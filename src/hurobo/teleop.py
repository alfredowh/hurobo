"""Teleoperation: maps a tracked human hand (or the keyboard) to LIBERO end-effector targets, step by step."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from scipy.spatial.transform import Rotation

from hurobo.control import Waypoint, WaypointFollower
from hurobo.demo import HumanDemo
from hurobo.props import CubeProp
from hurobo.sim import LiberoTask
from hurobo.twin import SIM_TABLE_TOP_Z
from hurobo.utils.transforms import cv_camera_pose_to_mujoco, transform_to_pos_quat_wxyz


def yaw_aligning_views(t_table_cam_cv: np.ndarray, t_world_cam_mj: np.ndarray) -> float:
    """Yaw about world z that makes the human camera's ground heading match a sim camera's.

    The phone and the LIBERO camera generally look at the table from different sides. Rotating
    the table frame by this yaw makes "hand moves right/away in the phone image" show up as
    "gripper moves right/away" in the sim camera too.

    Args:
        t_table_cam_cv: human camera pose in the table frame, OpenCV axes (looks down +z).
        t_world_cam_mj: sim camera pose in the world frame, MuJoCo axes (looks down -z).

    Returns:
        Angle in radians, R_world_table = Rz(yaw).
    """
    forward_table = t_table_cam_cv[:3, 2]
    forward_world = -t_world_cam_mj[:3, 2]
    yaw = np.arctan2(forward_world[1], forward_world[0]) - np.arctan2(forward_table[1], forward_table[0])
    return float((yaw + np.pi) % (2 * np.pi) - np.pi)


@dataclass(frozen=True)
class TableToWorld:
    """Similarity map from the table (ArUco) frame to the sim world frame, anchored on the first pose.

    Teleoperation uses *deltas*: the hand's first-frame TCP is pinned to the robot's starting
    end-effector position and everything after moves relative to it. That removes the need
    for the real table to coincide with the sim table (heights and offsets differ anyway),
    while the table-frame axes (not the camera's) define the directions, so two clips filmed
    from different camera poses drive the robot the same way.

    Attributes:
        p_table_anchor: shape (3,), hand TCP at the first sample, table frame.
        p_world_anchor: shape (3,), robot end-effector at the start, world frame.
        yaw_rad: rotation of the table axes about world z.
        scale: metres of robot motion per metre of hand motion.
        rotation: optional full 3x3 rotation from the reference frame to the world, replacing
            `yaw_rad`. Markerless teleoperation uses the camera frame as the reference frame
            ("table" then means "camera") and passes the agentview camera's orientation here.
    """

    p_table_anchor: np.ndarray
    p_world_anchor: np.ndarray
    yaw_rad: float = 0.0
    scale: float = 1.0
    rotation: np.ndarray | None = None

    @classmethod
    def grounded(
        cls, p_table_anchor: np.ndarray, p_world_eef: np.ndarray, yaw_rad: float = 0.0, scale: float = 1.0
    ) -> TableToWorld:
        """Like the plain delta map, but the anchor height is the hand's height above the table.

        x and y stay deltas from the robot's current position; z is absolute: the table plane
        (table-frame z = 0) lands on the sim table top. The robot has to be moved to the
        anchor first (`ApproachCommandSource`), so grasps at table level happen at table level.

        Args:
            p_table_anchor: shape (3,), hand TCP at the first sample, table frame.
            p_world_eef: shape (3,), robot end effector now, world frame.
            yaw_rad: see the class.
            scale: see the class.
        """
        p_world_anchor = np.array(
            [p_world_eef[0], p_world_eef[1], SIM_TABLE_TOP_Z + scale * p_table_anchor[2]]
        )
        return cls(np.asarray(p_table_anchor, dtype=float), p_world_anchor, yaw_rad, scale)

    @property
    def r_world_table(self) -> np.ndarray:
        """shape (3, 3), rotation taking table-frame vectors to world-frame vectors."""
        if self.rotation is not None:
            return np.asarray(self.rotation, dtype=float)
        return Rotation.from_euler("z", self.yaw_rad).as_matrix()

    def point_to_world(self, p_table: np.ndarray) -> np.ndarray:
        """Map table-frame points, shape (3,) or (N, 3), to the world frame."""
        return self.p_world_anchor + self.scale * (np.asarray(p_table) - self.p_table_anchor) @ (
            self.r_world_table.T
        )

    def point_to_table(self, p_world: np.ndarray) -> np.ndarray:
        """Inverse of `point_to_world`."""
        return self.p_table_anchor + (np.asarray(p_world) - self.p_world_anchor) @ self.r_world_table / (
            self.scale
        )

    def pose_to_world(self, t_table_x: np.ndarray) -> np.ndarray:
        """Re-express a 4x4 pose given in the table frame in the world frame (position scaled too)."""
        t_world_x = np.eye(4)
        t_world_x[:3, :3] = self.r_world_table @ t_table_x[:3, :3]
        t_world_x[:3, 3] = self.point_to_world(t_table_x[:3, 3])
        return t_world_x


def place_twin_camera(
    task: LiberoTask, mapping: TableToWorld, t_table_cam_cv: np.ndarray, fovy_degrees: float
) -> None:
    """Put the sim's twin camera where the human camera is, in the world frame `mapping` defines.

    Args:
        task: the sim whose `TWIN_CAMERA_NAME` camera is repositioned.
        mapping: table-to-world map of this teleoperation.
        t_table_cam_cv: human camera pose in the table frame, OpenCV axes.
        fovy_degrees: the human camera's vertical field of view.
    """
    t_world_cam_mj = cv_camera_pose_to_mujoco(mapping.pose_to_world(t_table_cam_cv))
    pos, quat_wxyz = transform_to_pos_quat_wxyz(t_world_cam_mj)
    task.set_camera_pose(task.TWIN_CAMERA_NAME, pos, quat_wxyz, fovy_degrees)


@dataclass(frozen=True)
class TeleopCommand:
    """What the operator asks for at one control step.

    Attributes:
        pos_world: shape (3,), desired end-effector position, world frame.
        gripper_closed: desired gripper state.
        source_time_s: time in the human clip this command comes from, if it comes from one.
        starts_stream: True on the command that begins following a hand (first lock-on, or a
            re-anchor): its gripper state is the starting state, not an open-to-closed edge.
    """

    pos_world: np.ndarray
    gripper_closed: bool
    source_time_s: float | None = None
    starts_stream: bool = False


class CommandSource(Protocol):
    """Anything that produces one `TeleopCommand` per control step."""

    def next_command(self, p_world_eef: np.ndarray) -> TeleopCommand | None:
        """The next command given the current end-effector position, or None when finished."""
        ...


class HandCommandSource:
    """Replays a `HumanDemo` as end-effector commands, optionally slowed down.

    Slowing down stretches the *human* trajectory in time, so the robot is asked to move
    more slowly (it tracks better) and the recording is longer and easier to inspect.
    """

    def __init__(
        self,
        demo: HumanDemo,
        mapping: TableToWorld,
        control_hz: float = 20.0,
        slowdown: float = 1.0,
        hold_end_s: float = 0.5,
    ) -> None:
        """Set up the replay.

        Args:
            demo: processed human demo; positions are in the table frame.
            mapping: table-to-world map; its anchor should be `demo.tcp_pos_table[0]`.
            control_hz: sim control rate.
            slowdown: > 1 plays the human motion this many times slower.
            hold_end_s: sim seconds to keep commanding the final pose after the clip ends.
        """
        if slowdown <= 0:
            raise ValueError("slowdown must be positive")
        self._demo = demo
        self._mapping = mapping
        self._dt_s = 1.0 / control_hz
        self._slowdown = slowdown
        self._hold_steps = int(round(hold_end_s * control_hz))
        self._step = 0

    @property
    def num_steps(self) -> int:
        """Control steps the whole replay (including the end hold) takes."""
        return int(np.ceil(self._demo.timestamps_s[-1] * self._slowdown / self._dt_s)) + 1 + self._hold_steps

    def next_command(self, p_world_eef: np.ndarray) -> TeleopCommand | None:
        """See `CommandSource`; `p_world_eef` is unused (the hand does not look at the robot)."""
        if self._step >= self.num_steps:
            return None
        demo_time_s = min(self._step * self._dt_s / self._slowdown, float(self._demo.timestamps_s[-1]))
        self._step += 1
        p_table = np.array(
            [np.interp(demo_time_s, self._demo.timestamps_s, axis) for axis in self._demo.tcp_pos_table.T]
        )
        nearest = int(np.argmin(np.abs(self._demo.timestamps_s - demo_time_s)))
        return TeleopCommand(
            pos_world=self._mapping.point_to_world(p_table),
            gripper_closed=bool(self._demo.is_grasping[nearest]),
            source_time_s=demo_time_s + self._demo.source_start_time_s,
        )


class ApproachCommandSource:
    """Holds one target until the arm is within a tolerance of it, or a step budget runs out."""

    def __init__(
        self,
        p_world_target: np.ndarray,
        gripper_closed: bool = False,
        tolerance_m: float = 0.01,
        max_steps: int = 80,
    ) -> None:
        """Set the point to reach.

        Args:
            p_world_target: shape (3,), world-frame position.
            gripper_closed: gripper state to hold while approaching; pass the next source's
                first state so the hand-over does not look like a grasp.
            tolerance_m: distance at which the approach counts as done.
            max_steps: give up (and let the next source start) after this many steps.
        """
        self._closed = gripper_closed
        self._target = np.asarray(p_world_target, dtype=float)
        self._tolerance_m = tolerance_m
        self._steps_left = max_steps

    def next_command(self, p_world_eef: np.ndarray) -> TeleopCommand | None:
        """See `CommandSource`."""
        if self._steps_left <= 0 or np.linalg.norm(self._target - p_world_eef) < self._tolerance_m:
            return None
        self._steps_left -= 1
        return TeleopCommand(self._target.copy(), self._closed)


class SequenceCommandSource:
    """Runs several sources one after another; each starts when the previous one is exhausted."""

    def __init__(self, sources: list[CommandSource]) -> None:
        """Order matters: the first source is consumed completely before the second is asked."""
        self._sources = list(sources)

    def next_command(self, p_world_eef: np.ndarray) -> TeleopCommand | None:
        """See `CommandSource`."""
        while self._sources:
            command = self._sources[0].next_command(p_world_eef)
            if command is not None:
                return command
            self._sources.pop(0)
        return None


class KeySource(Protocol):
    """Keyboard state as `KeyboardCommandSource` needs it (names are lower-case Tk keysyms)."""

    def pressed(self) -> set[str]:
        """Keys currently held down."""
        ...

    def consume(self, key: str) -> bool:
        """True once per fresh press of `key` (edge-triggered), False otherwise."""
        ...


@dataclass
class ScriptedKeys:
    """A `KeySource` driven by a step schedule instead of a keyboard, for tests and headless demos.

    One `pressed()` call is one control step (`KeyboardCommandSource` makes exactly one per
    step, after any `consume` calls), so the schedule advances by itself.

    Attributes:
        holds: (first_step, last_step, key) triples; the key is held for steps first..last.
        presses: (step, key) pairs; a single edge press at that step.
    """

    holds: list[tuple[int, int, str]] = field(default_factory=list)
    presses: list[tuple[int, str]] = field(default_factory=list)
    step: int = 0

    def pressed(self) -> set[str]:
        """See `KeySource`; also moves the schedule on to the next step."""
        held = {key for first, last, key in self.holds if first <= self.step <= last}
        self.step += 1
        return held

    def consume(self, key: str) -> bool:
        """See `KeySource`."""
        return (self.step, key) in self.presses


class KeyboardCommandSource:
    """Moves a target with held keys and toggles the gripper with the space bar.

    Arrow keys are screen-relative to the `agentview` camera (up = away from the viewer),
    `w`/`s` raise/lower, `space` toggles the gripper, `escape` or `q` ends the session.
    """

    #: key -> unit direction in the world frame. LIBERO's agentview looks along world -x with +y to the right.
    DIRECTIONS = {
        "up": np.array([-1.0, 0.0, 0.0]),
        "down": np.array([1.0, 0.0, 0.0]),
        "left": np.array([0.0, -1.0, 0.0]),
        "right": np.array([0.0, 1.0, 0.0]),
        "w": np.array([0.0, 0.0, 1.0]),
        "s": np.array([0.0, 0.0, -1.0]),
    }
    QUIT_KEYS = ("escape", "q")

    def __init__(
        self,
        keys: KeySource,
        start_pos_world: np.ndarray,
        control_hz: float = 20.0,
        speed_mps: float = 0.15,
        max_lead_m: float = 0.04,
    ) -> None:
        """Start with the target at the robot's current position, gripper open.

        Args:
            keys: keyboard state provider.
            start_pos_world: shape (3,), initial target, world frame.
            control_hz: sim control rate (sets the distance covered per step).
            speed_mps: target speed while a key is held.
            max_lead_m: how far the target may run ahead of the arm; stops it winding up when
                the arm is blocked (e.g. pressed against the table).
        """
        self._keys = keys
        self._target = np.asarray(start_pos_world, dtype=float).copy()
        self._dt_s = 1.0 / control_hz
        self._speed_mps = speed_mps
        self._max_lead_m = max_lead_m
        self._closed = False

    def next_command(self, p_world_eef: np.ndarray) -> TeleopCommand | None:
        """See `CommandSource`."""
        if any(self._keys.consume(key) for key in self.QUIT_KEYS):
            return None
        if self._keys.consume("space"):
            self._closed = not self._closed
        held = self._keys.pressed()
        direction = sum((v for k, v in self.DIRECTIONS.items() if k in held), np.zeros(3))
        self._target = self._target + direction * self._speed_mps * self._dt_s
        lead = np.clip(self._target - p_world_eef, -self._max_lead_m, self._max_lead_m)
        self._target = p_world_eef + lead
        return TeleopCommand(self._target.copy(), self._closed)


@dataclass(frozen=True)
class WorkspaceLimits:
    """Axis-aligned box the commanded position is clamped into (the arm's usable, collision-safe region).

    Attributes:
        lower: shape (3,), minimum x, y, z in the world frame.
        upper: shape (3,), maximum x, y, z in the world frame.
    """

    lower: np.ndarray
    upper: np.ndarray

    @classmethod
    def around(
        cls, p_world_anchor: np.ndarray, half_extent_xy_m: float = 0.30, z_above_m: float = 0.35
    ) -> WorkspaceLimits:
        """Box centred (in x, y) on the start position, bounded below by just above the table top."""
        anchor = np.asarray(p_world_anchor, dtype=float)
        lower = np.array(
            [anchor[0] - half_extent_xy_m, anchor[1] - half_extent_xy_m, SIM_TABLE_TOP_Z + 0.025]
        )
        upper = np.array([anchor[0] + half_extent_xy_m, anchor[1] + half_extent_xy_m, anchor[2] + z_above_m])
        return cls(lower, upper)

    def clamp(self, p_world: np.ndarray) -> tuple[np.ndarray, bool]:
        """Clamp a position; also report whether it had to move."""
        clamped = np.clip(p_world, self.lower, self.upper)
        return clamped, bool(np.any(clamped != p_world))


@dataclass
class StepInfo:
    """Everything observable after one control step.

    Attributes:
        step: 0-based control step index.
        t_s: sim time after the step.
        command: what was asked for (before workspace clamping).
        p_world_target: the clamped position actually tracked.
        p_world_eef: end-effector position after the step.
        gripper_closed: commanded gripper state.
        clamped: True if the workspace limits modified the command.
        cube_spawned: True once the cube prop is in play.
        p_world_cube: shape (3,) cube centre after the step once spawned, else None.
        obs: the env observation after the step.
    """

    step: int
    t_s: float
    command: TeleopCommand
    p_world_target: np.ndarray
    p_world_eef: np.ndarray
    gripper_closed: bool
    clamped: bool
    cube_spawned: bool
    p_world_cube: np.ndarray | None
    obs: dict


class TeleopSession:
    """Closed-loop teleoperation of one LIBERO task from any `CommandSource`.

    Each step: take the source's command, clamp it to the workspace, convert the position
    error to a clipped relative action (`WaypointFollower.action_toward`), step the sim.
    The end-effector orientation is held at its starting value (top-down).
    """

    #: Steps the cube is supported against gravity at most, waiting for the fingers to close on it.
    CUBE_SUPPORT_STEPS = 20

    def __init__(
        self,
        task: LiberoTask,
        source: CommandSource,
        limits: WorkspaceLimits,
        cube: CubeProp | None = None,
        action_limit: float = 1.0,
        control_hz: float = 20.0,
    ) -> None:
        """Wire up a session around an already reset task.

        Args:
            task: the sim; `reset` must have been called (and the cube installed, if any).
            source: where commands come from.
            limits: workspace clamp.
            cube: optional prop spawned between the fingers the first time the gripper is told to
                close after having been open.
            action_limit: fraction of the controller's maximum per-step motion used. Commands are
                the position error itself (at most 5 cm), which tracked a 4x-slowed hand to ~1 cm;
                0.3 lagged by 4-9 cm. Lower it if the arm oscillates.
            control_hz: sim control rate, used for the reported sim time.
        """
        pos_scale_m, rot_scale_rad = task.controller_action_scale
        self._task = task
        self._source = source
        self._limits = limits
        self._cube = cube
        self._action_limit = action_limit
        self._dt_s = 1.0 / control_hz
        self._follower = WaypointFollower(max_pos_delta_m=pos_scale_m, max_rot_delta_rad=rot_scale_rad)

    def run(self, obs: dict) -> Iterator[StepInfo]:
        """Step the sim until the source is exhausted, yielding after every step.

        Args:
            obs: the observation returned by `task.reset()`.

        Yields:
            One `StepInfo` per control step. The consumer decides what to do with it
            (render, record, stop early by simply not iterating further).
        """
        p_world_eef, r_world_eef = self._task.get_eef_pose(obs)
        r_world_hold = r_world_eef
        spawn_step: int | None = None
        was_closed = True  # a clip that starts closed is not a grasp: only an open -> closed edge spawns
        step = 0
        while (command := self._source.next_command(p_world_eef)) is not None:
            target, clamped = self._limits.clamp(command.pos_world)
            if command.starts_stream:
                was_closed = command.gripper_closed
            closing = command.gripper_closed and not was_closed
            was_closed = command.gripper_closed
            if self._cube and not self._cube.spawned and closing:
                yaw = float(np.arctan2(r_world_eef.as_matrix()[1, 0], r_world_eef.as_matrix()[0, 0]))
                self._cube.spawn(self._task, p_world_eef, yaw)
                spawn_step = step
            action = self._follower.action_toward(
                Waypoint(target, r_world_hold, command.gripper_closed),
                p_world_eef,
                r_world_eef,
                self._action_limit,
            )
            obs, _, _ = self._task.step(action)
            p_world_eef, r_world_eef = self._task.get_eef_pose(obs)
            if self._cube and self._cube.supported and spawn_step is not None:
                gripped = self._task.finger_opening_m(obs) <= self._cube.width_m + 0.002
                if gripped or step - spawn_step >= self.CUBE_SUPPORT_STEPS:
                    self._cube.release_support(self._task)
            yield StepInfo(
                step=step,
                t_s=(step + 1) * self._dt_s,
                command=command,
                p_world_target=target,
                p_world_eef=p_world_eef.copy(),
                gripper_closed=command.gripper_closed,
                clamped=clamped,
                cube_spawned=bool(self._cube and self._cube.spawned),
                p_world_cube=self._cube.position(self._task) if self._cube and self._cube.spawned else None,
                obs=obs,
            )
            step += 1
