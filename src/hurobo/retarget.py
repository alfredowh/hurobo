"""Object-centric retargeting: a human pick-and-place becomes a robot pick-and-place from the home pose."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from hurobo.control import Waypoint, WaypointFollower
from hurobo.demo import HumanDemo
from hurobo.scaled_native import (
    BOWL_JOINT,
    PLATE_JOINT,
    ROBOT_BASE_XY,
    ScaledNativeScene,
    SimLayout,
)
from hurobo.teleop import TableToWorld
from hurobo.twin import SIM_TABLE_TOP_Z

#: The Panda's fingers close along world y at the home orientation (checked in the sim, not assumed).
FINGER_AXIS_ANGLE_RAD = np.pi / 2


@dataclass(frozen=True)
class RetargetConfig:
    """How a human demo is turned into robot goals.

    Attributes:
        approach_radius_m: only the last part of the human approach, the part within this distance of
            the grasp point, is replayed. What the hand did before that (where it started) is dropped:
            the robot always starts from its own home pose.
        transit_height_m: TCP height above the table for the move from home to the start of the approach.
        spacing_m: distance between consecutive goals along a replayed path.
        grasp_height_m: height above the table at which the fingers close. The human value is replaced by
            this one: the estimate's depth error (about 2 cm) is larger than the part of the wall the
            fingertip pads can hold, and a pinch at the very top of a 2 mm wall does not lift the bowl.
        release_clearance_m: height of the bowl's base above the plate surface when it is let go.
        carry_clearance_m: height of the bowl's base above the plate rim while it travels over it.
        max_tcp_reach_m: horizontal distance from the robot base beyond which the Panda cannot hold the
            fingers upright at table height (measured: `docs/experiment_scaled_native.md`); the replayed
            approach is pulled back inside it, which matters when the human came in from the far side.
        retreat_height_m: how far the TCP rises after the release.
        action_limit: fraction of the controller's per-step motion limit used (1 = fastest).
        gripper_steps: control steps spent waiting for the gripper to close or open.
        pass_tolerance_m: distance at which an intermediate goal counts as reached; the arm does not slow
            down for it, so the motion is smooth.
        final_tolerance_m: the same for the last goal of a path (the grasp point, the release point).
        max_steps_per_goal: give up on a goal after this many steps.
        max_total_steps: abort the whole rollout after this many steps.
        lost_bowl_distance_m: the bowl is counted as dropped when it is farther than this from the TCP
            while it should be in the fingers.
        paced: follow the demo's own clock. The approach and the carry are tracked by a goal that moves
            along the path as the demo's timestamps say (a demo slowed with `HumanDemo.time_scaled` gives a
            slower robot), instead of passing goals as fast as the controller allows. There is no separate
            straight-up lift: the carry starts right after the grasp and rises with the path.
        control_hz: control rate, to turn steps into demo seconds when `paced`.
        carry_rise_samples: when `paced`, the minimum carry height above the plate rim is reached
            gradually over this many demo samples (20 per second), so the bowl does not jump up.
    """

    approach_radius_m: float = 0.20
    transit_height_m: float = 0.20
    spacing_m: float = 0.03
    grasp_height_m: float = 0.040
    release_clearance_m: float = 0.004
    carry_clearance_m: float = 0.04
    max_tcp_reach_m: float = 0.76
    retreat_height_m: float = 0.15
    action_limit: float = 0.5
    gripper_steps: int = 14
    pass_tolerance_m: float = 0.02
    final_tolerance_m: float = 0.005
    max_steps_per_goal: int = 150
    max_total_steps: int = 800
    lost_bowl_distance_m: float = 0.20
    paced: bool = False
    control_hz: float = 20.0
    carry_rise_samples: int = 20


@dataclass
class RetargetPlan:
    """The world-frame paths a demo gives for one layout.

    Attributes:
        approach_world: shape (K, 3), TCP path ending at the grasp point, over the bowl.
        bowl_path_world: shape (M, 3), the path of the bowl's base centre while carried, starting at the
            bowl's resting position and ending over the plate centre.
        gripper_yaw_rad: rotation about world z, relative to the home orientation, that points the
            finger-closing axis radially at the rim.
        grasp_radius_m: horizontal TCP-to-bowl-centre distance at the grasp, after snapping to the rim.
        release_z_m: world z of the bowl's base centre at the release.
        approach_times_s: shape (K,), demo time of each approach point (from the demo's timestamps).
        bowl_path_times_s: shape (M,), demo time of each bowl-path point.
    """

    approach_world: np.ndarray
    bowl_path_world: np.ndarray
    gripper_yaw_rad: float
    grasp_radius_m: float
    release_z_m: float
    approach_times_s: np.ndarray
    bowl_path_times_s: np.ndarray


@dataclass
class RolloutStep:
    """One control step of a rollout, for recording.

    Attributes:
        obs: the observation the action was computed from.
        action: shape (7,), what was sent to the env.
        phase: name of the phase that produced it.
        success: True once the task's success check is true.
    """

    obs: dict
    action: np.ndarray
    phase: str
    success: bool


@dataclass
class RolloutResult:
    """Outcome of one executed plan.

    Attributes:
        success: the LIBERO success check fired.
        steps: control steps taken.
        failure: why it stopped early ("" when it did not).
        phases: names of the phases that ran, in order.
        phase_start_steps: phase name -> index of the control step at which it began.
    """

    success: bool = False
    steps: int = 0
    failure: str = ""
    phases: list[str] = field(default_factory=list)
    phase_start_steps: dict[str, int] = field(default_factory=dict)


def resample_path(path: np.ndarray, spacing_m: float) -> np.ndarray:
    """Points along a polyline at roughly equal arc-length spacing; both end points are kept.

    Args:
        path: shape (N, 3).
        spacing_m: target spacing, > 0.

    Returns:
        shape (M, 3) with M >= 2.
    """
    path = np.asarray(path, dtype=float)
    if len(path) < 2:
        return np.repeat(path, 2, axis=0)
    segment = np.linalg.norm(np.diff(path, axis=0), axis=1)
    arclength = np.concatenate([[0.0], np.cumsum(segment)])
    if arclength[-1] < 1e-9:
        return path[[0, -1]]
    count = max(2, int(np.ceil(arclength[-1] / spacing_m)) + 1)
    samples = np.linspace(0.0, arclength[-1], count)
    return np.stack([np.interp(samples, arclength, path[:, axis]) for axis in range(3)], axis=1)


def resample_timed_path(
    path: np.ndarray, times_s: np.ndarray, spacing_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """`resample_path`, also giving the time at which each resampled point was reached.

    Args:
        path: shape (N, 3).
        times_s: shape (N,), non-decreasing time of each path point.
        spacing_m: target spacing, > 0.

    Returns:
        (points, times): shapes (M, 3) and (M,). While the path stands still the time passes between two
        neighbouring points, so a goal that follows the times waits there.
    """
    path = np.asarray(path, dtype=float)
    points = resample_path(path, spacing_m)
    arclength = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    resampled_arclength = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    # A tiny ramp keeps the arc length strictly increasing where the path stands still.
    times = np.interp(resampled_arclength, arclength + 1e-9 * np.arange(len(arclength)), times_s)
    # The resampled path is a little shorter than the original, which would drop a dwell at its end.
    times[-1] = times_s[-1]
    return points, times


def wrap_symmetric_yaw(angle_rad: float) -> float:
    """Wrap an angle to (-pi/2, pi/2]: a parallel gripper looks the same after a half turn."""
    return float((angle_rad + np.pi / 2) % np.pi - np.pi / 2) if angle_rad != np.pi / 2 else np.pi / 2


class ObjectCentricRetargeter:
    """Turns a `HumanDemo` into robot paths anchored on the bowl and the plate of a sim layout.

    The rules (all in the world frame, by way of the table-to-world map):

    1. **Approach**: the human TCP path, relative to the bowl centre, for the last `approach_radius_m`
       before the grasp. The grasp point is snapped radially onto the real bowl's rim and its height
       clipped, so a noisy hand estimate still makes a graspable grasp.
    2. **Carry**: the human TCP path from grasp to release becomes the path of the *bowl* (the TCP minus
       the grasp offset), relative to the plate. It is bent so that it starts at the bowl's real resting
       place and ends with the bowl centred on the plate at `release_clearance_m`, which removes the
       human's placement error. The executor then moves the TCP so that the actual bowl follows this
       path, whatever its tilt in the fingers.
    3. **Start**: nothing of the human's start pose is kept; the executor goes from the robot's home pose
       to the first approach point at a safe height.

    Orientation is not taken from the hand: the gripper stays top-down, turned about z only so that the
    fingers close across the rim wall.
    """

    def __init__(
        self,
        mapping: TableToWorld,
        bowl_radius_m: float,
        plate_height_m: float,
        config: RetargetConfig | None = None,
    ) -> None:
        """Set up with the frame map and the sim object sizes.

        Args:
            mapping: table-to-world map (`scaled_native.table_to_world_map`).
            bowl_radius_m: bowl radius at the rim (real, equal to the sim's after scaling).
            plate_height_m: the sim plate's height above the table.
            config: see `RetargetConfig`.
        """
        self._mapping = mapping
        self._bowl_radius_m = bowl_radius_m
        self._plate_height_m = plate_height_m
        self._config = config or RetargetConfig()

    def replayed_sample_range(self, demo: HumanDemo) -> tuple[int, int]:
        """The samples (first, last) of the human motion the robot replays: from where the hand comes within
        `approach_radius_m` of the grasp point to the release.

        Raises:
            ValueError: if the demo has no grasp or no release.
        """
        if demo.grasp_frame_idx is None or demo.release_frame_idx is None:
            raise ValueError("the demo needs a detected grasp and release")
        tcp = np.asarray(demo.tcp_pos_table, dtype=float)
        distance = np.linalg.norm(tcp[: demo.grasp_frame_idx + 1] - tcp[demo.grasp_frame_idx], axis=1)
        far = np.flatnonzero(distance > self._config.approach_radius_m)
        return (int(far[-1]) + 1 if len(far) else 0), demo.release_frame_idx

    def plan(self, demo: HumanDemo, layout: SimLayout) -> RetargetPlan:
        """Paths for one demo on one layout.

        Args:
            demo: a processed demo with grasp and release found.
            layout: where the bowl and plate are in the world.

        Returns:
            The plan.

        Raises:
            ValueError: if the demo has no grasp or no release.
        """
        if demo.grasp_frame_idx is None or demo.release_frame_idx is None:
            raise ValueError("the demo needs a detected grasp and release")
        cfg = self._config
        grasp_idx, release_idx = demo.grasp_frame_idx, demo.release_frame_idx
        tcp = np.asarray(demo.tcp_pos_table, dtype=float)
        bowl_xy_table = np.asarray(demo.t_table_bowl)[:2, 3]
        plate_xy_table = np.asarray(demo.t_table_plate)[:2, 3]

        # 1. approach: the stretch before the grasp that lies within approach_radius_m of it
        start_idx = self.replayed_sample_range(demo)[0]
        approach_xy = self._to_world_xy(tcp[start_idx : grasp_idx + 1, :2], bowl_xy_table, layout.bowl_xy)
        approach_z = tcp[start_idx : grasp_idx + 1, 2].copy()
        radial = approach_xy[-1] - layout.bowl_xy
        radial_dir = (
            radial / np.linalg.norm(radial) if np.linalg.norm(radial) > 1e-6 else np.array([-1.0, 0.0])
        )
        snapped_xy = layout.bowl_xy + self._bowl_radius_m * radial_dir
        ramp = np.linspace(0.0, 1.0, len(approach_xy))
        approach_xy = approach_xy + ramp[:, None] * (snapped_xy - approach_xy[-1])
        approach_z = approach_z + ramp * (cfg.grasp_height_m - approach_z[-1])
        from_base = approach_xy - np.asarray(ROBOT_BASE_XY)
        distance = np.linalg.norm(from_base, axis=1, keepdims=True)
        approach_xy = np.asarray(ROBOT_BASE_XY) + from_base * np.minimum(1.0, cfg.max_tcp_reach_m / distance)
        approach, approach_times = resample_timed_path(
            np.column_stack([approach_xy, SIM_TABLE_TOP_Z + approach_z]),
            np.asarray(demo.timestamps_s)[start_idx : grasp_idx + 1],
            cfg.spacing_m,
        )

        # 2. carry: the bowl's base path (TCP minus the grasp offset), bent to start and end where it must
        grasp_offset_table = tcp[grasp_idx, :2] - bowl_xy_table
        carried = tcp[grasp_idx : release_idx + 1]
        carry_xy = self._to_world_xy(carried[:, :2] - grasp_offset_table, plate_xy_table, layout.plate_xy)
        carry_z = np.maximum(carried[:, 2] - carried[0, 2], 0.0)
        release_z = self._plate_height_m + cfg.release_clearance_m
        ramp = np.linspace(0.0, 1.0, len(carry_xy))
        carry_xy = (
            carry_xy
            + (1.0 - ramp)[:, None] * (layout.bowl_xy - carry_xy[0])
            + ramp[:, None] * (layout.plate_xy - carry_xy[-1])
        )
        carry_z = carry_z + (1.0 - ramp) * (0.0 - carry_z[0]) + ramp * (release_z - carry_z[-1])
        # Keep the base above the plate rim until it is nearly over the plate centre.
        over_plate = np.linalg.norm(carry_xy - layout.plate_xy, axis=1) < 0.05
        floor_z = self._plate_height_m + cfg.carry_clearance_m
        if cfg.paced:
            # No jump at the grasp: the floor is reached gradually.
            floor_z = floor_z * np.clip(np.arange(len(carry_z)) / max(cfg.carry_rise_samples, 1), 0.0, 1.0)
        carry_z = np.where(over_plate, carry_z, np.maximum(carry_z, floor_z))
        carry, carry_times = resample_timed_path(
            np.column_stack([carry_xy, SIM_TABLE_TOP_Z + carry_z]),
            np.asarray(demo.timestamps_s)[grasp_idx : release_idx + 1],
            cfg.spacing_m,
        )

        # 3. finger axis across the rim wall
        yaw = wrap_symmetric_yaw(float(np.arctan2(radial_dir[1], radial_dir[0])) - FINGER_AXIS_ANGLE_RAD)
        return RetargetPlan(
            approach_world=approach,
            bowl_path_world=carry,
            gripper_yaw_rad=yaw,
            grasp_radius_m=self._bowl_radius_m,
            release_z_m=SIM_TABLE_TOP_Z + release_z,
            approach_times_s=approach_times,
            bowl_path_times_s=carry_times,
        )

    def _to_world_xy(
        self, p_table_xy: np.ndarray, anchor_table_xy: np.ndarray, anchor_world_xy: np.ndarray
    ) -> np.ndarray:
        """Table-frame (x, y) rows to world (x, y), keeping each row's offset from the anchor object."""
        relative = np.atleast_2d(p_table_xy) - anchor_table_xy
        return anchor_world_xy + relative @ self._mapping.r_world_table[:2, :2].T


class _EpisodeOver(Exception):
    """Raised inside the executor to unwind when the rollout is decided (success, failure or timeout)."""


class PlanExecutor:
    """Runs a `RetargetPlan` on a `ScaledNativeScene`, closed loop, yielding every control step.

    Uses privileged simulator state (the bowl's current pose) to keep the bowl on its planned path; the
    policy that is later trained on the recorded steps does not get that.
    """

    #: A bowl lifted less than this above the table after the grasp is counted as not picked up.
    MIN_LIFT_M = 0.02

    def __init__(self, scene: ScaledNativeScene, config: RetargetConfig | None = None) -> None:
        """Bind to a scene.

        Args:
            scene: the scene to act in; `reset` must have been called, and its observation is passed to `run`.
            config: speed, tolerances and dwell times.
        """
        self._scene = scene
        self._config = config or RetargetConfig()
        pos_scale_m, rot_scale_rad = scene.task.controller_action_scale
        self._follower = WaypointFollower(max_pos_delta_m=pos_scale_m, max_rot_delta_rad=rot_scale_rad)
        self._obs: dict = {}
        self._rot_goal = Rotation.identity()
        self._result = RolloutResult()

    @property
    def result(self) -> RolloutResult:
        """Outcome of the latest run (complete once `run` is exhausted)."""
        return self._result

    def run(self, obs: dict, plan: RetargetPlan) -> Iterator[RolloutStep]:
        """Execute the plan, yielding one `RolloutStep` per env step.

        Args:
            obs: the observation after `scene.reset`.
            plan: from `ObjectCentricRetargeter.plan` for the layout the scene was reset with.

        Yields:
            Each step taken. Ends on success, on a failure (see `result.failure`), or when the plan is done.
        """
        self._obs = obs
        self._result = RolloutResult()
        _, home_rot = self._scene.task.get_eef_pose(obs)
        self._rot_goal = Rotation.from_euler("z", plan.gripper_yaw_rad) * home_rot
        try:
            yield from self._script(plan)
        except _EpisodeOver:
            pass

    def _script(self, plan: RetargetPlan) -> Iterator[RolloutStep]:
        """The phases: transit (home to the approach), approach, grasp, lift, carry, release, retreat.

        When the config is `paced` the approach and the carry follow the demo's clock and there is no lift.
        """
        cfg = self._config
        above_start = plan.approach_world[0].copy()
        above_start[2] = SIM_TABLE_TOP_Z + cfg.transit_height_m
        if cfg.paced:
            yield from self._reach(lambda: above_start, False, "transit", cfg.pass_tolerance_m)
            yield from self._follow_timed(
                plan.approach_world, plan.approach_times_s, lambda: np.zeros(3), False, "approach"
            )
        else:
            goals = [above_start, *plan.approach_world]
            for index, goal in enumerate(goals):
                last = index == len(goals) - 1
                tolerance = cfg.final_tolerance_m if last else cfg.pass_tolerance_m
                phase = "transit" if index == 0 else "approach"
                yield from self._reach(lambda goal=goal: goal, False, phase, tolerance)
        yield from self._hold_gripper(True, "grasp")
        if not cfg.paced:
            lift_to = self._eef_pos() + np.array([0.0, 0.0, 0.6 * cfg.transit_height_m])
            yield from self._reach(lambda: lift_to, True, "lift", 0.01)
            if self._bowl_height_m() < self.MIN_LIFT_M:
                self._abort("bowl not lifted")

        # TCP target = planned bowl base position + the offset the bowl actually has in the fingers,
        # measured every step because the bowl swings in the grip.
        if cfg.paced:
            yield from self._follow_timed(
                plan.bowl_path_world, plan.bowl_path_times_s, self._tcp_minus_bowl, True, "carry"
            )
        else:
            carry_goals = plan.bowl_path_world[1:]
            for index, goal in enumerate(carry_goals):
                last = index == len(carry_goals) - 1
                tolerance = cfg.final_tolerance_m if last else cfg.pass_tolerance_m
                yield from self._reach(
                    lambda goal=goal: goal + self._tcp_minus_bowl(), True, "carry", tolerance
                )
                if np.linalg.norm(self._tcp_minus_bowl()) > cfg.lost_bowl_distance_m:
                    self._abort("bowl dropped")
        yield from self._hold_gripper(False, "release", steps=2 * cfg.gripper_steps)
        retreat_to = self._eef_pos() + np.array([0.0, 0.0, cfg.retreat_height_m])
        yield from self._reach(lambda: retreat_to, False, "retreat", 0.02)
        self._abort("plan finished without success")

    def _follow_timed(
        self, path: np.ndarray, times_s: np.ndarray, offset_fn, closed: bool, phase: str
    ) -> Iterator[RolloutStep]:
        """Track a goal that moves along `path` as `times_s` says, then settle on its end.

        Args:
            path: shape (K, 3), world positions the TCP (plus `offset_fn()`) should pass.
            times_s: shape (K,), demo time of each point; the goal starts at the first and takes the demo's
                duration to reach the last.
            offset_fn: added to the goal every step (the bowl's offset from the TCP when carrying).
            closed: gripper state.
            phase: name recorded for each step.

        After the duration has passed, the arm gets `max_steps_per_goal` more steps to get within
        `final_tolerance_m` of the end; a bowl lost while carrying aborts the rollout.
        """
        cfg = self._config
        duration_s = float(times_s[-1] - times_s[0])
        first_step = self._result.steps
        for _ in range(int(np.ceil(duration_s * cfg.control_hz)) + cfg.max_steps_per_goal):
            elapsed_s = (self._result.steps - first_step) / cfg.control_hz
            goal = np.array([np.interp(times_s[0] + elapsed_s, times_s, path[:, axis]) for axis in range(3)])
            pos, rot = self._scene.task.get_eef_pose(self._obs)
            target = goal + offset_fn()
            if elapsed_s >= duration_s and np.linalg.norm(target - pos) < cfg.final_tolerance_m:
                return
            action = self._follower.action_toward(
                Waypoint(target, self._rot_goal, closed), pos, rot, cfg.action_limit
            )
            yield from self._send(action, phase)
            if phase == "carry" and np.linalg.norm(self._tcp_minus_bowl()) > cfg.lost_bowl_distance_m:
                self._abort("bowl dropped")

    def _abort(self, reason: str) -> None:
        self._result.failure = reason
        raise _EpisodeOver

    def _eef_pos(self) -> np.ndarray:
        return self._scene.task.get_eef_pose(self._obs)[0]

    def _bowl_height_m(self) -> float:
        """Height of the bowl's base above the table."""
        return float(self._scene.task.get_object_pose(BOWL_JOINT).pos[2] - SIM_TABLE_TOP_Z)

    def _tcp_minus_bowl(self) -> np.ndarray:
        """TCP position minus the bowl's base centre, world frame, from the live sim."""
        return self._eef_pos() - self._scene.task.get_object_pose(BOWL_JOINT).pos

    def _reach(self, target_fn, closed: bool, phase: str, tolerance_m: float) -> Iterator[RolloutStep]:
        """Step until the TCP is within tolerance of `target_fn()`, or the step budget is spent.

        Running out of steps on one goal is tolerated, since the next goal pulls the arm on; a lost bowl is
        detected where it matters (after the lift, and while carrying).
        """
        for _ in range(self._config.max_steps_per_goal):
            pos, rot = self._scene.task.get_eef_pose(self._obs)
            target = target_fn()
            if np.linalg.norm(target - pos) < tolerance_m:
                return
            action = self._follower.action_toward(
                Waypoint(target, self._rot_goal, closed), pos, rot, self._config.action_limit
            )
            yield from self._send(action, phase)

    def _hold_gripper(self, closed: bool, phase: str, steps: int | None = None) -> Iterator[RolloutStep]:
        """Hold the arm still while the gripper closes or opens (and, after a release, the bowl settles)."""
        pos, rot = self._scene.task.get_eef_pose(self._obs)
        for _ in range(steps or self._config.gripper_steps):
            action = self._follower.action_toward(Waypoint(pos, rot, closed), pos, rot, 1.0)
            yield from self._send(action, phase)

    def _send(self, action: np.ndarray, phase: str) -> Iterator[RolloutStep]:
        """Step the env once and record it. A terminated episode is a success in LIBERO (it auto-resets)."""
        before = self._obs
        self._obs, success, terminated = self._scene.task.step(action)
        result = self._result
        result.steps += 1
        if not result.phases or result.phases[-1] != phase:
            result.phases.append(phase)
            result.phase_start_steps.setdefault(phase, result.steps - 1)
        result.success = result.success or success or terminated
        yield RolloutStep(obs=before, action=action, phase=phase, success=result.success)
        if result.success:
            raise _EpisodeOver
        if result.steps >= self._config.max_total_steps:
            self._abort("timeout")


def bowl_plate_distance_m(scene: ScaledNativeScene) -> float:
    """Horizontal distance between the bowl and plate origins (LIBERO's success check wants < 3 cm)."""
    task = scene.task
    return float(
        np.linalg.norm(task.get_object_pose(BOWL_JOINT).pos[:2] - task.get_object_pose(PLATE_JOINT).pos[:2])
    )
