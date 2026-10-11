"""Glue for teleoperation runs: human-clip preparation, side-by-side frames, live view and recording."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from hurobo.camera import CameraIntrinsics
from hurobo.camera_source import FrameSource
from hurobo.demo import GraspSegmenter, HumanDemo
from hurobo.extract import ClipReport, ExtractionConfig, extract_demo
from hurobo.hand import HandTcpTracker
from hurobo.live import LiveWindow
from hurobo.live_hand import LiveHandCommandSource, LiveHandSample, LiveHandTracker
from hurobo.markers import MarkerLayout, MarkerPoseEstimator
from hurobo.props import CubeProp
from hurobo.sim import LiberoTask
from hurobo.teleop import (
    ApproachCommandSource,
    HandCommandSource,
    SequenceCommandSource,
    StepInfo,
    TableToWorld,
    TeleopSession,
    WorkspaceLimits,
    place_twin_camera,
    yaw_aligning_views,
)
from hurobo.utils.synthetic import StandInScene
from hurobo.utils.transforms import apply_transform, invert_transform, level_camera_rotation
from hurobo.utils.video import (
    FrameReel,
    VideoWriter,
    caption_frame,
    draw_hand_skeleton,
    draw_outlined_text,
    hstack_resized,
    slow_down_video,
)

_RED, _CYAN = (0, 0, 255), (255, 255, 0)
_GREEN, _ORANGE, _BLUE = (0, 220, 0), (0, 140, 255), (255, 80, 0)


@dataclass
class PreparedClip:
    """A human clip ready to drive the robot.

    Attributes:
        demo: table-frame hand trajectory and grasp signal.
        report: extraction QC report (gates may be flagged for stand-in clips; teleop does not need them).
        intrinsics: camera that filmed the clip.
        video_path: the clip as extracted from (stand-in: the marker-augmented render).
        reel: its frames, for synced display.
        layout: the marker ids and sizes the clip was extracted with (also what a live camera
            pointed at the same scene needs).
    """

    demo: HumanDemo
    report: ClipReport
    intrinsics: CameraIntrinsics
    video_path: Path
    reel: FrameReel
    layout: MarkerLayout


def prepare_stand_in_clip(
    clip_dir: Path,
    out_dir: Path,
    camera_position_table: tuple[float, float, float] = (0.0, -0.45, 0.40),
    first_frame: int = 0,
) -> PreparedClip:
    """Paste ArUco markers into a frame folder at a chosen camera pose, then extract the hand.

    The table marker (not a ChArUco board) is what gives the camera pose: `T_cam_table` per
    clip, so the hand ends up in the table frame whatever the camera's position. ChArUco is only
    needed once per phone for intrinsics; here they are guessed (fx = fy = 800 px).

    Args:
        clip_dir: folder of `.png` frames.
        out_dir: where the marker-augmented clip (`standin_1x.mp4`) is written.
        camera_position_table: pretend camera position in the table frame (looks at the origin).
        first_frame: frames to skip at the start.

    Returns:
        The extracted clip. Depth is not anchored to the (fake) bowl marker.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    scene = StandInScene.from_clip_dir(clip_dir, first_frame, camera_position_table)
    video_path = scene.render_clip(out_dir / "standin_1x.mp4")
    scene.intrinsics.save(out_dir / "standin_intrinsics.npz")
    demo, report = extract_demo(
        video_path, scene.intrinsics, scene.layout, config=ExtractionConfig(anchor_depth=False)
    )
    return PreparedClip(
        demo, report, scene.intrinsics, video_path, FrameReel.from_video(video_path), scene.layout
    )


def prepare_video_clip(
    video_path: Path, intrinsics: CameraIntrinsics, layout: MarkerLayout, anchor_depth: bool = True
) -> PreparedClip:
    """Extract the hand from a real clip with printed ArUco markers in view.

    Args:
        video_path: phone clip.
        intrinsics: calibration of the camera that filmed it (ChArUco, `scripts/01_calibrate_camera.py`).
        layout: marker ids and sizes.
        anchor_depth: scale depth so the grasp lands on the bowl marker (see `ExtractionConfig`).
    """
    demo, report = extract_demo(
        video_path, intrinsics, layout, config=ExtractionConfig(anchor_depth=anchor_depth)
    )
    return PreparedClip(demo, report, intrinsics, video_path, FrameReel.from_video(video_path), layout)


def reset_with_props(task: LiberoTask, cube: CubeProp | None) -> dict:
    """Reset the task, add the cube prop, and settle one step so the observation matches the new sim.

    Args:
        task: the sim.
        cube: prop to install, or None.

    Returns:
        A fresh observation.
    """
    task.reset()
    if cube is not None:
        cube.install(task)
    open_gripper = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
    obs, _, _ = task.step(open_gripper)
    return obs


class HumanView(Protocol):
    """The left-hand panel of the display: the human side of the teleoperation."""

    @property
    def aspect(self) -> float:
        """Width over height of the camera image (used for the sim twin camera render)."""
        ...

    def panel(self, info: StepInfo) -> np.ndarray:
        """BGR panel for one step."""
        ...


def _draw_marker_overlay(
    frame: np.ndarray,
    intrinsics: CameraIntrinsics,
    t_cam_table: np.ndarray,
    mapping: TableToWorld,
    hand_world: np.ndarray | None,
    robot_world: np.ndarray,
) -> None:
    """Draw the commanded hand TCP (red cross) and the robot gripper (cyan ring) onto a camera frame.

    Both are world-frame points taken back through the table frame into the camera, so the
    two coincide exactly when the robot sits where the hand is.
    """
    for p_world, colour, is_hand in ((hand_world, _RED, True), (robot_world, _CYAN, False)):
        if p_world is None:
            continue
        p_cam = apply_transform(t_cam_table, mapping.point_to_table(p_world))
        if p_cam[2] <= 0:
            continue
        u, v = intrinsics.project(p_cam)
        if is_hand:
            cv2.drawMarker(frame, (round(u), round(v)), colour, cv2.MARKER_CROSS, 28, 3)
        else:
            cv2.circle(frame, (round(u), round(v)), 16, colour, 3, cv2.LINE_AA)


#: Overlay font scales (Hershey letters are ~22 px tall per unit of scale): panel titles,
#: the human-panel caption, and the bottom status line.
_TITLE_SCALE, _CAPTION_SCALE, _STATUS_SCALE = 0.55, 0.45, 0.41


def _caption(frame: np.ndarray, text: str) -> None:
    draw_outlined_text(frame, text, (10, 28), _CAPTION_SCALE)


class ClipView:
    """Human panel for an offline clip: the frame at the step's source time, with overlays."""

    def __init__(self, clip: PreparedClip, mapping: TableToWorld) -> None:
        """Bind a prepared clip and the table-to-world map used for its replay."""
        self._clip = clip
        self._mapping = mapping

    @property
    def aspect(self) -> float:
        """See `HumanView`."""
        return self._clip.intrinsics.image_width / self._clip.intrinsics.image_height

    def panel(self, info: StepInfo) -> np.ndarray:
        """See `HumanView`."""
        clip = self._clip
        frame = clip.reel.frame_at(info.command.source_time_s or clip.reel.start_s).copy()
        _draw_marker_overlay(
            frame,
            clip.intrinsics,
            invert_transform(clip.demo.t_table_cam),
            self._mapping,
            info.command.pos_world,
            info.p_world_eef,
        )
        _caption(frame, "human clip: hand TCP (red cross), robot (cyan ring)")
        return frame


class LiveView:
    """Human panel for a live camera: newest frame, detected hand, TCP/robot markers, pipeline status."""

    def __init__(self, source: LiveHandCommandSource, intrinsics: CameraIntrinsics) -> None:
        """Bind the live command source (which owns the newest frame and tracking state)."""
        self._source = source
        self._intrinsics = intrinsics

    @property
    def aspect(self) -> float:
        """See `HumanView`."""
        return self._intrinsics.image_width / self._intrinsics.image_height

    def panel(self, info: StepInfo) -> np.ndarray:
        """See `HumanView`."""
        frame = self._source.latest_frame
        if frame is None:
            frame = np.zeros((self._intrinsics.image_height, self._intrinsics.image_width, 3), np.uint8)
        frame = frame.copy()
        sample, mapping = self._source.latest_sample, self._source.mapping
        if sample is not None and sample.hand is not None and sample.hand.landmarks_image_px is not None:
            draw_hand_skeleton(frame, sample.hand.landmarks_image_px)
        self._draw_marker_state(frame, sample)
        if sample is not None and sample.t_cam_table is not None and mapping is not None:
            _draw_marker_overlay(
                frame, self._intrinsics, sample.t_cam_table, mapping, info.command.pos_world, info.p_world_eef
            )
        _caption(frame, f"live: {self._source.status}")
        text, ok = self._source.marker_status
        draw_outlined_text(frame, text, (10, 52), _CAPTION_SCALE, _GREEN if ok else _ORANGE)
        return frame

    def _draw_marker_state(self, frame: np.ndarray, sample: LiveHandSample | None) -> None:
        """Outline the detected markers (table marker thick green, others thin) or, once the table is
        locked, draw its frame axes (x red, y green, z blue) so the lock can be checked by eye."""
        if sample is None:
            return
        if sample.t_cam_table is not None:
            self._draw_table_axes(frame, sample.t_cam_table)
            return
        detection = sample.detection
        if detection is None:
            return
        for marker_id, corners in zip(detection.ids, detection.corners, strict=True):
            is_table = marker_id == detection.table_id
            polygon = corners.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [polygon], True, _GREEN if is_table else _ORANGE, 4 if is_table else 2)
            label = f"id {marker_id}" + (" TABLE" if is_table else "")
            draw_outlined_text(
                frame, label, (int(corners[0, 0]), int(corners[0, 1]) - 8), _CAPTION_SCALE, _GREEN
            )

    def _draw_table_axes(self, frame: np.ndarray, t_cam_table: np.ndarray, length_m: float = 0.08) -> None:
        origin = apply_transform(t_cam_table, np.zeros(3))
        if origin[2] <= 0:
            return
        o = tuple(round(v) for v in self._intrinsics.project(origin))
        for axis, colour in zip(np.eye(3), (_RED, _GREEN, _BLUE), strict=True):
            tip = apply_transform(t_cam_table, axis * length_m)
            if tip[2] > 0:
                cv2.line(
                    frame, o, tuple(round(v) for v in self._intrinsics.project(tip)), colour, 4, cv2.LINE_AA
                )


#: Camera whose mounting leaves the upright render 180 degrees off from the agentview's orientation.
_ROTATED_CAMERAS = {"robot0_eye_in_hand"}

_PANEL_TITLES = {
    "agentview": "agentview (policy camera)",
    "galleryview": "gallery view (whole setup)",
    "robot0_eye_in_hand": "wrist camera (top-down)",
}

#: Layout token for the human panel.
HUMAN = "human"


def default_layout(with_human: bool, twin_camera: str) -> list[list[str]]:
    """Rows of panels left to right.

    With a human source: human clip | twin (sim camera at the human camera's pose) on top, then
    agentview | gallery view of the whole setup | top-down wrist camera below (the two rows come
    out about equally wide). Without one (keyboard): just the bottom row.
    """
    bottom = ["agentview", "galleryview", "robot0_eye_in_hand"]
    return [[HUMAN, twin_camera], bottom] if with_human else [bottom]


class TeleopFrameComposer:
    """Builds one display frame per step: the human side and sim camera views in a grid."""

    def __init__(
        self,
        task: LiberoTask,
        layout: list[list[str]],
        panel_height: int = 300,
        human_view: HumanView | None = None,
    ) -> None:
        """Set up the panels.

        Args:
            task: the sim to render from.
            layout: rows of panels, each entry a sim camera name or `HUMAN`; build one with
                `default_layout`. The task's twin camera is rendered at the human camera's aspect.
            panel_height: height of every panel in pixels.
            human_view: the panel for `HUMAN` (a `ClipView` or `LiveView`).

        Raises:
            ValueError: if the layout asks for `HUMAN` without a `human_view`.
        """
        if any(HUMAN in row for row in layout) and human_view is None:
            raise ValueError("layout contains the human panel but no human_view was given")
        self._task = task
        self._layout = layout
        self._height = panel_height
        self._human_view = human_view

    def compose(self, info: StepInfo) -> np.ndarray:
        """Render all panels for one step and tile them. Returns a BGR frame."""
        rows = [
            hstack_resized([self._panel(name, info) for name in row], self._height) for row in self._layout
        ]
        width = max(row.shape[1] for row in rows)
        padded = [cv2.copyMakeBorder(r, 0, 0, 0, width - r.shape[1], cv2.BORDER_CONSTANT) for r in rows]
        frame = np.concatenate(padded, axis=0)
        status = (
            f"t={info.t_s:4.1f}s  gripper={'CLOSED' if info.gripper_closed else 'open  '}"
            f"{'  cube' if info.cube_spawned else ''}{'  (workspace clamp)' if info.clamped else ''}"
        )
        _caption_bottom(frame, status)
        return frame

    def _panel(self, name: str, info: StepInfo) -> np.ndarray:
        if name == HUMAN:
            return self._human_view.panel(info)
        is_twin = name == self._task.TWIN_CAMERA_NAME
        aspect = self._human_view.aspect if is_twin and self._human_view is not None else 1.0
        image = self._task.render_camera(name, int(self._height * aspect), self._height)
        if name in _ROTATED_CAMERAS:
            image = np.ascontiguousarray(image[::-1, ::-1])
        title = "twin view (human camera pose)" if is_twin else _PANEL_TITLES.get(name, name)
        return caption_frame(image, title, _TITLE_SCALE)


def _caption_bottom(frame: np.ndarray, text: str) -> None:
    draw_outlined_text(frame, text, (10, frame.shape[0] - 10), _STATUS_SCALE)


@dataclass
class TeleopTrace:
    """Per-step record of a run, for plots and tests.

    Attributes:
        t_s: sim time after each step.
        p_world_target: tracked (clamped) target positions.
        p_world_eef: end-effector positions.
        gripper_closed: commanded gripper state.
        p_world_cube: cube centre after each step from the one it spawned in, else NaN rows.
        cube_spawn_step: step at which the cube appeared, or None.
        frames: composed display frames (only when recording was asked for).
    """

    t_s: list[float] = field(default_factory=list)
    p_world_target: list[np.ndarray] = field(default_factory=list)
    p_world_eef: list[np.ndarray] = field(default_factory=list)
    gripper_closed: list[bool] = field(default_factory=list)
    p_world_cube: list[np.ndarray] = field(default_factory=list)
    cube_spawn_step: int | None = None
    frames: list[np.ndarray] = field(default_factory=list)

    def tracking_error_m(self) -> np.ndarray:
        """shape (N,), distance between target and end effector after each step."""
        return np.linalg.norm(np.asarray(self.p_world_target) - np.asarray(self.p_world_eef), axis=1)


class TeleopRunner:
    """Drives a `TeleopSession`, optionally showing it live and keeping the composed frames."""

    def __init__(
        self,
        session: TeleopSession,
        composer: TeleopFrameComposer | None = None,
        window: LiveWindow | None = None,
        keep_frames: bool = False,
        control_hz: float = 20.0,
    ) -> None:
        """Configure outputs.

        Args:
            session: the session to run.
            composer: needed for `window` or `keep_frames`.
            window: live view; the loop is paced to wall-clock sim time and ends if it is closed.
            keep_frames: store every composed frame in the trace (for videos).
            control_hz: sim control rate, used for pacing.
        """
        if (window is not None or keep_frames) and composer is None:
            raise ValueError("a composer is needed to show or record frames")
        self._session = session
        self._composer = composer
        self._window = window
        self._keep_frames = keep_frames
        self._dt_s = 1.0 / control_hz

    def run(self, obs: dict, max_steps: int | None = None) -> TeleopTrace:
        """Run to the end of the command source, until the window is closed, or for `max_steps`."""
        trace = TeleopTrace()
        for info in self._session.run(obs):
            if max_steps is not None and info.step >= max_steps:
                break
            started = time.monotonic()
            trace.t_s.append(info.t_s)
            trace.p_world_target.append(info.p_world_target)
            trace.p_world_eef.append(info.p_world_eef)
            trace.gripper_closed.append(info.gripper_closed)
            trace.p_world_cube.append(
                info.p_world_cube if info.p_world_cube is not None else np.full(3, np.nan)
            )
            if info.cube_spawned and trace.cube_spawn_step is None:
                trace.cube_spawn_step = info.step
            if self._composer is not None and (self._window is not None or self._keep_frames):
                frame = self._composer.compose(info)
                if self._keep_frames:
                    trace.frames.append(frame)
                if self._window is not None:
                    self._window.show(frame)
                    if self._window.closed:
                        break
                    time.sleep(max(0.0, self._dt_s - (time.monotonic() - started)))
        return trace


def write_run_videos(
    trace: TeleopTrace, out_dir: Path, slowdown: float, control_hz: float = 20.0
) -> dict[str, Path]:
    """Write the recorded frames as the slowed-down video and a real-speed one.

    The run itself was executed `slowdown` times slower than the human; the real-speed video
    keeps every `slowdown`-th frame, so it lasts as long as the original clip.

    Args:
        trace: a trace recorded with `keep_frames=True`.
        out_dir: output directory.
        slowdown: the factor the human trajectory was stretched by during the run.
        control_hz: sim control rate (output frame rate of both videos).

    Returns:
        {"slow": path, "1x": path}.
    """
    height, width = trace.frames[0].shape[:2]
    paths = {"slow": write_trace_video(trace, out_dir / "teleop_slow.mp4", control_hz)}
    paths["1x"] = out_dir / "teleop_1x.mp4"
    keep = np.unique(np.round(np.arange(0, len(trace.frames), slowdown)).astype(int))
    with VideoWriter(paths["1x"], control_hz, (width, height)) as writer:
        for index in keep:
            writer.write(trace.frames[index])
    return paths


def write_standin_slow_video(clip: PreparedClip, out_dir: Path, slowdown: float) -> Path:
    """Slowed copy of the marker-augmented human clip, next to (never replacing) the 1x one."""
    return slow_down_video(clip.video_path, out_dir / "standin_slow.mp4", slowdown)


@dataclass
class HandTeleopResult:
    """Outcome of `run_hand_teleop`.

    Attributes:
        trace: per-step record (frames included when recording was on).
        mapping: the table-to-world map that was used.
        yaw_rad: rotation applied about world z (see `yaw_aligning_views`).
    """

    trace: TeleopTrace
    mapping: TableToWorld
    yaw_rad: float


def run_hand_teleop(
    task: LiberoTask,
    cube: CubeProp | None,
    clip: PreparedClip,
    slowdown: float = 4.0,
    align: str = "agentview",
    scale: float = 1.0,
    table_height: bool = False,
    marker_height_m: float = 0.0,
    panel_height: int = 300,
    record: bool = True,
    window: LiveWindow | None = None,
) -> HandTeleopResult:
    """Reset the sim and drive the arm with a prepared human clip.

    Args:
        task: the sim (reset here; the cube, if any, is installed after the reset).
        cube: prop spawned on the first open-to-closed edge, or None.
        clip: output of `prepare_stand_in_clip` / `prepare_video_clip`.
        slowdown: play the human motion this many times slower.
        align: `agentview` rotates the table frame so the hand moves across that camera's
            image as it does across the phone's; `table` leaves table and world axes equal;
            a number is a yaw in degrees.
        scale: robot metres per hand metre.
        table_height: map the hand's height above the table onto the sim table (the robot
            first moves to that height) instead of pure deltas from its start pose.
        marker_height_m: height of the marker plane above the real table top; only used with
            `table_height` (see `TableToWorld.grounded`).
        panel_height: height of each display panel.
        record: keep display frames in the trace (needed for `write_run_videos`).
        window: live view; None for headless.
    """
    demo = clip.demo
    obs = reset_with_props(task, cube)
    p_world_eef0, _ = task.get_eef_pose(obs)
    if align == "agentview":
        yaw = yaw_aligning_views(demo.t_table_cam, task.camera_pose_world("agentview"))
    elif align == "table":
        yaw = 0.0
    else:
        yaw = float(np.radians(float(align)))
    if table_height:
        mapping = TableToWorld.grounded(demo.tcp_pos_table[0], p_world_eef0, yaw, scale, marker_height_m)
    else:
        mapping = TableToWorld(demo.tcp_pos_table[0], p_world_eef0, yaw, scale)
    place_twin_camera(task, mapping, demo.t_table_cam, clip.intrinsics.fovy_degrees)

    source = SequenceCommandSource(
        [
            # Hold the clip's first gripper state while approaching, so the hand-over is not a grasp.
            ApproachCommandSource(mapping.p_world_anchor, bool(demo.is_grasping[0])),
            HandCommandSource(demo, mapping, slowdown=slowdown),
        ]
    )
    session = TeleopSession(task, source, WorkspaceLimits.around(mapping.p_world_anchor), cube)
    composer = TeleopFrameComposer(
        task, default_layout(True, task.TWIN_CAMERA_NAME), panel_height, ClipView(clip, mapping)
    )
    trace = TeleopRunner(session, composer, window, keep_frames=record).run(obs)
    return HandTeleopResult(trace, mapping, yaw)


def guess_intrinsics(frame_size: tuple[int, int], focal_over_width: float = 0.8) -> CameraIntrinsics:
    """Distortion-free pinhole intrinsics from the frame size alone, for cameras not yet calibrated.

    0.8 x width corresponds to a ~65 degree horizontal field of view, typical of webcams and
    phone main cameras. Metric scale (hand size, table marker) is then only roughly right;
    calibrate with `scripts/01_calibrate_camera.py capture` for real work.

    Args:
        frame_size: (width, height) in pixels.
        focal_over_width: focal length in units of the image width.
    """
    width, height = frame_size
    return CameraIntrinsics.pinhole(width, height, fy=focal_over_width * width)


@dataclass
class LiveTeleopResult:
    """Outcome of `run_live_teleop`.

    Attributes:
        trace: per-step record (frames included when recording was on).
        source: the command source, for the final mapping and status.
    """

    trace: TeleopTrace
    source: LiveHandCommandSource


def run_live_teleop(
    task: LiberoTask,
    cube: CubeProp | None,
    frames: FrameSource,
    intrinsics: CameraIntrinsics,
    layout: MarkerLayout | None,
    align: str = "agentview",
    scale: float = 1.0,
    table_height: bool = False,
    marker_height_m: float = 0.0,
    panel_height: int = 300,
    record: bool = False,
    window: LiveWindow | None = None,
    max_steps: int | None = None,
    calibration_frames: int = 10,
    camera_pitch_deg: float = 0.0,
    on_event: Callable[[str], None] | None = None,
) -> LiveTeleopResult:
    """Reset the sim and drive the arm from a live camera until quit, stream end or `max_steps`.

    Args:
        task: the sim (reset here).
        cube: prop spawned on the first open-to-closed edge, or None.
        frames: the camera.
        intrinsics: calibration of that camera (or `guess_intrinsics`).
        layout: ArUco ids and sizes (only the table marker is used). None means *markerless*:
            the camera is the reference frame and only relative hand motion is used. The camera
            is assumed level (see `camera_pitch_deg`) and looking along the agentview camera's
            heading, so image right is the agentview's right, image up is up, and moving the
            hand away from the camera moves the robot away from the agentview camera.
            `align` and `table_height` do not apply.
        align: `agentview`, `table`, or a yaw in degrees (see `run_hand_teleop`); marker mode only.
        scale: robot metres per hand metre.
        table_height: ground z to the sim table (see `TableToWorld.grounded`); marker mode only.
        marker_height_m: height of the marker plane above the real table top, e.g. a tablet's
            thickness; only used with `table_height`.
        panel_height: height of each display panel.
        record: keep display frames in the trace.
        window: live view and keyboard (`r` re-anchors, `escape`/`q` quits); None for headless.
        max_steps: stop after this many control steps.
        calibration_frames: table-marker detections averaged before locking (marker mode).
        camera_pitch_deg: markerless only: how far the real camera's optical axis points
            below horizontal (a laptop screen tilted back looks slightly up: negative).
        on_event: called with a line whenever the table-marker or tracking status changes.

    Raises:
        ValueError: if `table_height` is asked for without markers (there is no table plane).
    """
    if layout is None and table_height:
        raise ValueError("table_height needs the table marker: without markers there is no table plane")
    obs = reset_with_props(task, cube)
    p_world_eef0, _ = task.get_eef_pose(obs)

    def make_mapping(p_ref_tcp: np.ndarray, p_world_eef: np.ndarray, t_ref_cam: np.ndarray) -> TableToWorld:
        if layout is None:
            heading = -task.camera_pose_world("agentview")[:2, 2]  # horizontal part of its forward axis
            r_world_cam = level_camera_rotation(heading, np.radians(camera_pitch_deg))
            return TableToWorld(p_ref_tcp, p_world_eef, scale=scale, rotation=r_world_cam)
        if align == "agentview":
            yaw = yaw_aligning_views(t_ref_cam, task.camera_pose_world("agentview"))
        else:
            yaw = 0.0 if align == "table" else float(np.radians(float(align)))
        if table_height:
            return TableToWorld.grounded(p_ref_tcp, p_world_eef, yaw, scale, marker_height_m)
        return TableToWorld(p_ref_tcp, p_world_eef, yaw, scale)

    def on_anchor(mapping: TableToWorld, t_ref_cam: np.ndarray) -> None:
        place_twin_camera(task, mapping, t_ref_cam, intrinsics.fovy_degrees)

    tracker = LiveHandTracker(
        HandTcpTracker(intrinsics),
        None if layout is None else MarkerPoseEstimator(intrinsics, layout),
        GraspSegmenter(0.03, 0.05),
        calibration_frames=calibration_frames,
    )
    try:
        source = LiveHandCommandSource(
            frames, tracker, make_mapping, keys=window, on_anchor=on_anchor, on_event=on_event
        )
        session = TeleopSession(task, source, WorkspaceLimits.around(p_world_eef0), cube)
        composer = TeleopFrameComposer(
            task,
            default_layout(True, task.TWIN_CAMERA_NAME),
            panel_height,
            LiveView(source, intrinsics),
        )
        trace = TeleopRunner(session, composer, window, keep_frames=record).run(obs, max_steps)
    finally:
        frames.close()
    return LiveTeleopResult(trace, source)


def write_trace_video(trace: TeleopTrace, path: Path, control_hz: float = 20.0) -> Path:
    """Write every recorded frame at the control rate (a live run has no slowed-down variant)."""
    height, width = trace.frames[0].shape[:2]
    with VideoWriter(path, control_hz, (width, height)) as writer:
        for frame in trace.frames:
            writer.write(frame)
    return path
