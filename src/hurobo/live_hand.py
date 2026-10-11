"""Online hand tracking for live teleoperation: table-frame TCP and gripper state, frame by frame."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from hurobo.camera_source import FrameSource
from hurobo.demo import GraspSegmenter
from hurobo.extract import HandTracker, MarkerEstimator, mean_transform
from hurobo.hand import HandFrameResult
from hurobo.teleop import KeySource, TableToWorld, TeleopCommand
from hurobo.utils.signal import ExponentialSmoother
from hurobo.utils.transforms import apply_transform, invert_transform

#: (hand TCP in the reference frame, robot end effector in the world, T_ref_cam) -> reference-to-world map.
MappingFactory = Callable[[np.ndarray, np.ndarray, np.ndarray], TableToWorld]


@dataclass
class LiveHandSample:
    """What one camera frame yielded.

    Attributes:
        table_ready: True once the table marker pose has been locked.
        hand: the raw tracker result, or None if no hand was found.
        p_table_tcp: shape (3,) smoothed TCP in the table frame, or None without hand/table.
        closed: gripper state after hysteresis (holds its last value while the hand is lost).
        t_cam_table: the locked camera-from-table pose, or None before `table_ready`.
    """

    table_ready: bool
    hand: HandFrameResult | None
    p_table_tcp: np.ndarray | None
    closed: bool
    t_cam_table: np.ndarray | None


class LiveHandTracker:
    """Per-frame hand TCP in the table frame, assuming a static camera.

    The first `calibration_frames` table-marker detections are averaged and then locked
    (the batch extractor averages over the whole clip for the same reason: PnP jitter on the
    marker would otherwise move the whole hand trajectory). Move the camera and call `recalibrate`.

    Without a marker estimator (`markers=None`) there is no table: the camera frame is the
    reference frame (`T_cam_table` is the identity from the start), so the reported "table-frame"
    TCP is the TCP in camera axes. That is enough for relative-pose teleoperation.
    """

    def __init__(
        self,
        tracker: HandTracker,
        markers: MarkerEstimator | None,
        segmenter: GraspSegmenter,
        calibration_frames: int = 10,
        smoothing_alpha: float = 0.5,
    ) -> None:
        """Wire the perception components.

        Args:
            tracker: per-frame hand tracker (needs monotonically increasing timestamps).
            markers: per-frame ArUco estimator whose `table` entry defines the reference frame,
                or None to use the camera frame as the reference.
            segmenter: pinch hysteresis.
            calibration_frames: table detections to average before locking.
            smoothing_alpha: weight of the newest TCP sample (see `ExponentialSmoother`).
        """
        self._tracker = tracker
        self._markers = markers
        self._segmenter = segmenter
        self._calibration_frames = calibration_frames
        self._smoother = ExponentialSmoother(smoothing_alpha)
        self._poses: list[np.ndarray] = []
        self._t_cam_table: np.ndarray | None = np.eye(4) if markers is None else None
        self._closed = False
        self._last_ms = -1

    @property
    def table_ready(self) -> bool:
        """True once the table pose is locked."""
        return self._t_cam_table is not None

    @property
    def calibration_progress(self) -> tuple[int, int]:
        """(detections collected, needed)."""
        return len(self._poses), self._calibration_frames

    def recalibrate(self) -> None:
        """Discard the locked table pose and collect it again (after moving the camera)."""
        self._poses.clear()
        self._t_cam_table = np.eye(4) if self._markers is None else None
        self._smoother.reset()

    def process(self, frame_bgr: np.ndarray, t_s: float) -> LiveHandSample:
        """Track one frame.

        Args:
            frame_bgr: the newest camera frame.
            t_s: its time in seconds (any origin, increasing).
        """
        if self._t_cam_table is None and self._markers is not None:
            poses = self._markers.estimate(frame_bgr)
            if poses is not None:
                self._poses.append(poses["table"])
                if len(self._poses) >= self._calibration_frames:
                    self._t_cam_table = mean_transform(self._poses)
        self._last_ms = max(int(round(t_s * 1000)), self._last_ms + 1)
        hand = self._tracker.process_frame(frame_bgr, self._last_ms)
        p_table = None
        if hand is not None:
            self._closed = self._segmenter.next_state(hand.pinch_aperture_m, self._closed)
            if self._t_cam_table is not None:
                p_table = self._smoother.update(
                    apply_transform(invert_transform(self._t_cam_table), hand.tcp_pos_cam)
                )
        return LiveHandSample(self.table_ready, hand, p_table, self._closed, self._t_cam_table)


class LiveHandCommandSource:
    """Turns a live camera into end-effector commands (a `teleop.CommandSource`).

    The first frame with the table locked and a hand in view anchors the mapping: that hand
    position is the robot's current position, everything after is a delta (same convention as
    the clip replay). Press `r` to re-anchor (clutch: move your hand back to a comfortable
    spot, press `r`, carry on). The robot holds still while the table or hand is missing.
    `escape` or `q` ends the session, as does the camera stream ending.
    """

    QUIT_KEYS = ("escape", "q")

    def __init__(
        self,
        frames: FrameSource,
        tracker: LiveHandTracker,
        make_mapping: MappingFactory,
        keys: KeySource | None = None,
        on_anchor: Callable[[TableToWorld, np.ndarray], None] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        """Configure the source.

        Args:
            frames: the camera.
            tracker: online perception.
            make_mapping: builds the reference-to-world map at (re)anchoring from
                (hand TCP in the reference frame, robot end effector in the world, `T_ref_cam`);
                this is where yaw alignment, scale and height grounding are decided.
            keys: keyboard state for `r` and quit keys.
            on_anchor: called with (mapping, T_table_cam) whenever the mapping is (re)built,
                e.g. to move the twin camera.
            clock: time source in seconds, injectable for tests.
        """
        self._frames = frames
        self._tracker = tracker
        self._make_mapping = make_mapping
        self._keys = keys
        self._on_anchor = on_anchor
        self._clock = clock or time.monotonic
        self._t0 = self._clock()
        self._mapping: TableToWorld | None = None
        self._last_index = -1
        self._sample: LiveHandSample | None = None
        self._frame: np.ndarray | None = None
        self._last: TeleopCommand | None = None

    @property
    def mapping(self) -> TableToWorld | None:
        """The table-to-world map once anchored, else None."""
        return self._mapping

    @property
    def latest_frame(self) -> np.ndarray | None:
        """The newest camera frame seen."""
        return self._frame

    @property
    def latest_sample(self) -> LiveHandSample | None:
        """The newest tracking result."""
        return self._sample

    @property
    def status(self) -> str:
        """One line for the display: what the pipeline is waiting for, or that it is tracking."""
        if self._sample is None:
            return "waiting for camera"
        if not self._sample.table_ready:
            got, need = self._tracker.calibration_progress
            return f"looking for the table marker ({got}/{need})"
        if self._mapping is None:
            return "show your hand to start"
        if self._sample.hand is None:
            return "hand lost: holding"
        return "tracking"

    def next_command(self, p_world_eef: np.ndarray) -> TeleopCommand | None:
        """See `teleop.CommandSource`."""
        if self._keys is not None and any(self._keys.consume(k) for k in self.QUIT_KEYS):
            return None
        if self._frames.ended:
            return None
        recenter = self._keys is not None and self._keys.consume("r")
        newest = self._frames.latest()
        if newest is not None and newest[1] != self._last_index:
            self._frame, self._last_index = newest[0], newest[1]
            self._sample = self._tracker.process(self._frame, self._clock() - self._t0)
        sample = self._sample
        if sample is not None and sample.p_table_tcp is not None:
            anchoring = self._mapping is None or recenter
            if anchoring:
                self._anchor(sample, p_world_eef)
            target = self._mapping.point_to_world(sample.p_table_tcp)
            self._last = TeleopCommand(target, sample.closed, starts_stream=anchoring)
        if self._last is None:
            return TeleopCommand(np.asarray(p_world_eef, dtype=float).copy(), False)
        return self._last

    def _anchor(self, sample: LiveHandSample, p_world_eef: np.ndarray) -> None:
        t_table_cam = invert_transform(sample.t_cam_table)
        self._mapping = self._make_mapping(sample.p_table_tcp, p_world_eef, t_table_cam)
        if self._on_anchor is not None:
            self._on_anchor(self._mapping, t_table_cam)
