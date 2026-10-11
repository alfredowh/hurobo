"""Turns one phone clip into a table-frame `HumanDemo` plus a quality report."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
from scipy.spatial.transform import Rotation

from ego2robo.camera import CameraIntrinsics
from ego2robo.demo import GraspSegmenter, HumanDemo
from ego2robo.hand import HandFrameResult, HandTcpTracker
from ego2robo.markers import MarkerLayout, MarkerPoseEstimator
from ego2robo.utils.signal import fill_short_gaps, resample_to_rate, smooth_trajectory
from ego2robo.utils.transforms import apply_transform, invert_transform, make_transform
from ego2robo.utils.video import read_frames_with_timestamps


class HandTracker(Protocol):
    """What `DemoExtractor` needs from a hand tracker (`HandTcpTracker` satisfies it)."""

    def process_frame(self, image_bgr: np.ndarray, timestamp_ms: int) -> HandFrameResult | None: ...


class MarkerEstimator(Protocol):
    """What `DemoExtractor` needs from a marker detector (`MarkerPoseEstimator` satisfies it)."""

    def estimate(self, image_bgr: np.ndarray) -> dict[str, np.ndarray] | None: ...


class ExtractionError(RuntimeError):
    """Raised when a clip has no usable world frame or layout (nothing to salvage)."""


@dataclass
class ExtractionConfig:
    """Thresholds for extraction and for the per-clip quality gates.

    Attributes:
        target_hz: output sample rate of the `HumanDemo` (PLAN.md §4 step 3).
        max_gap_s: longest hand-detection dropout that is interpolated.
        layout_window_s: bowl/plate poses are taken from the first this-many seconds,
            before the hand occludes or moves them.
        smooth_window: Savitzky-Golay window (samples at `target_hz`, odd).
        anchor_scale_bounds: accepted range of the grasp-anchor depth scale; outside
            it the clip is flagged (the hand model or the intrinsics are off).
        min_hand_fraction: minimum share of frames with a hand detection, within the
            span from the first to the last detection.
        max_grasp_to_bowl_m: max TCP-to-bowl-marker distance at grasp, after anchoring.
        max_release_to_plate_xy_m: max TCP-to-plate-marker distance (table plane) at release.
    """

    target_hz: float = 20.0
    max_gap_s: float = 0.4
    layout_window_s: float = 0.6
    smooth_window: int = 9
    anchor_scale_bounds: tuple[float, float] = (0.6, 1.6)
    min_hand_fraction: float = 0.85
    max_grasp_to_bowl_m: float = 0.10
    max_release_to_plate_xy_m: float = 0.12


@dataclass
class ClipReport:
    """Per-clip quality summary; `usable` is False if any gate failed (see `reasons`).

    Clips are reported, never silently dropped, so the README can state how many
    were excluded and why.
    """

    source_video: Path
    num_frames: int = 0
    duration_s: float = 0.0
    hand_fraction: float = 0.0
    table_marker_fraction: float = 0.0
    longest_hand_gap_s: float = 0.0
    unfilled_gap_frames: int = 0
    grasp_found: bool = False
    release_found: bool = False
    anchor_scale: float | None = None
    grasp_to_bowl_m: float | None = None
    release_to_plate_xy_m: float | None = None
    reasons: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        """True when every quality gate passed."""
        return not self.reasons


def mean_transform(transforms: list[np.ndarray]) -> np.ndarray:
    """Average rigid transforms: mean translation, chordal-mean rotation.

    Args:
        transforms: non-empty list of 4x4 transforms of the same quantity.

    Returns:
        4x4 transform.
    """
    rotation = Rotation.from_matrix(np.stack([t[:3, :3] for t in transforms])).mean()
    translation = np.mean([t[:3, 3] for t in transforms], axis=0)
    return make_transform(rotation, translation)


class DemoExtractor:
    """Runs hand tracking and marker detection over a clip and emits a `HumanDemo`.

    Assumes a static camera (tripod, PLAN.md §4): T_cam_table is one robust
    average over the clip rather than a per-frame quantity, which removes
    marker-PnP jitter from the hand trajectory.
    """

    def __init__(
        self,
        tracker: HandTracker,
        marker_estimator: MarkerEstimator,
        segmenter: GraspSegmenter,
        config: ExtractionConfig | None = None,
    ) -> None:
        """Wire up the perception components.

        Args:
            tracker: per-frame hand TCP tracker (one instance per clip: VIDEO mode
                needs monotonically increasing timestamps).
            marker_estimator: per-frame ArUco pose estimator.
            segmenter: grasp/release segmenter with the hysteresis thresholds.
            config: thresholds; defaults to `ExtractionConfig()`.
        """
        self._tracker = tracker
        self._markers = marker_estimator
        self._segmenter = segmenter
        self._config = config or ExtractionConfig()

    def extract_video(self, video_path: Path) -> tuple[HumanDemo, ClipReport]:
        """Extract a demo from a video file. See `extract_frames`."""
        return self.extract_frames(read_frames_with_timestamps(video_path), video_path)

    def extract_frames(
        self, frames: Iterable[tuple[np.ndarray, float]], source_video: Path
    ) -> tuple[HumanDemo, ClipReport]:
        """Extract a demo from (BGR frame, time in s) pairs.

        Args:
            frames: frames in time order with strictly increasing timestamps.
            source_video: recorded in the demo and report for traceability.

        Returns:
            (demo, report). Check `report.usable` before using the demo.

        Raises:
            ExtractionError: if the table, bowl or plate marker was never seen
                (no world frame or layout exists, so there is no demo to build).
        """
        cfg = self._config
        times_s: list[float] = []
        tcp_cam: list[np.ndarray] = []
        aperture: list[float] = []
        t_cam_table_all: list[np.ndarray] = []
        layout_poses: dict[str, list[np.ndarray]] = {}
        last_ms = -1
        for frame, t_s in frames:
            times_s.append(t_s)
            last_ms = max(int(round(t_s * 1000)), last_ms + 1)
            hand = self._tracker.process_frame(frame, last_ms)
            tcp_cam.append(hand.tcp_pos_cam if hand else np.full(3, np.nan))
            aperture.append(hand.pinch_aperture_m if hand else np.nan)

            poses = self._markers.estimate(frame)
            if poses is None:
                continue
            t_cam_table_all.append(poses["table"])
            if t_s - times_s[0] <= cfg.layout_window_s:
                for role, t_table_role in poses.items():
                    if role != "table":
                        layout_poses.setdefault(role, []).append(t_table_role)

        if not times_s:
            raise ExtractionError(f"{source_video}: no frames decoded")
        if not t_cam_table_all:
            raise ExtractionError(f"{source_video}: table marker never detected")
        for role in ("bowl", "plate"):
            if role not in layout_poses:
                raise ExtractionError(
                    f"{source_video}: {role} marker not seen in the first {cfg.layout_window_s}s"
                )

        times = np.asarray(times_s)
        report = ClipReport(
            source_video=source_video,
            num_frames=len(times),
            duration_s=float(times[-1] - times[0]),
            table_marker_fraction=len(t_cam_table_all) / len(times),
        )
        t_cam_table = mean_transform(t_cam_table_all)
        t_table_cam = invert_transform(t_cam_table)
        t_table_bowl = mean_transform(layout_poses["bowl"])
        t_table_plate = mean_transform(layout_poses["plate"])

        hand_signal = np.column_stack([np.asarray(tcp_cam), np.asarray(aperture)])
        filled, missing = fill_short_gaps(times, hand_signal, cfg.max_gap_s)
        valid_idx = np.flatnonzero(~np.isnan(hand_signal).any(axis=1))
        if len(valid_idx) < 2:
            report.reasons.append("hand detected in fewer than 2 frames")
            return self._empty_demo(source_video, t_cam_table, t_table_bowl, t_table_plate), report
        # Trim to the span where the hand is in view: nothing is known outside it.
        first, last = int(valid_idx[0]), int(valid_idx[-1])
        times, filled, missing = times[first : last + 1], filled[first : last + 1], missing[first : last + 1]
        # Measured inside the span: frames before the hand enters / after it leaves are not dropouts.
        report.hand_fraction = len(valid_idx) / len(times)
        report.unfilled_gap_frames = int(missing.sum())
        report.longest_hand_gap_s = self._longest_gap_s(times, missing)
        if report.unfilled_gap_frames:
            # Bridged only so arrays stay finite; the clip is flagged unusable below.
            for d in range(filled.shape[1]):
                keep = ~missing
                filled[:, d] = np.interp(times, times[keep], filled[keep, d])

        new_times, resampled = resample_to_rate(times, filled, cfg.target_hz)
        smoothed = smooth_trajectory(resampled, window_length=cfg.smooth_window)
        tcp_cam_s, aperture_s = smoothed[:, :3], smoothed[:, 3]

        is_grasping = self._segmenter.segment(aperture_s)
        grasp_idx, release_idx = GraspSegmenter.find_transition_frames(is_grasping)
        report.grasp_found, report.release_found = grasp_idx is not None, release_idx is not None

        if grasp_idx is not None:
            tcp_cam_s, report.anchor_scale = self._anchor_correct(
                tcp_cam_s, grasp_idx, t_cam_table, t_table_bowl
            )
        tcp_table = apply_transform(t_table_cam, tcp_cam_s)

        if grasp_idx is not None:
            report.grasp_to_bowl_m = float(np.linalg.norm(tcp_table[grasp_idx] - t_table_bowl[:3, 3]))
        if release_idx is not None:
            report.release_to_plate_xy_m = float(
                np.linalg.norm(tcp_table[release_idx, :2] - t_table_plate[:2, 3])
            )
        self._apply_gates(report)

        demo = HumanDemo(
            timestamps_s=new_times - new_times[0],
            tcp_pos_table=tcp_table,
            pinch_aperture_m=aperture_s,
            is_grasping=is_grasping,
            t_table_bowl=t_table_bowl,
            t_table_plate=t_table_plate,
            t_table_cam=t_table_cam,
            source_video=source_video,
            grasp_frame_idx=grasp_idx,
            release_frame_idx=release_idx,
            source_start_time_s=float(new_times[0]),
        )
        return demo, report

    @staticmethod
    def _anchor_correct(
        tcp_cam: np.ndarray, grasp_idx: int, t_cam_table: np.ndarray, t_table_bowl: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """Scale the TCP trajectory along each camera ray so the grasp lands on the bowl marker.

        MediaPipe's world landmarks carry an assumed hand size, and solvePnP depth is
        proportional to it, while the 2D landmarks (hence the pixel ray) are accurate.
        A single scale s about the camera origin therefore corrects depth without
        moving the TCP in the image (PLAN.md §4 step 4).

        Args:
            tcp_cam: shape (N, 3), TCP in the camera frame.
            grasp_idx: grasp-onset sample index.
            t_cam_table: 4x4 camera-from-table transform of the clip.
            t_table_bowl: 4x4 bowl pose in the table frame.

        Returns:
            (corrected tcp_cam, s). s is the least-squares scale at the grasp.
        """
        window = tcp_cam[max(grasp_idx - 1, 0) : grasp_idx + 2]
        p_tcp_cam = np.median(window, axis=0)
        p_bowl_cam = apply_transform(t_cam_table, t_table_bowl[:3, 3])
        scale = float(p_tcp_cam @ p_bowl_cam / (p_tcp_cam @ p_tcp_cam))
        return tcp_cam * scale, scale

    def _apply_gates(self, report: ClipReport) -> None:
        """Append a reason to `report.reasons` for every failed quality gate."""
        cfg = self._config
        if report.hand_fraction < cfg.min_hand_fraction:
            report.reasons.append(f"hand detected in {report.hand_fraction:.0%} of frames")
        if report.unfilled_gap_frames:
            report.reasons.append(f"hand lost for {report.longest_hand_gap_s:.2f}s (> {cfg.max_gap_s}s)")
        if not report.grasp_found:
            report.reasons.append("no grasp detected")
        elif not report.release_found:
            report.reasons.append("no release detected")
        low, high = cfg.anchor_scale_bounds
        if report.anchor_scale is not None and not low <= report.anchor_scale <= high:
            report.reasons.append(f"anchor scale {report.anchor_scale:.2f} outside [{low}, {high}]")
        if report.grasp_to_bowl_m is not None and report.grasp_to_bowl_m > cfg.max_grasp_to_bowl_m:
            report.reasons.append(f"grasp {report.grasp_to_bowl_m:.2f} m from bowl marker")
        if (
            report.release_to_plate_xy_m is not None
            and report.release_to_plate_xy_m > cfg.max_release_to_plate_xy_m
        ):
            report.reasons.append(f"release {report.release_to_plate_xy_m:.2f} m from plate marker")

    @staticmethod
    def _longest_gap_s(times: np.ndarray, missing: np.ndarray) -> float:
        """Duration of the longest run of True in `missing`, bracket to bracket."""
        longest, start = 0.0, None
        for i, is_missing in enumerate(missing):
            if is_missing and start is None:
                start = i
            elif not is_missing and start is not None:
                # The span is trimmed to valid samples at both ends, so start >= 1.
                longest = max(longest, times[i] - times[start - 1])
                start = None
        return float(longest)

    @staticmethod
    def _empty_demo(
        source_video: Path, t_cam_table: np.ndarray, t_table_bowl: np.ndarray, t_table_plate: np.ndarray
    ) -> HumanDemo:
        """A zero-length placeholder so callers get a (demo, report) pair even for dead clips."""
        return HumanDemo(
            timestamps_s=np.zeros(0),
            tcp_pos_table=np.zeros((0, 3)),
            pinch_aperture_m=np.zeros(0),
            is_grasping=np.zeros(0, dtype=bool),
            t_table_bowl=t_table_bowl,
            t_table_plate=t_table_plate,
            t_table_cam=invert_transform(t_cam_table),
            source_video=source_video,
        )


def extract_demo(
    video_path: Path,
    intrinsics: CameraIntrinsics,
    layout: MarkerLayout,
    segmenter: GraspSegmenter | None = None,
    config: ExtractionConfig | None = None,
) -> tuple[HumanDemo, ClipReport]:
    """Extract one clip with the standard MediaPipe + ArUco perception stack.

    Builds a fresh `HandTcpTracker` per call: its VIDEO running mode needs
    monotonically increasing timestamps, so a tracker can't span clips.

    Args:
        video_path: phone clip.
        intrinsics: calibration of the camera that filmed it.
        layout: which ArUco ids mark the table and the objects.
        segmenter: grasp/release thresholds; defaults to 3 cm close / 5 cm open.
        config: QC gates; defaults to `ExtractionConfig()`.

    Returns:
        The demo and its quality report.

    Raises:
        ExtractionError: if the clip yields no usable samples.
    """
    with HandTcpTracker(intrinsics) as tracker:
        extractor = DemoExtractor(
            tracker,
            MarkerPoseEstimator(intrinsics, layout),
            segmenter or GraspSegmenter(0.03, 0.05),
            config,
        )
        return extractor.extract_video(video_path)
