"""DemoExtractor against a synthetic pick-place with known ground truth (no video, no MediaPipe)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hurobo.demo import GraspSegmenter
from hurobo.extract import DemoExtractor, ExtractionConfig, ExtractionError
from hurobo.hand import HandFrameResult
from hurobo.utils.signal import fill_short_gaps
from hurobo.utils.transforms import apply_transform, invert_transform, make_transform

FPS = 30.0
DEPTH_ERROR = 1.25  # simulated MediaPipe hand-size error: depth too large by 25 %
P_BOWL = np.array([0.10, 0.00, 0.03])
P_PLATE = np.array([-0.10, 0.10, 0.03])


def _camera_looking_at_table() -> np.ndarray:
    """T_cam_table for a tripod camera 0.7 m away, 0.4 m up, looking at the table origin."""
    position_table = np.array([0.0, -0.7, 0.4])
    forward = -position_table / np.linalg.norm(position_table)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    t_table_cam = make_transform(
        Rotation.from_matrix(np.column_stack([right, down, forward])), position_table
    )
    return invert_transform(t_table_cam)


# Piecewise-linear pick-place with dwells (a person pauses while closing / opening the fingers).
_WAYPOINT_T_S = np.array([0.0, 1.0, 1.3, 2.3, 2.6, 3.5])
_WAYPOINTS_TABLE = np.array(
    [[0.15, -0.10, 0.20], P_BOWL, P_BOWL, P_PLATE, P_PLATE, P_PLATE + [0.0, 0.0, 0.15]]
)
_GRASP_T_S, _RELEASE_T_S = 1.1, 2.4
DURATION_S = 3.5


def _true_trajectory_table(t_s: np.ndarray) -> np.ndarray:
    return np.column_stack([np.interp(t_s, _WAYPOINT_T_S, _WAYPOINTS_TABLE[:, d]) for d in range(3)])


def _aperture(t_s: np.ndarray) -> np.ndarray:
    return np.where((t_s >= _GRASP_T_S) & (t_s < _RELEASE_T_S), 0.015, 0.08)


class FakeTracker:
    """Replays the ground truth through the camera with a depth error and short dropouts."""

    def __init__(self, tcp_cam: np.ndarray, aperture: np.ndarray, dropped: set[int]) -> None:
        self._tcp_cam, self._aperture, self._dropped = tcp_cam, aperture, dropped
        self._i = 0

    def process_frame(self, image_bgr: np.ndarray, timestamp_ms: int) -> HandFrameResult | None:
        i, self._i = self._i, self._i + 1
        if i in self._dropped:
            return None
        return HandFrameResult(np.eye(4), self._tcp_cam[i], float(self._aperture[i]))


class FakeMarkers:
    def __init__(self, t_cam_table: np.ndarray) -> None:
        self._poses = {
            "table": t_cam_table,
            "bowl": make_transform(Rotation.identity(), P_BOWL),
            "plate": make_transform(Rotation.identity(), P_PLATE),
        }

    def estimate(self, image_bgr: np.ndarray) -> dict[str, np.ndarray]:
        return self._poses


def _run(dropped: set[int], config: ExtractionConfig | None = None):
    t_s = np.arange(0, DURATION_S, 1 / FPS)
    t_cam_table = _camera_looking_at_table()
    tcp_cam = apply_transform(t_cam_table, _true_trajectory_table(t_s)) * DEPTH_ERROR
    extractor = DemoExtractor(
        FakeTracker(tcp_cam, _aperture(t_s), dropped),
        FakeMarkers(t_cam_table),
        GraspSegmenter(close_threshold_m=0.03, open_threshold_m=0.05),
        config or ExtractionConfig(),
    )
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    return extractor.extract_frames(((frame, t) for t in t_s), Path("synthetic.mp4"))


def test_anchor_correction_recovers_true_depth_and_trajectory() -> None:
    demo, report = _run(dropped={20, 21, 22, 70})
    assert report.usable, report.reasons
    assert report.anchor_scale == pytest.approx(1 / DEPTH_ERROR, abs=0.02)
    assert report.grasp_to_bowl_m < 0.01
    assert report.release_to_plate_xy_m < 0.03
    truth = _true_trajectory_table(demo.timestamps_s)
    # Savitzky-Golay rounds the corners, so compare away from the grasp and release.
    interior = (np.abs(demo.timestamps_s - 1.15) > 0.3) & (np.abs(demo.timestamps_s - 2.45) > 0.3)
    assert np.abs(demo.tcp_pos_table - truth)[interior].max() < 0.01


def test_height_anchor_recovers_depth_without_a_marker_at_the_grasp() -> None:
    # The bowl marker is moved away from the grasp point (as when it lies on a pad beside the bowl),
    # so only the known grasp height can fix the depth.
    demo, report = _run(dropped=set(), config=ExtractionConfig(grasp_height_m=float(P_BOWL[2]) + 0.0))
    assert report.anchor_scale == pytest.approx(1 / DEPTH_ERROR, abs=0.02)
    truth = _true_trajectory_table(demo.timestamps_s)
    interior = (np.abs(demo.timestamps_s - 1.15) > 0.3) & (np.abs(demo.timestamps_s - 2.45) > 0.3)
    assert np.abs(demo.tcp_pos_table - truth)[interior].max() < 0.01


def test_long_dropout_flags_clip_unusable() -> None:
    _, report = _run(dropped=set(range(40, 60)))
    assert not report.usable
    assert any("hand lost" in reason for reason in report.reasons)


def test_missing_table_marker_raises() -> None:
    class NoMarkers:
        def estimate(self, image_bgr: np.ndarray) -> None:
            return None

    extractor = DemoExtractor(
        FakeTracker(np.zeros((2, 3)), np.zeros(2), set()), NoMarkers(), GraspSegmenter()
    )
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    with pytest.raises(ExtractionError):
        extractor.extract_frames([(frame, 0.0), (frame, 0.1)], Path("x.mp4"))


def test_fill_short_gaps_only_bridges_short_runs() -> None:
    t = np.arange(10) / 10.0
    v = np.arange(10, dtype=float)
    v[[2, 3]] = np.nan  # 0.3 s bracket
    v[6] = np.nan  # 0.2 s bracket
    filled, still_missing = fill_short_gaps(t, v, max_gap_s=0.25)
    assert still_missing[[2, 3]].all() and not still_missing[6]
    assert filled[6] == pytest.approx(6.0)
