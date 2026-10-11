"""Tracking views: interactive summary figure, TCP overlay video and the synced dashboard video."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ego2robo.camera import CameraIntrinsics
from ego2robo.hand import HandFrameResult, TrackedFrame
from ego2robo.utils.dashboard_video import SyncedDashboard, tracking_dashboard, write_tracking_dashboard_video
from ego2robo.utils.hand_landmarks import HAND_CONNECTIONS, INDEX_TIP, LANDMARK_NAMES, THUMB_TIP
from ego2robo.utils.transforms import apply_transform, make_transform
from ego2robo.utils.video import THUMB_BGR, draw_tcp_overlay, read_frames, write_tcp_overlay_video
from ego2robo.utils.viewer import pinch_figure, tracking_figure

INTRINSICS = CameraIntrinsics.pinhole(160, 120, fy=150.0)


def _tracked(num_frames: int = 6) -> list[TrackedFrame]:
    frames = []
    for i in range(num_frames):
        # Frame 2 has no hand: the figure must show a gap, the video a "no hand" caption.
        result = None
        if i != 2:
            result = HandFrameResult(
                t_cam_hand=np.eye(4),
                tcp_pos_cam=np.array([0.01 * i, 0.0, 0.5]),
                pinch_aperture_m=0.06 - 0.005 * i,
                reprojection_error_px=1.0,
            )
        frames.append(TrackedFrame(np.full((120, 160, 3), 40, np.uint8), i / 10.0, result))
    return frames


def test_tracking_figure_has_gap_for_missing_hand() -> None:
    figure = tracking_figure(_tracked())
    assert len(figure.data) == 3
    assert np.isnan(figure.data[0].y[2]) and np.isfinite(figure.data[0].y[3])


def test_overlay_video_has_all_frames_and_marks_tcp(tmp_path: Path) -> None:
    video = write_tcp_overlay_video(_tracked(), INTRINSICS, tmp_path / "overlay.mp4", fps=10.0)
    frames = list(read_frames(video))
    assert len(frames) == 6
    # The red cross is drawn at the projected TCP; the untouched background is grey 40.
    u, v = INTRINSICS.project(np.array([0.0, 0.0, 0.5]))
    assert frames[0][round(v), round(u), 2] > 150


def test_dashboard_video_has_all_frames_and_moves_with_time(tmp_path: Path) -> None:
    tracked = _tracked()
    video = write_tracking_dashboard_video(
        tracked,
        INTRINSICS,
        tmp_path / "dashboard.mp4",
        fps=10.0,
        close_threshold_m=0.03,
        open_threshold_m=0.05,
    )
    frames = list(read_frames(video))
    assert len(frames) == len(tracked)
    assert frames[0].shape[:2] == (720, 1280)
    # Every panel advances with the clip, so consecutive frames differ.
    assert np.abs(frames[0].astype(int) - frames[-1].astype(int)).mean() > 0.1


def test_dashboard_rejects_clip_without_hand() -> None:
    blank = [TrackedFrame(np.zeros((120, 160, 3), np.uint8), i / 10.0, None) for i in range(3)]
    with pytest.raises(ValueError, match="no valid TCP"):
        tracking_dashboard(blank, INTRINSICS)


def test_dashboard_rejects_mismatched_lengths() -> None:
    frames = [np.zeros((12, 16, 3), np.uint8)] * 3
    with pytest.raises(ValueError, match="same length"):
        SyncedDashboard(frames, np.arange(3.0), np.zeros((2, 3)), np.zeros(3))


def _tracked_with_landmarks(apertures_m: list[float]) -> list[TrackedFrame]:
    """A hand 0.5 m in front of the camera whose thumb and index tips are `aperture` apart."""
    rng = np.random.default_rng(0)
    frames = []
    for i, aperture in enumerate(apertures_m):
        landmarks_hand = rng.normal(scale=0.03, size=(21, 3))
        landmarks_hand[THUMB_TIP] = [-aperture / 2, 0.0, 0.0]
        landmarks_hand[INDEX_TIP] = [aperture / 2, 0.0, 0.0]
        t_cam_hand = make_transform(Rotation.identity(), np.array([0.0, 0.0, 0.5]))
        landmarks_cam = apply_transform(t_cam_hand, landmarks_hand)
        result = HandFrameResult(
            t_cam_hand=t_cam_hand,
            tcp_pos_cam=apply_transform(t_cam_hand, np.zeros(3)),
            pinch_aperture_m=aperture,
            landmarks_hand=landmarks_hand,
            landmarks_image_px=INTRINSICS.project(landmarks_cam),
        )
        frames.append(TrackedFrame(np.full((120, 160, 3), 40, np.uint8), i / 10.0, result))
    return frames


def test_overlay_draws_whole_hand_with_thumb_tip_highlighted() -> None:
    item = _tracked_with_landmarks([0.06])[0]
    frame = draw_tcp_overlay(item, INTRINSICS)
    u, v = item.result.landmarks_image_px[THUMB_TIP]
    # 5 px above the tip: inside its dot but clear of the TCP cross, whose arm runs through the tip row.
    assert tuple(frame[round(v) - 5, round(u)]) == THUMB_BGR
    # Without landmarks the overlay falls back to the single TCP cross, so the bones are the difference.
    bare = HandFrameResult(item.result.t_cam_hand, item.result.tcp_pos_cam, 0.06)
    plain = draw_tcp_overlay(TrackedFrame(item.frame_bgr, 0.0, bare), INTRINSICS)
    assert (frame != plain).any(axis=2).sum() > 200


def test_landmark_topology_is_the_21_point_mediapipe_hand() -> None:
    assert len(LANDMARK_NAMES) == 21 and len(HAND_CONNECTIONS) == 21
    assert LANDMARK_NAMES[THUMB_TIP] == "thumb_tip" and LANDMARK_NAMES[INDEX_TIP] == "index_tip"
    assert {i for pair in HAND_CONNECTIONS for i in pair} == set(range(21))


def test_pinch_figure_marks_grasp_state_per_frame() -> None:
    # open (6 cm) -> closing (4 cm, inside the band) -> closed (2 cm) -> back in band (4 cm: stays closed).
    tracked = _tracked_with_landmarks([0.06, 0.04, 0.02, 0.04])
    tracked.append(TrackedFrame(np.zeros((120, 160, 3), np.uint8), 0.4, None))
    figure = pinch_figure(tracked, close_threshold_m=0.03, open_threshold_m=0.05)
    assert len(figure.frames) == len(tracked)
    colours = [frame.data[-1].marker.color for frame in figure.frames]
    assert colours[:4] == ["royalblue", "royalblue", "crimson", "crimson"]
    assert figure.frames[-1].data[-1].y == (None,)
    aperture_curve = next(
        trace for trace in figure.data if trace.name == "aperture" and trace.mode != "lines"
    )
    assert np.allclose(aperture_curve.y[:4], [6, 4, 2, 4])


def test_pinch_figure_rejects_clip_without_landmarks() -> None:
    with pytest.raises(ValueError, match="landmarks"):
        pinch_figure(_tracked(), 0.03, 0.05)


def test_dashboard_with_hand_landmarks_renders_every_frame(tmp_path: Path) -> None:
    tracked = _tracked_with_landmarks([0.06, 0.05, 0.03, 0.02])
    video = write_tracking_dashboard_video(
        tracked, INTRINSICS, tmp_path / "hand.mp4", fps=10.0, close_threshold_m=0.03, open_threshold_m=0.05
    )
    assert len(list(read_frames(video))) == 4
