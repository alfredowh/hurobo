"""Tracking views: interactive summary figure, TCP overlay video and the synced dashboard video."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ego2robo.camera import CameraIntrinsics
from ego2robo.hand import HandFrameResult, TrackedFrame
from ego2robo.utils.dashboard_video import SyncedDashboard, tracking_dashboard, write_tracking_dashboard_video
from ego2robo.utils.video import read_frames, write_tcp_overlay_video
from ego2robo.utils.viewer import tracking_figure

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
