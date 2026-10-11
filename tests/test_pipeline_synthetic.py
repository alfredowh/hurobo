"""Full perception chain (real MediaPipe + real ArUco) on a real hand clip with synthetic markers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hurobo.camera import CameraIntrinsics
from hurobo.demo import GraspSegmenter
from hurobo.extract import DemoExtractor
from hurobo.hand import HandTcpTracker
from hurobo.markers import MarkerLayout, MarkerPoseEstimator
from hurobo.utils.synthetic import render_marker_clip
from hurobo.utils.transforms import look_at_origin_pose, make_transform

CLIP_DIR = Path("data/raw/overview_gif/first_clip")
MARKER_SIDE_M = 0.06
INTRINSICS = CameraIntrinsics(
    camera_matrix=np.array([[800.0, 0, 400.0], [0, 800.0, 225.0], [0, 0, 1]]),
    dist_coeffs=np.zeros(5),
    image_width=800,
    image_height=450,
)


@pytest.mark.skipif(not CLIP_DIR.exists(), reason="sample clip frames not present")
def test_marker_poses_recovered_and_extractor_runs(tmp_path: Path) -> None:
    t_table_cam = look_at_origin_pose(np.array([0.0, -0.45, 0.40]))
    t_table_bowl = make_transform(Rotation.from_euler("z", 20, degrees=True), [-0.12, 0.02, 0.0])
    t_table_plate = make_transform(Rotation.from_euler("z", -10, degrees=True), [0.13, 0.04, 0.0])
    video_path = tmp_path / "synthetic.mp4"
    render_marker_clip(
        sorted(CLIP_DIR.glob("*.png"))[20:],
        video_path,
        INTRINSICS,
        t_table_cam,
        {20: np.eye(4), 21: t_table_bowl, 22: t_table_plate},
        MARKER_SIDE_M,
        fps=10.0,
    )

    layout = MarkerLayout(20, MARKER_SIDE_M, {"bowl": 21, "plate": 22}, MARKER_SIDE_M)
    with HandTcpTracker(INTRINSICS) as tracker:
        extractor = DemoExtractor(
            tracker, MarkerPoseEstimator(INTRINSICS, layout), GraspSegmenter(0.03, 0.05)
        )
        demo, report = extractor.extract_video(video_path)

    assert report.table_marker_fraction > 0.9
    assert np.linalg.norm(demo.t_table_cam[:3, 3] - t_table_cam[:3, 3]) < 0.01
    assert np.linalg.norm(demo.t_table_bowl[:3, 3] - t_table_bowl[:3, 3]) < 0.01
    assert np.linalg.norm(demo.t_table_plate[:3, 3] - t_table_plate[:3, 3]) < 0.01
    assert np.isfinite(demo.tcp_pos_table).all() and len(demo.timestamps_s) > 5
