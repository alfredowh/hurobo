"""Helpers that notebooks and scripts share: intrinsics, pose error, frame changes, notebook bootstrap."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hurobo.camera import CameraIntrinsics
from hurobo.scan import PosedImage, SfmResult, object_frame_from_orbit
from hurobo.utils.notebook import existing_paths, find_project_root
from hurobo.utils.synthetic import synthetic_aperture_trace
from hurobo.utils.transforms import look_at_origin_pose, make_transform, pose_error


def test_pinhole_intrinsics_fovy_and_projection() -> None:
    intrinsics = CameraIntrinsics.pinhole(800, 450, fy=800.0)
    assert intrinsics.fx == intrinsics.fy == 800.0
    assert intrinsics.fovy_degrees == pytest.approx(2 * np.degrees(np.arctan(225 / 800)))
    np.testing.assert_allclose(intrinsics.project(np.array([0.0, 0.0, 2.0])), [400.0, 225.0])
    np.testing.assert_allclose(
        intrinsics.project(np.array([[0.1, 0.0, 1.0], [0.0, -0.1, 1.0]])), [[480.0, 225.0], [400.0, 145.0]]
    )


def test_pose_error_translation_and_rotation() -> None:
    truth = np.eye(4)
    estimated = make_transform(Rotation.from_euler("z", 3, degrees=True), [0.002, 0.0, 0.0])
    mm, deg = pose_error(estimated, truth)
    assert mm == pytest.approx(2.0)
    assert deg == pytest.approx(3.0)


def test_look_at_origin_pose_points_camera_z_at_origin() -> None:
    position = np.array([0.4, -0.3, 0.5])
    t_table_cam = look_at_origin_pose(position)
    np.testing.assert_allclose(t_table_cam[:3, 2], -position / np.linalg.norm(position))
    np.testing.assert_allclose(t_table_cam[:3, :3] @ t_table_cam[:3, :3].T, np.eye(3), atol=1e-12)


def test_sfm_in_frame_round_trips_points_and_cameras() -> None:
    rng = np.random.default_rng(0)
    t_world_cams = [
        make_transform(Rotation.from_rotvec(rng.normal(size=3) * 0.3), rng.normal(size=3)) for _ in range(4)
    ]
    sfm = SfmResult(
        images=[
            PosedImage(np.zeros((1, 1, 3), np.float32), np.linalg.inv(t), np.eye(3)) for t in t_world_cams
        ],
        points_world=rng.normal(size=(20, 3)),
        colors=rng.uniform(size=(20, 3)).astype(np.float32),
    )
    t_new_world = make_transform(Rotation.from_rotvec([0.2, -0.4, 0.1]), [1.0, 2.0, 3.0])
    moved = sfm.in_frame(t_new_world)
    for before, after in zip(sfm.t_world_cam_list, moved.t_world_cam_list, strict=True):
        np.testing.assert_allclose(after, t_new_world @ before, atol=1e-9)
    back = moved.in_frame(np.linalg.inv(t_new_world))
    np.testing.assert_allclose(back.points_world, sfm.points_world, atol=1e-9)
    # object_frame_from_orbit only needs poses and points, so it must run on a moved result too.
    assert object_frame_from_orbit(moved).shape == (4, 4)


def test_synthetic_aperture_trace_has_one_grasp() -> None:
    t, aperture = synthetic_aperture_trace()
    assert t.shape == aperture.shape
    assert aperture[(t > 2.0) & (t < 4.0)].max() < 0.03
    assert aperture[t < 1.0].min() > 0.05


def test_project_root_and_existing_paths(tmp_path: Path) -> None:
    root = find_project_root(Path(__file__).parent)
    assert (root / "src/hurobo").is_dir()
    present = tmp_path / "a.ply"
    present.touch()
    assert existing_paths({"a": present, "b": tmp_path / "missing.ply"}, "hint") == {"a": present}
    with pytest.raises(FileNotFoundError, match="hint"):
        existing_paths({"b": tmp_path / "missing.ply"}, "hint")
