"""Tests for the tabletop/background split of a table-frame splat."""

from pathlib import Path

import numpy as np
import pytest
from plyfile import PlyData, PlyElement

from hurobo.scan import Similarity, merge_plys, split_tabletop_background


def _write_ply(path: Path, xyz: np.ndarray) -> None:
    vertices = np.zeros(len(xyz), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    vertices["x"], vertices["y"], vertices["z"] = xyz.T
    PlyData([PlyElement.describe(vertices, "vertex")]).write(str(path))


def test_split_separates_tabletop_from_background(tmp_path: Path) -> None:
    xyz = np.array(
        [
            [0.0, 0.0, 0.0],  # table surface
            [0.1, 0.1, 0.05],  # object on the table
            [2.0, 0.0, 0.5],  # wall, far outside the radius
            [0.0, 0.0, -0.5],  # floor under the table
            [0.0, 0.0, 1.5],  # ceiling lamp above the table
        ]
    )
    table_ply = tmp_path / "table.ply"
    background_ply = tmp_path / "background.ply"
    _write_ply(table_ply, xyz)

    tabletop, background = split_tabletop_background(table_ply, background_ply, radius_m=0.5)

    assert (tabletop, background) == (2, 3)
    assert len(PlyData.read(str(table_ply))["vertex"].data) == 2
    assert len(PlyData.read(str(background_ply))["vertex"].data) == 3


def test_box_split_keeps_a_long_tabletop(tmp_path: Path) -> None:
    xyz = np.array(
        [
            [1.2, -0.3, 0.0],  # far end of a long table: outside any 0.3 m cylinder, inside the box
            [0.7, -0.3, 0.05],  # object on the table
            [1.2, -0.9, 0.0],  # floor beyond the near edge
            [1.2, -0.3, 1.0],  # above the work volume
        ]
    )
    table_ply = tmp_path / "table.ply"
    _write_ply(table_ply, xyz)

    tabletop, background = split_tabletop_background(
        table_ply, tmp_path / "background.ply", box_xy=(-0.03, 1.415, -0.58, 0.02)
    )

    assert (tabletop, background) == (2, 2)


def test_split_requires_a_region(tmp_path: Path) -> None:
    table_ply = tmp_path / "table.ply"
    _write_ply(table_ply, np.zeros((1, 3)))
    with pytest.raises(ValueError):
        split_tabletop_background(table_ply, tmp_path / "background.ply")


def test_similarity_shifted_lifts_destination_z() -> None:
    similarity = Similarity(2.0, np.eye(3), np.array([1.0, 0.0, 0.0]))
    lifted = similarity.shifted(np.array([0.0, 0.0, 0.008]))
    point = np.array([[1.0, 1.0, 1.0]])
    assert np.allclose(lifted.apply(point) - similarity.apply(point), [[0.0, 0.0, 0.008]])


def test_merge_is_the_inverse_of_a_split(tmp_path: Path) -> None:
    xyz = np.array([[0.0, 0.0, 0.0], [0.1, 0.1, 0.05], [2.0, 0.0, 0.5]])
    table_ply, background_ply, scene_ply = (tmp_path / f"{n}.ply" for n in ("table", "bg", "scene"))
    _write_ply(table_ply, xyz)
    split_tabletop_background(table_ply, background_ply, radius_m=0.5)

    count = merge_plys([table_ply, background_ply], scene_ply)

    merged = PlyData.read(str(scene_ply))["vertex"].data
    assert count == len(merged) == 3
    assert sorted(merged["x"]) == pytest.approx(sorted(xyz[:, 0]))


def test_merge_rejects_mismatched_fields(tmp_path: Path) -> None:
    a, b = tmp_path / "a.ply", tmp_path / "b.ply"
    _write_ply(a, np.zeros((1, 3)))
    vertices = np.zeros(1, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("opacity", "f4")])
    PlyData([PlyElement.describe(vertices, "vertex")]).write(str(b))

    with pytest.raises(ValueError):
        merge_plys([a, b], tmp_path / "out.ply")


def test_cameras_follow_the_similarity() -> None:
    from scipy.spatial.transform import Rotation

    from hurobo.scan import PosedImage, SfmResult

    t_cam_world = np.eye(4)
    t_cam_world[:3, :3] = Rotation.from_euler("xyz", [10, 20, 30], degrees=True).as_matrix()
    t_cam_world[:3, 3] = [0.1, -0.2, 2.0]
    k = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
    sfm = SfmResult(
        [PosedImage(np.zeros((480, 640, 3), np.float32), t_cam_world, k)], np.zeros((0, 3)), np.zeros((0, 3))
    )
    similarity = Similarity(
        0.14, Rotation.from_euler("z", 70, degrees=True).as_matrix(), np.array([0.3, 0.4, 0.008])
    )
    point_sfm = np.array([[0.2, -0.1, 0.5]])

    pixel_before = k @ (t_cam_world[:3, :3] @ point_sfm[0] + t_cam_world[:3, 3])
    t_cam_table = sfm.t_cam_world_in(similarity)[0]
    point_table = similarity.apply(point_sfm)[0]
    pixel_after = k @ (t_cam_table[:3, :3] @ point_table + t_cam_table[:3, 3])

    assert pixel_after[:2] / pixel_after[2] == pytest.approx(pixel_before[:2] / pixel_before[2])
