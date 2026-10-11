"""Tests for the tabletop/background split of a table-frame splat."""

from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from hurobo.scan import split_tabletop_background


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
