"""Tests for finding the plate and the bowl as blobs above a metric tabletop."""

from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from hurobo.scan_crop import ExpectedObject, locate_objects_on_table


def _disc(
    centre: tuple[float, float], radius: float, z: float, n: int, rng: np.random.Generator
) -> np.ndarray:
    angle = rng.uniform(0, 2 * np.pi, n)
    r = radius * np.sqrt(rng.uniform(0, 1, n))
    return np.stack([centre[0] + r * np.cos(angle), centre[1] + r * np.sin(angle), np.full(n, z)], axis=1)


def test_objects_are_matched_by_size(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    xyz = np.concatenate(
        [
            _disc((0.5, -0.3), 0.5, 0.0, 4000, rng),  # tabletop
            _disc((0.2, -0.3), 0.125, 0.03, 6000, rng),  # plate
            _disc((0.5, -0.3), 0.07, 0.06, 6000, rng),  # bowl
        ]
    )
    vertices = np.zeros(len(xyz), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    vertices["x"], vertices["y"], vertices["z"] = xyz.T
    ply = tmp_path / "table.ply"
    PlyData([PlyElement.describe(vertices, "vertex")]).write(str(ply))

    found = locate_objects_on_table(
        ply, [ExpectedObject("bowl", 0.14, 0.067), ExpectedObject("plate", 0.255, 0.028)]
    )

    assert np.allclose(found["plate"].centre_xy, [0.2, -0.3], atol=0.03)
    assert np.allclose(found["bowl"].centre_xy, [0.5, -0.3], atol=0.03)
    assert found["plate"].radius > found["bowl"].radius
