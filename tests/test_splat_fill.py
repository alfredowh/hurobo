"""Tests for the dish shape-prior fill: scale fit, haze removal, gap filling, coverage metric."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from hurobo.splat_fill import DishFiller, DishPrior, solid_coverage

FIELDS = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2"] + [
    f"rot_{i}" for i in range(4)
]
PRIOR = DishPrior(diameter_m=0.255, height_m=0.028)


def _write_dish(path: Path, rng: np.random.Generator, units_per_metre: float = 10.0) -> None:
    """A dish scan with a hole in one quadrant, white colour, plus brown haze above the rim."""
    radius = 0.1275 * units_per_metre
    r = radius * np.sqrt(rng.uniform(0, 1, 6000))
    theta = rng.uniform(0, 2 * np.pi, 6000)
    z = PRIOR.top_height(r / radius) * units_per_metre
    keep = ~((np.cos(theta) > 0.0) & (np.sin(theta) > 0.0) & (r < 0.8 * radius))
    body = np.c_[r * np.cos(theta), r * np.sin(theta), z][keep]
    haze = np.c_[rng.normal(0, 0.5 * radius, (300, 2)), rng.uniform(0.1, 0.4, 300) * units_per_metre]
    xyz = np.r_[body, haze]
    dtype = np.dtype([(name, "f4") for name in FIELDS])
    verts = np.zeros(len(xyz), dtype=dtype)
    verts["x"], verts["y"], verts["z"] = xyz.T
    white = (0.9 - 0.5) / 0.28209479177387814
    verts["f_dc_0"] = verts["f_dc_1"] = verts["f_dc_2"] = white
    verts["f_dc_0"][len(body) :] = (0.5 - 0.5) / 0.28209479177387814
    verts["opacity"] = 3.0
    verts["scale_0"] = verts["scale_1"] = verts["scale_2"] = np.log(0.05)
    verts["rot_0"] = 1.0
    PlyData([PlyElement.describe(verts, "vertex")]).write(str(path))


def test_fill_recovers_scale_closes_the_hole_and_drops_haze(tmp_path):
    src, out = tmp_path / "in.ply", tmp_path / "out.ply"
    _write_dish(src, np.random.default_rng(0))
    report = DishFiller(PRIOR).fill(src, out)
    assert abs(report.units_per_metre - 10.0) < 1.0
    assert report.added > 0 and report.dropped >= 250
    assert solid_coverage(out, report.radius_units) > solid_coverage(src, report.radius_units) + 0.03
    filled = PlyData.read(str(out))["vertex"].data
    assert (filled["z"] < 0.2 * 10.0).all()


def test_synthetic_gaussians_are_white_and_unit_quaternions(tmp_path):
    src, out = tmp_path / "in.ply", tmp_path / "out.ply"
    _write_dish(src, np.random.default_rng(1))
    DishFiller(PRIOR).fill(src, out)
    verts = PlyData.read(str(out))["vertex"].data
    quats = np.stack([verts[f"rot_{i}"] for i in range(4)], axis=1)
    np.testing.assert_allclose(np.linalg.norm(quats, axis=1), 1.0, atol=1e-4)
    rgb = verts["f_dc_1"] * 0.28209479177387814 + 0.5
    assert rgb.mean() > 0.8  # the haze colour was not spread into the fill


def test_dish_prior_profile_is_flat_then_rises():
    ratios = np.array([0.0, 0.5, 1.0])
    heights = PRIOR.top_height(ratios)
    assert heights[0] == heights[1] == PRIOR.thickness_m
    assert abs(heights[2] - PRIOR.height_m) < 1e-12
