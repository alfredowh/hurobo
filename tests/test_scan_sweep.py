"""Tests for the scan parameter sweep: SfM options, held-out split, KD-tree init, result storage."""

from __future__ import annotations

import numpy as np
import torch

from hurobo.scan import (
    PosedImage,
    SfmOptions,
    SfmResult,
    SfmSummary,
    SplatTrainer,
    colmap_extraction_options,
    colmap_matching_options,
    colmap_pipeline_options,
    holdout_split,
    orbit_coverage_deg,
    preview_grid,
)
from hurobo.scan_sweep import (
    ScanConfig,
    SweepResult,
    _result_from_json,
    _result_to_json,
    default_bowl_sweep,
    sweep_table,
)


def test_holdout_split_withholds_every_nth_view():
    train, heldout = holdout_split(12, 5)
    assert heldout == [0, 5, 10]
    assert train == [1, 2, 3, 4, 6, 7, 8, 9, 11]


def test_holdout_split_without_holdout_trains_on_everything():
    for every in (None, 0):
        train, heldout = holdout_split(4, every)
        assert train == [0, 1, 2, 3] and heldout == []


def test_colmap_options_default_to_colmap_defaults():
    import pycolmap

    options = SfmOptions()
    assert colmap_extraction_options(options).sift.peak_threshold == (
        pycolmap.FeatureExtractionOptions().sift.peak_threshold
    )
    assert colmap_matching_options(options).sift.max_ratio == pycolmap.FeatureMatchingOptions().sift.max_ratio
    assert colmap_pipeline_options(options).mapper.init_min_tri_angle == (
        pycolmap.IncrementalPipelineOptions().mapper.init_min_tri_angle
    )


def test_colmap_options_apply_overrides():
    options = SfmOptions(peak_threshold=0.002, max_ratio=0.9, init_min_tri_angle_deg=4.0)
    assert colmap_extraction_options(options).sift.peak_threshold == 0.002
    assert colmap_matching_options(options).sift.max_ratio == 0.9
    assert colmap_pipeline_options(options).mapper.init_min_tri_angle == 4.0


def test_init_scales_match_brute_force_three_nearest_neighbours():
    rng = np.random.default_rng(0)
    points = rng.normal(size=(200, 3))
    trainer = SplatTrainer(device="cpu")
    params, _ = trainer._init_params(points, np.full((200, 3), 0.5, dtype=np.float32), scene_scale=1.0)

    dists = np.linalg.norm(points[:, None] - points[None], axis=2)
    expected = np.sort(dists, axis=1)[:, 1:4].mean(axis=1)
    got = torch.exp(params["scales"][:, 0]).detach().numpy()
    np.testing.assert_allclose(got, expected, rtol=1e-4)


def test_preview_grid_stacks_photo_beside_render():
    render, photo = np.zeros((4, 6, 3)), np.ones((4, 6, 3))
    grid = preview_grid([(render, photo), (render, photo)])
    assert grid.shape == (8, 12, 3) and grid.dtype == np.uint8
    assert grid[0, 0, 0] == 255 and grid[0, -1, 0] == 0  # photo on the left, render on the right


def test_default_sweep_has_unique_names_and_starts_from_colmap_defaults():
    configs = default_bowl_sweep()
    assert len({c.name for c in configs}) == len(configs)
    assert configs[0].sfm == SfmOptions()


def _result(tmp_path, name="a", heldout=30.0):
    return SweepResult(
        config=ScanConfig(name, SfmOptions(peak_threshold=0.002, matching="exhaustive"), 40, 1000),
        sfm_summary=SfmSummary(40, 38, 1, 5000, 1.1),
        orbit_coverage_deg=349.0,
        keyframe_spacing_s=0.5,
        num_gaussians=1234,
        train_psnr_db=33.0,
        heldout_psnr_db=heldout,
        sfm_seconds=10.0,
        train_seconds=20.0,
        sparse_dir=tmp_path / "_sfm/k/sparse",
        ply_path=tmp_path / name / "splat.ply",
        preview_path=tmp_path / name / "p.png",
    )


def test_result_json_round_trip(tmp_path):
    result = _result(tmp_path)
    # Stored paths are relative to the sweep root; reloading makes them absolute again.
    data = _result_to_json(result)
    for key in ("sparse_dir", "ply_path", "preview_path"):
        data[key] = str(result.__dict__[key].relative_to(tmp_path))
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "result.json").write_text(__import__("json").dumps(data))
    loaded = _result_from_json(tmp_path, "a")
    assert loaded == result


def test_sweep_table_reports_registration_and_heldout(tmp_path):
    table = sweep_table([_result(tmp_path, "a", 31.7), _result(tmp_path, "b", None)])
    assert list(table["config"]) == ["a", "b"]
    assert table.loc[0, "registered"] == "38/40" and table.loc[0, "registered %"] == 95
    assert table.loc[0, "held-out PSNR (dB)"] == 31.7
    assert table["held-out PSNR (dB)"].isna().iloc[1]


def _orbit_sfm(azimuths_deg: np.ndarray) -> SfmResult:
    """Cameras on a radius-3 circle at height 2 looking at the origin, plus points near the origin."""
    images = []
    for azimuth in np.radians(azimuths_deg):
        centre = np.array([3 * np.cos(azimuth), 3 * np.sin(azimuth), 2.0])
        z_axis = -centre / np.linalg.norm(centre)
        x_axis = np.cross([0.0, 0.0, 1.0], z_axis)
        x_axis /= np.linalg.norm(x_axis)
        t_world_cam = np.eye(4)
        t_world_cam[:3, :3] = np.stack([x_axis, np.cross(z_axis, x_axis), z_axis], axis=1)
        t_world_cam[:3, 3] = centre
        images.append(PosedImage(np.zeros((1, 1, 3), np.float32), np.linalg.inv(t_world_cam), np.eye(3)))
    rng = np.random.default_rng(0)
    return SfmResult(images, rng.normal(scale=0.2, size=(200, 3)), np.zeros((200, 3), np.float32))


def test_orbit_coverage_reads_a_full_loop_as_about_360():
    coverage = orbit_coverage_deg(_orbit_sfm(np.arange(0, 360, 5.0)))
    assert 345 <= coverage <= 360


def test_orbit_coverage_reads_a_half_arc_as_about_180():
    coverage = orbit_coverage_deg(_orbit_sfm(np.arange(0, 181, 5.0)))
    assert 170 <= coverage <= 190
