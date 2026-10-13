"""Tests for the scan parameter sweep: SfM options, held-out split, KD-tree init, result storage."""

from __future__ import annotations

import warnings

import cv2
import numpy as np
import pytest
import torch

from hurobo.scan import (
    KeyframeSampler,
    PosedImage,
    SfmOptions,
    SfmReconstructor,
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
    combined_clip_sweep,
    default_bowl_sweep,
    default_clip_sweep,
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


def _write_video(path, amplitudes, size=(64, 48)):
    """AVI whose frame i carries a high-frequency pattern of the given amplitude and a block coding i."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, size)
    rng = np.random.default_rng(0)
    for i, amplitude in enumerate(amplitudes):
        frame = np.full((size[1], size[0], 3), 128, np.uint8)
        noise = (rng.normal(size=(size[1], size[0])) * amplitude).astype(np.int16)
        frame = np.clip(frame + noise[..., None], 0, 255).astype(np.uint8)
        frame[:8, :8] = 10 * i + 5  # which frame this is
        writer.write(frame)
    writer.release()


def test_keyframe_sampler_streams_and_keeps_the_sharpest_frame_per_bin(tmp_path):
    # 12 frames, 4 bins of 3: the sharpest (largest amplitude) in each bin is frame 1, 4, 8, 9.
    amplitudes = [5, 40, 10, 5, 30, 10, 5, 10, 40, 60, 10, 5]
    _write_video(tmp_path / "clip.avi", amplitudes)
    paths = KeyframeSampler(num_frames=4, max_side_px=64).sample(tmp_path / "clip.avi", tmp_path / "frames")
    assert [p.name for p in paths] == [f"frame_{i:04d}.png" for i in range(4)]
    chosen = [int(round((cv2.imread(str(p))[:8, :8].mean() - 5) / 10)) for p in paths]
    assert chosen == [1, 4, 8, 9]


def test_sample_many_splits_the_budget_by_clip_length_and_numbers_frames_continuously(tmp_path):
    _write_video(tmp_path / "short.avi", [10] * 6)
    _write_video(tmp_path / "long.avi", [200] * 12)
    paths = KeyframeSampler(num_frames=6, max_side_px=64).sample_many(
        [tmp_path / "short.avi", tmp_path / "long.avi"], tmp_path / "frames"
    )
    assert [p.name for p in paths] == [f"frame_{i:04d}.png" for i in range(6)]
    texture = [cv2.imread(str(p))[16:, 16:].std() for p in paths]  # the long clip is much noisier
    assert [t < 30 for t in texture] == [True, True, False, False, False, False]
    assert not (tmp_path / "frames_part").exists()


class _FakeReconstructor(SfmReconstructor):
    """SfmReconstructor whose COLMAP runs are replaced by canned registration counts."""

    def __init__(self, registered_by_matching: dict[str, int], **kwargs):
        super().__init__(**kwargs)
        self._registered = registered_by_matching
        self.calls: list[str] = []

    def _run_once(self, image_dir, work_dir, options):
        self.calls.append(options.matching)
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / "marker.txt").write_text(options.matching)
        summary = SfmSummary(120, self._registered[options.matching], 1, 1000, 1.0)
        return SfmResult([], np.zeros((0, 3)), np.zeros((0, 3))), summary


def test_reconstructor_retries_with_the_fallback_and_keeps_the_better_run(tmp_path):
    reconstructor = _FakeReconstructor(
        {"sequential": 60, "exhaustive": 120},
        options=SfmOptions(),
        fallback=SfmOptions(matching="exhaustive"),
    )
    reconstructor.run(tmp_path / "images", tmp_path / "work")
    assert reconstructor.calls == ["sequential", "exhaustive"]
    assert reconstructor.last_summary.num_registered == 120
    assert (tmp_path / "work" / "marker.txt").read_text() == "exhaustive"  # the better run is the one on disk
    assert not (tmp_path / "work_fallback").exists()


def test_reconstructor_keeps_the_first_run_when_the_fallback_is_worse(tmp_path):
    reconstructor = _FakeReconstructor(
        {"sequential": 90, "exhaustive": 50},
        options=SfmOptions(),
        fallback=SfmOptions(matching="exhaustive"),
        min_registered_fraction=0.95,
    )
    with pytest.warns(UserWarning, match="90/120"):
        reconstructor.run(tmp_path / "images", tmp_path / "work")
    assert reconstructor.last_summary.num_registered == 90
    assert (tmp_path / "work" / "marker.txt").read_text() == "sequential"


def test_reconstructor_does_not_retry_or_warn_when_enough_registered(tmp_path):
    reconstructor = _FakeReconstructor(
        {"sequential": 110}, options=SfmOptions(), fallback=SfmOptions(matching="exhaustive")
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        reconstructor.run(tmp_path / "images", tmp_path / "work")
    assert reconstructor.calls == ["sequential"]


def test_robust_presets_lower_the_threshold_and_fallback_is_exhaustive():
    assert SfmOptions.robust().peak_threshold == 0.002
    assert SfmOptions.robust().matching == "sequential"
    assert SfmOptions.robust_exhaustive().matching == "exhaustive"
    assert SfmOptions() == SfmOptions(
        None, None, "sequential", 15, None
    )  # plain SfmOptions stays COLMAP's defaults


def test_combined_clip_sweep_scales_frames_with_clip_count_and_matches_exhaustively():
    (config,) = combined_clip_sweep(3, frames_per_clip=50)
    assert config.num_frames == 150
    assert config.sfm.matching == "exhaustive"


def test_clip_sweep_names_are_unique_and_include_colmap_defaults():
    names = [c.name for c in default_clip_sweep()]
    assert len(set(names)) == len(names) and "colmap_defaults" in names
