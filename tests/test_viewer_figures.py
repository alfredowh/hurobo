"""Interactive QC figures (trajectory, aperture) build from synthetic arrays with the expected traces."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hurobo.scan import SfmOptions, SfmSummary
from hurobo.scan_sweep import ScanConfig, SweepResult
from hurobo.utils.viewer import aperture_figure, state_action_figure, sweep_figure, trajectory_figure

NUM_SAMPLES = 50


def _synthetic_arc() -> tuple[np.ndarray, np.ndarray]:
    angle = np.linspace(0.0, np.pi, NUM_SAMPLES)
    positions_table = np.stack([0.2 * np.cos(angle), 0.2 * np.sin(angle), 0.1 * angle], axis=1)
    is_grasping = np.zeros(NUM_SAMPLES, dtype=bool)
    is_grasping[20:35] = True
    return positions_table, is_grasping


def test_trajectory_figure_splits_free_and_grasping() -> None:
    positions_table, is_grasping = _synthetic_arc()
    figure = trajectory_figure(positions_table, is_grasping)
    path, free, grasping = figure.data
    assert len(path.x) == NUM_SAMPLES
    assert len(free.x) == int((~is_grasping).sum())
    assert len(grasping.x) == int(is_grasping.sum())
    assert list(grasping.customdata) == list(np.flatnonzero(is_grasping))
    assert figure.layout.scene.aspectmode == "data"


def test_trajectory_figure_without_grasp_mask_is_single_trace() -> None:
    positions_table, _ = _synthetic_arc()
    figure = trajectory_figure(positions_table)
    assert len(figure.data) == 1
    assert len(figure.data[0].x) == NUM_SAMPLES


def test_aperture_figure_has_band_trace_and_event_lines() -> None:
    timestamps_s = np.linspace(0.0, 2.0, NUM_SAMPLES)
    pinch_aperture_m = 0.04 + 0.03 * np.cos(np.linspace(0.0, 2 * np.pi, NUM_SAMPLES))
    figure = aperture_figure(
        timestamps_s, pinch_aperture_m, 0.03, 0.05, grasp_frame_idx=12, release_frame_idx=37
    )
    band, aperture, grasp, release = figure.data
    assert len(aperture.x) == len(aperture.y) == NUM_SAMPLES
    assert np.allclose(aperture.y, pinch_aperture_m * 100)
    assert set(band.y) == {3.0, 5.0}
    assert grasp.x[0] == grasp.x[1] == timestamps_s[12]
    assert release.x[0] == release.x[1] == timestamps_s[37]


def test_aperture_figure_omits_missing_events() -> None:
    timestamps_s = np.linspace(0.0, 1.0, NUM_SAMPLES)
    figure = aperture_figure(timestamps_s, np.full(NUM_SAMPLES, 0.04), 0.03, 0.05)
    assert len(figure.data) == 2


def test_state_action_figure_overlays_each_pair_and_marks_flags() -> None:
    timestamps_s = np.linspace(0.0, 1.0, NUM_SAMPLES)
    state = np.column_stack([np.linspace(0.0, 0.2, NUM_SAMPLES), np.full(NUM_SAMPLES, 0.04)])
    flag = (np.arange(NUM_SAMPLES) > 20).astype(float)
    action = np.column_stack([np.full(NUM_SAMPLES, 0.2 / (NUM_SAMPLES - 1)), flag])
    figure = state_action_figure(
        timestamps_s,
        state,
        action,
        ["tcp_x", "aperture"],
        ["d_tcp_x", "grasp"],
        [("tcp_x", "d_tcp_x"), ("aperture", "grasp")],
        replayed_state={"tcp_x": state[:, 0]},
        hysteresis_m=(0.03, 0.05),
    )
    names = [trace.name for trace in figure.data]
    assert names == [
        "observation.state",
        "state[0] + cumsum(action)",
        "action (mm / step)",
        "observation.state",
        "action (flag)",
    ]
    assert np.allclose(figure.data[0].y, state[:, 0] * 1000)
    assert np.allclose(figure.data[2].y, action[:, 0] * 1000)
    assert list(figure.data[4].y) == list(flag)
    assert len(figure.layout.shapes) == 2  # the two hysteresis edges, only on the flag row


def _sweep_result(name: str, registered: int, heldout: float | None) -> SweepResult:
    return SweepResult(
        config=ScanConfig(name, SfmOptions(), 120, 7000),
        sfm_summary=SfmSummary(120, registered, 1, 1000, 1.0),
        orbit_coverage_deg=180.0,
        keyframe_spacing_s=0.2,
        num_gaussians=1,
        train_psnr_db=30.0,
        heldout_psnr_db=heldout,
        sfm_seconds=1.0,
        train_seconds=1.0,
        sparse_dir=Path("s"),
        ply_path=Path("p"),
        preview_path=Path("v"),
    )


def test_sweep_figure_has_registration_coverage_and_psnr_bars_per_config() -> None:
    figure = sweep_figure([_sweep_result("a", 60, 18.0), _sweep_result("b", 120, None)])
    registered, coverage, train, heldout = figure.data
    assert list(registered.x) == ["a", "b"]
    assert list(registered.y) == [50.0, 100.0]
    assert list(registered.text) == ["60/120", "120/120"]
    assert list(coverage.y) == [180.0, 180.0]
    assert list(train.y) == [30.0, 30.0]
    assert heldout.y[0] == 18.0 and heldout.y[1] is None


def test_sweep_cameras_figure_subsamples_points_and_draws_cameras() -> None:
    from hurobo.scan import PosedImage, SfmResult
    from hurobo.utils.viewer import sweep_cameras_figure

    class _Result(SweepResult):
        def load_sfm(self) -> SfmResult:
            images = [PosedImage(np.zeros((1, 1, 3), np.float32), np.eye(4), np.eye(3)) for _ in range(3)]
            return SfmResult(images, np.random.default_rng(0).normal(size=(100, 3)), np.zeros((100, 3)))

    base = _sweep_result("a", 60, 18.0)
    figure = sweep_cameras_figure(_Result(**base.__dict__), every_nth_point=4)
    assert len(figure.data[0].x) == 25
    assert figure.layout.title.text == "a: 60/120 registered"


def test_crop_figure_draws_points_one_wireframe_per_cylinder_and_the_floor() -> None:
    from hurobo.scan import CropCylinder
    from hurobo.scan_crop import FloorPlane
    from hurobo.utils.viewer import crop_figure

    points = np.random.default_rng(0).uniform(-3, 3, size=(500, 3))
    cylinders = {
        "old": CropCylinder(5.0, -1.0, 3.0, np.zeros(2)),
        "new": CropCylinder(1.5, 0.0, 2.0, np.array([0.3, 0.1])),
    }
    floor = FloorPlane(np.array([0.0, 0.0, 1.0]), np.zeros(3), 0.5, 0.02)
    figure = crop_figure(points, np.full((500, 3), 0.5), cylinders, floor)
    names = [trace.name for trace in figure.data]
    assert names == ["sparse points", "old", "new", "floor (RANSAC)"]
    assert figure.layout.scene.aspectmode == "data"


def test_plot_image_grid_has_one_axis_per_cell() -> None:
    from hurobo.utils.plotting import plot_image_grid

    image = np.zeros((20, 30, 3), np.uint8)
    figure = plot_image_grid({"a": [image, image, image], "b": [image, image, image]}, ["0", "90", "180"])
    assert len(figure.axes) == 6


def test_splat_viewer_page_is_filled_in(tmp_path) -> None:
    import numpy as np
    from plyfile import PlyData, PlyElement

    from hurobo.utils.viewer import write_splat_viewer

    names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2"]
    names += [f"rot_{i}" for i in range(4)]
    vertices = np.zeros(50, dtype=[(n, "f4") for n in names])
    vertices["opacity"] = 2.0
    vertices["rot_0"] = 1.0
    vertices["x"] = np.linspace(0, 1, 50)
    ply = tmp_path / "scene.ply"
    PlyData([PlyElement.describe(vertices, "vertex")]).write(str(ply))

    page = write_splat_viewer(ply, tmp_path / "viewer", title="demo")

    html = page.read_text()
    assert "__TITLE__" not in html and "__PLY_NAME__" not in html
    assert 'PLY_URL = "scene.ply"' in html
    assert (tmp_path / "viewer" / "scene.ply").exists()
