"""Interactive QC figures (trajectory, aperture) build from synthetic arrays with the expected traces."""

from __future__ import annotations

import numpy as np

from ego2robo.utils.viewer import aperture_figure, trajectory_figure

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
