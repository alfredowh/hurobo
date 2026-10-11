"""The on-screen marker sheet: right ids, true physical size, and detectable by a camera."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from hurobo.markers import MarkerLayout, render_screen_marker_sheet

LAYOUT = MarkerLayout(20, 0.10, {"bowl": 21, "plate": 22}, 0.05)
PPI = 217.0
CAMERA_SCALE = 0.3  # sheet pixels to camera pixels: a webcam a bit under half a metre away


def _camera_view(sheet: np.ndarray) -> np.ndarray:
    """The sheet seen by a blurry, slightly noisy camera at a realistic distance (grayscale)."""
    height, width = sheet.shape[:2]
    view = cv2.resize(sheet, None, fx=CAMERA_SCALE, fy=CAMERA_SCALE, interpolation=cv2.INTER_AREA)
    canvas = np.full((720, 1280, 3), 100, dtype=np.uint8)
    canvas[50 : 50 + view.shape[0], 50 : 50 + view.shape[1]] = view
    noisy = cv2.GaussianBlur(canvas, (0, 0), 1.0) + np.random.default_rng(0).normal(0, 3, canvas.shape)
    return cv2.cvtColor(np.clip(noisy, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)


def test_sheet_markers_are_detected_at_true_size() -> None:
    sheet = render_screen_marker_sheet(LAYOUT, PPI, (1800, 1200))
    assert sheet.shape == (1200, 1800, 3)
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
    corners, ids, _ = detector.detectMarkers(_camera_view(sheet))
    assert sorted(ids.ravel()) == [20, 21, 22]
    px_per_m = PPI / 0.0254 * CAMERA_SCALE
    for marker_corners, marker_id in zip(corners, ids.ravel(), strict=True):
        side_m = np.linalg.norm(marker_corners[0, 0] - marker_corners[0, 1]) / px_per_m
        expected_m = LAYOUT.table_marker_length_m if marker_id == 20 else LAYOUT.object_marker_length_m
        assert side_m == pytest.approx(expected_m, abs=0.002)


def test_sheet_rejects_a_screen_that_is_too_small() -> None:
    with pytest.raises(ValueError):
        render_screen_marker_sheet(LAYOUT, PPI, (600, 400))


def test_transparent_sheet_is_opaque_only_on_the_markers() -> None:
    sheet = render_screen_marker_sheet(LAYOUT, PPI, (1800, 1200), transparent=True)
    assert sheet.shape == (1200, 1800, 4)
    alpha = sheet[..., 3]
    assert set(np.unique(alpha)) == {0, 255}
    assert alpha[0, 0] == 0 and 0 < (alpha == 255).mean() < 1
