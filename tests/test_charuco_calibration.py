"""ChArUco calibration recovers known intrinsics from synthetic board views."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hurobo.camera import ChArUcoCalibrator

IMAGE_SIZE = (1280, 720)
FX = 900.0
PIXELS_PER_SQUARE = 100
MARGIN_PX = 40


def _board_view(calibrator: ChArUcoCalibrator, rotation_xyz_deg: tuple[float, ...], t_cam_board: np.ndarray):
    """Board image warped as a camera with focal FX at the given board pose would see it."""
    board_image = calibrator.generate_board_image(PIXELS_PER_SQUARE, MARGIN_PX)
    metres_per_px = 0.035 / PIXELS_PER_SQUARE
    # image px -> board metres, with the board origin at the corner of the first square
    to_board = np.array(
        [
            [metres_per_px, 0, -MARGIN_PX * metres_per_px],
            [0, metres_per_px, -MARGIN_PX * metres_per_px],
            [0, 0, 1],
        ]
    )
    k = np.array([[FX, 0, IMAGE_SIZE[0] / 2], [0, FX, IMAGE_SIZE[1] / 2], [0, 0, 1]])
    r = Rotation.from_euler("xyz", rotation_xyz_deg, degrees=True).as_matrix()
    pose = np.column_stack([r[:, 0], r[:, 1], t_cam_board])
    return cv2.warpPerspective(board_image, k @ pose @ to_board, IMAGE_SIZE, borderValue=255)


def test_calibration_recovers_focal_length() -> None:
    calibrator = ChArUcoCalibrator()
    rng = np.random.default_rng(1)
    for _ in range(12):
        angles = tuple(rng.uniform(-25, 25, size=3))
        t = np.array([rng.uniform(-0.15, 0.0), rng.uniform(-0.1, 0.0), rng.uniform(0.5, 0.8)])
        view = _board_view(calibrator, angles, t)
        calibrator.add_frame(cv2.cvtColor(view, cv2.COLOR_GRAY2BGR))
    intrinsics = calibrator.calibrate()
    assert intrinsics.fx == pytest.approx(FX, rel=0.03)
    assert intrinsics.fy == pytest.approx(FX, rel=0.03)
    assert (intrinsics.image_width, intrinsics.image_height) == IMAGE_SIZE
