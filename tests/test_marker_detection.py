"""MarkerPoseEstimator diagnostics: what was seen, and a mirrored image is told apart from no marker."""

from __future__ import annotations

import cv2
import numpy as np

from hurobo.camera import CameraIntrinsics
from hurobo.markers import MarkerLayout, MarkerPoseEstimator, render_marker_image


def _estimator() -> MarkerPoseEstimator:
    intrinsics = CameraIntrinsics(
        camera_matrix=np.array([[900.0, 0, 320], [0, 900.0, 240], [0, 0, 1]]),
        dist_coeffs=np.zeros(5),
        image_width=640,
        image_height=480,
    )
    return MarkerPoseEstimator(intrinsics, MarkerLayout(20, 0.10, {}, 0.05))


def _frame_with_marker(marker_id: int) -> np.ndarray:
    frame = np.full((480, 640), 255, np.uint8)
    marker = cv2.resize(render_marker_image(marker_id), (260, 260))
    frame[110:370, 190:450] = marker
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)


def test_table_marker_is_reported_with_corners() -> None:
    estimator = _estimator()
    assert estimator.estimate(_frame_with_marker(20)) is not None
    detection = estimator.last_detection
    assert detection.table_found and detection.ids == [20] and detection.corners[0].shape == (4, 2)
    assert not detection.mirrored_table_found


def test_wrong_id_lists_what_was_seen() -> None:
    estimator = _estimator()
    assert estimator.estimate(_frame_with_marker(7)) is None
    detection = estimator.last_detection
    assert not detection.table_found and detection.ids == [7] and not detection.mirrored_table_found


def test_mirrored_image_is_flagged() -> None:
    estimator = _estimator()
    assert estimator.estimate(cv2.flip(_frame_with_marker(20), 1)) is None
    assert estimator.last_detection.mirrored_table_found


def test_empty_frame_sees_nothing() -> None:
    estimator = _estimator()
    assert estimator.estimate(np.full((480, 640, 3), 255, np.uint8)) is None
    detection = estimator.last_detection
    assert detection.ids == [] and not detection.table_found and not detection.mirrored_table_found
