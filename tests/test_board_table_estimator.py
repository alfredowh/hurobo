"""BoardMarkerPoseEstimator: the table frame from a ChArUco board, with marker offsets for objects."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from hurobo.camera import CameraIntrinsics
from hurobo.markers import BoardMarkerPoseEstimator, MarkerLayout
from hurobo.real_setup import BoardSpec
from hurobo.utils.synthetic import paste_marker
from hurobo.utils.transforms import invert_transform, look_at_origin_pose, make_transform

BOARD = BoardSpec(squares_x=7, squares_y=5, square_length_m=0.025, marker_length_m=0.0186, thickness_m=0.008)
INTRINSICS = CameraIntrinsics.pinhole(1280, 720, 1000.0)
PAD_MARKER_ID = 21


def _render_board(t_cam_table: np.ndarray) -> np.ndarray:
    """A camera frame showing the board on the tabletop at a known pose (zero distortion)."""
    px_per_m = 4000
    board_image = BOARD.make_board().generateImage(
        (round(BOARD.width_m * px_per_m), round(BOARD.height_m * px_per_m)), marginSize=0
    )
    # Board-image corners in board metres (x right, y down in the image), then to the table frame:
    # p_table = (bx, -by, thickness).
    corners_board = np.array(
        [[0, 0], [BOARD.width_m, 0], [BOARD.width_m, BOARD.height_m], [0, BOARD.height_m]], dtype=float
    )
    corners_table = np.column_stack(
        [corners_board[:, 0], -corners_board[:, 1], np.full(4, BOARD.thickness_m)]
    )
    corners_cam = corners_table @ t_cam_table[:3, :3].T + t_cam_table[:3, 3]
    projected = INTRINSICS.project(corners_cam).astype(np.float32)
    size = board_image.shape[::-1]
    source = np.array([[0, 0], [size[0], 0], [size[0], size[1]], [0, size[1]]], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(source, projected)
    frame = np.full((720, 1280), 255, np.uint8)
    warped = cv2.warpPerspective(board_image, homography, (1280, 720))
    mask = cv2.warpPerspective(np.full(board_image.shape, 255, np.uint8), homography, (1280, 720)) > 0
    frame[mask] = warped[mask]
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)


def _camera() -> np.ndarray:
    """T_cam_table for a camera 0.40 m up and 0.12 m in front of the board's far-left corner."""
    t_table_cam = look_at_origin_pose(np.array([0.12, -0.30, 0.30]))
    # Aim at the middle of the board rather than its corner.
    centre = np.array([BOARD.width_m / 2, -BOARD.height_m / 2, 0.0])
    forward = centre - t_table_cam[:3, 3]
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    t_table_cam[:3, :3] = np.column_stack([right, down, forward])
    return invert_transform(t_table_cam)


def _estimator(offsets: dict | None = None) -> BoardMarkerPoseEstimator:
    layout = MarkerLayout(0, 0.0, {"bowl": PAD_MARKER_ID}, 0.06, offsets or {})
    return BoardMarkerPoseEstimator(INTRINSICS, layout, BOARD.make_board(), BOARD.thickness_m)


def test_table_frame_is_the_board_frame_with_tabletop_at_z_zero() -> None:
    t_cam_table = _camera()
    poses = _estimator().estimate(_render_board(t_cam_table))
    assert poses is not None
    estimated = poses["table"]
    assert np.linalg.norm(estimated[:3, 3] - t_cam_table[:3, 3]) < 0.003
    angle = Rotation.from_matrix(estimated[:3, :3] @ t_cam_table[:3, :3].T).magnitude()
    assert np.degrees(angle) < 1.0


def test_object_marker_offset_moves_the_object_reference_point() -> None:
    t_cam_table = _camera()
    frame = _render_board(t_cam_table)
    t_table_marker = make_transform(Rotation.from_euler("z", 30, degrees=True), [0.22, -0.13, 0.0])
    paste_marker(frame, PAD_MARKER_ID, t_cam_table @ t_table_marker, 0.06, INTRINSICS)
    offset = (0.115, 0.0, 0.0)  # the bowl sits 11.5 cm along the marker's x axis
    poses = _estimator({"bowl": offset}).estimate(frame)
    assert poses is not None and "bowl" in poses
    expected = t_table_marker[:3, :3] @ np.array(offset) + t_table_marker[:3, 3]
    assert np.linalg.norm(poses["bowl"][:3, 3] - expected) < 0.005


def test_no_board_means_no_table_frame() -> None:
    blank = np.full((720, 1280, 3), 255, np.uint8)
    estimator = _estimator()
    assert estimator.estimate(blank) is None
    assert estimator.last_detection is not None and not estimator.last_detection.table_found
