"""ArUco-based object pose estimation: the table, bowl, and plate markers."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from hurobo.camera import CameraIntrinsics
from hurobo.utils.transforms import invert_transform, rvec_tvec_to_transform


@dataclass
class MarkerLayout:
    """Maps ArUco marker ids to their role and physical side length.

    Attributes:
        table_marker_id: id of the marker that defines the world (table) frame.
        table_marker_length_m: printed side length of the table marker, in metres.
        object_marker_ids: {role_name: marker_id}, e.g. {"bowl": 1, "plate": 2}.
        object_marker_length_m: printed side length of the object markers (assumed
            uniform across objects; use separate layouts if they differ).
    """

    table_marker_id: int
    table_marker_length_m: float
    object_marker_ids: dict[str, int]
    object_marker_length_m: float


class MarkerPoseEstimator:
    """Detects ArUco markers in a frame and returns poses in the table frame.

    The table marker fixes the world frame (`T_cam_table`); every other marker's
    pose is expressed relative to it, so the table frame is the one frame that
    stays meaningful across camera moves between episodes (PLAN.md §4 step 1).
    """

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        layout: MarkerLayout,
        dictionary: int = cv2.aruco.DICT_4X4_50,
    ) -> None:
        """Set up the detector for a fixed camera and marker layout.

        Args:
            intrinsics: calibrated intrinsics for the camera this estimator will
                be fed frames from. Must match the camera used at capture time.
            layout: marker id/role/size mapping, see `MarkerLayout`.
            dictionary: cv2.aruco predefined dictionary id. Must match the
                dictionary the markers were generated with.
        """
        self._intrinsics = intrinsics
        self._layout = layout
        aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
        self._detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

    def _estimate_marker_pose(self, corners: np.ndarray, marker_length_m: float) -> np.ndarray:
        """Run solvePnP for one marker's four corners and return T_cam_marker.

        Args:
            corners: shape (1, 4, 2) or (4, 2), the marker's detected image corners.
            marker_length_m: physical side length of this marker.

        Returns:
            4x4 transform T_cam_marker.
        """
        half = marker_length_m / 2.0
        object_points = np.array(
            [[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]],
            dtype=np.float64,
        )
        image_points = corners.reshape(4, 2).astype(np.float64)
        _, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            self._intrinsics.camera_matrix,
            self._intrinsics.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        return rvec_tvec_to_transform(rvec, tvec)

    def estimate(self, image_bgr: np.ndarray) -> dict[str, np.ndarray] | None:
        """Detect all markers in one frame and express their poses in the table frame.

        Args:
            image_bgr: a single BGR frame.

        Returns:
            {"table": T_cam_table, role_name: T_table_<role> for each detected
            object marker}, or None if the table marker itself wasn't found
            (without it there is no world frame to express anything else in).
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)
        if ids is None:
            return None
        ids = ids.flatten()

        table_mask = ids == self._layout.table_marker_id
        if not table_mask.any():
            return None
        table_idx = int(np.argmax(table_mask))
        t_cam_table = self._estimate_marker_pose(corners[table_idx], self._layout.table_marker_length_m)
        t_table_cam = invert_transform(t_cam_table)

        poses = {"table": t_cam_table}
        for role, marker_id in self._layout.object_marker_ids.items():
            role_mask = ids == marker_id
            if not role_mask.any():
                continue
            role_idx = int(np.argmax(role_mask))
            t_cam_object = self._estimate_marker_pose(corners[role_idx], self._layout.object_marker_length_m)
            poses[role] = t_table_cam @ t_cam_object
        return poses


def render_marker_image(
    marker_id: int,
    side_px: int = 700,
    margin_px: int = 150,
    dictionary: int = cv2.aruco.DICT_4X4_50,
) -> np.ndarray:
    """A printable marker: the black square plus the white quiet zone the detector needs.

    Print it so the *black square* measures `MarkerLayout.table_marker_length_m` (10 cm for the
    table marker by default): the metric scale of every pose comes from that number.

    Args:
        marker_id: id in the dictionary.
        side_px: side of the black square in pixels.
        margin_px: white border on each side.
        dictionary: cv2.aruco predefined dictionary id.

    Returns:
        shape (side_px + 2 * margin_px,) * 2 uint8 grayscale image.
    """
    marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(dictionary), marker_id, side_px)
    return cv2.copyMakeBorder(
        marker, margin_px, margin_px, margin_px, margin_px, cv2.BORDER_CONSTANT, value=255
    )
