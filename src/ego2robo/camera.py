"""Camera intrinsics container and ChArUco-based calibration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsics plus distortion, as used by cv2.solvePnP and friends.

    Attributes:
        camera_matrix: shape (3, 3), the standard OpenCV K matrix.
        dist_coeffs: shape (N,), OpenCV distortion coefficients (k1, k2, p1, p2, k3, ...).
        image_width: pixel width of the calibration images (and of any image this
            intrinsics object is used with — re-calibrate if resolution changes).
        image_height: pixel height of the calibration images.
    """

    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_width: int
    image_height: int

    @property
    def fx(self) -> float:
        """Focal length in pixels, x axis."""
        return float(self.camera_matrix[0, 0])

    @property
    def fy(self) -> float:
        """Focal length in pixels, y axis. Used to set a MuJoCo camera's `fovy`."""
        return float(self.camera_matrix[1, 1])

    @property
    def fovy_degrees(self) -> float:
        """Vertical field of view in degrees, derived from fy and image height.

        MuJoCo cameras are specified by a single vertical FOV (`cam_fovy`), not by
        a full K matrix, so this is the number `SceneTwin` (`twin.py`) needs when
        placing the sim's twin camera at the real camera's pose.
        """
        return float(2.0 * np.degrees(np.arctan2(self.image_height / 2.0, self.fy)))

    def save(self, path: Path) -> None:
        """Save intrinsics to an .npz file."""
        np.savez(
            path,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            image_width=self.image_width,
            image_height=self.image_height,
        )

    @classmethod
    def load(cls, path: Path) -> CameraIntrinsics:
        """Load intrinsics previously saved with `save`."""
        data = np.load(path)
        return cls(
            camera_matrix=data["camera_matrix"],
            dist_coeffs=data["dist_coeffs"],
            image_width=int(data["image_width"]),
            image_height=int(data["image_height"]),
        )


class ChArUcoCalibrator:
    """Calibrates a single camera from a ChArUco board shown across several frames.

    Usage: feed it frames with `add_frame`, then call `calibrate`. The board
    dimensions must match whatever was printed/displayed, or the metric scale
    of every downstream pose (hand, markers) will be wrong.
    """

    def __init__(
        self,
        squares_x: int = 7,
        squares_y: int = 5,
        square_length_m: float = 0.035,
        marker_length_m: float = 0.026,
        dictionary: int = cv2.aruco.DICT_4X4_50,
    ) -> None:
        """Set up the ChArUco board definition.

        Args:
            squares_x: number of chessboard squares along the board's x axis.
            squares_y: number of chessboard squares along the board's y axis.
            square_length_m: physical side length of one chessboard square, in
                metres. Must match the printed/displayed board exactly.
            marker_length_m: physical side length of the ArUco markers embedded
                in the board, in metres.
            dictionary: cv2.aruco predefined dictionary id. Must match the
                dictionary used for the table/object markers (`markers.py`) so
                a single `cv2.aruco.ArucoDetector` can be reused.
        """
        self._dictionary = cv2.aruco.getPredefinedDictionary(dictionary)
        self._board = cv2.aruco.CharucoBoard(
            (squares_x, squares_y), square_length_m, marker_length_m, self._dictionary
        )
        self._detector = cv2.aruco.CharucoDetector(self._board)
        self._all_corners: list[np.ndarray] = []
        self._all_ids: list[np.ndarray] = []
        self._image_size: tuple[int, int] | None = None

    def add_frame(self, image_bgr: np.ndarray) -> bool:
        """Detect the board in one frame and, if found, keep it for calibration.

        Args:
            image_bgr: a single BGR frame containing the ChArUco board.

        Returns:
            True if enough board corners were detected in this frame to use it.
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        self._image_size = gray.shape[::-1]
        charuco_corners, charuco_ids, _, _ = self._detector.detectBoard(gray)
        if charuco_corners is None or len(charuco_corners) < 6:
            return False
        self._all_corners.append(charuco_corners)
        self._all_ids.append(charuco_ids)
        return True

    def calibrate(self) -> CameraIntrinsics:
        """Run `cv2.aruco.calibrateCameraCharuco` over all frames added so far.

        Returns:
            The fitted CameraIntrinsics.

        Raises:
            RuntimeError: if fewer than 5 usable frames were collected.
        """
        if len(self._all_corners) < 5 or self._image_size is None:
            raise RuntimeError(
                f"Only {len(self._all_corners)} usable calibration frames; need >= 5. "
                "Show the board at more angles/distances and keep it fully in frame."
            )
        _, camera_matrix, dist_coeffs, _, _ = cv2.aruco.calibrateCameraCharuco(
            self._all_corners, self._all_ids, self._board, self._image_size, None, None
        )
        width, height = self._image_size
        return CameraIntrinsics(
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            image_width=width,
            image_height=height,
        )
