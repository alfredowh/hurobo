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

    @classmethod
    def pinhole(
        cls, image_width: int, image_height: int, fy: float, fx: float | None = None
    ) -> CameraIntrinsics:
        """Distortion-free intrinsics with the principal point at the image centre.

        Args:
            image_width: pixel width.
            image_height: pixel height.
            fy: vertical focal length in pixels.
            fx: horizontal focal length in pixels; defaults to `fy` (square pixels).
        """
        camera_matrix = np.array(
            [[fx or fy, 0.0, image_width / 2], [0.0, fy, image_height / 2], [0.0, 0.0, 1.0]]
        )
        return cls(camera_matrix, np.zeros(5), image_width, image_height)

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

    def project(self, points_cam: np.ndarray) -> np.ndarray:
        """Pixel coordinates of camera-frame points (OpenCV axes), ignoring lens distortion.

        Args:
            points_cam: shape (3,) or (N, 3), z > 0 in front of the camera.

        Returns:
            Shape (2,) or (N, 2) pixel coordinates (u, v).
        """
        homogeneous = np.atleast_2d(points_cam) @ self.camera_matrix.T
        pixels = homogeneous[:, :2] / homogeneous[:, 2:3]
        return pixels[0] if np.ndim(points_cam) == 1 else pixels

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

    @property
    def board(self) -> cv2.aruco.CharucoBoard:
        """The board definition, shared with `scan.ChArUcoAligner` so both match the printout."""
        return self._board

    def generate_board_image(self, pixels_per_square: int = 200, margin_px: int = 40) -> np.ndarray:
        """Render the board as a grayscale image, to print or show fullscreen.

        Args:
            pixels_per_square: resolution of one chessboard square, in pixels.
                Higher values help detection at distance but make for a bigger
                file; 200 is plenty for a laptop screen or a printed page.
            margin_px: white border around the board, in pixels. A margin is
                required so `detectBoard` doesn't lose corners that touch the
                image edge.

        Returns:
            shape (H, W) uint8 image, aspect-matched to squares_x/squares_y.
        """
        size = (
            self._board.getChessboardSize()[0] * pixels_per_square,
            self._board.getChessboardSize()[1] * pixels_per_square,
        )
        return self._board.generateImage(size, marginSize=margin_px)

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
        """Fit the pinhole + distortion model to every board view added so far.

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
        # OpenCV >= 4.7 dropped aruco.calibrateCameraCharuco: match corners to 3D board points
        # ourselves and use the plain calibrator.
        object_points, image_points = [], []
        for corners, ids in zip(self._all_corners, self._all_ids, strict=True):
            board_points, board_image_points = self._board.matchImagePoints(corners, ids)
            object_points.append(board_points)
            image_points.append(board_image_points)
        _, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
            object_points, image_points, self._image_size, None, None
        )
        width, height = self._image_size
        return CameraIntrinsics(
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            image_width=width,
            image_height=height,
        )
