"""ArUco-based object pose estimation: the table, bowl, and plate markers."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from hurobo.camera import CameraIntrinsics
from hurobo.utils.transforms import invert_transform, make_transform, rvec_tvec_to_transform


@dataclass
class MarkerLayout:
    """Maps ArUco marker ids to their role and physical side length.

    Attributes:
        table_marker_id: id of the marker that defines the world (table) frame.
        table_marker_length_m: printed side length of the table marker, in metres.
        object_marker_ids: {role_name: marker_id}, e.g. {"bowl": 1, "plate": 2}.
        object_marker_length_m: printed side length of the object markers (assumed
            uniform across objects; use separate layouts if they differ).
        object_marker_offsets_m: {role_name: (x, y, z)} position of the object's reference point
            (its centre) in the marker's own frame, for a marker that sits beside the object on a pad
            instead of on it. Roles left out use the marker origin.
    """

    table_marker_id: int
    table_marker_length_m: float
    object_marker_ids: dict[str, int]
    object_marker_length_m: float
    object_marker_offsets_m: dict[str, tuple[float, float, float]] = field(default_factory=dict)


@dataclass
class MarkerDetection:
    """What the ArUco detector saw in one frame, for on-screen and terminal diagnostics.

    Attributes:
        ids: ids of all detected markers.
        corners: one (4, 2) array of image corners (pixels) per detected marker, same order.
        table_id: the id that was being looked for.
        table_found: True if `table_id` is among `ids`.
        mirrored_table_found: True if the table marker is *not* found but is found in the
            left-right flipped image, i.e. the camera image is mirrored (ArUco cannot read mirrored
            markers).
    """

    ids: list[int]
    corners: list[np.ndarray]
    table_id: int
    table_found: bool
    mirrored_table_found: bool = False


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
        self._last_detection: MarkerDetection | None = None

    @property
    def last_detection(self) -> MarkerDetection | None:
        """What the most recent `estimate` call saw, or None before the first call."""
        return self._last_detection

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
            object marker}, or None if the table frame itself wasn't found
            (without it there is no world frame to express anything else in).
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)
        ids = np.empty(0, dtype=int) if ids is None else ids.flatten()
        t_cam_table = self._table_pose(gray, corners, ids)
        if t_cam_table is None:
            return None
        t_table_cam = invert_transform(t_cam_table)

        poses = {"table": t_cam_table}
        for role, marker_id in self._layout.object_marker_ids.items():
            role_mask = ids == marker_id
            if not role_mask.any():
                continue
            role_idx = int(np.argmax(role_mask))
            t_cam_marker = self._estimate_marker_pose(corners[role_idx], self._layout.object_marker_length_m)
            offset = self._layout.object_marker_offsets_m.get(role)
            if offset is not None:
                t_cam_marker = t_cam_marker @ make_transform(Rotation.identity(), np.asarray(offset))
            poses[role] = t_table_cam @ t_cam_marker
        return poses

    def _table_pose(self, gray: np.ndarray, corners: list, ids: np.ndarray) -> np.ndarray | None:
        """T_cam_table from the single table marker; records what was seen in `last_detection`."""
        table_mask = ids == self._layout.table_marker_id
        mirrored = False
        if not table_mask.any():
            # Looking at the flipped image tells "not in view" from "image is mirrored".
            _, flipped_ids, _ = self._detector.detectMarkers(cv2.flip(gray, 1))
            mirrored = flipped_ids is not None and bool((flipped_ids == self._layout.table_marker_id).any())
        self._last_detection = MarkerDetection(
            [int(i) for i in ids],
            [c.reshape(4, 2) for c in corners],
            self._layout.table_marker_id,
            bool(table_mask.any()),
            mirrored,
        )
        if not table_mask.any():
            return None
        table_idx = int(np.argmax(table_mask))
        return self._estimate_marker_pose(corners[table_idx], self._layout.table_marker_length_m)


class BoardMarkerPoseEstimator(MarkerPoseEstimator):
    """Like `MarkerPoseEstimator`, but the table frame is a ChArUco board (e.g. shown on a tablet).

    The frame is the same one `scan.align_charuco_table` gives the table scans: origin at the board's
    outer top-left corner, x along the long side, y toward the top of the board image, z up, and z = 0
    on the tabletop, i.e. `board_thickness_m` below the board surface. Clips filmed in this frame and
    the scan splats therefore share coordinates without an alignment step.

    `layout.table_marker_*` is ignored; `layout.object_marker_*` works as in the base class.
    """

    MIN_CORNERS = 6

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        layout: MarkerLayout,
        board: cv2.aruco.CharucoBoard,
        board_thickness_m: float,
        dictionary: int = cv2.aruco.DICT_4X4_50,
    ) -> None:
        """Set up the detector.

        Args:
            intrinsics: calibrated intrinsics of the camera that filmed the frames.
            layout: object marker ids, size and offsets.
            board: the board as displayed (squares must be measured on the screen).
            board_thickness_m: height of the board surface above the tabletop.
            dictionary: ArUco dictionary of the board and object markers.
        """
        super().__init__(intrinsics, layout, dictionary)
        self._board = board
        self._charuco = cv2.aruco.CharucoDetector(board)
        # p_board = F p_table + (0, 0, t): the board's own z points into the table, so y and z flip.
        flip = np.diag([1.0, -1.0, -1.0])
        self._t_board_table = np.eye(4)
        self._t_board_table[:3, :3] = flip
        self._t_board_table[:3, 3] = [0.0, 0.0, board_thickness_m]

    def _table_pose(self, gray: np.ndarray, corners: list, ids: np.ndarray) -> np.ndarray | None:
        """T_cam_table from the ChArUco corners (planar PnP, refined)."""
        charuco_corners, charuco_ids, _, _ = self._charuco.detectBoard(gray)
        found = charuco_ids is not None and len(charuco_ids) >= self.MIN_CORNERS
        self._last_detection = MarkerDetection(
            [int(i) for i in ids],
            [c.reshape(4, 2) for c in corners],
            self._layout.table_marker_id,
            bool(found),
        )
        if not found:
            return None
        object_points, image_points = self._board.matchImagePoints(charuco_corners, charuco_ids)
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            self._intrinsics.camera_matrix,
            self._intrinsics.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE,
        )
        if not ok:
            return None
        rvec, tvec = cv2.solvePnPRefineLM(
            object_points,
            image_points,
            self._intrinsics.camera_matrix,
            self._intrinsics.dist_coeffs,
            rvec,
            tvec,
        )
        return rvec_tvec_to_transform(rvec, tvec) @ self._t_board_table


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


def render_screen_marker_sheet(
    layout: MarkerLayout,
    ppi: float,
    screen_px: tuple[int, int],
    background_bgr: np.ndarray | None = None,
    transparent: bool = False,
    quiet_zone_modules: float = 1.0,
    gap_mm: float = 6.0,
    dictionary: int = cv2.aruco.DICT_4X4_50,
) -> np.ndarray:
    """One full-screen image with the table, bowl and plate markers at true physical size.

    Show it 1:1 (one image pixel per screen pixel, no OS scaling or zoom) and the black squares
    measure `layout.*_length_m` on the glass. The table marker sits on the left, the object
    markers stacked on the right, each inside a white quiet zone; everything else is
    `background_bgr` (e.g. a photo of the tabletop, so the tablet blends in) or light grey.

    Args:
        layout: marker ids and side lengths.
        ppi: pixels per inch of the display (Surface Go 1: 217).
        screen_px: (width, height) of the display in pixels.
        background_bgr: optional photo, cover-resized to the screen.
        transparent: return BGRA with alpha 0 everywhere except the markers and their quiet
            zones, to composite over a photo elsewhere. Ignores `background_bgr`.
        quiet_zone_modules: white border around each black square, in marker modules (side / 6).
            Keep it at 1: a narrower zone lets the detector decode the white rectangle around the
            marker as if it were the marker, which inflates its size.
        gap_mm: space between the table marker and the object column, and between the objects.
        dictionary: cv2.aruco predefined dictionary id.

    Returns:
        shape (height, width, 3) uint8 BGR image, or (height, width, 4) BGRA if `transparent`.

    Raises:
        ValueError: if the markers do not fit on the screen at this ppi.
    """
    width_px, height_px = screen_px
    px_per_m = ppi / 0.0254
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
    gap_px = round(gap_mm * 1e-3 * px_per_m)

    if background_bgr is None or transparent:
        sheet = np.full((height_px, width_px, 3), 200, dtype=np.uint8)
    else:
        scale = max(width_px / background_bgr.shape[1], height_px / background_bgr.shape[0])
        resized = cv2.resize(background_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        top, left = (resized.shape[0] - height_px) // 2, (resized.shape[1] - width_px) // 2
        sheet = resized[top : top + height_px, left : left + width_px].copy()

    table_side = round(layout.table_marker_length_m * px_per_m)
    object_side = round(layout.object_marker_length_m * px_per_m)
    table_quiet = round(quiet_zone_modules * table_side / 6)
    object_quiet = round(quiet_zone_modules * object_side / 6)
    object_ids = list(layout.object_marker_ids.values())
    table_box, object_box = table_side + 2 * table_quiet, object_side + 2 * object_quiet
    column_height = len(object_ids) * object_box + (len(object_ids) - 1) * gap_px
    block_width = table_box + gap_px + object_box
    block_height = max(table_box, column_height)
    if block_width > width_px or block_height > height_px:
        raise ValueError(
            f"markers need {block_width}x{block_height} px but the screen is {width_px}x{height_px}"
        )
    left0, top0 = (width_px - block_width) // 2, (height_px - block_height) // 2

    opaque = np.zeros((height_px, width_px), dtype=np.uint8)

    def paste(marker_id: int, side: int, quiet: int, x: int, y: int) -> None:
        patch = cv2.aruco.generateImageMarker(aruco_dict, marker_id, side)
        patch = cv2.copyMakeBorder(patch, quiet, quiet, quiet, quiet, cv2.BORDER_CONSTANT, value=255)
        sheet[y : y + patch.shape[0], x : x + patch.shape[1]] = patch[..., None]
        opaque[y : y + patch.shape[0], x : x + patch.shape[1]] = 255

    paste(layout.table_marker_id, table_side, table_quiet, left0, top0 + (block_height - table_box) // 2)
    column_x = left0 + table_box + gap_px
    column_y = top0 + (block_height - column_height) // 2
    for index, marker_id in enumerate(object_ids):
        paste(marker_id, object_side, object_quiet, column_x, column_y + index * (object_box + gap_px))
    return np.dstack([sheet, opaque]) if transparent else sheet
