"""What is stuck on the objects and how a recorded clip is processed, for the board-frame recordings."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from hurobo.camera import CameraIntrinsics
from hurobo.demo import GraspSegmenter, HumanDemo
from hurobo.extract import ClipReport, ExtractionConfig, ExtractionError, extract_demo
from hurobo.markers import BoardMarkerPoseEstimator, MarkerLayout
from hurobo.real_setup import RealSetup
from hurobo.utils.video import read_frames_with_timestamps

A4_LANDSCAPE_MM = (297.0, 210.0)


@dataclass(frozen=True)
class RecordingSpec:
    """The markers and thresholds of a board-frame recording (experiment E1 and the ones after it).

    Attributes:
        bowl_marker_id: ArUco id (DICT_4X4_50) on the bowl pad. Ids 0 to 16 belong to the ChArUco board.
        plate_marker_id: id on the plate.
        marker_length_m: side of the black square, measured after printing.
        pad_marker_to_bowl_m: distance from the pad marker's centre to the centre of the bowl circle
            printed next to it, along the marker's x axis (to the right on the printed sheet).
        pinch_below_rim_m: how far below the rim top the fingertips are pinched on the real bowl.
    """

    bowl_marker_id: int = 21
    plate_marker_id: int = 22
    marker_length_m: float = 0.06
    pad_marker_to_bowl_m: float = 0.115
    pinch_below_rim_m: float = 0.012

    def marker_layout(self) -> MarkerLayout:
        """Marker ids, sizes and the bowl-pad offset, in the form the estimator takes."""
        return MarkerLayout(
            table_marker_id=-1,
            table_marker_length_m=0.0,
            object_marker_ids={"bowl": self.bowl_marker_id, "plate": self.plate_marker_id},
            object_marker_length_m=self.marker_length_m,
            object_marker_offsets_m={"bowl": (self.pad_marker_to_bowl_m, 0.0, 0.0)},
        )

    def extraction_config(self, setup: RealSetup) -> ExtractionConfig:
        """QC gates and depth anchor for this setup.

        The depth is anchored on the pinch height (bowl height minus `pinch_below_rim_m`), because the pad
        marker lies on the table and says nothing about how far away a rim grasp is. A grasp is accepted
        up to 13 cm from the bowl centre (rim 6.9 cm plus the height of the pinch plus hand error).
        """
        return ExtractionConfig(
            grasp_height_m=setup.bowl.height_m - self.pinch_below_rim_m,
            max_grasp_to_bowl_m=0.13,
        )

    def estimator(self, intrinsics: CameraIntrinsics, setup: RealSetup) -> BoardMarkerPoseEstimator:
        """The marker estimator: ChArUco board for the table frame, pad and plate markers for objects."""
        return BoardMarkerPoseEstimator(
            intrinsics, self.marker_layout(), setup.board.make_board(), setup.board.thickness_m
        )


def extract_board_clip(
    video_path: Path,
    intrinsics: CameraIntrinsics,
    setup: RealSetup,
    spec: RecordingSpec | None = None,
    segmenter: GraspSegmenter | None = None,
) -> tuple[HumanDemo, ClipReport]:
    """One phone clip to a table-frame `HumanDemo`, with the board as the table frame.

    Args:
        video_path: the clip.
        intrinsics: calibration of the phone at the clip's resolution.
        setup: measured real setup (board, bowl height).
        spec: markers; defaults to `RecordingSpec()`.
        segmenter: pinch thresholds; defaults to 3 cm close, 5 cm open.

    Returns:
        (demo, report); check `report.usable` before using the demo.
    """
    spec = spec or RecordingSpec()
    return extract_demo(
        video_path,
        intrinsics,
        spec.marker_layout(),
        segmenter,
        spec.extraction_config(setup),
        marker_estimator=spec.estimator(intrinsics, setup),
    )


def render_bowl_pad(
    spec: RecordingSpec,
    bowl_diameter_m: float,
    dpi: int = 300,
    sheet_mm: tuple[float, float] = A4_LANDSCAPE_MM,
) -> np.ndarray:
    """The printable pad the bowl stands on: a marker, and a circle where the bowl goes.

    Print at 100 % ("actual size", no fit-to-page) and measure the black square: it must be
    `spec.marker_length_m`, because the pad's pose is only as metric as that number.

    Args:
        spec: marker id, size and marker-to-bowl distance.
        bowl_diameter_m: diameter of the circle to draw (the measured outer diameter of the bowl).
        dpi: print resolution.
        sheet_mm: paper size (width, height), landscape A4 by default.

    Returns:
        shape (H, W) uint8 grayscale image of the sheet.

    Raises:
        ValueError: if the marker and the circle do not fit on the sheet.
    """
    px_per_m = dpi / 0.0254
    width_px, height_px = (round(mm * 1e-3 * px_per_m) for mm in sheet_mm)
    side_px = round(spec.marker_length_m * px_per_m)
    quiet_px = round(side_px / 6)
    circle_radius_px = round(bowl_diameter_m / 2 * px_per_m)
    marker_centre = np.array([quiet_px + side_px / 2 + 0.01 * px_per_m, height_px / 2])
    circle_centre = marker_centre + np.array([spec.pad_marker_to_bowl_m * px_per_m, 0.0])
    if (
        circle_centre[0] + circle_radius_px + 0.01 * px_per_m > width_px
        or circle_radius_px * 2 + 0.02 * px_per_m > height_px
    ):
        raise ValueError("marker and bowl circle do not fit on the sheet")

    sheet = np.full((height_px, width_px), 255, np.uint8)
    marker = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), spec.bowl_marker_id, side_px
    )
    left, top = round(marker_centre[0] - side_px / 2), round(marker_centre[1] - side_px / 2)
    sheet[top : top + side_px, left : left + side_px] = marker
    centre = (round(circle_centre[0]), round(circle_centre[1]))
    cv2.circle(sheet, centre, circle_radius_px, 0, thickness=max(2, round(0.001 * px_per_m)))
    arm = round(0.012 * px_per_m)
    cv2.line(sheet, (centre[0] - arm, centre[1]), (centre[0] + arm, centre[1]), 0, 2)
    cv2.line(sheet, (centre[0], centre[1] - arm), (centre[0], centre[1] + arm), 0, 2)
    label = (
        f"bowl {bowl_diameter_m * 100:.1f} cm   marker {spec.marker_length_m * 100:.0f} cm   "
        f"id {spec.bowl_marker_id}"
    )
    cv2.putText(
        sheet,
        label,
        (round(0.01 * px_per_m), height_px - round(0.01 * px_per_m)),
        cv2.FONT_HERSHEY_SIMPLEX,
        px_per_m * 0.00012,
        0,
        2,
    )
    return sheet


def render_plate_marker(
    spec: RecordingSpec, dpi: int = 300, sheet_mm: tuple[float, float] = A4_LANDSCAPE_MM
) -> np.ndarray:
    """The plate's marker on an otherwise blank sheet: cut it out and tape it to the middle of the plate.

    Args:
        spec: marker id and size.
        dpi: print resolution.
        sheet_mm: paper size (width, height).

    Returns:
        shape (H, W) uint8 grayscale image.
    """
    px_per_m = dpi / 0.0254
    width_px, height_px = (round(mm * 1e-3 * px_per_m) for mm in sheet_mm)
    side_px = round(spec.marker_length_m * px_per_m)
    sheet = np.full((height_px, width_px), 255, np.uint8)
    marker = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), spec.plate_marker_id, side_px
    )
    left, top = (width_px - side_px) // 2, (height_px - side_px) // 2
    sheet[top : top + side_px, left : left + side_px] = marker
    return sheet


@dataclass
class CaptureCheckReport:
    """What a test clip says about the capture setup, before a whole session is recorded.

    Attributes:
        video: the clip.
        frames_checked: frames looked at for the marker statistics.
        board_fraction: share of those frames in which enough board corners were found.
        board_width_px: median width of the detected board corners' bounding box (the grid is 7 squares).
        pad_seen_early: the bowl-pad marker was found in the first second.
        plate_seen_early: the plate marker was found in the first second.
        clip: the extraction report of the whole clip, None if extraction failed outright.
        problems: what to fix, in plain words; empty when the clip would pass.
    """

    video: Path
    frames_checked: int = 0
    board_fraction: float = 0.0
    board_width_px: float = 0.0
    pad_seen_early: bool = False
    plate_seen_early: bool = False
    clip: ClipReport | None = None
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when nothing needs fixing."""
        return not self.problems


def check_capture(
    video_path: Path,
    intrinsics: CameraIntrinsics,
    setup: RealSetup,
    spec: RecordingSpec | None = None,
    min_board_width_px: float = 150.0,
) -> CaptureCheckReport:
    """Run the recording pipeline on a short test clip and say what is wrong with the setup.

    Record about 10 s: the hand comes in, pinches the bowl rim, moves it onto the plate, lets go. The
    check looks at the same things the extraction needs, so a pass here means the real clips will extract.

    Args:
        video_path: test clip.
        intrinsics: calibration of the phone at the clip's resolution.
        setup: measured real setup.
        spec: markers; defaults to `RecordingSpec()`.
        min_board_width_px: board grid width below which corner detection becomes unreliable.

    Returns:
        The report; `problems` lists what to change.
    """
    spec = spec or RecordingSpec()
    report = CaptureCheckReport(video=video_path)
    estimator = spec.estimator(intrinsics, setup)
    board = cv2.aruco.CharucoDetector(setup.board.make_board())
    seen = early_pad = early_plate = 0
    widths_px: list[float] = []
    for index, (frame, t_s) in enumerate(read_frames_with_timestamps(video_path)):
        if index % 3:
            continue
        report.frames_checked += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _, _ = board.detectBoard(gray)
        if ids is not None and len(ids) >= estimator.MIN_CORNERS:
            seen += 1
            widths_px.append(float(np.ptp(corners.reshape(-1, 2)[:, 0])))
        if t_s <= 1.0:
            poses = estimator.estimate(frame)
            if poses is not None:
                early_pad += "bowl" in poses
                early_plate += "plate" in poses
    report.board_fraction = seen / max(report.frames_checked, 1)
    report.board_width_px = float(np.median(widths_px)) if widths_px else 0.0
    report.pad_seen_early = early_pad > 0
    report.plate_seen_early = early_plate > 0

    if report.board_fraction < 0.5:
        report.problems.append(
            f"ChArUco board found in only {report.board_fraction:.0%} of frames: raise the tablet "
            "brightness, remove glare, make sure the white squares are white (night filter off), "
            "or move the phone closer"
        )
    # The grid spans 6 inner-corner intervals of the 7 squares; scale up to the whole grid width.
    grid_px = report.board_width_px * setup.board.squares_x / max(setup.board.squares_x - 1, 1)
    if report.board_fraction >= 0.5 and grid_px < min_board_width_px:
        report.problems.append(
            f"board is only {grid_px:.0f} px wide (want {min_board_width_px:.0f}+): move the phone "
            "closer or record in 4K and use intrinsics scaled to 4K"
        )
    if not report.pad_seen_early:
        report.problems.append(
            f"bowl pad marker (id {spec.bowl_marker_id}) not seen in the first second: keep the whole "
            "marker visible and your hands away from it at the start"
        )
    if not report.plate_seen_early:
        report.problems.append(
            f"plate marker (id {spec.plate_marker_id}) not seen in the first second: it must be visible "
            "and flat on the middle of the plate at the start"
        )
    try:
        _, report.clip = extract_board_clip(video_path, intrinsics, setup, spec)
        report.problems.extend(report.clip.reasons)
    except ExtractionError as error:
        report.problems.append(str(error))
    return report
