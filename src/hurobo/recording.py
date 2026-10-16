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
from hurobo.utils.stats import longest_true_run
from hurobo.utils.video import (
    VideoWriter,
    caption_frame,
    draw_hand_skeleton,
    draw_outlined_text,
    get_video_fps,
    read_frames_with_timestamps,
)

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
    marker_length_m: float = 0.0575
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
    window_s: tuple[float, float] | None = None,
) -> tuple[HumanDemo, ClipReport]:
    """One phone clip to a table-frame `HumanDemo`, with the board as the table frame.

    Args:
        video_path: the clip.
        intrinsics: calibration of the phone at the clip's resolution.
        setup: measured real setup (board, bowl height).
        spec: markers; defaults to `RecordingSpec()`.
        segmenter: pinch thresholds; defaults to 3 cm close, 5 cm open.
        window_s: (start, end) clip times to keep (the cleaning step's cut); the whole clip if None.

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
        window_s=window_s,
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


@dataclass
class CaptureDiagnosis:
    """Everything the capture check looks at, per frame, to see where a test clip goes wrong.

    Attributes:
        video: the clip.
        report: the pass/fail report of `check_capture`.
        times_s: shape (N,), frame times.
        board_corners: shape (N,), ChArUco corners found (the pose needs `MIN_CORNERS`).
        marker_ids: ArUco ids detected in each frame (the board's own markers included).
        marker_corners: per frame, one (4, 2) pixel array per detected marker.
        hand_landmarks_px: per frame, shape (21, 2), or None where no hand was detected.
        aperture_m: shape (N,), thumb-index distance, NaN where no hand was detected.
        reprojection_px: shape (N,), PnP reprojection error of the hand fit, NaN where no hand.
        tcp_depth_m: shape (N,), distance of the pinch point along the camera axis, NaN where no hand.
        spec: markers the clip was checked against.
        close_threshold_m: aperture below this counts as a grasp.
        open_threshold_m: aperture above this counts as a release.
        min_board_corners: corners needed for a board pose.
    """

    video: Path
    report: CaptureCheckReport
    times_s: np.ndarray
    board_corners: np.ndarray
    marker_ids: list[list[int]]
    marker_corners: list[list[np.ndarray]]
    hand_landmarks_px: list[np.ndarray | None]
    aperture_m: np.ndarray
    reprojection_px: np.ndarray
    tcp_depth_m: np.ndarray
    spec: RecordingSpec
    close_threshold_m: float
    open_threshold_m: float
    min_board_corners: int

    @property
    def board_found(self) -> np.ndarray:
        """shape (N,) bool, enough board corners for a pose."""
        return self.board_corners >= self.min_board_corners

    @property
    def hand_found(self) -> np.ndarray:
        """shape (N,) bool, a hand was detected."""
        return ~np.isnan(self.aperture_m)

    def marker_found(self, marker_id: int) -> np.ndarray:
        """shape (N,) bool, the marker with this id was detected."""
        return np.array([marker_id in ids for ids in self.marker_ids])

    def hand_gap(self) -> tuple[int, int]:
        """Frame range (start, stop) of the longest hand dropout between the first and last detection."""
        found = np.flatnonzero(self.hand_found)
        if len(found) < 2:
            return 0, 0
        first, last = found[0], found[-1]
        start, stop = longest_true_run(~self.hand_found[first : last + 1])
        return (first + start, first + stop) if stop > start else (0, 0)

    def key_moments(self) -> dict[str, int]:
        """Frames worth looking at, label -> frame index (later labels never repeat an earlier frame)."""
        moments = {"first frame": 0, "at 1 s": int(np.searchsorted(self.times_s, self.times_s[0] + 1.0))}
        start, stop = self.hand_gap()
        if stop > start:
            moments.update(
                {
                    "hand last seen": start - 1,
                    "hand lost (middle of longest gap)": (start + stop) // 2,
                    "hand back": stop,
                }
            )
        if self.hand_found.any():
            moments["tightest pinch"] = int(np.nanargmin(self.aperture_m))
            moments["widest pinch"] = int(np.nanargmax(self.aperture_m))
        seen: set[int] = set()
        return {
            label: min(index, len(self.times_s) - 1)
            for label, index in moments.items()
            if not (index in seen or seen.add(index))
        }

    def annotated_frames(self, indices: dict[str, int]) -> dict[str, np.ndarray]:
        """RGB frames with the detections drawn: marker outlines with ids, hand skeleton, and what is missing.

        Args:
            indices: label -> frame index, e.g. `key_moments()`.

        Returns:
            label -> RGB image, with a caption that says what the pipeline saw in that frame.
        """
        wanted = {index: label for label, index in indices.items()}
        panels: dict[str, np.ndarray] = {}
        expected = {"bowl pad": self.spec.bowl_marker_id, "plate marker": self.spec.plate_marker_id}
        for index, (frame, _) in enumerate(read_frames_with_timestamps(self.video)):
            if index not in wanted:
                continue
            self.draw_detections(frame, index, 1.0)
            missing = [
                name for name, marker_id in expected.items() if marker_id not in self.marker_ids[index]
            ]
            notes = [f"t={self.times_s[index]:.2f}s", f"board corners {self.board_corners[index]}"]
            notes += [f"NO {name}" for name in missing]
            if self.hand_landmarks_px[index] is None:
                notes.append("NO hand")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            panels[wanted[index]] = cv2.cvtColor(
                caption_frame(rgb, " | ".join(notes), 1.0), cv2.COLOR_BGR2RGB
            )
        return {label: panels[label] for label in indices if label in panels}

    def write_video(
        self,
        out_path: Path,
        width_px: int = 960,
        window_s: tuple[float, float] | None = None,
        only_window: bool = False,
    ) -> Path:
        """Diagnostic video: every frame with what was detected drawn on it and what is missing flagged.

        Marker outlines carry their role (bowl pad, plate marker, board), the hand skeleton is drawn when
        found, a red banner appears when it is lost, a status panel lists board / pad / plate / hand / pinch,
        and a strip along the bottom shows the whole clip's detections with a cursor at the current frame.

        Args:
            out_path: output .mp4; the parent folder is created.
            width_px: output width; the height keeps the aspect ratio.
            window_s: the clip's cut (start, end in clip time): frames outside it are dimmed and labelled
                "REMOVED BY CUT", and the kept range is marked on the timeline strip.
            only_window: write only the frames inside `window_s` (the cleaned clip).

        Returns:
            `out_path`.
        """
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = None
        try:
            for index, (frame, t_s) in enumerate(read_frames_with_timestamps(self.video)):
                kept = window_s is None or window_s[0] <= t_s <= window_s[1]
                if only_window and not kept:
                    continue
                scale = width_px / frame.shape[1]
                frame = cv2.resize(
                    frame, (width_px, round(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA
                )
                self.draw_detections(frame, index, scale)
                self._draw_status(frame, index, window_s)
                if not kept:
                    frame[:] = (frame * 0.45).astype(np.uint8)
                    draw_outlined_text(frame, "REMOVED BY CUT", (frame.shape[1] // 2 - 110, 100), 0.9)
                if writer is None:
                    writer = VideoWriter(
                        out_path, get_video_fps(self.video), (frame.shape[1], frame.shape[0])
                    )
                writer.write(frame)
        finally:
            if writer is not None:
                writer.close()
        return out_path

    def _marker_roles(self) -> dict[int, tuple[str, tuple[int, int, int]]]:
        """Marker id -> (label, BGR colour) for the objects; any other id belongs to the board."""
        return {
            self.spec.bowl_marker_id: ("bowl pad", (0, 140, 255)),
            self.spec.plate_marker_id: ("plate marker", (255, 0, 255)),
        }

    def draw_detections(self, frame_bgr: np.ndarray, index: int, scale: float) -> None:
        """Marker outlines (coloured by role) and the hand skeleton on a frame, in place.

        Args:
            frame_bgr: frame resized by `scale` relative to the video.
            index: frame index.
            scale: factor from video pixels to `frame_bgr` pixels.
        """
        roles = self._marker_roles()
        for marker_id, corners in zip(self.marker_ids[index], self.marker_corners[index], strict=True):
            label, colour = roles.get(marker_id, (None, (200, 200, 0)))
            points = (np.asarray(corners).reshape(4, 2) * scale).round().astype(np.int32)
            cv2.polylines(frame_bgr, [points], True, colour, 2 if label else 1, cv2.LINE_AA)
            if label:
                draw_outlined_text(
                    frame_bgr,
                    f"{label} (id {marker_id})",
                    tuple(points[0] - [0, 8]),
                    0.55 * scale + 0.1,
                    colour,
                )
        if self.hand_landmarks_px[index] is not None:
            draw_hand_skeleton(frame_bgr, self.hand_landmarks_px[index] * scale)

    def _draw_status(
        self, frame_bgr: np.ndarray, index: int, window_s: tuple[float, float] | None = None
    ) -> None:
        """Status panel, a red banner when the hand is lost, and the timeline strip, in place.

        With `window_s`, a bar above the strip marks the kept range.
        """
        height, width = frame_bgr.shape[:2]
        aperture = self.aperture_m[index]
        if np.isnan(aperture):
            pinch = ("pinch: no hand", False)
        else:
            state = (
                "closed"
                if aperture < self.close_threshold_m
                else "open"
                if aperture > self.open_threshold_m
                else "between"
            )
            pinch = (f"pinch {100 * aperture:.1f} cm ({state})", True)
        rows = [
            (f"board: {self.board_corners[index]} corners", bool(self.board_found[index])),
            ("bowl pad marker", self.spec.bowl_marker_id in self.marker_ids[index]),
            ("plate marker", self.spec.plate_marker_id in self.marker_ids[index]),
            ("hand", bool(self.hand_found[index])),
            pinch,
        ]
        for line, (text, found) in enumerate(rows):
            y = 26 + 24 * line
            cv2.circle(frame_bgr, (16, y - 6), 7, (0, 200, 0) if found else (0, 0, 255), -1, cv2.LINE_AA)
            draw_outlined_text(frame_bgr, text, (32, y), 0.55)
        draw_outlined_text(frame_bgr, f"t = {self.times_s[index]:.2f} s", (width - 130, 26), 0.55)
        if not self.hand_found[index]:
            cv2.rectangle(frame_bgr, (0, 0), (width - 1, height - 1), (0, 0, 255), 6)
            draw_outlined_text(frame_bgr, "HAND LOST", (width // 2 - 80, 60), 0.9, (0, 0, 255))
        lanes = [
            self.hand_found,
            self.marker_found(self.spec.plate_marker_id),
            self.marker_found(self.spec.bowl_marker_id),
            self.board_found,
        ]
        lane_h, top = 8, height - 8 - 8 * len(lanes)
        columns = np.minimum(np.arange(width) * len(self.times_s) // width, len(self.times_s) - 1)
        for lane, found in enumerate(lanes):
            strip = np.where(found[columns, None], np.array([0, 160, 0]), np.array([0, 0, 200]))
            frame_bgr[top + lane * lane_h : top + (lane + 1) * lane_h] = strip.astype(np.uint8)[None]
        if window_s is not None:
            x0, x1 = (int(np.searchsorted(self.times_s, t) * width // len(self.times_s)) for t in window_s)
            cv2.rectangle(frame_bgr, (x0, top - 8), (x1, top - 4), (0, 220, 255), -1)
        cursor = index * width // len(self.times_s)
        cv2.line(frame_bgr, (cursor, top - 4), (cursor, height - 1), (255, 255, 255), 2)


def diagnose_capture(
    video_path: Path,
    intrinsics: CameraIntrinsics,
    setup: RealSetup,
    spec: RecordingSpec | None = None,
    segmenter: GraspSegmenter | None = None,
) -> CaptureDiagnosis:
    """Run the capture check and also keep every frame's detections, to see why it failed.

    Args:
        video_path: test clip.
        intrinsics: calibration of the phone at the clip's resolution.
        setup: measured real setup.
        spec: markers; defaults to `RecordingSpec()`.
        segmenter: pinch thresholds; defaults to 3 cm close, 5 cm open (what the extraction uses).

    Returns:
        The per-frame record plus the `check_capture` report.
    """
    from hurobo.hand import HandTcpTracker

    spec = spec or RecordingSpec()
    segmenter = segmenter or GraspSegmenter()
    estimator = spec.estimator(intrinsics, setup)
    board = cv2.aruco.CharucoDetector(setup.board.make_board())
    times, corners_count, ids, corners = [], [], [], []
    landmarks, aperture, reprojection, depth = [], [], [], []
    with HandTcpTracker(intrinsics) as tracker:
        for frame, t_s in read_frames_with_timestamps(video_path):
            times.append(t_s)
            _, charuco_ids, _, _ = board.detectBoard(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            corners_count.append(0 if charuco_ids is None else len(charuco_ids))
            estimator.estimate(frame)
            detection = estimator.last_detection
            ids.append(list(detection.ids) if detection else [])
            corners.append(list(detection.corners) if detection else [])
            hand = tracker.process_frame(frame, int(round(t_s * 1000)))
            landmarks.append(None if hand is None else hand.landmarks_image_px)
            aperture.append(np.nan if hand is None else hand.pinch_aperture_m)
            reprojection.append(np.nan if hand is None else hand.reprojection_error_px)
            depth.append(np.nan if hand is None else float(hand.tcp_pos_cam[2]))
    return CaptureDiagnosis(
        video=video_path,
        report=check_capture(video_path, intrinsics, setup, spec),
        times_s=np.asarray(times),
        board_corners=np.asarray(corners_count),
        marker_ids=ids,
        marker_corners=corners,
        hand_landmarks_px=landmarks,
        aperture_m=np.asarray(aperture),
        reprojection_px=np.asarray(reprojection),
        tcp_depth_m=np.asarray(depth),
        spec=spec,
        close_threshold_m=segmenter.close_threshold_m,
        open_threshold_m=segmenter.open_threshold_m,
        min_board_corners=estimator.MIN_CORNERS,
    )


def session_commands(
    out_dir: Path = Path("data/raw/e1"),
    work_dir: Path = Path("data/e1"),
    setup_path: Path = Path("data/raw/setup_01/measurements.yaml"),
    intrinsics_path: Path = Path("data/camera_intrinsics_s10e.npz"),
    clips: int = 20,
    generated: int = 200,
) -> list[tuple[str, str]]:
    """The shell commands of the E1 recording session, in order, as (what it does, command).

    One definition, used by the guide, the notebook and the scripts' own help.

    Args:
        out_dir: where the raw clips and the plan live.
        work_dir: where extracted demos, replays and the generated dataset go.
        setup_path: the measurements file.
        intrinsics_path: the phone's intrinsics at the recording resolution.
        clips: number of clips to plan.
        generated: robot demos to generate.
    """
    demos = work_dir / "demos"
    generated_dir = work_dir / "generated"
    return [
        (
            "1. plan: phone pose, layouts and hand starts, plus the pad and plate marker to print",
            f"uv run python scripts/16_plan_recording.py --setup {setup_path} "
            f"--intrinsics {intrinsics_path} --clips {clips} --out {out_dir}",
        ),
        (
            "2. check a 10 s test clip before recording the session",
            f"uv run python scripts/17_check_capture.py {out_dir}/test.mp4 --setup {setup_path} "
            f"--intrinsics {intrinsics_path}",
        ),
        (
            "3. clean the clips: propose a cut per clip (one hand, one close-open cycle), then review them",
            f"uv run python scripts/20_clean_clips.py {out_dir} --setup {setup_path} "
            f"--intrinsics {intrinsics_path}\n"
            f"uv run python scripts/21_clip_cutter.py {out_dir} --setup {setup_path} "
            f"--intrinsics {intrinsics_path}",
        ),
        (
            "4. after recording: human clips to table-frame demos + QC report (inside the cuts)",
            f"uv run python scripts/02_extract_demos.py {out_dir}/clip_*.mp4 --setup {setup_path} "
            f"--intrinsics {intrinsics_path} --cuts {out_dir}/cuts.json --out {demos}",
        ),
        (
            "5. optional: the human dataset in LeRobot format (what you recorded, for the record)",
            f"uv run python scripts/11_export_human_dataset.py {demos} "
            f"--intrinsics {intrinsics_path} --out {work_dir}/human",
        ),
        (
            "6. replay every demo on the robot, from the home pose, in the scaled scene (videos)",
            f"MUJOCO_GL=egl uv run python scripts/03_replay.py {demos} --setup {setup_path} "
            f"--out {work_dir}/replay",
        ),
        (
            "7. generate the robot dataset (successful rollouts on new layouts)",
            f"MUJOCO_GL=egl uv run python scripts/04_generate_dataset.py {demos} --setup {setup_path} "
            f"--count {generated} --out {generated_dir}",
        ),
    ]


def render_board_screen(setup: RealSetup, ppi: float, screen_px: tuple[int, int]) -> np.ndarray:
    """The ChArUco board for a tablet: white screen, board at true size in the middle.

    Show it 1:1 (one image pixel per screen pixel). The squares are rounded to whole pixels, so measure one
    square on the glass afterwards and put the measured value in `measurements.yaml`.

    Args:
        setup: board squares, their size and marker size.
        ppi: pixels per inch of the tablet's display.
        screen_px: (width, height) of the display in pixels.

    Returns:
        shape (height, width) uint8 grayscale image.

    Raises:
        ValueError: if the board does not fit on the screen.
    """
    board = setup.board
    square_px = round(board.square_length_m * ppi / 0.0254)
    size = (board.squares_x * square_px, board.squares_y * square_px)
    width_px, height_px = screen_px
    if size[0] > width_px or size[1] > height_px:
        raise ValueError(f"board needs {size[0]}x{size[1]} px but the screen is {width_px}x{height_px}")
    # The board's own marker/square ratio must be kept, so render it from the board definition.
    image = board.make_board().generateImage(size, marginSize=0)
    sheet = np.full((height_px, width_px), 255, np.uint8)
    top, left = (height_px - size[1]) // 2, (width_px - size[0]) // 2
    sheet[top : top + size[1], left : left + size[0]] = image
    return sheet
