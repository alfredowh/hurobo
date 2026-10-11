#!/usr/bin/env python
"""Generates the ChArUco calibration board and calibrates a camera from it.

Run with:
    uv run python scripts/01_calibrate_camera.py generate-screen-markers --background table.jpg
    uv run python scripts/01_calibrate_camera.py generate-board
    uv run python scripts/01_calibrate_camera.py generate-markers
    uv run python scripts/01_calibrate_camera.py capture --source webcam:0
    uv run python scripts/01_calibrate_camera.py capture --source http://PHONE_IP:8080/video

Calibrate with the camera in the same mode (resolution, zoom, lens) it will have when teleoperating.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

from hurobo.camera import ChArUcoCalibrator
from hurobo.camera_source import open_frame_source, parse_source
from hurobo.live import LiveWindow
from hurobo.markers import MarkerLayout, render_marker_image, render_screen_marker_sheet

BOARD_IMAGE_PATH = Path("data/charuco_board.png")
INTRINSICS_PATH = Path("data/camera_intrinsics.npz")


def generate_board(calibrator: ChArUcoCalibrator, out_path: Path) -> None:
    """Render the board to a PNG for printing, or for showing fullscreen (PLAN.md §2/§4)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    board_image = calibrator.generate_board_image()
    cv2.imwrite(str(out_path), board_image)
    print(f"Saved board image to {out_path}")
    print("Print it, or show it fullscreen on a screen, for the 'capture' step.")


def generate_markers(out_dir: Path, ids: tuple[int, ...]) -> None:
    """Write printable table/bowl/plate markers (the black square must be printed at the stated size)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for marker_id in ids:
        path = out_dir / f"aruco_4x4_50_id{marker_id}.png"
        cv2.imwrite(str(path), render_marker_image(marker_id))
        print(f"Saved {path}")
    print(
        "Print at 100% scale; scale so the BLACK square is exactly 10.0 cm (table marker) / 5.0 cm (objects)."
    )


def generate_screen_markers(
    out_path: Path,
    ppi: float,
    screen_px: tuple[int, int],
    background: Path | None,
    transparent: bool = False,
) -> None:
    """Write a full-screen image with all three markers at true size, for a tablet instead of a printer."""
    layout = MarkerLayout(20, 0.10, {"bowl": 21, "plate": 22}, 0.05)
    photo = None if background is None else cv2.imread(str(background))
    if background is not None and photo is None:
        raise SystemExit(f"cannot read background image {background}")
    sheet = render_screen_marker_sheet(layout, ppi, screen_px, photo, transparent)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)
    print(f"Saved {out_path} ({screen_px[0]}x{screen_px[1]} px at {ppi:g} PPI)")
    print(
        "Show it fullscreen at 100% (1 image px = 1 screen px; Windows Photos may rescale, so check "
        "Display scaling), brightness high, matte/anti-glare if you can.\n"
        "Then measure the table marker's BLACK square with a ruler: it must be 100 mm (objects 50 mm). "
        "If it is not, pass the measured size via --ppi (PPI x measured/100)."
    )


def capture_and_calibrate(
    calibrator: ChArUcoCalibrator, source: str, width: int, height: int, out_path: Path
) -> None:
    """Live loop: 'c' captures the current frame, 'q'/Esc calibrates and quits."""
    frames = open_frame_source(parse_source(source), width, height)
    window = LiveWindow("ChArUco calibration")
    num_captured, last_index = 0, -1
    try:
        while not window.closed and not frames.ended:
            newest = frames.latest()
            if newest is None:
                window.process_events()
                time.sleep(0.01)
                continue
            frame, index = newest
            preview = frame.copy()
            text = f"captured: {num_captured}  ('c' capture, 'q' finish)"
            cv2.putText(preview, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            window.show(preview)
            if window.consume("c") and index != last_index:
                last_index = index
                if calibrator.add_frame(frame):
                    num_captured += 1
                    print(f"Captured frame {num_captured}")
                else:
                    print("Board not detected in this frame, try another angle/distance")
            if window.consume("q") or window.consume("escape"):
                break
    finally:
        window.close()
        frames.close()

    try:
        intrinsics = calibrator.calibrate()
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    out_path.parent.mkdir(parents=True, exist_ok=True)
    intrinsics.save(out_path)
    print(f"Calibrated from {num_captured} frames. Saved intrinsics to {out_path}")
    print(
        f"  {intrinsics.image_width}x{intrinsics.image_height} fx={intrinsics.fx:.1f} fy={intrinsics.fy:.1f}"
    )
    print(f"  fovy={intrinsics.fovy_degrees:.1f} deg")


def main() -> int:
    """Parse args, build the calibrator, and dispatch to the requested subcommand."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    screen_parser = subparsers.add_parser(
        "generate-screen-markers", help="Render all three markers at true size for a tablet screen"
    )
    screen_parser.add_argument("--out", type=Path, default=Path("data/markers/screen_markers.png"))
    screen_parser.add_argument("--ppi", type=float, default=217.0, help="display pixels per inch")
    screen_parser.add_argument("--screen", default="1800x1200", help="display resolution WxH")
    screen_parser.add_argument(
        "--transparent", action="store_true", help="PNG with alpha: only the markers are opaque"
    )
    screen_parser.add_argument("--background", type=Path, default=None, help="photo of the tabletop")

    generate_parser = subparsers.add_parser("generate-board", help="Render the board to a PNG")
    generate_parser.add_argument("--out", type=Path, default=BOARD_IMAGE_PATH)

    markers_parser = subparsers.add_parser("generate-markers", help="Render printable ArUco markers")
    markers_parser.add_argument("--out", type=Path, default=Path("data/markers"))
    markers_parser.add_argument("--ids", type=int, nargs="+", default=[20, 21, 22])

    capture_parser = subparsers.add_parser("capture", help="Capture frames from a camera and calibrate")
    capture_parser.add_argument(
        "--source",
        default="webcam:0",
        help="webcam[:N], /dev/videoN or a stream URL (see 13_teleop.py sources)",
    )
    capture_parser.add_argument("--width", type=int, default=1280)
    capture_parser.add_argument("--height", type=int, default=720)
    capture_parser.add_argument("--out", type=Path, default=INTRINSICS_PATH)

    args = parser.parse_args()
    calibrator = ChArUcoCalibrator()

    if args.command == "generate-board":
        generate_board(calibrator, args.out)
    elif args.command == "generate-screen-markers":
        width, height = (int(v) for v in args.screen.lower().split("x"))
        generate_screen_markers(args.out, args.ppi, (width, height), args.background, args.transparent)
    elif args.command == "generate-markers":
        generate_markers(args.out, tuple(args.ids))
    elif args.command == "capture":
        capture_and_calibrate(calibrator, args.source, args.width, args.height, args.out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
