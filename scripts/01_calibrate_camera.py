#!/usr/bin/env python
"""Generates the ChArUco calibration board and calibrates a camera from it.

Run with:
    uv run python scripts/01_calibrate_camera.py generate-board
    uv run python scripts/01_calibrate_camera.py capture --camera 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from ego2robo.camera import ChArUcoCalibrator

BOARD_IMAGE_PATH = Path("data/charuco_board.png")
INTRINSICS_PATH = Path("data/camera_intrinsics.npz")


def generate_board(calibrator: ChArUcoCalibrator, out_path: Path) -> None:
    """Render the board to a PNG for printing, or for showing fullscreen (PLAN.md §2/§4)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    board_image = calibrator.generate_board_image()
    cv2.imwrite(str(out_path), board_image)
    print(f"Saved board image to {out_path}")
    print("Print it, or show it fullscreen on a screen, for the 'capture' step.")


def capture_and_calibrate(calibrator: ChArUcoCalibrator, camera_index: int, out_path: Path) -> None:
    """Live webcam loop: 'c' to capture the current frame, 'q'/ESC to calibrate and quit."""
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    num_captured = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Camera read failed mid-capture")

            preview = frame.copy()
            cv2.putText(
                preview,
                f"captured: {num_captured}  ('c' capture, 'q' finish)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.imshow("ChArUco calibration", preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("c"):
                if calibrator.add_frame(frame):
                    num_captured += 1
                    print(f"Captured frame {num_captured}")
                else:
                    print("Board not detected in this frame, try another angle/distance")
            elif key in (ord("q"), 27):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    intrinsics = calibrator.calibrate()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    intrinsics.save(out_path)
    print(f"Calibrated from {num_captured} frames. Saved intrinsics to {out_path}")
    print(f"  fx={intrinsics.fx:.1f} fy={intrinsics.fy:.1f} fovy={intrinsics.fovy_degrees:.1f} deg")


def main() -> int:
    """Parse args, build the calibrator, and dispatch to the requested subcommand."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate-board", help="Render the board to a PNG")
    generate_parser.add_argument("--out", type=Path, default=BOARD_IMAGE_PATH)

    capture_parser = subparsers.add_parser("capture", help="Capture frames from a camera and calibrate")
    capture_parser.add_argument("--camera", type=int, default=0, help="cv2.VideoCapture index")
    capture_parser.add_argument("--out", type=Path, default=INTRINSICS_PATH)

    args = parser.parse_args()
    calibrator = ChArUcoCalibrator()

    if args.command == "generate-board":
        generate_board(calibrator, args.out)
    elif args.command == "capture":
        capture_and_calibrate(calibrator, args.camera, args.out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
