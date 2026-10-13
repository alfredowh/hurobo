#!/usr/bin/env python
"""Checks a short test clip against what the E1 extraction needs, before recording the whole session.

Record about 10 s with the final setup (board on, pad and plate marker in place, hand does one pick and
place), copy it over, then:

    uv run python scripts/17_check_capture.py data/raw/e1/test.mp4 \
        --intrinsics data/camera_intrinsics_s24.npz

Exit code 0 when nothing needs fixing, 1 otherwise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hurobo.camera import CameraIntrinsics
from hurobo.real_setup import DEFAULT_MEASUREMENTS, RealSetup
from hurobo.recording import check_capture


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("--setup", type=Path, default=DEFAULT_MEASUREMENTS)
    parser.add_argument("--intrinsics", type=Path, default=Path("data/camera_intrinsics_s24.npz"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = check_capture(args.video, CameraIntrinsics.load(args.intrinsics), RealSetup.load(args.setup))
    print(f"{args.video.name}: {report.frames_checked} frames checked")
    print(f"  board found in {report.board_fraction:.0%} of frames, {report.board_width_px:.0f} px wide")
    print(f"  bowl pad marker in first second: {report.pad_seen_early}")
    print(f"  plate marker in first second:    {report.plate_seen_early}")
    clip = report.clip
    if clip is not None:
        print(
            f"  hand in {clip.hand_fraction:.0%} of frames; grasp {clip.grasp_found}, release "
            f"{clip.release_found}; anchor scale {clip.anchor_scale}; grasp to bowl centre "
            f"{clip.grasp_to_bowl_m}; release to plate centre {clip.release_to_plate_xy_m}"
        )
    if report.ok:
        print("PASS: record the session.")
        return
    print("FIX BEFORE RECORDING:")
    for problem in report.problems:
        print(f"  - {problem}")
    sys.exit(1)


if __name__ == "__main__":
    main()
