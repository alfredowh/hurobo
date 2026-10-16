#!/usr/bin/env python
"""Plans an E1 recording session: where the phone goes, which layouts to record, what to print.

Run with:
    uv run python scripts/16_plan_recording.py --clips 20 --out data/raw/e1

Writes into --out: plan.csv (one row per clip), plan.html (interactive top view), plan_sheet.png (one small
top view per clip, to print and keep next to the table), bowl_pad.png and plate_marker.png (print at 100 %),
charuco_board.png (the table frame, shown 1:1 on the tablet) and table_marker.png (single ArUco id 20, 10 cm).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from hurobo.camera import CameraIntrinsics
from hurobo.capture_plan import CameraPlacementSearch, CapturePlanner, plan_rows, workspace_rect_table
from hurobo.markers import render_marker_image
from hurobo.real_setup import DEFAULT_MEASUREMENTS, RealSetup
from hurobo.recording import (
    RecordingSpec,
    render_board_screen,
    render_bowl_pad,
    render_plate_marker,
)
from hurobo.scaled_native import Workspace
from hurobo.utils.plotting import plot_clip_sheet
from hurobo.utils.viewer import capture_plan_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--setup", type=Path, default=DEFAULT_MEASUREMENTS)
    parser.add_argument("--intrinsics", type=Path, default=Path("data/camera_intrinsics_s10e.npz"))
    parser.add_argument("--clips", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("data/raw/e1"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tablet-ppi", type=float, default=217.0, help="pixels per inch of the tablet")
    parser.add_argument("--tablet-screen", default="1800x1200", help="tablet resolution WxH")
    parser.add_argument(
        "--min-board-px",
        type=float,
        default=250.0,
        help="board grid width the phone pose must give at the intrinsics' resolution",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    setup = RealSetup.load(args.setup)
    workspace = Workspace()
    spec = RecordingSpec()
    intrinsics = CameraIntrinsics.load(args.intrinsics)

    placements = CameraPlacementSearch(
        setup, workspace, intrinsics, min_board_width_px=args.min_board_px
    ).search()
    print(f"phone poses that see the board ({args.min_board_px:.0f}+ px), both objects and the whole reach:")
    if not placements:
        print("  none; lower --min-board-px or record in 4K")
    for placement in placements[:5]:
        x, y, z = placement.position_table
        print(
            f"  lens at x={x:+.2f} y={y:+.2f} z={z:.2f} m (table frame)  "
            f"board {placement.board_width_px:.0f} px  hand {placement.hand_width_px:.0f} px  "
            f"looks down {placement.elevation_deg:.0f} deg  margin {placement.min_margin:.0%}"
        )

    planner = CapturePlanner(setup, workspace, seed=args.seed)
    clips = planner.plan(args.clips)
    rows = plan_rows(clips, setup)
    with (args.out / "plan.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figure = capture_plan_figure(
        setup,
        workspace_rect_table(setup, workspace),
        clips,
        placements[0] if placements else None,
    )
    figure.write_html(args.out / "plan.html")
    plot_clip_sheet(setup, workspace_rect_table(setup, workspace), clips, planner).savefig(
        args.out / "plan_sheet.png", dpi=130
    )
    cv2.imwrite(str(args.out / "bowl_pad.png"), render_bowl_pad(spec, setup.bowl.diameter_m))
    cv2.imwrite(str(args.out / "plate_marker.png"), render_plate_marker(spec))
    screen = tuple(int(v) for v in args.tablet_screen.lower().split("x"))
    cv2.imwrite(str(args.out / "charuco_board.png"), render_board_screen(setup, args.tablet_ppi, screen))
    # Not used by E1 (the board is the table frame); the original single-marker table frame of scripts 02/13.
    cv2.imwrite(str(args.out / "table_marker.png"), render_marker_image(20))
    print(f"\nwrote the plan, the sheets and all marker images to {args.out}")
    print(
        "print bowl_pad.png and plate_marker.png at 100 % (A4 landscape); the black square must measure 60 mm"
    )


if __name__ == "__main__":
    main()
