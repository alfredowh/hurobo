#!/usr/bin/env python
"""Plans an E1 recording session: where the phone goes, which layouts to record, what to print.

Run with:
    uv run python scripts/16_plan_recording.py --clips 20 --out data/raw/e1

Writes into --out: plan.csv (one row per clip), plan.html (interactive top view), plan_sheet.png (one small
top view per clip, to print and keep next to the table), bowl_pad.png and plate_marker.png (print at 100 %).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from hurobo.camera import CameraIntrinsics
from hurobo.capture_plan import CameraPlacementSearch, CapturePlanner, workspace_rect_table
from hurobo.real_setup import DEFAULT_MEASUREMENTS, RealSetup
from hurobo.recording import RecordingSpec, render_bowl_pad, render_plate_marker
from hurobo.scaled_native import Workspace
from hurobo.utils.plotting import plot_clip_sheet
from hurobo.utils.viewer import capture_plan_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--setup", type=Path, default=DEFAULT_MEASUREMENTS)
    parser.add_argument("--intrinsics", type=Path, default=Path("data/camera_intrinsics_s24.npz"))
    parser.add_argument("--clips", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("data/raw/e1"))
    parser.add_argument("--seed", type=int, default=0)
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
    with (args.out / "plan.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "clip",
                "bowl_x_table_m",
                "bowl_y_table_m",
                "bowl_from_left_edge_m",
                "bowl_from_near_edge_m",
                "plate_x_table_m",
                "plate_y_table_m",
                "plate_from_left_edge_m",
                "plate_from_near_edge_m",
                "hand_start",
                "grasp_side",
            ]
        )
        for clip in clips:
            bowl_edges = clip.from_table_edges_m(clip.bowl_xy_table, setup)
            plate_edges = clip.from_table_edges_m(clip.plate_xy_table, setup)
            writer.writerow(
                [
                    f"clip_{clip.index:03d}",
                    *(f"{v:.3f}" for v in (*clip.bowl_xy_table, *bowl_edges)),
                    *(f"{v:.3f}" for v in (*clip.plate_xy_table, *plate_edges)),
                    clip.hand_start.name,
                    clip.grasp_side,
                ]
            )

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
    print(f"\nwrote plan.csv, plan.html, plan_sheet.png, bowl_pad.png, plate_marker.png to {args.out}")
    print(
        "print bowl_pad.png and plate_marker.png at 100 % (A4 landscape); the black square must measure 60 mm"
    )


if __name__ == "__main__":
    main()
