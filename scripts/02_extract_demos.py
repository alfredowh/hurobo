#!/usr/bin/env python
"""Extracts a table-frame HumanDemo (.npz) plus QC plots from each phone clip.

Run with:
    uv run python scripts/02_extract_demos.py data/raw/*.mp4 \
        --intrinsics data/camera_intrinsics.npz --out data/demos
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ego2robo.camera import CameraIntrinsics
from ego2robo.demo import GraspSegmenter
from ego2robo.extract import ClipReport, ExtractionError, extract_demo
from ego2robo.markers import MarkerLayout
from ego2robo.utils.plotting import plot_3d_trajectory, plot_aperture_trace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--intrinsics", type=Path, default=Path("data/camera_intrinsics.npz"))
    parser.add_argument("--out", type=Path, default=Path("data/demos"))
    parser.add_argument("--table-id", type=int, default=20)
    parser.add_argument("--bowl-id", type=int, default=21)
    parser.add_argument("--plate-id", type=int, default=22)
    parser.add_argument("--table-size-m", type=float, default=0.10)
    parser.add_argument("--object-size-m", type=float, default=0.05)
    parser.add_argument("--close-threshold-m", type=float, default=0.03)
    parser.add_argument("--open-threshold-m", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    intrinsics = CameraIntrinsics.load(args.intrinsics)
    layout = MarkerLayout(
        table_marker_id=args.table_id,
        table_marker_length_m=args.table_size_m,
        object_marker_ids={"bowl": args.bowl_id, "plate": args.plate_id},
        object_marker_length_m=args.object_size_m,
    )
    segmenter = GraspSegmenter(args.close_threshold_m, args.open_threshold_m)

    reports: list[ClipReport] = []
    for video_path in args.videos:
        try:
            demo, report = extract_demo(video_path, intrinsics, layout, segmenter)
        except ExtractionError as error:
            report = ClipReport(source_video=video_path, reasons=[str(error)])
            reports.append(report)
            print(f"[SKIP] {video_path.name}: {error}")
            continue
        demo.save(args.out / f"{video_path.stem}.npz")
        if len(demo.timestamps_s):
            plot_3d_trajectory(
                demo.tcp_pos_table,
                args.out / f"{video_path.stem}_traj.png",
                demo.is_grasping,
                video_path.stem,
            )
            plot_aperture_trace(
                demo.timestamps_s,
                demo.pinch_aperture_m,
                args.out / f"{video_path.stem}_aperture.png",
                args.close_threshold_m,
                args.open_threshold_m,
                demo.grasp_frame_idx,
                demo.release_frame_idx,
                video_path.stem,
            )
        reports.append(report)
        status = "OK  " if report.usable else "FLAG"
        print(f"[{status}] {video_path.name}: {'; '.join(report.reasons) or 'all gates passed'}")

    summary = [{**asdict(r), "source_video": str(r.source_video), "usable": r.usable} for r in reports]
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    usable_count = sum(r.usable for r in reports)
    print(f"{usable_count}/{len(reports)} clips usable; summary -> {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
