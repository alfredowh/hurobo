#!/usr/bin/env python
"""Exports extracted human demos (.npz from 02_extract_demos.py) as a LeRobotDataset.

Run with:
    uv run python scripts/11_export_human_dataset.py data/demos --out data/lerobot/human
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ego2robo.camera import CameraIntrinsics
from ego2robo.demo import HumanDemo
from ego2robo.human_dataset import export_human_dataset

DEFAULT_TASK = "put the bowl on the plate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("demos_dir", type=Path, help="directory with <clip>.npz and summary.json")
    parser.add_argument("--intrinsics", type=Path, default=Path("data/camera_intrinsics.npz"))
    parser.add_argument("--out", type=Path, default=Path("data/lerobot/human"))
    parser.add_argument("--repo-id", default="local/ego2robo_human_demos")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--image-height", type=int, default=360)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument(
        "--include-flagged", action="store_true", help="also export clips that failed a QC gate"
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = json.loads((args.demos_dir / "summary.json").read_text())
    selected = [r for r in summary if r["usable"] or args.include_flagged]
    skipped = len(summary) - len(selected)
    if not selected:
        raise SystemExit("No usable clips to export (see summary.json for the reasons).")

    demos = [HumanDemo.load(args.demos_dir / f"{Path(r['source_video']).stem}.npz") for r in selected]
    fps = round(1.0 / float(demos[0].timestamps_s[1] - demos[0].timestamps_s[0]))
    image_size_hw = (args.image_height, args.image_width)

    episodes = export_human_dataset(
        demos,
        args.out,
        CameraIntrinsics.load(args.intrinsics),
        args.task,
        image_size_hw,
        fps,
        args.repo_id,
        args.overwrite,
    )
    for episode, demo in zip(episodes, demos, strict=True):
        print(f"episode {episode}: {demo.source_video.name} ({len(demo.timestamps_s)} frames)")
    print(f"Wrote {len(demos)} episodes at {fps} fps to {args.out} ({skipped} clips skipped by QC)")


if __name__ == "__main__":
    main()
