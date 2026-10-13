#!/usr/bin/env python
"""Replays extracted human demos as robot pick-and-places in the scaled LIBERO scene, with videos.

Each demo is retargeted onto the layout the human recorded (the sim twin of the clip) or onto a random
layout, executed from the robot's home pose, and written as an agent-view video with phase captions.

    MUJOCO_GL=egl uv run python scripts/03_replay.py data/e1/demos \
        --setup data/raw/setup_01/measurements.yaml --out data/e1/replay

Use --random-layout to see the same demo on new object positions (what generation does).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hurobo.demo import HumanDemo
from hurobo.generate import DemoGenerator
from hurobo.real_setup import DEFAULT_MEASUREMENTS, RealSetup
from hurobo.scaled_native import ScaledNativeScene
from hurobo.utils.video import VideoWriter, caption_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("demos_dir", type=Path, help="folder of <clip>.npz from 02_extract_demos.py")
    parser.add_argument("--setup", type=Path, default=DEFAULT_MEASUREMENTS)
    parser.add_argument("--out", type=Path, default=Path("data/e1/replay"))
    parser.add_argument("--random-layout", action="store_true", help="new object positions, not the human's")
    parser.add_argument(
        "--keep-distractors", action="store_true", help="leave LIBERO's other objects in place"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=480, help="video side in pixels")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    setup = RealSetup.load(args.setup)
    scene = ScaledNativeScene(setup, keep_distractors=args.keep_distractors, seed=args.seed)
    paths = sorted(args.demos_dir.glob("*.npz"))
    demos = [HumanDemo.load(path) for path in paths]
    generator = DemoGenerator(scene, demos, seed=args.seed)

    results = []
    for path, demo in zip(paths, demos, strict=True):
        layout = None if args.random_layout else generator.layout_of_demo(demo)
        if layout is not None and not (
            scene.workspace.contains(layout.bowl_xy) and scene.workspace.contains(layout.plate_xy)
        ):
            print(f"[SKIP] {path.stem}: its layout is outside the reachable work area")
            results.append({"clip": path.stem, "skipped": "layout outside the work area"})
            continue
        steps, executor = generator.rollout(demo, layout)
        video = args.out / f"{path.stem}.mp4"
        with VideoWriter(video, 20.0, (args.size, args.size)) as writer:
            for step in steps:
                image = scene.task.render_camera("agentview", args.size, args.size)
                writer.write(caption_frame(np.ascontiguousarray(image), step.phase))
        result = executor.result
        outcome = "SUCCESS" if result.success else f"FAILED ({result.failure})"
        print(f"{path.stem}: {outcome}, {result.steps} steps -> {video}")
        results.append(
            {"clip": path.stem, "success": result.success, "steps": result.steps, "failure": result.failure}
        )
    (args.out / "results.json").write_text(json.dumps(results, indent=2))
    ran = [r for r in results if "success" in r]
    print(f"{sum(r['success'] for r in ran)}/{len(ran)} replays succeeded")


if __name__ == "__main__":
    main()
