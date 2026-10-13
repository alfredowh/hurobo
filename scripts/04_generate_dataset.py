#!/usr/bin/env python
"""Generates a LeRobotDataset of robot demos from human demos, in the scaled LIBERO scene.

Every rollout starts from the robot's home pose on a freshly sampled bowl/plate layout; only rollouts
that LIBERO's success check accepts are written.

    MUJOCO_GL=egl uv run python scripts/04_generate_dataset.py data/e1/demos --count 200 \
        --setup data/raw/setup_01/measurements.yaml --out data/e1/generated

Train on it with lerobot-train (see docs/experiment_scaled_native.md).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from hurobo.demo import HumanDemo
from hurobo.generate import FPS, DemoGenerator, sim_features
from hurobo.real_setup import DEFAULT_MEASUREMENTS, RealSetup
from hurobo.recorder import LeRobotRecorder
from hurobo.scaled_native import ScaledNativeScene

TASK = "put the bowl on the plate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("demos_dir", type=Path, help="folder of <clip>.npz from 02_extract_demos.py")
    parser.add_argument("--setup", type=Path, default=DEFAULT_MEASUREMENTS)
    parser.add_argument("--out", type=Path, default=Path("data/e1/generated"))
    parser.add_argument("--repo-id", default="local/hurobo_e1_scaled_native")
    parser.add_argument("--count", type=int, default=100, help="successful episodes to write")
    parser.add_argument("--max-attempts", type=int, default=None, help="default: 4 x count")
    parser.add_argument("--keep-distractors", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup = RealSetup.load(args.setup)
    demos = [HumanDemo.load(path) for path in sorted(args.demos_dir.glob("*.npz"))]
    scene = ScaledNativeScene(setup, keep_distractors=args.keep_distractors, seed=args.seed)
    generator = DemoGenerator(scene, demos, seed=args.seed)
    with LeRobotRecorder(
        args.out, args.repo_id, FPS, sim_features(), robot_type="panda", overwrite=args.overwrite
    ) as recorder:
        stats = generator.generate(args.count, recorder, TASK, args.max_attempts)
        recorder.write_sidecar(
            {
                "experiment": "E1 scaled-native",
                "human_demos": [d.source_video.name for d in demos],
                "attempted": stats.attempted,
                "succeeded": stats.succeeded,
                "failures": dict(stats.failures),
                "episode_steps": stats.steps,
                "object_scale_bowl": scene.scales.bowl,
                "object_scale_plate": scene.scales.plate,
                "distractors_kept": args.keep_distractors,
                "seed": args.seed,
            }
        )
    print(
        f"{stats.succeeded}/{stats.attempted} rollouts succeeded ({stats.success_rate:.0%}); "
        f"failures: {dict(stats.failures)}"
    )
    print(f"dataset: {args.out}")


if __name__ == "__main__":
    main()
