#!/usr/bin/env python
"""Benchmarks SmolVLA checkpoints in LIBERO: closed-loop success per scene condition, plus open-loop error.

Closed loop runs the policy in the simulator under three conditions (stock LIBERO; cleared table;
cleared table with the real-size bowl and plate) and reports the success rate with a 95 % Wilson interval.
Open loop compares predicted action chunks with a demonstration dataset (no simulator).

    MUJOCO_GL=egl uv run python scripts/18_benchmark_policy.py --checkpoints lerobot hfvla \
        --conditions stock clean scaled --replan 10 --episodes 10 --out results/benchmark_pretrained
    # open loop against generated data (checkpoints may also be local paths: --checkpoint-path name=dir)
    MUJOCO_GL=egl uv run python scripts/18_benchmark_policy.py --checkpoints lerobot --open-loop-only \
        --open-loop-dataset data/e1/generated --out results/benchmark_pretrained

Reruns resume: finished episodes in `<out>/closed_loop_episodes.jsonl` are skipped.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from hurobo.benchmark import OPEN_LOOP_FILE, ClosedLoopBenchmark, OpenLoopEvaluator
from hurobo.policy_rollout import CONDITIONS, KNOWN_CHECKPOINTS, MAX_STEPS, CheckpointSpec, SmolVLAAgent
from hurobo.real_setup import DEFAULT_MEASUREMENTS, RealSetup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoints", nargs="*", default=[], choices=sorted(KNOWN_CHECKPOINTS))
    parser.add_argument(
        "--checkpoint-path",
        nargs="*",
        default=[],
        metavar="NAME=PATH",
        help="extra checkpoints (a local run's pretrained_model folder or a Hub id). Use the camera1/camera2 "
        "key convention of lerobot-train fine-tunes.",
    )
    parser.add_argument("--conditions", nargs="*", default=list(CONDITIONS), choices=CONDITIONS)
    parser.add_argument(
        "--replan",
        nargs="*",
        type=int,
        default=[10],
        help="actions executed per policy call (50 = open-loop chunks of 2.5 s, 1 = look every step)",
    )
    parser.add_argument("--episodes", type=int, default=10, help="per checkpoint x condition x replan")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS, help="episode length limit")
    parser.add_argument("--videos", type=int, default=1, help="episodes per cell saved as mp4")
    parser.add_argument("--setup", type=Path, default=DEFAULT_MEASUREMENTS)
    parser.add_argument("--out", type=Path, default=Path("results/benchmark"))
    parser.add_argument("--open-loop-dataset", type=Path, default=None)
    parser.add_argument("--open-loop-frames", type=int, default=200)
    parser.add_argument("--open-loop-only", action="store_true")
    return parser.parse_args()


def build_specs(args: argparse.Namespace) -> list[CheckpointSpec]:
    from hurobo.training import CAMERA_RENAME

    specs = [KNOWN_CHECKPOINTS[key] for key in args.checkpoints]
    for item in args.checkpoint_path:
        name, path = item.split("=", 1)
        specs.append(CheckpointSpec(name, path, rename_map=dict(CAMERA_RENAME)))
    if not specs:
        raise SystemExit("give --checkpoints and/or --checkpoint-path")
    return specs


def main() -> None:
    args = parse_args()
    specs = build_specs(args)
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.open_loop_only:
        benchmark = ClosedLoopBenchmark(
            RealSetup.load(args.setup), args.out, video_episodes=args.videos, max_steps=args.max_steps
        )
        summary = benchmark.run(specs, args.conditions, args.replan, args.episodes)
        summary.to_csv(args.out / "closed_loop_summary.csv", index=False)
        print(summary.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    if args.open_loop_dataset is not None:
        evaluator = OpenLoopEvaluator(args.open_loop_dataset, "local/open_loop")
        results = [
            asdict(evaluator.evaluate(SmolVLAAgent(spec), frames=args.open_loop_frames)) for spec in specs
        ]
        (args.out / OPEN_LOOP_FILE).write_text(json.dumps(results, indent=2))
        for result in results:
            print(
                f"{result['checkpoint']}: gripper accuracy {result['gripper_accuracy']:.2f}, "
                f"MAE per dim {[round(v, 2) for v in result['mae_per_dim']]} "
                f"(zero-action baseline {[round(v, 2) for v in result['zero_mae_per_dim']]})"
            )


if __name__ == "__main__":
    main()
