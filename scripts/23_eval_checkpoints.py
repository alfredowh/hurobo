#!/usr/bin/env python
"""Evaluates every checkpoint of a SmolVLA fine-tuning run (closed loop in the scaled scene + open loop).

    MUJOCO_GL=egl uv run python scripts/23_eval_checkpoints.py outputs/e1_smolvla_ckpts \
        --dataset data/e1/generated_clips --out results/smolvla_ckpts --episodes 10 --max-steps 400
    ... --watch --final-step 4000     # keep polling for new checkpoints until step 4000 is evaluated

Resumable: finished episodes and open-loop results in `--out` are skipped.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from hurobo.checkpoint_eval import CheckpointEvaluator, discover_checkpoints
from hurobo.real_setup import DEFAULT_MEASUREMENTS, RealSetup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_dir", type=Path, help="training output folder (--out of 19_train_smolvla.py)")
    parser.add_argument(
        "--dataset", type=Path, required=True, help="generated dataset for the open-loop error"
    )
    parser.add_argument("--out", type=Path, default=Path("results/smolvla_ckpts"))
    parser.add_argument("--setup", type=Path, default=DEFAULT_MEASUREMENTS)
    parser.add_argument("--condition", default="scaled")
    parser.add_argument("--replan", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--open-loop-frames", type=int, default=100)
    parser.add_argument("--videos", type=int, default=1)
    parser.add_argument("--watch", action="store_true", help="poll for new checkpoints")
    parser.add_argument("--final-step", type=int, default=None, help="with --watch: stop after this step")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluator = CheckpointEvaluator(
        args.run_dir.resolve(),
        args.out.resolve(),
        args.dataset.resolve(),
        RealSetup.load(args.setup),
        condition=args.condition,
        replan_every=args.replan,
        episodes=args.episodes,
        max_steps=args.max_steps,
        open_loop_frames=args.open_loop_frames,
        videos=args.videos,
    )
    while True:
        pending = evaluator.pending()
        for checkpoint in pending:
            evaluator.evaluate(checkpoint)
        if not args.watch:
            break
        steps = [checkpoint.step for checkpoint in discover_checkpoints(args.run_dir)]
        if not pending and args.final_step is not None and args.final_step in steps:
            break
        if not pending:
            time.sleep(30)


if __name__ == "__main__":
    main()
