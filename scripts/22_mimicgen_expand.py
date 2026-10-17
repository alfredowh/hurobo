#!/usr/bin/env python
"""Expands a few source demos into more robot demos, MimicGen-style, under data/e1/mimicgen/.

Differences from scripts/04_generate_dataset.py: the source demo is the one whose own layout is nearest to the
new layout (separately for the approach and the carry), the executed actions carry Gaussian noise, and an
episode longer than the evaluation's 300 steps is dropped. `--stock` rolls the same layouts out the 04 way
(uniform source, no noise) for a side-by-side count; `--dry-run` measures without writing a dataset.

    MUJOCO_GL=egl uv run python scripts/22_mimicgen_expand.py data/e1/demos_clips \
        --out data/e1/mimicgen/run_01 --attempts 100 --stock
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from hurobo.generate import FPS, sim_features
from hurobo.mimicgen import (
    EVAL_STEP_LIMIT,
    MimicGenConfig,
    MimicGenExpander,
    StockBaseline,
    load_source_demos,
    sample_layouts,
)
from hurobo.real_setup import DEFAULT_MEASUREMENTS, RealSetup
from hurobo.recorder import LeRobotRecorder
from hurobo.retarget import RetargetConfig
from hurobo.scaled_native import ScaledNativeScene

TASK = "put the bowl on the plate"
OUTPUT_ROOT = Path("data/e1/mimicgen")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "demos_dir",
        type=Path,
        help="folder of <clip>.npz and summary.json (read only); only usable clips are sources",
    )
    parser.add_argument("--setup", type=Path, default=DEFAULT_MEASUREMENTS)
    parser.add_argument("--out", type=Path, required=True, help=f"run folder under {OUTPUT_ROOT}")
    parser.add_argument("--attempts", type=int, default=100, help="rollouts (layouts) to try")
    parser.add_argument("--target", type=int, default=None, help="stop after this many kept episodes")
    parser.add_argument("--nn-k", type=int, default=3, help="0 = uniform source choice")
    parser.add_argument("--shared-source", action="store_true", help="one demo for approach and carry")
    parser.add_argument("--noise", type=float, default=0.05, help="action noise sigma (0 = off)")
    parser.add_argument("--max-steps", type=int, default=EVAL_STEP_LIMIT)
    parser.add_argument("--action-limit", type=float, default=None, help="default: RetargetConfig's")
    parser.add_argument("--approach-radius", type=float, default=None, help="m of approach replayed")
    parser.add_argument("--paced", action="store_true", help="follow the demo clock")
    parser.add_argument("--slowdown", type=float, nargs=2, default=(1.0, 1.0), metavar=("LOW", "HIGH"))
    parser.add_argument("--screen", type=int, default=0, help="screen each source on this many layouts first")
    parser.add_argument("--screen-min", type=float, default=0.25, help="success share a source must reach")
    parser.add_argument("--stock", action="store_true", help="also roll the same layouts out the 04 way")
    parser.add_argument("--dry-run", action="store_true", help="measure only; write no dataset")
    parser.add_argument("--keep-distractors", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def retarget_config(args: argparse.Namespace) -> RetargetConfig:
    config = RetargetConfig(paced=args.paced)
    if args.action_limit is not None:
        config = replace(config, action_limit=args.action_limit)
    if args.approach_radius is not None:
        config = replace(config, approach_radius_m=args.approach_radius)
    return config


def main() -> None:
    args = parse_args()
    if OUTPUT_ROOT.resolve() not in args.out.resolve().parents:
        raise SystemExit(f"--out must be inside {OUTPUT_ROOT}")
    demos, names = load_source_demos(args.demos_dir)
    print(f"{len(demos)} source demos: {names}")
    scene = ScaledNativeScene(
        RealSetup.load(args.setup), keep_distractors=args.keep_distractors, seed=args.seed
    )
    layouts = sample_layouts(scene, args.attempts, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    config = MimicGenConfig(
        nn_k=args.nn_k,
        per_subtask=not args.shared_source,
        action_noise=args.noise,
        max_episode_steps=args.max_steps,
        slowdown_range=tuple(args.slowdown),
        retarget=retarget_config(args),
        seed=args.seed,
    )
    expander = MimicGenExpander(scene, demos, names, config)
    results = {}
    if args.screen:
        screening = expander.screen(sample_layouts(scene, args.screen, args.seed + 1000), args.screen_min)
        print("screening (rollouts, kept):", screening)
        results["screening"] = screening
    if args.stock:
        stock = StockBaseline(scene, demos, names, config.retarget, args.seed).run(layouts)
        print(stock.summary("stock (04-style)"))
        results["stock"] = stock.to_dict()
    if args.dry_run:
        report = expander.expand(layouts, None, TASK, args.target)
    else:
        with LeRobotRecorder(
            args.out / "dataset", "local/hurobo_e1_mimicgen", FPS, sim_features(), "panda", overwrite=True
        ) as recorder:
            report = expander.expand(layouts, recorder, TASK, args.target)
            recorder.write_sidecar(
                {
                    "experiment": "E1 mimicgen-style expansion",
                    "config": repr(config),
                    "screening": results.get("screening"),
                    **report.to_dict(),
                }
            )
    print(report.summary("mimicgen-style"))
    results["mimicgen"] = report.to_dict()
    (args.out / "report.json").write_text(json.dumps({"config": repr(config), **results}, indent=2))
    print(f"report: {args.out / 'report.json'}")


if __name__ == "__main__":
    main()
