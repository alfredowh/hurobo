#!/usr/bin/env python
"""Fine-tunes SmolVLA on the generated robot dataset (wraps lerobot-train with the right camera mapping).

    uv run python scripts/19_train_smolvla.py data/e1/generated --out outputs/e1_smolvla --steps 20000
    uv run python scripts/19_train_smolvla.py data/e1/generated --dry-run     # print the command only

Then benchmark the result:
    MUJOCO_GL=egl uv run python scripts/18_benchmark_policy.py \
        --checkpoint-path e1=outputs/e1_smolvla/checkpoints/last/pretrained_model
"""

from __future__ import annotations

import argparse
from pathlib import Path

from hurobo.training import BASE_CHECKPOINT, SmolVLATrainConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dataset_root", type=Path, help="folder written by 04_generate_dataset.py")
    parser.add_argument("--repo-id", default="local/hurobo_e1_scaled_native")
    parser.add_argument("--out", type=Path, default=Path("outputs/e1_smolvla"))
    parser.add_argument("--init-from", default=BASE_CHECKPOINT, help="Hub id or local checkpoint")
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--save-freq", type=int, default=5_000)
    parser.add_argument("--lr", type=float, default=None, help="peak learning rate (default 1e-4)")
    parser.add_argument("--warmup-steps", type=int, default=None, help="default 1000")
    parser.add_argument(
        "--decay-steps", type=int, default=None, help="cosine decay length (default 30000; use --steps)"
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-freq", type=int, default=200)
    parser.add_argument(
        "--train-vlm", action="store_true", help="also train the language model (more memory)"
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SmolVLATrainConfig(
        dataset_root=args.dataset_root.resolve(),
        repo_id=args.repo_id,
        output_dir=args.out.resolve(),
        init_from=args.init_from,
        steps=args.steps,
        batch_size=args.batch_size,
        save_freq=args.save_freq,
        train_expert_only=not args.train_vlm,
        num_workers=args.num_workers,
        log_freq=args.log_freq,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        decay_steps=args.decay_steps,
    )
    print(" ".join(config.command()))
    if not args.dry_run:
        raise SystemExit(config.run())


if __name__ == "__main__":
    main()
