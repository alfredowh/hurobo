"""Fine-tune SmolVLA on a generated robot dataset by composing the `lerobot-train` command."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

#: SmolVLA's pretrained base (6-D state and action, camera1..3); fine-tuning adapts it to LIBERO's 8-D / 7-D.
BASE_CHECKPOINT = "lerobot/smolvla_base"
#: Our dataset uses LIBERO's camera keys; the checkpoints expect camera1/camera2.
CAMERA_RENAME = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.image2": "observation.images.camera2",
}


@dataclass
class SmolVLATrainConfig:
    """Settings of one fine-tuning run.

    Attributes:
        dataset_root: local LeRobotDataset (from `scripts/04_generate_dataset.py`).
        repo_id: its repo id.
        output_dir: where checkpoints go.
        init_from: checkpoint to start from; `BASE_CHECKPOINT` or an already LIBERO-tuned one.
        steps: optimiser steps.
        batch_size: samples per step (32 is what `lerobot/smolvla_libero` used; lower it if the GPU runs out).
        save_freq: checkpoint every this many steps.
        train_expert_only: freeze the VLM and train only the action expert (cheap; SmolVLA's default for
            fine-tuning). False also trains the language model, as `lerobot/smolvla_libero` did.
        seed: training seed.
        num_workers: dataloader workers.
        learning_rate: peak learning rate; None keeps SmolVLA's default (1e-4).
        warmup_steps: linear warm-up steps; None keeps SmolVLA's default (1000).
        log_freq: log the loss every this many steps.
        decay_steps: length of the cosine decay after warm-up; None keeps SmolVLA's default (30000), which
            for a short run means the rate never decays, so short runs should set it to `steps`.
    """

    dataset_root: Path
    repo_id: str
    output_dir: Path
    init_from: str = BASE_CHECKPOINT
    steps: int = 20_000
    batch_size: int = 32
    save_freq: int = 5_000
    train_expert_only: bool = True
    seed: int = 1000
    num_workers: int = 4
    learning_rate: float | None = None
    warmup_steps: int | None = None
    decay_steps: int | None = None
    log_freq: int = 200

    def to_cli_args(self) -> list[str]:
        """Arguments for `lerobot-train`."""
        args = [
            f"--policy.path={self.init_from}",
            "--policy.device=cuda",
            "--policy.push_to_hub=false",
            f"--policy.train_expert_only={str(self.train_expert_only).lower()}",
            f"--policy.freeze_vision_encoder={str(self.train_expert_only).lower()}",
            f"--dataset.repo_id={self.repo_id}",
            f"--dataset.root={self.dataset_root}",
            "--dataset.video_backend=pyav",
            f"--rename_map={json.dumps(CAMERA_RENAME)}",
            f"--output_dir={self.output_dir}",
            f"--job_name={self.output_dir.name}",
            f"--steps={self.steps}",
            f"--batch_size={self.batch_size}",
            f"--save_freq={self.save_freq}",
            f"--seed={self.seed}",
            f"--num_workers={self.num_workers}",
            f"--log_freq={self.log_freq}",
            "--wandb.enable=false",
        ]
        optional = {
            "optimizer_lr": self.learning_rate,
            "scheduler_warmup_steps": self.warmup_steps,
            "scheduler_decay_steps": self.decay_steps,
        }
        args += [f"--policy.{name}={value}" for name, value in optional.items() if value is not None]
        return args

    def command(self) -> list[str]:
        """The full command line (run through `uv run`)."""
        return ["uv", "run", "lerobot-train", *self.to_cli_args()]

    def run(self) -> int:
        """Run the training and return its exit code."""
        return subprocess.run(self.command(), check=False).returncode

    def final_checkpoint(self) -> Path:
        """Folder with the last checkpoint's `pretrained_model`, usable as `CheckpointSpec.path`."""
        return self.output_dir / "checkpoints" / "last" / "pretrained_model"
