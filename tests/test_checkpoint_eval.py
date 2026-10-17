"""Checkpoint discovery, training-log parsing and per-step tables (no simulator, no GPU)."""

from __future__ import annotations

import json
from pathlib import Path

from hurobo.benchmark import OpenLoopResult
from hurobo.checkpoint_eval import (
    append_open_loop,
    checkpoint_name,
    checkpoint_table,
    discover_checkpoints,
    parse_training_log,
)
from hurobo.training import SmolVLATrainConfig


def _make_checkpoint(run: Path, step: int, complete: bool = True) -> None:
    folder = run / "checkpoints" / f"{step:06d}" / "pretrained_model"
    folder.mkdir(parents=True)
    if complete:
        (folder / "model.safetensors").write_bytes(b"x")


def test_discover_checkpoints_sorts_and_skips_incomplete(tmp_path: Path) -> None:
    for step, complete in [(2000, True), (500, True), (1000, False)]:
        _make_checkpoint(tmp_path, step, complete)
    (tmp_path / "checkpoints" / "last").mkdir()
    found = discover_checkpoints(tmp_path)
    assert [c.step for c in found] == [500, 2000]
    assert found[0].name == "step000500"
    assert found[0].spec().rename_map["observation.images.image"] == "observation.images.camera1"
    assert discover_checkpoints(tmp_path / "missing") == []


def test_parse_training_log_rebuilds_steps_from_position(tmp_path: Path) -> None:
    log = tmp_path / "train.log"
    log.write_text(
        "INFO x step:100 smpl:2K ep:7 epch:0.1 loss:0.512 grdn:3.1 lr:1.0e-04 updt_s:0.5\n"
        "noise line\n"
        "INFO x step:2K smpl:4K ep:9 epch:0.2 loss:0.250 grdn:2.0 lr:9.0e-05 updt_s:0.5\n"
    )
    table = parse_training_log(log, log_freq=100)
    assert table["step"].tolist() == [100, 200]
    assert table["loss"].tolist() == [0.512, 0.25]
    assert parse_training_log(tmp_path / "none.log", 100).empty


def _episode(step: int, episode: int, success: bool, lift: float = 0.0, near: float = 0.1) -> dict:
    name = checkpoint_name(step)
    return {
        "cell": f"{name}|scaled|10",
        "checkpoint": name,
        "replan_every": 10,
        "condition": "scaled",
        "episode": episode,
        "success": success,
        "steps": 200 if success else 400,
        "min_eef_bowl_m": near,
        "max_bowl_lift_m": lift,
        "final_bowl_plate_xy_m": 0.1,
        "policy_calls": 20,
    }


def test_checkpoint_table_joins_closed_and_open_loop(tmp_path: Path) -> None:
    rows = [_episode(500, 0, False), _episode(500, 1, False, near=0.03), _episode(1000, 0, True)]
    (tmp_path / "closed_loop_episodes.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    append_open_loop(
        tmp_path,
        OpenLoopResult("step000500", 50, [0.1] * 7, [0.3] * 7, 0.8, [0.1] * 50),
    )
    table = checkpoint_table(tmp_path)
    assert table["step"].tolist() == [500, 1000]
    first = table.iloc[0]
    assert first["episodes"] == 2 and first["successes"] == 0
    assert first["never_near_bowl_failed"] == 1
    assert first["gripper_accuracy"] == 0.8 and first["zero_mae_pos"] == 0.3
    assert table.iloc[1]["successes"] == 1


def test_checkpoint_table_is_empty_without_results(tmp_path: Path) -> None:
    assert checkpoint_table(tmp_path).empty


def test_train_config_passes_schedule_overrides_only_when_set(tmp_path: Path) -> None:
    base = SmolVLATrainConfig(tmp_path, "local/x", tmp_path / "out")
    assert not any("optimizer_lr" in arg for arg in base.to_cli_args())
    tuned = SmolVLATrainConfig(
        tmp_path, "local/x", tmp_path / "out", learning_rate=2e-4, warmup_steps=100, decay_steps=4000
    )
    args = tuned.to_cli_args()
    assert "--policy.optimizer_lr=0.0002" in args
    assert "--policy.scheduler_warmup_steps=100" in args
    assert "--policy.scheduler_decay_steps=4000" in args
