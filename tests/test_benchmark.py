"""Benchmark bookkeeping: intervals, summaries, resume keys and the training command (no simulator or GPU)."""

from __future__ import annotations

from pathlib import Path

import pytest

from hurobo.benchmark import ClosedLoopBenchmark, ClosedLoopCell, summarize_rows
from hurobo.policy_rollout import CONDITIONS, KNOWN_CHECKPOINTS
from hurobo.real_setup import RealSetup
from hurobo.training import CAMERA_RENAME, SmolVLATrainConfig
from hurobo.utils.stats import wilson_interval


def test_wilson_interval_is_sensible_at_the_edges() -> None:
    assert wilson_interval(0, 0) == (0.0, 1.0)
    low, high = wilson_interval(0, 10)
    assert low == 0.0 and 0.2 < high < 0.35
    low, high = wilson_interval(10, 10)
    assert high == 1.0 and 0.65 < low < 0.8
    low, high = wilson_interval(5, 10)
    assert low < 0.5 < high


def _row(checkpoint: str, condition: str, episode: int, success: bool, lift: float = 0.0) -> dict:
    return {
        "cell": f"{checkpoint}|{condition}|10",
        "checkpoint": checkpoint,
        "replan_every": 10,
        "condition": condition,
        "episode": episode,
        "success": success,
        "steps": 100 if success else 300,
        "min_eef_bowl_m": 0.05,
        "max_bowl_lift_m": lift,
        "final_bowl_plate_xy_m": 0.1,
        "policy_calls": 10,
    }


def test_summary_counts_successes_and_lifts_per_cell() -> None:
    rows = [_row("a", "stock", i, i < 3, lift=0.1 if i < 3 else 0.0) for i in range(4)]
    rows += [_row("a", "scaled", i, False) for i in range(2)]
    table = summarize_rows(rows).set_index(["checkpoint", "condition"])
    stock = table.loc[("a", "stock")]
    assert (stock["episodes"], stock["successes"]) == (4, 3)
    assert stock["success_rate"] == pytest.approx(0.75)
    assert stock["lifted"] == pytest.approx(0.75)
    assert stock["mean_steps_success"] == 100
    assert table.loc[("a", "scaled"), "success_rate"] == 0.0
    assert summarize_rows([]).empty


def test_results_file_lets_a_rerun_skip_finished_episodes(tmp_path: Path) -> None:
    benchmark = ClosedLoopBenchmark(None, tmp_path)  # no simulator is built until `run`
    cell = ClosedLoopCell("a", "scaled", 10)
    assert benchmark.done_episodes() == set()
    (tmp_path / "closed_loop_episodes.jsonl").write_text(
        f'{{"cell": "{cell.key}", "episode": 0, "checkpoint": "a", "replan_every": 10}}\n'
    )
    assert benchmark.done_episodes() == {(cell.key, 0)}


def test_the_registry_covers_the_two_popular_checkpoints_with_the_right_camera_keys() -> None:
    assert set(KNOWN_CHECKPOINTS) == {"hfvla", "lerobot"}
    assert KNOWN_CHECKPOINTS["hfvla"].rename_map == {}
    assert KNOWN_CHECKPOINTS["lerobot"].rename_map == CAMERA_RENAME
    assert CONDITIONS == ("stock", "clean", "scaled", "clean_stock_layout", "scaled_stock_layout")


def test_train_command_maps_cameras_and_uses_the_given_paths(tmp_path: Path) -> None:
    config = SmolVLATrainConfig(tmp_path / "data", "local/x", tmp_path / "run", steps=7, batch_size=3)
    args = config.to_cli_args()
    assert "--policy.path=lerobot/smolvla_base" in args
    assert f"--dataset.root={tmp_path / 'data'}" in args
    assert "--steps=7" in args and "--batch_size=3" in args
    assert any(a.startswith("--rename_map=") and "camera1" in a for a in args)
    assert config.final_checkpoint() == tmp_path / "run" / "checkpoints" / "last" / "pretrained_model"


def test_scaled_condition_needs_the_real_measurements() -> None:
    # The scaled scene is built from measurements.yaml; make sure the file the benchmark defaults to loads.
    setup = RealSetup.load()
    assert setup.bowl.diameter_m < setup.plate.diameter_m
