"""Choosing the best checkpoint of a run and the comparison figures."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from hurobo.model_comparison import best_checkpoint
from hurobo.utils.comparison_view import progress_figure, success_figure


def _fake_run(tmp_path: Path, successes: dict[int, int], errors: dict[int, float]) -> tuple[Path, Path]:
    run, results = tmp_path / "run", tmp_path / "results"
    results.mkdir()
    rows, open_rows = [], []
    for step, count in successes.items():
        (run / "checkpoints" / f"{step:06d}" / "pretrained_model").mkdir(parents=True)
        (run / "checkpoints" / f"{step:06d}" / "pretrained_model" / "model.safetensors").write_text("x")
        for episode in range(10):
            rows.append(
                {
                    "cell": f"step{step:06d}|scaled|10",
                    "checkpoint": f"step{step:06d}",
                    "replan_every": 10,
                    "condition": "scaled",
                    "episode": episode,
                    "success": episode < count,
                    "steps": 100,
                    "min_eef_bowl_m": 0.05,
                    "max_bowl_lift_m": 0.0,
                }
            )
        open_rows.append(
            {
                "checkpoint": f"step{step:06d}",
                "frames": 10,
                "gripper_accuracy": 0.9,
                "mae_per_dim": [errors[step]] * 7,
                "zero_mae_per_dim": [0.2] * 7,
            }
        )
    (results / "closed_loop_episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (results / "open_loop_by_step.jsonl").write_text("".join(json.dumps(r) + "\n" for r in open_rows))
    return run, results


def test_best_checkpoint_has_the_highest_success(tmp_path: Path) -> None:
    run, results = _fake_run(tmp_path, {500: 1, 1000: 4, 1500: 2}, {500: 0.1, 1000: 0.2, 1500: 0.05})
    assert best_checkpoint(run, results).step == 1000


def test_best_checkpoint_breaks_ties_by_open_loop_error(tmp_path: Path) -> None:
    run, results = _fake_run(tmp_path, {500: 3, 1000: 3}, {500: 0.05, 1000: 0.2})
    assert best_checkpoint(run, results).step == 500


def test_best_checkpoint_without_results_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        best_checkpoint(tmp_path / "run", tmp_path / "empty")


def test_figures_draw_one_bar_or_line_per_entry() -> None:
    summary = pd.DataFrame(
        {
            "checkpoint": ["a", "b"],
            "episodes": [10, 10],
            "successes": [2, 5],
            "success_rate": [0.2, 0.5],
            "ci_low": [0.06, 0.24],
            "ci_high": [0.51, 0.76],
        }
    )
    assert len(success_figure(summary).data[0].x) == 2
    table = summary.assign(step=[500, 1000])
    assert len(progress_figure({"run": table}).data) == 1
