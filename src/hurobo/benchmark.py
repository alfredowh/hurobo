"""Benchmark automation: checkpoints x scene conditions x replan intervals, closed-loop and open-loop."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from hurobo.policy_rollout import (
    CONDITIONS,
    MAX_STEPS,
    CheckpointSpec,
    ClosedLoopRunner,
    EpisodeResult,
    EvalScene,
    SmolVLAAgent,
)
from hurobo.real_setup import RealSetup
from hurobo.utils.stats import wilson_interval

EPISODES_FILE = "closed_loop_episodes.jsonl"
OPEN_LOOP_FILE = "open_loop.json"


@dataclass(frozen=True)
class ClosedLoopCell:
    """One benchmark cell: a checkpoint in a condition at a replan interval.

    Attributes:
        checkpoint: the checkpoint's display name.
        condition: one of `CONDITIONS`.
        replan_every: actions executed per policy call.
    """

    checkpoint: str
    condition: str
    replan_every: int

    @property
    def key(self) -> str:
        """Stable identifier used in the results file."""
        return f"{self.checkpoint}|{self.condition}|{self.replan_every}"


class ClosedLoopBenchmark:
    """Runs every cell for N episodes, appending one JSON line per episode so a rerun resumes."""

    def __init__(
        self, setup: RealSetup, out_dir: Path, video_episodes: int = 1, max_steps: int = MAX_STEPS
    ) -> None:
        """Prepare the output folder.

        Args:
            setup: the real measurements (for the `scaled` condition).
            out_dir: results folder; created.
            video_episodes: how many episodes per cell are saved as an mp4 (the first ones).
            max_steps: episode length limit.
        """
        self._setup = setup
        self._out_dir = out_dir
        self._video_episodes = video_episodes
        self._max_steps = max_steps
        out_dir.mkdir(parents=True, exist_ok=True)
        self._episodes_path = out_dir / EPISODES_FILE

    def done_episodes(self) -> set[tuple[str, int]]:
        """(cell key, episode) pairs already in the results file."""
        return {(row["cell"], row["episode"]) for row in self.load_rows()}

    def load_rows(self) -> list[dict]:
        """All stored episode rows."""
        return load_episode_rows(self._out_dir)

    def run(
        self,
        checkpoints: Iterable[CheckpointSpec],
        conditions: Iterable[str],
        replan_values: Iterable[int | None],
        episodes: int,
    ) -> pd.DataFrame:
        """Run all missing episodes.

        Args:
            checkpoints: policies to test.
            conditions: subset of `CONDITIONS`.
            replan_values: replan intervals; None uses each checkpoint's own.
            episodes: episodes per cell.

        Returns:
            The summary table (`summarize`) of everything stored so far.
        """
        conditions = list(conditions)
        unknown = set(conditions) - set(CONDITIONS)
        if unknown:
            raise ValueError(f"unknown conditions {sorted(unknown)}; choose from {CONDITIONS}")
        scene = EvalScene(self._setup)
        runner = ClosedLoopRunner(scene, max_steps=self._max_steps)
        try:
            for spec in checkpoints:
                for replan in replan_values:
                    agent = SmolVLAAgent(spec, replan_every=replan)
                    for condition in conditions:
                        self._run_cell(
                            runner, agent, ClosedLoopCell(spec.name, condition, agent.replan_every), episodes
                        )
        finally:
            scene.close()
        return self.summarize()

    def _run_cell(
        self, runner: ClosedLoopRunner, agent: SmolVLAAgent, cell: ClosedLoopCell, episodes: int
    ) -> None:
        done = self.done_episodes()
        for episode in range(episodes):
            if (cell.key, episode) in done:
                continue
            video = None
            if episode < self._video_episodes:
                videos = self._out_dir / "videos"
                videos.mkdir(exist_ok=True)
                safe = cell.key.replace("/", "_").replace("|", "__")
                video = videos / f"{safe}__ep{episode}.mp4"
            result = runner.run_episode(agent, cell.condition, episode, video_path=video)
            self._append(cell, result)
            print(
                f"{cell.key} ep{episode}: {'SUCCESS' if result.success else 'fail'} "
                f"({result.steps} steps, bowl lifted {100 * result.max_bowl_lift_m:.0f} cm)",
                flush=True,
            )

    def _append(self, cell: ClosedLoopCell, result: EpisodeResult) -> None:
        row = {"cell": cell.key, "checkpoint": cell.checkpoint, "replan_every": cell.replan_every}
        row.update(result.as_dict())
        with self._episodes_path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")

    def summarize(self) -> pd.DataFrame:
        """One row per cell: success rate with its Wilson interval and the failure-stage statistics."""
        return summarize_rows(self.load_rows())


def load_episode_rows(out_dir: Path) -> list[dict]:
    """Episode rows of a benchmark folder (empty if nothing ran yet).

    Args:
        out_dir: the folder given to `ClosedLoopBenchmark`.
    """
    path = out_dir / EPISODES_FILE
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize_rows(rows: list[dict]) -> pd.DataFrame:
    """Aggregate episode rows into per-cell results.

    Args:
        rows: dicts as written by `ClosedLoopBenchmark`.

    Returns:
        Columns: checkpoint, condition, replan_every, episodes, successes, success_rate, ci_low, ci_high,
        lifted (share of episodes where the bowl rose > 3 cm), mean_steps_success, median_min_eef_bowl_cm.
    """
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    records = []
    for (checkpoint, condition, replan), group in frame.groupby(["checkpoint", "condition", "replan_every"]):
        successes = int(group["success"].sum())
        low, high = wilson_interval(successes, len(group))
        succeeded = group[group["success"]]
        records.append(
            {
                "checkpoint": checkpoint,
                "condition": condition,
                "replan_every": int(replan),
                "episodes": len(group),
                "successes": successes,
                "success_rate": successes / len(group),
                "ci_low": low,
                "ci_high": high,
                "lifted": float((group["max_bowl_lift_m"] > 0.03).mean()),
                "mean_steps_success": float(succeeded["steps"].mean()) if len(succeeded) else float("nan"),
                "median_min_eef_bowl_cm": 100 * float(group["min_eef_bowl_m"].median()),
            }
        )
    return pd.DataFrame(records)


@dataclass
class OpenLoopResult:
    """Offline action-chunk error of a policy against demonstrations.

    Attributes:
        checkpoint: display name.
        frames: number of dataset frames evaluated.
        mae_per_dim: mean absolute error per action dimension, in env units, shape (7,).
        zero_mae_per_dim: the error of the do-nothing chunk (all zeros) on the same steps, shape (7,); a
            policy is only informative on a dimension where `mae_per_dim` is below this.
        gripper_accuracy: share of steps where the predicted and demonstrated gripper commands share a sign.
        mae_by_horizon: mean absolute error (all dims) at each step of the chunk, shape (chunk,).
    """

    checkpoint: str
    frames: int
    mae_per_dim: list[float]
    zero_mae_per_dim: list[float]
    gripper_accuracy: float
    mae_by_horizon: list[float]


class OpenLoopEvaluator:
    """Compares predicted action chunks with demonstrated ones on recorded frames (no simulator)."""

    def __init__(self, dataset_root: Path, repo_id: str, chunk_size: int = 50, fps: int = 20) -> None:
        """Open the dataset with `chunk_size` future actions per frame.

        Args:
            dataset_root: local LeRobotDataset folder.
            repo_id: its repo id.
            chunk_size: actions per frame to compare (the policy's chunk size).
            fps: dataset frame rate, sets the spacing of the future actions.
        """
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self._chunk_size = chunk_size
        self._dataset = LeRobotDataset(
            repo_id,
            root=dataset_root,
            delta_timestamps={"action": [i / fps for i in range(chunk_size)]},
            video_backend="pyav",
        )

    def evaluate(self, agent: SmolVLAAgent, frames: int = 200, seed: int = 0) -> OpenLoopResult:
        """Predict a chunk at evenly random frames and compare it with the demonstration.

        The dataset's images are already in the layout the policy sees (`hurobo.generate.frame_from_obs`
        applies the same 180-degree turn as the evaluation processor), so the env processor is skipped.
        Frames whose chunk runs past the episode end are padded by LeRobot; those steps are excluded.

        Args:
            agent: policy.
            frames: number of frames to sample.
            seed: for the sample.
        """
        import torch

        rng = np.random.default_rng(seed)
        indices = rng.choice(len(self._dataset), size=min(frames, len(self._dataset)), replace=False)
        errors, signs, valid_all, actions_all = [], [], [], []
        for index in indices:
            item = self._dataset[int(index)]
            batch = {
                key: value.unsqueeze(0) if isinstance(value, torch.Tensor) else [value]
                for key, value in item.items()
                if key.startswith("observation.") or key in ("action", "task", "action_is_pad")
            }
            truth = item["action"].numpy()
            valid = ~item["action_is_pad"].numpy()
            prediction = agent.predict_chunk(batch)[0][: len(truth)]
            errors.append(np.abs(prediction - truth))
            signs.append(np.sign(prediction[:, 6]) == np.sign(truth[:, 6]))
            valid_all.append(valid)
            actions_all.append(truth)
        errors, signs, valid = np.stack(errors), np.stack(signs), np.stack(valid_all)
        mask = valid[..., None]
        mae = (errors * mask).sum(axis=(0, 1)) / mask.sum()
        zero_mae = (np.abs(np.stack(actions_all)) * mask).sum(axis=(0, 1)) / mask.sum()
        by_horizon = (errors.mean(axis=2) * valid).sum(axis=0) / np.maximum(valid.sum(axis=0), 1)
        return OpenLoopResult(
            checkpoint=agent.spec.name,
            frames=len(indices),
            mae_per_dim=mae.tolist(),
            zero_mae_per_dim=zero_mae.tolist(),
            gripper_accuracy=float(signs[valid].mean()),
            mae_by_horizon=by_horizon.tolist(),
        )
