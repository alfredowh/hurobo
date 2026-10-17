"""Compare fine-tuned and pretrained SmolVLA policies on the same scaled scene, on unseen layouts."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from hurobo.benchmark import (
    EPISODES_FILE,
    OpenLoopEvaluator,
    load_episode_rows,
    summarize_rows,
)
from hurobo.checkpoint_eval import RunCheckpoint, checkpoint_table, discover_checkpoints
from hurobo.policy_rollout import CheckpointSpec

#: Layout seeds of the held-out episodes. The per-step evaluation of a run uses seeds 0..9, so the checkpoint
#: chosen there is not also judged on the layouts it was chosen on.
HELD_OUT_FIRST_EPISODE = 100


def best_checkpoint(run_dir: Path, results_dir: Path, condition: str = "scaled") -> RunCheckpoint:
    """The checkpoint of a run with the highest closed-loop success; ties go to the lower open-loop error.

    Args:
        run_dir: training output folder with the checkpoints.
        results_dir: the run's per-step evaluation (`CheckpointEvaluator`).
        condition: scene condition whose success rate decides.

    Returns:
        The checkpoint. Ties on both go to the later step.

    Raises:
        ValueError: if no evaluated checkpoint of the run exists.
    """
    table = checkpoint_table(results_dir, condition)
    if table.empty or "success_rate" not in table:
        raise ValueError(f"no closed-loop results for {run_dir.name} in {results_dir}")
    error = table["mae_all"] if "mae_all" in table else pd.Series(0.0, index=table.index)
    ranked = table.assign(_error=error.fillna(np.inf)).sort_values(
        ["success_rate", "_error", "step"], ascending=[False, True, False]
    )
    best_step = int(ranked.iloc[0]["step"])
    for checkpoint in discover_checkpoints(run_dir):
        if checkpoint.step == best_step:
            return checkpoint
    raise ValueError(f"checkpoint {best_step} is in the results but not in {run_dir}")


def fine_tuned_contender(
    label: str, run_dir: Path, results_dir: Path, final_step: int, condition: str = "scaled"
) -> CheckpointSpec:
    """The policy to enter for a training run: its best checkpoint, once the run has finished.

    Args:
        label: name of the run in the results, e.g. "ours, smolvla_base init"; the step is appended.
        run_dir: training output folder.
        results_dir: the run's per-step evaluation.
        final_step: the last step the run is to reach; a run that has not saved it is not entered, so a
            half-trained run is never compared.
        condition: scene condition whose success rate picks the checkpoint.

    Returns:
        The checkpoint as a `CheckpointSpec` (with our camera mapping) named "<label> (step N)".

    Raises:
        ValueError: if the run has not reached `final_step` or has no per-step results yet.
    """
    if final_step not in {checkpoint.step for checkpoint in discover_checkpoints(run_dir)}:
        raise ValueError(f"{run_dir.name} has not reached step {final_step} yet")
    best = best_checkpoint(run_dir, results_dir, condition)
    return replace(best.spec(), name=f"{label} (step {best.step})")


class ModelComparison:
    """Runs several policies on the same held-out layouts of one scene condition; resumable.

    Every policy sees the same layouts (the layout seed is the episode number), so the success counts are
    paired. One JSON line per episode goes into `<out_dir>/closed_loop_episodes.jsonl` in the format of
    `ClosedLoopBenchmark`, so `summarize_rows` and the other benchmark tables read it.
    """

    def __init__(
        self,
        setup,  # RealSetup; untyped to keep the simulator import out of the table helpers
        out_dir: Path,
        condition: str = "scaled",
        replan_every: int = 10,
        max_steps: int = 500,
        first_episode: int = HELD_OUT_FIRST_EPISODE,
        videos: int = 1,
    ) -> None:
        """Configure the comparison.

        Args:
            setup: real measurements for the scene (`RealSetup`).
            out_dir: results folder; created.
            condition: scene condition (see `policy_rollout.CONDITIONS`); `scaled` has the real-size objects.
            replan_every: actions executed per policy call, the same for every policy.
            max_steps: episode length limit, the same for every policy (the generated demonstrations are
                longer than LIBERO's 300).
            first_episode: layout seed of the first episode.
            videos: episodes per policy saved as mp4.
        """
        self._setup = setup
        self._out_dir = out_dir
        self._condition = condition
        self._replan_every = replan_every
        self._max_steps = max_steps
        self._first_episode = first_episode
        self._videos = videos
        out_dir.mkdir(parents=True, exist_ok=True)

    def run(self, specs: list[CheckpointSpec], episodes: int) -> pd.DataFrame:
        """Run the missing episodes of every policy.

        Args:
            specs: the policies, each with a unique `name`.
            episodes: held-out layouts per policy.

        Returns:
            The summary table (`summary`).
        """
        from hurobo.policy_rollout import ClosedLoopRunner, EvalScene, SmolVLAAgent

        done = {(row["checkpoint"], row["episode"]) for row in load_episode_rows(self._out_dir)}
        scene = EvalScene(self._setup)
        runner = ClosedLoopRunner(scene, max_steps=self._max_steps)
        try:
            for spec in specs:
                agent = SmolVLAAgent(spec, replan_every=self._replan_every)
                for episode in range(self._first_episode, self._first_episode + episodes):
                    if (spec.name, episode) in done:
                        continue
                    video = None
                    if episode < self._first_episode + self._videos:
                        videos = self._out_dir / "videos"
                        videos.mkdir(exist_ok=True)
                        video = videos / f"{spec.name.replace('/', '_')}__ep{episode}.mp4"
                    result = runner.run_episode(agent, self._condition, episode, video_path=video)
                    row = {
                        "cell": f"{spec.name}|{self._condition}|{self._replan_every}",
                        "checkpoint": spec.name,
                        "replan_every": self._replan_every,
                        **result.as_dict(),
                    }
                    with (self._out_dir / EPISODES_FILE).open("a") as handle:
                        handle.write(json.dumps(row) + "\n")
                    print(f"{spec.name} ep{episode}: {'SUCCESS' if result.success else 'fail'}", flush=True)
        finally:
            scene.close()
        return self.summary()

    def summary(self) -> pd.DataFrame:
        """One row per policy: success rate with its Wilson interval and the failure-stage statistics."""
        return summarize_rows(load_episode_rows(self._out_dir))

    def open_loop(self, specs: list[CheckpointSpec], dataset_root: Path, frames: int = 100) -> pd.DataFrame:
        """Offline action-chunk error of each policy against a demonstration dataset.

        Args:
            specs: the policies.
            dataset_root: the generated dataset (the fine-tuned policies were trained on it).
            frames: dataset frames sampled, the same ones for every policy.

        Returns:
            One row per policy: gripper accuracy and mean absolute error of position, rotation and gripper
            dimensions, next to the error of a do-nothing chunk.
        """
        from hurobo.policy_rollout import SmolVLAAgent

        evaluator = OpenLoopEvaluator(dataset_root, "local/open_loop")
        records = []
        for spec in specs:
            result = asdict(evaluator.evaluate(SmolVLAAgent(spec), frames=frames))
            mae, zero = np.array(result["mae_per_dim"]), np.array(result["zero_mae_per_dim"])
            records.append(
                {
                    "policy": spec.name,
                    "gripper_accuracy": result["gripper_accuracy"],
                    "mae_pos": float(mae[:3].mean()),
                    "mae_rot": float(mae[3:6].mean()),
                    "mae_gripper": float(mae[6]),
                    "do_nothing_mae_pos": float(zero[:3].mean()),
                }
            )
        return pd.DataFrame(records)
