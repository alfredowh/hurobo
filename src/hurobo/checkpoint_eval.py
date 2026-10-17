"""Evaluate every checkpoint of one training run and tabulate success, open-loop error and loss per step."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from hurobo.benchmark import OpenLoopResult, load_episode_rows, summarize_rows
from hurobo.policy_rollout import CheckpointSpec
from hurobo.training import CAMERA_RENAME

OPEN_LOOP_STEPS_FILE = "open_loop_by_step.jsonl"
_LOG_LINE = re.compile(r"step:\S+\s.*?loss:(?P<loss>[\d.eE+-]+)\s.*?lr:(?P<lr>[\d.eE+-]+)")


@dataclass(frozen=True)
class RunCheckpoint:
    """One saved checkpoint of a `lerobot-train` run.

    Attributes:
        step: optimiser step at which it was saved.
        path: its `pretrained_model` folder.
    """

    step: int
    path: Path

    @property
    def name(self) -> str:
        """Label used in result files; zero-padded so it sorts by step."""
        return checkpoint_name(self.step)

    def spec(self) -> CheckpointSpec:
        """The `CheckpointSpec` for the policy loader (adds the camera mapping of our fine-tunes)."""
        return CheckpointSpec(self.name, str(self.path), rename_map=dict(CAMERA_RENAME))


def checkpoint_name(step: int) -> str:
    """Result-file label of the checkpoint saved at `step`."""
    return f"step{step:06d}"


def step_of_name(name: str) -> int:
    """Inverse of `checkpoint_name`."""
    return int(name.removeprefix("step"))


def discover_checkpoints(run_dir: Path) -> list[RunCheckpoint]:
    """Complete checkpoints of a run, ordered by step.

    Args:
        run_dir: the `--out` folder of the training run (holds `checkpoints/<step>/pretrained_model`).

    Returns:
        Checkpoints whose folder has a `model.safetensors`; the `last` symlink is skipped.
    """
    root = run_dir / "checkpoints"
    if not root.exists():
        return []
    found = [
        RunCheckpoint(int(folder.name), folder / "pretrained_model")
        for folder in root.iterdir()
        if folder.name.isdigit() and (folder / "pretrained_model" / "model.safetensors").exists()
    ]
    return sorted(found, key=lambda checkpoint: checkpoint.step)


def parse_training_log(log_path: Path, log_freq: int) -> pd.DataFrame:
    """Loss and learning rate per logged step from a `lerobot-train` log.

    lerobot prints the step abbreviated ("2K" for 1500), so the step is rebuilt from the position of the line:
    line i is step `(i + 1) * log_freq`. This holds for a run that started at step 0 and was not resumed.

    Args:
        log_path: text file with lerobot's `step:200 ... loss:0.123 ... lr:1.0e-04` lines.
        log_freq: the `--log_freq` the run used.

    Returns:
        Columns step, loss, lr (empty if the file is missing or has no such lines).
    """
    if not log_path.exists():
        return pd.DataFrame(columns=["step", "loss", "lr"])
    matches = [
        _LOG_LINE.search(line)
        for line in log_path.read_text(errors="ignore").replace("\r", "\n").splitlines()
    ]
    records = [
        {"step": (index + 1) * log_freq, "loss": float(match["loss"]), "lr": float(match["lr"])}
        for index, match in enumerate(m for m in matches if m)
    ]
    return pd.DataFrame(records, columns=["step", "loss", "lr"])


def load_open_loop(results_dir: Path) -> pd.DataFrame:
    """Open-loop results stored by `append_open_loop`, one row per checkpoint.

    Returns:
        Columns: step, frames, gripper_accuracy, mae_pos (mean of dims 0-2), mae_rot (3-5), mae_gripper,
        zero_mae_pos (the do-nothing chunk's error on the same steps), mae_all.
    """
    path = results_dir / OPEN_LOOP_STEPS_FILE
    if not path.exists():
        return pd.DataFrame()
    records = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        mae, zero = np.array(row["mae_per_dim"]), np.array(row["zero_mae_per_dim"])
        records.append(
            {
                "step": step_of_name(row["checkpoint"]),
                "frames": row["frames"],
                "gripper_accuracy": row["gripper_accuracy"],
                "mae_pos": float(mae[:3].mean()),
                "mae_rot": float(mae[3:6].mean()),
                "mae_gripper": float(mae[6]),
                "zero_mae_pos": float(zero[:3].mean()),
                "mae_all": float(mae.mean()),
            }
        )
    return pd.DataFrame(records).drop_duplicates("step", keep="last").sort_values("step")


def append_open_loop(results_dir: Path, result: OpenLoopResult) -> None:
    """Append one checkpoint's open-loop result to the JSONL (re-running a checkpoint replaces it on load)."""
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / OPEN_LOOP_STEPS_FILE).open("a") as handle:
        handle.write(json.dumps(asdict(result)) + "\n")


def open_loop_done(results_dir: Path) -> set[int]:
    """Steps that already have an open-loop result."""
    table = load_open_loop(results_dir)
    return set(table["step"]) if len(table) else set()


def closed_loop_table(results_dir: Path, condition: str = "scaled") -> pd.DataFrame:
    """Closed-loop summary per checkpoint step for one scene condition.

    Returns:
        `summarize_rows` columns plus step, failure-reason shares (`bowl_lifted`: bowl rose above 3 cm but the
        episode failed; `no_grasp`: end effector never came within 6 cm of the bowl centre) and
        mean_steps_all (steps over all episodes, a failed episode counts its full length).
    """
    rows = [row for row in load_episode_rows(results_dir) if row["condition"] == condition]
    summary = summarize_rows(rows)
    if summary.empty:
        return summary
    frame = pd.DataFrame(rows)
    extra = []
    for checkpoint, group in frame.groupby("checkpoint"):
        failed = group[~group["success"]]
        extra.append(
            {
                "checkpoint": checkpoint,
                "bowl_lifted_failed": int((failed["max_bowl_lift_m"] > 0.03).sum()),
                "never_near_bowl_failed": int((failed["min_eef_bowl_m"] > 0.06).sum()),
                "mean_steps_all": float(group["steps"].mean()),
            }
        )
    merged = summary.merge(pd.DataFrame(extra), on="checkpoint")
    merged.insert(0, "step", merged["checkpoint"].map(step_of_name))
    return merged.sort_values("step").reset_index(drop=True)


def checkpoint_table(results_dir: Path, condition: str = "scaled") -> pd.DataFrame:
    """Closed-loop and open-loop numbers side by side, one row per checkpoint step."""
    closed = closed_loop_table(results_dir, condition)
    opened = load_open_loop(results_dir)
    if closed.empty and opened.empty:
        return pd.DataFrame()
    if closed.empty:
        return opened.reset_index(drop=True)
    if opened.empty:
        return closed
    return closed.merge(opened, on="step", how="outer").sort_values("step").reset_index(drop=True)


class CheckpointEvaluator:
    """Evaluates the not yet evaluated checkpoints of a run, open loop then closed loop, resumably."""

    def __init__(
        self,
        run_dir: Path,
        results_dir: Path,
        dataset_root: Path,
        setup,  # RealSetup; untyped here to keep the simulator import out of the table helpers
        condition: str = "scaled",
        replan_every: int = 10,
        episodes: int = 10,
        max_steps: int = 400,
        open_loop_frames: int = 100,
        videos: int = 1,
    ) -> None:
        """Configure the evaluation.

        Args:
            run_dir: training output folder with `checkpoints/<step>/pretrained_model`.
            results_dir: where the JSONL files and videos go; created.
            dataset_root: generated dataset used for the open-loop error (the training data).
            setup: real measurements for the scene (`RealSetup`).
            condition: closed-loop scene condition (see `policy_rollout.CONDITIONS`).
            replan_every: actions executed per policy call.
            episodes: closed-loop episodes per checkpoint; the same layout seeds 0..n-1 for every checkpoint.
            max_steps: episode length limit; must cover the longest demonstration.
            open_loop_frames: frames sampled for the open-loop error (same seed for every checkpoint).
            videos: closed-loop episodes per checkpoint saved as mp4.
        """
        self._run_dir = run_dir
        self._results_dir = results_dir
        self._dataset_root = dataset_root
        self._setup = setup
        self._condition = condition
        self._replan_every = replan_every
        self._episodes = episodes
        self._max_steps = max_steps
        self._open_loop_frames = open_loop_frames
        self._videos = videos
        self._open_loop = None
        results_dir.mkdir(parents=True, exist_ok=True)

    def pending(self) -> list[RunCheckpoint]:
        """Checkpoints that still lack open-loop or closed-loop episodes."""
        done = open_loop_done(self._results_dir)
        counts = {}
        for row in load_episode_rows(self._results_dir):
            if row["condition"] == self._condition:
                counts[row["checkpoint"]] = counts.get(row["checkpoint"], 0) + 1
        return [
            checkpoint
            for checkpoint in discover_checkpoints(self._run_dir)
            if checkpoint.step not in done or counts.get(checkpoint.name, 0) < self._episodes
        ]

    def evaluate(self, checkpoint: RunCheckpoint) -> None:
        """Run the open-loop error, then the closed-loop episodes, of one checkpoint."""
        from hurobo.benchmark import ClosedLoopBenchmark, OpenLoopEvaluator
        from hurobo.policy_rollout import SmolVLAAgent

        spec = checkpoint.spec()
        if checkpoint.step not in open_loop_done(self._results_dir):
            if self._open_loop is None:
                self._open_loop = OpenLoopEvaluator(self._dataset_root, "local/open_loop")
            result = self._open_loop.evaluate(SmolVLAAgent(spec), frames=self._open_loop_frames)
            append_open_loop(self._results_dir, result)
            print(f"{spec.name}: open loop done (gripper accuracy {result.gripper_accuracy:.2f})", flush=True)
        benchmark = ClosedLoopBenchmark(
            self._setup, self._results_dir, video_episodes=self._videos, max_steps=self._max_steps
        )
        benchmark.run([spec], [self._condition], [self._replan_every], self._episodes)


def display_table(table: pd.DataFrame) -> pd.DataFrame:
    """Readable per-step table: success as "k/n (low-high)", failure reasons and open-loop numbers.

    Args:
        table: from `checkpoint_table`.
    """
    if table.empty:
        return table
    out = pd.DataFrame({"step": table["step"]})
    if "successes" in table:
        has = table["episodes"].notna()
        out["success (95 % Wilson)"] = [
            f"{int(k)}/{int(n)} ({lo:.2f}-{hi:.2f})" if ok else "not run"
            for ok, k, n, lo, hi in zip(
                has,
                table["successes"].fillna(0),
                table["episodes"].fillna(0),
                table["ci_low"],
                table["ci_high"],
                strict=True,
            )
        ]
        out["failed: bowl lifted"] = table["bowl_lifted_failed"]
        out["failed: never within 6 cm"] = table["never_near_bowl_failed"]
        out["mean steps (success)"] = table["mean_steps_success"].round(0)
        out["median closest eef-bowl (cm)"] = table["median_min_eef_bowl_cm"].round(1)
    if "mae_pos" in table:
        out["open-loop frames"] = table["frames"]
        out["pos MAE"] = table["mae_pos"].round(3)
        out["do-nothing pos MAE"] = table["zero_mae_pos"].round(3)
        out["gripper sign acc."] = table["gripper_accuracy"].round(2)
    return out


def sample_size_note(table: pd.DataFrame, enough: int = 30) -> str:
    """One plain sentence about how little the closed-loop numbers can show.

    Args:
        table: from `checkpoint_table`.
        enough: episodes per checkpoint below which differences are called noise.
    """
    if table.empty or "episodes" not in table or table["episodes"].dropna().empty:
        return "No closed-loop episodes stored yet."
    n_min, n_max = int(table["episodes"].min()), int(table["episodes"].max())
    span = f"{n_min}" if n_min == n_max else f"{n_min}-{n_max}"
    verdict = (
        "Differences between checkpoints are within noise unless the Wilson intervals do not overlap."
        if n_max < enough
        else "Intervals are still wide; read them, not the point estimates."
    )
    return (
        f"Each closed-loop point is {span} episodes per checkpoint "
        f"(same layout seeds for every checkpoint). {verdict}"
    )
