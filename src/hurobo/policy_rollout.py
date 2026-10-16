"""Run a pretrained or fine-tuned SmolVLA closed-loop in LIBERO: stock scene, cleared scene, scaled scene."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hurobo.generate import FPS
from hurobo.real_setup import RealSetup
from hurobo.scaled_native import (
    BOWL_JOINT,
    PLATE_JOINT,
    SUITE_NAME,
    TASK_ID,
    LayoutSampler,
    ScaledNativeScene,
    SimLayout,
)
from hurobo.sim import LiberoTask
from hurobo.utils.video import VideoWriter, caption_frame

#: lerobot's `TASK_SUITE_MAX_STEPS["libero_goal"]`: the longest training demo has 270 steps.
MAX_STEPS = 300

CONDITIONS = ("stock", "clean", "scaled", "clean_stock_layout", "scaled_stock_layout")
CONDITION_HELP = {
    "stock": "LIBERO's own scene and init states, untouched (reference for the checkpoint)",
    "clean": "LIBERO's bowl/plate sizes, other objects removed, our random layouts",
    "scaled": "bowl and plate at the measured real sizes, other objects removed, our random layouts (target)",
    "clean_stock_layout": "as clean, but bowl and plate start where LIBERO's init state puts them",
    "scaled_stock_layout": "as scaled, but bowl and plate start where LIBERO's init state puts them "
    "(the real-size plate may overlap the bowl there: LIBERO's spacing was made for the small plate)",
}


@dataclass(frozen=True)
class CheckpointSpec:
    """A policy checkpoint and how its camera names map onto LIBERO's.

    Attributes:
        name: short label used in result tables.
        path: Hub repo id or local checkpoint directory.
        rename_map: observation key -> key the checkpoint was trained with (empty if they already agree).
        likes: Hub likes when the registry entry was made (popularity evidence, informational only).
        note: what is known about the training data.
    """

    name: str
    path: str
    rename_map: dict[str, str] = field(default_factory=dict)
    likes: int | None = None
    note: str = ""


#: The two most-liked SmolVLA LIBERO checkpoints on the Hub (queried 2026-10-16, sorted by downloads).
KNOWN_CHECKPOINTS: dict[str, CheckpointSpec] = {
    "hfvla": CheckpointSpec(
        "HuggingFaceVLA/smolvla_libero",
        "HuggingFaceVLA/smolvla_libero",
        likes=20,
        note="41.7k downloads/month; image/image2 keys; training data not stated on the card",
    ),
    "lerobot": CheckpointSpec(
        "lerobot/smolvla_libero",
        "lerobot/smolvla_libero",
        rename_map={
            "observation.images.image": "observation.images.camera1",
            "observation.images.image2": "observation.images.camera2",
        },
        likes=7,
        note="trained on lerobot/libero (all four suites), 25k steps, batch 32; camera1/camera2 keys",
    ),
}


class SmolVLAAgent:
    """A SmolVLA checkpoint with the same pre/post-processing `lerobot-eval` applies in LIBERO."""

    def __init__(
        self,
        spec: CheckpointSpec,
        replan_every: int | None = None,
        device: str = "cuda",
    ) -> None:
        """Load the policy and its processors.

        Args:
            spec: which checkpoint.
            replan_every: how many actions of a predicted chunk are executed before the policy looks at the
                scene again (SmolVLA's `n_action_steps`). The checkpoint's own value if None; at most the
                chunk size.
            device: torch device.
        """
        from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
        from lerobot.policies import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        self.spec = spec
        policy = SmolVLAPolicy.from_pretrained(spec.path)
        if replan_every is not None:
            policy.config.n_action_steps = min(replan_every, policy.config.chunk_size)
        policy.config.device = device
        self._policy = policy.to(device).eval()
        self._preprocessor, self._postprocessor = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=spec.path,
            preprocessor_overrides={
                "device_processor": {"device": device},
                "rename_observations_processor": {"rename_map": spec.rename_map},
            },
        )
        # The LIBERO step turns the env's raw observation into the dataset's layout (180-degree image
        # turn, state = eef position + axis-angle + gripper), exactly as lerobot-eval does.
        self._env_preprocessor, _ = LiberoEnvConfig(task=SUITE_NAME).get_env_processors()

    @property
    def replan_every(self) -> int:
        """Actions executed per policy call."""
        return int(self._policy.config.n_action_steps)

    def reset(self) -> None:
        """Forget the queued actions; call at the start of every episode."""
        self._policy.reset()

    @property
    def will_query(self) -> bool:
        """True if the next `act` runs the network (the action queue is empty)."""
        return len(self._policy._queues["action"]) == 0

    @torch.inference_mode()
    def act(self, obs: dict[str, Any], task: str) -> np.ndarray:
        """Next action for an environment observation.

        Args:
            obs: observation as returned by `LiberoTask.reset` / `step` (leading n_envs = 1 axis).
            task: language instruction.

        Returns:
            shape (7,), the env action (3 position, 3 rotation, 1 gripper).
        """
        from lerobot.envs import preprocess_observation

        batch = preprocess_observation(obs)
        batch["task"] = [task]
        batch = self._env_preprocessor(batch)
        batch = self._preprocessor(batch)
        action = self._postprocessor(self._policy.select_action(batch))
        return action.to("cpu").numpy()[0]

    def predict_chunk(self, batch: dict[str, Any]) -> np.ndarray:
        """Action chunk for an already env-processed batch (a dataset frame), for open-loop evaluation.

        Args:
            batch: LeRobotDataset items collated to a batch, with a "task" list.

        Returns:
            shape (B, chunk_size, 7), in env units.
        """
        with torch.inference_mode():
            self._policy.reset()
            batch = self._preprocessor({k: v for k, v in batch.items() if k != "action"})
            chunk = self._policy.predict_action_chunk(batch)
            return self._postprocessor(chunk).to("cpu").numpy()


@dataclass
class EpisodeResult:
    """Outcome of one closed-loop episode.

    Attributes:
        condition: scene condition (see `CONDITIONS`).
        episode: episode index (also the layout / init-state seed).
        success: LIBERO's success check fired.
        steps: control steps taken.
        min_eef_bowl_m: closest the end effector came to the bowl centre.
        max_bowl_lift_m: highest the bowl rose above its start.
        final_bowl_plate_xy_m: horizontal bowl-plate centre distance at the end.
        policy_calls: number of policy forward passes.
        start_bowl_plate_xy_m: horizontal bowl-plate centre distance after the scene settled, before the first
            action (shows starts where overlapping objects pushed each other apart).
    """

    condition: str
    episode: int
    success: bool
    steps: int
    min_eef_bowl_m: float
    max_bowl_lift_m: float
    final_bowl_plate_xy_m: float
    policy_calls: int
    start_bowl_plate_xy_m: float = float("nan")

    def as_dict(self) -> dict[str, Any]:
        """Plain dict for JSON."""
        return asdict(self)


class EvalScene:
    """One LIBERO task that can be reset into any of the three benchmark conditions."""

    def __init__(self, setup: RealSetup) -> None:
        """Build the task once; conditions only differ in how `reset` edits it.

        Args:
            setup: the real measurements the `scaled` condition is built from.
        """
        self.task = LiberoTask(SUITE_NAME, TASK_ID)
        self._scaled = ScaledNativeScene(setup, task=self.task)
        self._setup = setup

    def reset(self, condition: str, episode: int) -> dict:
        """Reset into a condition; the same (condition, episode) always gives the same start.

        Args:
            condition: one of `CONDITIONS`.
            episode: index; selects LIBERO's init state (`stock`) or the layout seed (others).

        Returns:
            the first observation.
        """
        if condition == "stock":
            self.task.select_init_state(episode)
            return self.task.reset()
        scaled = condition.startswith("scaled")
        if condition.endswith("_stock_layout"):
            return self._scaled.reset(self.stock_layout(episode), scaled=scaled)
        sampler = LayoutSampler(
            self._scaled.workspace, self._setup.bowl.radius_m, self._setup.plate.radius_m, seed=episode
        )
        return self._scaled.reset(sampler.sample(), scaled=scaled)

    def stock_layout(self, episode: int) -> SimLayout:
        """Bowl and plate centres (world xy) of LIBERO's init state `episode`.

        Only xy is taken: `ScaledNativeScene.place` sets both objects upright (yaw 0) at rest height.
        This resets the task, and `reset` advances LIBERO's init-state counter, so it is selected again.

        Args:
            episode: init state index (wraps around LIBERO's 50 states).
        """
        self.task.select_init_state(episode)
        self.task.reset()
        layout = SimLayout(bowl_xy=self.bowl_pos()[:2].copy(), plate_xy=self.plate_pos()[:2].copy())
        self.task.select_init_state(episode)
        return layout

    def bowl_pos(self) -> np.ndarray:
        """World position of the bowl, shape (3,)."""
        return self.task.get_object_pose(BOWL_JOINT).pos

    def plate_pos(self) -> np.ndarray:
        """World position of the plate, shape (3,)."""
        return self.task.get_object_pose(PLATE_JOINT).pos

    def close(self) -> None:
        """Release the simulator."""
        self.task.close()


class ClosedLoopRunner:
    """Plays a `SmolVLAAgent` in an `EvalScene` and records the outcome."""

    def __init__(self, scene: EvalScene, max_steps: int = MAX_STEPS, video_size: int = 256) -> None:
        """Hold the scene.

        Args:
            scene: where to run.
            max_steps: episode length limit.
            video_size: side of the optional episode video.
        """
        self._scene = scene
        self._max_steps = max_steps
        self._video_size = video_size

    def run_episode(
        self,
        agent: SmolVLAAgent,
        condition: str,
        episode: int,
        video_path: Path | None = None,
        on_step: Callable[[int, dict, np.ndarray], None] | None = None,
    ) -> EpisodeResult:
        """One episode; stops at the first success (LIBERO resets the scene after it).

        Args:
            agent: the policy.
            condition: one of `CONDITIONS`.
            episode: episode index.
            video_path: write an agent-view mp4 here if given.
            on_step: called with (step, observation, action) after every step.
        """
        scene = self._scene
        obs = scene.reset(condition, episode)
        task_text = scene.task.task_description
        agent.reset()
        bowl_start = scene.bowl_pos().copy()
        start_distance = float(np.linalg.norm(bowl_start[:2] - scene.plate_pos()[:2]))
        min_eef_bowl, max_lift, success, steps, calls = np.inf, 0.0, False, 0, 0
        video = VideoWriter(video_path, float(FPS), (self._video_size,) * 2) if video_path else nullcontext()
        with video as writer:
            while steps < self._max_steps and not success:
                calls += int(agent.will_query)
                action = agent.act(obs, task_text)
                obs, success, terminated = scene.task.step(action)
                steps += 1
                eef = np.asarray(obs["robot_state"]["eef"]["pos"][0])
                bowl = scene.bowl_pos()
                min_eef_bowl = min(min_eef_bowl, float(np.linalg.norm(eef - bowl)))
                max_lift = max(max_lift, float(bowl[2] - bowl_start[2]))
                if writer is not None:
                    rgb = scene.task.render_camera("agentview", self._video_size, self._video_size)
                    # VideoWriter takes BGR.
                    writer.write(
                        caption_frame(np.ascontiguousarray(rgb[..., ::-1]), f"{condition} #{episode}")
                    )
                if on_step is not None:
                    on_step(steps, obs, action)
                if terminated and not success:
                    break
        return EpisodeResult(
            condition=condition,
            episode=episode,
            success=bool(success),
            steps=steps,
            min_eef_bowl_m=min_eef_bowl,
            max_bowl_lift_m=max_lift,
            final_bowl_plate_xy_m=float(np.linalg.norm(scene.bowl_pos()[:2] - scene.plate_pos()[:2])),
            policy_calls=calls,
            start_bowl_plate_xy_m=start_distance,
        )
