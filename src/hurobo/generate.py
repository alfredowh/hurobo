"""Robot demonstrations from human demos: retarget, roll out in the scaled scene, keep the successes."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from hurobo.demo import HumanDemo
from hurobo.recorder import LeRobotRecorder, vector_feature, video_feature
from hurobo.retarget import (
    ObjectCentricRetargeter,
    PlanExecutor,
    RetargetConfig,
    RolloutResult,
    RolloutStep,
)
from hurobo.scaled_native import ScaledNativeScene, SimLayout, Workspace, table_to_world_map
from hurobo.teleop import TableToWorld
from hurobo.utils.signal import speed_percentile
from hurobo.utils.video import VideoWriter, caption_frame

STATE_NAMES = ["eef_x", "eef_y", "eef_z", "eef_aa_x", "eef_aa_y", "eef_aa_z", "gripper_q1", "gripper_q2"]
ACTION_NAMES = ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]
FPS = 20  # LIBERO's control rate, and `lerobot-eval`'s


def sim_features(image_size: int = 256) -> dict[str, dict[str, Any]]:
    """Feature spec of the generated dataset, with the names and shapes lerobot uses for LIBERO.

    Args:
        image_size: side of the square agent-view and wrist images.
    """
    return {
        "observation.images.image": video_feature(image_size, image_size),
        "observation.images.image2": video_feature(image_size, image_size),
        "observation.state": vector_feature(STATE_NAMES),
        "action": vector_feature(ACTION_NAMES),
    }


def quat_xyzw_to_axis_angle(quat_xyzw: np.ndarray) -> np.ndarray:
    """Axis-angle vector of a quaternion, computed exactly as lerobot's LIBERO observation processor does.

    The formula (angle = 2 acos(w), no wrap to [0, pi]) is kept on purpose: a different but equivalent
    vector would put the state in a different place than the one policies are evaluated with.

    Args:
        quat_xyzw: shape (4,).

    Returns:
        shape (3,).
    """
    w = float(np.clip(quat_xyzw[3], -1.0, 1.0))
    den = np.sqrt(max(1.0 - w * w, 0.0))
    if den <= 1e-10:
        return np.zeros(3)
    return np.asarray(quat_xyzw[:3]) / den * (2.0 * np.arccos(w))


def frame_from_obs(obs: dict, action: np.ndarray) -> dict[str, Any]:
    """One dataset frame from an env observation and the action taken in it.

    Images are turned by 180 degrees, as `lerobot-eval` does before a policy sees them, so a policy trained
    on this data gets the same pixels at evaluation.

    Args:
        obs: LeRobot-style observation (leading n_envs = 1 axis).
        action: shape (7,).
    """
    robot_state = obs["robot_state"]
    state = np.concatenate(
        [
            np.asarray(robot_state["eef"]["pos"][0]),
            quat_xyzw_to_axis_angle(np.asarray(robot_state["eef"]["quat"][0])),
            np.asarray(robot_state["gripper"]["qpos"][0]),
        ]
    ).astype(np.float32)
    return {
        "observation.images.image": np.ascontiguousarray(obs["pixels"]["image"][0][::-1, ::-1]),
        "observation.images.image2": np.ascontiguousarray(obs["pixels"]["image2"][0][::-1, ::-1]),
        "observation.state": state,
        "action": np.asarray(action, dtype=np.float32),
    }


@dataclass
class GenerationStats:
    """What a generation run did.

    Attributes:
        attempted: rollouts started.
        succeeded: rollouts that the LIBERO success check accepted (and were written).
        failures: failure reason -> count.
        steps: control steps of every written episode.
    """

    attempted: int = 0
    succeeded: int = 0
    failures: Counter[str] = field(default_factory=Counter)
    steps: list[int] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Share of rollouts that succeeded."""
        return self.succeeded / self.attempted if self.attempted else 0.0


class DemoGenerator:
    """Rolls human demos out on random layouts in a `ScaledNativeScene` and writes the successes."""

    def __init__(
        self,
        scene: ScaledNativeScene,
        demos: list[HumanDemo],
        mapping_workspace: Workspace | None = None,
        config: RetargetConfig | None = None,
        seed: int = 0,
    ) -> None:
        """Wire up the scene, the demos and the retargeter.

        Args:
            scene: the scaled-native scene (any state; `reset` is called per rollout).
            demos: processed human demos with grasp and release found.
            mapping_workspace: workspace the table-to-world map is built for; the scene's own by default.
            config: retargeting and execution settings.
            seed: for the choice of demo and layout.

        Raises:
            ValueError: if no demo has both a grasp and a release.
        """
        usable = [d for d in demos if d.grasp_frame_idx is not None and d.release_frame_idx is not None]
        if not usable:
            raise ValueError("no demo has a detected grasp and release")
        self._scene = scene
        self._demos = usable
        self._config = config or RetargetConfig()
        self._rng = np.random.default_rng(seed)
        self._workspace = mapping_workspace or scene.workspace
        self._mapping = table_to_world_map(scene.setup, self._workspace)

    @property
    def mapping(self) -> TableToWorld:
        """The table-to-world map in use."""
        return self._mapping

    def _retargeter(self) -> ObjectCentricRetargeter:
        return ObjectCentricRetargeter(
            self._mapping, self._scene.setup.bowl.radius_m, self._scene.plate_height_m, self._config
        )

    def slowdown_to_robot(
        self, demo: HumanDemo, eef_path: np.ndarray, percentile: float = 99.0
    ) -> tuple[float, float]:
        """How much slower than recorded the human demo must run to stay within the robot's speed.

        The robot's limit is measured, not assumed: a high percentile of the end-effector speed in a
        rollout. The human's speed is compared with it over the part of the demo the robot replays.

        Args:
            demo: the human demo.
            eef_path: shape (N, 3), end-effector positions of a rollout, one per control step.
            percentile: speed percentile taken as the robot's limit.

        Returns:
            (factor, robot_limit_m_s): factor >= 1, and the limit.
        """
        limit = speed_percentile(eef_path, np.arange(len(eef_path)) / FPS, percentile)
        start, end = self._retargeter().replayed_sample_range(demo)
        return demo.slowdown_for(limit, start, end), limit

    def replayed_sample_range(self, demo: HumanDemo) -> tuple[int, int]:
        """The samples (first, last) of the human motion the robot replays (see the retargeter)."""
        return self._retargeter().replayed_sample_range(demo)

    def layout_of_demo(self, demo: HumanDemo, clamp_to_workspace: bool = False) -> SimLayout:
        """The sim layout that is the human's own layout of that clip (the 'twin' of the recording).

        Args:
            demo: the human demo.
            clamp_to_workspace: snap bowl and plate into the scene's work area (for layouts that miss it
                by a few mm or cm) instead of returning positions the arm may not reach.
        """
        bowl = self._mapping.point_to_world(np.asarray(demo.t_table_bowl)[:3, 3])[:2]
        plate = self._mapping.point_to_world(np.asarray(demo.t_table_plate)[:3, 3])[:2]
        if clamp_to_workspace:
            bowl, plate = self._scene.workspace.clamp(bowl), self._scene.workspace.clamp(plate)
        return SimLayout(bowl_xy=bowl, plate_xy=plate)

    def rollout(
        self, demo: HumanDemo, layout: SimLayout | None = None
    ) -> tuple[Iterator[RolloutStep], PlanExecutor]:
        """Reset the scene on `layout` (sampled if None) and return the step iterator to consume.

        Args:
            demo: the human demo to retarget.
            layout: where the objects go.

        Returns:
            (steps, executor): iterate `steps` to run the rollout; `executor.result` is complete once the
            iterator is exhausted.
        """
        layout = layout or self._scene.sampler.sample()
        obs = self._scene.reset(layout)
        plan = self._retargeter().plan(demo, layout)
        executor = PlanExecutor(self._scene, self._config)
        return executor.run(obs, plan), executor

    def rollout_with_video(
        self, demo: HumanDemo, video_path: Path, layout: SimLayout | None = None, size: int = 480
    ) -> tuple[RolloutResult, np.ndarray]:
        """Run one rollout and write an agent-view video with the phase as a caption.

        Args:
            demo: the human demo to retarget.
            video_path: output .mp4 (20 fps).
            layout: where the objects go; sampled if None.
            size: video side in pixels.

        Returns:
            (result, eef_path): the outcome, and the end-effector world positions, shape (N, 3), one per step.
        """
        steps, executor = self.rollout(demo, layout)
        eef_path = []
        with VideoWriter(video_path, float(FPS), (size, size)) as writer:
            for step in steps:
                image = self._scene.task.render_camera("agentview", size, size)
                writer.write(caption_frame(np.ascontiguousarray(image), step.phase))
                eef_path.append(np.asarray(step.obs["robot_state"]["eef"]["pos"][0]))
        return executor.result, np.asarray(eef_path)

    def replay_on_own_layout(
        self, demo: HumanDemo, video_path: Path, size: int = 480, clamp_to_workspace: bool = True
    ) -> tuple[RolloutResult, SimLayout, np.ndarray]:
        """Replay a demo on the bowl and plate positions the human recorded, with a video.

        Args:
            demo: the human demo.
            video_path: output .mp4 (20 fps).
            size: video side in pixels.
            clamp_to_workspace: snap the layout into the work area (see `layout_of_demo`).

        Returns:
            (result, layout, eef_path): the outcome, the layout that was used and the end-effector world
            positions, shape (N, 3), one per step.
        """
        layout = self.layout_of_demo(demo, clamp_to_workspace)
        result, eef_path = self.rollout_with_video(demo, video_path, layout, size)
        return result, layout, eef_path

    def generate(
        self, count: int, recorder: LeRobotRecorder, task: str, max_attempts: int | None = None
    ) -> GenerationStats:
        """Roll out until `count` successful episodes are written.

        Args:
            count: successful episodes wanted.
            recorder: destination dataset (features from `sim_features`).
            task: instruction stored with each episode.
            max_attempts: stop after this many rollouts; default 4 x `count`.

        Returns:
            What happened, including why failed rollouts failed.
        """
        stats = GenerationStats()
        limit = max_attempts or 4 * count
        while stats.succeeded < count and stats.attempted < limit:
            demo = self._demos[int(self._rng.integers(len(self._demos)))]
            steps, executor = self.rollout(demo)
            frames = [frame_from_obs(step.obs, step.action) for step in steps]
            result = executor.result
            stats.attempted += 1
            if result.success and frames:
                recorder.add_episode(frames, task)
                stats.succeeded += 1
                stats.steps.append(len(frames))
            else:
                stats.failures[result.failure or "no success"] += 1
        return stats
