"""MimicGen-style expansion of a few source demos: nearest-layout sources, per-subtask mixing, action noise.

What this adds on top of `generate.DemoGenerator` (which already does MimicGen's core loop: transform a source
demo to a new layout, execute it, keep the successes). Existing modules are only imported, never changed.

- `SourceSelector`: picks the source demo whose own layout is nearest to the new one (MimicGen's
  nearest-neighbour selection), separately for the approach and for the carry (its per-subtask selection).
- `MixedRetargeter`: builds one plan from the approach of one demo and the carry of another.
- `NoisyPlanExecutor`: Gaussian noise on the executed actions, which the closed loop then has to correct.
- `MimicGenExpander`: the loop, with a step-count filter and a per-source record of what failed.
- `StockBaseline`: the same layouts rolled out the way `scripts/04_generate_dataset.py` does, for comparison.

Frames: `p_*_world` is the sim world frame, `*_table` the real table frame; `TableToWorld` maps between them.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from hurobo.demo import HumanDemo
from hurobo.generate import DemoGenerator, frame_from_obs
from hurobo.recorder import LeRobotRecorder
from hurobo.retarget import (
    ObjectCentricRetargeter,
    PlanExecutor,
    RetargetConfig,
    RetargetPlan,
    RolloutStep,
)
from hurobo.scaled_native import LayoutSampler, ScaledNativeScene, SimLayout, table_to_world_map
from hurobo.teleop import TableToWorld

#: LIBERO's episode limit, and the one `policy_rollout.MAX_STEPS` evaluates with.
EVAL_STEP_LIMIT = 300


@dataclass(frozen=True)
class MimicGenConfig:
    """Settings of the expansion.

    Attributes:
        nn_k: pick the source uniformly among the `nn_k` demos nearest to the new layout (MimicGen uses 3 for
            its tasks with a nearest-neighbour rule); 0 picks uniformly among all demos, as `DemoGenerator`.
        per_subtask: choose the approach source and the carry source independently (MimicGen's per-subtask
            toggle); False uses one demo for both, chosen by the sum of the two distances.
        action_noise: standard deviation of the Gaussian noise added to the six pose dimensions of every
            action, in the controller's normalised units (MimicGen's simulation default is 0.05; 0 is off).
        noise_free_phases: phases that get no noise, because a nudge while the fingers close or open throws
            the bowl (our choice; MimicGen perturbs every step).
        max_episode_steps: a successful episode longer than this is dropped, since the evaluation allows
            `EVAL_STEP_LIMIT` steps and a policy cannot learn to finish in time from a longer demo.
        slowdown_range: when the retarget config is `paced`, the source demos are slowed by a factor drawn
            uniformly from this range (>= 1); ignored otherwise.
        retarget: how a demo becomes robot goals (passed to the retargeter and the executor).
        seed: for source choice, noise and speed.
    """

    nn_k: int = 3
    per_subtask: bool = True
    action_noise: float = 0.05
    noise_free_phases: tuple[str, ...] = ("grasp", "release")
    max_episode_steps: int = EVAL_STEP_LIMIT
    slowdown_range: tuple[float, float] = (1.0, 1.0)
    retarget: RetargetConfig = field(default_factory=RetargetConfig)
    seed: int = 0

    @classmethod
    def stock(cls, retarget: RetargetConfig | None = None, seed: int = 0) -> MimicGenConfig:
        """The settings under which this expander does what `DemoGenerator` does (no selection, no noise)."""
        return cls(
            nn_k=0,
            per_subtask=False,
            action_noise=0.0,
            max_episode_steps=10**9,
            retarget=retarget or RetargetConfig(),
            seed=seed,
        )


@dataclass(frozen=True)
class SourceChoice:
    """Indices (into the demo list) of the demo each subtask is taken from."""

    approach: int
    carry: int


class SourceSelector:
    """Chooses source demos for a new layout by how close their own layouts are.

    The distances are the ones the retargeter's distortion depends on: the approach is moved rigidly with the
    bowl (so what matters is the bowl's absolute place, which decides the reach limit and the side the hand
    came from); the carry is bent from the bowl to the plate (so what matters is the bowl-to-plate vector).
    """

    def __init__(
        self,
        source_layouts_world: list[SimLayout],
        nn_k: int,
        per_subtask: bool,
        rng: np.random.Generator,
    ) -> None:
        """Store the sources' layouts.

        Args:
            source_layouts_world: where each source demo's bowl and plate are, in the world frame.
            nn_k: pool size; 0 means uniform over all sources.
            per_subtask: choose approach and carry independently.
            rng: for the choice within the pool.
        """
        self._bowl_world = np.array([layout.bowl_xy for layout in source_layouts_world])
        self._bowl_to_plate_world = np.array(
            [layout.plate_xy - layout.bowl_xy for layout in source_layouts_world]
        )
        self._nn_k = nn_k
        self._per_subtask = per_subtask
        self._rng = rng
        self._allowed = np.ones(len(source_layouts_world), dtype=bool)

    def restrict(self, allowed_sources: list[int]) -> None:
        """Only choose among these sources from now on (e.g. the ones that passed screening).

        Raises:
            ValueError: if the list is empty.
        """
        if not allowed_sources:
            raise ValueError("at least one source must stay allowed")
        self._allowed = np.zeros(len(self._allowed), dtype=bool)
        self._allowed[allowed_sources] = True

    def approach_distances_m(self, layout: SimLayout) -> np.ndarray:
        """shape (n_sources,), distance of each source's bowl from the new bowl, world frame."""
        return np.linalg.norm(self._bowl_world - layout.bowl_xy, axis=1)

    def carry_distances_m(self, layout: SimLayout) -> np.ndarray:
        """shape (n_sources,), distance of each source's bowl-to-plate vector from the new one."""
        return np.linalg.norm(self._bowl_to_plate_world - (layout.plate_xy - layout.bowl_xy), axis=1)

    def choose(self, layout: SimLayout) -> SourceChoice:
        """Pick the sources for one rollout on `layout`."""
        approach_d, carry_d = self.approach_distances_m(layout), self.carry_distances_m(layout)
        if self._per_subtask:
            return SourceChoice(self._pick(approach_d), self._pick(carry_d))
        shared = self._pick(approach_d + carry_d)
        return SourceChoice(shared, shared)

    def _pick(self, distances_m: np.ndarray) -> int:
        allowed = np.flatnonzero(self._allowed)
        if self._nn_k <= 0 or self._nn_k >= len(allowed):
            return int(self._rng.choice(allowed))
        pool = allowed[np.argsort(distances_m[allowed])[: self._nn_k]]
        return int(self._rng.choice(pool))


class MixedRetargeter(ObjectCentricRetargeter):
    """`ObjectCentricRetargeter` that can take the approach and the carry from different demos.

    This works because the parent already bends every carry to start at the bowl's real resting place and end
    on the plate, whatever demo it came from, and anchors every approach on the bowl: the two parts meet at
    the grasp regardless of which demos they came from.
    """

    def plan_mixed(self, approach_demo: HumanDemo, carry_demo: HumanDemo, layout: SimLayout) -> RetargetPlan:
        """One plan: the approach and the finger yaw of `approach_demo`, the carry of `carry_demo`.

        Args:
            approach_demo: source of the TCP path to the grasp (and with it the side the rim is gripped from).
            carry_demo: source of the bowl's path from the grasp to the release.
            layout: where the bowl and plate are in the world.

        Returns:
            The combined plan. The two parts keep their own clocks (`*_times_s`), as the executor follows
            each phase's clock separately.
        """
        approach_plan = self.plan(approach_demo, layout)
        if carry_demo is approach_demo:
            return approach_plan
        carry_plan = self.plan(carry_demo, layout)
        return replace(
            approach_plan,
            bowl_path_world=carry_plan.bowl_path_world,
            bowl_path_times_s=carry_plan.bowl_path_times_s,
            release_z_m=carry_plan.release_z_m,
        )


class NoisyPlanExecutor(PlanExecutor):
    """`PlanExecutor` with Gaussian noise on the commanded pose deltas; the noisy action is what is recorded.

    The executor re-aims at its goal every step from the measured pose, so a perturbation is corrected on the
    next steps; the recorded data then contains small deviations and the corrections, which a noise-free
    scripted rollout never shows.
    """

    def __init__(
        self,
        scene: ScaledNativeScene,
        config: RetargetConfig,
        noise_std: float,
        rng: np.random.Generator,
        noise_free_phases: tuple[str, ...] = (),
    ) -> None:
        """Bind to a scene.

        Args:
            scene: the scene to act in.
            config: speed, tolerances and dwell times.
            noise_std: standard deviation of the noise on the six pose dimensions (normalised units).
            rng: noise source.
            noise_free_phases: phase names that are left unperturbed.
        """
        super().__init__(scene, config)
        self._noise_std = noise_std
        self._noise_rng = rng
        self._noise_free_phases = noise_free_phases

    def _send(self, action: np.ndarray, phase: str) -> Iterator[RolloutStep]:
        """Add the noise (the gripper dimension is never touched), clip to the action range, and step."""
        if self._noise_std > 0.0 and phase not in self._noise_free_phases:
            action = np.asarray(action, dtype=np.float32).copy()
            action[:6] = np.clip(action[:6] + self._noise_rng.normal(0.0, self._noise_std, 6), -1.0, 1.0)
        yield from super()._send(action, phase)


@dataclass
class Attempt:
    """One rollout of the expansion.

    Attributes:
        layout: the bowl and plate positions (world frame).
        approach_source: index of the demo the approach came from.
        carry_source: index of the demo the carry came from.
        success: LIBERO's check fired and the episode is not longer than `max_episode_steps`.
        failure: why not ("" on success): the executor's reason, "no success", or "too long".
        steps: control steps taken.
        seconds: wall-clock time of the rollout.
        slowdown: factor the source demos were slowed by (1 unless paced with a range).
        episode_index: index in the written dataset (None when not written).
        eef_path_world: shape (steps, 3), end-effector position at each step (empty when read back from
            a saved report).
    """

    layout: SimLayout
    approach_source: int
    carry_source: int
    success: bool
    failure: str
    steps: int
    seconds: float
    slowdown: float = 1.0
    episode_index: int | None = None
    eef_path_world: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))


def eef_position_world(step: RolloutStep) -> np.ndarray:
    """End-effector position (world frame) the action of a step was computed from, shape (3,)."""
    return np.asarray(step.obs["robot_state"]["eef"]["pos"][0])


@dataclass
class ExpansionReport:
    """What an expansion did, with the demo list it indexes into.

    Attributes:
        source_names: name of each source demo (file stem).
        attempts: every rollout in order.
        wall_seconds: wall-clock time of the whole run, including writing.
    """

    source_names: list[str]
    attempts: list[Attempt] = field(default_factory=list)
    wall_seconds: float = 0.0

    @property
    def succeeded(self) -> int:
        """Number of kept episodes."""
        return sum(a.success for a in self.attempts)

    @property
    def success_rate(self) -> float:
        """Share of attempts that were kept."""
        return self.succeeded / len(self.attempts) if self.attempts else 0.0

    @property
    def failures(self) -> Counter[str]:
        """Failure reason -> count."""
        return Counter(a.failure for a in self.attempts if not a.success)

    @property
    def episodes_per_hour(self) -> float:
        """Kept episodes per hour of wall-clock time."""
        return 3600.0 * self.succeeded / self.wall_seconds if self.wall_seconds > 0 else 0.0

    def success_by_source(self, role: str) -> dict[str, tuple[int, int]]:
        """name -> (attempts, successes) of the demos used for the `role` ("approach" or "carry").

        A success is credited to both parts of a mixed plan, so it says how often a demo took part in a
        success, not how good that demo alone is.
        """
        counts: dict[str, list[int]] = {name: [0, 0] for name in self.source_names}
        for attempt in self.attempts:
            name = self.source_names[getattr(attempt, f"{role}_source")]
            counts[name][0] += 1
            counts[name][1] += attempt.success
        return {name: (n, k) for name, (n, k) in counts.items()}

    def step_counts(self) -> np.ndarray:
        """Steps of the kept episodes."""
        return np.array([a.steps for a in self.attempts if a.success])

    def summary(self, title: str = "") -> str:
        """A few lines for a terminal: counts, rate, failure reasons, speed and episode length."""
        steps = self.step_counts()
        length = (
            f"{np.median(steps):.0f} median, {steps.max()} max steps, "
            f"{(steps <= EVAL_STEP_LIMIT).sum()} within {EVAL_STEP_LIMIT}"
            if len(steps)
            else "no kept episodes"
        )
        lines = [
            f"{title}: {self.succeeded}/{len(self.attempts)} kept ({self.success_rate:.0%}), {length}",
            f"  failures: {dict(self.failures)}",
            f"  {self.episodes_per_hour:.0f} kept episodes/hour ({self.wall_seconds:.0f} s wall)",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """JSON-serialisable summary for the dataset's sidecar."""
        return {
            "attempted": len(self.attempts),
            "succeeded": self.succeeded,
            "success_rate": self.success_rate,
            "failures": dict(self.failures),
            "wall_seconds": self.wall_seconds,
            "episodes_per_hour": self.episodes_per_hour,
            "source_names": self.source_names,
            "attempts": [
                {
                    "bowl_xy": a.layout.bowl_xy.round(4).tolist(),
                    "plate_xy": a.layout.plate_xy.round(4).tolist(),
                    "approach_source": a.approach_source,
                    "carry_source": a.carry_source,
                    "success": a.success,
                    "failure": a.failure,
                    "steps": a.steps,
                    "slowdown": a.slowdown,
                    "episode_index": a.episode_index,
                }
                for a in self.attempts
            ],
        }


def report_from_dict(data: dict) -> ExpansionReport:
    """Read back what `ExpansionReport.to_dict` wrote (the end-effector paths are not stored)."""
    report = ExpansionReport(
        source_names=data["source_names"],
        wall_seconds=data["wall_seconds"],
    )
    for row in data["attempts"]:
        report.attempts.append(
            Attempt(
                layout=SimLayout(bowl_xy=np.array(row["bowl_xy"]), plate_xy=np.array(row["plate_xy"])),
                approach_source=row["approach_source"],
                carry_source=row["carry_source"],
                success=row["success"],
                failure=row["failure"],
                steps=row["steps"],
                seconds=0.0,
                slowdown=row["slowdown"],
                episode_index=row["episode_index"],
            )
        )
    return report


def load_run_reports(run_dir: Path) -> dict[str, ExpansionReport]:
    """The reports a run of `scripts/22_mimicgen_expand.py` saved: "mimicgen", and "stock" if asked for."""
    saved = json.loads((run_dir / "report.json").read_text())
    return {name: report_from_dict(saved[name]) for name in ("stock", "mimicgen") if name in saved}


def usable_demos(demos: list[HumanDemo]) -> list[HumanDemo]:
    """The demos that have a grasp and a release.

    Raises:
        ValueError: if there are none.
    """
    usable = [d for d in demos if d.grasp_frame_idx is not None and d.release_frame_idx is not None]
    if not usable:
        raise ValueError("no demo has a detected grasp and release")
    return usable


def load_source_demos(demos_dir: Path) -> tuple[list[HumanDemo], list[str]]:
    """The demos in a folder that may be used as sources, with their names.

    With a `summary.json` (written by `scripts/02_extract_demos.py`) only the clips marked `"usable": true`
    are loaded; flagged clips are not trusted. Without one, every `.npz` that has a grasp and a release.

    Args:
        demos_dir: folder of `<clip>.npz` files (read only).

    Returns:
        (demos, names), the names being the file stems.
    """
    summary_path = demos_dir / "summary.json"
    usable_stems = None
    if summary_path.exists():
        usable_stems = {
            Path(row["source_video"]).stem for row in json.loads(summary_path.read_text()) if row["usable"]
        }
    demos, names = [], []
    for path in sorted(demos_dir.glob("*.npz")):
        if usable_stems is not None and path.stem not in usable_stems:
            continue
        demo = HumanDemo.load(path)
        if demo.grasp_frame_idx is not None and demo.release_frame_idx is not None:
            demos.append(demo)
            names.append(path.stem)
    return demos, names


def source_layouts_world(demos: list[HumanDemo], mapping: TableToWorld) -> list[SimLayout]:
    """Each demo's own bowl and plate position in the world frame (not clamped to the work area)."""
    return [
        SimLayout(
            bowl_xy=mapping.point_to_world(np.asarray(d.t_table_bowl)[:3, 3])[:2],
            plate_xy=mapping.point_to_world(np.asarray(d.t_table_plate)[:3, 3])[:2],
        )
        for d in demos
    ]


def sample_layouts(scene: ScaledNativeScene, count: int, seed: int) -> list[SimLayout]:
    """`count` layouts from a sampler of its own, so that two runs can be compared on the same layouts."""
    sampler = LayoutSampler(scene.workspace, scene.setup.bowl.radius_m, scene.setup.plate.radius_m, seed=seed)
    return [sampler.sample() for _ in range(count)]


class MimicGenExpander:
    """Rolls source demos out on random layouts MimicGen-style and writes the successes."""

    def __init__(
        self,
        scene: ScaledNativeScene,
        demos: list[HumanDemo],
        names: list[str],
        config: MimicGenConfig | None = None,
    ) -> None:
        """Wire up the scene, the sources and the selector.

        Args:
            scene: the scaled-native scene.
            demos: processed human demos with grasp and release found.
            names: a name per demo, for the report.
            config: see `MimicGenConfig`.

        Raises:
            ValueError: if a demo lacks a grasp or release, or `names` does not match `demos`.
        """
        if len(names) != len(demos):
            raise ValueError("one name per demo")
        self._demos = usable_demos(demos)
        if len(self._demos) != len(demos):
            raise ValueError("every demo needs a detected grasp and release")
        self._names = list(names)
        self._scene = scene
        self._config = config or MimicGenConfig()
        self._rng = np.random.default_rng(self._config.seed)
        self._mapping = table_to_world_map(scene.setup, scene.workspace)
        self._selector = SourceSelector(
            source_layouts_world(self._demos, self._mapping),
            self._config.nn_k,
            self._config.per_subtask,
            self._rng,
        )

    def _retargeter(self) -> MixedRetargeter:
        """Built per rollout: the plate's height is only known once the scene has been reset."""
        return MixedRetargeter(
            self._mapping, self._scene.setup.bowl.radius_m, self._scene.plate_height_m, self._config.retarget
        )

    @property
    def selector(self) -> SourceSelector:
        """The source selector (for inspecting distances)."""
        return self._selector

    @property
    def mapping(self) -> TableToWorld:
        """The table-to-world map in use (the same as `DemoGenerator`'s)."""
        return self._mapping

    def attempt(
        self, layout: SimLayout, collect_frames: bool = True, choice: SourceChoice | None = None
    ) -> tuple[Attempt, list[dict]]:
        """One rollout on `layout`.

        Args:
            layout: where the bowl and plate go.
            collect_frames: build the dataset frames (needed to write the episode; skip for a dry run).
            choice: force these sources instead of letting the selector choose.

        Returns:
            (attempt, frames): `frames` is empty unless `collect_frames`.
        """
        cfg = self._config
        choice = choice or self._selector.choose(layout)
        slowdown = float(self._rng.uniform(*cfg.slowdown_range)) if cfg.retarget.paced else 1.0
        approach_demo = self._demos[choice.approach].time_scaled(slowdown)
        carry_demo = (
            approach_demo
            if choice.carry == choice.approach
            else self._demos[choice.carry].time_scaled(slowdown)
        )
        started = time.perf_counter()
        obs = self._scene.reset(layout)
        plan = self._retargeter().plan_mixed(approach_demo, carry_demo, layout)
        executor = NoisyPlanExecutor(
            self._scene, cfg.retarget, cfg.action_noise, self._rng, cfg.noise_free_phases
        )
        frames, eef_path = [], []
        steps = 0
        for step in executor.run(obs, plan):
            steps += 1
            eef_path.append(eef_position_world(step))
            if collect_frames:
                frames.append(frame_from_obs(step.obs, step.action))
        result = executor.result
        failure = result.failure or "no success"
        success = result.success
        if success and steps > cfg.max_episode_steps:
            success, failure = False, "too long"
        return (
            Attempt(
                layout=layout,
                approach_source=choice.approach,
                carry_source=choice.carry,
                success=success,
                failure="" if success else failure,
                steps=steps,
                seconds=time.perf_counter() - started,
                slowdown=slowdown,
                eef_path_world=np.array(eef_path),
            ),
            frames if success else [],
        )

    def screen(self, layouts: list[SimLayout], min_success: float = 0.25) -> dict[str, tuple[int, int]]:
        """Roll every source out alone (for approach and carry) on shared layouts, and drop the poor ones.

        MimicGen assumes every source demo is a good one. Ours come from a hand tracker, and a source whose
        grasp point cannot be gripped fails on every layout; with a nearest-neighbour choice that source then
        gets picked for the layouts it is nearest to. Nothing is written for these rollouts.

        Args:
            layouts: the same layouts for every source (a few are enough to tell 0 from 50 %).
            min_success: a source is kept if at least this share of its rollouts is kept. If none passes,
                every source stays allowed.

        Returns:
            name -> (rollouts, kept) of each source. The selector is restricted to the sources that passed.
        """
        outcome = {}
        for index, name in enumerate(self._names):
            kept = sum(
                self.attempt(layout, collect_frames=False, choice=SourceChoice(index, index))[0].success
                for layout in layouts
            )
            outcome[name] = (len(layouts), kept)
        passed = [i for i, name in enumerate(self._names) if outcome[name][1] >= min_success * len(layouts)]
        if passed:
            self._selector.restrict(passed)
        return outcome

    def expand(
        self,
        layouts: list[SimLayout],
        recorder: LeRobotRecorder | None,
        task: str,
        target_successes: int | None = None,
    ) -> ExpansionReport:
        """Roll out on each layout in turn and write the successes.

        Args:
            layouts: one rollout per layout (see `sample_layouts`).
            recorder: destination dataset (features from `generate.sim_features`); None for a dry run that
                only measures.
            task: instruction stored with each episode.
            target_successes: stop once this many episodes are kept.

        Returns:
            The record of every attempt.
        """
        report = ExpansionReport(source_names=self._names)
        started = time.perf_counter()
        for layout in layouts:
            if target_successes is not None and report.succeeded >= target_successes:
                break
            attempt, frames = self.attempt(layout, collect_frames=recorder is not None)
            if attempt.success and recorder is not None:
                attempt.episode_index = recorder.add_episode(frames, task)
            report.attempts.append(attempt)
        report.wall_seconds = time.perf_counter() - started
        return report


class StockBaseline:
    """The rollouts of `scripts/04_generate_dataset.py` (uniform source, no noise) on given layouts.

    Drives `DemoGenerator.rollout` itself, so that the comparison is against the real code path and not
    against this module's re-implementation of it.
    """

    def __init__(
        self,
        scene: ScaledNativeScene,
        demos: list[HumanDemo],
        names: list[str],
        retarget: RetargetConfig | None = None,
        seed: int = 0,
    ) -> None:
        """Build a `DemoGenerator` over the demos.

        Args:
            scene: the scaled-native scene.
            demos: processed human demos.
            names: a name per demo.
            retarget: settings, as for `DemoGenerator`.
            seed: for the choice of demo.
        """
        self._generator = DemoGenerator(scene, demos, config=retarget, seed=seed)
        self._demos = demos
        self._names = list(names)
        self._rng = np.random.default_rng(seed)

    def run(self, layouts: list[SimLayout]) -> ExpansionReport:
        """One rollout per layout with a uniformly drawn demo; nothing is written."""
        report = ExpansionReport(source_names=self._names)
        started = time.perf_counter()
        for layout in layouts:
            index = int(self._rng.integers(len(self._demos)))
            t0 = time.perf_counter()
            steps_iter, executor = self._generator.rollout(self._demos[index], layout)
            eef_path = [eef_position_world(step) for step in steps_iter]
            result = executor.result
            report.attempts.append(
                Attempt(
                    layout=layout,
                    approach_source=index,
                    carry_source=index,
                    success=result.success,
                    failure="" if result.success else (result.failure or "no success"),
                    steps=len(eef_path),
                    seconds=time.perf_counter() - t0,
                    eef_path_world=np.array(eef_path),
                )
            )
        report.wall_seconds = time.perf_counter() - started
        return report
