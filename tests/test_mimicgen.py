"""MimicGen-style expansion: source selection, mixed plans, action noise and the report (no simulator)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hurobo.mimicgen import (
    Attempt,
    ExpansionReport,
    MimicGenConfig,
    MixedRetargeter,
    NoisyPlanExecutor,
    SourceSelector,
    load_run_reports,
    load_source_demos,
)
from hurobo.retarget import RetargetConfig
from hurobo.scaled_native import SimLayout
from hurobo.teleop import TableToWorld
from hurobo.utils.synthetic import synthetic_pick_place_demo


def layout(bowl: tuple[float, float], plate: tuple[float, float]) -> SimLayout:
    return SimLayout(bowl_xy=np.array(bowl), plate_xy=np.array(plate))


SOURCES = [
    layout((0.0, 0.0), (0.0, 0.3)),  # bowl-to-plate vector (0, 0.3)
    layout((0.1, 0.0), (0.1, -0.3)),  # (0, -0.3)
    layout((-0.1, 0.1), (0.2, 0.1)),  # (0.3, 0)
    layout((0.05, -0.1), (0.05, 0.2)),  # (0, 0.3)
]


def test_approach_distance_is_about_the_bowl_and_carry_distance_about_the_vector() -> None:
    selector = SourceSelector(SOURCES, nn_k=1, per_subtask=True, rng=np.random.default_rng(0))
    target = layout((0.1, 0.0), (0.1, 0.3))
    assert selector.approach_distances_m(target).argmin() == 1
    assert np.allclose(selector.carry_distances_m(target)[[0, 3]], 0.0)


def test_nearest_neighbour_choice_is_per_subtask_and_stays_in_the_pool() -> None:
    selector = SourceSelector(SOURCES, nn_k=1, per_subtask=True, rng=np.random.default_rng(0))
    target = layout((0.1, 0.0), (0.1, 0.3))
    choice = selector.choose(target)
    assert choice.approach == 1  # nearest bowl
    assert choice.carry in (0, 3)  # same bowl-to-plate vector; tie broken by argsort order
    pool = SourceSelector(SOURCES, nn_k=2, per_subtask=True, rng=np.random.default_rng(1))
    picks = {pool.choose(target).approach for _ in range(60)}
    assert picks == set(np.argsort(pool.approach_distances_m(target))[:2])


def test_shared_source_uses_one_demo_for_both_subtasks() -> None:
    selector = SourceSelector(SOURCES, nn_k=2, per_subtask=False, rng=np.random.default_rng(0))
    for _ in range(20):
        choice = selector.choose(layout((0.0, 0.05), (0.0, 0.35)))
        assert choice.approach == choice.carry


def test_nn_k_zero_is_uniform_over_all_sources() -> None:
    selector = SourceSelector(SOURCES, nn_k=0, per_subtask=True, rng=np.random.default_rng(0))
    target = layout((0.0, 0.0), (0.0, 0.3))
    assert {selector.choose(target).approach for _ in range(100)} == {0, 1, 2, 3}


def test_mixed_plan_takes_approach_from_one_demo_and_carry_from_the_other() -> None:
    mapping = TableToWorld(p_table_anchor=np.zeros(3), p_world_anchor=np.zeros(3), yaw_rad=-np.pi / 2)
    retargeter = MixedRetargeter(mapping, bowl_radius_m=0.069, plate_height_m=0.03, config=RetargetConfig())
    demo_a = synthetic_pick_place_demo(
        np.array([0.1, 0.2]), np.array([-0.2, 0.2]), np.array([0.3, -0.4, 0.2])
    )
    demo_b = synthetic_pick_place_demo(
        np.array([-0.1, 0.0]), np.array([0.2, 0.3]), np.array([0.0, -0.5, 0.25]), seed=3
    )
    target = layout((-0.05, 0.1), (0.0, -0.15))
    plan_a, plan_b = retargeter.plan(demo_a, target), retargeter.plan(demo_b, target)
    mixed = retargeter.plan_mixed(demo_a, demo_b, target)
    assert np.allclose(mixed.approach_world, plan_a.approach_world)
    assert mixed.gripper_yaw_rad == plan_a.gripper_yaw_rad
    assert np.allclose(mixed.bowl_path_world, plan_b.bowl_path_world)
    # Both parts are anchored on the real objects, so the parts meet at the grasp whatever mixed.
    assert np.allclose(mixed.bowl_path_world[0, :2], target.bowl_xy)
    assert np.allclose(mixed.bowl_path_world[-1, :2], target.plate_xy)
    assert retargeter.plan_mixed(demo_a, demo_a, target).approach_world.shape == plan_a.approach_world.shape


class FakeTask:
    """Records the actions an executor sends; a stand-in for `LiberoTask`."""

    controller_action_scale = (0.05, 0.5)

    def __init__(self) -> None:
        self.sent: list[np.ndarray] = []

    def step(self, action: np.ndarray) -> tuple[dict, bool, bool]:
        self.sent.append(np.asarray(action).copy())
        return {}, False, False


def send_all(executor: NoisyPlanExecutor, action: np.ndarray, phase: str, count: int) -> None:
    for _ in range(count):
        for _step in executor._send(action, phase):
            pass


def make_executor(noise: float, quiet: tuple[str, ...] = ()) -> tuple[NoisyPlanExecutor, FakeTask]:
    task = FakeTask()
    executor = NoisyPlanExecutor(
        SimpleNamespace(task=task),
        RetargetConfig(max_total_steps=10**6),
        noise,
        np.random.default_rng(0),
        quiet,
    )
    return executor, task


def test_noise_perturbs_the_pose_dimensions_only_and_keeps_the_range() -> None:
    executor, task = make_executor(0.05)
    action = np.array([0.2, -0.2, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    send_all(executor, action, "carry", 400)
    sent = np.array(task.sent)
    assert np.all(sent[:, 6] == 1.0)
    assert np.allclose(sent[:, :6].mean(axis=0), action[:6], atol=0.01)
    assert np.allclose(sent[:, :6].std(axis=0), 0.05, atol=0.01)
    assert np.abs(sent).max() <= 1.0


def test_noise_is_off_in_quiet_phases_and_when_sigma_is_zero() -> None:
    action = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
    executor, task = make_executor(0.05, quiet=("grasp",))
    send_all(executor, action, "grasp", 5)
    assert all(np.array_equal(sent, action) for sent in task.sent)
    executor, task = make_executor(0.0)
    send_all(executor, action, "carry", 5)
    assert all(np.array_equal(sent, action) for sent in task.sent)


def test_clipping_keeps_a_full_scale_action_inside_the_range() -> None:
    executor, task = make_executor(0.5)
    send_all(executor, np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0], dtype=np.float32), "carry", 200)
    assert np.abs(np.array(task.sent)).max() <= 1.0


def attempt(approach: int, carry: int, success: bool, failure: str = "", steps: int = 200) -> Attempt:
    return Attempt(layout((0, 0), (0, 0.3)), approach, carry, success, failure, steps, seconds=1.0)


def test_report_counts_rates_and_sources() -> None:
    report = ExpansionReport(source_names=["a", "b"], wall_seconds=36.0)
    report.attempts = [
        attempt(0, 0, True),
        attempt(0, 1, False, "bowl dropped"),
        attempt(1, 1, True, steps=250),
        attempt(1, 0, False, "too long"),
    ]
    assert report.succeeded == 2 and report.success_rate == 0.5
    assert dict(report.failures) == {"bowl dropped": 1, "too long": 1}
    assert report.episodes_per_hour == pytest.approx(200.0)
    assert report.success_by_source("approach") == {"a": (2, 1), "b": (2, 1)}
    assert report.success_by_source("carry") == {"a": (2, 1), "b": (2, 1)}
    assert report.step_counts().tolist() == [200, 250]
    assert report.to_dict()["attempts"][1]["failure"] == "bowl dropped"
    assert "2/4" in report.summary("x")


def test_stock_config_switches_every_mimicgen_feature_off() -> None:
    stock = MimicGenConfig.stock()
    assert stock.nn_k == 0 and not stock.per_subtask and stock.action_noise == 0.0
    assert stock.max_episode_steps > 10**6


def test_report_survives_a_save_and_load(tmp_path: Path) -> None:
    report = ExpansionReport(source_names=["a", "b"], wall_seconds=10.0)
    report.attempts = [attempt(0, 1, True, steps=210), attempt(1, 0, False, "timeout")]
    (tmp_path / "report.json").write_text(json.dumps({"mimicgen": report.to_dict()}))
    loaded = load_run_reports(tmp_path)["mimicgen"]
    assert loaded.source_names == ["a", "b"]
    assert [a.failure for a in loaded.attempts] == ["", "timeout"]
    assert loaded.attempts[0].layout.plate_xy.tolist() == [0.0, 0.3]
    assert loaded.step_counts().tolist() == [210]


def test_only_clips_marked_usable_are_loaded(tmp_path: Path) -> None:
    for stem in ("good", "flagged"):
        synthetic_pick_place_demo(np.zeros(2), np.ones(2) * 0.3, np.zeros(3)).save(tmp_path / f"{stem}.npz")
    rows = [{"source_video": f"clips/{stem}.mp4", "usable": stem == "good"} for stem in ("good", "flagged")]
    (tmp_path / "summary.json").write_text(json.dumps(rows))
    demos, names = load_source_demos(tmp_path)
    assert names == ["good"] and len(demos) == 1
    (tmp_path / "summary.json").unlink()
    assert load_source_demos(tmp_path)[1] == ["flagged", "good"]


def test_restricted_selector_never_picks_a_dropped_source() -> None:
    selector = SourceSelector(SOURCES, nn_k=2, per_subtask=True, rng=np.random.default_rng(0))
    selector.restrict([1, 3])
    target = layout((-0.1, 0.1), (0.2, 0.1))  # nearest to source 2, which is dropped
    for _ in range(50):
        choice = selector.choose(target)
        assert choice.approach in (1, 3) and choice.carry in (1, 3)
    with pytest.raises(ValueError):
        selector.restrict([])
