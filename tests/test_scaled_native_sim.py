"""The scaled-native scene in the real LIBERO simulator: sizes, layout, and one retargeted rollout."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import pytest

pytest.importorskip("libero")

from hurobo.generate import DemoGenerator
from hurobo.object_scale import measure_body_extent
from hurobo.real_setup import DEFAULT_MEASUREMENTS, RealSetup
from hurobo.scaled_native import BOWL_BODY, PLATE_BODY, ScaledNativeScene, SimLayout
from hurobo.utils.synthetic import synthetic_pick_place_demo

pytestmark = pytest.mark.skipif(
    not Path(DEFAULT_MEASUREMENTS).exists(), reason="needs data/raw/setup_01/measurements.yaml"
)


@pytest.fixture(scope="module")
def scene() -> ScaledNativeScene:
    return ScaledNativeScene(RealSetup.load(), seed=0)


def test_objects_have_the_real_diameters_and_sit_where_asked(scene: ScaledNativeScene) -> None:
    layout = SimLayout(bowl_xy=np.array([-0.05, 0.12]), plate_xy=np.array([0.02, -0.12]))
    scene.reset(layout)
    model, data = scene.task.mj_model, scene.task.mj_data
    setup = scene.setup
    bowl = measure_body_extent(model, data, BOWL_BODY)
    plate = measure_body_extent(model, data, PLATE_BODY)
    assert bowl.diameter_m == pytest.approx(setup.bowl.diameter_m, rel=0.02)
    assert plate.diameter_m == pytest.approx(setup.plate.diameter_m, rel=0.02)
    placed = scene.object_xy()
    assert np.allclose(placed.bowl_xy, layout.bowl_xy, atol=0.005)
    assert np.allclose(placed.plate_xy, layout.plate_xy, atol=0.005)


def test_a_retargeted_demo_places_the_scaled_bowl_on_the_scaled_plate(scene: ScaledNativeScene) -> None:
    demo_layout = SimLayout(bowl_xy=np.array([-0.08, 0.12]), plate_xy=np.array([0.0, -0.12]))
    mapping = DemoGenerator(scene, [synthetic_pick_place_demo(np.zeros(2), np.ones(2), np.zeros(3))]).mapping
    bowl_t = mapping.point_to_table(np.array([*demo_layout.bowl_xy, 0.9]))[:2]
    plate_t = mapping.point_to_table(np.array([*demo_layout.plate_xy, 0.9]))[:2]
    demo = synthetic_pick_place_demo(bowl_t, plate_t, np.array([0.2, -0.5, 0.25]), grasp_angle_rad=0.0)
    generator = DemoGenerator(scene, [demo])
    steps, executor = generator.rollout(demo, generator.layout_of_demo(demo))
    for _ in steps:
        pass
    assert executor.result.success, executor.result.failure
