"""Measuring and uniformly scaling a body in MJCF, on a toy cup that needs no simulator assets."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from hurobo.object_scale import measure_body_extent, resting_origin_height, scale_body_xml

CUP_XML = """
<mujoco>
  <asset><mesh name="cup_mesh" vertex="0 0 0  1 0 0  0 1 0  0 0 1" scale="0.5 0.5 0.5"/></asset>
  <worldbody>
    <body name="cup" pos="0 0 0.1">
      <joint name="cup_free" type="free"/>
      <geom name="floor" type="box" size="0.04 0.04 0.002" pos="0 0 0.002" density="100"/>
      <geom name="wall_l" type="box" size="0.002 0.04 0.03" pos="-0.04 0 0.03" density="100"/>
      <geom name="wall_r" type="box" size="0.002 0.04 0.03" pos="0.04 0 0.03" density="100"/>
      <site name="top" pos="0 0 0.06" size="0.005"/>
      <body name="handle" pos="0.05 0 0.03">
        <geom type="box" size="0.01 0.01 0.01" density="100"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _compile(xml: str) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _scaled_cup_xml(factor: float) -> str:
    root = ET.fromstring(CUP_XML)
    scale_body_xml(root, "cup", factor)
    return ET.tostring(root, encoding="unicode")


def test_extent_of_the_toy_cup() -> None:
    model, data = _compile(CUP_XML)
    extent = measure_body_extent(model, data, "cup", collision=True)
    assert extent.height_m == pytest.approx(0.06, abs=1e-6)
    assert extent.diameter_m == pytest.approx(2 * np.hypot(0.042, 0.04), abs=1e-6)
    assert extent.base_offset_m == pytest.approx(0.0, abs=1e-6)


def test_uniform_scale_scales_size_mass_and_sites() -> None:
    model, data = _compile(CUP_XML)
    scaled_model, scaled_data = _compile(_scaled_cup_xml(1.5))
    before = measure_body_extent(model, data, "cup", collision=True)
    after = measure_body_extent(scaled_model, scaled_data, "cup", collision=True)
    assert after.diameter_m == pytest.approx(1.5 * before.diameter_m, rel=1e-6)
    assert after.height_m == pytest.approx(1.5 * before.height_m, rel=1e-6)
    cup, scaled_cup = model.body("cup"), scaled_model.body("cup")
    assert scaled_model.body_subtreemass[scaled_cup.id] == pytest.approx(
        1.5**3 * model.body_subtreemass[cup.id], rel=1e-6
    )
    top = mujoco.mj_name2id(scaled_model, mujoco.mjtObj.mjOBJ_SITE, "top")
    assert scaled_model.site_pos[top][2] == pytest.approx(0.09)
    handle = scaled_model.body("handle")
    assert scaled_model.body_pos[handle.id][0] == pytest.approx(0.075)


def test_body_placement_is_left_alone_and_other_bodies_untouched() -> None:
    root = ET.fromstring(CUP_XML)
    scale_body_xml(root, "cup", 2.0)
    cup = next(b for b in root.iter("body") if b.get("name") == "cup")
    assert cup.get("pos") == "0 0 0.1"


def test_mesh_asset_scale_is_multiplied() -> None:
    root = ET.fromstring(
        CUP_XML.replace('<site name="top"', '<geom type="mesh" mesh="cup_mesh"/><site name="top"')
    )
    scale_body_xml(root, "cup", 2.0)
    assert root.find("asset/mesh").get("scale") == "1 1 1"


def test_invalid_factor_and_missing_body_raise() -> None:
    root = ET.fromstring(CUP_XML)
    with pytest.raises(ValueError):
        scale_body_xml(root, "cup", 0.0)
    with pytest.raises(ValueError):
        scale_body_xml(root, "saucer", 1.2)


def test_resting_height_puts_the_lowest_point_on_the_surface() -> None:
    assert resting_origin_height(base_offset_m=-0.005, surface_z_m=0.9, clearance_m=0.001) == pytest.approx(
        0.906
    )
