"""Measure a LIBERO object's size and rescale it, collision and visual geometry together."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

import mujoco
import numpy as np

_BOX_SIGNS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)


@dataclass(frozen=True)
class ObjectExtent:
    """Size of an upright, roughly round object, from its geometry in the current pose.

    Attributes:
        diameter_m: twice the largest horizontal distance of any point from the object's vertical axis
            (the bounding circle, so it does not depend on yaw).
        height_m: highest minus lowest point.
        base_offset_m: z of the lowest point relative to the body origin; the origin has to sit this far
            above the surface for the object to rest on it.
    """

    diameter_m: float
    height_m: float
    base_offset_m: float


def _geom_points_world(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> np.ndarray:
    """Corner points (box) or vertices (mesh) of one geom in the world frame, shape (N, 3)."""
    rotation = np.asarray(data.geom_xmat[geom_id]).reshape(3, 3)
    centre = np.asarray(data.geom_xpos[geom_id])
    if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH:
        mesh_id = model.geom_dataid[geom_id]
        start, count = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
        local = np.asarray(model.mesh_vert[start : start + count])
    elif model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX:
        local = _BOX_SIGNS * np.asarray(model.geom_size[geom_id])
    else:
        raise ValueError(f"geom type {model.geom_type[geom_id]} is not supported for size measurement")
    return local @ rotation.T + centre


def measure_body_extent(
    model: mujoco.MjModel, data: mujoco.MjData, body_name: str, collision: bool = False
) -> ObjectExtent:
    """Size of a free body from its box and mesh geoms in the current pose.

    Args:
        model: compiled model.
        data: its data with kinematics up to date (`mj_forward`).
        body_name: e.g. "plate_1_main".
        collision: True measures the geoms that collide (what the gripper touches), False the visual
            mesh (what a camera and a ruler see).

    Returns:
        The extent. The body is assumed to be upright.

    Raises:
        ValueError: if the body has no geom of the requested kind.
    """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"no body named {body_name!r}")
    points = []
    for geom_id in range(model.ngeom):
        if model.geom_bodyid[geom_id] != body_id:
            continue
        is_collision = model.geom_contype[geom_id] != 0 or model.geom_conaffinity[geom_id] != 0
        if is_collision == collision:
            points.append(_geom_points_world(model, data, geom_id))
    if not points:
        raise ValueError(f"{body_name!r} has no {'collision' if collision else 'visual'} geoms")
    cloud = np.concatenate(points)
    axis_xy = cloud[:, :2].mean(axis=0) if collision else (cloud[:, :2].min(0) + cloud[:, :2].max(0)) / 2
    radius = float(np.linalg.norm(cloud[:, :2] - axis_xy, axis=1).max())
    origin_z = float(data.xpos[body_id][2])
    return ObjectExtent(
        diameter_m=2.0 * radius,
        height_m=float(cloud[:, 2].max() - cloud[:, 2].min()),
        base_offset_m=float(cloud[:, 2].min() - origin_z),
    )


def _scale_floats(element: ET.Element, attribute: str, factor: float) -> None:
    """Multiply a whitespace-separated numeric attribute by `factor`, if present."""
    value = element.get(attribute)
    if value is not None:
        element.set(attribute, " ".join(f"{float(v) * factor:.8g}" for v in value.split()))


def scale_body_xml(root: ET.Element, body_name: str, factor: float) -> None:
    """Uniformly scale one body in an MJCF tree, about the body origin, in place.

    Covers what makes up a LIBERO object: geom positions and sizes, the visual mesh's asset scale, sites,
    nested bodies, joint anchors and an explicit inertial (mass by factor^3, inertia by factor^5).
    Masses that MuJoCo derives from geom density follow automatically, so the object keeps its density.
    The body's own `pos` is left alone: it is the placement, not part of the shape.

    Args:
        root: the `<mujoco>` element, e.g. from `ET.fromstring(task.model_xml)`.
        body_name: the body to scale.
        factor: > 0.

    Raises:
        ValueError: if `factor` is not positive or the body does not exist.
    """
    if factor <= 0:
        raise ValueError("factor must be positive")
    body = next((b for b in root.iter("body") if b.get("name") == body_name), None)
    if body is None:
        raise ValueError(f"no body named {body_name!r}")
    meshes = {m.get("name"): m for m in root.iter("mesh")}
    scaled_meshes: set[str] = set()
    for element in body.iter():
        if element.tag == "geom":
            _scale_floats(element, "pos", factor)
            if element.get("type") != "mesh":
                _scale_floats(element, "size", factor)
            mesh_name = element.get("mesh")
            if mesh_name is not None and mesh_name not in scaled_meshes:
                mesh = meshes[mesh_name]
                if mesh.get("scale") is None:
                    mesh.set("scale", "1 1 1")
                _scale_floats(mesh, "scale", factor)
                scaled_meshes.add(mesh_name)
        elif element.tag == "site":
            _scale_floats(element, "pos", factor)
            _scale_floats(element, "size", factor)
        elif element.tag == "body" and element is not body:
            _scale_floats(element, "pos", factor)
        elif element.tag == "joint" and element.get("type") != "free":
            _scale_floats(element, "pos", factor)
        elif element.tag == "inertial":
            _scale_floats(element, "pos", factor)
            _scale_floats(element, "mass", factor**3)
            _scale_floats(element, "diaginertia", factor**5)


def resting_origin_height(base_offset_m: float, surface_z_m: float, clearance_m: float = 0.001) -> float:
    """World z for a body origin so that the object's lowest point sits just above a surface."""
    return surface_z_m - base_offset_m + clearance_m
