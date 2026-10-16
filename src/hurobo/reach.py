"""Where the Panda can hold its fingers upright on the LIBERO table: a position-error map."""

from __future__ import annotations

import numpy as np

from hurobo.control import Waypoint, WaypointFollower
from hurobo.scaled_native import ScaledNativeScene


def measure_reach_map(
    scene: ScaledNativeScene,
    xs: np.ndarray,
    ys: np.ndarray,
    z_world: float,
    steps_per_point: int = 45,
    tolerance_m: float = 0.004,
) -> np.ndarray:
    """Drive the end effector to each grid point with the home orientation and record the final error.

    Each row (one y) starts from the home pose and visits the x values in order, so a row costs fewer
    steps than the episode horizon.

    Args:
        scene: the scene; its env is reset once per row.
        xs: world x values, shape (X,).
        ys: world y values, shape (Y,).
        z_world: end-effector height to test, world frame.
        steps_per_point: control steps allowed to converge to a point.
        tolerance_m: error below which a point counts as reached early.

    Returns:
        shape (Y, X) distance between target and reached position, in metres.
    """
    task = scene.task
    errors = np.zeros((len(ys), len(xs)))
    follower: WaypointFollower | None = None
    for row, y in enumerate(ys):
        obs = scene.reset()
        if follower is None:
            pos_scale_m, rot_scale_rad = task.controller_action_scale
            follower = WaypointFollower(max_pos_delta_m=pos_scale_m, max_rot_delta_rad=rot_scale_rad)
        _, home_rot = task.get_eef_pose(obs)
        for column, x in enumerate(xs):
            target = np.array([x, y, z_world])
            for _ in range(steps_per_point):
                pos, rot = task.get_eef_pose(obs)
                if np.linalg.norm(target - pos) < tolerance_m:
                    break
                obs, _, _ = task.step(
                    follower.action_toward(Waypoint(target, home_rot, False), pos, rot, 0.8)
                )
            errors[row, column] = np.linalg.norm(target - task.get_eef_pose(obs)[0])
    return errors
