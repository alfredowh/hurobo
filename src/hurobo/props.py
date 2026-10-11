"""Props added to a LIBERO scene at runtime: a simple cube that appears between the fingers on grasp."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from hurobo.sim import LiberoTask


class CubeProp:
    """A small box with a collider that is invisible and inert until `spawn`.

    The cube is a free body added to the scene's MJCF (robosuite reloads the sim from the edited
    XML). Before `spawn` it has no collision and zero alpha; `spawn` drops it at a pose with
    collision and colour on. For a few steps afterwards gravity is cancelled (`body_gravcomp`) so
    it hovers between the open fingers while they close, then `release_support` hands it to
    friction and gravity. `LiberoTask.reset` rebuilds the stock task, so `install` is needed
    after every reset.
    """

    NAME = "teleop_cube"

    def __init__(
        self,
        half_extent_m: float = 0.018,
        rgba: tuple[float, float, float, float] = (0.95, 0.75, 0.10, 1.0),
        mass_kg: float = 0.05,
    ) -> None:
        """Define the cube.

        Args:
            half_extent_m: half the side length; must fit the gripper's opening (~8 cm total).
            rgba: colour; the default matches the yellow cube in the dummy hand clip.
            mass_kg: mass.
        """
        self._half_extent_m = half_extent_m
        self._rgba = np.array(rgba)
        self._mass_kg = mass_kg
        self._spawned = False
        self._supported = False
        self._body_id = -1
        self._geom_id = -1
        self._joint_id = -1

    @property
    def width_m(self) -> float:
        """Side length in metres."""
        return 2.0 * self._half_extent_m

    @property
    def spawned(self) -> bool:
        """True once `spawn` has been called since the last `install`."""
        return self._spawned

    @property
    def supported(self) -> bool:
        """True while gravity is cancelled for the cube."""
        return self._supported

    def install(self, task: LiberoTask) -> None:
        """Add the cube to the task's scene, parked: invisible, collision-free, weightless.

        Args:
            task: a freshly reset task.
        """
        root = ET.fromstring(task.model_xml)
        # gravcomp="1" at compile time matters: MuJoCo skips gravity compensation entirely when no
        # body had it at compile time (`ngravcomp == 0`), so toggling it at runtime alone is a no-op.
        body = ET.SubElement(root.find("worldbody"), "body", name=self.NAME, pos="0 0 1.5", gravcomp="1")
        ET.SubElement(body, "freejoint", name=f"{self.NAME}_joint")
        half = self._half_extent_m
        ET.SubElement(
            body,
            "geom",
            name=f"{self.NAME}_geom",
            type="box",
            size=f"{half} {half} {half}",
            mass=str(self._mass_kg),
            rgba=" ".join(str(c) for c in self._rgba),
            friction="1.5 0.005 0.0001",
            condim="4",
        )
        task.reload_model(ET.tostring(root, encoding="unicode"))
        model = task.mj_model
        self._body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.NAME)
        self._geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{self.NAME}_geom")
        self._joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{self.NAME}_joint")
        self._spawned = False
        self._set_active(task, active=False)
        model.body_gravcomp[self._body_id] = 1.0
        task.sim.forward()

    def spawn(self, task: LiberoTask, p_world: np.ndarray, yaw_rad: float = 0.0) -> None:
        """Make the cube real at `p_world` (its centre), at rest, supported against gravity.

        Args:
            task: the task it was installed in.
            p_world: shape (3,), centre position in the world frame.
            yaw_rad: rotation about world z, so its faces line up with the gripper's.
        """
        model, data = task.mj_model, task.mj_data
        qpos_adr = model.jnt_qposadr[self._joint_id]
        dof_adr = model.jnt_dofadr[self._joint_id]
        w, x, y, z = np.cos(yaw_rad / 2), 0.0, 0.0, np.sin(yaw_rad / 2)
        data.qpos[qpos_adr : qpos_adr + 7] = [*p_world, w, x, y, z]
        data.qvel[dof_adr : dof_adr + 6] = 0.0
        self._set_active(task, active=True)
        model.body_gravcomp[self._body_id] = 1.0
        self._spawned = True
        self._supported = True
        task.sim.forward()

    def release_support(self, task: LiberoTask) -> None:
        """Let gravity act on the cube (fingers are expected to hold it by friction by now)."""
        task.mj_model.body_gravcomp[self._body_id] = 0.0
        self._supported = False

    def position(self, task: LiberoTask) -> np.ndarray:
        """shape (3,), the cube centre in the world frame."""
        return np.array(task.mj_data.xpos[self._body_id])

    def _set_active(self, task: LiberoTask, active: bool) -> None:
        """Toggle collision and visibility together."""
        model = task.mj_model
        model.geom_contype[self._geom_id] = 1 if active else 0
        model.geom_conaffinity[self._geom_id] = 1 if active else 0
        model.geom_rgba[self._geom_id] = self._rgba if active else [*self._rgba[:3], 0.0]
