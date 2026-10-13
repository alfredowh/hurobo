"""Experiment E1 "Scaled-Native": LIBERO's own scene with the bowl and plate scaled to the real objects."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from hurobo.object_scale import ObjectExtent, measure_body_extent, resting_origin_height, scale_body_xml
from hurobo.real_setup import RealSetup
from hurobo.sim import LiberoTask, ObjectPose
from hurobo.teleop import TableToWorld
from hurobo.twin import SIM_TABLE_TOP_Z

SUITE_NAME = "libero_goal"
TASK_ID = 8  # "put the bowl on the plate"
BOWL_BODY, BOWL_JOINT = "akita_black_bowl_1_main", "akita_black_bowl_1_joint0"
PLATE_BODY, PLATE_JOINT = "plate_1_main", "plate_1_joint0"

#: The other objects of the libero_goal scene. With the plate at its real size there is no room left
#: for them (see `ScaledNativeScene`), so by default they are parked far from the table.
DISTRACTOR_BODIES = (
    "cream_cheese_1_main",
    "wine_bottle_1_main",
    "wooden_cabinet_1_main",
    "flat_stove_1_main",
    "wine_rack_1_main",
)
_DISTRACTOR_JOINTS = {
    "cream_cheese_1_main": "cream_cheese_1_joint0",
    "wine_bottle_1_main": "wine_bottle_1_joint0",
}
_PARKING_XY = np.array([0.0, 8.0])

#: The robot's base is at world x = -0.66; this table-to-world yaw puts the person's reach direction
#: (table +y, away from the person) along the robot's reach direction (world +x).
YAW_WORLD_TABLE_RAD = -np.pi / 2.0


@dataclass(frozen=True)
class ObjectScales:
    """Uniform scale factors that make the LIBERO bowl and plate as wide as the real ones.

    Attributes:
        bowl: factor for the bowl.
        plate: factor for the plate.
        native_bowl: LIBERO's own visual extent of the bowl (unscaled).
        native_plate: LIBERO's own visual extent of the plate (unscaled).
    """

    bowl: float
    plate: float
    native_bowl: ObjectExtent
    native_plate: ObjectExtent

    def height_error(self, real_bowl_height_m: float, real_plate_height_m: float) -> dict[str, float]:
        """Relative height error of each scaled object against the real one (positive = sim taller).

        A uniform scale matches the diameter exactly; the height follows from LIBERO's proportions.
        """
        return {
            "bowl": self.bowl * self.native_bowl.height_m / real_bowl_height_m - 1.0,
            "plate": self.plate * self.native_plate.height_m / real_plate_height_m - 1.0,
        }


@dataclass(frozen=True)
class SimLayout:
    """Where the bowl and plate are in the world frame (centres; the objects are round).

    Attributes:
        bowl_xy: shape (2,).
        plate_xy: shape (2,).
    """

    bowl_xy: np.ndarray
    plate_xy: np.ndarray


@dataclass(frozen=True)
class Workspace:
    """Rectangle of world positions where object centres may be put.

    Attributes:
        x_range: (min, max) world x. The upper end is limited by the Panda's reach from its base at x = -0.66.
        y_range: (min, max) world y.
    """

    x_range: tuple[float, float] = (-0.18, 0.10)
    y_range: tuple[float, float] = (-0.30, 0.30)

    @property
    def centre(self) -> np.ndarray:
        """shape (2,), centre of the rectangle."""
        return np.array([sum(self.x_range) / 2.0, sum(self.y_range) / 2.0])

    def contains(self, xy: np.ndarray) -> bool:
        """True if a world (x, y) is inside the rectangle."""
        return bool(
            self.x_range[0] <= xy[0] <= self.x_range[1] and self.y_range[0] <= xy[1] <= self.y_range[1]
        )


def table_to_world_map(setup: RealSetup, workspace: Workspace) -> TableToWorld:
    """Map the table frame to the world so that the real table centre lands on the workspace centre.

    Table z = 0 (the tabletop) maps to the sim table top, scale is 1 (the objects are real size).

    Args:
        setup: measured real setup.
        workspace: world rectangle the real work area should correspond to.

    Returns:
        The similarity (a yaw and a translation).
    """
    centre_xy = setup.table.centre_xy_m
    return TableToWorld(
        p_table_anchor=np.array([centre_xy[0], centre_xy[1], 0.0]),
        p_world_anchor=np.array([workspace.centre[0], workspace.centre[1], SIM_TABLE_TOP_Z]),
        yaw_rad=YAW_WORLD_TABLE_RAD,
        scale=1.0,
    )


class LayoutSampler:
    """Draws bowl/plate layouts that do not overlap and are inside the workspace."""

    def __init__(
        self,
        workspace: Workspace,
        bowl_radius_m: float,
        plate_radius_m: float,
        gap_m: float = 0.03,
        seed: int = 0,
    ) -> None:
        """Set the constraints.

        Args:
            workspace: where centres may be.
            bowl_radius_m: real bowl radius.
            plate_radius_m: real plate radius.
            gap_m: free space between the two rims at the start.
            seed: for the random generator.
        """
        self._workspace = workspace
        self._min_distance_m = bowl_radius_m + plate_radius_m + gap_m
        self._rng = np.random.default_rng(seed)

    @property
    def min_centre_distance_m(self) -> float:
        """Smallest allowed distance between the bowl and plate centres."""
        return self._min_distance_m

    def sample(self, max_tries: int = 1000) -> SimLayout:
        """One layout.

        Raises:
            RuntimeError: if the workspace is too small for the two objects.
        """
        x_low, x_high = self._workspace.x_range
        y_low, y_high = self._workspace.y_range
        for _ in range(max_tries):
            bowl = np.array([self._rng.uniform(x_low, x_high), self._rng.uniform(y_low, y_high)])
            plate = np.array([self._rng.uniform(x_low, x_high), self._rng.uniform(y_low, y_high)])
            if np.linalg.norm(bowl - plate) >= self._min_distance_m:
                return SimLayout(bowl_xy=bowl, plate_xy=plate)
        raise RuntimeError("no non-overlapping layout in the workspace; enlarge it or shrink the objects")


class ScaledNativeScene:
    """libero_goal task 8 with the bowl and plate scaled to the real ones.

    What changes against stock LIBERO, and why:

    - **Bowl and plate are uniformly scaled** so their diameters equal the measured ones (the visual mesh,
      the collision boxes and the mass follow). The real plate is 1.86x LIBERO's.
    - **The five other objects are parked off the table** (no collision, 8 m away). With the real-size plate
      almost no layout fits between them (0.06 % of random bowl/plate pairs in the reachable area, 29 % when
      cleared; `scripts/..`, notebook 10). Pass `keep_distractors=True` to leave them where LIBERO puts them.
    - **The initial layout is ours** (`SimLayout`), not LIBERO's init state: its 2 cm ranges were made for
      the small plate, which now overlaps the bowl there.

    Everything else (table, robot, camera, light, task, success check) is untouched.
    `LiberoTask.reset` rebuilds the stock task, so the edits are re-applied by `reset` here.
    """

    #: Zero-motion steps after a layout is applied, so the objects come to rest before the first observation.
    SETTLE_STEPS = 25

    def __init__(
        self,
        setup: RealSetup,
        task: LiberoTask | None = None,
        keep_distractors: bool = False,
        workspace: Workspace | None = None,
        seed: int = 0,
    ) -> None:
        """Build the LIBERO task (or take one) and measure the stock objects.

        Args:
            setup: the real measurements to scale to.
            task: an existing `LiberoTask` of libero_goal task 8; built if None.
            keep_distractors: leave the five other objects in place.
            workspace: where object centres may go; defaults to `Workspace()`.
            seed: for layout sampling.
        """
        self._setup = setup
        self._task = task or LiberoTask(SUITE_NAME, TASK_ID)
        self._keep_distractors = keep_distractors
        self.workspace = workspace or Workspace()
        self._sampler = LayoutSampler(self.workspace, setup.bowl.radius_m, setup.plate.radius_m, seed=seed)
        self._scales: ObjectScales | None = None
        self._base_offsets: dict[str, float] = {}

    @property
    def setup(self) -> RealSetup:
        """The real measurements the scene is scaled to."""
        return self._setup

    @property
    def plate_height_m(self) -> float:
        """Height of the scaled plate above the table (visual mesh); available after the first `reset`."""
        return self.scales.plate * self.scales.native_plate.height_m

    @property
    def task(self) -> LiberoTask:
        """The underlying `LiberoTask` (cameras, stepping, raw MuJoCo access)."""
        return self._task

    @property
    def scales(self) -> ObjectScales:
        """The factors in use; available after the first `reset`."""
        if self._scales is None:
            raise RuntimeError("call reset() first")
        return self._scales

    @property
    def sampler(self) -> LayoutSampler:
        """The layout sampler (for generating many layouts)."""
        return self._sampler

    def reset(self, layout: SimLayout | None = None) -> dict:
        """Reset the episode, apply the scaling and the layout, and let things settle.

        Args:
            layout: where to put bowl and plate; sampled if None.

        Returns:
            The LeRobot-style observation after settling (leading n_envs = 1 axis, like `LiberoTask.reset`).
        """
        self._task.reset()
        model, data = self._task.mj_model, self._task.mj_data
        mujoco.mj_forward(model, data)
        if self._scales is None:
            self._scales = self._measure_scales(model, data)
        self._apply_scaling()
        mujoco.mj_forward(self._task.mj_model, self._task.mj_data)
        self._base_offsets = {
            body: measure_body_extent(
                self._task.mj_model, self._task.mj_data, body, collision=True
            ).base_offset_m
            for body in (BOWL_BODY, PLATE_BODY)
        }
        if not self._keep_distractors:
            self._park_distractors()
        self.place(layout or self._sampler.sample())
        return self._settle()

    def place(self, layout: SimLayout) -> None:
        """Put the bowl and plate upright on the table at the layout, at rest height, yaw 0."""
        for joint, body, xy in (
            (BOWL_JOINT, BOWL_BODY, layout.bowl_xy),
            (PLATE_JOINT, PLATE_BODY, layout.plate_xy),
        ):
            z = resting_origin_height(self._base_offsets[body], SIM_TABLE_TOP_Z, clearance_m=0.0005)
            self._task.set_object_pose(joint, ObjectPose(np.array([xy[0], xy[1], z]), Rotation.identity()))
        self._task.mj_data.qvel[:] = 0.0
        self._task.sim.forward()

    def object_xy(self) -> SimLayout:
        """Current world (x, y) of the bowl and plate centres."""
        return SimLayout(
            bowl_xy=self._task.get_object_pose(BOWL_JOINT).pos[:2].copy(),
            plate_xy=self._task.get_object_pose(PLATE_JOINT).pos[:2].copy(),
        )

    def _measure_scales(self, model: mujoco.MjModel, data: mujoco.MjData) -> ObjectScales:
        bowl = measure_body_extent(model, data, BOWL_BODY)
        plate = measure_body_extent(model, data, PLATE_BODY)
        return ObjectScales(
            bowl=self._setup.bowl.diameter_m / bowl.diameter_m,
            plate=self._setup.plate.diameter_m / plate.diameter_m,
            native_bowl=bowl,
            native_plate=plate,
        )

    def _apply_scaling(self) -> None:
        scales = self.scales
        root = ET.fromstring(self._task.model_xml)
        scale_body_xml(root, BOWL_BODY, scales.bowl)
        scale_body_xml(root, PLATE_BODY, scales.plate)
        self._task.reload_model(ET.tostring(root, encoding="unicode"))

    def _park_distractors(self) -> None:
        """Move the other objects far away and make them non-colliding, without removing them."""
        model = self._task.mj_model
        for slot, body_name in enumerate(DISTRACTOR_BODIES):
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                continue
            for geom_id in range(model.ngeom):
                if model.body_rootid[model.geom_bodyid[geom_id]] == body_id:
                    model.geom_contype[geom_id] = 0
                    model.geom_conaffinity[geom_id] = 0
            target_xy = _PARKING_XY + np.array([slot * 1.0, 0.0])
            joint = _DISTRACTOR_JOINTS.get(body_name)
            if joint is not None:
                qpos = self._task.sim.data.get_joint_qpos(joint).copy()
                qpos[:3] = [target_xy[0], target_xy[1], 2.0]
                self._task.sim.data.set_joint_qpos(joint, qpos)
            else:
                model.body_pos[body_id][:2] = target_xy
        self._task.sim.forward()

    def _settle(self) -> dict:
        hold = np.zeros(7, dtype=np.float32)
        hold[6] = -1.0  # gripper open, as at reset
        obs = None
        for _ in range(self.SETTLE_STEPS):
            obs, _success, _terminated = self._task.step(hold)
        return obs
