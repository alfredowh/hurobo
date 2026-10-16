"""Experiment E1 "Scaled-Native": LIBERO's own scene with the bowl and plate scaled to the real objects."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from hurobo.object_scale import (
    ObjectExtent,
    body_footprint_xy,
    measure_body_extent,
    resting_origin_height,
    scale_body_xml,
)
from hurobo.real_setup import RealSetup
from hurobo.sim import LiberoTask, ObjectPose
from hurobo.teleop import TableToWorld
from hurobo.twin import SIM_TABLE_TOP_Z
from hurobo.utils.transforms import cv_camera_pose_to_mujoco, transform_to_pos_quat_wxyz

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

#: World (x, y) of the Panda's base (read from the sim).
ROBOT_BASE_XY = (-0.66, 0.0)

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

    def comparison(self, setup: RealSetup) -> list[dict[str, float | str]]:
        """Rows (object, real, LIBERO stock, scaled, factor), in centimetres, for a table.

        Args:
            setup: the measured objects.
        """
        rows: list[dict[str, float | str]] = []
        for name, real, native, factor in (
            ("bowl", setup.bowl, self.native_bowl, self.bowl),
            ("plate", setup.plate, self.native_plate, self.plate),
        ):
            rows.append(
                {
                    "object": name,
                    "real diameter": 100 * real.diameter_m,
                    "LIBERO diameter": 100 * native.diameter_m,
                    "scale factor": factor,
                    "sim diameter": 100 * factor * native.diameter_m,
                    "real height": 100 * real.height_m,
                    "LIBERO height": 100 * native.height_m,
                    "sim height": 100 * factor * native.height_m,
                }
            )
        return rows

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
        y_range: (min, max) world y. Narrower than the table: at |y| = 0.3 the arm no longer reaches x = 0.1
            with the fingers upright (position error 2 cm at table height, see `hurobo.reach`).
    """

    x_range: tuple[float, float] = (-0.18, 0.08)
    y_range: tuple[float, float] = (-0.25, 0.25)

    @property
    def centre(self) -> np.ndarray:
        """shape (2,), centre of the rectangle."""
        return np.array([sum(self.x_range) / 2.0, sum(self.y_range) / 2.0])

    def contains(self, xy: np.ndarray) -> bool:
        """True if a world (x, y) is inside the rectangle."""
        return bool(
            self.x_range[0] <= xy[0] <= self.x_range[1] and self.y_range[0] <= xy[1] <= self.y_range[1]
        )

    def clamp(self, xy: np.ndarray) -> np.ndarray:
        """The world (x, y) nearest to `xy` that lies inside the rectangle."""
        return np.array(
            [np.clip(xy[0], *self.x_range), np.clip(xy[1], *self.y_range)],
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
      almost no layout fits between them (0.00 % of random bowl/plate pairs in the work area, 38 % when
      cleared; notebook 08). Pass `keep_distractors=True` to leave them where LIBERO puts them.
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

    def reset(self, layout: SimLayout | None = None, scaled: bool = True) -> dict:
        """Reset the episode, apply the scaling and the layout, and let things settle.

        Args:
            layout: where to put bowl and plate; sampled if None.
            scaled: False keeps LIBERO's own bowl and plate sizes (everything else as usual), to compare
                against the scaled scene on the same layout.

        Returns:
            The LeRobot-style observation after settling (leading n_envs = 1 axis, like `LiberoTask.reset`).
        """
        self._task.reset()
        model, data = self._task.mj_model, self._task.mj_data
        mujoco.mj_forward(model, data)
        if self._scales is None:
            self._scales = self._measure_scales(model, data)
        if scaled:
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

    def render_top_down(self, size: int = 480) -> np.ndarray:
        """Render the table from straight above (the robot at the top of the image), for size comparisons.

        Moves the twin camera; the policy cameras are untouched.

        Args:
            size: image side in pixels.
        """
        height_above_table_m = 1.0
        centre = np.array([*self.workspace.centre, SIM_TABLE_TOP_Z])
        # OpenCV axes: image right = world +y, image down = world +x, optical axis = world -z.
        t_world_cam_cv = np.eye(4)
        t_world_cam_cv[:3, :3] = np.column_stack([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
        t_world_cam_cv[:3, 3] = centre + np.array([0.0, 0.0, height_above_table_m])
        pos, quat_wxyz = transform_to_pos_quat_wxyz(cv_camera_pose_to_mujoco(t_world_cam_cv))
        self._task.set_camera_pose(self._task.TWIN_CAMERA_NAME, pos, quat_wxyz, 50.0)
        return self._task.render_camera(self._task.TWIN_CAMERA_NAME, size, size)

    def distractor_footprints(self) -> dict[str, tuple[float, float, float, float]]:
        """World (x_min, x_max, y_min, y_max) of each other object in the stock scene.

        This resets the task; call `reset` again before using the scaled scene.
        """
        self._task.reset()
        model, data = self._task.mj_model, self._task.mj_data
        mujoco.mj_forward(model, data)
        return {name: body_footprint_xy(model, data, name) for name in DISTRACTOR_BODIES}

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


def layout_feasible_fraction(
    workspace: Workspace,
    bowl_radius_m: float,
    plate_radius_m: float,
    footprints: dict[str, tuple[float, float, float, float]],
    samples: int = 50000,
    clearance_m: float = 0.015,
    gap_m: float = 0.03,
    seed: int = 0,
) -> float:
    """Share of random bowl/plate placements in the workspace that touch neither each other nor an object.

    Args:
        workspace: where object centres are drawn from, uniformly and independently.
        bowl_radius_m: real bowl radius.
        plate_radius_m: real plate radius.
        footprints: rectangles to stay clear of, e.g. `ScaledNativeScene.distractor_footprints()`; empty
            for a cleared table.
        samples: number of random pairs.
        clearance_m: extra distance kept from those rectangles.
        gap_m: free space between the bowl and plate rims.
        seed: for the generator.
    """
    rng = np.random.default_rng(seed)
    low = np.array([workspace.x_range[0], workspace.y_range[0]])
    high = np.array([workspace.x_range[1], workspace.y_range[1]])
    bowl = rng.uniform(low, high, (samples, 2))
    plate = rng.uniform(low, high, (samples, 2))
    ok = np.linalg.norm(bowl - plate, axis=1) >= bowl_radius_m + plate_radius_m + gap_m
    for x_min, x_max, y_min, y_max in footprints.values():
        for centre, radius in ((bowl, bowl_radius_m), (plate, plate_radius_m)):
            nearest = np.column_stack(
                [np.clip(centre[:, 0], x_min, x_max), np.clip(centre[:, 1], y_min, y_max)]
            )
            ok &= np.linalg.norm(centre - nearest, axis=1) >= radius + clearance_m
    return float(ok.mean())


def render_unscaled_vs_scaled(
    asset_sets: dict[str, list],
    setup: RealSetup,
    size: int = 320,
    camera_names: tuple[str, ...] = ("agentview", "frontview"),
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Each splat set drawn on the stock LIBERO bodies and on bodies scaled to the real measurements.

    The splats are fitted with `MATCH_BODY_XY`, so a bigger body gives a bigger splat: this shows the
    object size the policy would see before and after scaling. The scaled scene uses `ScaledNativeScene`
    (distractors parked, our layout); the stock one uses LIBERO's own initial state, so object positions
    differ between the two rows by design.

    Args:
        asset_sets: label -> splats (see `hurobo.splat.build_assets`).
        setup: the real measurements.
        size: square size of one camera image in pixels.
        camera_names: cameras joined left to right.
        seed: layout seed of the scaled scene.

    Returns:
        "<label> | stock" and "<label> | scaled" -> (size, size * len(camera_names), 3) uint8 composite.
    """
    from hurobo.splat import LIBERO_GOAL_DISTRACTOR_ROOTS, SplatCompositor, render_asset_composites

    images = {
        f"{label} | stock LIBERO size": image
        for label, image in render_asset_composites(asset_sets, size=size, camera_names=camera_names).items()
    }
    scene = ScaledNativeScene(setup, seed=seed)
    scene.reset()
    try:
        for label, assets in asset_sets.items():
            compositor = SplatCompositor(
                scene.task.mj_model,
                assets,
                hidden_body_roots=LIBERO_GOAL_DISTRACTOR_ROOTS,
                width=size,
                height=size,
            )
            try:
                views = [
                    compositor.render_layers(scene.task.mj_data, name)["composite"] for name in camera_names
                ]
                images[f"{label} | scaled to measurements"] = np.concatenate(views, axis=1)
            finally:
                compositor.close()
    finally:
        scene.task.close()
    return images
