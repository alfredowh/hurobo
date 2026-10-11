"""Thin wrapper around the LeRobot LIBERO env: object poses, success check, render.

Built against `lerobot.envs.libero.create_libero_envs` and the underlying
robosuite `MjSim` it exposes via `LiberoEnv._env.sim` (verified interactively
against the installed packages, not guessed — see REAL2SIM_IDEAS.md for why
that mattered: a previous `uv sync` silently pulled an incompatible torch build,
and the same "verify, don't guess" discipline applies to this API).
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation

# Two different quaternion conventions are in play here, confirmed against the
# installed robosuite source rather than assumed (see PLAN.md's "frames are
# explicit" rule): free-joint `qpos` (bowl/plate) is raw MuJoCo, scalar-first
# (w, x, y, z). The `robot0_eef_quat` *observable* is explicitly converted to
# scalar-last (x, y, z, w) by robosuite itself (robots/single_arm.py:
# `T.convert_quat(..., to="xyzw")`) — i.e. already scipy's native order.
_WXYZ_TO_XYZW = [1, 2, 3, 0]


def _wxyz_to_scipy_rotation(quat_wxyz: np.ndarray) -> Rotation:
    """Convert a mujoco-native (w, x, y, z) quaternion (e.g. object qpos) to a scipy Rotation."""
    return Rotation.from_quat(np.asarray(quat_wxyz)[_WXYZ_TO_XYZW])


def _scipy_rotation_to_wxyz(rotation: Rotation) -> np.ndarray:
    """Convert a scipy Rotation to a mujoco-convention (w, x, y, z) quaternion."""
    x, y, z, w = rotation.as_quat()
    return np.array([w, x, y, z])


@dataclass
class ObjectPose:
    """A free-jointed object's pose, as read from / written to `qpos`.

    Attributes:
        pos: shape (3,), world-frame position.
        rot: scipy Rotation, world-frame orientation.
    """

    pos: np.ndarray
    rot: Rotation


class LiberoTask:
    """Wraps one LIBERO task (one suite + task id) for retargeting and generation.

    One instance owns one underlying env; call `reset` before each rollout.
    Not vectorized on purpose — `generate.py`'s `DemoGenerator` runs many of
    these in parallel worker processes instead, which keeps this class simple.
    """

    #: Reused for the real2sim twin render (`twin.py`, teleoperation). The policy observation
    #: only uses "agentview"/"robot0_eye_in_hand", and "galleryview" (the whole-room view) is
    #: kept stock for display, so the near-duplicate of the agentview is the one to repoint.
    TWIN_CAMERA_NAME = "frontview"

    def __init__(
        self,
        suite_name: str,
        task_id: int,
        camera_name: str = "agentview_image,robot0_eye_in_hand_image",
        observation_size: int = 256,
    ) -> None:
        """Build the vectorized (n_envs=1) LIBERO env for one task.

        Args:
            suite_name: a LIBERO suite, e.g. "libero_goal".
            task_id: index into that suite, e.g. 8 for "put the bowl on the plate".
            camera_name: comma-separated camera names for the policy observation,
                matching the convention `lerobot-eval` expects.
            observation_size: square pixel size for the observation cameras.
        """
        from lerobot.envs.libero import create_libero_envs

        envs = create_libero_envs(
            task=suite_name,
            n_envs=1,
            gym_kwargs={
                "task_ids": [task_id],
                "obs_type": "pixels_agent_pos",
                "observation_width": observation_size,
                "observation_height": observation_size,
            },
            camera_name=camera_name,
            env_cls=gym.vector.SyncVectorEnv,
        )
        self._vec_env = envs[suite_name][task_id]
        self._libero_env = self._vec_env.envs[0]  # the single underlying LiberoEnv
        self.task_description = None

    def reset(self) -> dict:
        """Reset the episode and return the first observation.

        Returns:
            The LeRobot-style obs dict (keys: "pixels", "robot_state"), with the
            leading n_envs=1 batch axis still present (index [0] to drop it).
        """
        obs, _info = self._vec_env.reset()
        self._libero_env._ensure_env()
        self.task_description = self._libero_env.task_description
        return obs

    def step(self, action: np.ndarray) -> tuple[dict, bool, bool]:
        """Step the env with a 7-D relative action.

        Args:
            action: shape (7,), clipped to [-1, 1]: 3 position delta + 3
                axis-angle delta + 1 gripper command (see `control.py`).

        Returns:
            (obs, is_success, terminated).
        """
        obs, _reward, terminated, _truncated, info = self._vec_env.step(action[None, :])
        return obs, bool(info["is_success"][0]), bool(terminated[0])

    @property
    def sim(self):
        """The underlying robosuite `MjSim` (model + data), for direct mujoco access."""
        return self._libero_env._env.sim

    @property
    def mj_model(self):
        """The raw `mujoco.MjModel`, as `SplatCompositor` and `mujoco.Renderer` expect."""
        return self.sim.model._model

    @property
    def mj_data(self):
        """The raw `mujoco.MjData` of the current episode."""
        return self.sim.data._data

    @property
    def controller_action_scale(self) -> tuple[float, float]:
        """The OSC_POSE controller's per-step (position_m, rotation_rad) output scale.

        robosuite's `scale_action` (base_controller.py) linearly maps an action
        component in [-1, 1] to `[output_min, output_max]` *before* adding it to
        the current pose (see `osc_pose.json`: output_max = (0.05, 0.05, 0.05,
        0.5, 0.5, 0.5)). `WaypointFollower.step` (control.py) must divide its
        desired metric delta by this real scale, not by its own clip bound —
        using a smaller constant there would silently command a larger motion
        than intended every step (verified against the installed robosuite
        source, not assumed; see PLAN.md §1 fix 6 / §2 "frames are explicit").
        """
        controller = self._libero_env._env.robots[0].controller
        output_max = np.asarray(controller.output_max, dtype=np.float64)
        return float(output_max[0]), float(output_max[3])

    def get_eef_pose(self, obs: dict) -> tuple[np.ndarray, Rotation]:
        """Extract the current end-effector pose from an observation dict.

        Args:
            obs: an observation as returned by `reset`/`step`.

        Returns:
            (pos, rot): pos has shape (3,); rot is a scipy Rotation.
        """
        pos = np.asarray(obs["robot_state"]["eef"]["pos"][0])
        # robosuite's eef_quat observable is already (x, y, z, w) — scipy's native
        # order — unlike object qpos below, which is raw MuJoCo (w, x, y, z).
        quat_xyzw = np.asarray(obs["robot_state"]["eef"]["quat"][0])
        return pos, Rotation.from_quat(quat_xyzw)

    def get_object_pose(self, joint_name: str) -> ObjectPose:
        """Read a free-jointed object's current world pose from the sim.

        Args:
            joint_name: e.g. "akita_black_bowl_1_joint0" or "plate_1_joint0"
                (`obj_of_interest` on the underlying env lists the task's
                objects; append "_joint0" for the standard LIBERO naming).

        Returns:
            The object's current `ObjectPose`.
        """
        qpos = self.sim.data.get_joint_qpos(joint_name)
        return ObjectPose(pos=np.array(qpos[:3]), rot=_wxyz_to_scipy_rotation(qpos[3:7]))

    def set_object_pose(self, joint_name: str, pose: ObjectPose) -> None:
        """Overwrite a free-jointed object's world pose (position + orientation).

        Used by `SceneTwin` (`twin.py`) to match a real recorded layout, and by
        `DemoGenerator` (`generate.py`) to sample new layouts for generation.
        Caller is responsible for keeping the pose physically sensible (e.g.
        resting height) — this only writes qpos, it doesn't validate it.

        Args:
            joint_name: see `get_object_pose`.
            pose: the new pose to write.
        """
        qpos = np.concatenate([pose.pos, _scipy_rotation_to_wxyz(pose.rot)])
        self.sim.data.set_joint_qpos(joint_name, qpos)
        self.sim.forward()

    def translate_object(self, joint_name: str, delta_world: np.ndarray) -> None:
        """Shift a free-jointed object by `delta_world` metres, keeping its orientation.

        Args:
            joint_name: see `get_object_pose`.
            delta_world: shape (3,), displacement in the world frame.
        """
        pose = self.get_object_pose(joint_name)
        self.set_object_pose(joint_name, ObjectPose(pose.pos + delta_world, pose.rot))

    def render_camera(self, camera_name: str, width: int = 480, height: int = 480) -> np.ndarray:
        """Render an arbitrary named camera (e.g. the twin camera) off-screen.

        Args:
            camera_name: a camera defined in the task's MJCF, e.g. `TWIN_CAMERA_NAME`.
            width: output image width in pixels.
            height: output image height in pixels.

        Returns:
            shape (height, width, 3) uint8 RGB image, upright (robosuite returns GL
            bottom-up images; this flips rows, unlike the 180-degree-rotated policy
            observations the LeRobot wrapper produces).
        """
        image = self.sim.render(camera_name=camera_name, width=width, height=height)
        return np.ascontiguousarray(np.asarray(image)[::-1])

    def set_camera_pose(
        self, camera_name: str, pos: np.ndarray, quat_wxyz: np.ndarray, fovy_degrees: float
    ) -> None:
        """Reposition a named camera at runtime (no XML recompile needed).

        Verified interactively: `mjModel.cam_pos`/`cam_quat`/`cam_fovy` are
        writable numpy views, and a render taken right after an assignment
        reflects the new pose immediately.

        Args:
            camera_name: an existing camera name, e.g. `TWIN_CAMERA_NAME`.
            pos: shape (3,), world-frame camera position.
            quat_wxyz: shape (4,), world-frame camera orientation, (w, x, y, z),
                already in MuJoCo's own camera-axis convention (apply
                `utils.transforms.cv_camera_pose_to_mujoco` first if the pose
                came from OpenCV).
            fovy_degrees: vertical field of view in degrees.
        """
        cam_idx = list(self.sim.model.camera_names).index(camera_name)
        self.sim.model.cam_pos[cam_idx] = pos
        self.sim.model.cam_quat[cam_idx] = quat_wxyz
        self.sim.model.cam_fovy[cam_idx] = fovy_degrees
        # Without this, the next render uses the camera pose baked in at the
        # last forward() call (e.g. from set_object_pose), not this one —
        # confirmed by hand: omitting it rendered from the camera's old
        # default pose instead of the one just assigned here.
        self.sim.forward()

    def camera_pose_world(self, camera_name: str) -> np.ndarray:
        """4x4 pose of a named camera in the world frame, MuJoCo camera axes (looks down -z).

        Args:
            camera_name: an existing camera name.
        """
        cam_idx = list(self.sim.model.camera_names).index(camera_name)
        pose = np.eye(4)
        pose[:3, :3] = np.array(self.mj_data.cam_xmat[cam_idx]).reshape(3, 3)
        pose[:3, 3] = self.mj_data.cam_xpos[cam_idx]
        return pose

    @property
    def model_xml(self) -> str:
        """The current scene as MJCF text, for editing and `reload_model`."""
        return self.sim.model.get_xml()

    def reload_model(self, xml: str) -> None:
        """Rebuild the sim from edited MJCF while keeping the current joint state of shared joints.

        robosuite re-creates the sim and re-samples placements on reload, so the joint
        positions (robot, objects) are saved by name and written back afterwards. The edit
        lasts until the next `reset`, which rebuilds the stock task.

        Args:
            xml: scene MJCF, normally `model_xml` plus additions.
        """
        model = self.mj_model
        saved = {
            model.joint(j).name: self.mj_data.qpos[
                model.jnt_qposadr[j] : model.jnt_qposadr[j] + (7 if model.jnt_type[j] == 0 else 1)
            ].copy()
            for j in range(model.njnt)
        }
        self._libero_env._env.env.reset_from_xml_string(xml)
        new_model = self.mj_model
        names = {new_model.joint(j).name for j in range(new_model.njnt)}
        for name, qpos in saved.items():
            if name in names:
                self.sim.data.set_joint_qpos(name, qpos)
        self.sim.forward()

    def finger_opening_m(self, obs: dict) -> float:
        """Distance between the two gripper fingertips' joint positions, summed (0 closed, ~0.08 open).

        Args:
            obs: an observation as returned by `reset`/`step`.
        """
        fingers = np.asarray(obs["robot_state"]["gripper"]["qpos"][0], dtype=float)
        return float(abs(fingers[0]) + abs(fingers[1]))

    def close(self) -> None:
        """Close the underlying vector env."""
        self._vec_env.close()
