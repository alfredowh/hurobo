"""Converts Cartesian waypoints into LIBERO's clipped relative end-effector actions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class Waypoint:
    """One target pose for the gripper to reach, in the frame the rollout runs in.

    Attributes:
        pos: shape (3,), target end-effector position.
        rot: scipy Rotation, target end-effector orientation. The retargeter
            (`retarget.py`) keeps this fixed top-down with yaw from the object
            (PLAN.md §4 step 6), but `WaypointFollower` itself is agnostic to that.
        gripper_closed: target gripper state at this waypoint.
    """

    pos: np.ndarray
    rot: Rotation
    gripper_closed: bool


class WaypointFollower:
    """Closed-loop controller: at each step, moves toward the current waypoint
    by a clipped relative delta, so tracking error doesn't compound (PLAN.md §1 fix 5).

    LIBERO's action space is a 7-D vector: 3 position delta + 3 axis-angle
    orientation delta + 1 gripper command, each component clipped to [-1, 1]
    and interpreted relative to the *current* simulated end-effector pose, not
    an absolute target. Computing the delta from the sim's own current pose
    (rather than open-loop from the plan) is what makes this robust to
    retargeting/tracking error.
    """

    def __init__(
        self,
        max_pos_delta_m: float = 0.02,
        max_rot_delta_rad: float = 0.15,
        pos_reach_threshold_m: float = 0.005,
        rot_reach_threshold_rad: float = 0.05,
    ) -> None:
        """Set the per-step motion limits and the "waypoint reached" thresholds.

        Args:
            max_pos_delta_m: largest position step commanded per control step,
                before the [-1, 1] normalization LIBERO expects. Tune down if
                the arm overshoots or oscillates around waypoints.
            max_rot_delta_rad: largest orientation step per control step.
            pos_reach_threshold_m: distance below which a waypoint counts as
                reached and the follower advances to the next one.
            rot_reach_threshold_rad: angular distance below which a waypoint's
                orientation counts as reached.
        """
        self._max_pos_delta_m = max_pos_delta_m
        self._max_rot_delta_rad = max_rot_delta_rad
        self._pos_reach_threshold_m = pos_reach_threshold_m
        self._rot_reach_threshold_rad = rot_reach_threshold_rad
        self._waypoints: list[Waypoint] = []
        self._current_idx = 0

    def set_waypoints(self, waypoints: list[Waypoint]) -> None:
        """Load a new waypoint sequence and reset progress to the first one."""
        self._waypoints = waypoints
        self._current_idx = 0

    @property
    def done(self) -> bool:
        """True once every waypoint has been reached."""
        return self._current_idx >= len(self._waypoints)

    def step(self, current_pos: np.ndarray, current_rot: Rotation) -> np.ndarray:
        """Compute one 7-D clipped relative action toward the current waypoint.

        Args:
            current_pos: shape (3,), the sim end-effector's current position.
            current_rot: scipy Rotation, the sim end-effector's current orientation.

        Returns:
            shape (7,) float array in [-1, 1]: (dx, dy, dz, drx, dry, drz, gripper),
            where the gripper component is +1 (close) or -1 (open), matching
            LIBERO/robosuite's OSC_POSE delta-pose convention.
        """
        if self.done:
            return np.zeros(7, dtype=np.float32)

        target = self._waypoints[self._current_idx]
        pos_error = target.pos - current_pos
        rot_error = (target.rot * current_rot.inv()).as_rotvec()

        pos_dist = float(np.linalg.norm(pos_error))
        rot_dist = float(np.linalg.norm(rot_error))
        if pos_dist < self._pos_reach_threshold_m and rot_dist < self._rot_reach_threshold_rad:
            self._current_idx += 1
            if self.done:
                return np.zeros(7, dtype=np.float32)
            target = self._waypoints[self._current_idx]
            pos_error = target.pos - current_pos
            rot_error = (target.rot * current_rot.inv()).as_rotvec()

        pos_delta = np.clip(pos_error, -self._max_pos_delta_m, self._max_pos_delta_m)
        rot_delta = np.clip(rot_error, -self._max_rot_delta_rad, self._max_rot_delta_rad)
        pos_action = pos_delta / self._max_pos_delta_m
        rot_action = rot_delta / self._max_rot_delta_rad
        gripper_action = np.array([1.0 if target.gripper_closed else -1.0], dtype=np.float32)

        return np.concatenate([pos_action, rot_action, gripper_action]).astype(np.float32)
