"""A robot rollout recorded per control step, and its dashboard (image, path, gripper, state, action)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hurobo.retarget import RolloutResult
from hurobo.utils.dashboard_video import SignalPanel, SyncedDashboard, write_dashboard_video

STATE_NAMES = ["eef_x", "eef_y", "eef_z", "gripper_opening"]
ACTION_NAMES = ["a_x", "a_y", "a_z", "gripper_cmd"]


@dataclass
class RolloutTrace:
    """One robot rollout, step by step.

    Attributes:
        result: the outcome (success, steps, phases).
        frames_bgr: one agent-view image per step, with the phase caption and a gripper symbol drawn on it.
        times_s: shape (N,), time of each step.
        eef_pos_world: shape (N, 3), end-effector position in the world frame.
        gripper_opening_m: shape (N,), distance between the two fingers.
        actions: shape (N, 7), what was sent to the env (position and rotation deltas in [-1, 1], then the
            gripper command: > 0 closes).
        phases: name of the phase of each step.
    """

    result: RolloutResult
    frames_bgr: list[np.ndarray]
    times_s: np.ndarray
    eef_pos_world: np.ndarray
    gripper_opening_m: np.ndarray
    actions: np.ndarray
    phases: list[str]

    @property
    def gripper_closed(self) -> np.ndarray:
        """shape (N,) bool, the gripper command was "close"."""
        return self.actions[:, 6] > 0


def rollout_dashboard(trace: RolloutTrace) -> SyncedDashboard:
    """Dashboard of a rollout like the human episode's: image | TCP path | gripper | state | action.

    Args:
        trace: from `DemoGenerator.rollout_traced`.

    Returns:
        The dashboard: agent-view image, end-effector path in the world frame (red while the gripper is
        commanded closed), gripper opening, the state and the action curves, with a shared time cursor.
    """
    state = np.column_stack([trace.eef_pos_world, trace.gripper_opening_m])
    action = np.column_stack([trace.actions[:, :3], trace.gripper_closed.astype(float)])
    return SyncedDashboard(
        trace.frames_bgr,
        trace.times_s,
        trace.eef_pos_world,
        trace.gripper_opening_m,
        image_title="robot agent view",
        path_title="end effector, world frame (m)",
        path_axis_labels=("x", "y", "z up"),
        grasp_mask=trace.gripper_closed,
        signal_panels=[
            SignalPanel("robot state", state, STATE_NAMES, y_label="m (world frame)"),
            SignalPanel(
                "action sent",
                action,
                ACTION_NAMES,
                right_axis_labels=["gripper_cmd"],
                y_label="delta (normalised)",
                right_y_label="gripper closed",
            ),
        ],
        aperture_title="gripper opening",
        aperture_label="opening (cm)",
    )


def write_rollout_dashboard_video(trace: RolloutTrace, video_path: Path, fps: float) -> Path:
    """Video of a rollout's dashboard, played at `fps` (the control rate plays it in real time)."""
    return write_dashboard_video(rollout_dashboard(trace), video_path, fps)
