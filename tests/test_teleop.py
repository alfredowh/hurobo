"""Teleoperation logic: frame mapping, command sources, keyboard, workspace limits, retiming."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hurobo.control import Waypoint, WaypointFollower
from hurobo.demo import HumanDemo
from hurobo.teleop import (
    ApproachCommandSource,
    HandCommandSource,
    KeyboardCommandSource,
    ScriptedKeys,
    SequenceCommandSource,
    TableToWorld,
    WorkspaceLimits,
    yaw_aligning_views,
)
from hurobo.utils.transforms import apply_transform, invert_transform, look_at_origin_pose, make_transform
from hurobo.utils.video import (
    FrameReel,
    VideoWriter,
    dense_flow,
    interpolate_frames,
    read_frames,
    slow_down_video,
)


def make_demo(num: int = 21, closed_from: int = 10) -> HumanDemo:
    """A straight 20 cm move along table x over 1 s, closing the gripper halfway."""
    t = np.linspace(0.0, 1.0, num)
    tcp = np.column_stack([0.2 * t, np.zeros(num), 0.05 + 0.1 * t])
    return HumanDemo(
        timestamps_s=t,
        tcp_pos_table=tcp,
        pinch_aperture_m=np.full(num, 0.05),
        is_grasping=np.arange(num) >= closed_from,
        t_table_bowl=np.eye(4),
        t_table_plate=np.eye(4),
        t_table_cam=np.eye(4),
        source_video=Path("none.mp4"),
        source_start_time_s=2.0,
    )


def test_anchor_maps_to_anchor_and_deltas_follow_yaw_and_scale() -> None:
    mapping = TableToWorld(np.array([0.1, 0.2, 0.3]), np.array([-0.2, 0.0, 1.1]), np.pi / 2, scale=2.0)
    assert np.allclose(mapping.point_to_world(np.array([0.1, 0.2, 0.3])), [-0.2, 0.0, 1.1])
    # +0.1 m along table x, rotated by 90 deg about z -> +y, doubled.
    assert np.allclose(mapping.point_to_world(np.array([0.2, 0.2, 0.3])), [-0.2, 0.2, 1.1])
    points = np.random.default_rng(0).normal(size=(5, 3))
    assert np.allclose(mapping.point_to_table(mapping.point_to_world(points)), points)


def test_grounded_mapping_puts_table_plane_on_sim_table() -> None:
    mapping = TableToWorld.grounded(np.array([0.1, 0.2, 0.12]), np.array([-0.2, 0.0, 1.1]))
    assert mapping.p_world_anchor[2] == pytest.approx(0.9 + 0.12)
    assert np.allclose(mapping.p_world_anchor[:2], [-0.2, 0.0])
    assert mapping.point_to_world(np.array([0.1, 0.2, 0.0]))[2] == pytest.approx(0.9)


def test_camera_view_is_preserved_by_the_mapping() -> None:
    """A table point seen by the human camera and its world image seen by the mapped camera agree."""
    t_table_cam = look_at_origin_pose(np.array([0.1, -0.45, 0.4]))
    mapping = TableToWorld(np.array([0.0, 0.0, 0.1]), np.array([-0.3, 0.1, 1.0]), 0.7, scale=1.5)
    t_world_cam = mapping.pose_to_world(t_table_cam)
    p_table = np.array([0.05, 0.03, 0.02])
    in_cam_from_table = apply_transform(invert_transform(t_table_cam), p_table)
    in_cam_from_world = apply_transform(invert_transform(t_world_cam), mapping.point_to_world(p_table))
    # Uniform scaling of the whole scene scales camera-frame coordinates by `scale`; directions match.
    assert np.allclose(in_cam_from_world, 1.5 * in_cam_from_table)


def test_yaw_aligns_ground_headings_and_wraps() -> None:
    t_table_cam = look_at_origin_pose(np.array([0.0, -0.45, 0.4]))  # looks along +y (and down)
    # A MuJoCo camera looking along world -x: its -z axis must point to -x.
    r_world_cam = Rotation.from_matrix(np.column_stack([[0, 1, 0], [0, 0, 1], [1, 0, 0]])).as_matrix()
    t_world_cam_mj = make_transform(Rotation.from_matrix(r_world_cam), np.zeros(3))
    assert np.allclose(-r_world_cam[:, 2], [-1, 0, 0])
    yaw = yaw_aligning_views(t_table_cam, t_world_cam_mj)
    assert yaw == pytest.approx(np.pi / 2)  # +y heading is turned +90 deg onto -x
    heading = Rotation.from_euler("z", yaw).apply([0.0, 1.0, 0.0])
    assert np.allclose(heading, [-1, 0, 0], atol=1e-9)
    assert -np.pi <= yaw <= np.pi


def test_hand_source_replays_trajectory_and_gripper_from_demo() -> None:
    demo = make_demo()
    mapping = TableToWorld(demo.tcp_pos_table[0], np.array([0.0, 0.0, 1.0]))
    source = HandCommandSource(demo, mapping, control_hz=20.0, slowdown=1.0, hold_end_s=0.0)
    commands = []
    while (command := source.next_command(np.zeros(3))) is not None:
        commands.append(command)
    assert len(commands) == source.num_steps == 21
    assert np.allclose(commands[0].pos_world, [0, 0, 1.0])
    assert np.allclose(commands[-1].pos_world, [0.2, 0, 1.1])
    assert not commands[0].gripper_closed and commands[-1].gripper_closed
    assert commands[0].source_time_s == pytest.approx(2.0)  # offset into the video, not the demo


def test_slowdown_stretches_duration_but_not_path() -> None:
    demo = make_demo()
    mapping = TableToWorld(demo.tcp_pos_table[0], np.zeros(3))

    def run(slowdown: float) -> list:
        source = HandCommandSource(demo, mapping, slowdown=slowdown, hold_end_s=0.0)
        out = []
        while (c := source.next_command(np.zeros(3))) is not None:
            out.append(c)
        return out

    normal, slow = run(1.0), run(4.0)
    assert len(slow) == pytest.approx(4 * (len(normal) - 1) + 1, abs=1)
    assert np.allclose(normal[-1].pos_world, slow[-1].pos_world)
    with pytest.raises(ValueError):
        HandCommandSource(demo, mapping, slowdown=0.0)


def test_sequence_runs_approach_then_hand_and_ends() -> None:
    demo = make_demo()
    mapping = TableToWorld(demo.tcp_pos_table[0], np.array([0.0, 0.0, 1.0]))
    eef = np.array([0.0, 0.0, 1.3])
    approach = ApproachCommandSource(mapping.p_world_anchor, tolerance_m=0.01, max_steps=5)
    source = SequenceCommandSource([approach, HandCommandSource(demo, mapping, hold_end_s=0.0)])
    first = source.next_command(eef)
    assert first is not None and first.source_time_s is None  # still approaching
    near = np.array([0.0, 0.0, 1.0])
    second = source.next_command(near)  # arrived: hand source takes over
    assert second is not None and second.source_time_s == pytest.approx(2.0)


def test_keyboard_moves_with_held_keys_toggles_gripper_and_quits() -> None:
    keys = ScriptedKeys(holds=[(0, 9, "right")], presses=[(5, "space"), (12, "escape")])
    source = KeyboardCommandSource(keys, np.zeros(3), control_hz=20.0, speed_mps=0.2, max_lead_m=1.0)
    eef = np.zeros(3)
    commands = []
    for _ in range(12):
        commands.append(source.next_command(eef))
    # 10 steps of 0.2 m/s at 20 Hz = 0.1 m along world +y (right in the agentview).
    assert np.allclose(commands[9].pos_world, [0.0, 0.1, 0.0])
    assert np.allclose(commands[11].pos_world, commands[9].pos_world)
    assert not commands[4].gripper_closed and commands[5].gripper_closed
    assert source.next_command(eef) is None  # escape pressed at step 12


def test_keyboard_target_cannot_wind_up_past_the_lead_limit() -> None:
    keys = ScriptedKeys(holds=[(0, 100, "w")])
    source = KeyboardCommandSource(keys, np.zeros(3), speed_mps=1.0, max_lead_m=0.03)
    for _ in range(50):
        command = source.next_command(np.zeros(3))  # arm never moves
    assert command.pos_world[2] == pytest.approx(0.03)


def test_workspace_limits_clamp_and_report() -> None:
    limits = WorkspaceLimits.around(np.array([-0.2, 0.0, 1.1]), half_extent_xy_m=0.1, z_above_m=0.2)
    inside, changed = limits.clamp(np.array([-0.2, 0.05, 1.0]))
    assert not changed and np.allclose(inside, [-0.2, 0.05, 1.0])
    outside, changed = limits.clamp(np.array([0.0, 0.0, 0.5]))
    assert changed and outside[0] == pytest.approx(-0.1) and outside[2] > 0.9  # above the table top


def test_action_toward_points_at_target_and_respects_limits() -> None:
    follower = WaypointFollower(max_pos_delta_m=0.05, max_rot_delta_rad=0.5)
    here, rot = np.zeros(3), Rotation.identity()
    target = Waypoint(np.array([0.01, -0.2, 0.0]), rot, gripper_closed=True)
    action = follower.action_toward(target, here, rot)
    assert action.shape == (7,)
    assert action[0] == pytest.approx(0.2) and action[1] == pytest.approx(-1.0)  # 1 cm of 5 cm; clipped
    assert action[6] == 1.0
    gentle = follower.action_toward(target, here, rot, action_limit=0.3)
    assert gentle[1] == pytest.approx(-0.3)


def write_moving_square_clip(path: Path, num: int = 5, fps: float = 10.0) -> None:
    with VideoWriter(path, fps, (64, 48)) as writer:
        for k in range(num):
            frame = np.zeros((48, 64, 3), np.uint8)
            frame[20:28, 4 + 10 * k : 12 + 10 * k] = 255
            writer.write(frame)


def test_slow_down_keeps_source_and_stretches_duration(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    write_moving_square_clip(source)
    before = source.read_bytes()
    slowed = slow_down_video(source, tmp_path / "clip_slow.mp4", factor=3.0, output_fps=20.0)
    assert source.read_bytes() == before
    original, stretched = list(read_frames(source)), list(read_frames(slowed))
    assert len(original) == 5
    assert len(stretched) == pytest.approx(3.0 * (len(original) - 1) / 10.0 * 20.0 + 1, abs=2)
    with pytest.raises(ValueError):
        slow_down_video(source, source, 2.0)


def test_flow_interpolation_beats_cross_fade_on_a_moving_texture() -> None:
    rng = np.random.default_rng(1)
    texture = cv2.GaussianBlur(rng.integers(0, 255, (24, 24, 3), dtype=np.uint8), (3, 3), 0)

    def frame_with_patch_at(x: int) -> np.ndarray:
        frame = np.full((64, 96, 3), 40, np.uint8)
        frame[20:44, x : x + 24] = texture
        return frame

    first, last, truth = frame_with_patch_at(20), frame_with_patch_at(36), frame_with_patch_at(28)
    interpolated = interpolate_frames(first, last, 0.5, dense_flow(first, last), dense_flow(last, first))
    cross_fade = cv2.addWeighted(first, 0.5, last, 0.5, 0)

    def mse(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.mean((a.astype(float) - b.astype(float)) ** 2))

    assert mse(interpolated, truth) < 0.5 * mse(cross_fade, truth)


def test_frame_reel_clamps_and_returns_real_frames_at_the_ends(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    write_moving_square_clip(source)
    reel = FrameReel.from_video(source)
    assert np.array_equal(reel.frame_at(-5.0), reel.frame_at(reel.start_s))
    assert np.array_equal(reel.frame_at(1e6), reel.frame_at(reel.start_s + reel.duration_s))


def has_libero() -> bool:
    return (Path.home() / ".libero" / "config.yaml").exists()


@pytest.mark.skipif(not has_libero(), reason="LIBERO is not configured")
def test_cube_prop_is_inert_until_spawned_then_held_by_gravcomp() -> None:
    from hurobo.props import CubeProp
    from hurobo.sim import LiberoTask

    task = LiberoTask("libero_goal", 8)
    try:
        task.reset()
        cube = CubeProp()
        cube.install(task)
        assert not cube.spawned
        zero = np.zeros(7, dtype=np.float32)
        for _ in range(5):
            task.step(zero)
        assert cube.position(task)[2] == pytest.approx(1.5, abs=1e-3)  # parked, weightless

        cube.spawn(task, np.array([0.0, 0.0, 1.3]), yaw_rad=0.0)
        for _ in range(5):
            task.step(zero)
        assert cube.position(task)[2] == pytest.approx(1.3, abs=5e-3)  # supported, not falling

        cube.release_support(task)
        for _ in range(40):
            task.step(zero)
        assert cube.position(task)[2] < 1.0  # now it falls to the table
    finally:
        task.close()
