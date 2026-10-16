"""Demo overlay: sample lookup, phases and the gripper gauge."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hurobo.demo import HumanDemo
from hurobo.demo_overlay import DemoOverlay
from hurobo.utils.video import draw_gripper_gauge


def _demo() -> HumanDemo:
    n = 10
    return HumanDemo(
        timestamps_s=np.arange(n) * 0.05,
        tcp_pos_table=np.tile([0.5, -0.2, 0.1], (n, 1)),
        pinch_aperture_m=np.full(n, 0.08),
        is_grasping=np.zeros(n, dtype=bool),
        t_table_bowl=np.eye(4),
        t_table_plate=np.eye(4),
        t_table_cam=np.diag([1.0, -1.0, -1.0, 1.0])
        + np.array([[0, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 1.0], [0, 0, 0, 0]]),
        source_video=Path("none.mp4"),
        grasp_frame_idx=3,
        release_frame_idx=7,
        source_start_time_s=2.0,
    )


def _overlay() -> DemoOverlay:
    from hurobo.camera import CameraIntrinsics

    intrinsics = CameraIntrinsics(
        camera_matrix=np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]]),
        dist_coeffs=np.zeros(5),
        image_width=640,
        image_height=480,
    )
    return DemoOverlay(_demo(), diagnosis=None, intrinsics=intrinsics)  # type: ignore[arg-type]


def test_sample_at_maps_clip_time_to_the_nearest_demo_sample() -> None:
    overlay = _overlay()
    assert overlay.sample_at(2.0) == 0
    assert overlay.sample_at(2.21) == 4


def test_phase_follows_grasp_and_release() -> None:
    overlay = _overlay()
    assert [overlay.phase_at(i) for i in (0, 3, 6, 7, 9)] == [
        "approach",
        "carry",
        "carry",
        "retreat",
        "retreat",
    ]


def test_gripper_gauge_spreads_the_fingers_with_the_aperture() -> None:
    narrow, wide = (np.zeros((120, 200, 3), np.uint8) for _ in range(2))
    draw_gripper_gauge(narrow, 0.01, True, (20, 20))
    draw_gripper_gauge(wide, 0.10, False, (20, 20))

    def finger_columns(image: np.ndarray, colour: tuple[int, int, int]) -> np.ndarray:
        return np.flatnonzero((image[30] == colour).all(axis=1))

    assert np.ptp(finger_columns(wide, (0, 200, 0))) > np.ptp(finger_columns(narrow, (0, 0, 230)))


def test_rollout_result_event_times_end_at_the_last_step_when_there_was_no_release() -> None:
    from hurobo.retarget import RolloutResult

    result = RolloutResult(steps=100, phase_start_steps={"approach": 30, "grasp": 40})
    assert result.event_times_s(20.0) == [0.0, 1.5, 2.0, 5.0, 5.0]


def test_phase_aligned_video_plays_on_in_real_time_after_the_last_event(tmp_path: Path) -> None:
    from hurobo.utils.video import VideoWriter, read_frames, write_phase_aligned_video

    def clip(name: str, seconds: float) -> Path:
        path = tmp_path / name
        with VideoWriter(path, 10.0, (64, 48)) as writer:
            for _ in range(int(seconds * 10)):
                writer.write(np.zeros((48, 64, 3), np.uint8))
        return path

    out = write_phase_aligned_video(
        clip("a.mp4", 2.0),
        clip("b.mp4", 4.0),
        tmp_path / "out.mp4",
        ("a", "b"),
        left_events_s=[0.0, 1.0],
        right_events_s=[0.0, 2.0],
        fps=10.0,
        play_on_after_last_event=True,
    )
    # 2 s aligned segment (the right clip needs it) + 1.9 s tail (the right clip's remaining time)
    assert sum(1 for _ in read_frames(out)) == 39


def test_pause_faster_waits_instead_of_slowing_and_the_slowdown_stretches_the_slow_side(
    tmp_path: Path,
) -> None:
    from hurobo.utils.video import VideoWriter, read_frames, write_phase_aligned_video

    def clip(name: str, seconds: float) -> Path:
        path = tmp_path / name
        with VideoWriter(path, 10.0, (64, 48)) as writer:
            for _ in range(int(seconds * 10)):
                writer.write(np.zeros((48, 64, 3), np.uint8))
        return path

    # left waits 1 s (equal events), then 1 s of its video slowed x2 takes 2 s; right needs 1 s then 1 s
    out = write_phase_aligned_video(
        clip("a.mp4", 2.0),
        clip("b.mp4", 2.0),
        tmp_path / "out.mp4",
        ("a", "b"),
        left_events_s=[0.0, 0.0, 1.0],
        right_events_s=[0.0, 1.0, 2.0],
        fps=10.0,
        slowdowns=(2.0, 1.0),
        pause_faster=True,
    )
    assert sum(1 for _ in read_frames(out)) == 30


def test_slowdown_for_scales_a_demo_to_a_speed_limit() -> None:
    demo = _demo()
    demo.tcp_pos_table = np.column_stack([np.arange(10) * 0.05, np.zeros(10), np.zeros(10)])  # 1 m/s
    assert demo.slowdown_for(0.25) == 4.0
    assert demo.slowdown_for(5.0) == 1.0
    assert demo.time_scaled(4.0).timestamps_s[-1] == 4.0 * demo.timestamps_s[-1]
