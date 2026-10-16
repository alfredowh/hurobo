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


def test_slowdown_for_scales_a_demo_to_a_speed_limit() -> None:
    demo = _demo()
    demo.tcp_pos_table = np.column_stack([np.arange(10) * 0.05, np.zeros(10), np.zeros(10)])  # 1 m/s
    assert demo.slowdown_for(0.25) == 4.0
    assert demo.slowdown_for(5.0) == 1.0
    assert demo.time_scaled(4.0).timestamps_s[-1] == 4.0 * demo.timestamps_s[-1]


def test_annotate_sample_draws_on_a_downscaled_frame_with_the_matching_clip_frame() -> None:
    from types import SimpleNamespace

    drawn: list[tuple[int, float]] = []
    diagnosis = SimpleNamespace(
        times_s=2.0 + np.arange(20) * 0.025,
        draw_detections=lambda _frame, index, scale: drawn.append((index, scale)),
    )
    overlay = _overlay()
    overlay._diagnosis = diagnosis  # type: ignore[assignment]
    frame = np.zeros((240, 320, 3), np.uint8)

    overlay.annotate_sample(frame, 4)  # sample 4 is at 2.0 + 0.2 s of the clip = clip frame 8

    assert drawn == [(8, 0.5)]
    assert frame.any()


def test_overlay_video_is_slowed_and_held_by_repeating_frames(tmp_path: Path) -> None:
    from hurobo.camera import CameraIntrinsics
    from hurobo.demo import HumanDemo
    from hurobo.utils.video import VideoWriter, read_frames

    source = tmp_path / "clip.mp4"
    with VideoWriter(source, 10.0, (64, 48)) as writer:
        for _ in range(20):
            writer.write(np.zeros((48, 64, 3), np.uint8))
    demo = _demo()
    demo = HumanDemo(**{**vars(demo), "source_video": source, "source_start_time_s": 0.0})
    demo.timestamps_s = np.arange(10) * 0.1
    intrinsics = CameraIntrinsics(np.array([[500.0, 0, 32], [0, 500.0, 24], [0, 0, 1]]), np.zeros(5), 64, 48)

    class _NoDetections:
        def draw_detections(self, *_args: object) -> None:
            return None

    overlay = DemoOverlay(demo, _NoDetections(), intrinsics)  # type: ignore[arg-type]
    out = overlay.write_video(tmp_path / "out.mp4", width_px=64, slowdown=2.0, hold_start_s=1.0, fps=10.0)
    # 1 s held + 0.9 s of motion x2 = 2.8 s at 10 fps
    assert sum(1 for _ in read_frames(out)) == 28
