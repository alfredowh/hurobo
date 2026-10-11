"""Round trip: HumanDemo -> LeRobotDataset on disk -> reload and compare against the source."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from scipy.spatial.transform import Rotation

from ego2robo.demo import HumanDemo
from ego2robo.human_dataset import (
    HumanDemoExporter,
    human_features,
    replay_tcp_from_action,
    sample_nearest_frames,
)
from ego2robo.recorder import LeRobotRecorder
from ego2robo.utils.transforms import make_transform
from ego2robo.utils.video import VideoWriter

SOURCE_FPS = 30.0
IMAGE_SIZE_HW = (96, 128)


BAR_PX_PER_FRAME = 2
BAR_WIDTH_PX = 6
SOURCE_WIDTH_PX = 160


def _write_clip(path: Path, num_frames: int) -> None:
    """A clip whose frame i has a white bar at x = 2*i, so frame identity survives lossy coding."""
    with VideoWriter(path, SOURCE_FPS, (SOURCE_WIDTH_PX, 120)) as writer:
        for i in range(num_frames):
            frame = np.zeros((120, SOURCE_WIDTH_PX, 3), dtype=np.uint8)
            frame[:, BAR_PX_PER_FRAME * i : BAR_PX_PER_FRAME * i + BAR_WIDTH_PX] = 255
            writer.write(frame)


def _frame_index(image_gray_columns: np.ndarray, scale: float = 1.0) -> int:
    """Recover the source frame index from a per-column brightness profile (see `_write_clip`)."""
    bright = np.flatnonzero(image_gray_columns > 0.5 * image_gray_columns.max())
    left_edge_px = bright.mean() / scale - (BAR_WIDTH_PX - 1) / 2
    return int(round(left_edge_px / BAR_PX_PER_FRAME))


def _demo(video: Path, start_time_s: float, num_samples: int) -> HumanDemo:
    t = np.arange(num_samples) / 20.0
    return HumanDemo(
        timestamps_s=t,
        tcp_pos_table=np.column_stack([0.01 * np.arange(num_samples), np.zeros(num_samples), 0.1 + 0 * t]),
        pinch_aperture_m=np.full(num_samples, 0.06),
        is_grasping=(np.arange(num_samples) >= num_samples // 2),
        t_table_bowl=make_transform(Rotation.from_euler("z", 30, degrees=True), [0.1, 0.0, 0.0]),
        t_table_plate=make_transform(Rotation.identity(), [-0.1, 0.05, 0.0]),
        t_table_cam=np.eye(4),
        source_video=video,
        grasp_frame_idx=num_samples // 2,
        release_frame_idx=None,
        source_start_time_s=start_time_s,
    )


def test_sample_nearest_frames_picks_closest_in_time(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    _write_clip(clip, 30)
    # Source frame i is at i/30 s: 0.30 s -> frame 9, 0.5 s -> frame 15, 0.31 s -> frame 9 (nearest).
    frames = list(sample_nearest_frames(clip, np.array([0.30, 0.31, 0.5])))
    assert [_frame_index(f.mean(axis=(0, 2))) for f in frames] == [9, 9, 15]


def test_roundtrip_two_episodes(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    _write_clip(clip, 60)
    demos = [_demo(clip, 0.2, 20), _demo(clip, 0.5, 24)]

    root = tmp_path / "ds"
    with LeRobotRecorder(root, "local/test_human", 20, human_features(*IMAGE_SIZE_HW)) as recorder:
        exporter = HumanDemoExporter(recorder, IMAGE_SIZE_HW)
        for demo in demos:
            exporter.add_demo(demo, "put the bowl on the plate")

    dataset = LeRobotDataset("local/test_human", root=root, video_backend="pyav")
    assert dataset.num_episodes == 2
    assert dataset.num_frames == 20 + 24
    assert dataset.fps == 20

    first = dataset[0]
    assert first["task"] == "put the bowl on the plate"
    np.testing.assert_allclose(first["observation.state"][:3].numpy(), demos[0].tcp_pos_table[0], atol=1e-6)
    np.testing.assert_allclose(first["action"][:3].numpy(), [0.01, 0.0, 0.0], atol=1e-6)
    assert first["observation.images.phone"].shape == (3, *IMAGE_SIZE_HW)

    # Frame identity survives encode/decode: sample 10 of episode 0 is source time 0.2 + 0.5 = 0.7 s.
    columns = dataset[10]["observation.images.phone"].mean(dim=(0, 1)).numpy()
    assert _frame_index(columns, scale=IMAGE_SIZE_HW[1] / SOURCE_WIDTH_PX) == round(0.7 * SOURCE_FPS)
    bowl_pose = dataset[0]["observation.bowl_pose"].numpy()
    np.testing.assert_allclose(bowl_pose[:3], [0.1, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(bowl_pose[3:]), 1.0, atol=1e-5)


def test_integrating_actions_reproduces_state(tmp_path: Path) -> None:
    demo = _demo(tmp_path / "unused.mp4", 0.0, 12)
    demo.tcp_pos_table[:, 1] = np.sin(np.arange(12) / 3.0)  # a non-monotonic y, unlike the default ramp
    vectors = HumanDemoExporter.demo_to_vectors(demo)
    replayed = replay_tcp_from_action(vectors["state"], vectors["action"])
    for i, name in enumerate(("tcp_x", "tcp_y", "tcp_z")):
        np.testing.assert_allclose(replayed[name], vectors["state"][:, i], atol=1e-6)
