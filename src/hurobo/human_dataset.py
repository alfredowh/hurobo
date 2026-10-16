"""Exports `HumanDemo`s (plus their source video frames) as a LeRobotDataset."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from hurobo.camera import CameraIntrinsics
from hurobo.demo import HumanDemo
from hurobo.demo_overlay import DemoOverlay
from hurobo.recorder import LeRobotRecorder, vector_feature, video_feature
from hurobo.utils.dashboard_video import SignalPanel, SyncedDashboard, write_dashboard_video
from hurobo.utils.transforms import transform_to_pos_quat_wxyz
from hurobo.utils.video import read_frames_with_timestamps

STATE_NAMES = ["tcp_x", "tcp_y", "tcp_z", "pinch_aperture"]
# `action` has no robot behind it for human data: it is the human's next-step TCP
# displacement plus the grasp flag, the same quantities the retargeter consumes.
ACTION_NAMES = ["d_tcp_x", "d_tcp_y", "d_tcp_z", "grasp"]
POSE_NAMES = ["x", "y", "z", "qw", "qx", "qy", "qz"]
# Which action channel is the "next step" of which state channel, for overlay plots.
STATE_ACTION_PAIRS = [
    ("tcp_x", "d_tcp_x"),
    ("tcp_y", "d_tcp_y"),
    ("tcp_z", "d_tcp_z"),
    ("pinch_aperture", "grasp"),
]


def human_features(image_height: int, image_width: int) -> dict:
    """Feature spec of the human-demo dataset.

    Args:
        image_height: stored phone-image height.
        image_width: stored phone-image width.

    Returns:
        Feature dict for `LeRobotRecorder`.
    """
    return {
        "observation.images.phone": video_feature(image_height, image_width),
        "observation.state": vector_feature(STATE_NAMES),
        "action": vector_feature(ACTION_NAMES),
        "observation.bowl_pose": vector_feature(POSE_NAMES),
        "observation.plate_pose": vector_feature(POSE_NAMES),
    }


def sample_nearest_frames(video_path: Path, target_times_s: np.ndarray) -> Iterator[np.ndarray]:
    """Yield, for each target time, the BGR frame of the video nearest in time.

    Nearest-frame (not blended) so that stored pixels are always real captured frames.

    Args:
        video_path: source clip.
        target_times_s: shape (M,), increasing times in the clip's own timeline.

    Yields:
        M BGR frames, one per target time, in order.
    """
    next_target = 0
    previous: tuple[np.ndarray, float] | None = None
    for frame, t_s in read_frames_with_timestamps(video_path):
        while next_target < len(target_times_s) and t_s >= target_times_s[next_target]:
            target = target_times_s[next_target]
            use_previous = previous is not None and (target - previous[1]) <= (t_s - target)
            yield previous[0] if use_previous else frame
            next_target += 1
        previous = (frame, t_s)
        if next_target == len(target_times_s):
            return
    while next_target < len(target_times_s) and previous is not None:
        yield previous[0]  # targets past the last decoded frame
        next_target += 1


class HumanDemoExporter:
    """Writes a set of `HumanDemo`s into one LeRobotDataset (one episode per demo)."""

    def __init__(self, recorder: LeRobotRecorder, image_size_hw: tuple[int, int]) -> None:
        """Bind to a recorder created with `human_features(*image_size_hw)`.

        Args:
            recorder: an open recorder; the exporter does not finalize it.
            image_size_hw: (height, width) every phone frame is resized to.
        """
        self._recorder = recorder
        self._image_size_hw = image_size_hw

    @staticmethod
    def demo_to_vectors(demo: HumanDemo) -> dict[str, np.ndarray]:
        """Compute the per-frame state/action/pose arrays of one demo.

        Args:
            demo: a processed demo (table frame, uniform sampling).

        Returns:
            {"state": (N, 4), "action": (N, 4), "bowl_pose": (N, 7), "plate_pose": (N, 7)},
            float32. The last action holds position (zero delta); `grasp` is the grasp
            state at the *next* sample, so the action is "what the hand does next".
        """
        positions = demo.tcp_pos_table
        next_positions = np.vstack([positions[1:], positions[-1:]])
        next_grasp = np.append(demo.is_grasping[1:], demo.is_grasping[-1])
        state = np.column_stack([positions, demo.pinch_aperture_m])
        action = np.column_stack([next_positions - positions, next_grasp.astype(np.float64)])

        def pose_row(t_table_object: np.ndarray) -> np.ndarray:
            pos, quat_wxyz = transform_to_pos_quat_wxyz(t_table_object)
            return np.concatenate([pos, quat_wxyz])

        n = len(positions)
        return {
            "state": state.astype(np.float32),
            "action": action.astype(np.float32),
            "bowl_pose": np.tile(pose_row(demo.t_table_bowl), (n, 1)).astype(np.float32),
            "plate_pose": np.tile(pose_row(demo.t_table_plate), (n, 1)).astype(np.float32),
        }

    def add_demo(self, demo: HumanDemo, task: str) -> int:
        """Export one demo as an episode.

        Args:
            demo: a processed demo whose `source_video` is readable.
            task: instruction string for the episode.

        Returns:
            The episode index.
        """
        vectors = self.demo_to_vectors(demo)
        target_times_s = demo.source_start_time_s + demo.timestamps_s
        height, width = self._image_size_hw

        def frames() -> Iterator[dict]:
            images = sample_nearest_frames(demo.source_video, target_times_s)
            for i, image_bgr in enumerate(images):
                resized = cv2.resize(image_bgr, (width, height), interpolation=cv2.INTER_AREA)
                yield {
                    "observation.images.phone": cv2.cvtColor(resized, cv2.COLOR_BGR2RGB),
                    "observation.state": vectors["state"][i],
                    "action": vectors["action"][i],
                    "observation.bowl_pose": vectors["bowl_pose"][i],
                    "observation.plate_pose": vectors["plate_pose"][i],
                }

        return self._recorder.add_episode(frames(), task)

    @staticmethod
    def sidecar_metadata(intrinsics: CameraIntrinsics, demos: list[HumanDemo]) -> dict:
        """Provenance the LeRobot schema has no slot for.

        Args:
            intrinsics: calibration of the capture camera (at source resolution).
            demos: the exported demos, in episode order.

        Returns:
            JSON-serialisable dict for `LeRobotRecorder.write_sidecar`.
        """
        return {
            "frames": "all poses are in the table (ArUco marker) frame; pose = [x, y, z, qw, qx, qy, qz]",
            "action_semantics": "human proxy: next-step TCP displacement (m), grasp flag; not robot commands",
            "camera_matrix_source_resolution": intrinsics.camera_matrix.tolist(),
            "dist_coeffs": intrinsics.dist_coeffs.tolist(),
            "source_resolution_wh": [intrinsics.image_width, intrinsics.image_height],
            "episodes": [
                {
                    "source_video": str(d.source_video),
                    "source_start_time_s": d.source_start_time_s,
                    "grasp_frame_idx": d.grasp_frame_idx,
                    "release_frame_idx": d.release_frame_idx,
                    "t_table_cam": d.t_table_cam.tolist(),
                }
                for d in demos
            ],
        }


def export_human_dataset(
    demos: list[HumanDemo],
    root: Path,
    intrinsics: CameraIntrinsics,
    task: str,
    image_size_hw: tuple[int, int] = (360, 640),
    fps: int = 20,
    repo_id: str = "local/hurobo_human_demos",
    overwrite: bool = False,
) -> list[int]:
    """Write demos as one LeRobotDataset plus its metadata sidecar.

    Args:
        demos: extracted demos; all are stored under the same `task`.
        root: dataset directory.
        intrinsics: calibration of the capture camera, stored in the sidecar.
        task: natural-language instruction.
        image_size_hw: stored (height, width) of the phone image.
        fps: dataset sampling rate; must match the demos' resampling rate.
        repo_id: Hub-style dataset id.
        overwrite: replace an existing `root`.

    Returns:
        Episode index of each demo, in order.
    """
    with LeRobotRecorder(
        root, repo_id, fps, human_features(*image_size_hw), robot_type="human_hand", overwrite=overwrite
    ) as recorder:
        exporter = HumanDemoExporter(recorder, image_size_hw)
        episodes = [exporter.add_demo(demo, task) for demo in demos]
        recorder.write_sidecar(HumanDemoExporter.sidecar_metadata(intrinsics, demos))
    return episodes


def read_episode_vectors(dataset, num_frames: int, start: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """`observation.state` and `action` of consecutive frames of a loaded LeRobotDataset.

    Args:
        dataset: a `LeRobotDataset`.
        num_frames: how many frames to read.
        start: global index of the first frame.

    Returns:
        (state, action), shapes (num_frames, S) and (num_frames, A).
    """
    frames = [dataset[start + i] for i in range(num_frames)]
    state = np.stack([f["observation.state"].numpy() for f in frames])
    action = np.stack([f["action"].numpy() for f in frames])
    return state, action


def replay_tcp_from_action(state: np.ndarray, action: np.ndarray) -> dict[str, np.ndarray]:
    """Integrate the displacement actions from the first state, to check they reproduce `observation.state`.

    Args:
        state: shape (N, 4), `observation.state` of one episode (`STATE_NAMES` order).
        action: shape (N, 4), `action` of the same episode (`ACTION_NAMES` order).

    Returns:
        {"tcp_x": ..., "tcp_y": ..., "tcp_z": ...}, each shape (N,): `state[0]` plus the
        running sum of the displacements of the *previous* samples, so row i is where
        the actions say the hand is at sample i.
    """
    steps = action[:, [ACTION_NAMES.index(f"d_{name}") for name in STATE_NAMES[:3]]]
    moved = np.vstack([np.zeros((1, 3)), np.cumsum(steps[:-1], axis=0)])
    replayed = state[0, :3] + moved
    return {name: replayed[:, i] for i, name in enumerate(STATE_NAMES[:3])}


def episode_dashboard(
    dataset,
    episode_index: int,
    close_threshold_m: float | None = None,
    open_threshold_m: float | None = None,
    overlay: DemoOverlay | None = None,
) -> SyncedDashboard:
    """Dashboard of one recorded episode, built only from what a dataset consumer reads.

    Args:
        dataset: a `LeRobotDataset` written by `export_human_dataset`.
        episode_index: episode to show.
        close_threshold_m: lower edge of the grasp hysteresis band to draw, if any.
        open_threshold_m: upper edge of the grasp hysteresis band to draw, if any.
        overlay: overlay of the demo this episode was exported from; if given, the phone image shows its
            markers, hand, TCP with path, object centres and gripper (the curves still come from the dataset).

    Returns:
        Phone image | TCP path in the table frame | aperture | observation.state | action,
        with grasping samples marked and a shared time cursor.
    """
    episode = dataset.meta.episodes[episode_index]
    samples = [dataset[i] for i in range(episode["dataset_from_index"], episode["dataset_to_index"])]
    state = np.stack([s["observation.state"].numpy() for s in samples])
    action = np.stack([s["action"].numpy() for s in samples])
    # Dataset images are CHW float RGB in [0, 1].
    frames_bgr = [
        cv2.cvtColor(
            (s["observation.images.phone"].permute(1, 2, 0).numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
        )
        for s in samples
    ]
    if overlay is not None:
        for sample, frame_bgr in enumerate(frames_bgr):
            overlay.annotate_sample(frame_bgr, sample)
    return SyncedDashboard(
        frames_bgr,
        np.array([float(s["timestamp"]) for s in samples]),
        state[:, :3],
        state[:, STATE_NAMES.index("pinch_aperture")],
        image_title=f"observation.images.phone, episode {episode_index}",
        path_title="observation.state TCP, table frame (m)",
        path_axis_labels=("x", "y", "z up"),
        grasp_mask=action[:, ACTION_NAMES.index("grasp")] > 0.5,
        close_threshold_m=close_threshold_m,
        open_threshold_m=open_threshold_m,
        signal_panels=[
            SignalPanel("observation.state", state, STATE_NAMES, y_label="m (table frame)"),
            SignalPanel(
                "action (human proxy)",
                action,
                ACTION_NAMES,
                right_axis_labels=["grasp"],
                y_label="d_tcp (m / step)",
                right_y_label="grasp flag",
            ),
        ],
    )


def write_episode_dashboard_video(
    dataset,
    episode_index: int,
    video_path: Path,
    close_threshold_m: float | None = None,
    open_threshold_m: float | None = None,
    overlay: DemoOverlay | None = None,
) -> Path:
    """Video of one recorded episode: image, TCP path, aperture, state and action on one clock.

    Args:
        dataset: a `LeRobotDataset` written by `export_human_dataset`.
        episode_index: episode to show.
        video_path: output .mp4 (H.264, browser-playable).
        close_threshold_m: lower edge of the grasp hysteresis band to draw, if any.
        open_threshold_m: upper edge of the grasp hysteresis band to draw, if any.
        overlay: see `episode_dashboard`.

    Returns:
        `video_path`, played at the dataset's fps.
    """
    dashboard = episode_dashboard(dataset, episode_index, close_threshold_m, open_threshold_m, overlay)
    return write_dashboard_video(dashboard, video_path, dataset.fps)


def describe_features(dataset) -> list[str]:
    """One line per user-defined feature (name, dtype, shape), skipping LeRobot's bookkeeping columns."""
    skipped = ("timestamp", "frame_", "episode_", "index", "task_")
    return [
        f"{name:28s} {spec['dtype']:8s} {tuple(spec['shape'])}"
        for name, spec in dataset.features.items()
        if not name.startswith(skipped)
    ]
