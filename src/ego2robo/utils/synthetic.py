"""Paste ArUco markers with known poses into real frames, to check the pipeline against ground truth."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from ego2robo.camera import CameraIntrinsics
from ego2robo.demo import HumanDemo
from ego2robo.extract import ClipReport, extract_demo
from ego2robo.markers import MarkerLayout
from ego2robo.utils.transforms import invert_transform, look_at_origin_pose, make_transform
from ego2robo.utils.video import VideoWriter

_PAD_FRACTION = 0.25  # white quiet zone around the marker, as a fraction of its side


def paste_marker(
    frame_bgr: np.ndarray,
    marker_id: int,
    t_cam_marker: np.ndarray,
    side_m: float,
    intrinsics: CameraIntrinsics,
) -> None:
    """Render one 4x4_50 ArUco marker at a known pose onto `frame_bgr`, in place.

    Corner order matches `MarkerPoseEstimator._estimate_marker_pose`: marker-frame
    (-h, +h), (+h, +h), (+h, -h), (-h, -h), i.e. image TL, TR, BR, BL of the marker.

    Args:
        frame_bgr: image to draw on (modified in place).
        marker_id: id in DICT_4X4_50.
        t_cam_marker: 4x4 marker pose in the camera frame.
        side_m: black-square side length in metres.
        intrinsics: camera intrinsics used for the projection (zero distortion assumed).
    """
    side_px = 240
    pad_px = int(side_px * _PAD_FRACTION)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, side_px)
    padded = cv2.copyMakeBorder(marker, pad_px, pad_px, pad_px, pad_px, cv2.BORDER_CONSTANT, value=255)

    half = side_m / 2.0 * (1 + 2 * _PAD_FRACTION)
    corners_marker = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]])
    corners_cam = corners_marker @ t_cam_marker[:3, :3].T + t_cam_marker[:3, 3]
    projected = cv2.projectPoints(
        corners_cam, np.zeros(3), np.zeros(3), intrinsics.camera_matrix, intrinsics.dist_coeffs
    )[0].reshape(4, 2)

    size = padded.shape[0]
    source = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(source, projected.astype(np.float32))
    height, width = frame_bgr.shape[:2]
    warped = cv2.warpPerspective(cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR), homography, (width, height))
    mask = cv2.warpPerspective(np.full(padded.shape, 255, np.uint8), homography, (width, height)) > 0
    frame_bgr[mask] = warped[mask]


def render_marker_clip(
    frame_paths: list[Path],
    video_path: Path,
    intrinsics: CameraIntrinsics,
    t_table_cam: np.ndarray,
    t_table_markers: dict[int, np.ndarray],
    marker_side_m: float,
    fps: float,
) -> None:
    """Write a video of real frames with ArUco markers pasted in at known table-frame poses.

    Args:
        frame_paths: image files, in time order, all of the size the intrinsics describe.
        video_path: output .mp4.
        intrinsics: camera intrinsics used to project the markers.
        t_table_cam: 4x4 camera pose in the table frame.
        t_table_markers: {marker id: 4x4 T_table_marker}.
        marker_side_m: side length of every marker, in metres.
        fps: frame rate of the output video.
    """
    t_cam_table = invert_transform(t_table_cam)
    with VideoWriter(video_path, fps, (intrinsics.image_width, intrinsics.image_height)) as writer:
        for path in frame_paths:
            frame = cv2.imread(str(path))
            for marker_id, t_table_marker in t_table_markers.items():
                paste_marker(frame, marker_id, t_cam_table @ t_table_marker, marker_side_m, intrinsics)
            writer.write(frame)


@dataclass
class StandInScene:
    """A real hand clip with ArUco markers pasted in at known poses, until phone captures exist.

    The ground-truth poses make the whole perception chain checkable: extraction
    must recover `t_table_cam`, `t_table_bowl` and `t_table_plate`.

    Attributes:
        frame_paths: hand-clip frames, in time order, of the size `intrinsics` describes.
        intrinsics: the pretend phone camera.
        t_table_cam: ground-truth camera pose in the table frame.
        t_table_bowl: ground-truth bowl-marker pose in the table frame.
        t_table_plate: ground-truth plate-marker pose in the table frame.
        marker_side_m: side of every marker.
        fps: frame rate of the rendered clip.
    """

    frame_paths: list[Path]
    intrinsics: CameraIntrinsics
    t_table_cam: np.ndarray
    t_table_bowl: np.ndarray
    t_table_plate: np.ndarray
    marker_side_m: float = 0.06
    fps: float = 10.0

    #: ArUco ids: the table marker defines the table frame, the others sit on the objects.
    TABLE_ID = 20
    BOWL_ID = 21
    PLATE_ID = 22

    @classmethod
    def from_clip_dir(cls, clip_dir: Path, first_frame: int = 20) -> StandInScene:
        """The default stand-in: GIF frames of a real hand, camera 0.45 m in front, 0.4 m up.

        Args:
            clip_dir: folder of `.png` frames (800x450).
            first_frame: skip the frames before the hand enters the view.
        """
        return cls(
            frame_paths=sorted(clip_dir.glob("*.png"))[first_frame:],
            intrinsics=CameraIntrinsics.pinhole(800, 450, 800.0),
            t_table_cam=look_at_origin_pose(np.array([0.0, -0.45, 0.40])),
            t_table_bowl=make_transform(Rotation.from_euler("z", 20, degrees=True), [-0.12, 0.02, 0.0]),
            t_table_plate=make_transform(Rotation.from_euler("z", -10, degrees=True), [0.13, 0.04, 0.0]),
        )

    @property
    def layout(self) -> MarkerLayout:
        """Marker ids and sizes as the extractor expects them."""
        return MarkerLayout(
            self.TABLE_ID,
            self.marker_side_m,
            {"bowl": self.BOWL_ID, "plate": self.PLATE_ID},
            self.marker_side_m,
        )

    def render_clip(self, video_path: Path) -> Path:
        """Write the marker-augmented clip to `video_path` and return it."""
        render_marker_clip(
            self.frame_paths,
            video_path,
            self.intrinsics,
            self.t_table_cam,
            {self.TABLE_ID: np.eye(4), self.BOWL_ID: self.t_table_bowl, self.PLATE_ID: self.t_table_plate},
            self.marker_side_m,
            self.fps,
        )
        return video_path

    def extract(self, video_path: Path) -> tuple[HumanDemo, ClipReport]:
        """Run the standard extraction on a clip rendered by `render_clip`."""
        return extract_demo(video_path, self.intrinsics, self.layout)


def synthetic_aperture_trace(
    seed: int = 0, duration_s: float = 6.0, rate_hz: float = 50.0
) -> tuple[np.ndarray, np.ndarray]:
    """Noisy pinch-aperture series with one grasp (open 7 cm, closed 1.5 cm between 1.5 s and 4.2 s).

    Args:
        seed: noise seed.
        duration_s: length of the series.
        rate_hz: sampling rate.

    Returns:
        (timestamps_s, aperture_m), both shape (N,).
    """
    t = np.arange(0.0, duration_s, 1.0 / rate_hz)
    noise = 0.004 * np.random.default_rng(seed).standard_normal(t.size)
    return t, 0.07 - 0.055 * ((t > 1.5) & (t < 4.2)) + noise
