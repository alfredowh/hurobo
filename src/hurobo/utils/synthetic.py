"""Paste ArUco markers with known poses into real frames, to check the pipeline against ground truth."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from hurobo.camera import CameraIntrinsics
from hurobo.demo import GraspSegmenter, HumanDemo
from hurobo.extract import ClipReport, extract_demo
from hurobo.markers import MarkerLayout
from hurobo.utils.transforms import invert_transform, look_at_origin_pose, make_transform
from hurobo.utils.video import VideoWriter

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
    def from_clip_dir(
        cls,
        clip_dir: Path,
        first_frame: int = 20,
        camera_position_table: tuple[float, float, float] = (0.0, -0.45, 0.40),
    ) -> StandInScene:
        """The default stand-in: GIF frames of a real hand, camera 0.45 m in front, 0.4 m up.

        Args:
            clip_dir: folder of `.png` frames (800x450).
            first_frame: skip the frames before the hand enters the view.
            camera_position_table: where the pretend camera sits in the table frame, looking at
                the table origin. Changing it re-interprets the same pixels from another
                viewpoint, which is how the camera-pose handling is exercised without a capture.
        """
        return cls(
            frame_paths=sorted(clip_dir.glob("*.png"))[first_frame:],
            intrinsics=CameraIntrinsics.pinhole(800, 450, 800.0),
            t_table_cam=look_at_origin_pose(np.array(camera_position_table, dtype=float)),
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


def _min_jerk(fraction: np.ndarray) -> np.ndarray:
    """Minimum-jerk easing, 0 -> 1, with zero speed at both ends."""
    return 10 * fraction**3 - 15 * fraction**4 + 6 * fraction**5


def synthetic_pick_place_demo(
    bowl_xy_table: np.ndarray,
    plate_xy_table: np.ndarray,
    start_xyz_table: np.ndarray,
    grasp_angle_rad: float = -np.pi / 2,
    rim_radius_m: float = 0.069,
    rim_height_m: float = 0.055,
    plate_height_m: float = 0.028,
    rate_hz: float = 20.0,
    seed: int = 0,
) -> HumanDemo:
    """A made-up human pick-and-place in the table frame: reach, pinch the bowl rim, carry, release, retreat.

    Stand-in data for tests and notebooks until real clips exist. The hand starts at `start_xyz_table`
    (anywhere), pinches the rim of the bowl at `grasp_angle_rad` around its centre, carries it so that the
    bowl centre ends over the plate centre, opens, and rises. Small seeded wobble and a few millimetres of
    placement error are added so it is not a perfect path.

    Args:
        bowl_xy_table: shape (2,), bowl centre.
        plate_xy_table: shape (2,), plate centre.
        start_xyz_table: shape (3,), where the hand is at t = 0.
        grasp_angle_rad: direction from the bowl centre to the pinch point, in the table xy plane
            (-pi/2 is the rim nearest the person).
        rim_radius_m: bowl radius at the rim.
        rim_height_m: pinch height above the table.
        plate_height_m: plate rim height; the bowl is released this far above the table plus the bowl base.
        rate_hz: sample rate.
        seed: for the wobble.

    Returns:
        A `HumanDemo` with grasp and release found; the marker poses are the object centres (no rotation).
    """
    rng = np.random.default_rng(seed)
    bowl_xy, plate_xy = np.asarray(bowl_xy_table, float), np.asarray(plate_xy_table, float)
    rim_offset = rim_radius_m * np.array([np.cos(grasp_angle_rad), np.sin(grasp_angle_rad)])
    grasp = np.array([*(bowl_xy + rim_offset), rim_height_m])
    place = np.array([*(plate_xy + rim_offset), plate_height_m + rim_height_m + 0.01])
    start = np.asarray(start_xyz_table, float)
    above_grasp = grasp + np.array([0.0, 0.0, 0.10])
    above_place = place + np.array([0.0, 0.0, 0.10])
    retreat = place + np.array([0.0, 0.0, 0.15])

    # (target, duration_s, aperture_after): aperture is 9 cm open or 1.5 cm pinched
    legs = [
        (above_grasp, 1.4, 0.09),
        (grasp, 0.8, 0.09),
        (grasp, 0.5, 0.015),
        (above_grasp, 0.7, 0.015),
        (above_place, 1.2, 0.015),
        (place, 0.7, 0.015),
        (place, 0.5, 0.09),
        (retreat, 0.8, 0.09),
    ]
    positions, apertures = [start], [0.09]
    for target, duration_s, aperture_after in legs:
        steps = max(2, int(round(duration_s * rate_hz)))
        eased = _min_jerk(np.linspace(0.0, 1.0, steps + 1)[1:])[:, None]
        segment = positions[-1] + eased * (target - positions[-1])
        positions.extend(segment)
        apertures.extend(
            apertures[-1] + _min_jerk(np.linspace(0.0, 1.0, steps + 1)[1:]) * (aperture_after - apertures[-1])
        )
    tcp = np.asarray(positions)
    tcp += rng.normal(0.0, 0.002, tcp.shape)
    aperture = np.asarray(apertures)
    is_grasping = GraspSegmenter(0.03, 0.05).segment(aperture)
    grasp_idx, release_idx = GraspSegmenter.find_transition_frames(is_grasping)
    return HumanDemo(
        timestamps_s=np.arange(len(tcp)) / rate_hz,
        tcp_pos_table=tcp,
        pinch_aperture_m=aperture,
        is_grasping=is_grasping,
        t_table_bowl=make_transform(Rotation.identity(), [*bowl_xy, 0.0]),
        t_table_plate=make_transform(Rotation.identity(), [*plate_xy, 0.0]),
        t_table_cam=np.eye(4),
        source_video=Path("synthetic"),
        grasp_frame_idx=grasp_idx,
        release_frame_idx=release_idx,
    )
