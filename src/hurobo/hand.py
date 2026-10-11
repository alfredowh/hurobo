"""Hand TCP tracking via MediaPipe Tasks HandLandmarker, metric position via solvePnP.

Why solvePnP instead of using the world landmarks directly: MediaPipe's
`hand_world_landmarks` are metric but expressed in a hand-local frame with an
arbitrary, unanchored translation (no guaranteed relation to the camera). What we
actually need is T_cam_hand. Pairing the metric world landmarks (as 3D object
points, in the hand's own local frame) with their corresponding 2D image
landmarks lets `cv2.solvePnP` recover the camera-frame pose of that local frame,
which is exactly T_cam_hand (PLAN.md §4 step 2).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision as mp_vision

from hurobo.camera import CameraIntrinsics
from hurobo.utils.hand_landmarks import INDEX_TIP, THUMB_TIP
from hurobo.utils.transforms import apply_transform, rvec_tvec_to_transform
from hurobo.utils.video import read_frames_with_timestamps

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
_DEFAULT_MODEL_PATH = Path.home() / ".cache" / "hurobo" / "hand_landmarker.task"


def ensure_model_downloaded(model_path: Path = _DEFAULT_MODEL_PATH) -> Path:
    """Download the HandLandmarker model file if it isn't already cached.

    Args:
        model_path: where to look for / save the .task model file.

    Returns:
        The same path, guaranteed to exist on return.
    """
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(_MODEL_URL, model_path)  # noqa: S310 - fixed, known Google-hosted URL
    return model_path


@dataclass
class HandFrameResult:
    """One frame's hand-tracking result.

    Attributes:
        t_cam_hand: 4x4 transform, the hand's local landmark frame expressed in
            the camera frame (from solvePnP). Not directly the TCP; use
            `tcp_pos_cam` for that.
        tcp_pos_cam: shape (3,), the TCP (thumb/index-tip midpoint) position in
            the camera frame.
        pinch_aperture_m: distance between thumb tip and index tip, in metres
            (computed from the metric world landmarks, so it's camera-pose-independent).
        reprojection_error_px: mean distance between the 21 detected image landmarks
            and the PnP-reprojected world landmarks; a per-frame confidence signal.
        landmarks_hand: shape (21, 3), the metric world landmarks in the hand's local frame
            (the PnP object points); the pinch aperture is measured between rows 4 and 8.
        landmarks_image_px: shape (21, 2), the detected landmarks in the image (the PnP
            image points).
    """

    t_cam_hand: np.ndarray
    tcp_pos_cam: np.ndarray
    pinch_aperture_m: float
    reprojection_error_px: float = 0.0
    landmarks_hand: np.ndarray | None = None
    landmarks_image_px: np.ndarray | None = None

    @property
    def landmarks_cam(self) -> np.ndarray | None:
        """shape (21, 3), the landmarks in the camera frame, or None if not recorded."""
        if self.landmarks_hand is None:
            return None
        return apply_transform(self.t_cam_hand, self.landmarks_hand)


class HandTcpTracker:
    """Runs MediaPipe's HandLandmarker on a video stream and extracts a metric TCP.

    One instance is meant to process one clip's frames in order via
    `process_frame` (it uses VIDEO running mode, which requires monotonically
    increasing timestamps).
    """

    def __init__(self, intrinsics: CameraIntrinsics, model_path: Path | None = None) -> None:
        """Set up the HandLandmarker for a specific camera's intrinsics.

        Args:
            intrinsics: calibrated intrinsics for the camera the frames come from.
            model_path: path to hand_landmarker.task; downloaded automatically
                to a cache dir if not given and not already present there.
        """
        self._intrinsics = intrinsics
        resolved_model_path = model_path or ensure_model_downloaded()
        options = mp_vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(resolved_model_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)

    def process_frame(self, image_bgr: np.ndarray, timestamp_ms: int) -> HandFrameResult | None:
        """Run hand detection on one frame and recover a metric TCP pose.

        Args:
            image_bgr: a single BGR frame.
            timestamp_ms: strictly increasing timestamp for this frame, in
                milliseconds from the start of the clip (required by VIDEO mode).

        Returns:
            A `HandFrameResult`, or None if no hand was detected in this frame.
        """
        import mediapipe as mp

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        if not result.hand_landmarks:
            return None

        image_landmarks = result.hand_landmarks[0]
        world_landmarks = result.hand_world_landmarks[0]
        height, width = image_bgr.shape[:2]

        object_points = np.array([[lm.x, lm.y, lm.z] for lm in world_landmarks], dtype=np.float64)
        image_points = np.array([[lm.x * width, lm.y * height] for lm in image_landmarks], dtype=np.float64)

        # EPnP seeds the pose, then LM refines it. A plain ITERATIVE solve from the
        # default guess diverges on hands that are small or fingers-spread in the frame
        # (reprojection error in the thousands of px, z ~ 0), which poisons the trajectory.
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            self._intrinsics.camera_matrix,
            self._intrinsics.dist_coeffs,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not ok:
            return None
        rvec, tvec = cv2.solvePnPRefineLM(
            object_points,
            image_points,
            self._intrinsics.camera_matrix,
            self._intrinsics.dist_coeffs,
            rvec,
            tvec,
        )
        t_cam_hand = rvec_tvec_to_transform(rvec, tvec)
        if (apply_transform(t_cam_hand, object_points)[:, 2] <= 0).any():
            return None  # a hand with points behind the camera is a flipped solution
        projected, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            self._intrinsics.camera_matrix,
            self._intrinsics.dist_coeffs,
        )
        reprojection_error_px = float(np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1).mean())

        thumb_tip_hand = object_points[THUMB_TIP]
        index_tip_hand = object_points[INDEX_TIP]
        tcp_pos_hand = (thumb_tip_hand + index_tip_hand) / 2.0
        tcp_pos_cam = apply_transform(t_cam_hand, tcp_pos_hand)
        pinch_aperture_m = float(np.linalg.norm(thumb_tip_hand - index_tip_hand))

        return HandFrameResult(
            t_cam_hand=t_cam_hand,
            tcp_pos_cam=tcp_pos_cam,
            pinch_aperture_m=pinch_aperture_m,
            reprojection_error_px=reprojection_error_px,
            landmarks_hand=object_points,
            landmarks_image_px=image_points,
        )

    def close(self) -> None:
        """Release the underlying MediaPipe task graph."""
        self._landmarker.close()

    def __enter__(self) -> HandTcpTracker:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


@dataclass
class TrackedFrame:
    """One video frame with its hand-tracking result, for inspection and plotting.

    Attributes:
        frame_bgr: the decoded frame.
        timestamp_s: seconds from the start of the clip.
        result: tracking result, or None if no hand was detected.
    """

    frame_bgr: np.ndarray
    timestamp_s: float
    result: HandFrameResult | None


def track_video(video_path: Path, intrinsics: CameraIntrinsics) -> list[TrackedFrame]:
    """Run `HandTcpTracker` over every frame of a clip and keep frames with their results.

    Holds all frames in memory; meant for short inspection clips, not full recordings.

    Args:
        video_path: input clip.
        intrinsics: calibration of the camera that filmed it.
    """
    with HandTcpTracker(intrinsics) as tracker:
        return [
            TrackedFrame(frame, t_s, tracker.process_frame(frame, int(round(t_s * 1000))))
            for frame, t_s in read_frames_with_timestamps(video_path)
        ]
