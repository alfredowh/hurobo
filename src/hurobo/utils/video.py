"""Video I/O and the real/sim side-by-side compositor used throughout the pipeline."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from hurobo.utils.hand_landmarks import HAND_CONNECTIONS, INDEX_CHAIN, INDEX_TIP, THUMB_CHAIN, THUMB_TIP

if TYPE_CHECKING:
    from hurobo.camera import CameraIntrinsics
    from hurobo.hand import TrackedFrame


def read_frames(video_path: Path) -> Iterator[np.ndarray]:
    """Yield BGR frames from a video file one at a time.

    Args:
        video_path: path to a video file readable by OpenCV's backend.

    Yields:
        shape (H, W, 3) uint8 BGR frames, in file order.

    Raises:
        RuntimeError: if the file can't be opened at all.
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                return
            yield frame
    finally:
        capture.release()


def read_frames_with_timestamps(video_path: Path) -> Iterator[tuple[np.ndarray, float]]:
    """Yield (BGR frame, presentation time in seconds) pairs from a video file.

    Uses the container's per-frame timestamps rather than `index / fps`, because
    phone clips are often variable frame rate and drift would otherwise leak into
    every resampled trajectory.

    Args:
        video_path: path to a video file readable by OpenCV's backend.

    Yields:
        (frame, t_s): shape (H, W, 3) uint8 BGR frame and its time from the start
        of the clip. Timestamps are forced strictly increasing (some backends
        report 0 or repeat values for the first frames).

    Raises:
        RuntimeError: if the file can't be opened at all.
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fallback_dt_s = 1.0 / (capture.get(cv2.CAP_PROP_FPS) or 30.0)
    last_t_s = -fallback_dt_s
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                return
            t_s = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if t_s <= last_t_s:
                t_s = last_t_s + fallback_dt_s
            last_t_s = t_s
            yield frame, t_s
    finally:
        capture.release()


def get_video_fps(video_path: Path) -> float:
    """Return a video's reported frame rate in Hz."""
    capture = cv2.VideoCapture(str(video_path))
    try:
        return float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()


class VideoWriter:
    """Writes BGR frames to an H.264/yuv420p .mp4 by piping them to ffmpeg.

    H.264 rather than OpenCV's mp4v because Chromium-based viewers (VS Code,
    Jupyter) refuse to play mp4v. Using the context manager is required: ffmpeg
    only finalises the file on `close`.
    """

    def __init__(self, path: Path, fps: float, frame_size: tuple[int, int]) -> None:
        """Open a video file for writing.

        Args:
            path: output .mp4; parent directory must exist, an existing file is overwritten.
            fps: output frame rate in Hz.
            frame_size: (width, height) in pixels. Every frame passed to
                `write` must match this exactly.

        Raises:
            RuntimeError: if `ffmpeg` is not on PATH.
        """
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg not found on PATH; needed to write browser-playable video")
        self._frame_size = frame_size
        width, height = frame_size
        command = [ffmpeg, "-y", "-loglevel", "error"]
        command += [
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
        ]
        # yuv420p subsamples chroma 2x2, so odd sizes are padded by one pixel instead of failing.
        command += ["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p"]
        command += ["-movflags", "+faststart", str(path)]
        self._process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self._closed = False

    def write(self, frame_bgr: np.ndarray) -> None:
        """Append one BGR frame.

        Args:
            frame_bgr: shape (height, width, 3) uint8, matching `frame_size`.

        Raises:
            ValueError: if the frame's size or dtype doesn't match the writer's.
            RuntimeError: if ffmpeg exited early (its error output is included).
        """
        width, height = self._frame_size
        if frame_bgr.shape != (height, width, 3) or frame_bgr.dtype != np.uint8:
            raise ValueError(
                f"expected a ({height}, {width}, 3) uint8 frame, got {frame_bgr.shape} {frame_bgr.dtype}"
            )
        try:
            self._process.stdin.write(np.ascontiguousarray(frame_bgr).tobytes())
        except BrokenPipeError:
            self.close()

    def close(self) -> None:
        """Flush and close the file. Safe to call more than once.

        Raises:
            RuntimeError: if ffmpeg failed; its error output is included.
        """
        if self._closed:
            return
        self._closed = True
        _, stderr = self._process.communicate()
        if self._process.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed with exit code {self._process.returncode}:\n{stderr.decode(errors='replace')}"
            )

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


def make_side_by_side(
    frame_left: np.ndarray, frame_right: np.ndarray, target_height: int = 480
) -> np.ndarray:
    """Resize two frames to a common height and concatenate them horizontally.

    Used for the real | sim-twin comparison video (PLAN.md §4 step 5b), which
    is the MVP deliverable, and for the real | sim replay in the generic case.

    Args:
        frame_left: shape (H1, W1, 3) BGR frame, e.g. the real phone footage.
        frame_right: shape (H2, W2, 3) BGR frame, e.g. the sim render.
        target_height: output height in pixels; both frames are scaled to this
            height, preserving their own aspect ratios.

    Returns:
        shape (target_height, W1' + W2', 3) BGR frame.
    """

    def _resize_to_height(frame: np.ndarray, height: int) -> np.ndarray:
        scale = height / frame.shape[0]
        width = int(round(frame.shape[1] * scale))
        return cv2.resize(frame, (width, height))

    left = _resize_to_height(frame_left, target_height)
    right = _resize_to_height(frame_right, target_height)
    return np.concatenate([left, right], axis=1)


def alpha_composite(
    base_bgr: np.ndarray, overlay_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    """Blend a rendered-robot overlay onto a real frame, masked to the robot's silhouette.

    Stretch goal from PLAN.md §4 step 5d: a sanity check that the whole
    calibration chain (intrinsics, T_cam_table, hand PnP) is correct, by seeing
    whether the rendered robot lands on the real hand's pinch point. Not used
    to generate training data — the policy never sees a blended frame.

    Args:
        base_bgr: shape (H, W, 3), the real camera frame.
        overlay_bgr: shape (H, W, 3), the sim render (e.g. a MuJoCo segmentation
            pass of just the robot), same size as base_bgr.
        mask: shape (H, W), float in [0, 1] or bool, where the overlay is opaque
            (e.g. a segmentation mask of the robot body, zero elsewhere).
        alpha: blend weight for the overlay where mask == 1.

    Returns:
        shape (H, W, 3) BGR frame, base_bgr with the overlay blended in.
    """
    mask_3ch = mask.astype(np.float32)[..., None] * alpha
    return (base_bgr.astype(np.float32) * (1 - mask_3ch) + overlay_bgr.astype(np.float32) * mask_3ch).astype(
        np.uint8
    )


#: BGR colours shared by the overlay: thumb orange, index green, pinch red.
THUMB_BGR, INDEX_BGR, PINCH_BGR = (0, 140, 255), (0, 200, 0), (0, 0, 255)


def draw_hand_skeleton(frame_bgr: np.ndarray, landmarks_px: np.ndarray) -> None:
    """Draw the 21 hand landmarks and bones in place, with the pinch fingertips marked.

    Thumb bones are orange and index bones green; the two fingertips get large dots and a
    red segment joins them, which is the distance reported as the pinch aperture.

    Args:
        frame_bgr: image to draw on, modified in place.
        landmarks_px: shape (21, 2) detected landmark pixel coordinates.
    """
    points = [(round(u), round(v)) for u, v in landmarks_px]
    for start, end in HAND_CONNECTIONS:
        colour = THUMB_BGR if end in THUMB_CHAIN else INDEX_BGR if end in INDEX_CHAIN else (255, 255, 255)
        cv2.line(frame_bgr, points[start], points[end], colour, 2, cv2.LINE_AA)
    for point in points:
        cv2.circle(frame_bgr, point, 3, (30, 30, 30), -1, cv2.LINE_AA)
    cv2.line(frame_bgr, points[THUMB_TIP], points[INDEX_TIP], PINCH_BGR, 3, cv2.LINE_AA)
    cv2.circle(frame_bgr, points[THUMB_TIP], 8, THUMB_BGR, -1, cv2.LINE_AA)
    cv2.circle(frame_bgr, points[INDEX_TIP], 8, INDEX_BGR, -1, cv2.LINE_AA)


def draw_tcp_overlay(item: TrackedFrame, intrinsics: CameraIntrinsics) -> np.ndarray:
    """Copy of a tracked frame with the hand pose, the TCP (red cross) and a time/aperture caption.

    The whole detected hand is drawn when the result carries its landmarks; otherwise only
    the TCP cross is.

    Args:
        item: one `TrackedFrame`; a frame without a hand is captioned "no hand".
        intrinsics: camera that filmed the clip, for projecting the TCP.

    Returns:
        shape (H, W, 3) BGR frame.
    """
    white = (255, 255, 255)
    frame = item.frame_bgr.copy()
    caption = f"t={item.timestamp_s:.1f}s  no hand"
    if item.result is not None:
        if item.result.landmarks_image_px is not None:
            draw_hand_skeleton(frame, item.result.landmarks_image_px)
        u, v = intrinsics.project(item.result.tcp_pos_cam)
        cv2.drawMarker(frame, (round(u), round(v)), PINCH_BGR, cv2.MARKER_CROSS, 28, 3)
        caption = f"t={item.timestamp_s:.1f}s  aperture={item.result.pinch_aperture_m * 100:.1f}cm"
    cv2.putText(frame, caption, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, caption, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, white, 2, cv2.LINE_AA)
    return frame


def write_tcp_overlay_video(
    tracked: list[TrackedFrame], intrinsics: CameraIntrinsics, video_path: Path, fps: float
) -> Path:
    """Video of a tracked clip with the detected TCP (red cross) and a time/aperture caption.

    Args:
        tracked: output of `track_video`; frames without a hand are kept, captioned "no hand".
        intrinsics: camera that filmed the clip, for projecting the TCP.
        video_path: output .mp4 (H.264, browser-playable).
        fps: playback rate; the clip's own frame rate plays it in real time.

    Returns:
        `video_path`.
    """
    size = (intrinsics.image_width, intrinsics.image_height)
    with VideoWriter(video_path, fps, size) as writer:
        for item in tracked:
            writer.write(draw_tcp_overlay(item, intrinsics))
    return video_path


def caption_frame(image_rgb: np.ndarray, text: str) -> np.ndarray:
    """Return a BGR copy of an RGB frame with a caption (outlined, for legibility) top-left.

    Args:
        image_rgb: shape (H, W, 3) uint8 RGB.
        text: caption.

    Returns:
        shape (H, W, 3) uint8 BGR, ready for `VideoWriter` / `cv2.imwrite`.
    """
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.putText(image_bgr, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image_bgr, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return image_bgr
