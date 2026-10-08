"""Video I/O and the real/sim side-by-side compositor used throughout the pipeline."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np


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


def get_video_fps(video_path: Path) -> float:
    """Return a video's reported frame rate in Hz."""
    capture = cv2.VideoCapture(str(video_path))
    try:
        return float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()


class VideoWriter:
    """Thin wrapper around cv2.VideoWriter with a context-manager interface."""

    def __init__(self, path: Path, fps: float, frame_size: tuple[int, int]) -> None:
        """Open a video file for writing.

        Args:
            path: output path; the container is inferred from the extension
                (use .mp4 with the 'mp4v' fourcc, widely compatible without
                extra codec installs).
            fps: output frame rate in Hz.
            frame_size: (width, height) in pixels. Every frame passed to
                `write` must match this exactly.
        """
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(path), fourcc, fps, frame_size)

    def write(self, frame_bgr: np.ndarray) -> None:
        """Append one BGR frame."""
        self._writer.write(frame_bgr)

    def close(self) -> None:
        """Flush and close the file. Safe to call more than once."""
        self._writer.release()

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
