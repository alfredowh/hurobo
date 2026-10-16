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


def save_gif(frames: list[np.ndarray], path: Path, fps: float = 12.0, boomerang: bool = True) -> Path:
    """Write RGB frames as a looping GIF (README media, where an MP4 does not autoplay).

    Args:
        frames: (H, W, 3) uint8 RGB images of equal size.
        path: output `.gif`; parent folders are created.
        fps: playback rate.
        boomerang: append the frames in reverse so a camera sweep loops without a jump.

    Returns:
        `path`.
    """
    from PIL import Image

    sequence = frames + frames[-2:0:-1] if boomerang else frames
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(frame) for frame in sequence]
    images[0].save(
        path, save_all=True, append_images=images[1:], duration=round(1000 / fps), loop=0, optimize=True
    )
    return path


def hstack_resized(frames: list[np.ndarray], target_height: int = 480) -> np.ndarray:
    """Scale frames to a common height (keeping aspect ratios) and join them left to right.

    Args:
        frames: non-empty list of (H, W, 3) frames of the same dtype.
        target_height: output height in pixels.

    Returns:
        shape (target_height, sum of scaled widths, 3).
    """
    resized = []
    for frame in frames:
        width = int(round(frame.shape[1] * target_height / frame.shape[0]))
        resized.append(cv2.resize(frame, (width, target_height)))
    return np.concatenate(resized, axis=1)


def fit_in_box(frame: np.ndarray, box_width: int, box_height: int) -> np.ndarray:
    """Scale a frame, keeping its aspect ratio, to the largest size that fits a box.

    Args:
        frame: shape (H, W, 3).
        box_width: available width in pixels.
        box_height: available height in pixels.

    Returns:
        the frame scaled by min(box_width / W, box_height / H); unchanged if the box is degenerate.
    """
    height, width = frame.shape[:2]
    scale = min(box_width / width, box_height / height)
    if scale <= 0 or abs(scale - 1.0) < 1e-3:
        return frame
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)


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
    return hstack_resized([frame_left, frame_right], target_height)


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


def draw_gripper_gauge(
    frame_bgr: np.ndarray,
    aperture_m: float,
    grasping: bool,
    origin_px: tuple[int, int],
    max_aperture_m: float = 0.12,
    width_px: int = 150,
) -> None:
    """Draws a two-finger gripper symbol whose gap follows the pinch aperture, in place.

    Args:
        frame_bgr: image to draw on.
        aperture_m: thumb-index distance; the gap is drawn to scale up to `max_aperture_m`.
        grasping: hysteresis-thresholded grasp state; the fingers are red when closed, green when open.
        origin_px: top-left corner of the symbol.
        max_aperture_m: aperture that fills `width_px`.
        width_px: width of the symbol.
    """
    x0, y0 = origin_px
    gap = round(width_px * min(max(aperture_m, 0.0) / max_aperture_m, 1.0))
    finger_w, finger_h = 14, 46
    centre = x0 + width_px // 2
    colour = (0, 0, 230) if grasping else (0, 200, 0)
    cv2.rectangle(frame_bgr, (x0 - 6, y0 - 8), (x0 + width_px + 6, y0 + finger_h + 30), (30, 30, 30), -1)
    cv2.rectangle(
        frame_bgr, (centre - gap // 2 - finger_w, y0), (centre - gap // 2, y0 + finger_h), colour, -1
    )
    cv2.rectangle(
        frame_bgr, (centre + gap // 2, y0), (centre + gap // 2 + finger_w, y0 + finger_h), colour, -1
    )
    label = f"{'CLOSED' if grasping else 'open'}  {100 * aperture_m:.1f} cm"
    cv2.putText(
        frame_bgr,
        label,
        (x0, y0 + finger_h + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def draw_outlined_text(
    image_bgr: np.ndarray,
    text: str,
    origin: tuple[int, int],
    font_scale: float,
    colour: tuple[int, int, int] = (255, 255, 255),
) -> None:
    """Draw text with a dark outline (legible on any background) onto an image, in place.

    Hershey letters are about 22 * `font_scale` pixels tall. The stroke is 2 px from scale 0.6
    up and 1 px below, with a 2 px wider outline.

    Args:
        image_bgr: image to draw on, modified in place.
        text: the string.
        origin: (x, y) of the text's bottom-left corner.
        font_scale: OpenCV font scale.
        colour: BGR text colour.
    """
    stroke = 2 if font_scale >= 0.6 else 1
    cv2.putText(
        image_bgr, text, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), stroke + 2, cv2.LINE_AA
    )
    cv2.putText(image_bgr, text, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, colour, stroke, cv2.LINE_AA)


def caption_frame(image_rgb: np.ndarray, text: str, font_scale: float = 0.7) -> np.ndarray:
    """Return a BGR copy of an RGB frame with a caption (outlined, for legibility) top-left.

    Args:
        image_rgb: shape (H, W, 3) uint8 RGB.
        text: caption.
        font_scale: OpenCV font scale of the caption.

    Returns:
        shape (H, W, 3) uint8 BGR, ready for `VideoWriter` / `cv2.imwrite`.
    """
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    draw_outlined_text(image_bgr, text, (10, 28), font_scale)
    return image_bgr


def interpolate_frames(
    frame_a: np.ndarray, frame_b: np.ndarray, alpha: float, flow_ab: np.ndarray, flow_ba: np.ndarray
) -> np.ndarray:
    """Synthesise the frame a fraction `alpha` of the way from `frame_a` to `frame_b`.

    Each source frame is warped along the dense optical flow toward the in-between time and the
    two are cross-faded. Plain cross-fading would ghost a moving hand into two copies; warping
    keeps it one hand at an intermediate position.

    Args:
        frame_a: shape (H, W, 3) frame at alpha = 0.
        frame_b: shape (H, W, 3) frame at alpha = 1.
        alpha: in [0, 1].
        flow_ab: shape (H, W, 2), flow from `frame_a` to `frame_b` (see `dense_flow`).
        flow_ba: shape (H, W, 2), flow from `frame_b` to `frame_a`.

    Returns:
        shape (H, W, 3) frame of the same dtype.
    """
    height, width = frame_a.shape[:2]
    alpha = float(alpha)
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    # Backward warp using the flow evaluated at the output pixel: a first-order approximation
    # that is good for the small inter-frame motions of a 10-30 fps clip.
    from_a = cv2.remap(
        frame_a,
        (grid_x - alpha * flow_ab[..., 0]).astype(np.float32),
        (grid_y - alpha * flow_ab[..., 1]).astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    from_b = cv2.remap(
        frame_b,
        (grid_x - (1 - alpha) * flow_ba[..., 0]).astype(np.float32),
        (grid_y - (1 - alpha) * flow_ba[..., 1]).astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return cv2.addWeighted(from_a, 1 - alpha, from_b, alpha, 0)


def dense_flow(frame_from: np.ndarray, frame_to: np.ndarray) -> np.ndarray:
    """Dense optical flow (DIS, medium preset) taking `frame_from` to `frame_to`, shape (H, W, 2)."""
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    gray_from = cv2.cvtColor(frame_from, cv2.COLOR_BGR2GRAY)
    gray_to = cv2.cvtColor(frame_to, cv2.COLOR_BGR2GRAY)
    return dis.calc(gray_from, gray_to, None)


class FrameReel:
    """Frames with their times, queryable at any time with motion-compensated interpolation.

    Lets a 10 fps clip be shown at 20 Hz sim steps, or slowed several times, without
    stuttering. Flow between neighbouring frames is computed on first use and cached.
    """

    def __init__(self, frames_bgr: list[np.ndarray], times_s: np.ndarray) -> None:
        """Hold decoded frames.

        Args:
            frames_bgr: frames in time order.
            times_s: shape (N,), strictly increasing presentation times.
        """
        if len(frames_bgr) != len(times_s) or len(frames_bgr) == 0:
            raise ValueError("need one time per frame and at least one frame")
        self._frames = frames_bgr
        self._times_s = np.asarray(times_s, dtype=float)
        self._flows: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    @classmethod
    def from_video(cls, video_path: Path) -> FrameReel:
        """Decode a whole (short) clip with its container timestamps."""
        pairs = list(read_frames_with_timestamps(video_path))
        return cls([f for f, _ in pairs], np.array([t for _, t in pairs]))

    @property
    def duration_s(self) -> float:
        """Time span from the first to the last frame."""
        return float(self._times_s[-1] - self._times_s[0])

    @property
    def start_s(self) -> float:
        """Time of the first frame."""
        return float(self._times_s[0])

    def frame_at(self, t_s: float, interpolate: bool = True) -> np.ndarray:
        """The frame at time `t_s`, clamped to the clip.

        Args:
            t_s: time in the clip's own timeline (same origin as `times_s`).
            interpolate: synthesise in-between frames; False returns the nearest real frame.
        """
        t_s = float(np.clip(t_s, self._times_s[0], self._times_s[-1]))
        right = int(np.searchsorted(self._times_s, t_s, side="right"))
        if right >= len(self._frames):
            return self._frames[-1]
        left = right - 1
        span_s = self._times_s[right] - self._times_s[left]
        alpha = (t_s - self._times_s[left]) / span_s
        if not interpolate:
            return self._frames[right if alpha >= 0.5 else left]
        if left not in self._flows:
            self._flows[left] = (
                dense_flow(self._frames[left], self._frames[right]),
                dense_flow(self._frames[right], self._frames[left]),
            )
        flow_ab, flow_ba = self._flows[left]
        return interpolate_frames(self._frames[left], self._frames[right], alpha, flow_ab, flow_ba)


def write_side_by_side_video(
    left_path: Path,
    right_path: Path,
    out_path: Path,
    labels: tuple[str, str],
    fps: float = 20.0,
    height: int = 480,
) -> Path:
    """Two videos side by side on one clock; the shorter one holds its last frame.

    Args:
        left_path: video shown on the left.
        right_path: video shown on the right.
        out_path: output .mp4 (H.264); the parent folder is created.
        labels: captions of the left and right video.
        fps: output rate; the longer video sets the duration.
        height: output height in pixels.

    Returns:
        `out_path`.
    """
    reels = [FrameReel.from_video(path) for path in (left_path, right_path)]
    duration_s = max(reel.duration_s for reel in reels)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames = hstack_resized(
        [reels[0].frame_at(reels[0].start_s), reels[1].frame_at(reels[1].start_s)], height
    )
    with VideoWriter(out_path, fps, (frames.shape[1], frames.shape[0])) as writer:
        for t_s in np.arange(0.0, duration_s, 1.0 / fps):
            panels = [reel.frame_at(reel.start_s + t_s, interpolate=False).copy() for reel in reels]
            for panel, label, reel in zip(panels, labels, reels, strict=True):
                ended = t_s > reel.duration_s
                draw_outlined_text(
                    panel, f"{label}{' (ended)' if ended else ''}", (10, panel.shape[0] - 14), 0.8
                )
            frame = hstack_resized(panels, height)
            draw_outlined_text(
                frame, f"t = {t_s:.1f} s", (frame.shape[1] // 2 - 50, frame.shape[0] - 14), 0.8
            )
            writer.write(frame)
    return out_path


def write_phase_aligned_video(
    left_path: Path,
    right_path: Path,
    out_path: Path,
    labels: tuple[str, str],
    left_events_s: list[float],
    right_events_s: list[float],
    fps: float = 20.0,
    height: int = 480,
    play_on_after_last_event: bool = False,
    slowdowns: tuple[float, float] = (1.0, 1.0),
    pause_faster: bool = False,
) -> Path:
    """Two videos side by side with their event boundaries lined up.

    Between two consecutive events the output lasts as long as the slower side needs. By default the faster
    side is slowed to fit; with `pause_faster` it plays at its own speed and then waits (holds its last frame
    of the segment) until the other side reaches the event. The speed of each side is drawn on it.

    Args:
        left_path: video shown on the left.
        right_path: video shown on the right.
        out_path: output .mp4 (H.264); the parent folder is created.
        labels: captions of the left and right video.
        left_events_s: increasing times in the left video (its own clock, from 0), one per shared event,
            first = 0. Equal consecutive times mean the video waits there (e.g. while the other moves).
        right_events_s: the same events in the right video; same length as `left_events_s`.
        fps: output rate.
        height: output height in pixels.
        play_on_after_last_event: after the last event both videos play at their own speed until the longer
            one ends, the shorter holding its last frame; the events then need not end at the videos'
            durations, which lets only the start of a clip be aligned.
        slowdowns: how many times slower than recorded each side plays (the human clip slowed to the
            robot's speed limit is `(factor, 1.0)`).
        pause_faster: see above.

    Returns:
        `out_path`.
    """
    if len(left_events_s) != len(right_events_s) or len(left_events_s) < 2:
        raise ValueError("need the same number (>= 2) of events for both videos")
    reels = [FrameReel.from_video(path) for path in (left_path, right_path)]
    events = (np.asarray(left_events_s, dtype=float), np.asarray(right_events_s, dtype=float))
    needed = [np.diff(marks) * slowdown for marks, slowdown in zip(events, slowdowns, strict=True)]
    lengths = np.maximum(needed[0], needed[1]).clip(min=1e-6)
    starts = np.concatenate([[0.0], np.cumsum(lengths)])
    tail_s = max(
        (reel.duration_s - marks[-1]) * slowdown
        for reel, marks, slowdown in zip(reels, events, slowdowns, strict=True)
    )
    total_s = starts[-1] + (max(tail_s, 0.0) if play_on_after_last_event else 0.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    first = hstack_resized([reel.frame_at(reel.start_s) for reel in reels], height)
    with VideoWriter(out_path, fps, (first.shape[1], first.shape[0])) as writer:
        for t_s in np.arange(0.0, total_s, 1.0 / fps):
            in_tail = t_s >= starts[-1]
            segment = min(int(np.searchsorted(starts, t_s, side="right")) - 1, len(lengths) - 1)
            panels = []
            for reel, label, marks, slowdown in zip(reels, labels, events, slowdowns, strict=True):
                if in_tail:
                    own_t, speed = marks[-1] + (t_s - starts[-1]) / slowdown, 1.0 / slowdown
                else:
                    elapsed = t_s - starts[segment]
                    own_s = marks[segment + 1] - marks[segment]
                    if pause_faster:
                        progress_s = min(elapsed / slowdown, own_s)
                        speed = 0.0 if progress_s >= own_s else 1.0 / slowdown
                    else:
                        progress_s = elapsed / lengths[segment] * own_s
                        speed = own_s / lengths[segment]
                    own_t = marks[segment] + progress_s
                panel = reel.frame_at(reel.start_s + own_t, interpolate=False).copy()
                note = f"x{speed:.2f}" if speed > 0 else "paused"
                draw_outlined_text(panel, f"{label}  ({note})", (10, panel.shape[0] - 14), 0.8)
                panels.append(panel)
            writer.write(hstack_resized(panels, height))
    return out_path


def slow_down_video(source_path: Path, output_path: Path, factor: float, output_fps: float = 20.0) -> Path:
    """Write a slowed-down copy of a clip; the source file is left untouched.

    Args:
        source_path: input video.
        output_path: where the slowed copy goes (must differ from `source_path`).
        factor: > 1 slows playback by this factor (2 = half speed).
        output_fps: frame rate of the result; in-between frames are interpolated so
            motion stays smooth even when the source has few frames per second.

    Returns:
        `output_path`.
    """
    if output_path.resolve() == source_path.resolve():
        raise ValueError("refusing to overwrite the source clip; give a different output path")
    reel = FrameReel.from_video(source_path)
    height, width = reel.frame_at(reel.start_s).shape[:2]
    num_frames = max(2, int(round(reel.duration_s * factor * output_fps)) + 1)
    with VideoWriter(output_path, output_fps, (width, height)) as writer:
        for k in range(num_frames):
            writer.write(reel.frame_at(reel.start_s + k / output_fps / factor))
    return output_path
