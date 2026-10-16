"""Overlay of an extracted `HumanDemo` on its source clip: markers, hand, TCP path and the gripper state."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from hurobo.camera import CameraIntrinsics
from hurobo.demo import HumanDemo
from hurobo.recording import CaptureDiagnosis
from hurobo.utils.transforms import apply_transform, invert_transform
from hurobo.utils.video import (
    VideoWriter,
    draw_gripper_gauge,
    draw_outlined_text,
    get_video_fps,
    read_frames_with_timestamps,
)

TCP_BGR = (0, 0, 255)
OPEN_TRAIL_BGR = (230, 130, 30)
GRASP_TRAIL_BGR = (0, 0, 230)


class DemoOverlay:
    """Draws, on every frame of a demo's kept stretch, what the dataset holds for that instant.

    Per frame: the detected marker outlines and hand skeleton (from the capture diagnosis), the demo's TCP
    (projected back into the image through the demo's camera pose) with its path so far, the bowl and plate
    centres, a gripper symbol driven by the pinch aperture, and the demo phase.
    """

    def __init__(self, demo: HumanDemo, diagnosis: CaptureDiagnosis, intrinsics: CameraIntrinsics) -> None:
        """Pre-project the demo's TCP and the object centres into the image.

        Args:
            demo: the extracted demo.
            diagnosis: `diagnose_capture` of the demo's source clip (same frame order).
            intrinsics: camera that filmed the clip.
        """
        self._demo = demo
        self._diagnosis = diagnosis
        self._source_width_px = intrinsics.image_width
        t_cam_table = invert_transform(np.asarray(demo.t_table_cam))
        self._tcp_px = intrinsics.project(apply_transform(t_cam_table, np.asarray(demo.tcp_pos_table)))
        self._object_px = {
            name: intrinsics.project(apply_transform(t_cam_table, np.asarray(t_table_object)[:3, 3]))
            for name, t_table_object in (("bowl", demo.t_table_bowl), ("plate", demo.t_table_plate))
        }

    def sample_at(self, t_s: float) -> int:
        """Index of the demo sample nearest to a time in the source clip."""
        return int(np.argmin(np.abs(self._demo.timestamps_s + self._demo.source_start_time_s - t_s)))

    def phase_at(self, sample: int) -> str:
        """'approach', 'carry' or 'retreat', from the demo's grasp and release events."""
        grasp, release = self._demo.grasp_frame_idx, self._demo.release_frame_idx
        if grasp is None or release is None or sample < grasp:
            return "approach"
        return "carry" if sample < release else "retreat"

    def draw(self, frame_bgr: np.ndarray, frame_index: int, sample: int, scale: float) -> None:
        """Annotate one video frame in place.

        Args:
            frame_bgr: frame resized by `scale` relative to the video.
            frame_index: index of the frame in the source clip.
            sample: demo sample shown at this frame (`sample_at`).
            scale: factor from video pixels to `frame_bgr` pixels.
        """
        self._diagnosis.draw_detections(frame_bgr, frame_index, scale)
        grasping = self._demo.is_grasping
        for i in range(1, sample + 1):
            colour = GRASP_TRAIL_BGR if grasping[i] else OPEN_TRAIL_BGR
            cv2.line(
                frame_bgr,
                tuple((self._tcp_px[i - 1] * scale).round().astype(int)),
                tuple((self._tcp_px[i] * scale).round().astype(int)),
                colour,
                2,
                cv2.LINE_AA,
            )
        tcp = tuple((self._tcp_px[sample] * scale).round().astype(int))
        cv2.drawMarker(frame_bgr, tcp, TCP_BGR, cv2.MARKER_CROSS, 30, 3, cv2.LINE_AA)
        for name, pixel in self._object_px.items():
            centre = tuple((pixel * scale).round().astype(int))
            cv2.circle(frame_bgr, centre, 5, (255, 255, 255), -1, cv2.LINE_AA)
            draw_outlined_text(frame_bgr, f"{name} centre", (centre[0] + 8, centre[1] - 6), 0.5)
        height, width = frame_bgr.shape[:2]
        draw_gripper_gauge(
            frame_bgr,
            float(self._demo.pinch_aperture_m[sample]),
            bool(grasping[sample]),
            (width - 170, height - 100),
        )
        draw_outlined_text(frame_bgr, f"phase: {self.phase_at(sample)}", (width - 190, 26), 0.7)

    def annotate_sample(self, frame_bgr: np.ndarray, sample: int) -> None:
        """Annotate, in place, the image stored for one demo sample (e.g. a frame of the exported dataset).

        Args:
            frame_bgr: the sample's image at any resolution of the clip's aspect ratio.
            sample: demo sample the image belongs to.
        """
        # The stored image is the clip frame nearest to the sample's time, so detections come from that frame.
        t_s = self._demo.source_start_time_s + self._demo.timestamps_s[sample]
        frame_index = int(np.argmin(np.abs(self._diagnosis.times_s - t_s)))
        self.draw(frame_bgr, frame_index, sample, frame_bgr.shape[1] / self._source_width_px)

    def write_video(
        self,
        out_path: Path,
        width_px: int = 960,
        slowdown: float = 1.0,
        hold_start_s: float = 0.0,
        fps: float | None = None,
        start_sample: int = 0,
    ) -> Path:
        """Video of the demo's kept stretch with the overlay, optionally slowed and held at the start.

        Args:
            out_path: output .mp4; the parent folder is created.
            width_px: output width; the height keeps the aspect ratio.
            slowdown: how many times slower than recorded to play (1 = real time); every source frame is
                shown for `slowdown` times as long, so no frames are invented.
            hold_start_s: seconds the first frame is held before the motion starts (the human waits for
                the robot).
            fps: output rate; the clip's own by default, 20 when slowed.
            start_sample: demo sample the video starts at (the part before it is left out), e.g. where the
                part the robot replays begins.

        Returns:
            `out_path`.
        """
        out_path.parent.mkdir(parents=True, exist_ok=True)
        start = self._demo.source_start_time_s + float(
            self._demo.timestamps_s[start_sample] - self._demo.timestamps_s[0]
        )
        stop = self._demo.source_start_time_s + float(
            self._demo.timestamps_s[-1] - self._demo.timestamps_s[0]
        )
        video = self._demo.source_video
        out_fps = fps or (get_video_fps(video) if slowdown == 1.0 else 20.0)
        writer = None
        written = 0
        previous = None
        try:
            for index, (frame, t_s) in enumerate(read_frames_with_timestamps(video)):
                if not start <= t_s <= stop:
                    continue
                scale = width_px / frame.shape[1]
                frame = cv2.resize(
                    frame, (width_px, round(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA
                )
                self.draw(frame, index, self.sample_at(t_s), scale)
                if writer is None:
                    writer = VideoWriter(out_path, out_fps, (frame.shape[1], frame.shape[0]))
                else:
                    # the previous frame stays until this one's turn comes
                    due = round((hold_start_s + (t_s - start) * slowdown) * out_fps)
                    for _ in range(due - written):
                        writer.write(previous)
                    written = max(written, due)
                previous = frame
            if writer is not None:
                due = round((hold_start_s + (stop - start) * slowdown) * out_fps)
                for _ in range(due - written):
                    writer.write(previous)
        finally:
            if writer is not None:
                writer.close()
        return out_path
