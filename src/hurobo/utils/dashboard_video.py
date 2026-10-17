"""Time-synced dashboard video: camera view, 3D TCP path and pinch aperture on one clock."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from hurobo.demo import GraspSegmenter
from hurobo.utils.hand_landmarks import INDEX_TIP, THUMB_TIP, polyline_with_breaks
from hurobo.utils.video import VideoWriter, draw_tcp_overlay

if TYPE_CHECKING:
    from hurobo.camera import CameraIntrinsics
    from hurobo.hand import TrackedFrame

_BLUE, _RED, _GREY = "royalblue", "crimson", "0.7"
_THUMB, _INDEX = "darkorange", "green"  # same hues as the image overlay
_SERIES_COLOURS = ("tab:blue", "tab:orange", "tab:green", "tab:purple")


@dataclass
class SignalPanel:
    """A multi-series time series shown with a time cursor next to the 3D path.

    Attributes:
        title: panel title.
        values: shape (N, K), one column per series, aligned with the dashboard's timestamps.
        labels: K series names.
        right_axis_labels: names (from `labels`) drawn against a second y-axis, for a series
            on a very different scale (e.g. a 0/1 grasp flag next to millimetre displacements).
        y_label: label of the left y-axis.
        right_y_label: label of the right y-axis.
    """

    title: str
    values: np.ndarray
    labels: Sequence[str]
    right_axis_labels: Sequence[str] = ()
    y_label: str = ""
    right_y_label: str = ""


class SyncedDashboard:
    """Draws one dashboard frame per time step: image | 3D TCP path | aperture over time.

    Top row: image and 3D path. Bottom row: the aperture curve plus any extra `SignalPanel`s,
    side by side. The static parts (full path, full curves, axis limits) are drawn once; each
    `render` call only moves the time cursors, so every panel shows the same instant.
    """

    def __init__(
        self,
        frames_bgr: list[np.ndarray],
        timestamps_s: np.ndarray,
        path_xyz: np.ndarray,
        aperture_m: np.ndarray,
        *,
        image_title: str = "camera view",
        path_title: str = "TCP path (m)",
        path_axis_labels: tuple[str, str, str] = ("x", "y", "z"),
        grasp_mask: np.ndarray | None = None,
        close_threshold_m: float | None = None,
        open_threshold_m: float | None = None,
        signal_panels: Sequence[SignalPanel] = (),
        hand_landmarks_xyz: np.ndarray | None = None,
        size_px: tuple[int, int] = (1280, 720),
        aperture_title: str = "pinch aperture",
        aperture_label: str = "aperture (cm)",
    ) -> None:
        """Build the figure from a whole clip or episode.

        Args:
            frames_bgr: N images (H, W, 3) BGR, already annotated if wanted.
            timestamps_s: shape (N,), increasing seconds.
            path_xyz: shape (N, 3) TCP positions, already in the axes to plot (NaN = no sample).
            aperture_m: shape (N,) thumb-index distance in metres (NaN = no sample).
            image_title: title of the image panel.
            path_title: title of the 3D panel; say which frame the path is in.
            path_axis_labels: labels of the three plotted axes.
            grasp_mask: shape (N,) bool; grasping samples are drawn red on the path and shaded
                in the aperture panel.
            close_threshold_m: lower edge of the grasp hysteresis band, drawn if both are given.
            open_threshold_m: upper edge of the grasp hysteresis band.
            signal_panels: extra time series (e.g. observation.state, action) drawn after the
                aperture panel, each with a cursor synced to the same clock.
            hand_landmarks_xyz: shape (N, 21, 3) hand landmarks in the same axes as `path_xyz`
                (NaN = no hand); the 3D panel then draws the whole hand pose, with the pinch
                fingertips and the segment between them.
            size_px: (width, height) of the output frames; keep both even for H.264.
            aperture_title: title of the aperture panel ("gripper opening" for a robot).
            aperture_label: y-axis label of the aperture panel.

        Raises:
            ValueError: if the inputs differ in length or there is no valid sample at all.
        """
        count = len(frames_bgr)
        if not (len(timestamps_s) == len(path_xyz) == len(aperture_m) == count):
            raise ValueError("frames, timestamps, path and aperture must have the same length")
        if np.isnan(path_xyz).all():
            raise ValueError("no valid TCP sample; nothing to plot")
        self._frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
        self._t_s = np.asarray(timestamps_s, dtype=float)
        self._path = np.asarray(path_xyz, dtype=float)
        self._aperture_cm = np.asarray(aperture_m, dtype=float) * 100
        self._hand = None if hand_landmarks_xyz is None else np.asarray(hand_landmarks_xyz, dtype=float)
        if self._hand is not None and self._hand.shape != (count, 21, 3):
            raise ValueError(f"hand_landmarks_xyz must have shape ({count}, 21, 3), got {self._hand.shape}")
        self._grasp_path = (
            None
            if grasp_mask is None
            else np.where(np.asarray(grasp_mask, dtype=bool)[:, None], self._path, np.nan)
        )

        dpi = 100
        self._figure = Figure(figsize=(size_px[0] / dpi, size_px[1] / dpi), dpi=dpi)
        self._canvas = FigureCanvasAgg(self._figure)
        # Two grids so the top row (image, 3D) is not forced onto the bottom row's column gaps.
        margins = {"left": 0.05, "right": 0.94}
        top_grid = self._figure.add_gridspec(
            1, 2, top=0.94, bottom=0.46, wspace=0.05, width_ratios=[1.2, 1], **margins
        )
        bottom_grid = self._figure.add_gridspec(
            1, 1 + len(signal_panels), top=0.34, bottom=0.1, wspace=0.4, **margins
        )
        self._build_image_panel(self._figure.add_subplot(top_grid[0, 0]), image_title)
        self._build_path_panel(
            self._figure.add_subplot(top_grid[0, 1], projection="3d"), path_title, path_axis_labels
        )
        self._cursors: list = []
        self._dots: list[tuple] = []  # (line, series values) pairs that follow the cursor
        self._build_aperture_panel(
            self._figure.add_subplot(bottom_grid[0, 0]),
            grasp_mask,
            close_threshold_m,
            open_threshold_m,
            aperture_title,
            aperture_label,
        )
        for k, panel in enumerate(signal_panels, start=1):
            self._build_signal_panel(self._figure.add_subplot(bottom_grid[0, k]), panel)

    def _build_image_panel(self, axes, title: str) -> None:
        axes.set_title(title)
        axes.axis("off")
        self._image = axes.imshow(self._frames_rgb[0])

    def _build_path_panel(self, axes, title: str, labels: tuple[str, str, str]) -> None:
        axes.set_title(title)
        axes.plot(*self._path.T, color=_GREY, lw=1.5)
        (self._trail,) = axes.plot([], [], [], color=_BLUE, lw=2.5)
        (self._grasp_trail,) = axes.plot([], [], [], color=_RED, lw=4)
        (self._head,) = axes.plot([], [], [], "o", color="black", ms=8)
        if self._hand is not None:
            (self._skeleton,) = axes.plot([], [], [], color="0.25", lw=1.2)
            (self._pinch_line,) = axes.plot([], [], [], color=_RED, lw=3)
            (self._thumb_tip,) = axes.plot([], [], [], "o", color=_THUMB, ms=7)
            (self._index_tip,) = axes.plot([], [], [], "o", color=_INDEX, ms=7)
        extent_points = (
            self._path if self._hand is None else np.vstack([self._path, self._hand.reshape(-1, 3)])
        )
        valid = extent_points[~np.isnan(extent_points).any(axis=1)]
        low, high = valid.min(axis=0), valid.max(axis=0)
        pad = np.maximum(high - low, 0.02) * 0.1
        axes.set_xlim(low[0] - pad[0], high[0] + pad[0])
        axes.set_ylim(low[1] - pad[1], high[1] + pad[1])
        axes.set_zlim(low[2] - pad[2], high[2] + pad[2])
        # True proportions (metres on every axis), zoomed so the small path fills the panel.
        axes.set_box_aspect(np.maximum(high - low + 2 * pad, 0.02), zoom=1.3)
        axes.locator_params(nbins=3)  # the default ticks overlap on a ~10 cm path
        axes.set_xlabel(labels[0])
        axes.set_ylabel(labels[1])
        axes.set_zlabel(labels[2])

    def _build_aperture_panel(
        self,
        axes,
        grasp_mask: np.ndarray | None,
        close_threshold_m: float | None,
        open_threshold_m: float | None,
        title: str,
        y_label: str,
    ) -> None:
        axes.set_title(title)
        axes.set_xlabel("time (s)")
        axes.set_ylabel(y_label)
        if close_threshold_m is not None and open_threshold_m is not None:
            axes.axhspan(close_threshold_m * 100, open_threshold_m * 100, color=_GREY, alpha=0.3)
        if grasp_mask is not None:
            axes.fill_between(
                self._t_s,
                0,
                1,
                where=np.asarray(grasp_mask, dtype=bool),
                transform=axes.get_xaxis_transform(),  # y in axes fraction: full-height band
                color=_RED,
                alpha=0.12,
                step="mid",
            )
        axes.plot(self._t_s, self._aperture_cm, color=_GREY, lw=1.5)
        (self._aperture_so_far,) = axes.plot([], [], color=_BLUE, lw=2.5)
        self._cursors.append(axes.axvline(self._t_s[0], color=_RED, lw=1.5))
        (aperture_dot,) = axes.plot([], [], "o", color="black", ms=7)
        self._dots.append((aperture_dot, self._aperture_cm))
        axes.set_xlim(self._t_s[0], self._t_s[-1])
        axes.margins(y=0.15)

    def _build_signal_panel(self, axes, panel: SignalPanel) -> None:
        axes.set_title(panel.title)
        axes.set_xlabel("time (s)")
        axes.set_ylabel(panel.y_label)
        right_axes = axes.twinx() if panel.right_axis_labels else None
        if right_axes is not None:
            right_axes.set_ylabel(panel.right_y_label)
        handles = []
        for k, label in enumerate(panel.labels):
            on_right = label in panel.right_axis_labels
            colour = _SERIES_COLOURS[k % len(_SERIES_COLOURS)]
            values = np.asarray(panel.values)[:, k]
            (line,) = (right_axes if on_right else axes).plot(
                self._t_s, values, color=colour, lw=1.8, ls="--" if on_right else "-", label=label
            )
            (dot,) = (right_axes if on_right else axes).plot([], [], "o", color=colour, ms=6)
            self._dots.append((dot, values))
            handles.append(line)
        axes.legend(handles=handles, fontsize=7, loc="best")
        self._cursors.append(axes.axvline(self._t_s[0], color=_RED, lw=1.5))
        axes.set_xlim(self._t_s[0], self._t_s[-1])

    @property
    def frame_size(self) -> tuple[int, int]:
        """(width, height) of the rendered frames, in pixels."""
        return self._canvas.get_width_height()

    def __len__(self) -> int:
        return len(self._t_s)

    def render(self, index: int) -> np.ndarray:
        """Dashboard frame for time step `index`.

        Returns:
            shape (H, W, 3) BGR frame.
        """
        self._image.set_data(self._frames_rgb[index])
        so_far = self._path[: index + 1]
        self._trail.set_data_3d(*so_far.T)
        # A step without a sample is NaN here, so the head and dot vanish with it.
        self._head.set_data_3d(*so_far[-1:].T)
        if self._grasp_path is not None:
            self._grasp_trail.set_data_3d(*self._grasp_path[: index + 1].T)
        if self._hand is not None:
            hand = self._hand[index]
            self._skeleton.set_data_3d(*polyline_with_breaks(hand).T)
            self._pinch_line.set_data_3d(*hand[[THUMB_TIP, INDEX_TIP]].T)
            self._thumb_tip.set_data_3d(*hand[THUMB_TIP : THUMB_TIP + 1].T)
            self._index_tip.set_data_3d(*hand[INDEX_TIP : INDEX_TIP + 1].T)
        self._aperture_so_far.set_data(self._t_s[: index + 1], self._aperture_cm[: index + 1])
        for cursor in self._cursors:
            cursor.set_xdata([self._t_s[index]] * 2)
        for dot, values in self._dots:
            dot.set_data([self._t_s[index]], [values[index]])
        self._canvas.draw()
        rgba = np.asarray(self._canvas.buffer_rgba())
        return cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2BGR)


def write_dashboard_video(dashboard: SyncedDashboard, video_path: Path, fps: float) -> Path:
    """Render every step of a dashboard to an H.264 video.

    Args:
        dashboard: the built dashboard.
        video_path: output .mp4 (browser-playable).
        fps: playback rate; the source's own rate plays it in real time.

    Returns:
        `video_path`.
    """
    with VideoWriter(video_path, fps, dashboard.frame_size) as writer:
        for index in range(len(dashboard)):
            writer.write(dashboard.render(index))
    return video_path


def tracking_dashboard(
    tracked: list[TrackedFrame],
    intrinsics: CameraIntrinsics,
    close_threshold_m: float | None = None,
    open_threshold_m: float | None = None,
) -> SyncedDashboard:
    """Dashboard of a raw tracked clip: hand overlay | hand pose + TCP path in 3D | aperture.

    When both thresholds are given the raw aperture is run through `GraspSegmenter`, so the
    grasping state the video shows is exactly the hysteresis logic of the extraction pipeline
    (minus the resampling and smoothing `DemoExtractor` adds).

    Args:
        tracked: output of `track_video`; frames without a hand are gaps in the curves.
        intrinsics: camera that filmed the clip, for projecting the TCP in the overlay.
        close_threshold_m: lower edge of the grasp hysteresis band to draw, if any.
        open_threshold_m: upper edge of the grasp hysteresis band to draw, if any.
    """

    def to_plot_axes(points_cam: np.ndarray) -> np.ndarray:
        # Plot as (x, z, -y): depth runs along the ground plane and up is up, whereas the
        # camera frame has y pointing down.
        return points_cam[..., [0, 2, 1]] * np.array([1.0, 1.0, -1.0])

    nan_point = np.full(3, np.nan)
    tcp_cam = np.array([f.result.tcp_pos_cam if f.result else nan_point for f in tracked])
    aperture_m = np.array([f.result.pinch_aperture_m if f.result else np.nan for f in tracked])
    landmarks = [f.result.landmarks_cam if f.result else None for f in tracked]
    hand_cam = None
    if any(lm is not None for lm in landmarks):
        hand_cam = to_plot_axes(
            np.array([lm if lm is not None else np.full((21, 3), np.nan) for lm in landmarks])
        )
    grasp_mask = None
    if close_threshold_m is not None and open_threshold_m is not None:
        # NaN apertures compare False both ways, so a missing hand keeps the previous state.
        grasp_mask = GraspSegmenter(close_threshold_m, open_threshold_m).segment(aperture_m)
    return SyncedDashboard(
        [draw_tcp_overlay(f, intrinsics) for f in tracked],
        np.array([f.timestamp_s for f in tracked]),
        to_plot_axes(tcp_cam),
        aperture_m,
        hand_landmarks_xyz=hand_cam,
        grasp_mask=grasp_mask,
        image_title="detected hand, pinch fingertips and TCP",
        path_title="hand pose + TCP path, camera frame (m)",
        path_axis_labels=("x right", "z depth", "up (-y)"),
        close_threshold_m=close_threshold_m,
        open_threshold_m=open_threshold_m,
    )


def write_tracking_dashboard_video(
    tracked: list[TrackedFrame],
    intrinsics: CameraIntrinsics,
    video_path: Path,
    fps: float,
    close_threshold_m: float | None = None,
    open_threshold_m: float | None = None,
) -> Path:
    """Video of a tracked clip with overlay, 3D TCP path and aperture curve on one clock.

    Args:
        tracked: output of `track_video`.
        intrinsics: camera that filmed the clip.
        video_path: output .mp4 (H.264, browser-playable).
        fps: playback rate; the clip's own frame rate plays it in real time.
        close_threshold_m: lower edge of the grasp hysteresis band to draw, if any.
        open_threshold_m: upper edge of the grasp hysteresis band to draw, if any.

    Returns:
        `video_path`.
    """
    dashboard = tracking_dashboard(tracked, intrinsics, close_threshold_m, open_threshold_m)
    return write_dashboard_video(dashboard, video_path, fps)
