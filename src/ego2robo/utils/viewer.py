"""Interactive plotly 3D views of Gaussian-splat `.ply` files and SfM reconstructions, for notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from gaussian_renderer import load_ply
from plotly.subplots import make_subplots

from ego2robo.demo import GraspSegmenter
from ego2robo.utils.hand_landmarks import INDEX_TIP, LANDMARK_NAMES, THUMB_TIP, polyline_with_breaks

if TYPE_CHECKING:
    from ego2robo.hand import TrackedFrame

#: Zeroth spherical-harmonic basis constant: colour = 0.5 + C0 * f_dc.
_SH_C0 = 0.28209479177387814


@dataclass
class SplatPoints:
    """Gaussian centres of one splat, ready to plot.

    Attributes:
        xyz: shape (N, 3) centres in the splat's own frame (z up).
        rgb: shape (N, 3) uint8 base colour (SH degree 0 only, no view dependence).
        opacity: shape (N,) in [0, 1].
        total_count: Gaussians in the file before filtering/subsampling.
    """

    xyz: np.ndarray
    rgb: np.ndarray
    opacity: np.ndarray
    total_count: int


def load_splat_points(
    ply_path: Path, max_points: int = 15_000, min_opacity: float = 0.1, seed: int = 0
) -> SplatPoints:
    """Read a 3DGS `.ply`, drop near-transparent Gaussians, and subsample for plotting.

    Args:
        ply_path: 3DGS `.ply`, standard or SuperSplat-compressed (via DISCOVERSE's loader).
        max_points: cap on plotted points; a browser handles ~50k Scatter3d points smoothly.
        min_opacity: Gaussians below this opacity are floaters and are dropped.
        seed: subsampling seed, so a notebook re-run draws the same cloud.

    Returns:
        The filtered, subsampled `SplatPoints`.
    """
    gaussians = load_ply(str(ply_path))
    total = len(gaussians.xyz)
    opacity = np.asarray(gaussians.opacity, dtype=np.float64)
    keep = np.flatnonzero(opacity >= min_opacity)
    if len(keep) > max_points:
        keep = np.sort(np.random.default_rng(seed).choice(keep, size=max_points, replace=False))
    rgb = (np.clip(0.5 + _SH_C0 * gaussians.sh[keep, :3], 0.0, 1.0) * 255).astype(np.uint8)
    return SplatPoints(xyz=np.asarray(gaussians.xyz)[keep], rgb=rgb, opacity=opacity[keep], total_count=total)


def _rgb_strings(rgb: np.ndarray) -> list[str]:
    return [f"rgb({r},{g},{b})" for r, g, b in rgb]


def splat_trace(points: SplatPoints, name: str, marker_size: float = 1.6) -> go.Scatter3d:
    """One plotly trace of splat centres, coloured by their own base colour."""
    return go.Scatter3d(
        # Millimetre rounding keeps the figure JSON (embedded in the notebook) small.
        x=np.round(points.xyz[:, 0], 3),
        y=np.round(points.xyz[:, 1], 3),
        z=np.round(points.xyz[:, 2], 3),
        mode="markers",
        name=name,
        marker={"size": marker_size, "color": _rgb_strings(points.rgb), "opacity": 0.9},
        hoverinfo="skip",
    )


def axis_layout(title: str, height: int = 620) -> dict:
    """Shared 3D layout: metric-aspect axes (`data`), z up, compact margins."""
    return {
        "title": title,
        "height": height,
        "margin": {"l": 0, "r": 0, "t": 40, "b": 0},
        "scene": {
            "aspectmode": "data",
            "xaxis_title": "x [m]",
            "yaxis_title": "y [m]",
            "zaxis_title": "z [m]",
            "camera": {"up": {"x": 0, "y": 0, "z": 1}},
        },
    }


def splat_figure(
    ply_paths: dict[str, Path], max_points: int = 15_000, title: str = "Gaussian splat"
) -> go.Figure:
    """Interactive 3D view of one or more splats, with a dropdown to switch between them.

    Args:
        ply_paths: display name -> `.ply` path. One trace per entry; the dropdown shows one at a time.
        max_points: per-splat plotting cap, see `load_splat_points`.
        title: figure title.

    Returns:
        A `go.Figure`; orbit with the mouse, hover-free for speed.
    """
    figure = go.Figure()
    for index, (name, path) in enumerate(ply_paths.items()):
        points = load_splat_points(path, max_points=max_points)
        trace = splat_trace(points, f"{name} ({len(points.xyz):,}/{points.total_count:,})")
        trace.visible = index == 0
        figure.add_trace(trace)
    buttons = [
        {
            "label": trace.name,
            "method": "update",
            "args": [{"visible": [i == j for j in range(len(figure.data))]}],
        }
        for i, trace in enumerate(figure.data)
    ]
    figure.update_layout(**axis_layout(title), updatemenus=[{"buttons": buttons, "x": 0.0, "y": 1.0}])
    return figure


def camera_frustum_trace(
    t_world_cam_list: list[np.ndarray], scale: float, name: str = "cameras"
) -> go.Scatter3d:
    """Wireframe pyramids for OpenCV cameras (+z forward) given as T_world_cam.

    Args:
        t_world_cam_list: 4x4 camera-to-world transforms.
        scale: pyramid depth in world units.
        name: legend entry.
    """
    corners_cam = scale * np.array([[-0.6, -0.4, 1.0], [0.6, -0.4, 1.0], [0.6, 0.4, 1.0], [-0.6, 0.4, 1.0]])
    polylines: list[np.ndarray] = []
    for t_world_cam in t_world_cam_list:
        centre = t_world_cam[:3, 3]
        corners = corners_cam @ t_world_cam[:3, :3].T + centre
        polylines.append(np.vstack([corners, corners[:1]]))
        polylines.extend(np.vstack([centre, corner]) for corner in corners)
    # NaN rows break the single Scatter3d line into separate segments.
    gap = np.full((1, 3), np.nan)
    xyz = np.vstack([np.vstack([line, gap]) for line in polylines])
    xs, ys, zs = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", name=name, line={"width": 2, "color": "crimson"})


def sfm_figure(
    points_world: np.ndarray,
    colors: np.ndarray,
    t_world_cam_list: list[np.ndarray],
    title: str = "SfM reconstruction",
) -> go.Figure:
    """Sparse SfM points plus every registered camera's frustum, interactive.

    Args:
        points_world: shape (P, 3) sparse points.
        colors: shape (P, 3) in [0, 1].
        t_world_cam_list: camera-to-world transforms of the registered keyframes.
        title: figure title.
    """
    rgb = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
    figure = go.Figure(
        [
            go.Scatter3d(
                x=points_world[:, 0],
                y=points_world[:, 1],
                z=points_world[:, 2],
                mode="markers",
                name="sparse points",
                marker={"size": 1.8, "color": _rgb_strings(rgb)},
                hoverinfo="skip",
            ),
            camera_frustum_trace(t_world_cam_list, scale=0.08 * float(np.ptp(points_world, axis=0).max())),
        ]
    )
    figure.update_layout(**axis_layout(title))
    return figure


def splat_statistics(ply_paths: dict[str, Path]) -> pd.DataFrame:
    """Per-splat Gaussian counts and axis-aligned extents (metres, splat frame), unsubsampled.

    Args:
        ply_paths: display name -> `.ply` path.

    Returns:
        One row per splat: total Gaussians, opaque ones, and x/y/z extents of the opaque ones.
    """
    rows = []
    for name, path in ply_paths.items():
        points = load_splat_points(path, max_points=np.iinfo(np.int64).max)
        extent = points.xyz.max(axis=0) - points.xyz.min(axis=0)
        rows.append(
            {
                "splat": name,
                "gaussians": points.total_count,
                "opaque (>0.1)": len(points.xyz),
                "extent x": extent[0],
                "extent y": extent[1],
                "extent z": extent[2],
            }
        )
    return pd.DataFrame(rows).round(3)


def frame_traces(t_parent_frame: np.ndarray, name: str, length: float = 0.12) -> list[go.Scatter3d]:
    """Red/green/blue axis lines of one coordinate frame, as plotly traces.

    Args:
        t_parent_frame: 4x4 pose of the frame in its parent (the plot's) frame.
        name: legend entry; the three axes share one legend group.
        length: axis length in the plot's units.
    """
    origin = t_parent_frame[:3, 3]
    traces = []
    for axis, colour in enumerate(("red", "green", "blue")):
        tip = origin + length * t_parent_frame[:3, axis]
        traces.append(
            go.Scatter3d(
                x=[origin[0], tip[0]],
                y=[origin[1], tip[1]],
                z=[origin[2], tip[2]],
                mode="lines",
                line={"color": colour, "width": 6},
                name=f"{name} {'xyz'[axis]}",
                showlegend=axis == 0,
                legendgroup=name,
            )
        )
    return traces


def frames_figure(
    t_parent_frames: dict[str, np.ndarray], title: str = "Coordinate frames", length: float = 0.12
) -> go.Figure:
    """Interactive 3D view of several named frames, all expressed in one parent frame.

    Args:
        t_parent_frames: frame name -> 4x4 pose in the parent frame,
            e.g. `{"table": np.eye(4), "camera": T_table_cam}`.
        title: figure title.
        length: axis length in the parent frame's units.
    """
    figure = go.Figure(
        [t for name, pose in t_parent_frames.items() for t in frame_traces(pose, name, length)]
    )
    figure.update_layout(**axis_layout(title, height=520))
    return figure


def trajectory_figure(
    positions_table: np.ndarray,
    is_grasping: np.ndarray | None = None,
    title: str = "Hand TCP trajectory (table frame)",
) -> go.Figure:
    """Interactive 3D TCP trajectory, free vs grasping samples coloured differently.

    Args:
        positions_table: shape (N, 3), TCP positions in the table frame.
        is_grasping: shape (N,) bool; if given, free samples are blue and grasping ones red.
        title: figure title.

    Returns:
        A `go.Figure`; hovering a marker shows its sample index and xyz.
    """
    hover = "sample %{customdata}<br>x=%{x:.3f} y=%{y:.3f} z=%{z:.3f} m<extra>%{fullData.name}</extra>"
    sample_idx = np.arange(len(positions_table))

    def markers(mask: np.ndarray, name: str, colour: str) -> go.Scatter3d:
        xyz = positions_table[mask]
        return go.Scatter3d(
            x=xyz[:, 0],
            y=xyz[:, 1],
            z=xyz[:, 2],
            mode="markers",
            name=name,
            customdata=sample_idx[mask],
            marker={"size": 3, "color": colour},
            hovertemplate=hover,
        )

    x, y, z = positions_table.T
    if is_grasping is None:
        traces = [
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines+markers",
                name="TCP",
                customdata=sample_idx,
                line={"width": 3, "color": "royalblue"},
                marker={"size": 3, "color": "royalblue"},
                hovertemplate=hover,
            )
        ]
    else:
        is_grasping = np.asarray(is_grasping, dtype=bool)
        path = go.Scatter3d(
            x=x, y=y, z=z, mode="lines", name="path", line={"width": 2, "color": "gray"}, hoverinfo="skip"
        )
        traces = [
            path,
            markers(~is_grasping, "free", "royalblue"),
            markers(is_grasping, "grasping", "crimson"),
        ]
    figure = go.Figure(traces)
    figure.update_layout(**axis_layout(title))
    return figure


def aperture_figure(
    timestamps_s: np.ndarray,
    pinch_aperture_m: np.ndarray,
    close_threshold_m: float,
    open_threshold_m: float,
    grasp_frame_idx: int | None = None,
    release_frame_idx: int | None = None,
    title: str = "Pinch aperture",
) -> go.Figure:
    """Interactive pinch-aperture time series with the hysteresis band and grasp/release marks.

    Args:
        timestamps_s: shape (N,), seconds from the start of the episode.
        pinch_aperture_m: shape (N,), thumb-index distance in metres.
        close_threshold_m: aperture below this registers a grasp.
        open_threshold_m: aperture above this registers a release.
        grasp_frame_idx: detected grasp-onset sample index, if any.
        release_frame_idx: detected release-onset sample index, if any.
        title: figure title.

    Returns:
        A `go.Figure` with aperture in cm against time; hover shows time and aperture.
    """
    aperture_cm = np.asarray(pinch_aperture_m) * 100
    close_cm, open_cm = close_threshold_m * 100, open_threshold_m * 100
    t_start, t_end = float(timestamps_s[0]), float(timestamps_s[-1])
    y_min = min(float(aperture_cm.min()), close_cm)
    y_max = max(float(aperture_cm.max()), open_cm)

    figure = go.Figure(
        [
            # A filled polygon (not a layout shape) so the band gets a legend entry and can be toggled.
            go.Scatter(
                x=[t_start, t_end, t_end, t_start, t_start],
                y=[close_cm, close_cm, open_cm, open_cm, close_cm],
                mode="lines",
                fill="toself",
                fillcolor="rgba(128,128,128,0.2)",
                line={"width": 0},
                name=f"hysteresis ({close_cm:.1f}-{open_cm:.1f} cm)",
                hoverinfo="skip",
            ),
            go.Scatter(
                x=timestamps_s,
                y=aperture_cm,
                mode="lines",
                name="aperture",
                line={"color": "royalblue"},
                customdata=np.arange(len(aperture_cm)),
                hovertemplate="t=%{x:.2f} s<br>%{y:.2f} cm<br>sample %{customdata}<extra></extra>",
            ),
        ]
    )
    for idx, name, colour in (
        (grasp_frame_idx, "grasp", "crimson"),
        (release_frame_idx, "release", "seagreen"),
    ):
        if idx is None:
            continue
        t_event = float(timestamps_s[idx])
        figure.add_trace(
            go.Scatter(
                x=[t_event, t_event],
                y=[y_min, y_max],
                mode="lines",
                name=name,
                line={"color": colour, "dash": "dash"},
                hovertemplate=f"{name}: sample {idx}, t={t_event:.2f} s<extra></extra>",
            )
        )
    figure.update_layout(
        title=title,
        height=360,
        margin={"l": 50, "r": 10, "t": 40, "b": 40},
        xaxis_title="time [s]",
        yaxis_title="aperture [cm]",
        hovermode="x unified",
    )
    return figure


def tracking_figure(tracked: list[TrackedFrame], title: str = "Hand tracking per frame") -> go.Figure:
    """TCP depth, pinch aperture and PnP reprojection error over a tracked clip, interactive.

    Frames without a detected hand are gaps in the curves.

    Args:
        tracked: output of `track_video`.
        title: figure title.
    """
    t = np.array([f.timestamp_s for f in tracked])

    def series(value) -> np.ndarray:
        return np.array([value(f.result) if f.result else np.nan for f in tracked])

    curves = {
        "TCP depth z_cam (m)": series(lambda r: r.tcp_pos_cam[2]),
        "pinch aperture (cm)": series(lambda r: r.pinch_aperture_m * 100),
        "reprojection error (px)": series(lambda r: r.reprojection_error_px),
    }
    figure = make_subplots(rows=1, cols=3, subplot_titles=list(curves), horizontal_spacing=0.07)
    for column, y in enumerate(curves.values(), start=1):
        figure.add_trace(go.Scatter(x=t, y=y, mode="lines+markers", showlegend=False), row=1, col=column)
        figure.update_xaxes(title_text="time (s)", row=1, col=column)
    figure.update_layout(title=title, height=340, margin={"l": 40, "r": 10, "t": 70, "b": 40})
    return figure


def pinch_figure(
    tracked: list[TrackedFrame],
    close_threshold_m: float,
    open_threshold_m: float,
    title: str = "How the pinch aperture is measured",
) -> go.Figure:
    """Step through a tracked clip: the 21-landmark hand, the thumb-index segment, and its length over time.

    Left, the metric hand landmarks in the hand's own frame (the PnP object points). The red
    segment between the thumb tip (orange) and index tip (green) is the pinch aperture, its
    midpoint is the TCP. Right, that length per frame against the hysteresis band; the dot is
    red while `GraspSegmenter` says "grasping" and blue while it says "open".

    Args:
        tracked: output of `track_video`; frames without a hand show an empty hand and a gap.
        close_threshold_m: aperture below this closes the grasp state.
        open_threshold_m: aperture above this opens it.
        title: figure title.

    Returns:
        A `go.Figure` with a frame slider and a play button.

    Raises:
        ValueError: if no frame has recorded landmarks.
    """
    with_hand = [i for i, f in enumerate(tracked) if f.result and f.result.landmarks_hand is not None]
    if not with_hand:
        raise ValueError("no frame has hand landmarks; track the clip with HandTcpTracker first")
    t = np.array([f.timestamp_s for f in tracked])
    aperture_cm = np.array([f.result.pinch_aperture_m * 100 if f.result else np.nan for f in tracked])
    closed = GraspSegmenter(close_threshold_m, open_threshold_m).segment(aperture_cm / 100)
    close_cm, open_cm = close_threshold_m * 100, open_threshold_m * 100

    def plot_axes(landmarks_hand: np.ndarray) -> np.ndarray:
        # MediaPipe's world axes follow the image (x right, y down, z depth); plot (x, z, -y)
        # so the hand stands upright, in cm.
        return landmarks_hand[:, [0, 2, 1]] * 100 * np.array([1.0, 1.0, -1.0])

    def frame_traces(i: int) -> list:
        """The seven traces that change per frame, in figure order (bones ... current dot)."""
        result = tracked[i].result
        if result is None or result.landmarks_hand is None:
            nothing = [None]
            return [
                go.Scatter3d(x=nothing, y=nothing, z=nothing),
                go.Scatter3d(x=nothing, y=nothing, z=nothing),
                go.Scatter3d(x=nothing, y=nothing, z=nothing),
                go.Scatter3d(x=nothing, y=nothing, z=nothing),
                go.Scatter3d(x=nothing, y=nothing, z=nothing),
                go.Scatter3d(x=nothing, y=nothing, z=nothing),
                go.Scatter(x=[t[i]], y=[None]),
            ]
        hand = plot_axes(result.landmarks_hand)
        bones = polyline_with_breaks(hand)
        thumb, index = hand[THUMB_TIP], hand[INDEX_TIP]
        middle = (thumb + index) / 2
        return [
            go.Scatter3d(x=bones[:, 0], y=bones[:, 1], z=bones[:, 2]),
            go.Scatter3d(x=hand[:, 0], y=hand[:, 1], z=hand[:, 2]),
            go.Scatter3d(x=[thumb[0]], y=[thumb[1]], z=[thumb[2]]),
            go.Scatter3d(x=[index[0]], y=[index[1]], z=[index[2]]),
            go.Scatter3d(x=[thumb[0], index[0]], y=[thumb[1], index[1]], z=[thumb[2], index[2]]),
            go.Scatter3d(
                x=[middle[0]],
                y=[middle[1]],
                z=[middle[2]],
                text=[f"aperture {aperture_cm[i]:.1f} cm"],
                hovertext=[f"TCP (midpoint), aperture {aperture_cm[i]:.2f} cm"],
            ),
            go.Scatter(
                x=[t[i]], y=[aperture_cm[i]], marker={"color": "crimson" if closed[i] else "royalblue"}
            ),
        ]

    first = frame_traces(with_hand[0])
    figure = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "xy"}]],
        column_widths=[0.55, 0.45],
        horizontal_spacing=0.06,
    )
    names = list(LANDMARK_NAMES)
    static_3d = [
        {"mode": "lines", "line": {"color": "gray", "width": 4}, "name": "bones", "hoverinfo": "skip"},
        {
            "mode": "markers",
            "marker": {"size": 3, "color": "black"},
            "name": "21 landmarks",
            "customdata": names,
            "hovertemplate": "%{customdata}<extra></extra>",
        },
        {"mode": "markers", "marker": {"size": 9, "color": "darkorange"}, "name": "thumb tip (4)"},
        {"mode": "markers", "marker": {"size": 9, "color": "green"}, "name": "index tip (8)"},
        {
            "mode": "lines",
            "line": {"color": "crimson", "width": 7},
            "name": "thumb-index segment",
            "hoverinfo": "skip",
        },
        {
            "mode": "markers+text",
            "marker": {"size": 6, "color": "black", "symbol": "diamond"},
            "textposition": "top center",
            "name": "TCP",
        },
    ]
    for trace, style in zip(first[:6], static_3d, strict=True):
        trace.update(**style)
        figure.add_trace(trace, row=1, col=1)
    # Static curve and hysteresis band, then the moving dot. The band is a filled polygon so
    # it gets a legend entry and can be toggled.
    figure.add_trace(
        go.Scatter(
            x=[t[0], t[-1], t[-1], t[0], t[0]],
            y=[close_cm, close_cm, open_cm, open_cm, close_cm],
            mode="lines",
            fill="toself",
            fillcolor="rgba(128,128,128,0.2)",
            line={"width": 0},
            name=f"hysteresis ({close_cm:.0f}-{open_cm:.0f} cm)",
            hoverinfo="skip",
        ),
        row=1,
        col=2,
    )
    figure.add_trace(
        go.Scatter(
            x=t,
            y=aperture_cm,
            mode="lines+markers",
            name="aperture",
            line={"color": "lightgray"},
            marker={"size": 4},
        ),
        row=1,
        col=2,
    )
    dot = first[6]
    dot.update(mode="markers", marker={"size": 14, "color": dot.marker.color}, name="now", showlegend=False)
    figure.add_trace(dot, row=1, col=2)
    moving = [0, 1, 2, 3, 4, 5, 8]

    figure.frames = [go.Frame(data=frame_traces(i), traces=moving, name=str(i)) for i in range(len(tracked))]
    all_points = np.vstack([plot_axes(tracked[i].result.landmarks_hand) for i in with_hand])
    half = float(np.abs(all_points).max()) * 1.1
    axis = {"range": [-half, half], "backgroundcolor": "rgba(0,0,0,0)"}
    steps = [
        {
            "label": f"{t[i]:.1f}s",
            "method": "animate",
            "args": [[str(i)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}],
        }
        for i in range(len(tracked))
    ]
    figure.update_layout(
        title=title,
        height=520,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        scene={
            "xaxis": {**axis, "title": "x right (cm)"},
            "yaxis": {**axis, "title": "z depth (cm)"},
            "zaxis": {**axis, "title": "up = -y (cm)"},
            "aspectmode": "cube",
        },
        xaxis2={"title": "time (s)"},
        yaxis2={"title": "aperture (cm)"},
        sliders=[{"active": with_hand[0], "steps": steps, "currentvalue": {"prefix": "t = "}}],
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.0,
                "y": -0.05,
                "buttons": [
                    {
                        "label": "play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 200, "redraw": True}, "fromcurrent": True}],
                    }
                ],
            }
        ],
    )
    return figure
