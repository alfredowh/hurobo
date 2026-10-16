"""Interactive plotly 3D views of Gaussian-splat `.ply` files and SfM reconstructions, for notebooks."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from gaussian_renderer import load_ply
from plotly.subplots import make_subplots

from hurobo.demo import GraspSegmenter
from hurobo.utils.hand_landmarks import INDEX_TIP, LANDMARK_NAMES, THUMB_TIP, polyline_with_breaks

if TYPE_CHECKING:
    from hurobo.capture_plan import CameraPlacement, PlannedClip
    from hurobo.clip_clean import ClipCleaner, ClipCut, ClipSignals
    from hurobo.hand import TrackedFrame
    from hurobo.real_setup import RealSetup
    from hurobo.recording import CaptureDiagnosis
    from hurobo.scan import CropCylinder
    from hurobo.scan_crop import FloorPlane
    from hurobo.scan_sweep import SweepResult
    from hurobo.teleop_app import TeleopTrace

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


def orbit_figure(
    images: list[np.ndarray],
    title: str = "Splat orbit render",
    height: int = 560,
    labels: list[str] | None = None,
) -> go.Figure:
    """Rendered views as a slider-driven image, one frame per camera (by default per azimuth).

    Args:
        images: (H, W, 3) uint8 RGB views in azimuth order (see `hurobo.splat.render_splat_orbit`).
        title: figure title.
        height: figure height in pixels.
        labels: slider label per image, for views that are not an orbit; default is the azimuth.

    Returns:
        A `go.Figure`; drag the slider to orbit the object.
    """

    def jpeg_trace(image: np.ndarray) -> go.Image:
        # JPEG data URIs keep a many-view figure small enough to embed in a notebook.
        ok, buffer = cv2.imencode(".jpg", image[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
        assert ok
        return go.Image(source="data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode())

    figure = go.Figure(
        data=[jpeg_trace(images[0])],
        frames=[go.Frame(data=[jpeg_trace(image)], name=str(i)) for i, image in enumerate(images)],
    )
    steps = [
        {
            "label": labels[i] if labels else f"{round(360 * i / len(images))}°",
            "method": "animate",
            "args": [[str(i)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}],
        }
        for i in range(len(images))
    ]
    figure.update_layout(
        title=title,
        height=height,
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        sliders=[{"steps": steps, "currentvalue": {"prefix": "view: " if labels else "azimuth: "}}],
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
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


def _cylinder_wireframe(cylinder: CropCylinder, name: str, colour: str, segments: int = 48) -> go.Scatter3d:
    """Top and bottom circles of a crop cylinder plus four vertical edges, as one broken polyline."""
    angle = np.linspace(0.0, 2 * np.pi, segments + 1)
    cx, cy = cylinder.centre_xy
    xs, ys = cx + cylinder.radius * np.cos(angle), cy + cylinder.radius * np.sin(angle)
    x, y, z = [], [], []
    for height in (cylinder.z_min, cylinder.z_max):
        x += [*xs, None]
        y += [*ys, None]
        z += [*np.full(len(xs), height), None]
    for k in range(0, segments, segments // 4):
        x += [xs[k], xs[k], None]
        y += [ys[k], ys[k], None]
        z += [cylinder.z_min, cylinder.z_max, None]
    return go.Scatter3d(x=x, y=y, z=z, mode="lines", name=name, line={"color": colour, "width": 5})


def crop_figure(
    points: np.ndarray,
    colors: np.ndarray,
    cylinders: dict[str, CropCylinder],
    floor: FloorPlane | None = None,
    title: str = "Object crop",
    max_points: int = 2500,
) -> go.Figure:
    """Sparse points with one or more crop cylinders (and the fitted floor), to judge a crop by eye.

    Args:
        points: shape (P, 3) sparse points in the cylinders' z-up frame.
        colors: shape (P, 3) in [0, 1].
        cylinders: label -> crop cylinder; each is drawn as a wireframe.
        floor: optional fitted floor plane, drawn as a translucent sheet under the points.
        title: figure title.
        max_points: points are subsampled to about this many to keep the figure light.
    """
    keep = slice(None, None, max(1, len(points) // max_points))
    rgb = (np.clip(colors[keep], 0, 1) * 255).astype(np.uint8)
    traces: list[go.BaseTraceType] = [
        go.Scatter3d(
            x=points[keep, 0],
            y=points[keep, 1],
            z=points[keep, 2],
            mode="markers",
            name="sparse points",
            marker={"size": 1.8, "color": _rgb_strings(rgb)},
            hoverinfo="skip",
        )
    ]
    palette = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]
    reach = max(c.radius for c in cylinders.values())
    for (label, cylinder), colour in zip(cylinders.items(), palette, strict=False):
        traces.append(_cylinder_wireframe(cylinder, label, colour))
    if floor is not None:
        centre = next(iter(cylinders.values())).centre_xy
        half = min(1.1 * reach, 1.5 * float(np.ptp(points[:, :2], axis=0).max()))
        gx, gy = np.meshgrid([centre[0] - half, centre[0] + half], [centre[1] - half, centre[1] + half])
        gz = np.vectorize(lambda a, b: floor.z_at(np.array([a, b])))(gx, gy)
        traces.append(
            go.Mesh3d(
                x=gx.ravel(),
                y=gy.ravel(),
                z=gz.ravel(),
                i=[0, 0],
                j=[1, 2],
                k=[3, 3],
                color="grey",
                opacity=0.25,
                name="floor (RANSAC)",
                hoverinfo="skip",
            )
        )
    figure = go.Figure(traces)
    figure.update_layout(**axis_layout(title))
    return figure


def sweep_cameras_figure(
    result: SweepResult, title: str | None = None, every_nth_point: int = 1
) -> go.Figure:
    """The largest COLMAP model of one sweep config: sparse points and every registered camera.

    Args:
        result: one entry of `ScanSweep.load`.
        title: figure title; defaults to the config name and how many keyframes registered.
        every_nth_point: keep one sparse point in N, to keep a stored notebook small.
    """
    sfm = result.load_sfm()
    summary = result.sfm_summary
    keep = slice(None, None, every_nth_point)
    return sfm_figure(
        sfm.points_world[keep],
        sfm.colors[keep],
        sfm.t_world_cam_list,
        title=title or f"{result.config.name}: {summary.num_registered}/{summary.num_images} registered",
    )


def sweep_figure(results: list[SweepResult], title: str = "Scan parameter sweep") -> go.Figure:
    """One column per config: registration, how much of the orbit it covers, and splat PSNR.

    PSNR is only measured on views that registered, so a config that reconstructs a short arc
    can score well; the coverage row is what keeps that honest.

    Args:
        results: from `ScanSweep.load`; held-out PSNR may be None for a config.
        title: figure title.

    Returns:
        A three-row `go.Figure`: registered share, orbit covered, train and held-out PSNR.
    """
    names = [r.config.name for r in results]
    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=(
            "Keyframes registered in the largest COLMAP model",
            "Orbit the registered cameras cover (approximate for fragments)",
            "Splat PSNR (only on views that registered)",
        ),
    )
    details = [
        f"matching: {r.config.sfm.matching}<br>SIFT threshold: {r.config.sfm.peak_threshold or 'default'}"
        f"<br>frames: {r.config.num_frames} (every {r.keyframe_spacing_s:.2f} s)"
        f"<br>iterations: {r.config.iterations}<br>COLMAP models: {r.sfm_summary.num_models}"
        for r in results
    ]
    figure.add_trace(
        go.Bar(
            x=names,
            y=[100 * r.sfm_summary.registered_fraction for r in results],
            text=[f"{r.sfm_summary.num_registered}/{r.sfm_summary.num_images}" for r in results],
            customdata=details,
            hovertemplate="%{x}<br>%{y:.0f} % registered<br>%{customdata}<extra></extra>",
            marker_color="#4c78a8",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            x=names,
            y=[r.orbit_coverage_deg for r in results],
            text=[f"{r.orbit_coverage_deg:.0f}°" for r in results],
            hovertemplate="%{x}<br>%{y:.0f}° of 360° covered<extra></extra>",
            marker_color="#54a24b",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            x=names,
            y=[r.train_psnr_db for r in results],
            name="training views",
            marker_color="#b0b0b0",
            hovertemplate="%{x}<br>%{y:.1f} dB on views it trained on<extra></extra>",
        ),
        row=3,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            x=names,
            y=[r.heldout_psnr_db for r in results],
            name="held-out views",
            marker_color="#f58518",
            hovertemplate="%{x}<br>%{y:.1f} dB on views it never saw<extra></extra>",
        ),
        row=3,
        col=1,
    )
    figure.update_yaxes(title_text="% of keyframes", range=[0, 105], row=1, col=1)
    figure.update_yaxes(title_text="degrees", range=[0, 380], row=2, col=1)
    figure.update_yaxes(title_text="PSNR [dB]", row=3, col=1)
    figure.update_layout(title=title, height=760, barmode="group", legend={"orientation": "h", "y": -0.1})
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


def state_action_figure(
    timestamps_s: np.ndarray,
    state: np.ndarray,
    action: np.ndarray,
    state_names: list[str],
    action_names: list[str],
    pairs: list[tuple[str, str]],
    replayed_state: dict[str, np.ndarray] | None = None,
    hysteresis_m: tuple[float, float] | None = None,
    title: str = "observation.state vs action, per channel",
) -> go.Figure:
    """One row per (state, action) channel pair: the state curve with its action overlaid.

    Left axis is the state in mm; right axis is the action (mm per step for a displacement,
    a 0/1 step line for a flag). A displacement action is the *derivative* of its state, so
    the two curves only look alike once the action is integrated: `replayed_state` draws that.

    Args:
        timestamps_s: shape (N,), seconds.
        state: shape (N, S), lengths in metres.
        action: shape (N, A), displacements in metres per step, or 0/1 flags.
        state_names: names of the `state` columns.
        action_names: names of the `action` columns.
        pairs: (state name, action name) per row.
        replayed_state: optional {state name: shape (N,) metres} drawn dashed over that state.
        hysteresis_m: optional (close, open) band drawn on rows whose action is a flag.
        title: figure title.

    Returns:
        A `go.Figure` with shared time axis; hover shows time, sample index and value.
    """
    to_mm = 1000.0
    figure = make_subplots(
        rows=len(pairs),
        cols=1,
        shared_xaxes=True,
        specs=[[{"secondary_y": True}]] * len(pairs),
        vertical_spacing=0.05,
    )
    sample_idx = np.arange(len(timestamps_s))
    hover = "t=%{x:.2f} s<br>sample %{customdata}<br>%{y:.2f}<extra>%{fullData.name}</extra>"
    for row, (state_name, action_name) in enumerate(pairs, start=1):
        first = row == 1
        state_mm = state[:, state_names.index(state_name)] * to_mm
        action_col = action[:, action_names.index(action_name)]
        is_flag = bool(np.isin(action_col, (0.0, 1.0)).all())
        figure.add_trace(
            go.Scatter(
                x=timestamps_s,
                y=state_mm,
                mode="lines+markers",
                name="observation.state",
                legendgroup="state",
                showlegend=first,
                line={"color": "royalblue"},
                customdata=sample_idx,
                hovertemplate=hover,
            ),
            row=row,
            col=1,
        )
        if replayed_state and state_name in replayed_state:
            figure.add_trace(
                go.Scatter(
                    x=timestamps_s,
                    y=replayed_state[state_name] * to_mm,
                    mode="lines",
                    name="state[0] + cumsum(action)",
                    legendgroup="replay",
                    showlegend=first,
                    line={"color": "black", "dash": "dash"},
                    customdata=sample_idx,
                    hovertemplate=hover,
                ),
                row=row,
                col=1,
            )
        if is_flag:
            figure.add_trace(
                go.Scatter(
                    x=timestamps_s,
                    y=action_col,
                    mode="lines",
                    name="action (flag)",
                    legendgroup="flag",
                    line={"color": "crimson", "shape": "hv"},
                    customdata=sample_idx,
                    hovertemplate=hover,
                ),
                row=row,
                col=1,
                secondary_y=True,
            )
            if hysteresis_m is not None:
                for edge_m in hysteresis_m:
                    figure.add_hline(
                        y=edge_m * to_mm,
                        line={"color": "gray", "dash": "dot", "width": 1},
                        row=row,
                        col=1,
                    )
            figure.update_yaxes(title_text="mm", row=row, col=1, secondary_y=False)
            figure.update_yaxes(
                title_text="flag", range=[-0.1, 1.1], row=row, col=1, secondary_y=True, showgrid=False
            )
        else:
            figure.add_trace(
                go.Bar(
                    x=timestamps_s,
                    y=action_col * to_mm,
                    name="action (mm / step)",
                    legendgroup="delta",
                    showlegend=first,
                    marker={"color": "darkorange", "opacity": 0.45},
                    customdata=sample_idx,
                    hovertemplate=hover,
                ),
                row=row,
                col=1,
                secondary_y=True,
            )
            figure.update_yaxes(title_text="mm", row=row, col=1, secondary_y=False)
            figure.update_yaxes(title_text="mm / step", row=row, col=1, secondary_y=True, showgrid=False)
        figure.add_annotation(
            text=f"<b>{state_name}</b> vs <b>{action_name}</b>",
            xref="x domain",
            yref="y domain",
            x=0.0,
            y=1.0,
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            row=row,
            col=1,
        )
    figure.update_xaxes(title_text="time [s]", row=len(pairs), col=1)
    figure.update_layout(
        title=title,
        height=220 * len(pairs) + 110,
        margin={"l": 60, "r": 60, "t": 60, "b": 40},
        hovermode="x unified",
        bargap=0.3,
        legend={"orientation": "h", "y": 1.0, "yanchor": "bottom"},
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


def trajectories_figure(paths: dict[str, np.ndarray], title: str = "Trajectories") -> go.Figure:
    """Several 3D paths in one interactive figure, e.g. the same clip seen from different cameras.

    Args:
        paths: name -> (N, 3) positions in one common frame.
        title: figure title.
    """
    figure = go.Figure(
        [
            go.Scatter3d(
                x=xyz[:, 0],
                y=xyz[:, 1],
                z=xyz[:, 2],
                mode="lines+markers",
                name=name,
                marker={"size": 2.5},
                line={"width": 4},
            )
            for name, xyz in paths.items()
        ]
    )
    figure.update_layout(**axis_layout(title))
    return figure


def teleop_tracking_figure(trace: TeleopTrace, title: str = "Teleoperation tracking") -> go.Figure:
    """Commanded target vs achieved end effector per axis (relative to the start), error, and gripper.

    Args:
        trace: a finished run.
        title: figure title.

    Returns:
        A `go.Figure` with three position rows and one error row; the dashed vertical line
        marks the cube spawn, the shaded bands the commanded-closed periods.
    """
    t = np.asarray(trace.t_s)
    target, eef = np.asarray(trace.p_world_target), np.asarray(trace.p_world_eef)
    start = eef[0]
    figure = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("x (m)", "y (m)", "z (m)", "tracking error (cm)"),
    )
    for row in (1, 2, 3):
        figure.add_scatter(
            x=t,
            y=target[:, row - 1] - start[row - 1],
            name="target (hand)",
            row=row,
            col=1,
            line={"color": "crimson"},
            legendgroup="t",
            showlegend=row == 1,
        )
        figure.add_scatter(
            x=t,
            y=eef[:, row - 1] - start[row - 1],
            name="robot end effector",
            row=row,
            col=1,
            line={"color": "royalblue"},
            legendgroup="e",
            showlegend=row == 1,
        )
    figure.add_scatter(
        x=t,
        y=100 * trace.tracking_error_m(),
        name="error",
        row=4,
        col=1,
        line={"color": "gray"},
        showlegend=False,
    )
    closed = np.asarray(trace.gripper_closed, dtype=bool)
    edges = np.flatnonzero(np.diff(np.concatenate([[False], closed, [False]]).astype(int)))
    for first, last in zip(edges[::2], edges[1::2], strict=True):
        figure.add_vrect(
            x0=t[first], x1=t[min(last, len(t) - 1)], fillcolor="orange", opacity=0.15, line_width=0
        )
    if trace.cube_spawn_step is not None:
        figure.add_vline(x=t[trace.cube_spawn_step], line_dash="dash", annotation_text="cube spawns")
    figure.update_layout(title=title, height=720, xaxis4_title="sim time (s)")
    return figure


def capture_diagnosis_figure(diagnosis: CaptureDiagnosis, title: str | None = None) -> go.Figure:
    """Timeline of a test clip: what was detected in each frame, and the pinch against its thresholds.

    Top: one lane per thing the extraction needs (board, bowl pad, plate marker, hand); a gap in a lane is
    a frame where it was not found. The first second, where the markers must be visible, is shaded, and the
    longest hand dropout is marked. Middle: pinch aperture with the close / open thresholds; a grasp needs
    it to dip under the lower line, a release to rise over the upper one. Bottom: PnP reprojection error of
    the hand fit (large means the hand pose, and so the depth, is unreliable).

    Args:
        diagnosis: output of `hurobo.recording.diagnose_capture`.
        title: figure title; defaults to the clip name.
    """
    t = diagnosis.times_s
    spec = diagnosis.spec
    lanes = {
        "hand": diagnosis.hand_found,
        f"plate marker (id {spec.plate_marker_id})": diagnosis.marker_found(spec.plate_marker_id),
        f"bowl pad (id {spec.bowl_marker_id})": diagnosis.marker_found(spec.bowl_marker_id),
        "board (enough corners)": diagnosis.board_found,
    }
    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.4, 0.35, 0.25],
        vertical_spacing=0.06,
        subplot_titles=("found in this frame", "pinch aperture (cm)", "hand-fit reprojection error (px)"),
    )
    for level, (name, found) in enumerate(lanes.items()):
        figure.add_trace(
            go.Scatter(
                x=t[found],
                y=np.full(found.sum(), level),
                mode="markers",
                marker={"symbol": "square", "size": 7, "color": "seagreen"},
                name=name,
                showlegend=False,
                hovertemplate="%{x:.2f} s<extra>" + name + "</extra>",
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=t[~found],
                y=np.full((~found).sum(), level),
                mode="markers",
                marker={"symbol": "x", "size": 6, "color": "crimson"},
                name=f"{name}: missing",
                showlegend=False,
                hovertemplate="%{x:.2f} s missing<extra>" + name + "</extra>",
            ),
            row=1,
            col=1,
        )
    figure.update_yaxes(tickvals=list(range(len(lanes))), ticktext=list(lanes), row=1, col=1)
    figure.add_vrect(
        x0=t[0],
        x1=t[0] + 1.0,
        fillcolor="royalblue",
        opacity=0.1,
        line_width=0,
        annotation_text="markers must be seen",
        annotation_position="top left",
        row=1,
        col=1,
    )
    start, stop = diagnosis.hand_gap()
    if stop > start:
        for row in (1, 2, 3):
            figure.add_vrect(
                x0=t[start - 1],
                x1=t[min(stop, len(t) - 1)],
                fillcolor="crimson",
                opacity=0.12,
                line_width=0,
                row=row,
                col=1,
                annotation_text="longest hand dropout" if row == 1 else None,
            )
    figure.add_trace(
        go.Scatter(x=t, y=diagnosis.aperture_m * 100, mode="lines+markers", showlegend=False),
        row=2,
        col=1,
    )
    for threshold, label in (
        (diagnosis.close_threshold_m, "grasp below"),
        (diagnosis.open_threshold_m, "release above"),
    ):
        figure.add_hline(y=threshold * 100, line_dash="dash", annotation_text=label, row=2, col=1)
    figure.add_trace(
        go.Scatter(x=t, y=diagnosis.reprojection_px, mode="lines+markers", showlegend=False), row=3, col=1
    )
    figure.update_xaxes(title_text="time (s)", row=3, col=1)
    figure.update_layout(
        title=title or f"Capture check: {diagnosis.video.name}",
        height=760,
        margin={"l": 40, "r": 10, "t": 80, "b": 40},
    )
    return figure


def clip_cut_figure(
    signals: ClipSignals, cleaner: ClipCleaner, cut: ClipCut, title: str | None = None
) -> go.Figure:
    """A clip's per-frame filter results and pinch with the kept stretch highlighted.

    Top: one lane per frame filter (a bar where the frame is rejected for that reason) and the pinch-closed
    lane. Bottom: the thumb-index distance with the close and open thresholds. Shaded: what the cut removes.

    Args:
        signals: output of `ClipAnalyzer.analyze`.
        cleaner: gives the filters and the pinch thresholds.
        cut: the decision to draw (a proposal or a stored cut).
        title: figure title; defaults to the clip name and the verdict.
    """
    from hurobo.clip_clean import FrameFilter

    t = signals.times_s
    lanes = {**FrameFilter(cleaner.config).flags(signals), "pinch closed": cleaner.closed_signal(signals)}
    figure = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.55, 0.45], vertical_spacing=0.07,
        subplot_titles=("frames rejected by each filter, and the pinch state", "pinch aperture (cm)"),
    )  # fmt: skip
    for level, (name, mask) in enumerate(lanes.items()):
        colour = "darkorange" if name == "pinch closed" else "crimson"
        figure.add_trace(
            go.Scatter(
                x=t[mask], y=np.full(mask.sum(), level), mode="markers", name=name, showlegend=False,
                marker={"symbol": "square", "size": 6, "color": colour},
                hovertemplate="%{x:.2f} s<extra>" + name + "</extra>",
            ),
            row=1, col=1,
        )  # fmt: skip
    figure.update_yaxes(tickvals=list(range(len(lanes))), ticktext=list(lanes), row=1, col=1)
    figure.add_trace(
        go.Scatter(x=t, y=signals.aperture_m * 100, mode="lines", showlegend=False), row=2, col=1
    )
    for threshold in (cleaner.segmenter.close_threshold_m, cleaner.segmenter.open_threshold_m):
        figure.add_hline(y=100 * threshold, line_dash="dash", line_color="grey", row=2, col=1)
    start, end = cut.window_s or (t[0], t[0])  # a rejected clip without a window removes everything
    for x0, x1 in ((t[0], start), (end, t[-1])):
        for row in (1, 2):
            figure.add_vrect(x0=x0, x1=x1, fillcolor="black", opacity=0.18, line_width=0, row=row, col=1)
    verdict = "kept" if cut.accepted else "rejected: " + "; ".join(cut.reasons)
    figure.update_xaxes(title_text="time (s)", row=2, col=1)
    figure.update_layout(
        title=title or f"{signals.video.name}: {verdict}",
        height=520,
        margin={"l": 40, "r": 10, "t": 80, "b": 40},
    )
    return figure


_SPLAT_VIEWER_TEMPLATE = Path(__file__).with_name("splat_viewer.html")


def write_splat_viewer(ply_path: Path, out_dir: Path, title: str = "Gaussian splat") -> Path:
    """Write a WebGL viewer page that renders the full Gaussians of a splat, orbitable with the mouse.

    The page has a panel to crop by a 3D box (x/y/z min and max sliders, drawn as a wireframe), keep or
    remove what is inside it, drop low-opacity Gaussians, and download the filtered `.ply`; it prints
    the box as `CropBox(...)` arguments for `hurobo.scan.crop_ply`.

    The page loads the `.ply` over HTTP (browsers block `file://` fetches), so serve `out_dir`,
    e.g. `python -m http.server -d <out_dir>`. Three.js and the splat viewer come from jsDelivr.

    Args:
        ply_path: standard 3DGS `.ply`, z up.
        out_dir: directory for the page; the `.ply` is symlinked next to it.
        title: page title.

    Returns:
        Path of the written `index.html`.
    """
    from hurobo.splat import splat_centre_and_radius

    gaussians = load_ply(str(ply_path))
    centre, radius = splat_centre_and_radius(gaussians.xyz, gaussians.opacity)
    position = (centre + radius * np.array([0.0, -1.2, 0.9])).round(3).tolist()
    out_dir.mkdir(parents=True, exist_ok=True)
    link = out_dir / ply_path.name
    link.unlink(missing_ok=True)
    link.symlink_to(ply_path.resolve())
    page = out_dir / "index.html"
    html = (
        _SPLAT_VIEWER_TEMPLATE.read_text()
        .replace("__TITLE__", title)
        .replace("__PLY_NAME__", ply_path.name)
        .replace("__POSITION__", str(position))
        .replace("__LOOK_AT__", str(centre.round(3).tolist()))
    )
    page.write_text(html)
    return page


def capture_plan_figure(
    setup: RealSetup,
    work_rect_table: tuple[float, float, float, float],
    clips: list[PlannedClip],
    camera: CameraPlacement | None = None,
    hand_start_xy: dict[str, np.ndarray] | None = None,
    title: str = "Recording plan (top view, table frame)",
) -> go.Figure:
    """Top view of the real table: board, work area, the planned object positions, and the phone.

    Args:
        setup: measured table, board and objects.
        work_rect_table: (x_min, x_max, y_min, y_max) of the work area in the table frame.
        clips: the planned clips; bowls are drawn small and plates large, numbered by clip.
        camera: the phone pose to draw, with its horizontal field of view.
        hand_start_xy: hand start name -> table-frame xy, drawn as diamonds around the work area.
        title: figure title.
    """
    table, board = setup.table, setup.board
    figure = go.Figure()

    def rectangle(x0: float, x1: float, y0: float, y1: float, colour: str, fill: str, name: str) -> None:
        figure.add_shape(
            type="rect", x0=x0, x1=x1, y0=y0, y1=y1, line={"color": colour, "width": 2}, fillcolor=fill
        )
        figure.add_annotation(x=(x0 + x1) / 2, y=y1, text=name, showarrow=False, yshift=10, font={"size": 11})

    rectangle(
        table.left_edge_x_m,
        table.right_edge_x_m,
        table.near_edge_y_m,
        table.far_edge_y_m,
        "#8c6d46",
        "rgba(190,160,120,0.25)",
        "table",
    )
    rectangle(0.0, board.width_m, -board.height_m, 0.0, "#222", "rgba(40,40,40,0.5)", "board")
    x0, x1, y0, y1 = work_rect_table
    rectangle(x0, x1, y0, y1, "#2a9d8f", "rgba(42,157,143,0.10)", "object centres go in here")

    def circle(centre: np.ndarray, radius: float, colour: str, label: str) -> None:
        figure.add_shape(
            type="circle",
            x0=centre[0] - radius,
            x1=centre[0] + radius,
            y0=centre[1] - radius,
            y1=centre[1] + radius,
            line={"color": colour, "width": 1},
            fillcolor=colour.replace("1)", "0.12)"),
        )
        figure.add_annotation(
            x=centre[0], y=centre[1], text=label, showarrow=False, font={"size": 10, "color": colour}
        )

    for clip in clips:
        circle(clip.plate_xy_table, setup.plate.radius_m, "rgba(220,60,60,1)", str(clip.index))
        circle(clip.bowl_xy_table, setup.bowl.radius_m, "rgba(40,90,200,1)", str(clip.index))
    if hand_start_xy:
        points = np.array(list(hand_start_xy.values()))
        figure.add_trace(
            go.Scatter(
                x=points[:, 0],
                y=points[:, 1],
                mode="markers+text",
                text=list(hand_start_xy),
                textposition="top center",
                marker={"symbol": "diamond", "size": 10, "color": "#e9a000"},
                name="hand starts",
            )
        )
    if camera is not None:
        position = camera.position_table[:2]
        heading = camera.t_table_cam[:2, 2]
        heading = heading / np.linalg.norm(heading)
        half_fov = np.radians(camera.horizontal_fov_deg / 2)
        for sign in (-1, 1):
            c, s = np.cos(sign * half_fov), np.sin(sign * half_fov)
            edge = np.array([c * heading[0] - s * heading[1], s * heading[0] + c * heading[1]])
            tip = position + 1.3 * edge
            figure.add_trace(
                go.Scatter(
                    x=[position[0], tip[0]],
                    y=[position[1], tip[1]],
                    mode="lines",
                    line={"color": "#6a4c93", "dash": "dot"},
                    showlegend=False,
                )
            )
        figure.add_trace(
            go.Scatter(
                x=[position[0]],
                y=[position[1]],
                mode="markers+text",
                text=[f"phone ({camera.height_m:.2f} m up)"],
                textposition="bottom center",
                marker={"symbol": "triangle-up", "size": 14, "color": "#6a4c93"},
                name="phone",
            )
        )
    figure.update_layout(
        title=title,
        height=620,
        xaxis={"title": "table x (m), along the board's long side", "scaleanchor": "y", "scaleratio": 1},
        yaxis={"title": "table y (m), away from you"},
        showlegend=False,
    )
    return figure


def reach_map_figure(
    xs: np.ndarray, ys: np.ndarray, errors_m: dict[str, np.ndarray], title: str = "Panda reach"
) -> go.Figure:
    """Heat maps of the end-effector position error over a grid of world (x, y), one panel per height.

    Args:
        xs: world x values, shape (X,).
        ys: world y values, shape (Y,).
        errors_m: panel title -> errors in metres, shape (Y, X), from `hurobo.reach.measure_reach_map`.
        title: figure title.
    """
    figure = make_subplots(rows=1, cols=len(errors_m), subplot_titles=list(errors_m), shared_yaxes=True)
    for column, errors in enumerate(errors_m.values(), start=1):
        figure.add_trace(
            go.Heatmap(
                x=xs,
                y=ys,
                z=100 * np.minimum(errors, 0.10),
                zmin=0,
                zmax=10,
                colorscale="RdYlGn_r",
                colorbar={"title": "error (cm)"} if column == len(errors_m) else None,
                showscale=column == len(errors_m),
                hovertemplate="x %{x:.2f} y %{y:.2f}: %{z:.1f} cm<extra></extra>",
            ),
            row=1,
            col=column,
        )
        figure.update_xaxes(title_text="world x (m), away from the robot base", row=1, col=column)
    figure.update_yaxes(title_text="world y (m)", row=1, col=1)
    figure.update_layout(title=title, height=420)
    return figure


def size_comparison_figure(
    rows: list[dict[str, float | str]], title: str = "Real vs LIBERO vs scaled"
) -> go.Figure:
    """Grouped bars of diameter and height: the measured object, LIBERO's stock one, and the scaled sim one.

    Args:
        rows: from `hurobo.scaled_native.ObjectScales.comparison`.
        title: figure title.
    """
    figure = make_subplots(rows=1, cols=2, subplot_titles=["diameter (cm)", "height (cm)"])
    series = (
        ("measurements.yaml", "real", "#2a9d8f"),
        ("LIBERO stock", "LIBERO", "#9aa0a6"),
        ("scaled in the sim", "sim", "#e76f51"),
    )
    names = [str(row["object"]) for row in rows]
    for column, quantity in enumerate(("diameter", "height"), start=1):
        for label, key, colour in series:
            values = [float(row[f"{key} {quantity}"]) for row in rows]
            figure.add_trace(
                go.Bar(
                    x=names,
                    y=values,
                    name=label,
                    marker_color=colour,
                    text=[f"{v:.1f}" for v in values],
                    textposition="outside",
                    showlegend=column == 1,
                ),
                row=1,
                col=column,
            )
    figure.update_layout(title=title, barmode="group", height=420)
    return figure


def benchmark_success_figure(summary: pd.DataFrame, title: str = "Closed-loop success rate") -> go.Figure:
    """Success rate per checkpoint, scene condition and replan interval, with 95 % Wilson intervals.

    Args:
        summary: from `hurobo.benchmark.summarize_rows` (needs success_rate, ci_low, ci_high, episodes).
        title: figure title.

    Returns:
        One group of bars per condition, one bar per (checkpoint, replan interval); hover shows n.
    """
    figure = go.Figure()
    for (checkpoint, replan), group in summary.groupby(["checkpoint", "replan_every"]):
        figure.add_trace(
            go.Bar(
                name=f"{checkpoint} (replan {replan})",
                x=group["condition"],
                y=100 * group["success_rate"],
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": 100 * (group["ci_high"] - group["success_rate"]),
                    "arrayminus": 100 * (group["success_rate"] - group["ci_low"]),
                },
                customdata=group[["successes", "episodes"]],
                hovertemplate="%{y:.0f} % (%{customdata[0]}/%{customdata[1]})<extra>%{fullData.name}</extra>",
            )
        )
    figure.update_layout(
        title=title, yaxis_title="success (%)", yaxis_range=[0, 105], barmode="group", legend_title="policy"
    )
    return figure


def failure_stage_figure(summary: pd.DataFrame, title: str = "How far episodes got") -> go.Figure:
    """Share of episodes in which the bowl was lifted and in which the task succeeded, per cell.

    Separates 'never grasped' from 'grasped but misplaced'.

    Args:
        summary: from `hurobo.benchmark.summarize_rows`.
        title: figure title.
    """
    labels = [f"{r.checkpoint} | {r.condition} | {r.replan_every}" for r in summary.itertuples()]
    figure = go.Figure()
    figure.add_trace(go.Bar(name="bowl lifted > 3 cm", x=labels, y=100 * summary["lifted"]))
    figure.add_trace(go.Bar(name="success", x=labels, y=100 * summary["success_rate"]))
    figure.update_layout(title=title, yaxis_title="episodes (%)", barmode="group", xaxis_tickangle=-40)
    return figure


def open_loop_figure(
    results: list[dict], title: str = "Open-loop action error against demonstrations"
) -> go.Figure:
    """Per-dimension action error of each checkpoint beside the do-nothing baseline, and along the chunk.

    Args:
        results: dicts from `hurobo.benchmark.OpenLoopResult` (as asdict).
        title: figure title.
    """
    figure = make_subplots(rows=1, cols=2, subplot_titles=("MAE per action dimension", "MAE along the chunk"))
    dims = ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]
    for result in results:
        figure.add_trace(go.Bar(name=result["checkpoint"], x=dims, y=result["mae_per_dim"]), row=1, col=1)
        figure.add_trace(
            go.Scatter(name=result["checkpoint"], y=result["mae_by_horizon"], showlegend=False), row=1, col=2
        )
    figure.add_trace(
        go.Bar(name="zero action", x=dims, y=results[0]["zero_mae_per_dim"], marker_color="lightgrey"),
        row=1,
        col=1,
    )
    figure.update_xaxes(title_text="step in the chunk", row=1, col=2)
    figure.update_layout(title=title, barmode="group")
    return figure
