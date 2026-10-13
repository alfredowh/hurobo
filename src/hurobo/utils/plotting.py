"""Trajectory and results figures for the README and the extraction sanity checks."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from hurobo.capture_plan import CapturePlanner, PlannedClip
    from hurobo.real_setup import RealSetup


def plot_3d_trajectory(
    positions_table: np.ndarray,
    output_path: Path,
    is_grasping: np.ndarray | None = None,
    title: str = "Hand TCP trajectory (table frame)",
) -> None:
    """Save a 3D scatter/line plot of a TCP trajectory, colored by grasp state.

    Used right after extraction (PLAN.md §4, Thursday capture block) as the
    fast visual check that a demo looks like a clean pick-place arc before
    spending time on retargeting.

    Args:
        positions_table: shape (N, 3), TCP positions in the table frame.
        output_path: where to save the figure (.png).
        is_grasping: shape (N,) bool, if given colors grasping samples
            differently from free-moving ones.
        title: plot title.
    """
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    if is_grasping is None:
        ax.plot(positions_table[:, 0], positions_table[:, 1], positions_table[:, 2], "-o", markersize=2)
    else:
        ax.plot(
            positions_table[:, 0],
            positions_table[:, 1],
            positions_table[:, 2],
            "-",
            color="gray",
            linewidth=0.5,
        )
        free = ~is_grasping
        ax.scatter(*positions_table[free].T, s=4, color="tab:blue", label="free")
        ax.scatter(*positions_table[is_grasping].T, s=4, color="tab:red", label="grasping")
        ax.legend()
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_human_vs_robot_trajectory(
    human_positions: np.ndarray, robot_positions: np.ndarray, output_path: Path
) -> None:
    """Save a 3D overlay of the human TCP trajectory and the retargeted robot EEF trajectory.

    One of the results figures called for in PLAN.md §6.

    Args:
        human_positions: shape (N, 3), human TCP positions (table frame).
        robot_positions: shape (M, 3), robot EEF positions (same frame as the
            human trajectory after retargeting, i.e. already object-relative
            then re-expressed into whichever object frame was used for the plot).
        output_path: where to save the figure (.png).
    """
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(*human_positions.T, "-", color="tab:blue", label="human TCP")
    ax.plot(*robot_positions.T, "-", color="tab:orange", label="robot EEF")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.legend()
    ax.set_title("Human demonstration vs. retargeted robot rollout")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_data_multiplier_curve(
    num_seed_demos: list[int], num_generated_successes: list[int], output_path: Path
) -> None:
    """Save the generation-success-vs-seed-demos curve called for in PLAN.md §5/§6.

    Args:
        num_seed_demos: x values, e.g. [1, 3, 5, 10, 15].
        num_generated_successes: y values, number of verified successful sim
            rollouts generated from that many seed human demos.
        output_path: where to save the figure (.png).
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(num_seed_demos, num_generated_successes, "-o")
    ax.set_xlabel("number of seed human demos")
    ax.set_ylabel("number of verified sim successes")
    ax.set_title("Data multiplier: human demos -> generated sim demos")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_aperture_trace(
    timestamps_s: np.ndarray,
    pinch_aperture_m: np.ndarray,
    output_path: Path,
    close_threshold_m: float,
    open_threshold_m: float,
    grasp_frame_idx: int | None = None,
    release_frame_idx: int | None = None,
    title: str = "Pinch aperture",
) -> None:
    """Save the pinch-aperture time series with the hysteresis band and grasp/release marks.

    This is the plot to tune `GraspSegmenter` thresholds against real clips.

    Args:
        timestamps_s: shape (N,), seconds from the start of the episode.
        pinch_aperture_m: shape (N,), thumb-index distance in metres.
        output_path: where to save the figure (.png).
        close_threshold_m: aperture below this registers a grasp.
        open_threshold_m: aperture above this registers a release.
        grasp_frame_idx: detected grasp-onset sample index, if any.
        release_frame_idx: detected release-onset sample index, if any.
        title: plot title.
    """
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(timestamps_s, pinch_aperture_m * 100, "-", color="tab:blue")
    ax.axhspan(close_threshold_m * 100, open_threshold_m * 100, color="gray", alpha=0.2, label="hysteresis")
    for idx, label, color in (
        (grasp_frame_idx, "grasp", "tab:red"),
        (release_frame_idx, "release", "tab:green"),
    ):
        if idx is not None:
            ax.axvline(timestamps_s[idx], color=color, linestyle="--", label=label)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("aperture (cm)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_image_row(
    panels: dict[str, np.ndarray],
    title: str | None = None,
    height_in: float = 4.4,
    cmap: str | None = None,
) -> Figure:
    """Titled images side by side (one panel works too).

    Args:
        panels: panel title -> image (RGB, or 2D with `cmap`).
        title: figure title.
        height_in: figure height in inches; width follows the panels' aspect.
        cmap: matplotlib colormap for 2D images, e.g. "gray".
    """
    aspects = [image.shape[1] / image.shape[0] for image in panels.values()]
    fig, axes = plt.subplots(1, len(panels), figsize=(height_in * sum(aspects), height_in), squeeze=False)
    for ax, (name, image) in zip(axes[0], panels.items(), strict=True):
        ax.imshow(image, cmap=cmap)
        ax.set_title(name)
        ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_image_grid(
    rows: dict[str, list[np.ndarray]],
    column_titles: list[str] | None = None,
    cell_in: float = 2.0,
) -> Figure:
    """Rows of equally many images, each row labelled; e.g. one render per method across several views.

    Args:
        rows: row label -> list of RGB images (same length for every row).
        column_titles: title above each column, e.g. the azimuth of the view.
        cell_in: height of one image cell in inches; the width follows the first image's aspect.

    Returns:
        The figure (not saved or shown).
    """
    first = next(iter(rows.values()))[0]
    aspect = first.shape[1] / first.shape[0]
    columns = len(next(iter(rows.values())))
    fig, axes = plt.subplots(
        len(rows), columns, figsize=(cell_in * aspect * columns + 1.2, cell_in * len(rows)), squeeze=False
    )
    for r, (label, images) in enumerate(rows.items()):
        for c, image in enumerate(images):
            ax = axes[r][c]
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(label, fontsize=8, rotation=90, labelpad=6)
            if r == 0 and column_titles:
                ax.set_title(column_titles[c], fontsize=9)
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_episode_overview(
    image_rgb: np.ndarray,
    state: np.ndarray,
    action: np.ndarray,
    state_names: list[str],
    action_names: list[str],
    image_title: str = "observation image",
) -> Figure:
    """One image next to the state and action time series of an episode.

    Args:
        image_rgb: (H, W, 3) image to show.
        state: shape (T, S).
        action: shape (T, A).
        state_names: legend labels for the S state components.
        action_names: legend labels for the A action components.
        image_title: title of the image panel.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 2.8))
    axes[0].imshow(image_rgb)
    axes[0].set_title(image_title)
    axes[0].axis("off")
    for ax, values, names, title in (
        (axes[1], state, state_names, "observation.state"),
        (axes[2], action, action_names, "action (human proxy)"),
    ):
        for k, name in enumerate(names):
            ax.plot(values[:, k], label=name)
        ax.set_title(title)
        ax.legend(fontsize=7)
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_push_traces(traces: dict[str, dict[str, np.ndarray]], output_path: Path) -> None:
    """Save bowl x-y paths, tilt and height over time for several collider variants.

    Args:
        traces: {label: {"pos": (T, 3) bowl position in metres, "tilt_deg": (T,)}}.
        output_path: where to write the PNG.
    """
    fig, (ax_xy, ax_tilt, ax_z) = plt.subplots(1, 3, figsize=(15, 4.5))
    for name, trace in traces.items():
        pos, tilt = trace["pos"], trace["tilt_deg"]
        ax_xy.plot(pos[:, 1], pos[:, 0], label=name)
        ax_xy.plot(pos[0, 1], pos[0, 0], "o", color="k", markersize=4)
        ax_tilt.plot(tilt, label=name)
        ax_z.plot(1000 * (pos[:, 2] - pos[0, 2]), label=name)
    ax_xy.set(xlabel="world y (m)", ylabel="world x (m)", title="bowl path (dot = start)")
    ax_xy.set_aspect("equal")
    ax_tilt.set(xlabel="control step", ylabel="deg", title="bowl tilt from upright")
    ax_z.set(xlabel="control step", ylabel="mm", title="bowl height change")
    ax_xy.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


def plot_clip_sheet(
    setup: RealSetup,
    work_rect_table: tuple[float, float, float, float],
    clips: list[PlannedClip],
    planner: CapturePlanner,
    columns: int = 5,
) -> Figure:
    """One small top view of the table per planned clip, to print and keep next to the table.

    Each panel shows the table, the board, the work area, where the bowl (blue) and plate (red) go, the
    hand's start (orange diamond) and an arrow from the bowl to the side of the rim to pinch.

    Args:
        setup: measured table, board and objects.
        work_rect_table: (x_min, x_max, y_min, y_max) of the work area, table frame.
        clips: the planned clips.
        planner: used to turn a hand-start name into a table-frame position.
        columns: panels per row.

    Returns:
        The figure (not saved).
    """
    rows = int(np.ceil(len(clips) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(3.4 * columns, 2.3 * rows + 0.4), squeeze=False)
    table, board = setup.table, setup.board
    x0, x1, y0, y1 = work_rect_table
    for axis in axes.ravel():
        axis.axis("off")
    for axis, clip in zip(axes.ravel(), clips, strict=False):
        axis.axis("on")
        axis.set_aspect("equal")
        axis.add_patch(
            plt.Rectangle(
                (table.left_edge_x_m, table.near_edge_y_m),
                table.width_m,
                table.depth_m,
                fill=True,
                facecolor="#eadfcb",
                edgecolor="#8c6d46",
            )
        )
        axis.add_patch(plt.Rectangle((0, -board.height_m), board.width_m, board.height_m, color="#333"))
        axis.add_patch(
            plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#2a9d8f", linestyle="--")
        )
        axis.add_patch(plt.Circle(clip.plate_xy_table, setup.plate.radius_m, color="#dc3c3c", alpha=0.35))
        axis.add_patch(plt.Circle(clip.bowl_xy_table, setup.bowl.radius_m, color="#285ac8", alpha=0.6))
        start = planner.hand_start_table(clip.hand_start)
        axis.plot(*start[:2], "D", color="#e9a000", markersize=7)
        side = -1.0 if clip.grasp_side == "left" else 1.0
        axis.annotate(
            "",
            xy=clip.bowl_xy_table + np.array([side * setup.bowl.radius_m, 0.0]),
            xytext=clip.bowl_xy_table,
            arrowprops={"arrowstyle": "->", "color": "white", "lw": 1.5},
        )
        left, near = clip.from_table_edges_m(clip.bowl_xy_table, setup)
        axis.set_title(
            f"clip {clip.index:03d}: hand {clip.hand_start.name}\npinch {clip.grasp_side} rim; bowl "
            f"{left * 100:.0f} cm from left, {near * 100:.0f} from you",
            fontsize=7,
        )
        axis.set_xlim(table.left_edge_x_m - 0.05, table.right_edge_x_m + 0.05)
        axis.set_ylim(table.near_edge_y_m - 0.05, table.far_edge_y_m + 0.05)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle("Recording plan: blue bowl, red plate, orange hand start, board in black", fontsize=10)
    figure.tight_layout()
    return figure
