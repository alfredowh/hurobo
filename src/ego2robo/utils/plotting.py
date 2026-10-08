"""Trajectory and results figures for the README and the extraction sanity checks."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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
