"""Interactive plotly views of the MimicGen-style expansion: source choice, success per source, steps."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from hurobo.utils.viewer import trajectories_figure

if TYPE_CHECKING:
    from hurobo.mimicgen import ExpansionReport
    from hurobo.scaled_native import SimLayout

STEP_LIMIT = 300


def source_selection_figure(
    source_layouts_world: list[SimLayout],
    names: list[str],
    target: SimLayout,
    approach_distances_m: np.ndarray,
    carry_distances_m: np.ndarray,
    nn_k: int,
) -> go.Figure:
    """Where the sources lie relative to a new layout, as the selector sees them.

    Left: bowl positions (world x, y), what the approach is matched on. Right: bowl-to-plate vectors, what
    the carry is matched on. The `nn_k` nearest sources in each panel are drawn filled.

    Args:
        source_layouts_world: each source demo's own layout.
        names: a name per source.
        target: the new layout.
        approach_distances_m: per-source distance in the left panel.
        carry_distances_m: per-source distance in the right panel.
        nn_k: pool size of the nearest-neighbour choice.
    """
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("approach: bowl position (world xy, m)", "carry: bowl to plate vector (m)"),
    )
    bowls = np.array([layout.bowl_xy for layout in source_layouts_world])
    vectors = np.array([layout.plate_xy - layout.bowl_xy for layout in source_layouts_world])
    target_vector = target.plate_xy - target.bowl_xy
    for col, points, goal, distances in (
        (1, bowls, target.bowl_xy, approach_distances_m),
        (2, vectors, target_vector, carry_distances_m),
    ):
        pool = set(np.argsort(distances)[:nn_k].tolist())
        figure.add_trace(
            go.Scatter(
                x=points[:, 1],
                y=points[:, 0],
                mode="markers+text",
                text=[f"{n[-6:]} ({d * 100:.0f} cm)" for n, d in zip(names, distances, strict=True)],
                textposition="top center",
                marker={
                    "size": 11,
                    "color": ["#1f77b4" if i in pool else "rgba(0,0,0,0)" for i in range(len(points))],
                    "line": {"width": 2, "color": "#1f77b4"},
                },
                name="sources",
                showlegend=col == 1,
            ),
            row=1,
            col=col,
        )
        figure.add_trace(
            go.Scatter(
                x=[goal[1]],
                y=[goal[0]],
                mode="markers",
                marker={"size": 15, "symbol": "star", "color": "#d62728"},
                name="new layout",
                showlegend=col == 1,
            ),
            row=1,
            col=col,
        )
        figure.update_xaxes(title_text="world y", row=1, col=col)
        figure.update_yaxes(title_text="world x", scaleanchor=f"x{col}" if col > 1 else "x", row=1, col=col)
    figure.update_layout(height=460, title="Source choice for one new layout (filled = in the nearest pool)")
    return figure


def comparison_figure(reports: dict[str, ExpansionReport]) -> go.Figure:
    """Success rate and episode length of several runs side by side.

    Args:
        reports: run name -> report.
    """
    figure = make_subplots(
        rows=1, cols=2, subplot_titles=("share of rollouts kept", "steps of kept episodes")
    )
    names = list(reports)
    figure.add_trace(
        go.Bar(
            x=names,
            y=[reports[n].success_rate for n in names],
            text=[f"{reports[n].succeeded}/{len(reports[n].attempts)}" for n in names],
            textposition="outside",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    for name in names:
        figure.add_trace(go.Box(y=reports[name].step_counts(), name=name, boxpoints="all"), row=1, col=2)
    figure.add_hline(y=STEP_LIMIT, line_dash="dash", annotation_text="evaluation limit", row=1, col=2)
    figure.update_yaxes(range=[0, 1.1], row=1, col=1)
    figure.update_layout(height=420, title="Stock (04-style) against MimicGen-style, same layouts")
    return figure


def source_usage_figure(report: ExpansionReport) -> go.Figure:
    """Attempts and successes per source demo, for the approach and for the carry.

    Args:
        report: a finished run.
    """
    figure = make_subplots(rows=1, cols=2, subplot_titles=("as approach source", "as carry source"))
    for col, role in enumerate(("approach", "carry"), start=1):
        usage = report.success_by_source(role)
        names = [name[-6:] for name in usage]
        figure.add_trace(
            go.Bar(
                x=names,
                y=[n for n, _ in usage.values()],
                name="attempts",
                marker_color="#9ecae1",
                showlegend=col == 1,
            ),
            row=1,
            col=col,
        )
        figure.add_trace(
            go.Bar(
                x=names,
                y=[k for _, k in usage.values()],
                name="kept",
                marker_color="#1f77b4",
                showlegend=col == 1,
            ),
            row=1,
            col=col,
        )
    figure.update_layout(
        barmode="overlay", height=380, title="Which source demos were used, and how often it worked"
    )
    return figure


def kept_paths_figure(
    report: ExpansionReport, limit: int = 8, title: str = "Kept end-effector paths"
) -> go.Figure:
    """The end-effector paths (world frame) of the first `limit` kept rollouts, in one 3D view.

    Args:
        report: a run made in this process (paths are not stored in saved reports).
        limit: how many paths to draw.
        title: figure title.
    """
    paths = {
        f"{a.approach_source}/{a.carry_source} ({a.steps} steps)": a.eef_path_world
        for a in report.attempts
        if a.success and len(a.eef_path_world)
    }
    return trajectories_figure(dict(list(paths.items())[:limit]), title)
