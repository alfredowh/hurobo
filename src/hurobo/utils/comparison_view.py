"""Plotly figures for the comparison of SmolVLA policies."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def success_figure(summary: pd.DataFrame, title: str = "Closed-loop success, same layouts") -> go.Figure:
    """Success rate of each policy with its 95 % Wilson interval.

    Args:
        summary: `summarize_rows` table (checkpoint, episodes, successes, success_rate, ci_low, ci_high).
        title: figure title.

    Returns:
        A bar chart; the labels say successes / episodes.
    """
    rate = summary["success_rate"]
    figure = go.Figure(
        go.Bar(
            x=summary["checkpoint"],
            y=rate,
            error_y={
                "type": "data",
                "symmetric": False,
                "array": summary["ci_high"] - rate,
                "arrayminus": rate - summary["ci_low"],
            },
            text=[f"{s}/{n}" for s, n in zip(summary["successes"], summary["episodes"], strict=True)],
            textposition="outside",
        )
    )
    figure.update_layout(title=title, yaxis={"title": "success rate", "range": [0, 1.05]}, height=420)
    return figure


def progress_figure(
    tables: dict[str, pd.DataFrame], title: str = "Success against training step"
) -> go.Figure:
    """Closed-loop success rate per checkpoint step, one line per training run, with the Wilson interval.

    Args:
        tables: run name -> `checkpoint_table` (columns step, success_rate, ci_low, ci_high, episodes).
        title: figure title.
    """
    figure = go.Figure()
    for name, table in tables.items():
        if table.empty or "success_rate" not in table:
            continue
        rate = table["success_rate"]
        figure.add_trace(
            go.Scatter(
                x=table["step"],
                y=rate,
                mode="lines+markers",
                name=name,
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": table["ci_high"] - rate,
                    "arrayminus": rate - table["ci_low"],
                },
            )
        )
    figure.update_layout(
        title=title, xaxis={"title": "training step"}, yaxis={"title": "success rate", "range": [0, 1.05]}
    )
    return figure
