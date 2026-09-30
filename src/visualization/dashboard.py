"""Shared Plotly styling."""

from __future__ import annotations

import plotly.graph_objects as go

COLORS = ["#0e7c74", "#c9842a", "#3d5a80", "#9b4d6a", "#4d6b4a"]


def apply_layout(figure: go.Figure, title: str | None = None, *, height: int = 460) -> go.Figure:
    """Shared layout: centered title, centered legend, and room around the plot."""
    figure.update_layout(
        template="plotly_white",
        title=(
            {"text": title, "x": 0.5, "xanchor": "center", "y": 0.98, "yanchor": "top"}
            if title
            else None
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Segoe UI, Helvetica, Arial", "color": "#1b2430", "size": 13},
        margin={"l": 64, "r": 40, "t": 72, "b": 104},
        height=height,
        colorway=COLORS,
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": -0.22,
            "yanchor": "top",
            "bgcolor": "rgba(0,0,0,0)",
        },
    )
    figure.update_xaxes(automargin=True)
    figure.update_yaxes(automargin=True)
    return figure
