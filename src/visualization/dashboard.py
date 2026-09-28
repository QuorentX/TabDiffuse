"""Shared Plotly styling."""

from __future__ import annotations

import plotly.graph_objects as go

COLORS = ["#0e7c74", "#c9842a", "#3d5a80", "#9b4d6a", "#4d6b4a"]


def apply_layout(figure: go.Figure, title: str | None = None) -> go.Figure:
    figure.update_layout(
        template="plotly_white",
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Segoe UI, Helvetica, Arial", "color": "#1b2430", "size": 13},
        margin={"l": 48, "r": 24, "t": 48, "b": 40},
        colorway=COLORS,
        legend={"orientation": "h", "y": 1.12},
    )
    return figure
