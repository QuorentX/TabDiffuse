"""Correlation heatmaps."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from src.visualization.dashboard import apply_layout


def heatmap(matrix: list[list[float]] | np.ndarray, labels: list[str], title: str) -> go.Figure:
    values = np.asarray(matrix, dtype=float)
    figure = go.Figure(
        data=go.Heatmap(
            z=values,
            x=labels,
            y=labels,
            colorscale="Tealrose",
            zmid=0,
            colorbar={"title": "corr"},
        )
    )
    return apply_layout(figure, title)
