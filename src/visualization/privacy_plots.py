"""Privacy diagnostic charts."""

from __future__ import annotations

import plotly.graph_objects as go

from src.visualization.dashboard import apply_layout


def distance_histogram(distances: list[float], title: str) -> go.Figure:
    figure = go.Figure(data=[go.Histogram(x=distances, name="Distance", nbinsx=30, marker_color="#0e7c74")])
    figure.update_layout(xaxis_title="Nearest-neighbor distance", yaxis_title="Count")
    return apply_layout(figure, title)


def train_holdout_histogram(train_distances: list[float], holdout_distances: list[float]) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Histogram(x=train_distances, name="Distance to training", opacity=0.7, nbinsx=30))
    figure.add_trace(go.Histogram(x=holdout_distances, name="Distance to holdout", opacity=0.7, nbinsx=30))
    figure.update_layout(barmode="overlay", xaxis_title="Distance", yaxis_title="Count")
    return apply_layout(figure, "Training vs held-out nearest-neighbor distance")
