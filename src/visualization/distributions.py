"""Distribution charts for real and synthetic columns."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.visualization.dashboard import apply_layout


def histogram_figure(real: pd.Series, synthetic: pd.Series, title: str) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Histogram(x=pd.to_numeric(real, errors="coerce"), name="Real", opacity=0.65, nbinsx=30))
    figure.add_trace(go.Histogram(x=pd.to_numeric(synthetic, errors="coerce"), name="Synthetic", opacity=0.65, nbinsx=30))
    figure.update_layout(barmode="overlay")
    return apply_layout(figure, title)


def box_figure(real: pd.Series, synthetic: pd.Series, title: str) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Box(y=pd.to_numeric(real, errors="coerce"), name="Real"))
    figure.add_trace(go.Box(y=pd.to_numeric(synthetic, errors="coerce"), name="Synthetic"))
    return apply_layout(figure, title)


def ecdf_figure(real: pd.Series, synthetic: pd.Series, title: str) -> go.Figure:
    figure = go.Figure()
    for name, series in ("Real", real), ("Synthetic", synthetic):
        values = np.sort(pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float))
        if len(values) == 0:
            continue
        y = np.arange(1, len(values) + 1) / len(values)
        figure.add_trace(go.Scatter(x=values, y=y, mode="lines", name=name))
    return apply_layout(figure, title)


def category_figure(real: pd.Series, synthetic: pd.Series, title: str) -> go.Figure:
    real_freq = real.astype("string").fillna("<NA>").value_counts(normalize=True)
    syn_freq = synthetic.astype("string").fillna("<NA>").value_counts(normalize=True)
    categories = list(dict.fromkeys(list(real_freq.index[:12]) + [item for item in syn_freq.index if item not in real_freq.index[:12]]))[:16]
    figure = go.Figure()
    figure.add_trace(go.Bar(x=categories, y=[float(real_freq.get(item, 0.0)) for item in categories], name="Real"))
    figure.add_trace(go.Bar(x=categories, y=[float(syn_freq.get(item, 0.0)) for item in categories], name="Synthetic"))
    figure.update_layout(barmode="group")
    return apply_layout(figure, title)


def loss_figure(history: dict) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Scatter(y=history.get("train_loss", []), name="Training loss", mode="lines"))
    figure.add_trace(go.Scatter(y=history.get("val_loss", []), name="Validation loss", mode="lines"))
    figure.update_layout(xaxis_title="Epoch", yaxis_title="Score matching loss")
    return apply_layout(figure, "Training history")
