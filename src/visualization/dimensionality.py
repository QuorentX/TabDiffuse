"""Qualitative projections. These are not privacy or fidelity proofs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.utils.exceptions import TabDiffuseError
from src.visualization.dashboard import apply_layout


def project_coordinates(
    real_x: np.ndarray,
    synthetic_x: np.ndarray,
    method: str,
    *,
    max_rows: int = 800,
    seed: int = 42,
) -> pd.DataFrame:
    """Project a sample of real and synthetic feature rows into two dimensions."""
    real_s, syn_s = _sample(real_x, max_rows // 2, seed), _sample(synthetic_x, max_rows // 2, seed + 1)
    combined = np.vstack([real_s, syn_s])
    if method == "pca":
        points = PCA(n_components=2, random_state=seed).fit_transform(combined)
    elif method == "tsne":
        perplexity = max(5, min(30, len(combined) // 4))
        points = TSNE(n_components=2, random_state=seed, perplexity=perplexity, init="pca", learning_rate="auto").fit_transform(combined)
    elif method == "umap":
        points = _umap(combined, seed)
    else:
        raise TabDiffuseError("Projection method must be pca, tsne, or umap.")
    return pd.DataFrame(
        {
            "x": points[:, 0],
            "y": points[:, 1],
            "source": ["Real"] * len(real_s) + ["Synthetic"] * len(syn_s),
        }
    )


def projection_figure(coordinates: pd.DataFrame, title: str):
    figure = px.scatter(coordinates, x="x", y="y", color="source", color_discrete_sequence=["#0e7c74", "#c9842a"])
    return apply_layout(figure, title)


def _umap(combined: np.ndarray, seed: int) -> np.ndarray:
    try:
        import umap
    except ImportError as exc:
        raise TabDiffuseError("UMAP is not installed. Install the optional umap-learn package, or use PCA or t-SNE.") from exc
    reducer = umap.UMAP(n_components=2, random_state=seed)
    return reducer.fit_transform(combined)


def _sample(matrix: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float64)
    if len(array) <= max_rows:
        return array
    rng = np.random.default_rng(seed)
    return array[rng.choice(len(array), size=max_rows, replace=False)]
