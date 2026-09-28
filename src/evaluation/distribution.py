"""Histogram divergence and population stability."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import jensenshannon


def histogram_jsd(real: np.ndarray, synthetic: np.ndarray, bins: int = 20) -> float | None:
    """Jensen-Shannon divergence between shared-bin histograms."""
    if len(real) < 5 or len(synthetic) < 5:
        return None
    low = float(np.min(real))
    high = float(np.max(real))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return None
    edges = np.linspace(low, high, bins + 1)
    real_hist, _ = np.histogram(real, bins=edges)
    syn_hist, _ = np.histogram(synthetic, bins=edges)
    real_prob = real_hist.astype(float) + 1e-8
    syn_prob = syn_hist.astype(float) + 1e-8
    real_prob /= real_prob.sum()
    syn_prob /= syn_prob.sum()
    return float(jensenshannon(real_prob, syn_prob, base=2) ** 2)


def population_stability_index(real: np.ndarray, synthetic: np.ndarray, bins: int = 10) -> float | None:
    """PSI using quantile bins defined on the real column."""
    if len(real) < 20 or len(synthetic) < 5:
        return None
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(real, quantiles))
    if len(edges) < 3:
        return None
    real_hist, _ = np.histogram(real, bins=edges)
    syn_hist, _ = np.histogram(synthetic, bins=edges)
    real_ratio = np.clip(real_hist / max(real_hist.sum(), 1), 1e-6, None)
    syn_ratio = np.clip(syn_hist / max(syn_hist.sum(), 1), 1e-6, None)
    return float(np.sum((syn_ratio - real_ratio) * np.log(syn_ratio / real_ratio)))
