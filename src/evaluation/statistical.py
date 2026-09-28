"""Univariate numerical comparisons between real and synthetic columns."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from src.evaluation.distribution import histogram_jsd, population_stability_index
from src.utils.metrics import to_jsonable


def numeric_column_metrics(real: pd.Series, synthetic: pd.Series) -> dict[str, Any] | None:
    real_values = pd.to_numeric(real, errors="coerce").dropna().to_numpy(dtype=float)
    syn_values = pd.to_numeric(synthetic, errors="coerce").dropna().to_numpy(dtype=float)
    if len(real_values) < 5 or len(syn_values) < 5:
        return None
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    real_q = np.quantile(real_values, quantiles)
    syn_q = np.quantile(syn_values, quantiles)
    ks_stat = float(ks_2samp(real_values, syn_values, method="asymp").statistic)
    return {
        "mean_difference": float(np.mean(syn_values) - np.mean(real_values)),
        "median_difference": float(np.median(syn_values) - np.median(real_values)),
        "std_difference": float(np.std(syn_values) - np.std(real_values)),
        "quantile_mean_abs_difference": float(np.mean(np.abs(syn_q - real_q))),
        "quantiles": {"real": real_q.tolist(), "synthetic": syn_q.tolist(), "levels": quantiles},
        "ks_statistic": ks_stat,
        "wasserstein": float(wasserstein_distance(real_values, syn_values)),
        "jensen_shannon": histogram_jsd(real_values, syn_values),
        "psi": population_stability_index(real_values, syn_values),
    }


def numerical_report(real: pd.DataFrame, synthetic: pd.DataFrame) -> dict[str, Any]:
    columns = [
        column
        for column in real.columns
        if column in synthetic.columns and pd.api.types.is_numeric_dtype(real[column])
    ]
    per_column = {}
    for column in columns:
        metrics = numeric_column_metrics(real[column], synthetic[column])
        if metrics is not None:
            per_column[column] = metrics
    ks_values = [item["ks_statistic"] for item in per_column.values()]
    summary = {
        "columns": len(per_column),
        "mean_ks": float(np.mean(ks_values)) if ks_values else None,
        "mean_wasserstein": float(np.mean([item["wasserstein"] for item in per_column.values()])) if per_column else None,
        "mean_psi": float(np.mean([item["psi"] for item in per_column.values() if item["psi"] is not None])) if per_column else None,
    }
    return to_jsonable({"summary": summary, "columns": per_column})
