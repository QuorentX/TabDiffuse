"""Categorical frequency comparisons."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from src.utils.metrics import to_jsonable


def _probabilities(series: pd.Series) -> pd.Series:
    filled = series.astype("string").fillna("<NA>")
    counts = filled.value_counts(dropna=False)
    return counts / counts.sum()


def categorical_column_metrics(real: pd.Series, synthetic: pd.Series) -> dict[str, Any]:
    real_prob = _probabilities(real)
    syn_prob = _probabilities(synthetic)
    categories = sorted(set(real_prob.index).union(syn_prob.index))
    real_vec = np.array([float(real_prob.get(category, 0.0)) for category in categories]) + 1e-12
    syn_vec = np.array([float(syn_prob.get(category, 0.0)) for category in categories]) + 1e-12
    real_vec /= real_vec.sum()
    syn_vec /= syn_vec.sum()
    total_variation = float(0.5 * np.abs(real_vec - syn_vec).sum())
    return {
        "total_variation": total_variation,
        "jensen_shannon": float(jensenshannon(real_vec, syn_vec, base=2) ** 2),
        "missing_rate_real": float(real.isna().mean()),
        "missing_rate_synthetic": float(synthetic.isna().mean()),
        "missing_rate_difference": float(synthetic.isna().mean() - real.isna().mean()),
        "max_abs_frequency_difference": float(np.max(np.abs(real_vec - syn_vec))),
    }


def categorical_report(real: pd.DataFrame, synthetic: pd.DataFrame) -> dict[str, Any]:
    columns = []
    for column in real.columns:
        if column not in synthetic.columns:
            continue
        unique = int(real[column].nunique(dropna=True))
        if unique > 40:
            continue
        if pd.api.types.is_numeric_dtype(real[column]) and unique > 12:
            continue
        columns.append(column)
    per_column = {column: categorical_column_metrics(real[column], synthetic[column]) for column in columns}
    tv_values = [item["total_variation"] for item in per_column.values()]
    summary = {
        "columns": len(per_column),
        "mean_total_variation": float(np.mean(tv_values)) if tv_values else None,
    }
    return to_jsonable({"summary": summary, "columns": per_column})
