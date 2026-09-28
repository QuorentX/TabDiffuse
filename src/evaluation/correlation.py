"""Correlation preservation between real and synthetic numerical columns."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.utils.metrics import to_jsonable


def _numeric_frame(real: pd.DataFrame, synthetic: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        column
        for column in real.columns
        if column in synthetic.columns
        and pd.api.types.is_numeric_dtype(real[column])
        and pd.api.types.is_numeric_dtype(synthetic[column])
        and real[column].nunique(dropna=True) > 1
    ]
    return real[columns].apply(pd.to_numeric, errors="coerce"), synthetic[columns].apply(pd.to_numeric, errors="coerce")


def correlation_report(real: pd.DataFrame, synthetic: pd.DataFrame) -> dict[str, Any]:
    real_num, syn_num = _numeric_frame(real, synthetic)
    if real_num.shape[1] < 2:
        return {"available": False, "message": "At least two varying numerical columns are required."}
    pearson_real = real_num.corr(method="pearson")
    pearson_syn = syn_num.corr(method="pearson")
    spearman_real = real_num.corr(method="spearman")
    spearman_syn = syn_num.corr(method="spearman")
    pearson_diff = pearson_syn - pearson_real
    spearman_diff = spearman_syn - spearman_real
    return to_jsonable(
        {
            "available": True,
            "columns": list(real_num.columns),
            "pearson_real": pearson_real.fillna(0).to_numpy().tolist(),
            "pearson_synthetic": pearson_syn.fillna(0).to_numpy().tolist(),
            "pearson_difference": pearson_diff.fillna(0).to_numpy().tolist(),
            "spearman_real": spearman_real.fillna(0).to_numpy().tolist(),
            "spearman_synthetic": spearman_syn.fillna(0).to_numpy().tolist(),
            "spearman_difference": spearman_diff.fillna(0).to_numpy().tolist(),
            "pearson_mean_abs_difference": _upper_mean_abs(pearson_diff),
            "spearman_mean_abs_difference": _upper_mean_abs(spearman_diff),
        }
    )


def _upper_mean_abs(diff: pd.DataFrame) -> float | None:
    values = diff.to_numpy(dtype=float)
    mask = np.triu(np.ones(values.shape, dtype=bool), k=1)
    selected = np.abs(values[mask])
    selected = selected[np.isfinite(selected)]
    if selected.size == 0:
        return None
    return float(selected.mean())
