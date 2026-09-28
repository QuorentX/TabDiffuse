"""Composite similarity scores with explicit formulas.

These scores summarize measured discrepancies. They are not proof that the
synthetic data matches the real distribution.
"""

from __future__ import annotations

from typing import Any

from src.utils.metrics import to_jsonable


def similarity_scores(numerical: dict[str, Any], correlation: dict[str, Any], categorical: dict[str, Any]) -> dict[str, Any]:
    ks = (numerical.get("summary") or {}).get("mean_ks")
    distribution = None if ks is None else float(max(0.0, 1.0 - min(float(ks), 1.0)))
    corr_gap = correlation.get("pearson_mean_abs_difference") if correlation.get("available", False) else None
    correlation_score = None if corr_gap is None else float(max(0.0, 1.0 - min(float(corr_gap), 1.0)))
    tv = (categorical.get("summary") or {}).get("mean_total_variation")
    categorical_score = None if tv is None else float(max(0.0, 1.0 - min(float(tv), 1.0)))
    return to_jsonable(
        {
            "distribution_similarity": distribution,
            "correlation_similarity": correlation_score,
            "categorical_similarity": categorical_score,
            "formulas": {
                "distribution_similarity": "1 - mean Kolmogorov-Smirnov statistic, clipped to [0, 1].",
                "correlation_similarity": "1 - mean absolute Pearson correlation difference, clipped to [0, 1].",
                "categorical_similarity": "1 - mean total variation distance, clipped to [0, 1].",
            },
        }
    )
