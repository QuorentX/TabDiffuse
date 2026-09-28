"""Missing-value statistics for numeric columns."""

from __future__ import annotations

import pandas as pd

from src.utils.exceptions import TabDiffuseError


def fit_numeric_fill(series: pd.Series, strategy: str) -> tuple[float, int]:
    """Return the fill value and how many training values were missing."""
    numeric = pd.to_numeric(series, errors="coerce")
    missing = int(numeric.isna().sum())
    observed = numeric.dropna()
    if observed.empty:
        raise TabDiffuseError(
            f"Column '{series.name}' has no observed numeric values, so it cannot be imputed."
        )
    if strategy == "mean":
        fill = float(observed.mean())
    elif strategy == "median":
        fill = float(observed.median())
    elif strategy == "constant":
        fill = 0.0
    else:
        raise TabDiffuseError(
            f"Unknown numeric imputer '{strategy}'. Choose mean, median, or constant."
        )
    return fill, missing
