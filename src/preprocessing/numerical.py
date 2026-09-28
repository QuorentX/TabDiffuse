"""Numeric imputation and scaling with an exact inverse path."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.preprocessing.missing_values import fit_numeric_fill
from src.preprocessing.outliers import clip_count, iqr_bounds
from src.utils.exceptions import TabDiffuseError


def _integer_like(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return False
    if pd.api.types.is_integer_dtype(series):
        return True
    values = numeric.to_numpy(dtype=float)
    return bool(np.allclose(values, np.round(values), atol=1e-6))


class NumericTransform:
    """Fit-once numeric pipeline for a single column.

    Min-max scaling is mapped onto [-1, 1] so it sits closer to the noise
    scale of the variance-preserving SDE. Standard scaling remains the
    recommended default.
    """

    def __init__(self) -> None:
        self.fitted = False

    def fit(
        self,
        series: pd.Series,
        *,
        imputer: str = "median",
        scaler: str = "standard",
        outlier: str = "none",
        iqr_factor: float = 1.5,
    ) -> "NumericTransform":
        numeric = pd.to_numeric(series, errors="coerce")
        self.name = str(series.name)
        self.integer_like = _integer_like(numeric)
        self.scaler = scaler
        self.outlier = outlier
        self.fill, self.missing_filled = fit_numeric_fill(numeric, imputer)
        self.imputer = imputer
        filled = numeric.fillna(self.fill)
        self.clipped = 0
        self.clip_low = None
        self.clip_high = None
        if outlier == "iqr_clip":
            self.clip_low, self.clip_high = iqr_bounds(filled, iqr_factor)
            self.clipped = clip_count(filled, self.clip_low, self.clip_high)
            filled = filled.clip(self.clip_low, self.clip_high)
        elif outlier != "none":
            raise TabDiffuseError("Outlier handling must be 'none' or 'iqr_clip'.")
        values = filled.to_numpy(dtype=np.float64)
        self.value_min = float(np.min(values)) if len(values) else 0.0
        self.value_max = float(np.max(values)) if len(values) else 0.0
        if self.value_max < self.value_min:
            self.value_min, self.value_max = self.value_max, self.value_min
        if scaler == "standard":
            self.center = float(values.mean())
            self.scale = float(values.std())
        elif scaler == "robust":
            self.center = float(np.median(values))
            q1, q3 = np.percentile(values, [25, 75])
            self.scale = float(q3 - q1)
        elif scaler == "minmax":
            self.center = float(values.min())
            self.scale = float(values.max() - values.min())
        elif scaler == "none":
            self.center = 0.0
            self.scale = 1.0
        else:
            raise TabDiffuseError(f"Unknown numeric scaler '{scaler}'.")
        if self.scale < 1e-8:
            self.scale = 1.0
        self.fitted = True
        return self

    def transform(self, series: pd.Series) -> np.ndarray:
        self._check()
        numeric = pd.to_numeric(series, errors="coerce").fillna(self.fill)
        if self.outlier == "iqr_clip":
            numeric = numeric.clip(self.clip_low, self.clip_high)
        values = numeric.to_numpy(dtype=np.float64)
        if self.scaler == "minmax":
            return ((values - self.center) / self.scale) * 2.0 - 1.0
        if self.scaler == "none":
            return values
        return (values - self.center) / self.scale

    def inverse(self, values: np.ndarray) -> np.ndarray:
        self._check()
        array = np.asarray(values, dtype=np.float64)
        if self.scaler == "minmax":
            restored = ((array + 1.0) / 2.0) * self.scale + self.center
        elif self.scaler == "none":
            restored = array
        else:
            restored = array * self.scale + self.center
        outside = (restored < self.value_min) | (restored > self.value_max)
        self.last_projected = int(np.sum(outside))
        restored = np.clip(restored, self.value_min, self.value_max)
        if self.integer_like:
            restored = np.clip(np.rint(restored), self.value_min, self.value_max)
        return restored

    def _check(self) -> None:
        if not self.fitted:
            raise TabDiffuseError(f"Numeric transform for '{getattr(self, 'name', '?')}' is not fitted.")
