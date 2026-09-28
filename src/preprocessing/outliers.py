"""IQR fences used when the user explicitly chooses outlier clipping."""

from __future__ import annotations

import numpy as np
import pandas as pd


def iqr_bounds(values: pd.Series | np.ndarray, factor: float = 1.5) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    q1, q3 = np.percentile(array, [25, 75])
    iqr = float(q3 - q1)
    return float(q1 - factor * iqr), float(q3 + factor * iqr)


def clip_count(values: pd.Series | np.ndarray, low: float, high: float) -> int:
    array = np.asarray(values, dtype=float)
    return int(np.sum((array < low) | (array > high)))
