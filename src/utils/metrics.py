"""Shared numeric helpers and JSON conversion."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def to_jsonable(value: Any) -> Any:
    """Convert NumPy and pandas objects into JSON-safe Python values."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [to_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if np.isnan(number) or np.isinf(number):
            return None
        return number
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def memory_usage_bytes(df: pd.DataFrame) -> int:
    return int(df.memory_usage(deep=True).sum())


def format_bytes(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def finite_float(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if np.isnan(number) or np.isinf(number):
        return default
    return number
