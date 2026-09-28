"""Automatic type detection and data-quality profiling."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from src.data.schema import ColumnRole, SchemaSummary
from src.utils.metrics import memory_usage_bytes, to_jsonable

_ID_NAME = re.compile(r"(^id$|_id$|^id_|uuid|guid|ssn|customer_?id)", re.I)
_TARGET_NAME = re.compile(r"(target|label|class|status|churn|outcome|approved)", re.I)


def _integer_like(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return False
    values = numeric.to_numpy(dtype=float)
    return bool(np.allclose(values, np.round(values)))


def _datetime_parse_rate(series: pd.Series) -> float:
    non_null = series.dropna()
    if non_null.empty:
        return 0.0
    sample = non_null.astype(str).head(500)
    if sample.str.fullmatch(r"-?\d+(\.\d+)?").mean() > 0.9:
        return 0.0
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return float(parsed.notna().mean())


def _outlier_fraction(series: pd.Series) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if len(numeric) < 8:
        return None
    q1, q3 = np.percentile(numeric, [25, 75])
    iqr = q3 - q1
    if iqr == 0:
        return 0.0
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return float(((numeric < low) | (numeric > high)).mean())


def infer_column(series: pd.Series, name: str, n_rows: int, thresholds: dict[str, Any]) -> dict[str, Any]:
    """Infer a column type and collect column-level quality statistics."""
    missing = float(series.isna().mean()) if n_rows else 0.0
    n_unique = int(series.nunique(dropna=True))
    cardinality = n_unique
    warnings: list[str] = []
    notes: list[str] = []
    detected = "categorical"

    if n_unique <= 1:
        detected = "constant"
        warnings.append("Constant column.")
    elif pd.api.types.is_bool_dtype(series):
        detected = "boolean"
    elif pd.api.types.is_datetime64_any_dtype(series):
        detected = "datetime"
    elif pd.api.types.is_numeric_dtype(series):
        unique_ratio = n_unique / max(n_rows, 1)
        if _ID_NAME.search(name) or (unique_ratio >= thresholds["id_uniqueness_threshold"] and _integer_like(series)):
            detected = "id"
            warnings.append("ID-like column.")
        else:
            detected = "numerical"
            if n_unique <= 12 and _integer_like(series):
                notes.append("Low-cardinality integer. Switch to categorical if the values are codes.")
    else:
        parse_rate = _datetime_parse_rate(series)
        unique_ratio = n_unique / max(series.notna().sum(), 1)
        if parse_rate >= 0.8 and n_unique > 3:
            detected = "datetime"
        elif _ID_NAME.search(name) or unique_ratio >= thresholds["id_uniqueness_threshold"]:
            detected = "id"
            warnings.append("ID-like column.")
        else:
            detected = "categorical"

    if detected == "categorical" and n_unique >= thresholds["high_cardinality_threshold"]:
        warnings.append("High cardinality.")
    if missing >= thresholds["missing_warning_threshold"]:
        warnings.append("High missingness.")

    numeric = pd.to_numeric(series, errors="coerce") if detected in {"numerical", "id"} else pd.Series(dtype=float)
    stats: dict[str, Any] = {
        "name": name,
        "dtype": str(series.dtype),
        "detected_type": detected,
        "missing_pct": missing,
        "missing_count": int(series.isna().sum()),
        "unique_count": n_unique,
        "cardinality": cardinality,
        "min": None,
        "max": None,
        "mean": None,
        "median": None,
        "std": None,
        "outlier_pct": None,
        "warnings": warnings,
        "notes": notes,
    }
    if detected == "numerical" and numeric.notna().any():
        stats.update(
            {
                "min": float(numeric.min()),
                "max": float(numeric.max()),
                "mean": float(numeric.mean()),
                "median": float(numeric.median()),
                "std": float(numeric.std(ddof=0) or 0.0),
                "outlier_pct": _outlier_fraction(series),
            }
        )
    elif detected == "datetime":
        parsed = pd.to_datetime(series, errors="coerce", format="mixed")
        if parsed.notna().any():
            stats["min"] = parsed.min().isoformat()
            stats["max"] = parsed.max().isoformat()
    return stats


def suggest_target(columns: list[dict[str, Any]]) -> str | None:
    candidates = []
    for column in columns:
        if column["detected_type"] in {"id", "constant", "datetime"}:
            continue
        name_match = bool(_TARGET_NAME.search(column["name"]))
        low_card = column["unique_count"] <= 12
        if name_match or (low_card and column["detected_type"] in {"categorical", "boolean"}):
            candidates.append(column)
    named = [column for column in candidates if _TARGET_NAME.search(column["name"])]
    pool = named or candidates
    if not pool:
        return None
    pool.sort(key=lambda column: (0 if _TARGET_NAME.search(column["name"]) else 1, column["unique_count"]))
    return str(pool[0]["name"])


def profile_dataframe(df: pd.DataFrame, thresholds: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a profiling report and default column roles."""
    thresholds = {
        "id_uniqueness_threshold": 0.98,
        "high_cardinality_threshold": 50,
        "missing_warning_threshold": 0.30,
        "duplicate_warning_threshold": 0.05,
        "imbalance_warning_threshold": 0.85,
        **(thresholds or {}),
    }
    columns = [infer_column(df[name], str(name), len(df), thresholds) for name in df.columns]
    numeric_count = sum(column["detected_type"] == "numerical" for column in columns)
    categorical_count = sum(column["detected_type"] in {"categorical", "boolean"} for column in columns)
    duplicate_rows = int(df.duplicated().sum())
    duplicate_rate = duplicate_rows / len(df) if len(df) else 0.0
    missing_cells = int(df.isna().sum().sum())
    missing_rate = missing_cells / max(df.size, 1)
    warnings: list[str] = []
    if duplicate_rate >= thresholds["duplicate_warning_threshold"]:
        warnings.append(
            f"Suspicious duplicate records: {duplicate_rows} rows ({duplicate_rate:.1%}) are exact duplicates."
        )
    elif duplicate_rows:
        warnings.append(f"{duplicate_rows} duplicate rows are present and were not removed.")
    for column in columns:
        for warning in column["warnings"]:
            warnings.append(f"{column['name']}: {warning}")

    target = suggest_target(columns)
    if target is not None:
        mode_rate = float(df[target].astype("string").value_counts(normalize=True, dropna=False).iloc[0])
        if mode_rate >= thresholds["imbalance_warning_threshold"]:
            warnings.append(
                f"Severe class imbalance on suggested target '{target}': the most common value covers {mode_rate:.1%} of rows."
            )

    roles = []
    for column in columns:
        detected = column["detected_type"]
        include = detected not in {"id", "constant"}
        assigned = detected if detected != "drop" else "drop"
        roles.append(
            ColumnRole(
                name=column["name"],
                detected_type=detected,
                assigned_type=assigned,
                include=include,
                scaler="standard",
                encoding="auto",
                notes="; ".join(column["notes"]),
            )
        )
    schema = SchemaSummary(columns=roles, suggested_target=target, warnings=warnings)
    report = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "numeric_features": int(numeric_count),
        "categorical_features": int(categorical_count),
        "missing_cells": missing_cells,
        "missing_rate": float(missing_rate),
        "duplicate_rows": duplicate_rows,
        "duplicate_rate": float(duplicate_rate),
        "memory_bytes": memory_usage_bytes(df),
        "column_profiles": columns,
        "warnings": warnings,
        "suggested_target": target,
        "roles": schema.as_records(),
    }
    return to_jsonable(report)
