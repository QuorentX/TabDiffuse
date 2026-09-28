"""Dataset validation before profiling and training."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.utils.exceptions import TabDiffuseError


def validate_dataframe(df: pd.DataFrame, min_rows: int = 32) -> dict[str, Any]:
    """Check structural problems that should block training."""
    if not isinstance(df, pd.DataFrame):
        raise TabDiffuseError("A tabular dataset is required.")
    errors: list[str] = []
    warnings: list[str] = []
    if df.shape[1] == 0:
        errors.append("The dataset has no columns.")
    if len(df) == 0:
        errors.append("The dataset has no rows.")
    elif len(df) < min_rows:
        errors.append(
            f"Only {len(df)} rows were found. At least {min_rows} rows are required before training."
        )

    numeric = df.select_dtypes(include=[np.number])
    if not numeric.empty:
        infinite_columns = [column for column in numeric.columns if np.isinf(numeric[column].to_numpy()).any()]
        if infinite_columns:
            errors.append(
                "Infinite values were found in: "
                + ", ".join(map(str, infinite_columns[:12]))
                + ". Replace them before training."
            )

    all_missing = [column for column in df.columns if df[column].isna().all()]
    if all_missing:
        warnings.append(
            "These columns are entirely missing and should be excluded: " + ", ".join(map(str, all_missing[:12]))
        )

    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows:
        warnings.append(f"{duplicate_rows} duplicate rows were detected. They are kept unless you remove them.")

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "duplicate_rows": duplicate_rows if len(df) else 0,
    }


def assert_valid(df: pd.DataFrame, min_rows: int = 32) -> dict[str, Any]:
    report = validate_dataframe(df, min_rows=min_rows)
    if not report["ok"]:
        raise TabDiffuseError(" ".join(report["errors"]))
    return report
