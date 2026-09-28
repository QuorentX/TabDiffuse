"""Inverse scaling, schema checks, and optional constraints."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.generation.constraints import apply_constraints
from src.preprocessing.transformer import TabularPreprocessor
from src.utils.exceptions import TabDiffuseError


def postprocess_samples(
    matrix: np.ndarray,
    preprocessor: TabularPreprocessor,
    constraints: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Map continuous samples back to the original schema and apply rules."""
    frame, inverse_report = preprocessor.inverse_transform(matrix)
    expected = list(preprocessor.output_columns)
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise TabDiffuseError("Synthetic schema validation failed. Restored columns do not match the training schema.")
    frame = frame[expected]
    invalid_numeric = 0
    for column in frame.columns:
        if pd.api.types.is_numeric_dtype(frame[column]):
            bad = ~np.isfinite(pd.to_numeric(frame[column], errors="coerce"))
            invalid_numeric += int(bad.sum())
            frame.loc[bad, column] = np.nan
    before_rules = len(frame)
    constrained, rule_report = apply_constraints(frame, constraints or [])
    return {
        "frame": constrained,
        "inverse": inverse_report,
        "constraints": rule_report,
        "invalid_numeric_values": invalid_numeric,
        "rows_before_constraints": before_rules,
        "rows": int(len(constrained)),
    }
