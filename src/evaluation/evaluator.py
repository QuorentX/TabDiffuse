"""Run the quality evaluation bundle."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.evaluation.categorical import categorical_report
from src.evaluation.correlation import correlation_report
from src.evaluation.similarity import similarity_scores
from src.evaluation.statistical import numerical_report
from src.utils.exceptions import TabDiffuseError
from src.utils.metrics import to_jsonable


def evaluate_quality(real: pd.DataFrame, synthetic: pd.DataFrame) -> dict[str, Any]:
    """Compare real and synthetic tables on distribution, category, and correlation."""
    if real.empty or synthetic.empty:
        raise TabDiffuseError("Quality evaluation needs both a real table and a synthetic table.")
    shared = [column for column in real.columns if column in synthetic.columns]
    if not shared:
        raise TabDiffuseError("The real and synthetic tables have no columns in common.")
    real_shared = real[shared]
    synthetic_shared = synthetic[shared]
    numerical = numerical_report(real_shared, synthetic_shared)
    categorical = categorical_report(real_shared, synthetic_shared)
    correlation = correlation_report(real_shared, synthetic_shared)
    similarity = similarity_scores(numerical, correlation, categorical)
    return to_jsonable(
        {
            "rows_real": int(len(real_shared)),
            "rows_synthetic": int(len(synthetic_shared)),
            "columns": shared,
            "numerical": numerical,
            "categorical": categorical,
            "correlation": correlation,
            "similarity": similarity,
        }
    )
