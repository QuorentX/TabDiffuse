"""Fit the preprocessor on the training split and transform every split."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.preprocessing.transformer import TabularPreprocessor, preprocessor_config_from_app
from src.utils.logger import log_event


def fit_splits(
    splits: dict[str, Any],
    roles: list[dict[str, Any]],
    config: Any,
    seed: int,
) -> dict[str, Any]:
    """Fit on training rows only, then transform train, validation, and holdout."""
    preprocessor = TabularPreprocessor(preprocessor_config_from_app(config, seed))
    preprocessor.fit(splits["train"], roles)
    transformed = {
        "train": preprocessor.transform(splits["train"]),
        "validation": preprocessor.transform(splits["validation"]),
        "holdout": preprocessor.transform(splits["holdout"]) if len(splits["holdout"]) else None,
    }
    log_event(
        "preprocessing",
        "preprocessor_fit",
        status="ok",
        error=None,
    )
    return {
        "preprocessor": preprocessor,
        "arrays": transformed,
        "metadata": preprocessor.metadata(),
        "actions": preprocessor.actions,
        "feature_dim": preprocessor.feature_dim,
        "output_columns": list(preprocessor.output_columns),
    }
