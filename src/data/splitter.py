"""Train, validation, and held-out splits.

The held-out split is reserved for ML utility and membership diagnostics.
The diffusion model is fit only on the training split. The preprocessor is
also fit only on the training split.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.exceptions import TabDiffuseError


def split_dataframe(
    df: pd.DataFrame,
    holdout_ratio: float = 0.15,
    validation_ratio: float = 0.15,
    seed: int = 42,
    stratify_column: str | None = None,
) -> dict[str, Any]:
    """Split a frame into diffusion-train, diffusion-validation, and holdout."""
    if df.empty:
        raise TabDiffuseError("Cannot split an empty dataset.")
    if holdout_ratio < 0 or validation_ratio < 0:
        raise TabDiffuseError("Split ratios cannot be negative.")
    if holdout_ratio + validation_ratio >= 0.9:
        raise TabDiffuseError(
            "Holdout and validation ratios leave too little training data. Reduce one of them."
        )

    stratify = _stratify_series(df, stratify_column) if stratify_column else None
    try:
        remain, holdout = train_test_split(
            df,
            test_size=holdout_ratio if holdout_ratio > 0 else None,
            random_state=seed,
            stratify=stratify if holdout_ratio > 0 else None,
        ) if holdout_ratio > 0 else (df, df.iloc[0:0].copy())
    except ValueError:
        remain, holdout = train_test_split(
            df,
            test_size=holdout_ratio,
            random_state=seed,
            stratify=None,
        )

    val_size = validation_ratio / max(1e-8, 1 - holdout_ratio) if holdout_ratio < 1 else validation_ratio
    stratify_remain = _stratify_series(remain, stratify_column) if stratify_column else None
    try:
        train, validation = train_test_split(
            remain,
            test_size=val_size,
            random_state=seed,
            stratify=stratify_remain,
        )
    except ValueError:
        train, validation = train_test_split(remain, test_size=val_size, random_state=seed, stratify=None)

    if len(train) < 8 or len(validation) < 2:
        raise TabDiffuseError(
            "The split left too few training or validation rows. Use a larger dataset or smaller split ratios."
        )
    return {
        "train": train.reset_index(drop=True),
        "validation": validation.reset_index(drop=True),
        "holdout": holdout.reset_index(drop=True),
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "holdout_rows": int(len(holdout)),
        "seed": seed,
        "holdout_ratio": holdout_ratio,
        "validation_ratio": validation_ratio,
        "stratify_column": stratify_column,
    }


def _stratify_series(df: pd.DataFrame, column: str | None) -> pd.Series | None:
    if column is None or column not in df.columns:
        return None
    series = df[column]
    if series.nunique(dropna=True) < 2 or series.nunique(dropna=True) > 20:
        return None
    counts = series.value_counts(dropna=False)
    if counts.min() < 3:
        return None
    return series
