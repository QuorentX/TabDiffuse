"""Train-on-synthetic, test-on-real utility diagnostics.

Models are lightly configured diagnostic learners. A higher score is a
measurement, not an automatic claim that one training source is better.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.exceptions import TabDiffuseError
from src.utils.metrics import to_jsonable


def evaluate_utility(
    real_train: pd.DataFrame,
    real_holdout: pd.DataFrame,
    synthetic: pd.DataFrame,
    target: str,
    *,
    task: str | None = None,
    seed: int = 42,
    max_rows: int = 20000,
    n_estimators: int = 60,
) -> dict[str, Any]:
    """Compare real, synthetic, and combined training against a real holdout."""
    for name, frame in ("real training", real_train), ("holdout", real_holdout), ("synthetic", synthetic):
        if target not in frame.columns:
            raise TabDiffuseError(f"The target column is missing from the {name} table.")
    if len(real_holdout) < 5:
        raise TabDiffuseError("The real holdout split is too small for a utility evaluation.")
    resolved = task or ("classification" if _classification(real_train[target]) else "regression")
    if resolved not in {"classification", "regression"}:
        raise TabDiffuseError("Task must be classification or regression.")

    real_train = _sample(real_train, max_rows, seed)
    real_holdout = _sample(real_holdout, max_rows, seed + 1)
    synthetic = _sample(synthetic, max_rows, seed + 2)
    features = [
        column
        for column in real_train.columns
        if column != target and column in synthetic.columns and column in real_holdout.columns
    ]
    if not features:
        raise TabDiffuseError("No shared feature columns remain for the utility evaluation.")

    sources = {
        "Real": real_train,
        "Synthetic": synthetic,
        "Real + Synthetic": pd.concat([real_train, synthetic], ignore_index=True),
    }
    rows = []
    notes = []
    for source_name, frame in sources.items():
        trainable = frame.dropna(subset=[target])
        if len(trainable) < 8:
            notes.append(f"{source_name} was skipped because fewer than 8 labeled rows were available.")
            continue
        for model_name, estimator in _estimators(resolved, seed, n_estimators):
            try:
                metrics = _fit_and_score(
                    estimator,
                    trainable[features],
                    trainable[target],
                    real_holdout[features],
                    real_holdout[target],
                    resolved,
                )
            except (ValueError, TypeError) as exc:
                notes.append(f"{model_name} on {source_name} could not be scored ({type(exc).__name__}).")
                continue
            rows.append({"model": model_name, "training_data": source_name, **metrics, "train_rows": int(len(trainable))})
    return to_jsonable(
        {
            "task": resolved,
            "target": target,
            "features": features,
            "holdout_rows": int(len(real_holdout)),
            "real_train_rows": int(len(real_train)),
            "synthetic_rows": int(len(synthetic)),
            "max_rows": max_rows,
            "subsampled": len(real_train) == max_rows or len(synthetic) == max_rows,
            "note": "Diagnostic models use fixed hyperparameters. Compare the measured scores yourself.",
            "rows": rows,
            "notes": notes,
        }
    )


def _fit_and_score(estimator, x_train, y_train, x_test, y_test, task: str) -> dict[str, float | None]:
    x_train, x_test = _model_frames(x_train, x_test)
    if x_train.empty or x_train.shape[1] == 0:
        raise ValueError("No usable feature columns remain.")
    pipeline = Pipeline([("prep", _preprocessor(x_train)), ("model", estimator)])
    y_train_fit = y_train.astype(str) if task == "classification" else pd.to_numeric(y_train, errors="coerce")
    y_test_fit = y_test.astype(str) if task == "classification" else pd.to_numeric(y_test, errors="coerce")
    if task == "regression":
        train_mask = y_train_fit.notna()
        test_mask = y_test_fit.notna()
        pipeline.fit(x_train.loc[train_mask], y_train_fit.loc[train_mask])
        predictions = pipeline.predict(x_test.loc[test_mask])
        actual = y_test_fit.loc[test_mask].to_numpy(dtype=float)
        return {
            "mae": float(mean_absolute_error(actual, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(actual, predictions))),
            "r2": float(r2_score(actual, predictions)),
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "roc_auc": None,
        }
    pipeline.fit(x_train, y_train_fit)
    predictions = pipeline.predict(x_test)
    metrics: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_test_fit, predictions)),
        "precision": float(precision_score(y_test_fit, predictions, average="macro", zero_division=0)),
        "recall": float(recall_score(y_test_fit, predictions, average="macro", zero_division=0)),
        "f1": float(f1_score(y_test_fit, predictions, average="macro", zero_division=0)),
        "roc_auc": _roc_auc(pipeline, x_test, y_test_fit),
        "mae": None,
        "rmse": None,
        "r2": None,
    }
    return metrics


def _roc_auc(pipeline: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> float | None:
    try:
        probabilities = pipeline.predict_proba(x_test)
        classes = list(pipeline.named_steps["model"].classes_)
        if len(classes) == 2:
            positive = probabilities[:, 1]
            return float(roc_auc_score(y_test, positive))
        return float(roc_auc_score(y_test, probabilities, multi_class="ovr", average="macro"))
    except (ValueError, AttributeError):
        return None


def _model_frames(x_train: pd.DataFrame, x_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop columns the training table cannot impute and align dates to numbers."""
    train = x_train.copy()
    test = x_test.reindex(columns=train.columns).copy()
    empty = [column for column in train.columns if int(train[column].notna().sum()) == 0]
    if empty:
        train = train.drop(columns=empty)
        test = test.drop(columns=empty)
    for column in list(train.columns):
        train_time = pd.api.types.is_datetime64_any_dtype(train[column])
        test_time = pd.api.types.is_datetime64_any_dtype(test[column])
        if train_time or test_time:
            train[column] = _epoch_seconds(train[column])
            test[column] = _epoch_seconds(test[column])
    train = train.replace([np.inf, -np.inf], np.nan)
    test = test.replace([np.inf, -np.inf], np.nan)
    return train, test


def _epoch_seconds(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    seconds = parsed.astype("int64").astype("float64") / 1e9
    return seconds.where(parsed.notna(), np.nan)


def _preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    numeric = [column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]
    categorical = [column for column in frame.columns if column not in numeric]
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ]
    )


def _estimators(task: str, seed: int, n_estimators: int) -> list[tuple[str, Any]]:
    if task == "classification":
        return [
            ("Logistic Regression", LogisticRegression(max_iter=400)),
            ("Random Forest", RandomForestClassifier(n_estimators=n_estimators, max_depth=8, random_state=seed)),
            ("Gradient Boosting", GradientBoostingClassifier(random_state=seed)),
        ]
    return [
        ("Linear Regression", LinearRegression()),
        ("Random Forest", RandomForestRegressor(n_estimators=n_estimators, max_depth=8, random_state=seed)),
        ("Gradient Boosting", GradientBoostingRegressor(random_state=seed)),
    ]


def _classification(series: pd.Series) -> bool:
    if not pd.api.types.is_numeric_dtype(series):
        return series.nunique(dropna=True) <= 30
    return int(series.nunique(dropna=True)) <= 12


def _sample(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=seed)
