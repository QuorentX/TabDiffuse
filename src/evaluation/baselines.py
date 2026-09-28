"""Non-diffusion baselines. Results are measured, not ranked in code."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.utils.exceptions import TabDiffuseError
from src.utils.metrics import to_jsonable


def generate_baselines(
    train: pd.DataFrame,
    n_samples: int,
    seed: int = 42,
    *,
    resample: bool = True,
    gaussian: bool = True,
    copula: bool = True,
    ctgan: bool = False,
    ctgan_epochs: int = 30,
) -> dict[str, Any]:
    """Fit simple generators on the training split and draw synthetic rows."""
    if n_samples < 1:
        raise TabDiffuseError("Baseline sample size must be at least 1.")
    results: dict[str, Any] = {}
    if resample:
        results["Random resampling"] = _timed(resample_rows, train, n_samples, seed)
    if gaussian:
        results["Gaussian"] = _timed(gaussian_sample, train, n_samples, seed)
    if copula:
        results["Gaussian copula"] = _timed(copula_sample, train, n_samples, seed)
    if ctgan:
        results["CTGAN"] = ctgan_sample(train, n_samples, seed, ctgan_epochs)
    return results


def resample_rows(train: pd.DataFrame, n_samples: int, seed: int) -> pd.DataFrame:
    """Draw training rows with replacement. This copies real records."""
    return train.sample(n=n_samples, replace=True, random_state=seed).reset_index(drop=True)


def gaussian_sample(train: pd.DataFrame, n_samples: int, seed: int) -> pd.DataFrame:
    """Sample a Gaussian on numeric columns and independent categorical marginals.

    Categorical columns are sampled independently, so this baseline is not
    expected to preserve mixed-type dependence. Values outside the observed
    numeric range are kept and counted by the caller if needed.
    """
    rng = np.random.default_rng(seed)
    numeric_columns = [column for column in train.columns if pd.api.types.is_numeric_dtype(train[column])]
    other_columns = [column for column in train.columns if column not in numeric_columns]
    frame = pd.DataFrame(index=range(n_samples))
    if numeric_columns:
        matrix = train[numeric_columns].apply(pd.to_numeric, errors="coerce")
        means = matrix.mean().to_numpy()
        filled = matrix.fillna(matrix.median())
        covariance = np.cov(filled.to_numpy(dtype=float), rowvar=False)
        if np.ndim(covariance) == 0:
            covariance = np.array([[float(covariance)]])
        covariance = np.atleast_2d(covariance) + np.eye(len(numeric_columns)) * 1e-6
        draw = rng.multivariate_normal(means, covariance, size=n_samples)
        for index, column in enumerate(numeric_columns):
            values = draw[:, index]
            if _integer_like(train[column]):
                values = np.rint(values)
            frame[column] = values
    for column in other_columns:
        observed = train[column].dropna()
        if observed.empty:
            frame[column] = np.nan
            continue
        probabilities = observed.astype(str).value_counts(normalize=True)
        frame[column] = rng.choice(probabilities.index.to_numpy(), size=n_samples, p=probabilities.to_numpy())
    return frame[train.columns]


def copula_sample(train: pd.DataFrame, n_samples: int, seed: int) -> pd.DataFrame:
    """Gaussian copula on rank-transformed columns, with discrete inverse CDFs."""
    rng = np.random.default_rng(seed)
    transformed = []
    specs = []
    for column in train.columns:
        series = train[column]
        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce")
            fill = float(numeric.median()) if numeric.notna().any() else 0.0
            values = numeric.fillna(fill).to_numpy(dtype=float)
            ranks = pd.Series(values).rank(method="average").to_numpy()
            uniform = np.clip(ranks / (len(values) + 1), 1e-6, 1 - 1e-6)
            specs.append({"column": column, "kind": "numeric", "values": np.sort(values), "integer": _integer_like(series)})
        else:
            text = series.astype("string").fillna("<NA>")
            codes, uniques = pd.factorize(text, use_na_sentinel=False)
            jitter = (codes.astype(float) + rng.uniform(0.05, 0.95, size=len(codes))) / max(len(uniques), 1)
            uniform = np.clip(jitter, 1e-6, 1 - 1e-6)
            specs.append({"column": column, "kind": "categorical", "uniques": uniques.tolist()})
        transformed.append(norm.ppf(uniform))
    if not transformed:
        raise TabDiffuseError("The copula baseline needs at least one column.")
    latent = np.column_stack(transformed)
    covariance = np.cov(latent, rowvar=False)
    if np.ndim(covariance) == 0:
        covariance = np.array([[1.0]])
    covariance = np.atleast_2d(covariance) + np.eye(latent.shape[1]) * 1e-4
    draw = rng.multivariate_normal(np.zeros(latent.shape[1]), covariance, size=n_samples)
    uniform_draw = norm.cdf(draw)
    frame = pd.DataFrame(index=range(n_samples))
    for index, spec in enumerate(specs):
        probs = uniform_draw[:, index]
        if spec["kind"] == "numeric":
            positions = np.clip((probs * (len(spec["values"]) - 1)).astype(int), 0, len(spec["values"]) - 1)
            values = spec["values"][positions]
            frame[spec["column"]] = np.rint(values) if spec["integer"] else values
        else:
            positions = np.clip((probs * len(spec["uniques"])).astype(int), 0, len(spec["uniques"]) - 1)
            labels = [spec["uniques"][int(position)] for position in positions]
            frame[spec["column"]] = [np.nan if label == "<NA>" else label for label in labels]
    return frame[list(train.columns)]


def ctgan_sample(train: pd.DataFrame, n_samples: int, seed: int, epochs: int) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        from ctgan import CTGAN
    except ImportError:
        return {
            "available": False,
            "message": "CTGAN is not installed. Install the optional ctgan package to include this baseline.",
            "duration_seconds": 0.0,
        }
    discrete = [column for column in train.columns if not pd.api.types.is_numeric_dtype(train[column])]
    model = CTGAN(epochs=epochs, verbose=False, cuda=False)
    try:
        model.fit(train.copy(), discrete_columns=discrete)
        frame = model.sample(n_samples)
    except Exception as exc:
        return {
            "available": False,
            "message": f"CTGAN could not be trained ({type(exc).__name__}).",
            "duration_seconds": time.perf_counter() - started,
        }
    return {
        "available": True,
        "frame": frame,
        "duration_seconds": time.perf_counter() - started,
        "rows": int(len(frame)),
    }


def _timed(function, train: pd.DataFrame, n_samples: int, seed: int) -> dict[str, Any]:
    started = time.perf_counter()
    frame = function(train, n_samples, seed)
    return {
        "available": True,
        "frame": frame,
        "duration_seconds": time.perf_counter() - started,
        "rows": int(len(frame)),
    }


def _integer_like(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return False
    values = numeric.to_numpy(dtype=float)
    return bool(np.allclose(values, np.round(values), atol=1e-6))


def baseline_summary_table(comparisons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return to_jsonable(comparisons)
