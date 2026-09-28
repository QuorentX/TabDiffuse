"""Empirical privacy-risk diagnostics.

Nothing in this module is a differential-privacy guarantee. Threshold labels are
heuristics, and the thresholds are stored next to the results.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.metrics import to_jsonable

DISCLAIMER = (
    "These metrics are empirical privacy-risk diagnostics and do not constitute "
    "a formal differential privacy guarantee."
)

LOW = "Low observed risk"
MODERATE = "Moderate observed risk"
HIGHER = "Higher observed risk"


def privacy_report(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    train_matrix: np.ndarray | None = None,
    holdout_matrix: np.ndarray | None = None,
    synthetic_matrix: np.ndarray | None = None,
    quasi_identifiers: list[str] | None = None,
    sensitive_column: str | None = None,
    thresholds: dict[str, float] | None = None,
    max_rows: int = 4000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare synthetic rows with real rows using several diagnostic checks."""
    thresholds = _thresholds(thresholds)
    shared = [column for column in real.columns if column in synthetic.columns]
    exact = exact_match_diagnostic(real[shared], synthetic[shared], thresholds)
    real_space, syn_space, used_model_space = _spaces(real, synthetic, train_matrix, synthetic_matrix, max_rows, seed)
    neighbors = nearest_neighbor_diagnostic(real_space, syn_space, thresholds)
    membership = None
    matrices_ready = (
        holdout_matrix is not None
        and train_matrix is not None
        and synthetic_matrix is not None
        and len(holdout_matrix) > 1
        and np.asarray(train_matrix).ndim == 2
        and np.asarray(train_matrix).shape[1] == np.asarray(holdout_matrix).shape[1] == np.asarray(synthetic_matrix).shape[1]
    )
    if matrices_ready:
        membership = membership_diagnostic(train_matrix, holdout_matrix, synthetic_matrix, thresholds, max_rows, seed)
    disclosure = None
    if sensitive_column and quasi_identifiers:
        disclosure = attribute_disclosure_diagnostic(
            real, synthetic, quasi_identifiers, sensitive_column, thresholds, seed
        )
    labels = [item["label"] for item in (exact, neighbors, membership, disclosure) if item and item.get("label")]
    return to_jsonable(
        {
            "disclaimer": DISCLAIMER,
            "thresholds": thresholds,
            "threshold_note": "Labels use the thresholds stored in this report. Edit them in Settings.",
            "exact_match": exact,
            "nearest_neighbor": neighbors,
            "membership": membership,
            "attribute_disclosure": disclosure,
            "overall_label": _worst(labels),
            "space": "model feature space" if used_model_space else "standardized comparison of shared columns",
        }
    )


def exact_match_diagnostic(real: pd.DataFrame, synthetic: pd.DataFrame, thresholds: dict[str, float]) -> dict[str, Any]:
    real_keys = set(_signatures(real))
    synthetic_keys = _signatures(synthetic)
    matches = int(sum(key in real_keys for key in synthetic_keys))
    rate = matches / max(len(synthetic), 1)
    return {
        "exact_matches": matches,
        "exact_match_rate": rate,
        "synthetic_rows": int(len(synthetic)),
        "label": _higher_when_large(rate, thresholds["exact_match_low"], thresholds["exact_match_moderate"]),
        "definition": (
            "A row matches when categorical values are identical and integer columns match after "
            "rounding to the nearest integer. Other numeric columns match after rounding to 4 decimal places."
        ),
    }


def nearest_neighbor_diagnostic(real_x: np.ndarray, synthetic_x: np.ndarray, thresholds: dict[str, float]) -> dict[str, Any]:
    if len(real_x) < 3 or len(synthetic_x) < 1:
        return {"available": False, "message": "Not enough rows to compute nearest-neighbor distances."}
    real_x = np.nan_to_num(np.asarray(real_x, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    synthetic_x = np.nan_to_num(np.asarray(synthetic_x, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if real_x.shape[1] != synthetic_x.shape[1]:
        return {"available": False, "message": "Real and synthetic rows do not share a feature width, so distances were not computed."}
    real_nn = NearestNeighbors(n_neighbors=2).fit(real_x)
    real_distances, _ = real_nn.kneighbors(real_x)
    real_real = real_distances[:, 1]
    syn_nn = NearestNeighbors(n_neighbors=1).fit(real_x)
    syn_distances, _ = syn_nn.kneighbors(synthetic_x)
    nearest = syn_distances[:, 0]
    ratio = float(np.mean(nearest) / max(float(np.mean(real_real)), 1e-8))
    return {
        "available": True,
        "minimum_distance": float(np.min(nearest)),
        "mean_nearest_distance": float(np.mean(nearest)),
        "median_nearest_distance": float(np.median(nearest)),
        "mean_real_nearest_distance": float(np.mean(real_real)),
        "distance_ratio": ratio,
        "distance_sample": nearest[:400].astype(float).tolist(),
        "label": _higher_when_small(ratio, thresholds["nn_ratio_low"], thresholds["nn_ratio_moderate"]),
        "definition": (
            "Distance ratio is the mean synthetic-to-real nearest-neighbor distance divided by the "
            "mean real-to-real nearest-neighbor distance. Smaller ratios mean synthetic rows sit closer "
            "to real rows than real rows sit to each other."
        ),
    }


def membership_diagnostic(
    train_x: np.ndarray,
    holdout_x: np.ndarray,
    synthetic_x: np.ndarray,
    thresholds: dict[str, float],
    max_rows: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    train_s = np.nan_to_num(_sample(train_x, max_rows, rng), nan=0.0, posinf=0.0, neginf=0.0)
    hold_s = np.nan_to_num(_sample(holdout_x, max_rows, rng), nan=0.0, posinf=0.0, neginf=0.0)
    syn_s = np.nan_to_num(_sample(synthetic_x, max_rows, rng), nan=0.0, posinf=0.0, neginf=0.0)
    train_dist = NearestNeighbors(n_neighbors=1).fit(train_s).kneighbors(syn_s)[0][:, 0]
    hold_dist = NearestNeighbors(n_neighbors=1).fit(hold_s).kneighbors(syn_s)[0][:, 0]
    closer = float(np.mean(train_dist < hold_dist))
    advantage = closer - 0.5
    return {
        "available": True,
        "fraction_closer_to_training": closer,
        "advantage_over_half": float(advantage),
        "mean_distance_to_training": float(np.mean(train_dist)),
        "mean_distance_to_holdout": float(np.mean(hold_dist)),
        "train_distance_sample": train_dist[:400].astype(float).tolist(),
        "holdout_distance_sample": hold_dist[:400].astype(float).tolist(),
        "label": _higher_when_large(abs(advantage), thresholds["membership_advantage_low"], thresholds["membership_advantage_moderate"]),
        "definition": (
            "Compares each synthetic row's nearest training record with its nearest held-out record. "
            "The held-out records were not used to train the diffusion model. A large advantage means "
            "synthetic rows resemble the training split more than the held-out split."
        ),
        "sample_sizes": {"train": int(len(train_s)), "holdout": int(len(hold_s)), "synthetic": int(len(syn_s))},
    }


def attribute_disclosure_diagnostic(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    quasi_identifiers: list[str],
    sensitive_column: str,
    thresholds: dict[str, float],
    seed: int,
) -> dict[str, Any]:
    columns = [column for column in quasi_identifiers if column in real.columns and column in synthetic.columns]
    if sensitive_column not in real.columns or sensitive_column not in synthetic.columns:
        return {"available": False, "message": "The sensitive column is not in both tables."}
    if not columns or sensitive_column in columns:
        return {"available": False, "message": "Select quasi-identifiers that are distinct from the sensitive column."}
    classification = _is_classification(real[sensitive_column])
    attack = _attack_score(synthetic[columns + [sensitive_column]], real[columns], real[sensitive_column], columns, classification, seed)
    baseline = _majority_baseline(real[sensitive_column], classification)
    if attack is None or baseline is None or baseline <= 0:
        return {"available": False, "message": "The disclosure model could not be fit on these columns."}
    lift = float(attack / baseline) if classification else None
    label = None
    if lift is not None:
        label = _higher_when_large(max(lift - 1.0, 0.0) + 1.0 if False else lift, thresholds["disclosure_lift_low"], thresholds["disclosure_lift_moderate"]) if lift >= 1 else LOW
        if lift < thresholds["disclosure_lift_low"]:
            label = LOW
        elif lift < thresholds["disclosure_lift_moderate"]:
            label = MODERATE
        else:
            label = HIGHER
    return {
        "available": True,
        "quasi_identifiers": columns,
        "sensitive_column": sensitive_column,
        "task": "classification" if classification else "regression",
        "attack_score": attack,
        "baseline_score": baseline,
        "lift": lift,
        "label": label,
        "definition": (
            "A model is trained on synthetic rows to predict the sensitive column from the quasi-identifiers, "
            "then scored on real rows. The baseline is the real-data majority rate for classification, or the "
            "variance of the sensitive column for regression (reported as 1 - relative MSE, capped at 0)."
        ),
    }


def _attack_score(train_df, real_x, real_y, features, classification, seed) -> float | None:
    try:
        pipeline = _model(train_df[features], classification, seed)
        pipeline.fit(train_df[features], train_df.iloc[:, -1] if False else train_df[train_df.columns[-1]])
        predictions = pipeline.predict(real_x)
        if classification:
            return float(accuracy_score(real_y.astype(str), pd.Series(predictions).astype(str)))
        y = pd.to_numeric(real_y, errors="coerce")
        pred = pd.to_numeric(pd.Series(predictions), errors="coerce")
        mask = y.notna() & pred.notna()
        if mask.sum() < 5:
            return None
        return float(r2_score(y[mask], pred[mask]))
    except (ValueError, TypeError):
        return None


def _majority_baseline(series: pd.Series, classification: bool) -> float | None:
    if classification:
        return float(series.astype(str).value_counts(normalize=True).iloc[0])
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if len(numeric) < 5:
        return None
    variance = float(np.var(numeric))
    return 1.0 if variance == 0 else max(variance, 1e-8)


def _model(frame: pd.DataFrame, classification: bool, seed: int) -> Pipeline:
    numeric = [column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]
    categorical = [column for column in frame.columns if column not in numeric]
    preprocessor = ColumnTransformer(
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
    estimator = (
        RandomForestClassifier(n_estimators=40, max_depth=6, random_state=seed)
        if classification
        else RandomForestRegressor(n_estimators=40, max_depth=6, random_state=seed)
    )
    return Pipeline([("prep", preprocessor), ("model", estimator)])


def _spaces(real, synthetic, train_matrix, synthetic_matrix, max_rows, seed):
    """Return real and synthetic matrices with the same number of columns.

    Generated tables omit identifier columns, so a fallback built from the raw
    training table is wider than one built from the synthetic table. Distances
    are only meaningful after both sides use the shared columns.
    """
    rng = np.random.default_rng(seed)
    if train_matrix is not None and synthetic_matrix is not None:
        real_s = _sample(np.asarray(train_matrix, dtype=np.float64), max_rows, rng)
        syn_s = _sample(np.asarray(synthetic_matrix, dtype=np.float64), max_rows, rng)
        if real_s.ndim == 2 and syn_s.ndim == 2 and real_s.shape[1] == syn_s.shape[1] and real_s.shape[1] > 0:
            return real_s, syn_s, True
    shared = [column for column in real.columns if column in synthetic.columns]
    real_fb = _fallback_matrix(real.loc[:, shared] if shared else real.iloc[:, 0:0])
    syn_fb = _fallback_matrix(synthetic.loc[:, shared] if shared else synthetic.iloc[:, 0:0], reference=real.loc[:, shared] if shared else None)
    if real_fb.shape[1] != syn_fb.shape[1]:
        width = min(real_fb.shape[1], syn_fb.shape[1])
        real_fb = real_fb[:, :width]
        syn_fb = syn_fb[:, :width]
    return _sample(real_fb, max_rows, rng), _sample(syn_fb, max_rows, rng), False


def _fallback_matrix(df: pd.DataFrame, reference: pd.DataFrame | None = None) -> np.ndarray:
    source = reference if reference is not None else df
    columns = []
    for column in source.columns:
        if column not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(source[column]):
            values = pd.to_numeric(df[column], errors="coerce")
            center = pd.to_numeric(source[column], errors="coerce")
            fill = float(center.median()) if center.notna().any() else 0.0
            scale = float(center.std(ddof=0) or 1.0)
            columns.append(((values.fillna(fill) - fill) / (scale if scale > 1e-8 else 1.0)).to_numpy())
        else:
            categories = source[column].astype("string").fillna("<NA>")
            top = categories.value_counts().head(16).index.tolist()
            values = df[column].astype("string").fillna("<NA>")
            for category in top:
                columns.append((values == category).astype(float).to_numpy())
    if not columns:
        return np.zeros((len(df), 1))
    return np.column_stack(columns).astype(np.float64)


def _signatures(df: pd.DataFrame) -> list[str]:
    parts = []
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce")
            integer_like = numeric.dropna().empty or np.allclose(
                numeric.dropna().to_numpy(dtype=float), np.round(numeric.dropna().to_numpy(dtype=float))
            )
            rendered = numeric.round(0 if integer_like else 4).map(lambda value: "<NA>" if pd.isna(value) else f"{value:.4f}")
            parts.append(rendered.astype(str))
        else:
            parts.append(series.astype("string").fillna("<NA>"))
    if not parts:
        return [""] * len(df)
    frame = pd.concat(parts, axis=1)
    return frame.astype(str).agg("||".join, axis=1).tolist()


def _sample(matrix: np.ndarray, max_rows: int, rng: np.random.Generator) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float64)
    if len(array) <= max_rows:
        return array
    index = rng.choice(len(array), size=max_rows, replace=False)
    return array[index]


def _is_classification(series: pd.Series) -> bool:
    if not pd.api.types.is_numeric_dtype(series):
        return True
    return int(series.nunique(dropna=True)) <= 12


def _thresholds(thresholds: dict[str, float] | None) -> dict[str, float]:
    defaults = {
        "exact_match_low": 0.005,
        "exact_match_moderate": 0.02,
        "nn_ratio_low": 0.75,
        "nn_ratio_moderate": 0.45,
        "membership_advantage_low": 0.05,
        "membership_advantage_moderate": 0.12,
        "disclosure_lift_low": 1.1,
        "disclosure_lift_moderate": 1.4,
    }
    if thresholds:
        defaults.update({key: float(value) for key, value in thresholds.items() if key in defaults})
    return defaults


def _higher_when_large(value: float, low: float, moderate: float) -> str:
    if value < low:
        return LOW
    if value < moderate:
        return MODERATE
    return HIGHER


def _higher_when_small(value: float, low: float, moderate: float) -> str:
    if value >= low:
        return LOW
    if value >= moderate:
        return MODERATE
    return HIGHER


def _worst(labels: list[str]) -> str | None:
    order = {LOW: 0, MODERATE: 1, HIGHER: 2}
    if not labels:
        return None
    return max(labels, key=lambda label: order.get(label, 0))
