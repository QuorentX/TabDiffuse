"""Privacy diagnostic tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.privacy import DISCLAIMER, privacy_report


def test_exact_matches_are_counted():
    real = pd.DataFrame({"age": [30, 40, 50], "city": ["Austin", "Denver", "Miami"]})
    synthetic = pd.DataFrame({"age": [30, 22, 40], "city": ["Austin", "Boston", "Denver"]})
    report = privacy_report(real, synthetic, max_rows=10, seed=1)
    assert report["exact_match"]["exact_matches"] == 2
    assert DISCLAIMER in report["disclaimer"]
    assert report["overall_label"] is not None


def test_privacy_ignores_columns_absent_from_synthetic_rows():
    real = pd.DataFrame(
        {
            "customer_id": [f"C{i}" for i in range(12)],
            "age": np.linspace(20, 60, 12),
            "city": ["Austin", "Denver"] * 6,
        }
    )
    synthetic = real.drop(columns=["customer_id"]).copy()
    report = privacy_report(real, synthetic, train_matrix=np.random.default_rng(0).normal(size=(12, 8)), max_rows=12, seed=1)
    assert report["nearest_neighbor"]["available"] is True
    assert report["space"] == "standardized comparison of shared columns"


def test_nearest_neighbor_on_known_points():
    real = pd.DataFrame({"age": [0.0, 10.0, 20.0], "income": [0.0, 0.0, 0.0]})
    synthetic = real.copy()
    report = privacy_report(real, synthetic, thresholds={"nn_ratio_low": 0.75, "nn_ratio_moderate": 0.45, "exact_match_low": 0.5, "exact_match_moderate": 0.8})
    assert report["nearest_neighbor"]["minimum_distance"] < 1e-6
    assert report["nearest_neighbor"]["label"] == "Higher observed risk"
