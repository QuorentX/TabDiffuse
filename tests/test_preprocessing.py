"""Preprocessing round-trip tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.sample_data import generate_sample_dataset
from src.data.profiler import profile_dataframe
from src.preprocessing.transformer import TabularPreprocessor


def _config():
    return {
        "numerical_imputer": "median",
        "numerical_scaler": "standard",
        "outlier_method": "none",
        "iqr_factor": 1.5,
        "onehot_max_cardinality": 32,
        "embedding_dim": 4,
        "categorical_missing_token": "__MISSING__",
        "group_rare_categories": False,
        "rare_category_min_frequency": 0.01,
        "seed": 3,
    }


def test_numeric_and_categorical_roundtrip():
    frame = pd.DataFrame(
        {
            "age": [20, 30, 40, 50],
            "income": [1000.5, 2000.0, 1500.25, 3000.0],
            "city": ["Austin", "Denver", "Austin", "Miami"],
        }
    )
    roles = [
        {"name": "age", "assigned_type": "numerical", "include": True, "scaler": "standard", "encoding": "auto"},
        {"name": "income", "assigned_type": "numerical", "include": True, "scaler": "standard", "encoding": "auto"},
        {"name": "city", "assigned_type": "categorical", "include": True, "scaler": "standard", "encoding": "onehot"},
    ]
    preprocessor = TabularPreprocessor(_config()).fit(frame, roles)
    restored, report = preprocessor.inverse_transform(preprocessor.transform(frame))
    assert report["invalid_dates"] == 0
    assert np.allclose(restored["age"], frame["age"])
    assert np.allclose(restored["income"], frame["income"])
    assert restored["city"].tolist() == frame["city"].tolist()


def test_embedding_roundtrip_and_exclusion():
    frame = pd.DataFrame(
        {
            "customer_id": [f"C{i}" for i in range(10)],
            "city": [f"City {i % 6}" for i in range(10)],
        }
    )
    roles = [
        {"name": "customer_id", "assigned_type": "id", "include": False, "scaler": "standard", "encoding": "auto"},
        {"name": "city", "assigned_type": "categorical", "include": True, "scaler": "standard", "encoding": "embedding"},
    ]
    config = _config()
    config["onehot_max_cardinality"] = 1
    preprocessor = TabularPreprocessor(config).fit(frame, roles)
    restored, _ = preprocessor.inverse_transform(preprocessor.transform(frame))
    assert "customer_id" not in restored.columns
    assert restored["city"].tolist() == frame["city"].tolist()
    assert any(action["action"] == "excluded" for action in preprocessor.actions)


def test_date_parts_roundtrip():
    frame = pd.DataFrame({"opened": ["2020-01-15", "2021-06-02", "2022-12-31", "2020-01-15"]})
    roles = [{"name": "opened", "assigned_type": "datetime", "include": True, "scaler": "standard", "encoding": "auto"}]
    preprocessor = TabularPreprocessor(_config()).fit(frame, roles)
    restored, report = preprocessor.inverse_transform(preprocessor.transform(frame))
    assert report["invalid_dates"] == 0
    assert report["repaired_dates"] == 0
    assert restored["opened"].tolist() == frame["opened"].tolist()


def test_impossible_calendar_day_is_repaired():
    frame = pd.DataFrame({"opened": ["2020-02-15", "2020-01-31", "2021-06-02", "2020-02-15"]})
    roles = [{"name": "opened", "assigned_type": "datetime", "include": True, "scaler": "standard", "encoding": "auto"}]
    preprocessor = TabularPreprocessor(_config()).fit(frame, roles)
    matrix = preprocessor.transform(frame)
    parts = preprocessor.groups[0]["parts"]
    matrix = matrix.copy()
    matrix[0, 1] = parts["month"].transform(pd.Series([2.0], name="month"))[0]
    matrix[0, 2] = parts["day"].transform(pd.Series([31.0], name="day"))[0]
    restored, report = preprocessor.inverse_transform(matrix)
    assert report["invalid_dates"] == 0
    assert report["repaired_dates"] >= 1
    parsed = pd.to_datetime(restored["opened"], errors="coerce")
    assert parsed.notna().all()
    assert int(parsed.iloc[0].month) == 2
    assert int(parsed.iloc[0].day) <= 29


def test_sample_schema_has_no_silent_drop():
    frame = generate_sample_dataset(60, seed=4)
    profile = profile_dataframe(frame)
    preprocessor = TabularPreprocessor(_config()).fit(frame, profile["roles"])
    excluded = {item["column"] for item in preprocessor.metadata()["excluded"]}
    assert "customer_id" in excluded
    assert "income" in preprocessor.output_columns
    assert "loan_status" in preprocessor.output_columns
