"""Loader and schema tests."""

from __future__ import annotations

import pandas as pd

from src.data.loader import load_tabular
from src.data.profiler import profile_dataframe
from src.data.sample_data import generate_sample_dataset
from src.data.validator import validate_dataframe
from src.utils.exceptions import TabDiffuseError


def test_load_csv_excel_parquet(tmp_path):
    frame = generate_sample_dataset(80, seed=1)
    csv_path = tmp_path / "customers.csv"
    frame.to_csv(csv_path, index=False)
    loaded = load_tabular(csv_path)
    assert list(loaded.columns) == list(frame.columns)
    assert len(loaded) == len(frame)

    excel_path = tmp_path / "customers.xlsx"
    frame.to_excel(excel_path, index=False)
    assert len(load_tabular(excel_path)) == len(frame)

    parquet_path = tmp_path / "customers.parquet"
    frame.to_parquet(parquet_path, index=False)
    assert len(load_tabular(parquet_path)) == len(frame)


def test_reject_unsupported_and_empty(tmp_path):
    bad = tmp_path / "notes.txt"
    bad.write_text("hello", encoding="utf-8")
    try:
        load_tabular(bad)
        assert False, "expected an error"
    except TabDiffuseError as exc:
        assert "Unsupported" in exc.message

    empty = tmp_path / "empty.csv"
    empty.write_text("age,city\n", encoding="utf-8")
    try:
        load_tabular(empty)
        assert False, "expected an error"
    except TabDiffuseError as exc:
        assert "no data rows" in exc.message


def test_profile_detects_id_and_target():
    frame = generate_sample_dataset(200, seed=2)
    profile = profile_dataframe(frame)
    types = {column["name"]: column["detected_type"] for column in profile["column_profiles"]}
    assert types["customer_id"] == "id"
    assert types["income"] == "numerical"
    assert types["loan_status"] == "categorical"
    assert types["account_open_date"] == "datetime"
    assert profile["suggested_target"] == "loan_status"
    roles = {role["name"]: role for role in profile["roles"]}
    assert roles["customer_id"]["include"] is False


def test_two_demo_sets_are_ready_to_process():
    from src.data.sample_data import DEMO_SETS, write_demo_sets

    paths = write_demo_sets(n_rows=1000)
    frames = {key: load_tabular(path) for key, path in paths.items()}
    assert set(frames) == {"A", "B"}
    assert len(frames["A"]) >= 1000
    assert len(frames["B"]) >= 1000
    assert list(frames["A"].columns) != list(frames["B"].columns)
    for key, frame in frames.items():
        profile = profile_dataframe(frame)
        assert profile["rows"] == len(frame)
        assert profile["numeric_features"] >= 3
        assert profile["categorical_features"] >= 3
        assert profile["suggested_target"] == DEMO_SETS[key]["target"]
        report = validate_dataframe(frame, min_rows=32)
        assert report["ok"] is True


def test_validator_blocks_tiny_tables():
    report = validate_dataframe(pd.DataFrame({"age": [1, 2, 3]}), min_rows=32)
    assert report["ok"] is False
    assert "32" in report["errors"][0]
