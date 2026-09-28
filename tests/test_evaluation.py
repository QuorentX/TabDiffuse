"""Quality, utility, and report tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.sample_data import generate_sample_dataset
from src.evaluation.evaluator import evaluate_quality
from src.evaluation.utility import evaluate_utility
from src.reporting.report_generator import build_report


def test_identical_tables_have_high_similarity():
    frame = pd.DataFrame(
        {
            "age": np.arange(30),
            "income": np.linspace(1000, 5000, 30),
            "city": ["Austin", "Denver", "Miami"] * 10,
        }
    )
    quality = evaluate_quality(frame, frame.copy())
    assert quality["similarity"]["distribution_similarity"] > 0.95
    assert quality["similarity"]["correlation_similarity"] > 0.95
    assert quality["similarity"]["categorical_similarity"] > 0.95
    assert quality["numerical"]["columns"]["age"]["ks_statistic"] == 0


def test_utility_reports_three_training_sources():
    rng = np.random.default_rng(0)
    rows = 80
    age = rng.normal(40, 8, rows)
    income = age * 1000 + rng.normal(0, 2000, rows)
    status = np.where(income > 40000, "Approved", "Rejected")
    frame = pd.DataFrame({"age": age, "income": income, "loan_status": status})
    train, holdout = frame.iloc[:60], frame.iloc[60:]
    synthetic = train.sample(60, replace=True, random_state=0)
    result = evaluate_utility(train, holdout, synthetic, "loan_status", seed=0, n_estimators=10)
    sources = {row["training_data"] for row in result["rows"]}
    assert {"Real", "Synthetic", "Real + Synthetic"} <= sources


def test_utility_scores_synthetic_rows_when_a_column_is_entirely_missing():
    rng = np.random.default_rng(1)
    rows = 80
    frame = pd.DataFrame(
        {
            "age": rng.normal(40, 8, rows),
            "income": rng.normal(50000, 8000, rows),
            "opened": pd.date_range("2020-01-01", periods=rows, freq="D"),
            "loan_status": np.where(rng.random(rows) > 0.4, "Approved", "Rejected"),
        }
    )
    train, holdout = frame.iloc[:60], frame.iloc[60:]
    synthetic = train.drop(columns=["opened"]).copy()
    synthetic["opened"] = pd.NaT
    result = evaluate_utility(train, holdout, synthetic, "loan_status", seed=1, n_estimators=8)
    sources = {row["training_data"] for row in result["rows"]}
    assert "Synthetic" in sources
    assert result["task"] == "classification"
    assert all(row["f1"] is not None for row in result["rows"])


def test_report_files_are_written(tmp_path):
    experiment = {
        "experiment_id": "EXP-2026-0001",
        "dataset_name": "demo.csv",
        "rows": 10,
        "columns": 2,
        "seed": 1,
        "created_at": "2026-01-01T00:00:00+00:00",
        "packages": {"numpy": "test"},
        "python": "3.12",
        "model_config": {"sde": "vp", "hidden_dim": 32, "layers": 2, "steps": 10, "schedule": "linear", "dropout": 0.1, "sampler": "reverse_sde", "time_sampling": "discrete"},
        "training": {"best_val_loss": 0.8, "duration_seconds": 1.2, "epoch": 2, "history": {"train_loss": [1.0, 0.8], "val_loss": [1.1, 0.9]}},
        "evaluation": {"similarity": {"distribution_similarity": 0.7, "correlation_similarity": 0.6, "categorical_similarity": 0.8}, "numerical": {"summary": {"mean_ks": 0.2}}, "categorical": {"summary": {}}, "correlation": {"available": False}},
        "privacy": {"disclaimer": "diagnostic", "overall_label": "Low observed risk", "exact_match": {"exact_match_rate": 0.0, "label": "Low observed risk"}, "nearest_neighbor": {}, "membership": {}},
        "notes": {"conclusion": "The run produced finite metrics."},
    }
    real = generate_sample_dataset(40, seed=9)[["customer_age", "income"]]
    outputs = build_report(experiment, tmp_path / "report.html", tmp_path / "report.pdf", real=real, synthetic=real.copy())
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "100% private" not in html
    assert "EXP-2026-0001" in html
    assert "do not constitute" in html
    assert outputs["pdf"].endswith("report.pdf")
    assert (tmp_path / "report.pdf").stat().st_size > 500
