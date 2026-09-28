"""Generation and constraint tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.diffusion.model import build_model
from src.generation.constraints import apply_constraints, parse_constraints
from src.generation.generator import SyntheticGenerator
from src.preprocessing.transformer import TabularPreprocessor
from src.utils.exceptions import TabDiffuseError


def test_constraints_report_rejections_and_clips():
    frame = pd.DataFrame({"age": [10, 30, 120], "income": [15.0, -5.0, 40.0]})
    rules = parse_constraints(
        """
constraints:
  - column: age
    type: range
    min: 18
    max: 100
    action: reject
  - column: income
    type: minimum
    value: 0
    action: clip
"""
    )
    kept, report = apply_constraints(frame, rules)
    assert report["rejected_rows"] == 2
    assert report["modified_rows"] == 1
    assert kept["income"].min() >= 0
    assert len(kept) == 1


def test_invalid_constraint_is_explained():
    try:
        parse_constraints("constraints:\n  - column: age\n    type: mystery\n")
        assert False
    except TabDiffuseError as exc:
        assert "unsupported type" in exc.message


def test_generator_restores_schema():
    frame = pd.DataFrame(
        {
            "age": np.linspace(20, 60, 40),
            "income": np.linspace(20000, 80000, 40),
            "city": ["Austin", "Denver"] * 20,
            "loan_status": ["Approved", "Rejected"] * 20,
        }
    )
    roles = [
        {"name": "age", "assigned_type": "numerical", "include": True, "scaler": "standard", "encoding": "auto"},
        {"name": "income", "assigned_type": "numerical", "include": True, "scaler": "standard", "encoding": "auto"},
        {"name": "city", "assigned_type": "categorical", "include": True, "scaler": "standard", "encoding": "onehot"},
        {"name": "loan_status", "assigned_type": "categorical", "include": True, "scaler": "standard", "encoding": "onehot"},
    ]
    config = {
        "numerical_imputer": "median",
        "numerical_scaler": "standard",
        "outlier_method": "none",
        "onehot_max_cardinality": 8,
        "embedding_dim": 4,
        "categorical_missing_token": "__MISSING__",
        "group_rare_categories": False,
        "rare_category_min_frequency": 0.01,
        "seed": 5,
    }
    preprocessor = TabularPreprocessor(config).fit(frame, roles)
    matrix = preprocessor.transform(frame)
    model = build_model(matrix.shape[1], {"hidden_dim": 32, "layers": 2, "dropout": 0.0, "time_embedding_dim": 16, "steps": 8})
    generator = SyntheticGenerator(
        model,
        preprocessor,
        {"sampler": "probability_flow", "sampling_steps": 8, "guidance_scale": 1.0},
        train_frame=frame,
        train_matrix=matrix,
    )
    result = generator.generate(12, steps=8, seed=5, condition_column="loan_status", condition_value="Approved")
    assert list(result["frame"].columns) == ["age", "income", "city", "loan_status"]
    assert len(result["frame"]) == 12
    assert set(result["frame"]["loan_status"].unique()) == {"Approved"}
    assert result["frame"]["age"].between(20, 60).all()
    assert result["frame"]["income"].between(20000, 80000).all()
    assert result["duration_seconds"] >= 0
