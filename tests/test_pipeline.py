"""Small end-to-end pipeline test."""

from __future__ import annotations

from src.config.settings import load_config
from src.data.sample_data import generate_sample_dataset
from src.services.experiments import ExperimentStore
from src.services.pipeline import (
    generate_synthetic,
    load_bundle,
    prepare_training_data,
    profile_dataset,
    run_quality,
    train_model,
)


def test_pipeline_trains_generates_and_scores(tmp_path):
    config = load_config()
    frame = generate_sample_dataset(120, seed=8)
    profile = profile_dataset(
        frame,
        {
            "id_uniqueness_threshold": 0.98,
            "high_cardinality_threshold": 50,
            "missing_warning_threshold": 0.3,
            "duplicate_warning_threshold": 0.5,
            "imbalance_warning_threshold": 0.95,
        },
    )
    model_config = config.model_training_dict()
    model_config.update(
        {
            "epochs": 2,
            "batch_size": 32,
            "hidden_dim": 32,
            "layers": 2,
            "steps": 8,
            "sampling_steps": 8,
            "device": "cpu",
            "sampler": "probability_flow",
            "time_embedding_dim": 16,
            "learning_rate": 1e-3,
        }
    )
    prepared = prepare_training_data(
        frame,
        profile["roles"],
        config,
        seed=8,
        holdout_ratio=0.15,
        validation_ratio=0.15,
        stratify_column="loan_status",
        min_rows=32,
    )
    store = ExperimentStore(tmp_path)
    record = store.create({"dataset_name": "unit.csv", "rows": len(frame), "columns": frame.shape[1], "seed": 8, "model_config": model_config})
    summary = train_model(
        prepared["arrays"]["train"],
        prepared["arrays"]["validation"],
        model_config,
        root=tmp_path,
        experiment_id=record["experiment_id"],
        preprocessor=prepared["preprocessor"],
    )
    assert summary["history"]["train_loss"]
    assert summary["best_val_loss"] is not None
    bundle = load_bundle(tmp_path / "models" / "checkpoints" / record["experiment_id"] / "checkpoint_best.pt", "cpu")
    generated = generate_synthetic(
        bundle,
        prepared["splits"]["train"],
        prepared["arrays"]["train"],
        n_samples=20,
        steps=8,
        temperature=1.0,
        seed=8,
    )
    quality = run_quality(prepared["splits"]["train"], generated["frame"])
    assert quality["similarity"]["distribution_similarity"] is not None
    stored = store.load(record["experiment_id"])
    assert stored["experiment_id"].startswith("EXP-")
    second = store.create({"dataset_name": "other.csv", "rows": 1, "columns": 1, "seed": 1})
    assert second["experiment_id"] != record["experiment_id"]
