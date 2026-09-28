"""Run a small end-to-end demonstration without the Streamlit interface.

Usage:
    python -m src.demo_pipeline
"""

from __future__ import annotations

from src.config.settings import load_config, project_root
from src.data.loader import load_tabular
from src.data.sample_data import write_sample_dataset
from src.services.experiments import ExperimentStore
from src.services.pipeline import (
    generate_synthetic,
    load_bundle,
    prepare_training_data,
    profile_dataset,
    run_privacy,
    run_quality,
    run_utility,
    train_model,
)
from src.utils.logger import log_event


def main() -> None:
    config = load_config()
    root = project_root()
    sample_path = root / config.data.sample_path
    if not sample_path.exists():
        write_sample_dataset(sample_path, n_rows=1200, seed=config.project.seed)
    frame = load_tabular(sample_path).head(config.demo.max_rows).reset_index(drop=True)
    profile = profile_dataset(
        frame,
        {
            "id_uniqueness_threshold": config.data.id_uniqueness_threshold,
            "high_cardinality_threshold": config.data.high_cardinality_threshold,
            "missing_warning_threshold": config.data.missing_warning_threshold,
            "duplicate_warning_threshold": config.data.duplicate_warning_threshold,
            "imbalance_warning_threshold": config.data.imbalance_warning_threshold,
        },
    )
    model_config = config.model_training_dict()
    model_config.update(
        {
            "epochs": config.demo.epochs,
            "batch_size": config.demo.batch_size,
            "hidden_dim": config.demo.hidden_dim,
            "layers": config.demo.layers,
            "steps": config.demo.steps,
            "sampling_steps": config.demo.sampling_steps,
            "device": "cpu",
            "demo_mode": True,
        }
    )
    prepared = prepare_training_data(
        frame,
        profile["roles"],
        config,
        seed=config.project.seed,
        holdout_ratio=config.split.holdout_ratio,
        validation_ratio=config.split.validation_ratio,
        stratify_column=profile.get("suggested_target"),
        min_rows=config.data.min_rows,
    )
    store = ExperimentStore(root)
    record = store.create(
        {
            "dataset_name": "sample_customer_data.csv",
            "rows": int(len(frame)),
            "columns": int(frame.shape[1]),
            "seed": config.project.seed,
            "status": "training",
            "demo_mode": True,
            "model_config": model_config,
            "preprocessing": prepared["metadata"],
            "profile_warnings": profile.get("warnings", [])[:20],
            "notes": {
                "research_question": "Can a small score-based model capture this demonstration table?",
                "hypothesis": "Short training should produce finite loss and restorable synthetic rows.",
                "dataset_description": "Bundled synthetic customer demonstration data, not a real extract.",
                "observations": "",
                "conclusion": "",
            },
        }
    )
    summary = train_model(
        prepared["arrays"]["train"],
        prepared["arrays"]["validation"],
        model_config,
        root=root,
        experiment_id=record["experiment_id"],
        preprocessor=prepared["preprocessor"],
    )
    bundle = load_bundle(root / "models" / "checkpoints" / record["experiment_id"] / "checkpoint_best.pt", "cpu")
    generated = generate_synthetic(
        bundle,
        prepared["splits"]["train"],
        prepared["arrays"]["train"],
        n_samples=min(200, config.demo.samples),
        steps=config.demo.sampling_steps,
        temperature=1.0,
        seed=config.project.seed,
    )
    quality = run_quality(prepared["splits"]["train"], generated["frame"])
    synthetic_matrix = None
    try:
        synthetic_matrix = bundle["preprocessor"].transform(generated["frame"])
    except Exception:
        synthetic_matrix = None
    privacy = run_privacy(
        prepared["splits"]["train"],
        generated["frame"],
        train_matrix=prepared["arrays"]["train"],
        holdout_matrix=prepared["arrays"]["holdout"],
        synthetic_matrix=synthetic_matrix,
        thresholds=config.privacy_thresholds.model_dump(),
        max_rows=800,
        seed=config.project.seed,
    )
    target = profile.get("suggested_target")
    utility = None
    if target:
        utility = run_utility(
            prepared["splits"]["train"],
            prepared["splits"]["holdout"],
            generated["frame"],
            target,
            seed=config.project.seed,
            n_estimators=15,
            max_rows=800,
        )
    store.update(
        record["experiment_id"],
        {
            "status": "evaluated",
            "training": summary,
            "generation": {key: value for key, value in generated.items() if key != "frame"},
            "evaluation": quality,
            "privacy": privacy,
            "utility": utility,
        },
    )
    log_event("demo", "completed", experiment_id=record["experiment_id"], status="ok")
    print(f"Demo experiment {record['experiment_id']} finished.")
    print(f"Best validation loss: {summary.get('best_val_loss')}")
    print(f"Synthetic rows: {generated['rows']}")


if __name__ == "__main__":
    main()
