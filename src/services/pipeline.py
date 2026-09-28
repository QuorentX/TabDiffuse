"""Orchestration used by the dashboard, the demo script, and tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.data.profiler import profile_dataframe
from src.data.splitter import split_dataframe
from src.data.validator import assert_valid
from src.diffusion.model import build_model
from src.diffusion.trainer import create_trainer, resolve_device
from src.evaluation.baselines import generate_baselines
from src.evaluation.evaluator import evaluate_quality
from src.evaluation.privacy import privacy_report
from src.evaluation.utility import evaluate_utility
from src.generation.constraints import parse_constraints
from src.generation.generator import SyntheticGenerator
from src.preprocessing.pipeline import fit_splits
from src.services.experiments import ExperimentStore
from src.services.tracking import log_run
from src.utils.exceptions import TabDiffuseError
from src.utils.file_utils import experiment_paths
from src.utils.logger import log_event
from src.utils.seed import set_seed


def profile_dataset(frame: pd.DataFrame, thresholds: dict[str, Any]) -> dict[str, Any]:
    return profile_dataframe(frame, thresholds)


def prepare_training_data(
    frame: pd.DataFrame,
    roles: list[dict[str, Any]],
    config: Any,
    *,
    seed: int,
    holdout_ratio: float,
    validation_ratio: float,
    stratify_column: str | None,
    min_rows: int,
) -> dict[str, Any]:
    assert_valid(frame, min_rows=min_rows)
    splits = split_dataframe(
        frame,
        holdout_ratio=holdout_ratio,
        validation_ratio=validation_ratio,
        seed=seed,
        stratify_column=stratify_column,
    )
    prepared = fit_splits(splits, roles, config, seed)
    return {"splits": splits, **prepared}


def inverse_frequency_weights(labels: pd.Series) -> np.ndarray:
    text = labels.astype("string").fillna("<NA>")
    frequencies = text.value_counts(normalize=True)
    weights = text.map(lambda value: 1.0 / max(float(frequencies.get(value, 1.0)), 1e-6)).astype(float)
    return weights.to_numpy()


def train_model(
    train_x: np.ndarray,
    val_x: np.ndarray,
    model_config: dict[str, Any],
    *,
    root: Path,
    experiment_id: str,
    preprocessor: Any,
    sample_weights: np.ndarray | None = None,
    stop_event: Any = None,
    pause_event: Any = None,
    on_epoch: Any = None,
    mlflow_enabled: bool = False,
    mlflow_uri: str = "",
) -> dict[str, Any]:
    """Train, save the best and latest checkpoints, and return the training summary."""
    set_seed(int(model_config.get("seed", 42)))
    cond_indices = list(model_config.get("condition_indices") or [])
    trainer = create_trainer(train_x.shape[1], model_config, cond_dim=len(cond_indices))
    paths = experiment_paths(root, experiment_id)
    joblib.dump(preprocessor, paths["checkpoint"] / "preprocessor.joblib")
    metadata = preprocessor.metadata()
    metadata_path = paths["metadata"] / "preprocessing.json"
    from src.utils.file_utils import write_json

    write_json(metadata_path, metadata)
    extra = {
        "experiment_id": experiment_id,
        "seed": int(model_config.get("seed", 42)),
        "schema": metadata,
        "preprocessing_path": str(paths["checkpoint"] / "preprocessor.joblib"),
    }
    summary = trainer.fit(
        train_x,
        val_x,
        sample_weights=sample_weights,
        stop_event=stop_event,
        pause_event=pause_event,
        on_epoch=on_epoch,
        checkpoint_dir=paths["checkpoint"],
        extra_metadata=extra,
    )
    trainer.save(paths["checkpoint"] / "checkpoint_latest.pt", extra)
    if trainer.best_state is not None:
        trainer.save(paths["checkpoint"] / "checkpoint_best.pt", extra, best=True)
    summary["checkpoint"] = str(paths["checkpoint"] / "checkpoint_best.pt")
    summary["latest_checkpoint"] = str(paths["checkpoint"] / "checkpoint_latest.pt")
    summary["preprocessor_path"] = str(paths["checkpoint"] / "preprocessor.joblib")
    metrics = {}
    if summary.get("best_val_loss") is not None:
        metrics["best_val_loss"] = float(summary["best_val_loss"])
    log_run(
        mlflow_enabled,
        mlflow_uri,
        experiment_id,
        {key: model_config.get(key) for key in ("epochs", "batch_size", "learning_rate", "hidden_dim", "layers", "steps", "seed")},
        metrics,
    )
    log_event("pipeline", "model_trained", status=summary["state"], duration=summary.get("duration_seconds"), experiment_id=experiment_id)
    return summary


def resume_model(
    train_x: np.ndarray,
    val_x: np.ndarray,
    model_config: dict[str, Any],
    *,
    root: Path,
    experiment_id: str,
    sample_weights: np.ndarray | None = None,
    stop_event: Any = None,
    pause_event: Any = None,
    on_epoch: Any = None,
) -> dict[str, Any]:
    """Continue an existing experiment from its latest checkpoint."""
    paths = experiment_paths(root, experiment_id)
    latest = paths["checkpoint"] / "checkpoint_latest.pt"
    if not latest.exists():
        raise TabDiffuseError("No latest checkpoint exists for this experiment yet.")
    cond_indices = list(model_config.get("condition_indices") or [])
    trainer = create_trainer(train_x.shape[1], model_config, cond_dim=len(cond_indices))
    trainer.load(latest)
    target_epochs = int(model_config.get("epochs", trainer.epoch))
    if trainer.epoch >= target_epochs:
        raise TabDiffuseError(
            "The checkpoint is already at the configured epoch count. Increase epochs to continue training."
        )
    extra = {"experiment_id": experiment_id, "seed": int(model_config.get("seed", 42))}
    summary = trainer.fit(
        train_x,
        val_x,
        sample_weights=sample_weights,
        stop_event=stop_event,
        pause_event=pause_event,
        on_epoch=on_epoch,
        checkpoint_dir=paths["checkpoint"],
        extra_metadata=extra,
    )
    trainer.save(paths["checkpoint"] / "checkpoint_latest.pt", extra)
    if trainer.best_state is not None:
        trainer.save(paths["checkpoint"] / "checkpoint_best.pt", extra, best=True)
    summary["checkpoint"] = str(paths["checkpoint"] / "checkpoint_best.pt")
    return summary


def load_bundle(checkpoint: Path, device: str = "cpu") -> dict[str, Any]:
    """Load a score model and the preprocessor saved beside it."""
    preprocessor_path = checkpoint.parent / "preprocessor.joblib"
    if not preprocessor_path.exists():
        raise TabDiffuseError("The checkpoint is missing its preprocessing metadata.")
    preprocessor = joblib.load(preprocessor_path)
    resolved = resolve_device(device)
    try:
        import torch

        payload = torch.load(checkpoint, map_location=resolved, weights_only=False)
    except Exception as exc:
        raise TabDiffuseError("The checkpoint could not be read. Choose another saved experiment.") from exc
    config = payload.get("config") or {}
    model = build_model(int(payload["feature_dim"]), config, cond_dim=int(payload.get("cond_dim") or 0))
    model.load_state_dict(payload["model_state"])
    model.to(resolved)
    model.eval()
    return {"model": model, "preprocessor": preprocessor, "config": config, "payload": payload}


def generate_synthetic(
    bundle: dict[str, Any],
    train_frame: pd.DataFrame,
    train_matrix: np.ndarray,
    *,
    n_samples: int,
    steps: int,
    temperature: float,
    seed: int,
    condition_column: str | None = None,
    condition_value: Any = None,
    balance_column: str | None = None,
    constraint_text: str | None = None,
    on_progress: Any = None,
) -> dict[str, Any]:
    rules = parse_constraints(constraint_text) if constraint_text else []
    generator = SyntheticGenerator(
        bundle["model"],
        bundle["preprocessor"],
        bundle["config"],
        train_frame=train_frame,
        train_matrix=train_matrix,
    )
    return generator.generate(
        n_samples,
        steps=steps,
        temperature=temperature,
        seed=seed,
        condition_column=condition_column,
        condition_value=condition_value,
        balance_column=balance_column,
        constraints=rules,
        on_progress=on_progress,
    )


def run_quality(real: pd.DataFrame, synthetic: pd.DataFrame) -> dict[str, Any]:
    return evaluate_quality(real, synthetic)


def run_privacy(real: pd.DataFrame, synthetic: pd.DataFrame, **kwargs: Any) -> dict[str, Any]:
    return privacy_report(real, synthetic, **kwargs)


def run_utility(real_train: pd.DataFrame, holdout: pd.DataFrame, synthetic: pd.DataFrame, target: str, **kwargs: Any) -> dict[str, Any]:
    return evaluate_utility(real_train, holdout, synthetic, target, **kwargs)


def run_baseline_comparison(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    *,
    n_samples: int,
    seed: int,
    target: str | None,
    flags: dict[str, Any],
) -> list[dict[str, Any]]:
    generated = generate_baselines(
        train,
        n_samples,
        seed,
        resample=bool(flags.get("enable_resample", True)),
        gaussian=bool(flags.get("enable_gaussian", True)),
        copula=bool(flags.get("enable_copula", True)),
        ctgan=bool(flags.get("enable_ctgan", False)),
        ctgan_epochs=int(flags.get("ctgan_epochs", 30)),
    )
    rows = []
    for name, item in generated.items():
        if not item.get("available", False):
            rows.append({"method": name, "available": False, "message": item.get("message", ""), "duration_seconds": item.get("duration_seconds")})
            continue
        frame = item["frame"]
        quality = evaluate_quality(train, frame)
        privacy = privacy_report(train, frame, max_rows=min(1500, len(train)), seed=seed)
        utility_score = None
        if target and target in frame.columns and len(holdout):
            utility = evaluate_utility(
                train, holdout, frame, target, seed=seed, max_rows=min(2000, len(train)), n_estimators=20
            )
            utility_score = _primary_utility(utility)
        rows.append(
            {
                "method": name,
                "available": True,
                "duration_seconds": item.get("duration_seconds"),
                "distribution_similarity": (quality.get("similarity") or {}).get("distribution_similarity"),
                "correlation_similarity": (quality.get("similarity") or {}).get("correlation_similarity"),
                "categorical_similarity": (quality.get("similarity") or {}).get("categorical_similarity"),
                "exact_match_rate": (privacy.get("exact_match") or {}).get("exact_match_rate"),
                "distance_ratio": (privacy.get("nearest_neighbor") or {}).get("distance_ratio"),
                "utility_score": utility_score,
            }
        )
    return rows


def _primary_utility(utility: dict[str, Any]) -> float | None:
    for row in utility.get("rows") or []:
        if row.get("training_data") == "Synthetic" and row.get("model") == "Logistic Regression":
            return row.get("f1")
        if row.get("training_data") == "Synthetic" and row.get("model") == "Linear Regression":
            return row.get("r2")
    for row in utility.get("rows") or []:
        if row.get("training_data") == "Synthetic":
            return row.get("f1") if row.get("f1") is not None else row.get("r2")
    return None


def open_store(root: Path) -> ExperimentStore:
    return ExperimentStore(root)
