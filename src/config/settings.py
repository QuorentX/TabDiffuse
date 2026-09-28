"""Configuration loaded from config.yaml and optional environment variables."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


def project_root() -> Path:
    """Return the repository root (directory that contains config.yaml)."""
    return Path(__file__).resolve().parents[2]


def load_env_file(path: Path | None = None) -> None:
    """Load KEY=VALUE pairs from .env without overriding existing variables."""
    env_path = path or (project_root() / ".env")
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class ProjectSection(BaseModel):
    name: str = "TabDiffuse"
    seed: int = 42
    log_level: str = "INFO"


class DataSection(BaseModel):
    max_rows_warning: int = 200_000
    min_rows: int = 32
    sample_path: str = "data/raw/sample_customer_data.csv"
    onehot_max_cardinality: int = 32
    high_cardinality_threshold: int = 50
    id_uniqueness_threshold: float = 0.98
    missing_warning_threshold: float = 0.30
    embedding_dim: int = 8
    duplicate_warning_threshold: float = 0.05
    imbalance_warning_threshold: float = 0.85


class SplitSection(BaseModel):
    holdout_ratio: float = 0.15
    validation_ratio: float = 0.15


class PreprocessingSection(BaseModel):
    numerical_imputer: str = "median"
    numerical_scaler: str = "standard"
    outlier_method: str = "none"
    iqr_factor: float = 1.5
    categorical_missing_token: str = "__MISSING__"
    rare_category_min_frequency: float = 0.01
    group_rare_categories: bool = False


class DiffusionSection(BaseModel):
    sde: str = "vp"
    beta_min: float = 0.1
    beta_max: float = 20.0
    schedule: str = "linear"
    steps: int = 200
    t_epsilon: float = 1e-3
    sampler: str = "reverse_sde"
    time_sampling: str = "discrete"


class ModelSection(BaseModel):
    hidden_dim: int = 256
    layers: int = 4
    dropout: float = 0.1
    time_embedding_dim: int = 128


class TrainingSection(BaseModel):
    epochs: int = 80
    batch_size: int = 256
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    seed: int = 42
    device: str = "auto"
    grad_clip: float = 1.0
    condition_dropout: float = 0.15
    guidance_scale: float = 1.0
    balance_batches: bool = False
    early_stopping_patience: int = 0


class GenerationSection(BaseModel):
    samples: int = 1000
    sampling_steps: int = 200
    temperature: float = 1.0
    seed: int = 42
    batch_size: int = 512


class EvaluationSection(BaseModel):
    privacy: bool = True
    utility: bool = True
    quality: bool = True
    max_rows_for_privacy: int = 4000
    max_rows_for_utility: int = 20000
    tsne_sample: int = 800
    pca_sample: int = 4000
    utility_estimators: int = 60


class PrivacyThresholds(BaseModel):
    exact_match_low: float = 0.005
    exact_match_moderate: float = 0.02
    nn_ratio_low: float = 0.75
    nn_ratio_moderate: float = 0.45
    membership_advantage_low: float = 0.05
    membership_advantage_moderate: float = 0.12
    disclosure_lift_low: float = 1.1
    disclosure_lift_moderate: float = 1.4


class DemoSection(BaseModel):
    epochs: int = 8
    batch_size: int = 64
    hidden_dim: int = 128
    layers: int = 2
    steps: int = 40
    sampling_steps: int = 40
    max_rows: int = 600
    samples: int = 400


class BaselineSection(BaseModel):
    enable_resample: bool = True
    enable_gaussian: bool = True
    enable_copula: bool = True
    enable_ctgan: bool = False
    ctgan_epochs: int = 30
    baseline_samples: int = 1000


class MLflowSection(BaseModel):
    enabled: bool = False
    tracking_uri: str = "artifacts/mlruns"


class AppConfig(BaseModel):
    """Full application configuration."""

    project: ProjectSection = Field(default_factory=ProjectSection)
    data: DataSection = Field(default_factory=DataSection)
    split: SplitSection = Field(default_factory=SplitSection)
    preprocessing: PreprocessingSection = Field(default_factory=PreprocessingSection)
    diffusion: DiffusionSection = Field(default_factory=DiffusionSection)
    model: ModelSection = Field(default_factory=ModelSection)
    training: TrainingSection = Field(default_factory=TrainingSection)
    generation: GenerationSection = Field(default_factory=GenerationSection)
    evaluation: EvaluationSection = Field(default_factory=EvaluationSection)
    privacy_thresholds: PrivacyThresholds = Field(default_factory=PrivacyThresholds)
    demo: DemoSection = Field(default_factory=DemoSection)
    baselines: BaselineSection = Field(default_factory=BaselineSection)
    mlflow: MLflowSection = Field(default_factory=MLflowSection)

    def model_training_dict(self) -> dict[str, Any]:
        """Flatten the settings the trainer and sampler share."""
        return {
            "sde": self.diffusion.sde,
            "schedule": self.diffusion.schedule,
            "steps": self.diffusion.steps,
            "sampling_steps": self.generation.sampling_steps,
            "beta_min": self.diffusion.beta_min,
            "beta_max": self.diffusion.beta_max,
            "t_epsilon": self.diffusion.t_epsilon,
            "sampler": self.diffusion.sampler,
            "time_sampling": self.diffusion.time_sampling,
            "hidden_dim": self.model.hidden_dim,
            "layers": self.model.layers,
            "dropout": self.model.dropout,
            "time_embedding_dim": self.model.time_embedding_dim,
            "epochs": self.training.epochs,
            "batch_size": self.training.batch_size,
            "learning_rate": self.training.learning_rate,
            "weight_decay": self.training.weight_decay,
            "seed": self.training.seed,
            "device": self.training.device,
            "grad_clip": self.training.grad_clip,
            "condition_dropout": self.training.condition_dropout,
            "guidance_scale": self.training.guidance_scale,
            "balance_batches": self.training.balance_batches,
            "early_stopping_patience": self.training.early_stopping_patience,
            "temperature": self.generation.temperature,
            "samples": self.generation.samples,
            "generation_batch_size": self.generation.batch_size,
            "holdout_ratio": self.split.holdout_ratio,
            "validation_ratio": self.split.validation_ratio,
            "condition_column": None,
        }


def load_config(path: Path | None = None) -> AppConfig:
    """Load config.yaml and apply environment overrides for log level and seed."""
    load_env_file()
    config_path = path or (project_root() / "config.yaml")
    payload: dict[str, Any] = {}
    if config_path.exists():
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Configuration file {config_path} must be a mapping.")
        payload = loaded
    config = AppConfig.model_validate(payload)
    if os.environ.get("TABDIFFUSE_LOG_LEVEL"):
        config.project.log_level = os.environ["TABDIFFUSE_LOG_LEVEL"]
    if os.environ.get("TABDIFFUSE_SEED"):
        config.project.seed = int(os.environ["TABDIFFUSE_SEED"])
        config.training.seed = int(os.environ["TABDIFFUSE_SEED"])
    if os.environ.get("TABDIFFUSE_ENABLE_MLFLOW", "").lower() in {"1", "true", "yes"}:
        config.mlflow.enabled = True
    if os.environ.get("TABDIFFUSE_MLFLOW_URI"):
        config.mlflow.tracking_uri = os.environ["TABDIFFUSE_MLFLOW_URI"]
    return config


def demo_model_config(config: AppConfig | None = None) -> dict[str, Any]:
    """Return a small training configuration for interactive demonstration."""
    cfg = config or load_config()
    model = cfg.model_training_dict()
    model.update(
        {
            "epochs": cfg.demo.epochs,
            "batch_size": cfg.demo.batch_size,
            "hidden_dim": cfg.demo.hidden_dim,
            "layers": cfg.demo.layers,
            "steps": cfg.demo.steps,
            "sampling_steps": cfg.demo.sampling_steps,
            "samples": cfg.demo.samples,
            "demo_mode": True,
        }
    )
    return model
