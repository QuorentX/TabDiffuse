"""Optional MLflow logging. Training continues if MLflow is disabled or missing."""

from __future__ import annotations

from typing import Any

from src.utils.logger import log_event


def log_run(enabled: bool, tracking_uri: str, experiment_id: str, params: dict[str, Any], metrics: dict[str, float]) -> dict[str, Any]:
    if not enabled:
        return {"enabled": False}
    try:
        import mlflow
    except ImportError:
        return {"enabled": False, "message": "MLflow is not installed."}
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("tabdiffuse")
        with mlflow.start_run(run_name=experiment_id):
            mlflow.log_params({key: value for key, value in params.items() if _is_param(value)})
            mlflow.log_metrics({key: float(value) for key, value in metrics.items() if _is_metric(value)})
        return {"enabled": True}
    except Exception as exc:
        log_event("mlflow", "log_failed", status="error", error=type(exc).__name__, experiment_id=experiment_id)
        return {"enabled": False, "message": "MLflow logging failed. The local experiment record was still saved."}


def _is_param(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool))


def _is_metric(value: Any) -> bool:
    return isinstance(value, (int, float)) and value == value
