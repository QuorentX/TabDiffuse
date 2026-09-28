"""Structured logging that records events, not dataset values."""

from __future__ import annotations

import json
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config.settings import project_root

_LOGGER_NAME = "tabdiffuse"
_CONFIGURED = False


def _log_dir() -> Path:
    path = project_root() / "artifacts" / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_logger() -> logging.Logger:
    """Return the process-wide TabDiffuse logger."""
    global _CONFIGURED
    logger = logging.getLogger(_LOGGER_NAME)
    if _CONFIGURED:
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(_log_dir() / "tabdiffuse.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    stream = logging.StreamHandler()
    stream.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stream)
    _CONFIGURED = True
    return logger


def log_event(
    component: str,
    event: str,
    status: str = "ok",
    duration: float | None = None,
    experiment_id: str | None = None,
    error: str | None = None,
    level: str = "info",
) -> None:
    """Write one structured log line.

    Callers must pass summaries only. Do not pass column values or row contents.
    """
    payload: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "component": component,
        "event": event,
        "status": status,
        "experiment_id": experiment_id,
        "duration_sec": None if duration is None else round(float(duration), 4),
        "error": None if error is None else str(error)[:500],
    }
    line = json.dumps(payload, ensure_ascii=True)
    logger = get_logger()
    getattr(logger, level if level in {"info", "warning", "error"} else "info")(line)


def log_exception(component: str, event: str, exc: BaseException, experiment_id: str | None = None) -> None:
    """Log an exception type and a short message without a data dump."""
    message = f"{type(exc).__name__}: {exc}"
    log_event(component, event, status="error", experiment_id=experiment_id, error=message, level="error")
    logger = get_logger()
    logger.error(json.dumps({"traceback": traceback.format_exc()[-2000:]}))
