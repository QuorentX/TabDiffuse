"""Background training job for the local Streamlit process."""

from __future__ import annotations

import copy
import threading
from typing import Any, Callable

from src.utils.exceptions import TabDiffuseError
from src.utils.logger import log_exception

_JOBS: dict[str, "TrainingJob"] = {}


class TrainingJob:
    """One training run with pause and stop events checked between epochs."""

    def __init__(self) -> None:
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.status: dict[str, Any] = {"state": "idle", "published": True}

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return copy.deepcopy(self.status)

    def update(self, **fields: Any) -> None:
        with self.lock:
            self.status.update(fields)

    def start(self, worker: Callable[["TrainingJob"], dict[str, Any]]) -> None:
        if self.thread is not None and self.thread.is_alive():
            raise TabDiffuseError("Training is already running in this session.")
        self.stop_event.clear()
        self.pause_event.clear()
        with self.lock:
            self.status = {
                "state": "running",
                "epoch": 0,
                "epochs": 0,
                "train_loss": None,
                "val_loss": None,
                "history": {"train_loss": [], "val_loss": []},
                "message": "Training started.",
                "error": None,
                "published": False,
                "paused": False,
            }
        self.thread = threading.Thread(target=self._run, args=(worker,), daemon=True)
        self.thread.start()

    def _run(self, worker: Callable[["TrainingJob"], dict[str, Any]]) -> None:
        try:
            result = worker(self)
            self.update(state=result.get("state", "completed"), result=result, message="Training finished.", paused=False)
        except Exception as exc:
            log_exception("training_job", "failed", exc)
            message = exc.message if isinstance(exc, TabDiffuseError) else (
                "Model training failed. Try a smaller model or the demo configuration. "
                "Technical details were written to the log."
            )
            self.update(state="error", error=message, message=message, paused=False)

    def request_stop(self) -> None:
        self.stop_event.set()
        self.pause_event.clear()
        self.update(message="Stop requested. The current epoch will finish or abort between batches.")

    def request_pause(self) -> None:
        self.pause_event.set()
        self.update(paused=True, message="Pause requested. Training pauses at the next epoch boundary.")

    def request_resume(self) -> None:
        self.pause_event.clear()
        self.update(paused=False, message="Training resumed.")


def get_job(session_key: str) -> TrainingJob:
    if session_key not in _JOBS:
        _JOBS[session_key] = TrainingJob()
    return _JOBS[session_key]
