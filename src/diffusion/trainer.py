"""Training loop for denoising score matching."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from src.diffusion.loss import denoising_score_matching_loss
from src.diffusion.model import ScoreBasedDiffusion, build_model
from src.utils.exceptions import TabDiffuseError
from src.utils.logger import log_event, log_exception
from src.utils.seed import set_seed


def resolve_device(requested: str) -> torch.device:
    """Resolve Auto, CPU, or CUDA into a torch device."""
    name = str(requested or "auto").lower()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise TabDiffuseError(
                "CUDA was requested, but no compatible GPU is available. Switch the device to CPU or Auto."
            )
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    raise TabDiffuseError("Device must be Auto, CPU, or CUDA.")


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def estimate_memory_mb(feature_dim: int, hidden_dim: int, layers: int, batch_size: int) -> dict[str, float]:
    """Rough training-memory estimate used by the configuration screen."""
    params = feature_dim * hidden_dim + hidden_dim * feature_dim
    params += layers * (hidden_dim * hidden_dim * 2 + hidden_dim * 4)
    params += hidden_dim * 128 * 3
    model_mb = params * 4 / (1024**2)
    adam_mb = model_mb * 2
    activation_mb = batch_size * hidden_dim * max(layers, 1) * 4 * 2 / (1024**2)
    return {
        "parameters": float(params),
        "estimated_training_mb": float(model_mb + adam_mb + activation_mb),
    }


class DiffusionTrainer:
    """Minimize the denoising score matching loss and write checkpoints."""

    def __init__(self, model: ScoreBasedDiffusion, config: dict[str, Any], device: torch.device) -> None:
        self.model = model.to(device)
        self.config = dict(config)
        self.device = device
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(config.get("learning_rate", 1e-4)),
            weight_decay=float(config.get("weight_decay", 1e-4)),
        )
        self.history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        self.best_val = float("inf")
        self.best_state: dict[str, torch.Tensor] | None = None
        self.epoch = 0
        self.condition_indices = list(config.get("condition_indices") or [])

    def fit(
        self,
        train_x: np.ndarray,
        val_x: np.ndarray,
        *,
        sample_weights: np.ndarray | None = None,
        stop_event: threading.Event | None = None,
        pause_event: threading.Event | None = None,
        on_epoch: Callable[[dict[str, Any]], None] | None = None,
        checkpoint_dir: Path | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Train until the configured epoch, an early stop, or a stop request."""
        set_seed(int(self.config.get("seed", 42)))
        epochs = int(self.config["epochs"])
        batch_size = int(self.config.get("batch_size", 256))
        if len(train_x) < 8:
            raise TabDiffuseError("Too few training rows remain after splitting. Provide a larger dataset.")
        if len(val_x) < 2:
            raise TabDiffuseError("The validation split is empty. Increase the validation ratio or the dataset size.")

        train_loader = self._loader(train_x, batch_size, sample_weights, shuffle=True)
        val_tensor = torch.tensor(val_x, dtype=torch.float32, device=self.device)
        start_epoch = self.epoch
        started = time.perf_counter()
        stale = 0
        patience = int(self.config.get("early_stopping_patience", 0))
        state = "completed"

        try:
            for epoch in range(start_epoch, epochs):
                self._wait_if_paused(pause_event, stop_event)
                if stop_event is not None and stop_event.is_set():
                    state = "stopped"
                    break
                epoch_started = time.perf_counter()
                train_loss = self._train_epoch(train_loader, stop_event)
                if train_loss is None:
                    state = "stopped"
                    break
                val_loss = self._validate(val_tensor)
                self.epoch = epoch + 1
                self.history["train_loss"].append(float(train_loss))
                self.history["val_loss"].append(float(val_loss))
                if val_loss < self.best_val:
                    self.best_val = float(val_loss)
                    self.best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
                    stale = 0
                else:
                    stale += 1
                elapsed = time.perf_counter() - started
                done = self.epoch - start_epoch
                remaining = (elapsed / max(done, 1)) * max(epochs - self.epoch, 0)
                snapshot = {
                    "epoch": self.epoch,
                    "epochs": epochs,
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "best_val_loss": None if self.best_val == float("inf") else self.best_val,
                    "epoch_seconds": time.perf_counter() - epoch_started,
                    "elapsed_seconds": elapsed,
                    "remaining_seconds": remaining,
                    "history": self.history,
                    "state": "running",
                }
                if on_epoch is not None:
                    on_epoch(snapshot)
                if checkpoint_dir is not None:
                    self.save(checkpoint_dir / "checkpoint_latest.pt", extra_metadata)
                    if self.best_state is not None and val_loss <= self.best_val:
                        self.save(checkpoint_dir / "checkpoint_best.pt", extra_metadata, best=True)
                if patience and stale >= patience:
                    state = "early_stopped"
                    break
        except RuntimeError as exc:
            log_exception("trainer", "training_failed", exc)
            if "out of memory" in str(exc).lower():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise TabDiffuseError(
                    "GPU memory was insufficient for the selected model configuration. "
                    "Try reducing the batch size or the hidden dimension."
                ) from exc
            raise TabDiffuseError(
                "Model training failed. Try a smaller learning rate, fewer layers, or the demo configuration. "
                "Technical details were written to the log."
            ) from exc

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        duration = time.perf_counter() - started
        log_event("trainer", "training_finished", status=state, duration=duration)
        return {
            "state": state,
            "epoch": self.epoch,
            "epochs": epochs,
            "best_val_loss": None if self.best_val == float("inf") else self.best_val,
            "history": self.history,
            "duration_seconds": duration,
            "device": str(self.device),
            "parameters": parameter_count(self.model),
        }

    def save(self, path: Path, extra_metadata: dict[str, Any] | None = None, best: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        state = self.best_state if best and self.best_state is not None else self.model.state_dict()
        payload = {
            "model_state": state,
            "optimizer_state": self.optimizer.state_dict(),
            "epoch": self.epoch,
            "config": self.config,
            "history": self.history,
            "best_val_loss": None if self.best_val == float("inf") else self.best_val,
            "condition_indices": self.condition_indices,
            "feature_dim": self.model.feature_dim,
            "cond_dim": self.model.cond_dim,
            "extra": extra_metadata or {},
        }
        torch.save(payload, path)

    def load(self, path: Path) -> dict[str, Any]:
        try:
            payload = torch.load(path, map_location=self.device, weights_only=False)
        except Exception as exc:
            log_exception("trainer", "checkpoint_load_failed", exc)
            raise TabDiffuseError(
                "The checkpoint could not be read. Choose another experiment checkpoint."
            ) from exc
        self.model.load_state_dict(payload["model_state"])
        try:
            self.optimizer.load_state_dict(payload["optimizer_state"])
        except (ValueError, RuntimeError):
            pass
        self.epoch = int(payload.get("epoch", 0))
        self.history = payload.get("history", {"train_loss": [], "val_loss": []})
        best = payload.get("best_val_loss")
        self.best_val = float("inf") if best is None else float(best)
        self.condition_indices = list(payload.get("condition_indices") or [])
        self.best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
        return payload

    def _train_epoch(self, loader: DataLoader, stop_event: threading.Event | None) -> float | None:
        self.model.train()
        total = 0.0
        seen = 0
        steps = int(self.config.get("steps", 200))
        mode = str(self.config.get("time_sampling", "discrete"))
        dropout = float(self.config.get("condition_dropout", 0.15)) if self.condition_indices else 0.0
        for batch in loader:
            if stop_event is not None and stop_event.is_set():
                return None if seen == 0 else total / seen
            x0 = batch[0].to(self.device)
            cond = self._condition_batch(x0, dropout)
            t = self.model.sample_times(x0.shape[0], steps, mode, self.device)
            x_t, noise, _, std = self.model.perturb(x0, t)
            score = self.model(x_t, t, cond)
            loss = denoising_score_matching_loss(score, noise, std)
            if not torch.isfinite(loss):
                raise TabDiffuseError(
                    "Training loss became non-finite. Lower the learning rate or use standard scaling."
                )
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config.get("grad_clip", 1.0)))
            self.optimizer.step()
            total += float(loss.item()) * len(x0)
            seen += len(x0)
        return total / max(seen, 1)

    @torch.no_grad()
    def _validate(self, val_x: torch.Tensor) -> float:
        self.model.eval()
        steps = int(self.config.get("steps", 200))
        mode = str(self.config.get("time_sampling", "discrete"))
        batch_size = int(self.config.get("batch_size", 256))
        total = 0.0
        seen = 0
        for start in range(0, len(val_x), batch_size):
            x0 = val_x[start : start + batch_size]
            cond = self._condition_batch(x0, dropout=0.0)
            t = self.model.sample_times(x0.shape[0], steps, mode, self.device)
            x_t, noise, _, std = self.model.perturb(x0, t)
            score = self.model(x_t, t, cond)
            loss = denoising_score_matching_loss(score, noise, std)
            total += float(loss.item()) * len(x0)
            seen += len(x0)
        self.model.train()
        return total / max(seen, 1)

    def _condition_batch(self, x0: torch.Tensor, dropout: float) -> torch.Tensor | None:
        if not self.condition_indices or self.model.cond_dim == 0:
            return None
        cond = x0[:, self.condition_indices]
        if dropout > 0 and self.model.training:
            drop = torch.rand(x0.shape[0], device=x0.device) < dropout
            cond = cond.clone()
            cond[drop] = 0
        return cond

    def _loader(
        self,
        train_x: np.ndarray,
        batch_size: int,
        sample_weights: np.ndarray | None,
        shuffle: bool,
    ) -> DataLoader:
        tensor = torch.tensor(np.asarray(train_x), dtype=torch.float32)
        dataset = TensorDataset(tensor)
        sampler = None
        if sample_weights is not None:
            sampler = WeightedRandomSampler(
                weights=torch.tensor(sample_weights, dtype=torch.double),
                num_samples=len(sample_weights),
                replacement=True,
            )
            shuffle = False
        return DataLoader(
            dataset,
            batch_size=min(batch_size, len(dataset)),
            shuffle=shuffle and sampler is None,
            sampler=sampler,
            drop_last=False,
        )

    @staticmethod
    def _wait_if_paused(pause_event: threading.Event | None, stop_event: threading.Event | None) -> None:
        while pause_event is not None and pause_event.is_set():
            if stop_event is not None and stop_event.is_set():
                return
            time.sleep(0.2)


def create_trainer(feature_dim: int, config: dict[str, Any], cond_dim: int = 0) -> DiffusionTrainer:
    device = resolve_device(str(config.get("device", "auto")))
    model = build_model(feature_dim, config, cond_dim=cond_dim)
    config = dict(config)
    config["condition_indices"] = list(config.get("condition_indices") or [])
    if cond_dim and len(config["condition_indices"]) != cond_dim:
        raise TabDiffuseError("Condition indices do not match the condition vector width.")
    return DiffusionTrainer(model, config, device)
