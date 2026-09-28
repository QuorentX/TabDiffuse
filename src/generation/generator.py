"""Draw synthetic rows from a trained score-based model."""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

from src.diffusion.model import ScoreBasedDiffusion
from src.diffusion.sampler import sample_latent
from src.generation.postprocessor import postprocess_samples
from src.preprocessing.transformer import TabularPreprocessor
from src.utils.exceptions import TabDiffuseError
from src.utils.seed import set_seed


class SyntheticGenerator:
    """Sample the reverse process in batches and restore the original schema."""

    def __init__(
        self,
        model: ScoreBasedDiffusion,
        preprocessor: TabularPreprocessor,
        config: dict[str, Any],
        train_frame: pd.DataFrame | None = None,
        train_matrix: np.ndarray | None = None,
    ) -> None:
        self.model = model
        self.preprocessor = preprocessor
        self.config = config
        self.train_frame = train_frame
        self.train_matrix = train_matrix

    def generate(
        self,
        n_samples: int,
        *,
        steps: int | None = None,
        temperature: float | None = None,
        seed: int = 42,
        condition_column: str | None = None,
        condition_value: Any = None,
        balance_column: str | None = None,
        constraints: list[dict[str, Any]] | None = None,
        batch_size: int | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        """Generate ``n_samples`` rows. Progress counts completed latent rows."""
        if n_samples < 1:
            raise TabDiffuseError("Choose at least one synthetic record.")
        set_seed(seed)
        steps = int(steps or self.config.get("sampling_steps") or self.config.get("steps") or 50)
        temperature = float(self.config.get("temperature", 1.0) if temperature is None else temperature)
        batch_size = int(batch_size or self.config.get("generation_batch_size") or 512)
        started = time.perf_counter()
        batches = self._plan(n_samples, condition_column, condition_value, balance_column, seed)
        self._attach_feature_bounds()
        matrices = []
        done = 0
        total = sum(count for count, _, _ in batches)
        for count, indices, values in batches:
            remaining = count
            while remaining:
                current = min(batch_size, remaining)
                cond_embed = None
                cond_values = None
                if indices and values is not None:
                    value_row = torch.tensor(values, dtype=torch.float32).view(1, -1)
                    cond_values = value_row
                    trained = list(self.config.get("condition_indices") or [])
                    if self.model.cond_dim and indices == trained and value_row.shape[1] == self.model.cond_dim:
                        cond_embed = value_row
                latent = sample_latent(
                    self.model,
                    current,
                    steps=steps,
                    sampler=str(self.config.get("sampler", "reverse_sde")),
                    temperature=temperature,
                    condition=cond_embed,
                    condition_indices=indices,
                    condition_values=cond_values,
                    guidance_scale=float(self.config.get("guidance_scale", 1.0)),
                )
                matrices.append(latent.cpu().numpy())
                remaining -= current
                done += current
                if on_progress is not None:
                    on_progress(done, total)
        stacked = np.concatenate(matrices, axis=0)[:n_samples]
        if balance_column or (condition_column and condition_value is not None):
            stacked = stacked[:n_samples]
        processed = postprocess_samples(stacked, self.preprocessor, constraints)
        duration = time.perf_counter() - started
        frame = processed["frame"]
        if len(frame) > n_samples and not constraints:
            frame = frame.iloc[:n_samples].copy()
        return {
            "frame": frame,
            "rows": int(len(frame)),
            "requested": int(n_samples),
            "duration_seconds": duration,
            "rows_per_second": float(len(frame) / duration) if duration > 0 else None,
            "steps": steps,
            "temperature": temperature,
            "seed": seed,
            "sampler": self.config.get("sampler", "reverse_sde"),
            "condition_column": condition_column,
            "condition_value": None if condition_value is None else str(condition_value),
            "balance_column": balance_column,
            "conditioning_method": self._method(condition_column, balance_column),
            "inverse": processed["inverse"],
            "constraints": processed["constraints"],
            "invalid_numeric_values": processed["invalid_numeric_values"],
        }

    def _attach_feature_bounds(self) -> None:
        """Keep decoded samples inside the numeric range the training rows occupied."""
        matrix = self.train_matrix
        if matrix is None or getattr(matrix, "ndim", 0) != 2 or matrix.shape[1] != self.model.feature_dim:
            self.model.feature_low = None
            self.model.feature_high = None
            return
        finite = np.nan_to_num(np.asarray(matrix, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        self.model.feature_low = finite.min(axis=0)
        self.model.feature_high = finite.max(axis=0)

    def _plan(
        self,
        n_samples: int,
        condition_column: str | None,
        condition_value: Any,
        balance_column: str | None,
        seed: int,
    ) -> list[tuple[int, list[int] | None, np.ndarray | None]]:
        column = balance_column or (condition_column if condition_value is not None else None)
        if column is None:
            return [(n_samples, None, None)]
        if column not in self.preprocessor.output_columns:
            raise TabDiffuseError(f"Column '{column}' was excluded from the model, so it cannot be a condition.")
        if self.train_frame is None:
            raise TabDiffuseError("Conditional generation needs the training table that was used to fit the model.")
        indices = self.preprocessor.column_indices(column)
        if balance_column:
            values = [value for value in self.train_frame[column].dropna().unique().tolist()]
            if len(values) < 2:
                raise TabDiffuseError(f"Column '{column}' does not have multiple classes to balance.")
            base, extra = divmod(n_samples, len(values))
            plan = []
            for position, value in enumerate(values):
                count = base + (1 if position < extra else 0)
                if count:
                    plan.append((count, indices, self._encode(column, value)))
            return plan
        return [(n_samples, indices, self._encode(column, condition_value))]

    def _encode(self, column: str, value: Any) -> np.ndarray:
        if self.train_frame is None or column not in self.train_frame.columns:
            raise TabDiffuseError(f"Cannot encode a condition for '{column}'.")
        observed = self.train_frame[column].dropna()
        matches = observed.astype(str) == str(value)
        if not matches.any():
            raise TabDiffuseError(
                f"The condition value for '{column}' was not observed in the training split. "
                "Choose a value that appears in the training data."
            )
        template = self.train_frame.loc[[matches[matches].index[0]]].copy()
        encoded = self.preprocessor.transform(template)
        indices = self.preprocessor.column_indices(column)
        return encoded[0, indices]

    @staticmethod
    def _method(condition_column: str | None, balance_column: str | None) -> str:
        if balance_column or condition_column:
            return (
                "Coordinates of the selected column are re-injected during reverse diffusion "
                "and set to the encoded training value at the final step. If the model was trained "
                "with that column as a condition input, classifier-free guidance is also applied."
            )
        return "Unconditional reverse diffusion."
