"""Categorical representations for mixed-type diffusion.

Low-cardinality columns are one-hot encoded. High-cardinality columns use a
fixed spherical embedding and are decoded by nearest prototype. Both are
standardized using training moments. Integer category codes are not treated
as numeric magnitudes, because those codes do not carry a meaningful distance.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from src.utils.exceptions import TabDiffuseError


def embedding_seed(column: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{column}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


class CategoricalTransform:
    """Encode one categorical column and invert the encoding after sampling."""

    def __init__(self) -> None:
        self.fitted = False

    def fit(
        self,
        series: pd.Series,
        *,
        encoding: str = "auto",
        max_onehot: int = 32,
        embedding_dim: int = 8,
        missing_token: str = "__MISSING__",
        seed: int = 42,
        group_rare: bool = False,
        rare_min_frequency: float = 0.01,
    ) -> "CategoricalTransform":
        self.name = str(series.name)
        self.missing_token = missing_token
        text = series.astype("string")
        missing = text.isna() | text.str.strip().eq("")
        self.missing_filled = int(missing.sum())
        filled = text.mask(missing, missing_token).astype(str)
        self.rare_map: dict[str, str] = {}
        counts = filled.value_counts()
        if group_rare and rare_min_frequency > 0 and len(counts) > 2:
            for category, count in counts.items():
                if str(category) == missing_token:
                    continue
                if count / len(filled) < rare_min_frequency:
                    self.rare_map[str(category)] = "__OTHER__"
            if self.rare_map:
                filled = filled.map(lambda value: self.rare_map.get(value, value))
                counts = filled.value_counts()
        self.categories = [str(category) for category in counts.index.tolist()]
        if not self.categories:
            raise TabDiffuseError(f"Column '{self.name}' has no categorical values.")
        self.index = {category: position for position, category in enumerate(self.categories)}
        chosen = encoding
        if chosen == "auto":
            chosen = "onehot" if len(self.categories) <= max_onehot else "embedding"
        if chosen not in {"onehot", "embedding"}:
            raise TabDiffuseError("Categorical encoding must be auto, onehot, or embedding.")
        self.encoding = chosen
        if chosen == "onehot":
            raw = self._one_hot(filled.to_numpy())
            self.embeddings = None
            self.output_dim = len(self.categories)
        else:
            generator = np.random.default_rng(embedding_seed(self.name, seed))
            matrix = generator.normal(size=(len(self.categories), embedding_dim)).astype(np.float64)
            matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8
            self.embeddings = matrix
            raw = self.embeddings[self._codes(filled.to_numpy())]
            self.output_dim = embedding_dim
        self.center = raw.mean(axis=0)
        self.scale = raw.std(axis=0)
        self.scale = np.where(self.scale < 1e-8, 1.0, self.scale)
        self.fitted = True
        return self

    def transform(self, series: pd.Series) -> tuple[np.ndarray, int]:
        self._check()
        text = series.astype("string")
        missing = text.isna() | text.str.strip().eq("")
        filled = text.mask(missing, self.missing_token).astype(str)
        filled = filled.map(lambda value: self.rare_map.get(value, value)).to_numpy()
        codes, unknown = self._codes(filled, count_unknown=True)
        raw = self._one_hot_from_codes(codes) if self.encoding == "onehot" else self.embeddings[codes]
        scaled = (raw - self.center) / self.scale
        return scaled.astype(np.float64), unknown

    def inverse(self, values: np.ndarray) -> list:
        self._check()
        raw = np.asarray(values, dtype=np.float64) * self.scale + self.center
        if self.encoding == "onehot":
            indices = np.argmax(raw, axis=1)
        else:
            norms = np.linalg.norm(raw, axis=1, keepdims=True) + 1e-8
            table = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8)
            indices = np.argmax((raw / norms) @ table.T, axis=1)
        labels = []
        for index in indices:
            label = self.categories[int(index)]
            labels.append(np.nan if label == self.missing_token else label)
        return labels

    def _codes(self, values: np.ndarray, count_unknown: bool = False) -> np.ndarray | tuple[np.ndarray, int]:
        fallback = self.index.get(self.missing_token, 0)
        codes = np.empty(len(values), dtype=np.int64)
        unknown = 0
        for row, value in enumerate(values):
            key = str(value)
            if key in self.index:
                codes[row] = self.index[key]
            else:
                unknown += 1
                codes[row] = fallback
        if count_unknown:
            return codes, unknown
        return codes

    def _one_hot(self, values: np.ndarray) -> np.ndarray:
        return self._one_hot_from_codes(self._codes(values))

    def _one_hot_from_codes(self, codes: np.ndarray) -> np.ndarray:
        matrix = np.zeros((len(codes), len(self.categories)), dtype=np.float64)
        matrix[np.arange(len(codes)), codes] = 1.0
        return matrix

    def _check(self) -> None:
        if not self.fitted:
            raise TabDiffuseError(f"Categorical transform for '{getattr(self, 'name', '?')}' is not fitted.")
