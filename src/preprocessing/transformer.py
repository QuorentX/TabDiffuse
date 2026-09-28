"""Traceable mixed-type preprocessor with a stored inverse mapping."""

from __future__ import annotations

import calendar
from typing import Any

import numpy as np
import pandas as pd

from src.preprocessing.categorical import CategoricalTransform
from src.preprocessing.numerical import NumericTransform
from src.utils.exceptions import TabDiffuseError
from src.utils.metrics import to_jsonable


class TabularPreprocessor:
    """Convert a dataframe into the continuous representation used by the SDE.

    Every inclusion, exclusion, imputation, clip, and encoding is recorded in
    ``actions``. Synthetic samples are mapped back through ``inverse_transform``.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.groups: list[dict[str, Any]] = []
        self.actions: list[dict[str, Any]] = []
        self.excluded: list[dict[str, str]] = []
        self.output_columns: list[str] = []
        self.feature_dim = 0
        self.fitted = False
        self.seed = int(self.config.get("seed", 42))

    def fit(self, df: pd.DataFrame, roles: list[dict[str, Any]]) -> "TabularPreprocessor":
        self.groups = []
        self.actions = []
        self.excluded = []
        self.output_columns = []
        cursor = 0
        role_map = {str(role["name"]): role for role in roles}
        missing = [column for column in role_map if column not in df.columns]
        if missing:
            raise TabDiffuseError("Preprocessing roles refer to missing columns: " + ", ".join(missing[:12]))

        for name in df.columns:
            role = role_map.get(str(name))
            if role is None:
                self.excluded.append({"column": str(name), "reason": "No role was provided."})
                self.actions.append({"column": str(name), "action": "excluded", "reason": "No role was provided."})
                continue
            assigned = str(role.get("assigned_type", "numerical"))
            include = bool(role.get("include", True)) and assigned not in {"id", "constant", "drop"}
            if not include:
                reason = f"Assigned type '{assigned}' is excluded from the diffusion model."
                self.excluded.append({"column": str(name), "reason": reason})
                self.actions.append({"column": str(name), "action": "excluded", "reason": reason})
                continue
            if assigned == "datetime":
                width = self._fit_datetime(df[name], role)
            elif assigned in {"categorical", "boolean"}:
                width = self._fit_categorical(df[name], role)
            elif assigned == "numerical":
                width = self._fit_numeric(df[name], role)
            else:
                raise TabDiffuseError(
                    f"Column '{name}' has an unsupported assigned type '{assigned}'."
                )
            self.output_columns.append(str(name))
            cursor += width
        if cursor == 0:
            raise TabDiffuseError("No columns are included in training. Include at least one feature.")
        self.feature_dim = cursor
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        self._check()
        missing = [name for name in self.output_columns if name not in df.columns]
        if missing:
            raise TabDiffuseError("The dataset is missing columns required by the fitted preprocessor: " + ", ".join(missing[:12]))
        parts: list[np.ndarray] = []
        for group in self.groups:
            parts.append(self._transform_group(df, group))
        matrix = np.concatenate(parts, axis=1).astype(np.float32)
        if not np.isfinite(matrix).all():
            raise TabDiffuseError(
                "Preprocessing produced non-finite values. Check the included columns for values that cannot be scaled."
            )
        return matrix

    def inverse_transform(self, matrix: np.ndarray) -> tuple[pd.DataFrame, dict[str, Any]]:
        self._check()
        array = np.asarray(matrix, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != self.feature_dim:
            raise TabDiffuseError(
                f"Generated features have shape {getattr(array, 'shape', None)}; the preprocessor expects "
                f"{self.feature_dim} columns."
            )
        nonfinite_rows = ~np.isfinite(array).all(axis=1)
        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
        columns: dict[str, Any] = {}
        invalid_dates = 0
        repaired_dates = 0
        projected_numeric = 0
        for group in self.groups:
            block = array[:, group["start"] : group["end"]]
            if group["kind"] == "numerical":
                columns[group["name"]] = group["transform"].inverse(block[:, 0])
                projected_numeric += int(getattr(group["transform"], "last_projected", 0))
            elif group["kind"] == "categorical":
                columns[group["name"]] = group["transform"].inverse(block)
            elif group["kind"] == "datetime":
                restored, invalid, repaired, projected = self._inverse_datetime(group, block)
                columns[group["name"]] = restored
                invalid_dates += invalid
                repaired_dates += repaired
                projected_numeric += projected
        frame = pd.DataFrame(columns)
        report = {
            "rows": int(len(frame)),
            "nonfinite_rows": int(nonfinite_rows.sum()),
            "invalid_dates": int(invalid_dates),
            "repaired_dates": int(repaired_dates),
            "projected_numeric_values": int(projected_numeric),
            "columns": list(frame.columns),
        }
        return frame, report

    def column_indices(self, column: str) -> list[int]:
        self._check()
        for group in self.groups:
            if group["name"] == column:
                return list(range(group["start"], group["end"]))
        raise TabDiffuseError(
            f"Column '{column}' is not part of the model representation. It may have been excluded."
        )

    def metadata(self) -> dict[str, Any]:
        """Readable preprocessing record. Embedding matrices stay in the joblib file."""
        groups = []
        for group in self.groups:
            item = {
                "name": group["name"],
                "kind": group["kind"],
                "start": group["start"],
                "end": group["end"],
            }
            transform = group.get("transform")
            if group["kind"] == "categorical" and transform is not None:
                item.update(
                    {
                        "encoding": transform.encoding,
                        "categories": transform.categories,
                        "missing_filled": transform.missing_filled,
                        "rare_mapped": len(transform.rare_map),
                    }
                )
            if group["kind"] == "numerical" and transform is not None:
                item.update(
                    {
                        "scaler": transform.scaler,
                        "imputer": transform.imputer,
                        "outlier": transform.outlier,
                        "missing_filled": transform.missing_filled,
                        "clipped": transform.clipped,
                        "integer_like": transform.integer_like,
                    }
                )
            if group["kind"] == "datetime":
                item["parts"] = list(group["parts"].keys())
            groups.append(item)
        return to_jsonable(
            {
                "feature_dim": self.feature_dim,
                "output_columns": self.output_columns,
                "excluded": self.excluded,
                "actions": self.actions,
                "groups": groups,
                "seed": self.seed,
            }
        )

    def _fit_numeric(self, series: pd.Series, role: dict[str, Any]) -> int:
        transform = NumericTransform().fit(
            series,
            imputer=self.config.get("numerical_imputer", "median"),
            scaler=role.get("scaler") or self.config.get("numerical_scaler", "standard"),
            outlier=self.config.get("outlier_method", "none"),
            iqr_factor=float(self.config.get("iqr_factor", 1.5)),
        )
        start = self.feature_dim_so_far()
        self.groups.append(
            {"name": str(series.name), "kind": "numerical", "start": start, "end": start + 1, "transform": transform}
        )
        self.actions.append(
            {
                "column": str(series.name),
                "action": "numeric",
                "imputer": transform.imputer,
                "missing_filled": transform.missing_filled,
                "scaler": transform.scaler,
                "outlier": transform.outlier,
                "clipped": transform.clipped,
            }
        )
        return 1

    def _fit_categorical(self, series: pd.Series, role: dict[str, Any]) -> int:
        transform = CategoricalTransform().fit(
            series,
            encoding=str(role.get("encoding", "auto")),
            max_onehot=int(self.config.get("onehot_max_cardinality", 32)),
            embedding_dim=int(self.config.get("embedding_dim", 8)),
            missing_token=str(self.config.get("categorical_missing_token", "__MISSING__")),
            seed=self.seed,
            group_rare=bool(self.config.get("group_rare_categories", False)),
            rare_min_frequency=float(self.config.get("rare_category_min_frequency", 0.01)),
        )
        start = self.feature_dim_so_far()
        self.groups.append(
            {
                "name": str(series.name),
                "kind": "categorical",
                "start": start,
                "end": start + transform.output_dim,
                "transform": transform,
            }
        )
        self.actions.append(
            {
                "column": str(series.name),
                "action": transform.encoding,
                "categories": len(transform.categories),
                "missing_filled": transform.missing_filled,
                "rare_mapped": len(transform.rare_map),
                "output_dim": transform.output_dim,
            }
        )
        return transform.output_dim

    def _fit_datetime(self, series: pd.Series, role: dict[str, Any]) -> int:
        parsed = pd.to_datetime(series, errors="coerce", format="mixed")
        parts: dict[str, NumericTransform] = {}
        part_names = ["year", "month", "day", "dayofweek"]
        hours = parsed.dt.hour
        include_hour = bool(hours.dropna().nunique() > 1)
        if include_hour:
            part_names.append("hour")
        raw_parts = {
            "year": parsed.dt.year,
            "month": parsed.dt.month,
            "day": parsed.dt.day,
            "dayofweek": parsed.dt.dayofweek,
            "hour": hours,
        }
        indicator = None
        missing = parsed.isna()
        if int(missing.sum()) > 0:
            indicator = NumericTransform().fit(
                missing.astype(float),
                imputer="constant",
                scaler="none",
                outlier="none",
            )
        for part in part_names:
            part_series = raw_parts[part].astype(float)
            part_series.name = f"{series.name}__{part}"
            parts[part] = NumericTransform().fit(
                part_series,
                imputer=self.config.get("numerical_imputer", "median"),
                scaler=role.get("scaler") or "standard",
                outlier="none",
            )
        width = len(parts) + (1 if indicator is not None else 0)
        start = self.feature_dim_so_far()
        self.groups.append(
            {
                "name": str(series.name),
                "kind": "datetime",
                "start": start,
                "end": start + width,
                "parts": parts,
                "part_names": part_names,
                "indicator": indicator,
                "include_hour": include_hour,
            }
        )
        self.actions.append(
            {
                "column": str(series.name),
                "action": "datetime_parts",
                "parts": part_names + (["missing_indicator"] if indicator is not None else []),
                "missing_dates": int(missing.sum()),
                "hour_included": include_hour,
                "note": "Day of week is learned as a derived feature and is not written as its own output column.",
            }
        )
        return width

    def _transform_group(self, df: pd.DataFrame, group: dict[str, Any]) -> np.ndarray:
        name = group["name"]
        if group["kind"] == "numerical":
            return group["transform"].transform(df[name]).reshape(-1, 1)
        if group["kind"] == "categorical":
            values, unknown = group["transform"].transform(df[name])
            group["unknown_seen"] = int(unknown)
            return values
        parsed = pd.to_datetime(df[name], errors="coerce", format="mixed")
        blocks = []
        if group["indicator"] is not None:
            blocks.append(group["indicator"].transform(parsed.isna().astype(float)).reshape(-1, 1))
        for part in group["part_names"]:
            series = getattr(parsed.dt, "dayofweek" if part == "dayofweek" else part).astype(float)
            series.name = part
            blocks.append(group["parts"][part].transform(series).reshape(-1, 1))
        return np.concatenate(blocks, axis=1)

    def _inverse_datetime(self, group: dict[str, Any], block: np.ndarray) -> tuple[list, int, int, int]:
        cursor = 0
        missing_flag = None
        projected = 0
        if group["indicator"] is not None:
            missing_flag = group["indicator"].inverse(block[:, 0])
            projected += int(getattr(group["indicator"], "last_projected", 0))
            cursor = 1
        restored_parts = {}
        for part in group["part_names"]:
            restored_parts[part] = group["parts"][part].inverse(block[:, cursor])
            projected += int(getattr(group["parts"][part], "last_projected", 0))
            cursor += 1
        output: list[Any] = []
        invalid = 0
        repaired = 0
        for row in range(block.shape[0]):
            if missing_flag is not None and missing_flag[row] >= 0.5:
                output.append(pd.NA)
                continue
            try:
                year = int(np.rint(restored_parts["year"][row]))
                month = int(np.rint(restored_parts["month"][row]))
                day = int(np.rint(restored_parts["day"][row]))
                hour = int(np.rint(restored_parts["hour"][row])) if group["include_hour"] else 0
                original = (year, month, day, hour)
                year = int(np.clip(year, 1, 9999))
                month = int(np.clip(month, 1, 12))
                day_limit = calendar.monthrange(year, month)[1]
                day = int(np.clip(day, 1, day_limit))
                hour = int(np.clip(hour, 0, 23))
                if (year, month, day, hour) != original:
                    repaired += 1
                if group["include_hour"]:
                    output.append(f"{year:04d}-{month:02d}-{day:02d} {hour:02d}:00:00")
                else:
                    output.append(f"{year:04d}-{month:02d}-{day:02d}")
            except (ValueError, OverflowError, OSError):
                invalid += 1
                output.append(pd.NA)
        return output, invalid, repaired, projected

    def feature_dim_so_far(self) -> int:
        if not self.groups:
            return 0
        return int(self.groups[-1]["end"])

    def _check(self) -> None:
        if not self.fitted:
            raise TabDiffuseError("The preprocessor has not been fit.")


def preprocessor_config_from_app(config: Any, seed: int) -> dict[str, Any]:
    """Build the preprocessor config from the application settings object or a dict."""
    if isinstance(config, dict):
        data = config
        return {
            "numerical_imputer": data.get("numerical_imputer", "median"),
            "numerical_scaler": data.get("numerical_scaler", "standard"),
            "outlier_method": data.get("outlier_method", "none"),
            "iqr_factor": data.get("iqr_factor", 1.5),
            "onehot_max_cardinality": data.get("onehot_max_cardinality", 32),
            "embedding_dim": data.get("embedding_dim", 8),
            "categorical_missing_token": data.get("categorical_missing_token", "__MISSING__"),
            "group_rare_categories": data.get("group_rare_categories", False),
            "rare_category_min_frequency": data.get("rare_category_min_frequency", 0.01),
            "seed": seed,
        }
    return {
        "numerical_imputer": config.preprocessing.numerical_imputer,
        "numerical_scaler": config.preprocessing.numerical_scaler,
        "outlier_method": config.preprocessing.outlier_method,
        "iqr_factor": config.preprocessing.iqr_factor,
        "onehot_max_cardinality": config.data.onehot_max_cardinality,
        "embedding_dim": config.data.embedding_dim,
        "categorical_missing_token": config.preprocessing.categorical_missing_token,
        "group_rare_categories": config.preprocessing.group_rare_categories,
        "rare_category_min_frequency": config.preprocessing.rare_category_min_frequency,
        "seed": seed,
    }
