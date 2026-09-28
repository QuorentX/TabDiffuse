"""Column roles and inferred schema records."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

ASSIGNED_TYPES = ("numerical", "categorical", "boolean", "datetime", "id", "constant", "drop")
SCALERS = ("standard", "robust", "minmax", "none")
ENCODINGS = ("auto", "onehot", "embedding")


class ColumnRole(BaseModel):
    """User-editable decision about how a column enters the model."""

    name: str
    detected_type: str
    assigned_type: str = "numerical"
    include: bool = True
    scaler: str = "standard"
    encoding: str = "auto"
    notes: str = ""

    def model_treatment(self) -> str:
        if not self.include or self.assigned_type in {"id", "constant", "drop"}:
            return "Excluded"
        if self.assigned_type == "datetime":
            return "Date parts"
        if self.assigned_type in {"categorical", "boolean"}:
            return "Embedding" if self.encoding == "embedding" else "One-hot"
        scaler_names = {
            "standard": "Standardized",
            "robust": "Robust scaled",
            "minmax": "Min-max scaled",
            "none": "Unscaled",
        }
        return scaler_names.get(self.scaler, "Continuous")


class SchemaSummary(BaseModel):
    columns: list[ColumnRole] = Field(default_factory=list)
    suggested_target: str | None = None
    warnings: list[str] = Field(default_factory=list)

    def as_records(self) -> list[dict[str, Any]]:
        return [role.model_dump() for role in self.columns]
