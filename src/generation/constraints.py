"""Optional constraints, applied only when the user enables them.

Clipping is never the default. Each rule records how many rows it changed
or marked for rejection.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import yaml

from src.utils.exceptions import TabDiffuseError
from src.utils.metrics import to_jsonable


def parse_constraints(payload: str | dict | list | None) -> list[dict[str, Any]]:
    """Parse a YAML or JSON constraint document."""
    if payload is None or payload == "" or payload == []:
        return []
    if isinstance(payload, list):
        document = payload
    elif isinstance(payload, dict):
        document = payload.get("constraints", payload)
        if isinstance(document, dict):
            raise TabDiffuseError("Constraints must be a list of rules.")
    elif isinstance(payload, str):
        text = payload.strip()
        if not text:
            return []
        try:
            loaded = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise TabDiffuseError("The constraint file is not valid YAML or JSON.") from exc
        if loaded is None:
            return []
        if isinstance(loaded, dict):
            document = loaded.get("constraints")
        else:
            document = loaded
    else:
        raise TabDiffuseError("Constraints must be YAML text or a list of rules.")
    if not isinstance(document, list):
        raise TabDiffuseError("The constraint document must contain a list under 'constraints'.")
    rules = []
    for index, rule in enumerate(document, start=1):
        if not isinstance(rule, dict):
            raise TabDiffuseError(f"Constraint {index} is not a mapping.")
        kind = str(rule.get("type", "")).lower()
        action = str(rule.get("action", "reject")).lower()
        if action not in {"reject", "clip"}:
            raise TabDiffuseError(f"Constraint {index} action must be 'reject' or 'clip'.")
        if kind not in {"range", "minimum", "maximum", "date_order", "not_null"}:
            raise TabDiffuseError(
                f"Constraint {index} has unsupported type '{kind}'. "
                "Use range, minimum, maximum, date_order, or not_null."
            )
        if kind == "date_order":
            if "start" not in rule or "end" not in rule:
                raise TabDiffuseError(f"Constraint {index} needs 'start' and 'end' columns.")
        elif "column" not in rule:
            raise TabDiffuseError(f"Constraint {index} needs a 'column'.")
        rules.append({**rule, "type": kind, "action": action})
    return rules


def apply_constraints(df: pd.DataFrame, rules: list[dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply enabled rules and report modifications and rejections."""
    if not rules:
        return df.copy(), {"modified_rows": 0, "rejected_rows": 0, "rules": [], "enabled": False}
    working = df.copy()
    rejected = pd.Series(False, index=working.index)
    modified = pd.Series(False, index=working.index)
    details = []
    for rule in rules:
        detail = _apply_rule(working, rule, rejected, modified)
        details.append(detail)
    kept = working.loc[~rejected].copy()
    report = {
        "enabled": True,
        "modified_rows": int(modified.sum()),
        "rejected_rows": int(rejected.sum()),
        "input_rows": int(len(df)),
        "output_rows": int(len(kept)),
        "rules": details,
    }
    if kept.empty:
        raise TabDiffuseError(
            "Every generated row was rejected by the constraints. Relax the rules or generate more rows."
        )
    return kept, to_jsonable(report)


def _apply_rule(
    df: pd.DataFrame,
    rule: dict[str, Any],
    rejected: pd.Series,
    modified: pd.Series,
) -> dict[str, Any]:
    kind = rule["type"]
    action = rule["action"]
    if kind == "date_order":
        return _date_order(df, rule, rejected, modified)
    column = str(rule["column"])
    if column not in df.columns:
        raise TabDiffuseError(f"Constraint refers to missing column '{column}'.")
    if kind == "not_null":
        mask = df[column].isna()
        return _commit(df, column, mask, None, action, rejected, modified, kind)
    numeric = pd.to_numeric(df[column], errors="coerce")
    if kind == "minimum":
        if "value" not in rule:
            raise TabDiffuseError(f"Minimum constraint on '{column}' needs a value.")
        limit = float(rule["value"])
        mask = numeric.notna() & (numeric < limit)
        fixed = numeric.clip(lower=limit)
    elif kind == "maximum":
        if "value" not in rule:
            raise TabDiffuseError(f"Maximum constraint on '{column}' needs a value.")
        limit = float(rule["value"])
        mask = numeric.notna() & (numeric > limit)
        fixed = numeric.clip(upper=limit)
    else:
        if "min" not in rule or "max" not in rule:
            raise TabDiffuseError(f"Range constraint on '{column}' needs min and max.")
        low, high = float(rule["min"]), float(rule["max"])
        if low > high:
            raise TabDiffuseError(f"Range constraint on '{column}' has min greater than max.")
        mask = numeric.notna() & ((numeric < low) | (numeric > high))
        fixed = numeric.clip(lower=low, upper=high)
    return _commit(df, column, mask, fixed, action, rejected, modified, kind)


def _commit(
    df: pd.DataFrame,
    column: str,
    mask: pd.Series,
    fixed: pd.Series | None,
    action: str,
    rejected: pd.Series,
    modified: pd.Series,
    kind: str,
) -> dict[str, Any]:
    active = mask.fillna(False) & ~rejected
    count = int(active.sum())
    if action == "reject":
        rejected.loc[active] = True
        changed = 0
        dropped = count
    else:
        if fixed is None:
            raise TabDiffuseError(f"Column '{column}' cannot be clipped for constraint type '{kind}'. Use reject.")
        df.loc[active, column] = fixed.loc[active]
        modified.loc[active] = True
        changed = count
        dropped = 0
    return {"column": column, "type": kind, "action": action, "modified": changed, "rejected": dropped}


def _date_order(
    df: pd.DataFrame,
    rule: dict[str, Any],
    rejected: pd.Series,
    modified: pd.Series,
) -> dict[str, Any]:
    start, end = str(rule["start"]), str(rule["end"])
    for column in (start, end):
        if column not in df.columns:
            raise TabDiffuseError(f"Date constraint refers to missing column '{column}'.")
    start_values = pd.to_datetime(df[start], errors="coerce")
    end_values = pd.to_datetime(df[end], errors="coerce")
    mask = start_values.notna() & end_values.notna() & (start_values > end_values)
    action = rule["action"]
    if action == "clip":
        active = mask & ~rejected
        df.loc[active, start] = end_values.loc[active].dt.strftime("%Y-%m-%d")
        modified.loc[active] = True
        return {
            "column": f"{start}<={end}",
            "type": "date_order",
            "action": "clip",
            "modified": int(active.sum()),
            "rejected": 0,
            "note": "Start dates later than end dates were set equal to the end date.",
        }
    rejected.loc[mask & ~rejected] = True
    return {
        "column": f"{start}<={end}",
        "type": "date_order",
        "action": "reject",
        "modified": 0,
        "rejected": int(mask.sum()),
    }
