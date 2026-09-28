"""File helpers, experiment identifiers, and local artifact cleanup."""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.exceptions import TabDiffuseError

_EXP_RE = re.compile(r"^EXP-(\d{4})-(\d{4})$")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def next_experiment_id(root: Path, year: int | None = None) -> str:
    """Allocate EXP-YYYY-NNNN without reusing an existing identifier."""
    year = year or datetime.now().year
    highest = 0
    search_roots = [
        root / "artifacts" / "experiments",
        root / "models" / "checkpoints",
        root / "models" / "metadata",
    ]
    for folder in search_roots:
        if not folder.exists():
            continue
        for child in folder.iterdir():
            match = _EXP_RE.match(child.name)
            if match and int(match.group(1)) == year:
                highest = max(highest, int(match.group(2)))
    return f"EXP-{year}-{highest + 1:04d}"


def experiment_paths(root: Path, experiment_id: str) -> dict[str, Path]:
    if not _EXP_RE.match(experiment_id):
        raise TabDiffuseError(f"Experiment id '{experiment_id}' is not in the form EXP-YYYY-NNNN.")
    return {
        "experiment": ensure_dir(root / "artifacts" / "experiments" / experiment_id),
        "checkpoint": ensure_dir(root / "models" / "checkpoints" / experiment_id),
        "metadata": ensure_dir(root / "models" / "metadata" / experiment_id),
        "figures": ensure_dir(root / "artifacts" / "figures" / experiment_id),
        "reports": ensure_dir(root / "data" / "reports" / experiment_id),
        "synthetic": ensure_dir(root / "data" / "synthetic" / experiment_id),
    }


def assert_within(path: Path, parent: Path) -> Path:
    resolved = path.resolve()
    parent_resolved = parent.resolve()
    if parent_resolved != resolved and parent_resolved not in resolved.parents:
        raise TabDiffuseError("Refusing to touch a path outside the project artifact directories.")
    return resolved


def delete_directory(path: Path, root: Path) -> None:
    """Delete a directory only when it lives under the project root."""
    resolved = assert_within(path, root)
    if resolved == root.resolve():
        raise TabDiffuseError("Refusing to delete the project root.")
    if resolved.exists():
        shutil.rmtree(resolved)


def humanize_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {sec}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"
