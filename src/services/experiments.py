"""Persistent experiment records. Raw tables are not stored in the record."""

from __future__ import annotations

import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from src.utils.file_utils import experiment_paths, next_experiment_id, read_json, utc_now, write_json
from src.utils.metrics import to_jsonable

_PACKAGE_NAMES = (
    "numpy",
    "pandas",
    "torch",
    "scikit-learn",
    "scipy",
    "streamlit",
    "plotly",
    "pyarrow",
    "matplotlib",
    "pydantic",
)


def environment_fingerprint() -> dict[str, Any]:
    packages = {}
    for name in _PACKAGE_NAMES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "not installed"
    return {"python": sys.version.split()[0], "packages": packages}


class ExperimentStore:
    """Allocate experiment IDs and keep a JSON history on disk."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def create(self, fields: dict[str, Any]) -> dict[str, Any]:
        experiment_id = next_experiment_id(self.root)
        record = {
            "experiment_id": experiment_id,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "status": "created",
            **environment_fingerprint(),
        }
        record.update(to_jsonable(fields))
        record["experiment_id"] = experiment_id
        self._write(record)
        return record

    def update(self, experiment_id: str, fields: dict[str, Any]) -> dict[str, Any]:
        record = self.load(experiment_id)
        record.update(to_jsonable(fields))
        record["experiment_id"] = experiment_id
        record["updated_at"] = utc_now()
        self._write(record)
        return record

    def load(self, experiment_id: str) -> dict[str, Any]:
        path = experiment_paths(self.root, experiment_id)["experiment"] / "experiment.json"
        return read_json(path)

    def list_experiments(self) -> list[dict[str, Any]]:
        folder = self.root / "artifacts" / "experiments"
        if not folder.exists():
            return []
        records = []
        for child in sorted(folder.iterdir()):
            path = child / "experiment.json"
            if path.exists():
                records.append(read_json(path))
        return sorted(records, key=lambda item: item.get("created_at", ""), reverse=True)

    def _write(self, record: dict[str, Any]) -> None:
        paths = experiment_paths(self.root, record["experiment_id"])
        write_json(paths["experiment"] / "experiment.json", record)
        write_json(paths["metadata"] / "experiment.json", record)
