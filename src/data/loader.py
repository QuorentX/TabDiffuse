"""Load CSV, Excel, and Parquet tables from a path or an in-memory upload."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import pandas as pd

from src.utils.exceptions import TabDiffuseError
from src.utils.logger import log_event, log_exception

SUPPORTED_SUFFIXES = {".csv", ".tsv", ".xlsx", ".xls", ".parquet"}


def load_tabular(source: str | Path | BinaryIO, filename: str | None = None) -> pd.DataFrame:
    """Read a tabular file into a DataFrame.

    Parser errors are not forwarded to the caller, because some parsers echo
    cell contents. The log stores the exception type only.
    """
    if hasattr(source, "seek"):
        source.seek(0)
    name = filename or getattr(source, "name", None) or str(source)
    suffix = Path(str(name)).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise TabDiffuseError(
            f"Unsupported file type '{suffix or 'unknown'}'. Upload a CSV, Excel, or Parquet file."
        )
    try:
        if suffix == ".csv":
            frame = pd.read_csv(source)
        elif suffix == ".tsv":
            frame = pd.read_csv(source, sep="\t")
        elif suffix in {".xlsx", ".xls"}:
            frame = pd.read_excel(source)
        else:
            frame = pd.read_parquet(source)
    except TabDiffuseError:
        raise
    except Exception as exc:
        log_exception("loader", "read_failed", exc)
        raise TabDiffuseError(
            "The file could not be read. Check that it is a valid CSV, Excel, or Parquet table "
            "and that the first row contains column names."
        ) from exc

    if not isinstance(frame, pd.DataFrame) or frame.shape[1] == 0:
        raise TabDiffuseError("The uploaded file has no columns.")
    if len(frame) == 0:
        raise TabDiffuseError("The uploaded file has column names but no data rows.")

    frame = frame.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    if any(column == "" for column in frame.columns):
        raise TabDiffuseError("The file contains an empty column name. Name every column and upload again.")
    duplicates = frame.columns[frame.columns.duplicated()].unique().tolist()
    if duplicates:
        raise TabDiffuseError(
            "The file has duplicate column names ("
            + ", ".join(duplicates[:8])
            + "). Rename them and upload again."
        )
    log_event("loader", "dataset_loaded", status="ok")
    return frame
