"""Byte exports for downloads."""

from __future__ import annotations

import io
import json
from typing import Any

import pandas as pd

from src.utils.metrics import to_jsonable


def dataframe_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def dataframe_parquet_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    return buffer.getvalue()


def json_bytes(payload: Any) -> bytes:
    return json.dumps(to_jsonable(payload), indent=2).encode("utf-8")
