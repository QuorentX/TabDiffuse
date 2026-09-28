"""Dataset upload page."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from src.data.loader import load_tabular
from src.data.profiler import profile_dataframe
from src.ui.components import card, show_exception
from src.ui.state import go
from src.utils.metrics import format_bytes, memory_usage_bytes


def render(state: dict[str, Any]) -> None:
    st.title("Dataset Upload")
    st.caption("CSV, Excel, and Parquet files are read locally. The raw upload is kept in this session, not sent to a service.")
    uploaded = st.file_uploader("Drop a table here", type=["csv", "tsv", "xlsx", "xls", "parquet"])
    if uploaded is not None and st.button("Analyze dataset", type="primary"):
        try:
            frame = load_tabular(uploaded, filename=uploaded.name)
            config = state["config"]
            if len(frame) > config.data.max_rows_warning:
                st.warning(
                    f"This table has {len(frame):,} rows. Training is still possible, but consider a sample for the first run."
                )
            state["frame"] = frame
            state["dataset_name"] = uploaded.name
            state["file_size"] = uploaded.size
            state["demo_mode"] = False
            state["profile"] = profile_dataframe(frame, _thresholds(config))
            state["roles"] = state["profile"]["roles"]
            state["target"] = state["profile"].get("suggested_target")
            state["prepared"] = None
            state["synthetic"] = None
            state["bundle"] = None
            state["quality"] = None
            state["privacy"] = None
            state["utility"] = None
            st.success("Dataset analyzed. Review the summary, then open profiling.")
        except Exception as exc:
            show_exception(exc, "upload")

    frame: pd.DataFrame | None = state.get("frame")
    profile = state.get("profile")
    if frame is None or profile is None:
        st.info("Upload a file to see its shape, types, missing values, and a preview.")
        return
    _summary(state, frame, profile)
    st.dataframe(frame.head(20), use_container_width=True)
    if st.button("Continue to profiling"):
        go("Data Profiling")


def _summary(state: dict[str, Any], frame: pd.DataFrame, profile: dict[str, Any]) -> None:
    st.markdown(f"**{state.get('dataset_name')}**")
    columns = st.columns(4)
    values = [
        ("Rows", f"{profile['rows']:,}", ""),
        ("Columns", str(profile["columns"]), ""),
        ("File size", format_bytes(int(state.get("file_size") or 0)), ""),
        ("Memory", format_bytes(memory_usage_bytes(frame)), ""),
    ]
    for column, item in zip(columns, values):
        with column:
            card(*item)
    second = st.columns(4)
    second_values = [
        ("Numeric", str(profile["numeric_features"]), ""),
        ("Categorical", str(profile["categorical_features"]), ""),
        ("Missing", f"{profile['missing_rate']:.1%}", ""),
        ("Duplicates", f"{profile['duplicate_rate']:.1%}", f"{profile['duplicate_rows']} rows"),
    ]
    for column, item in zip(second, second_values):
        with column:
            card(*item)
    dtype_table = pd.DataFrame(profile["column_profiles"])[
        ["name", "dtype", "detected_type", "missing_pct", "unique_count"]
    ]
    st.dataframe(dtype_table, use_container_width=True, hide_index=True)


def _thresholds(config) -> dict:
    data = config.data
    return {
        "id_uniqueness_threshold": data.id_uniqueness_threshold,
        "high_cardinality_threshold": data.high_cardinality_threshold,
        "missing_warning_threshold": data.missing_warning_threshold,
        "duplicate_warning_threshold": data.duplicate_warning_threshold,
        "imbalance_warning_threshold": data.imbalance_warning_threshold,
    }
