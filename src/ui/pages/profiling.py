"""Profiling page."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from src.ui.components import card
from src.ui.state import go
from src.visualization.dashboard import apply_layout


def render(state: dict[str, Any]) -> None:
    st.title("Data Profiling")
    frame = state.get("frame")
    profile = state.get("profile")
    if frame is None or profile is None:
        st.info("Upload a dataset before profiling.")
        return
    columns = st.columns(4)
    values = [
        ("Rows", f"{profile['rows']:,}", ""),
        ("Columns", str(profile["columns"]), ""),
        ("Missing values", f"{profile['missing_rate']:.1%}", f"{profile['missing_cells']} cells"),
        ("Duplicates", str(profile["duplicate_rows"]), f"{profile['duplicate_rate']:.1%} of rows"),
    ]
    for column, item in zip(columns, values):
        with column:
            card(*item)
    if profile.get("warnings"):
        st.warning("Warnings are listed and not applied automatically.")
        for warning in profile["warnings"]:
            st.write("•", warning)
    tabs = st.tabs(["Overview", "Columns", "Missing values", "Distributions", "Outliers", "Cardinality"])
    column_frame = pd.DataFrame(profile["column_profiles"])
    with tabs[0]:
        st.write(
            {
                "Numeric features": profile["numeric_features"],
                "Categorical features": profile["categorical_features"],
                "Suggested target": profile.get("suggested_target") or "None suggested",
                "Memory bytes": profile["memory_bytes"],
            }
        )
        st.caption("The suggested target is a heuristic. Confirm it before utility evaluation.")
    with tabs[1]:
        st.dataframe(column_frame, use_container_width=True, hide_index=True)
    with tabs[2]:
        figure = apply_layout(px.bar(column_frame, x="name", y="missing_pct", title="Missing fraction by column"))
        st.plotly_chart(figure, use_container_width=True)
    with tabs[3]:
        name = st.selectbox("Column", frame.columns, key="profile_dist")
        if pd.api.types.is_numeric_dtype(frame[name]):
            figure = apply_layout(px.histogram(frame, x=name, nbins=30, title=str(name)))
        else:
            counts = frame[name].astype("string").fillna("<NA>").value_counts().head(20).reset_index()
            counts.columns = ["value", "count"]
            figure = apply_layout(px.bar(counts, x="value", y="count", title=str(name)))
        st.plotly_chart(figure, use_container_width=True)
    with tabs[4]:
        numeric = column_frame[column_frame["detected_type"] == "numerical"]
        if numeric.empty:
            st.info("No numerical columns were detected.")
        else:
            shown = numeric.copy()
            shown["outlier_pct"] = shown["outlier_pct"].fillna(0)
            figure = apply_layout(px.bar(shown, x="name", y="outlier_pct", title="IQR outlier fraction"))
            st.plotly_chart(figure, use_container_width=True)
    with tabs[5]:
        figure = apply_layout(px.bar(column_frame, x="name", y="unique_count", title="Unique values"))
        st.plotly_chart(figure, use_container_width=True)
    if st.button("Continue to preprocessing"):
        go("Preprocessing")
