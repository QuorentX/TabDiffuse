"""Home page."""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.ui.components import banner, card, workflow
from src.ui.state import go


def render(state: dict[str, Any]) -> None:
    st.title("TABDIFFUSE")
    st.subheader("Score-Based Diffusion for Privacy-Aware Synthetic Tabular Data")
    if state.get("demo_mode"):
        banner("Demo Mode – Reduced training configuration")
    frame = state.get("frame")
    prepared = state.get("prepared")
    synthetic = state.get("synthetic")
    quality = state.get("quality")
    columns = st.columns(4)
    with columns[0]:
        card("Dataset Status", "Loaded" if frame is not None else "Not loaded", state.get("dataset_name") or "Upload a table to begin")
    with columns[1]:
        card(
            "Model Status",
            "Trained" if state.get("training_summary") else "Not trained",
            state.get("experiment_id") or "No experiment yet",
        )
    with columns[2]:
        rows = 0 if synthetic is None else len(synthetic)
        card("Synthetic Records", f"{rows:,}", "Generated rows currently in memory")
    with columns[3]:
        card("Evaluation Status", "Available" if quality else "Not run", "Quality, privacy, and utility stay separate")
    st.markdown("#### Workflow")
    workflow()
    st.markdown(
        """
TabDiffuse learns a score function for a variance-preserving diffusion process on a mixed-type table.
It then draws new rows, restores the original schema, and measures statistical similarity, empirical privacy risk,
and downstream machine-learning utility. Synthetic data is not treated as automatically private.
        """
    )
    left, right = st.columns([1.4, 1])
    with left:
        st.markdown("#### Current experiment")
        experiment = state.get("experiment") or {}
        st.write(
            {
                "Experiment ID": state.get("experiment_id") or "—",
                "Dataset": state.get("dataset_name") or "—",
                "Model": "Variance-preserving score network",
                "Status": (experiment or {}).get("status") or ("trained" if state.get("training_summary") else "not started"),
            }
        )
        st.caption("Load a table from Dataset Upload, then return here or continue from that page.")
        if frame is not None and st.button("Continue to profiling"):
            go("Data Profiling")
    with right:
        st.markdown("#### Experiment notes")
        notes = state["notes"]
        notes["research_question"] = st.text_area("Research question", notes.get("research_question", ""))
        notes["hypothesis"] = st.text_area("Hypothesis", notes.get("hypothesis", ""))
        notes["dataset_description"] = st.text_area("Dataset description", notes.get("dataset_description", ""))
        notes["observations"] = st.text_area("Observations", notes.get("observations", ""))
        notes["conclusion"] = st.text_area("Conclusion", notes.get("conclusion", ""))
        st.caption("Notes are stored with the experiment record when training starts or when a report is saved.")
