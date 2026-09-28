"""Privacy diagnostics page."""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.config.settings import project_root
from src.evaluation.privacy import DISCLAIMER
from src.reporting.export import json_bytes
from src.services.experiments import ExperimentStore
from src.services.pipeline import run_privacy
from src.ui.components import card, disclaimer, fmt, show_exception
from src.visualization.privacy_plots import distance_histogram, train_holdout_histogram


def render(state: dict[str, Any]) -> None:
    st.title("Privacy Risk Diagnostics")
    disclaimer("Privacy metrics shown here are empirical diagnostics. They should not be interpreted as a formal privacy guarantee.")
    prepared = state.get("prepared")
    synthetic = state.get("synthetic")
    if prepared is None or synthetic is None:
        st.info("Generate synthetic data before running privacy diagnostics.")
        return
    columns = [column for column in prepared["output_columns"]]
    quasi = st.multiselect("Quasi-identifiers", columns)
    sensitive = st.selectbox("Sensitive column", ["(none)"] + columns)
    st.caption("Attribute disclosure trains a model on synthetic rows only, then scores it on real rows. Thresholds are edited in Settings.")
    if st.button("Run privacy diagnostics", type="primary"):
        try:
            holdout = prepared["arrays"]["holdout"]
            try:
                synthetic_matrix = prepared["preprocessor"].transform(synthetic)
            except Exception:
                synthetic_matrix = None
            report = run_privacy(
                prepared["splits"]["train"],
                synthetic,
                train_matrix=prepared["arrays"]["train"],
                holdout_matrix=holdout,
                synthetic_matrix=synthetic_matrix,
                quasi_identifiers=quasi,
                sensitive_column=None if sensitive == "(none)" else sensitive,
                thresholds=state["thresholds"],
                max_rows=state["config"].evaluation.max_rows_for_privacy,
                seed=int(state["model_config"].get("seed", 42)),
            )
            state["privacy"] = report
            if state.get("experiment_id"):
                ExperimentStore(project_root()).update(state["experiment_id"], {"privacy": report, "status": "privacy_evaluated"})
        except Exception as exc:
            show_exception(exc, "privacy")
    report = state.get("privacy")
    if not report:
        return
    exact = report.get("exact_match") or {}
    neighbor = report.get("nearest_neighbor") or {}
    membership = report.get("membership") or {}
    disclosure = report.get("attribute_disclosure") or {}
    cards = st.columns(4)
    blocks = [
        ("Exact match rate", fmt(exact.get("exact_match_rate")), exact.get("label") or ""),
        ("Nearest-neighbor ratio", fmt(neighbor.get("distance_ratio")), neighbor.get("label") or ""),
        ("Train/holdout advantage", fmt(membership.get("advantage_over_half")), membership.get("label") or ""),
        ("Disclosure label", disclosure.get("label") or "Not run", fmt(disclosure.get("lift"))),
    ]
    for column, item in zip(cards, blocks):
        with column:
            card(*item)
    st.info(report.get("overall_label") or "No overall label.")
    st.caption(DISCLAIMER)
    st.caption(f"Thresholds in use: {report.get('thresholds')}")
    if neighbor.get("distance_sample"):
        st.plotly_chart(distance_histogram(neighbor["distance_sample"], "Nearest-neighbor distance"), use_container_width=True)
    if membership and membership.get("available"):
        st.plotly_chart(
            train_holdout_histogram(membership.get("train_distance_sample") or [], membership.get("holdout_distance_sample") or []),
            use_container_width=True,
        )
    with st.expander("Definitions"):
        for key in ("exact_match", "nearest_neighbor", "membership", "attribute_disclosure"):
            item = report.get(key) or {}
            if item.get("definition"):
                st.markdown(f"**{key}**")
                st.write(item["definition"])
    st.download_button("Download privacy JSON", json_bytes(report), file_name="privacy_diagnostics.json")
