"""Quality evaluation page."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from src.reporting.export import json_bytes
from src.services.experiments import ExperimentStore
from src.services.pipeline import run_quality
from src.ui.components import card, fmt, show_exception
from src.ui.state import go
from src.config.settings import project_root
from src.visualization.correlations import heatmap
from src.visualization.dimensionality import project_coordinates, projection_figure
from src.visualization.distributions import box_figure, category_figure, ecdf_figure, histogram_figure


def render(state: dict[str, Any]) -> None:
    st.title("Quality Evaluation")
    prepared = state.get("prepared")
    synthetic = state.get("synthetic")
    if prepared is None or synthetic is None:
        st.info("Generate synthetic data before comparing distributions.")
        return
    if st.button("Run quality evaluation", type="primary"):
        try:
            state["quality"] = run_quality(prepared["splits"]["train"], synthetic)
            if state.get("experiment_id"):
                ExperimentStore(project_root()).update(state["experiment_id"], {"evaluation": state["quality"], "status": "quality_evaluated"})
            st.success("Quality metrics were calculated from the current real training split and synthetic sample.")
        except Exception as exc:
            show_exception(exc, "quality")
    quality = state.get("quality")
    if not quality:
        return
    similarity = quality.get("similarity") or {}
    columns = st.columns(4)
    items = [
        ("Distribution similarity", fmt(similarity.get("distribution_similarity")), "1 - mean KS"),
        ("Correlation similarity", fmt(similarity.get("correlation_similarity")), "1 - mean absolute Pearson gap"),
        ("Categorical similarity", fmt(similarity.get("categorical_similarity")), "1 - mean total variation"),
        ("Synthetic rows", f"{quality.get('rows_synthetic', 0):,}", ""),
    ]
    for column, item in zip(columns, items):
        with column:
            card(*item)
    st.caption("These scores summarize measured gaps. They do not prove that the synthetic data matches the real distribution.")
    real = prepared["splits"]["train"]
    shared = [column for column in real.columns if column in synthetic.columns]
    numeric = [column for column in shared if pd.api.types.is_numeric_dtype(real[column])]
    categorical = [column for column in shared if column not in numeric]
    tabs = st.tabs(["Distribution", "Correlation", "Categorical", "PCA", "t-SNE", "UMAP", "Metrics"])
    with tabs[0]:
        if not numeric:
            st.info("No shared numerical columns.")
        else:
            column = st.selectbox("Numerical column", numeric, key="quality_num")
            st.plotly_chart(histogram_figure(real[column], synthetic[column], column), use_container_width=True)
            left, right = st.columns(2)
            left.plotly_chart(box_figure(real[column], synthetic[column], "Box plot"), use_container_width=True)
            right.plotly_chart(ecdf_figure(real[column], synthetic[column], "ECDF"), use_container_width=True)
    with tabs[1]:
        correlation = quality.get("correlation") or {}
        if not correlation.get("available"):
            st.info(correlation.get("message", "Correlation is unavailable."))
        else:
            labels = correlation["columns"]
            st.plotly_chart(heatmap(correlation["pearson_real"], labels, "Real Pearson"), use_container_width=True)
            st.plotly_chart(heatmap(correlation["pearson_synthetic"], labels, "Synthetic Pearson"), use_container_width=True)
            st.plotly_chart(heatmap(correlation["pearson_difference"], labels, "Pearson difference"), use_container_width=True)
    with tabs[2]:
        if not categorical:
            st.info("No categorical columns to chart.")
        else:
            column = st.selectbox("Categorical column", categorical, key="quality_cat")
            st.plotly_chart(category_figure(real[column], synthetic[column], column), use_container_width=True)
    _projection_tab(tabs[3], state, "pca", "PCA", 4000)
    _projection_tab(tabs[4], state, "tsne", "t-SNE", 800)
    _projection_tab(tabs[5], state, "umap", "UMAP", 800)
    with tabs[6]:
        numerical = (quality.get("numerical") or {}).get("columns") or {}
        if numerical:
            st.dataframe(pd.DataFrame(numerical).T, use_container_width=True)
        categorical_metrics = (quality.get("categorical") or {}).get("columns") or {}
        if categorical_metrics:
            st.dataframe(pd.DataFrame(categorical_metrics).T, use_container_width=True)
        st.download_button("Download metrics JSON", json_bytes(quality), file_name="quality_metrics.json")
    if st.button("Open report page"):
        go("Reports")


def _projection_tab(tab, state: dict[str, Any], method: str, title: str, max_rows: int) -> None:
    with tab:
        st.caption("This projection is a qualitative diagnostic. It is not evidence of privacy or of statistical equivalence.")
        if st.button(f"Compute {title}", key=f"proj_{method}"):
            try:
                prepared = state["prepared"]
                synthetic = state["synthetic"]
                real_x = prepared["arrays"]["train"]
                syn_x = prepared["preprocessor"].transform(synthetic)
                coordinates = project_coordinates(real_x, syn_x, method, max_rows=max_rows, seed=int(state["model_config"].get("seed", 42)))
                state["projection"] = state.get("projection") or {}
                state["projection"][method] = coordinates
            except Exception as exc:
                show_exception(exc, f"projection_{method}")
        coordinates = (state.get("projection") or {}).get(method)
        if coordinates is not None:
            st.plotly_chart(projection_figure(coordinates, title), use_container_width=True)
