"""Machine-learning utility page."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config.settings import project_root
from src.reporting.export import json_bytes
from src.services.experiments import ExperimentStore
from src.services.pipeline import run_utility
from src.ui.components import show_exception
from src.visualization.dashboard import apply_layout


def render(state: dict[str, Any]) -> None:
    st.title("ML Utility")
    st.caption("Train on real, synthetic, or both. Always test on held-out real rows. The table does not declare a winner.")
    prepared = state.get("prepared")
    synthetic = state.get("synthetic")
    if prepared is None or synthetic is None:
        st.info("Generate synthetic data before evaluating utility.")
        return
    columns = list(prepared["output_columns"])
    default = state.get("target") if state.get("target") in columns else columns[0]
    target = st.selectbox("Target column", columns, index=columns.index(default))
    task = st.selectbox("Task", ["auto", "classification", "regression"], index=["auto", "classification", "regression"].index(state.get("task", "auto")))
    state["target"] = target
    state["task"] = task
    if st.button("Run utility evaluation", type="primary"):
        try:
            result = run_utility(
                prepared["splits"]["train"],
                prepared["splits"]["holdout"],
                synthetic,
                target,
                task=None if task == "auto" else task,
                seed=int(state["model_config"].get("seed", 42)),
                max_rows=state["config"].evaluation.max_rows_for_utility,
                n_estimators=state["config"].evaluation.utility_estimators,
            )
            state["utility"] = result
            if state.get("experiment_id"):
                ExperimentStore(project_root()).update(state["experiment_id"], {"utility": result, "status": "utility_evaluated"})
        except Exception as exc:
            show_exception(exc, "utility")
    result = state.get("utility")
    if not result:
        return
    st.write(f"Task: {result.get('task')} · Target: {result.get('target')} · Holdout rows: {result.get('holdout_rows')}")
    if result.get("subsampled"):
        st.caption(f"Training tables larger than {result.get('max_rows')} rows were sampled. This limit is recorded with the metrics.")
    for note in result.get("notes") or []:
        st.caption(note)
    frame = pd.DataFrame(result.get("rows") or [])
    if frame.empty:
        st.warning("No models could be scored.")
        return
    if result.get("task") == "classification":
        show = frame[["training_data", "model", "accuracy", "precision", "recall", "f1", "roc_auc"]]
        value = "f1"
    else:
        show = frame[["training_data", "model", "mae", "rmse", "r2"]]
        value = "r2"
    st.dataframe(show, use_container_width=True, hide_index=True)
    figure = apply_layout(px.bar(frame, x="model", y=value, color="training_data", barmode="group", title=value))
    st.plotly_chart(figure, use_container_width=True)
    st.download_button("Download utility JSON", json_bytes(result), file_name="utility_metrics.json")
