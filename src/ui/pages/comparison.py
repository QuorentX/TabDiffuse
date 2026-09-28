"""Compare saved experiments and optional baselines."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config.settings import project_root
from src.services.experiments import ExperimentStore
from src.services.pipeline import run_baseline_comparison
from src.ui.components import show_exception
from src.visualization.dashboard import apply_layout


def render(state: dict[str, Any]) -> None:
    st.title("Experiment Comparison")
    store = ExperimentStore(project_root())
    records = store.list_experiments()
    if not records:
        st.info("Train at least one experiment to build a comparison history.")
    else:
        labels = [record["experiment_id"] for record in records]
        selected = st.multiselect("Experiments", labels, default=labels[: min(3, len(labels))])
        chosen = [record for record in records if record["experiment_id"] in selected]
        rows = [_row(record) for record in chosen]
        if rows:
            frame = pd.DataFrame(rows)
            st.dataframe(frame, use_container_width=True, hide_index=True)
            if "best_val_loss" in frame and frame["best_val_loss"].notna().any():
                figure = apply_layout(px.bar(frame, x="experiment_id", y="best_val_loss", title="Best validation loss"))
                st.plotly_chart(figure, use_container_width=True)
            if frame["distribution_similarity"].notna().any():
                long = frame.melt(
                    id_vars=["experiment_id"],
                    value_vars=["distribution_similarity", "correlation_similarity", "categorical_similarity"],
                    var_name="metric",
                    value_name="score",
                )
                figure = apply_layout(px.bar(long, x="experiment_id", y="score", color="metric", barmode="group", title="Quality scores"))
                st.plotly_chart(figure, use_container_width=True)
    st.markdown("#### Baselines for the current dataset")
    st.caption("Random resampling copies real rows, so a high exact-match rate is expected. Other methods are measured the same way. No method is labeled the winner.")
    prepared = state.get("prepared")
    if prepared is None:
        st.info("Preprocess a dataset to compare baselines against TabDiffuse on the same training split.")
        return
    flags = state["config"].baselines
    enable_ctgan = st.checkbox("Include CTGAN if installed", value=flags.enable_ctgan)
    n_samples = st.number_input("Baseline rows", min_value=50, max_value=20000, value=int(min(flags.baseline_samples, 1000)))
    if st.button("Run baseline comparison"):
        try:
            rows = run_baseline_comparison(
                prepared["splits"]["train"],
                prepared["splits"]["holdout"],
                n_samples=int(n_samples),
                seed=int(state["model_config"].get("seed", 42)),
                target=state.get("target"),
                flags={
                    "enable_resample": flags.enable_resample,
                    "enable_gaussian": flags.enable_gaussian,
                    "enable_copula": flags.enable_copula,
                    "enable_ctgan": enable_ctgan,
                    "ctgan_epochs": flags.ctgan_epochs,
                },
            )
            if state.get("quality"):
                privacy = state.get("privacy") or {}
                rows.insert(
                    0,
                    {
                        "method": "TabDiffuse",
                        "available": True,
                        "duration_seconds": (state.get("generation") or {}).get("duration_seconds"),
                        "distribution_similarity": (state["quality"].get("similarity") or {}).get("distribution_similarity"),
                        "correlation_similarity": (state["quality"].get("similarity") or {}).get("correlation_similarity"),
                        "categorical_similarity": (state["quality"].get("similarity") or {}).get("categorical_similarity"),
                        "exact_match_rate": (privacy.get("exact_match") or {}).get("exact_match_rate"),
                        "distance_ratio": (privacy.get("nearest_neighbor") or {}).get("distance_ratio"),
                        "utility_score": _utility_score(state.get("utility")),
                    },
                )
            state["baselines"] = rows
            if state.get("experiment_id"):
                store.update(state["experiment_id"], {"baselines": rows})
        except Exception as exc:
            show_exception(exc, "baselines")
    if state.get("baselines"):
        frame = pd.DataFrame(state["baselines"])
        st.dataframe(frame, use_container_width=True, hide_index=True)
        available = frame[frame["available"].fillna(False)] if "available" in frame.columns else frame
        if not available.empty and "distribution_similarity" in available:
            figure = apply_layout(
                px.bar(available, x="method", y="distribution_similarity", title="Distribution similarity by method")
            )
            st.plotly_chart(figure, use_container_width=True)


def _row(record: dict[str, Any]) -> dict[str, Any]:
    training = record.get("training") or {}
    generation = record.get("generation") or {}
    similarity = ((record.get("evaluation") or {}).get("similarity")) or {}
    privacy = record.get("privacy") or {}
    config = record.get("model_config") or {}
    return {
        "experiment_id": record.get("experiment_id"),
        "dataset": record.get("dataset_name"),
        "rows": record.get("rows"),
        "columns": record.get("columns"),
        "hidden_dim": config.get("hidden_dim"),
        "layers": config.get("layers"),
        "epochs": config.get("epochs"),
        "steps": config.get("steps"),
        "seed": record.get("seed"),
        "best_val_loss": training.get("best_val_loss"),
        "training_seconds": training.get("duration_seconds"),
        "generation_seconds": generation.get("duration_seconds"),
        "synthetic_rows": generation.get("rows"),
        "distribution_similarity": similarity.get("distribution_similarity"),
        "correlation_similarity": similarity.get("correlation_similarity"),
        "categorical_similarity": similarity.get("categorical_similarity"),
        "exact_match_rate": (privacy.get("exact_match") or {}).get("exact_match_rate"),
        "privacy_label": privacy.get("overall_label"),
    }


def _utility_score(utility: dict[str, Any] | None) -> float | None:
    if not utility:
        return None
    for row in utility.get("rows") or []:
        if row.get("training_data") == "Synthetic":
            return row.get("f1") if row.get("f1") is not None else row.get("r2")
    return None
