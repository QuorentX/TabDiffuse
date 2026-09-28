"""Home page."""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.config.settings import project_root
from src.data.loader import load_tabular
from src.data.sample_data import DEMO_SETS, demo_set_path, write_demo_sets
from src.services.pipeline import profile_dataset
from src.ui.components import banner, card, show_exception, workflow
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
        st.caption("Both demonstration sets are simulated. Neither is a real extract.")
        load_a, load_b = st.columns(2)
        if load_a.button("Load demo set A — Lending", type="primary"):
            _load_demo(state, "A")
        if load_b.button("Load demo set B — Retail"):
            _load_demo(state, "B")
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


def _load_demo(state: dict[str, Any], key: str) -> None:
    try:
        config = state["config"]
        spec = DEMO_SETS[key]
        path = demo_set_path(key)
        if not path.exists():
            write_demo_sets(n_rows=int(spec["rows"]))
        frame = load_tabular(path)
        state["frame"] = frame
        state["dataset_name"] = str(spec["filename"])
        state["file_size"] = path.stat().st_size
        state["demo_mode"] = True
        state["profile"] = profile_dataset(frame, _thresholds(config))
        state["roles"] = state["profile"]["roles"]
        state["target"] = state["profile"].get("suggested_target")
        state["model_config"] = _demo_config(state)
        state["prepared"] = None
        state["synthetic"] = None
        state["bundle"] = None
        state["training_summary"] = None
        state["experiment_id"] = None
        state["experiment"] = None
        state["quality"] = None
        state["privacy"] = None
        state["utility"] = None
        state["baselines"] = None
        state["notes"]["dataset_description"] = str(spec["description"])
        st.success(
            f"{spec['label']} is loaded ({len(frame):,} rows). "
            "Demo Mode uses a smaller model. Continue with profiling."
        )
    except Exception as exc:
        show_exception(exc, "load_demo")


def _thresholds(config) -> dict:
    return {
        "id_uniqueness_threshold": config.data.id_uniqueness_threshold,
        "high_cardinality_threshold": config.data.high_cardinality_threshold,
        "missing_warning_threshold": config.data.missing_warning_threshold,
        "duplicate_warning_threshold": config.data.duplicate_warning_threshold,
        "imbalance_warning_threshold": config.data.imbalance_warning_threshold,
    }


def _demo_config(state: dict[str, Any]) -> dict[str, Any]:
    config = state["config"]
    model = config.model_training_dict()
    model.update(
        {
            "epochs": config.demo.epochs,
            "batch_size": config.demo.batch_size,
            "hidden_dim": config.demo.hidden_dim,
            "layers": config.demo.layers,
            "steps": config.demo.steps,
            "sampling_steps": config.demo.sampling_steps,
            "samples": config.demo.samples,
            "demo_mode": True,
            "device": "cpu",
        }
    )
    return model
