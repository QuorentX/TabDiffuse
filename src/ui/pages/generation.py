"""Synthetic generation page."""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.config.settings import project_root
from src.reporting.export import dataframe_csv_bytes, dataframe_parquet_bytes
from src.services.experiments import ExperimentStore
from src.services.pipeline import generate_synthetic, load_bundle
from src.ui.components import card, show_exception
from src.ui.state import go
from src.utils.file_utils import experiment_paths, humanize_seconds


def render(state: dict[str, Any]) -> None:
    st.title("Synthetic Generation")
    prepared = state.get("prepared")
    if prepared is None or not state.get("experiment_id"):
        st.info("Train a model before generating data.")
        return
    if state.get("bundle") is None:
        try:
            path = project_root() / "models" / "checkpoints" / state["experiment_id"] / "checkpoint_best.pt"
            state["bundle"] = load_bundle(path, state["model_config"].get("device", "cpu"))
        except Exception as exc:
            show_exception(exc, "load_for_generation")
            return
    st.write(f"Selected model: `{state['experiment_id']}`")
    presets = st.columns(5)
    counts = [1000, 5000, 10000, 50000]
    chosen = state["model_config"].get("samples", 1000)
    for column, count in zip(presets, counts):
        if column.button(f"{count:,}"):
            chosen = count
    custom = presets[4].number_input("Custom", min_value=1, max_value=200000, value=int(chosen), step=100)
    n_samples = int(custom)
    if state.get("frame") is not None:
        ratio = st.selectbox("Augmentation relative to uploaded rows", ["Custom count", "10%", "25%", "50%", "100%", "200%"])
        if ratio != "Custom count":
            n_samples = max(1, int(len(state["frame"]) * float(ratio.strip("%")) / 100))
            st.caption(
                f"Uploaded rows: {len(state['frame']):,}. Synthetic augmentation: {n_samples:,}. "
                f"Combined, excluding the untouched holdout: {len(prepared['splits']['train']) + n_samples:,} training rows."
            )
    steps = st.number_input("Sampling steps", min_value=10, max_value=2000, value=int(state["model_config"].get("sampling_steps", state["model_config"].get("steps", 100))))
    temperature = st.number_input("Temperature", min_value=0.1, max_value=2.0, value=float(state["model_config"].get("temperature", 1.0)), step=0.1, help="Scales the reverse-process noise. 1 leaves the sampler unchanged.")
    seed = st.number_input("Random seed", min_value=0, value=int(state["model_config"].get("seed", 42)))
    condition_column = st.selectbox("Condition column", ["(none)"] + list(prepared["output_columns"]))
    condition_value = None
    balance = False
    if condition_column != "(none)":
        series = prepared["splits"]["train"][condition_column].dropna()
        if series.nunique() <= 30:
            condition_value = st.selectbox("Condition value", series.astype(str).unique().tolist())
        else:
            condition_value = st.text_input("Condition value")
        balance = st.checkbox("Balance classes of this column", help="Generates an equal count per observed class. This changes the target marginal and is reported.")
    use_constraints = st.checkbox("Apply constraints", value=False)
    default_rules = _example_rules()
    rules = st.text_area("Constraint YAML", value=default_rules, height=180, disabled=not use_constraints)
    st.caption("Constraints do nothing until this box is enabled. Clipping happens only for rules whose action is clip, and every change is counted.")
    if st.button("Generate synthetic data", type="primary"):
        progress = st.progress(0.0, text="Starting sampler")
        status = st.empty()

        def on_progress(done: int, total: int) -> None:
            progress.progress(done / max(total, 1), text=f"{done:,} / {total:,} latent rows")
            status.caption("Sampling is running on this machine.")

        try:
            result = generate_synthetic(
                state["bundle"],
                prepared["splits"]["train"],
                prepared["arrays"]["train"],
                n_samples=n_samples,
                steps=int(steps),
                temperature=float(temperature),
                seed=int(seed),
                condition_column=None if condition_column == "(none)" else condition_column,
                condition_value=None if condition_column == "(none)" or balance else condition_value,
                balance_column=condition_column if balance else None,
                constraint_text=rules if use_constraints else None,
                on_progress=on_progress,
            )
            state["synthetic"] = result["frame"]
            state["generation"] = {key: value for key, value in result.items() if key != "frame"}
            paths = experiment_paths(project_root(), state["experiment_id"])
            csv_path = paths["synthetic"] / "synthetic.csv"
            parquet_path = paths["synthetic"] / "synthetic.parquet"
            result["frame"].to_csv(csv_path, index=False)
            result["frame"].to_parquet(parquet_path, index=False)
            state["generation"]["csv_path"] = str(csv_path)
            state["generation"]["parquet_path"] = str(parquet_path)
            ExperimentStore(project_root()).update(
                state["experiment_id"],
                {"status": "generated", "generation": state["generation"]},
            )
            progress.progress(1.0, text="Generation finished")
        except Exception as exc:
            show_exception(exc, "generate")
    synthetic = state.get("synthetic")
    generation = state.get("generation") or {}
    if synthetic is None:
        return
    if generation.get("exact_match_rate") is None and state.get("prepared") is not None:
        from src.evaluation.privacy import exact_match_diagnostic

        train = state["prepared"]["splits"]["train"]
        shared = [column for column in train.columns if column in synthetic.columns]
        exact = exact_match_diagnostic(train[shared], synthetic[shared], state["thresholds"])
        generation["exact_matches"] = exact["exact_matches"]
        generation["exact_match_rate"] = exact["exact_match_rate"]
        state["generation"] = generation
    columns = st.columns(5)
    items = [
        ("Generated records", f"{len(synthetic):,}", ""),
        ("Generation time", humanize_seconds(generation.get("duration_seconds")), ""),
        ("Rows/sec", f"{generation.get('rows_per_second'):.1f}" if generation.get("rows_per_second") else "—", ""),
        ("Constraint rejections", str((generation.get("constraints") or {}).get("rejected_rows", 0)), ""),
        ("Exact duplicates", str(generation.get("exact_matches", "—")), f"{generation.get('exact_match_rate', 0):.2%}" if generation.get("exact_match_rate") is not None else ""),
    ]
    for column, item in zip(columns, items):
        with column:
            card(*item)
    if generation.get("constraints", {}).get("enabled"):
        st.dataframe(generation["constraints"].get("rules", []), use_container_width=True)
    inverse = generation.get("inverse") or {}
    notes = []
    if inverse.get("projected_numeric_values"):
        notes.append(
            f"{int(inverse['projected_numeric_values'])} numeric values were moved back onto the range seen in the training split."
        )
    if inverse.get("repaired_dates"):
        notes.append(f"{int(inverse['repaired_dates'])} dates were repaired to a real calendar day.")
    if inverse.get("invalid_dates"):
        notes.append(f"{int(inverse['invalid_dates'])} dates could not be formed and were left missing.")
    if notes:
        st.caption(" ".join(notes))
    st.caption(generation.get("conditioning_method", ""))
    st.dataframe(synthetic.head(20), use_container_width=True)
    downloads = st.columns(3)
    downloads[0].download_button("Download CSV", dataframe_csv_bytes(synthetic), file_name=f"{state['experiment_id']}_synthetic.csv")
    downloads[1].download_button("Download Parquet", dataframe_parquet_bytes(synthetic), file_name=f"{state['experiment_id']}_synthetic.parquet")
    if downloads[2].button("Run evaluation"):
        go("Quality Evaluation")


def _example_rules() -> str:
    path = project_root() / "data" / "constraints.example.yaml"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return "constraints: []\n"
