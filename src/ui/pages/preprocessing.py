"""Preprocessing page."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from src.services.pipeline import prepare_training_data
from src.ui.components import show_exception
from src.ui.state import go

_TYPES = ["numerical", "categorical", "boolean", "datetime", "id", "constant", "drop"]
_SCALERS = ["standard", "robust", "minmax", "none"]
_ENCODINGS = ["auto", "onehot", "embedding"]


def render(state: dict[str, Any]) -> None:
    st.title("Preprocessing")
    st.caption("Detection is editable. Columns are not dropped unless you exclude them here.")
    if state.get("frame") is None or not state.get("roles"):
        st.info("Upload and profile a dataset first.")
        return
    config = state["config"]
    st.markdown(
        """
<div class='td-flow'>
<span class='td-step'>Raw</span><span class='td-arrow'>→</span>
<span class='td-step'>Missing values</span><span class='td-arrow'>→</span>
<span class='td-step'>Encoding</span><span class='td-arrow'>→</span>
<span class='td-step'>Scaling</span><span class='td-arrow'>→</span>
<span class='td-step'>Model ready</span>
</div>
""",
        unsafe_allow_html=True,
    )
    left, right = st.columns(2)
    with left:
        imputer = st.selectbox("Numeric imputation", ["median", "mean", "constant"], index=["median", "mean", "constant"].index(config.preprocessing.numerical_imputer))
        scaler = st.selectbox("Default numeric scaler", ["standard", "robust", "minmax", "none"], index=["standard", "robust", "minmax", "none"].index(config.preprocessing.numerical_scaler), help="Standard scaling is the recommended match for the variance-preserving SDE.")
    with right:
        outlier = st.selectbox("Outlier handling", ["none", "iqr_clip"], index=0 if config.preprocessing.outlier_method == "none" else 1, help="IQR clipping is reported per column. It is lossy and is not inverted.")
        rare = st.checkbox("Group rare categories into __OTHER__", value=config.preprocessing.group_rare_categories)
    config.preprocessing.numerical_imputer = imputer
    config.preprocessing.numerical_scaler = scaler
    config.preprocessing.outlier_method = outlier
    config.preprocessing.group_rare_categories = rare

    editor = pd.DataFrame(state["roles"])
    for column, default in (("scaler", "standard"), ("encoding", "auto"), ("include", True)):
        if column not in editor:
            editor[column] = default
    edited = st.data_editor(
        editor[["name", "detected_type", "assigned_type", "include", "scaler", "encoding", "notes"]],
        use_container_width=True,
        hide_index=True,
        disabled=["name", "detected_type", "notes"],
        column_config={
            "name": "Feature",
            "detected_type": "Detected type",
            "assigned_type": st.column_config.SelectboxColumn("Assigned type", options=_TYPES, required=True),
            "include": "Include",
            "scaler": st.column_config.SelectboxColumn("Scaler", options=_SCALERS, required=True),
            "encoding": st.column_config.SelectboxColumn("Encoding", options=_ENCODINGS, required=True),
        },
        key="role_editor",
    )
    preview = edited.copy()
    preview["Model treatment"] = preview.apply(_treatment, axis=1)
    st.dataframe(preview[["name", "detected_type", "Model treatment"]], use_container_width=True, hide_index=True)
    holdout = st.slider("Downstream holdout", 0.05, 0.4, float(state["model_config"].get("holdout_ratio", 0.15)), 0.05)
    validation = st.slider("Diffusion validation", 0.05, 0.4, float(state["model_config"].get("validation_ratio", 0.15)), 0.05)
    st.caption("The holdout split is not used to fit the preprocessor or the diffusion model. It is reserved for utility and membership checks.")
    if st.button("Apply preprocessing", type="primary"):
        try:
            roles = edited.to_dict(orient="records")
            state["roles"] = roles
            state["model_config"]["holdout_ratio"] = holdout
            state["model_config"]["validation_ratio"] = validation
            prepared = prepare_training_data(
                state["frame"],
                roles,
                config,
                seed=int(state["model_config"].get("seed", 42)),
                holdout_ratio=holdout,
                validation_ratio=validation,
                stratify_column=state.get("target"),
                min_rows=config.data.min_rows,
            )
            state["prepared"] = prepared
            st.success(
                f"Preprocessing fit on {prepared['splits']['train_rows']} training rows. "
                f"Model dimension: {prepared['feature_dim']}."
            )
        except Exception as exc:
            show_exception(exc, "preprocess")
    prepared = state.get("prepared")
    if prepared:
        st.markdown("#### Recorded actions")
        st.dataframe(pd.DataFrame(prepared["actions"]), use_container_width=True, hide_index=True)
        st.write(
            {
                "Train rows": prepared["splits"]["train_rows"],
                "Validation rows": prepared["splits"]["validation_rows"],
                "Holdout rows": prepared["splits"]["holdout_rows"],
                "Excluded": prepared["metadata"].get("excluded"),
            }
        )
        if st.button("Continue to model configuration"):
            go("Model Configuration")


def _treatment(row: pd.Series) -> str:
    if not row["include"] or row["assigned_type"] in {"id", "constant", "drop"}:
        return "Excluded"
    if row["assigned_type"] == "datetime":
        return "Date parts"
    if row["assigned_type"] in {"categorical", "boolean"}:
        return "Embedding" if row["encoding"] == "embedding" else "One-hot"
    names = {"standard": "Standardized", "robust": "Robust scaled", "minmax": "Min-max scaled", "none": "Unscaled"}
    return names.get(row["scaler"], "Continuous")
