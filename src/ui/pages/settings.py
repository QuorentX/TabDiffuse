"""Settings, thresholds, and local cleanup."""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.config.settings import project_root
from src.services.experiments import ExperimentStore
from src.ui.components import show_exception
from src.utils.file_utils import delete_directory, experiment_paths


def render(state: dict[str, Any]) -> None:
    st.title("Settings")
    st.caption("Privacy labels use these editable cutoffs. Changing them does not change the underlying measurements, only the label.")
    thresholds = state["thresholds"]
    columns = st.columns(2)
    keys = list(thresholds.keys())
    for index, key in enumerate(keys):
        thresholds[key] = columns[index % 2].number_input(key, value=float(thresholds[key]), format="%.4f", key=f"thr_{key}")
    st.markdown("#### Local cleanup")
    st.caption("Deletes session data or files stored under this project. Uploaded tables are not written to a third-party service.")
    confirm = st.checkbox("I understand this permanently deletes the selected local artifacts.")
    left, middle, right = st.columns(3)
    if left.button("Delete uploaded dataset"):
        state["frame"] = None
        state["profile"] = None
        state["roles"] = None
        state["prepared"] = None
        state["dataset_name"] = None
        state["synthetic"] = None
        st.success("The uploaded table was removed from this session.")
    if middle.button("Delete generated dataset"):
        if not confirm:
            st.warning("Enable the confirmation box before deleting generated files.")
        else:
            state["synthetic"] = None
            state["generation"] = None
            if state.get("experiment_id"):
                delete_directory(experiment_paths(project_root(), state["experiment_id"])["synthetic"], project_root())
            st.success("Generated files for the current experiment were deleted.")
    if right.button("Delete experiment artifacts"):
        if not confirm or not state.get("experiment_id"):
            st.warning("Select confirmation and have a current experiment before deleting artifacts.")
        else:
            try:
                root = project_root()
                paths = experiment_paths(root, state["experiment_id"])
                for path in paths.values():
                    delete_directory(path, root)
                state["experiment_id"] = None
                state["experiment"] = None
                state["bundle"] = None
                state["training_summary"] = None
                st.success("Experiment artifacts were deleted.")
            except Exception as exc:
                show_exception(exc, "delete_experiment")
    st.markdown("#### Environment")
    records = ExperimentStore(project_root()).list_experiments()
    st.write({"Saved experiments": len(records), "Log file": str(project_root() / "artifacts" / "logs" / "tabdiffuse.log")})
    if records:
        latest = records[0]
        st.write({"Python": latest.get("python"), "Packages": latest.get("packages")})
