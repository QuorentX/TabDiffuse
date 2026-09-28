"""Report page."""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.config.settings import project_root
from src.reporting.report_generator import build_report
from src.services.experiments import ExperimentStore
from src.ui.components import show_exception
from src.utils.file_utils import experiment_paths


def render(state: dict[str, Any]) -> None:
    st.title("Reports")
    store = ExperimentStore(project_root())
    records = store.list_experiments()
    if not records and not state.get("experiment_id"):
        st.info("Train a model to create a report. The report only includes sections that were actually run.")
        return
    labels = [record["experiment_id"] for record in records] or ([state["experiment_id"]] if state.get("experiment_id") else [])
    selected = st.selectbox("Experiment", labels)
    record = next((item for item in records if item["experiment_id"] == selected), state.get("experiment") or {"experiment_id": selected})
    st.write(
        {
            "Dataset": record.get("dataset_name"),
            "Status": record.get("status"),
            "Created": record.get("created_at"),
        }
    )
    notes = record.get("notes") or state.get("notes") or {}
    notes["observations"] = st.text_area("Observations", notes.get("observations", ""))
    notes["conclusion"] = st.text_area("Conclusion", notes.get("conclusion", ""))
    if st.button("Generate HTML and PDF report", type="primary"):
        try:
            record["notes"] = notes
            if selected == state.get("experiment_id"):
                state["notes"] = notes
            store.update(selected, {"notes": notes, "status": "report_ready"})
            paths = experiment_paths(project_root(), selected)
            real = None
            synthetic = None
            if selected == state.get("experiment_id") and state.get("prepared") and state.get("synthetic") is not None:
                real = state["prepared"]["splits"]["train"]
                synthetic = state["synthetic"]
            outputs = build_report(
                store.load(selected),
                paths["reports"] / "report.html",
                paths["reports"] / "report.pdf",
                real=real,
                synthetic=synthetic,
            )
            html = (paths["reports"] / "report.html").read_bytes()
            pdf = (paths["reports"] / "report.pdf").read_bytes()
            st.success("Report written with the measurements saved for this experiment.")
            st.download_button("Download HTML", html, file_name=f"{selected}_report.html")
            st.download_button("Download PDF", pdf, file_name=f"{selected}_report.pdf")
            st.caption(outputs["html"])
        except Exception as exc:
            show_exception(exc, "report")
