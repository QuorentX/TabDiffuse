"""TabDiffuse Streamlit application."""

from __future__ import annotations

import streamlit as st

from src.ui.pages import (
    comparison,
    generation,
    home,
    model_config,
    preprocessing,
    privacy,
    profiling,
    quality,
    reports,
    settings,
    training,
    upload,
    utility,
)
from src.ui.state import PAGES, apply_navigation, init_state
from src.ui.theme import inject_css

PAGES_MAP = {
    "Home": home.render,
    "Dataset Upload": upload.render,
    "Data Profiling": profiling.render,
    "Preprocessing": preprocessing.render,
    "Model Configuration": model_config.render,
    "Training": training.render,
    "Synthetic Generation": generation.render,
    "Quality Evaluation": quality.render,
    "Privacy Evaluation": privacy.render,
    "ML Utility": utility.render,
    "Experiment Comparison": comparison.render,
    "Reports": reports.render,
    "Settings": settings.render,
}


def main() -> None:
    st.set_page_config(
        page_title="TabDiffuse",
        page_icon="◇",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()
    state = init_state()
    apply_navigation()
    with st.sidebar:
        st.markdown("<div class='td-brand'>TABDIFFUSE</div>", unsafe_allow_html=True)
        st.markdown("<div class='td-sub'>Synthetic Tabular Data Generation</div>", unsafe_allow_html=True)
        st.radio("Navigation", PAGES, key="nav_page", label_visibility="collapsed")
        st.divider()
        if state.get("experiment_id"):
            st.caption("Current experiment")
            st.write(state["experiment_id"])
        if state.get("demo_mode"):
            st.caption("Demo Mode – Reduced training configuration")
        st.caption("Data stays on this machine. Nothing is uploaded to an external service.")
    page = st.session_state.nav_page
    PAGES_MAP[page](state)


if __name__ == "__main__":
    main()
