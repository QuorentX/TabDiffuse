"""Session state for the local dashboard."""

from __future__ import annotations

import uuid
from typing import Any

import streamlit as st

from src.config.settings import load_config

PAGES = [
    "Home",
    "Dataset Upload",
    "Data Profiling",
    "Preprocessing",
    "Model Configuration",
    "Training",
    "Synthetic Generation",
    "Quality Evaluation",
    "Privacy Evaluation",
    "ML Utility",
    "Experiment Comparison",
    "Reports",
    "Settings",
]


def init_state() -> dict[str, Any]:
    config = load_config()
    if "session_key" not in st.session_state:
        st.session_state.session_key = uuid.uuid4().hex
    if "nav_page" not in st.session_state:
        st.session_state.nav_page = "Home"
    if "td" not in st.session_state:
        st.session_state.td = {
            "config": config,
            "dataset_name": None,
            "file_size": None,
            "frame": None,
            "profile": None,
            "roles": None,
            "prepared": None,
            "model_config": config.model_training_dict(),
            "experiment_id": None,
            "experiment": None,
            "training_summary": None,
            "bundle": None,
            "synthetic": None,
            "generation": None,
            "quality": None,
            "privacy": None,
            "utility": None,
            "baselines": None,
            "demo_mode": False,
            "target": None,
            "task": "auto",
            "notes": {
                "research_question": "",
                "hypothesis": "",
                "dataset_description": "",
                "observations": "",
                "conclusion": "",
            },
            "thresholds": config.privacy_thresholds.model_dump(),
            "projection": None,
        }
    return st.session_state.td


def apply_navigation() -> None:
    """Apply a queued page before the navigation radio is created.

    The radio owns ``nav_page``. Streamlit rejects writes to that key after
    the widget exists, which is when page buttons run. ``go`` stores the
    request under a separate key and reruns; this copies it across first.
    """
    requested = st.session_state.pop("nav_request", None)
    if requested in PAGES:
        st.session_state.nav_page = requested


def go(page: str) -> None:
    st.session_state.nav_request = page
    st.rerun()
