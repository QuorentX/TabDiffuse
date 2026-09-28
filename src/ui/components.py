"""Shared Streamlit widgets."""

from __future__ import annotations

import html
from typing import Any

import streamlit as st

from src.utils.exceptions import TabDiffuseError
from src.utils.logger import log_exception


def card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        "<div class='td-card'>"
        f"<div class='td-label'>{html.escape(label)}</div>"
        f"<div class='td-value'>{html.escape(value)}</div>"
        f"<div class='td-note'>{html.escape(note)}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def workflow(active: str | None = None) -> None:
    steps = ["Upload", "Profile", "Preprocess", "Train", "Generate", "Evaluate", "Report"]
    parts = ["<div class='td-flow'>"]
    for index, step in enumerate(steps):
        parts.append(f"<span class='td-step'>{html.escape(step)}</span>")
        if index < len(steps) - 1:
            parts.append("<span class='td-arrow'>↓</span>" if False else "<span class='td-arrow'>→</span>")
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)
    if active:
        st.caption(f"Current stage: {active}")


def banner(text: str) -> None:
    st.markdown(f"<div class='td-banner'>{html.escape(text)}</div>", unsafe_allow_html=True)


def disclaimer(text: str) -> None:
    st.markdown(f"<div class='td-disclaimer'>{html.escape(text)}</div>", unsafe_allow_html=True)


def show_exception(exc: BaseException, event: str) -> None:
    log_exception("ui", event, exc)
    if isinstance(exc, TabDiffuseError):
        st.error(exc.message)
    else:
        st.error("The operation could not be completed. Technical details were written to the log.")


def metric_row(items: list[tuple[str, str, str]]) -> None:
    columns = st.columns(len(items))
    for column, item in zip(columns, items):
        with column:
            card(*item)


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)
