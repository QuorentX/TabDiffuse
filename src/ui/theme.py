"""Global styles for the research dashboard."""

from __future__ import annotations

import streamlit as st

CSS = """
<style>
    .block-container { padding-top: 1.4rem; max-width: 1180px; }
    h1, h2, h3 { letter-spacing: -0.02em; }
    [data-testid="stSidebar"] { background: #f7f8fa; border-right: 1px solid #e2e8ee; }
    .td-brand { font-size: 1.35rem; font-weight: 700; color: #14312f; margin-bottom: 0; }
    .td-sub { color: #5c6b7a; font-size: 0.86rem; margin-bottom: 0.8rem; }
    .td-card {
        background: #ffffff;
        border: 1px solid #e2e8ee;
        border-radius: 12px;
        padding: 14px 16px 12px 16px;
        min-height: 108px;
    }
    .td-label { color: #5c6b7a; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; }
    .td-value { color: #14312f; font-size: 1.45rem; font-weight: 650; margin: 4px 0; }
    .td-note { color: #5c6b7a; font-size: 0.84rem; }
    .td-flow { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 8px 0 4px 0; }
    .td-step { background: #e7f3f2; color: #0e7c74; border-radius: 999px; padding: 6px 10px; font-size: 0.84rem; font-weight: 600; }
    .td-arrow { color: #8b98a5; }
    .td-banner {
        background: #f7f1e6;
        border-left: 4px solid #c9842a;
        padding: 10px 12px;
        border-radius: 8px;
        color: #5c4314;
        margin-bottom: 12px;
    }
    .td-disclaimer {
        background: #f4f7fb;
        border: 1px solid #d7e1ea;
        border-radius: 10px;
        padding: 12px 14px;
        color: #243140;
        margin: 8px 0 14px 0;
    }
    div[data-testid="stMetric"] { background: white; border: 1px solid #e2e8ee; padding: 8px 12px; border-radius: 10px; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
