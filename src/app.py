"""
Oil-Geo Research Suite — unified Streamlit entry (GeoOil Intel + Monday Alpha).

Run from project root:
  streamlit run src/app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

_SRC = Path(__file__).resolve().parent

_SUITE_CSS = """
<style>
  .stApp {
    background: linear-gradient(180deg, #0f1116 0%, #131722 40%, #1a1d27 100%);
    color: #eaecef;
  }
  [data-testid="stHeader"] {
    background-color: #131722;
    border-bottom: 1px solid #2b3139;
  }
  [data-testid="stSidebarNav"] {
    background-color: #131722;
  }
  [data-testid="stSidebar"] {
    background-color: #131722 !important;
  }
  [data-testid="stMarkdownContainer"] p,
  [data-testid="stMarkdownContainer"] li {
    color: #eaecef;
  }
  h1, h2, h3 {
    color: #eaecef !important;
  }
  span[data-testid="stNavLinkLabel"] {
    color: #eaecef !important;
  }
</style>
"""

st.set_page_config(
    page_title="Oil-Geo Research Suite",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(_SUITE_CSS, unsafe_allow_html=True)

pages = [
    st.Page(
        str(_SRC / "geo_oil_intel.py"),
        title="GeoOil Intel",
        icon="🌍",
        default=True,
    ),
    st.Page(
        str(_SRC / "dashboard.py"),
        title="Monday Alpha",
        icon="📈",
    ),
]

st.navigation(pages).run()
