"""
Oil-Geo Research Suite — unified Streamlit entry (GeoOil Intel + Monday Alpha).

Run from project root:
  streamlit run src/app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

_SRC = Path(__file__).resolve().parent

_CHASE = "#003087"
_BG = "#f0f2f5"
_PANEL = "#ffffff"
_TEXT = "#1a1d21"
_MUTED = "#5c6370"
_BORDER = "rgba(0, 48, 135, 0.2)"

_SUITE_CSS = f"""
<style>
  .stApp {{
    background: linear-gradient(180deg, {_BG} 0%, #e8ebf0 50%, {_BG} 100%);
    color: {_TEXT};
  }}
  [data-testid="stHeader"] {{
    background-color: {_PANEL};
    border-bottom: 2px solid {_CHASE};
  }}
  [data-testid="stSidebarNav"] {{
    background-color: {_PANEL};
    border-right: 1px solid {_BORDER};
  }}
  [data-testid="stSidebar"] {{
    background-color: {_PANEL} !important;
  }}
  [data-testid="stMarkdownContainer"] p,
  [data-testid="stMarkdownContainer"] li {{
    color: {_TEXT};
  }}
  h1, h2, h3 {{
    color: {_CHASE} !important;
  }}
  span[data-testid="stNavLinkLabel"] {{
    color: {_TEXT} !important;
  }}
  a[data-testid="stNavLink"] {{
    border-radius: 6px;
  }}
  a[data-testid="stNavLink"][aria-current="page"] {{
    background-color: rgba(0, 48, 135, 0.08) !important;
    border-left: 3px solid {_CHASE} !important;
  }}
  button[kind="primary"] {{
    background-color: {_CHASE} !important;
    border-color: {_CHASE} !important;
  }}
  button[kind="primary"]:hover {{
    background-color: #002266 !important;
    border-color: #002266 !important;
  }}
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
        title="Energy & Equity",
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
