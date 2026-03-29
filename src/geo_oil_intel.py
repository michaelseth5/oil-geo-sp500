"""
Global Energy & Equity Monitor (GeoOil Intelligence): oil vs S&P 500 and rolling
correlation with a geopolitical event timeline.

Run standalone: streamlit run src/geo_oil_intel.py
Unified entry: streamlit run src/app.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent

# Design system — primary blue (interactive) + soft slate panels (fintech / corporate)
C_CHASE = "#003087"
C_PRIMARY = C_CHASE
C_BG_DEEP = "#f0f2f5"
C_BG_PANEL = "#ffffff"
C_SOFT_SLATE = "#e8eef4"
C_SOFT_CARD = "#eef3f9"
C_PANEL = "#f4f7fb"
C_GRID_SOFT = "rgba(0, 48, 135, 0.05)"
C_VLINE_EVENT = "rgba(0, 48, 135, 0.2)"
C_SPIKE = "rgba(0, 48, 135, 0.42)"
C_FILL_POS = "rgba(0, 48, 135, 0.09)"
C_TEXT = "#1a1d21"
C_MUTED = "#5c6370"
# Data lines — distinct but professional (gold oil, navy S&P); UI chrome stays C_PRIMARY
C_OIL_LINE = "#b8860b"
C_SPX_LINE = "#0a2342"
C_EVENT_BLUE = "#0b6ecf"
CORR_LINE_STYLE: dict[int, tuple[str, str]] = {
    30: ("rgba(0, 48, 135, 0.38)", "dot"),
    60: ("rgba(0, 48, 135, 0.58)", "dash"),
    90: (C_PRIMARY, "solid"),
    180: ("#001a52", "dashdot"),
}

# 18 curated events + structured detail panel fields
GEO_EVENTS: list[dict[str, str]] = [
    {
        "date": "2005-08-29",
        "label": "Gulf Coast hurricane disruption",
        "category": "Energy",
        "oil_impact": "Refinery & offshore outages → crude volatility and regional supply tightness.",
        "geo_context": "US Gulf Coast energy infrastructure under severe weather stress.",
        "market_reaction": "Energy names led; broad indices mixed as damage assessments evolved.",
        "fed_stance": "—",
    },
    {
        "date": "2007-08-09",
        "label": "Credit market liquidity stress",
        "category": "Financial",
        "oil_impact": "Risk-off lifted USD and pressured commodities; oil followed macro liquidity.",
        "geo_context": "Global interbank funding strains after structured-credit losses surfaced.",
        "market_reaction": "Credit spreads widened; equities and cyclicals de-risked.",
        "fed_stance": "Fed focused on market liquidity; easing bias into 2007–08.",
    },
    {
        "date": "2008-09-15",
        "label": "Global banking stress event",
        "category": "Financial",
        "oil_impact": "Demand destruction fears dominated → sharp crude drawdown alongside equities.",
        "geo_context": "Systemic financial stress with Lehman failure and global counterparty risk.",
        "market_reaction": "Severe equity drawdown; flight to quality and extreme volatility.",
        "fed_stance": "Emergency liquidity facilities; rapid policy easing cycle.",
    },
    {
        "date": "2010-05-06",
        "label": "Equity market structure shock",
        "category": "Financial",
        "oil_impact": "Flash-crash day correlations spiked; oil moved with macro risk appetite.",
        "geo_context": "US equity microstructure stress; transient liquidity vacuum.",
        "market_reaction": "Intraday crash/recovery; renewed focus on circuit breakers & depth.",
        "fed_stance": "Monitoring financial stability; post-crisis regulatory review phase.",
    },
    {
        "date": "2011-03-11",
        "label": "Pacific nuclear & supply-chain risk",
        "category": "Energy",
        "oil_impact": "Nuclear uncertainty + regional demand effects; LNG and risk premia in focus.",
        "geo_context": "Japan earthquake/tsunami with major industrial disruption.",
        "market_reaction": "Global supply-chain shock; insurance and industrial volatility.",
        "fed_stance": "Global coordination; G7 FX intervention context mid-2011.",
    },
    {
        "date": "2011-03-15",
        "label": "Middle East supply disruption concerns",
        "category": "Energy",
        "oil_impact": "Geopolitical risk premium in crude; supply routes under watch.",
        "geo_context": "Arab Spring dynamics; regional stability concerns for oil flows.",
        "market_reaction": "Energy outperformed intermittently; volatility in risk assets.",
        "fed_stance": "Inflation watch amid commodity spikes; policy still accommodative.",
    },
    {
        "date": "2014-06-12",
        "label": "Middle East supply uncertainty",
        "category": "Energy",
        "oil_impact": "Risk premium embedded in Brent; later balanced by shale supply growth.",
        "geo_context": "Regional conflict risk; Iraq supply headlines.",
        "market_reaction": "Energy-equity correlation regime shift into late-2014.",
        "fed_stance": "Taper ongoing; USD strength weighed on commodities.",
    },
    {
        "date": "2014-11-27",
        "label": "OPEC policy pivot — supply outlook",
        "category": "Policy",
        "oil_impact": "Bearish shift as spare-capacity narrative changed; crude entered structural slide.",
        "geo_context": "OPEC strategy vs US shale; market share vs price floor debate.",
        "market_reaction": "Energy sector repricing; HY credit stress in E&P.",
        "fed_stance": "USD strength channel post-taper favored disinflation narrative.",
    },
    {
        "date": "2015-08-11",
        "label": "Emerging-market FX volatility",
        "category": "Financial",
        "oil_impact": "China/EM demand worries pressured oil; USD moves amplified swings.",
        "geo_context": "PBOC FX adjustment; EM capital-flow stress.",
        "market_reaction": "Risk-off in EM; commodity complex correlated drawdown.",
        "fed_stance": "Lift-off debate; strong USD headwind to commodities.",
    },
    {
        "date": "2016-06-24",
        "label": "UK EU membership referendum outcome",
        "category": "Policy",
        "oil_impact": "Risk-off bid to USD; crude followed broad USD and demand sentiment.",
        "geo_context": "Brexit surprise; EU political uncertainty.",
        "market_reaction": "Sharp GBP move; global equity volatility spike.",
        "fed_stance": "Fed delayed hikes on global risk; flight to quality.",
    },
    {
        "date": "2018-10-10",
        "label": "Q4 global risk-off / rates pressure",
        "category": "Financial",
        "oil_impact": "Late-cycle growth fears; oil rolled over with cyclicals.",
        "geo_context": "Trade tensions; tightening financial conditions.",
        "market_reaction": "Tech and growth leadership cracked; VIX spike.",
        "fed_stance": "Hiking cycle nearing peak; QT ongoing.",
    },
    {
        "date": "2020-03-11",
        "label": "Global health emergency classification",
        "category": "Health",
        "oil_impact": "Demand shock narrative dominated; crude faced historic collapse into April.",
        "geo_context": "WHO pandemic declaration; mobility and travel outlook collapsed.",
        "market_reaction": "Cross-asset crash; credit stress; Fed backstop expectations.",
        "fed_stance": "Emergency cuts to zero; massive QE and liquidity programs.",
    },
    {
        "date": "2020-04-20",
        "label": "Energy futures dislocation",
        "category": "Energy",
        "oil_impact": "WTI front contract negative print; storage saturation & physical squeeze.",
        "geo_context": "Demand collapse + storage constraints at Cushing.",
        "market_reaction": "Historic energy volatility; ETF and derivative dislocations.",
        "fed_stance": "Deep accommodation; credit facilities for corporates/municipals.",
    },
    {
        "date": "2022-02-24",
        "label": "European regional security crisis",
        "category": "Conflict",
        "oil_impact": "Sharp risk premium; Brent spike; sanctions and rerouting of flows.",
        "geo_context": "Major military conflict in Europe; energy weaponization concerns.",
        "market_reaction": "Commodity surge; equity drawdown; volatility regime shift.",
        "fed_stance": "Inflation shock → front-loaded hiking cycle; USD strength.",
    },
    {
        "date": "2022-09-26",
        "label": "UK fiscal event — rates & FX volatility",
        "category": "Policy",
        "oil_impact": "USD strength and global rates repricing moved oil via macro channel.",
        "geo_context": "Gilt market stress; UK policy credibility shock.",
        "market_reaction": "GBP turmoil; pension/LDI stress; global risk-off.",
        "fed_stance": "Fed hiking; global financial tightening spillovers.",
    },
    {
        "date": "2023-03-10",
        "label": "US regional banking stress",
        "category": "Financial",
        "oil_impact": "Growth scare and USD moves; crude tracked risk appetite.",
        "geo_context": "SVB failure; deposit-flight fears; regional bank scrutiny.",
        "market_reaction": "Rate-cut pricing whipsaw; financials under pressure.",
        "fed_stance": "BTFP liquidity; later pause vs inflation trade-off.",
    },
    {
        "date": "2023-10-07",
        "label": "Middle East tensions escalation",
        "category": "Conflict",
        "oil_impact": "Geopolitical premium; tanker and supply-route risk reassessed.",
        "geo_context": "Major regional conflict risk; humanitarian and security crisis.",
        "market_reaction": "Energy equities volatile; safe-haven flows.",
        "fed_stance": "Focus on inflation pass-through; rates stay higher for longer narrative.",
    },
    {
        "date": "2024-08-05",
        "label": "Global risk-off / carry unwind",
        "category": "Financial",
        "oil_impact": "Recession fears and USD bid pressured crude; beta to equities high.",
        "geo_context": "Yen carry unwind; global growth scare and vol spike.",
        "market_reaction": "Sharp equity drawdown; VIX jump; cross-asset deleveraging.",
        "fed_stance": "Cut expectations repriced; policy path data-dependent.",
    },
]


def _load_raw_close(path: Path, name: str) -> pd.Series:
    s = pd.read_csv(path, skiprows=3, names=["date", name])
    s["date"] = pd.to_datetime(s["date"])
    return s.set_index("date")[name].astype(float).sort_index()


def _merge_prices() -> pd.DataFrame:
    brent = _load_raw_close(ROOT / "data/raw/brent.csv", "oil")
    spx = _load_raw_close(ROOT / "data/raw/sp500.csv", "spx")
    df = pd.concat([brent, spx], axis=1, join="inner")
    df = df.sort_index()
    return df


def _nearest_news_row(event_dt: pd.Timestamp, news: pd.DataFrame) -> pd.Series | None:
    if news.empty:
        return None
    news = news.copy()
    news["date"] = pd.to_datetime(news["date"])
    target = pd.Timestamp(event_dt).normalize()
    # CSV may contain duplicate dates — avoid Index.get_indexer (requires unique index).
    deltas = (news["date"].dt.normalize() - target).abs()
    pos = int(deltas.to_numpy().argmin())
    return news.iloc[pos]


def _plotly_selected_customdata(state_key: str) -> int | None:
    raw = st.session_state.get(state_key)
    if raw is None:
        return None
    sel = getattr(raw, "selection", None)
    if sel is None and isinstance(raw, dict):
        sel = raw.get("selection")
    if sel is None:
        return None
    pts = getattr(sel, "points", None)
    if pts is None and isinstance(sel, dict):
        pts = sel.get("points")
    if not pts:
        return None
    p0 = pts[0]
    if not isinstance(p0, dict):
        return None
    cd = p0.get("customdata")
    if cd is None:
        return None
    try:
        return int(np.asarray(cd).flatten()[0])
    except (TypeError, ValueError, IndexError):
        return None


def _make_geo_intel_figure(
    p: pd.DataFrame,
    ev_f: pd.DataFrame,
    *,
    sel_corr_windows: list[int],
    show_oil: bool,
    show_spx: bool,
    show_events: bool,
) -> go.Figure:
    """Single two-row figure so x-hover and spike lines stay synchronized across panels."""
    n_ev = len(ev_f)
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.11,
        row_heights=[0.6, 0.4],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
        subplot_titles=(
            "West Texas oil price vs. the S&P 500 (U.S. stock market)",
            "How closely oil and stocks moved together (rolling correlation)",
        ),
    )

    if show_oil:
        fig.add_trace(
            go.Scatter(
                x=p.index,
                y=p["oil"],
                name="Oil price (West Texas)",
                legendgroup="price",
                line=dict(color=C_OIL_LINE, width=1.65),
                hovertemplate="%{x|%Y-%m-%d}<br>Oil $%{y:.2f}/bbl<extra></extra>",
            ),
            row=1,
            col=1,
            secondary_y=False,
        )
    if show_spx:
        fig.add_trace(
            go.Scatter(
                x=p.index,
                y=p["spx"],
                name="S&P 500",
                legendgroup="price",
                line=dict(color=C_SPX_LINE, width=1.35),
                hovertemplate="%{x|%Y-%m-%d}<br>S&P 500 %{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
            secondary_y=True,
        )

    if show_events and n_ev > 0:
        for _, er in ev_f.iterrows():
            d = er["date"]
            if d < p.index.min() or d > p.index.max():
                continue
            fig.add_vline(
                x=d,
                line_width=1.25,
                line_dash="solid",
                line_color=C_VLINE_EVENT,
                opacity=1.0,
                row=1,
                col=1,
            )

    ev_f_reset = ev_f.reset_index(drop=True)
    markers_on_spx = show_spx and not show_oil
    ex: list[pd.Timestamp] = []
    ey: list[float] = []
    eidx: list[int] = []
    etext: list[str] = []
    for i, er in ev_f_reset.iterrows():
        d = er["date"]
        if d < p.index.min() or d > p.index.max():
            continue
        pos = int(p.index.searchsorted(d, side="right")) - 1
        pos = max(0, min(pos, len(p) - 1))
        ex.append(p.index[pos])
        ey.append(float(p["spx"].iloc[pos]) if markers_on_spx else float(p["oil"].iloc[pos]))
        eidx.append(i)
        lab = str(er["label"])
        etext.append(lab[:42] + "…" if len(lab) > 42 else lab)

    if show_events and n_ev > 0 and ex:
        _ht_tail = (
            "S&P 500 %{y:.2f}<extra></extra>"
            if markers_on_spx
            else "Oil $%{y:.2f}/bbl<extra></extra>"
        )
        fig.add_trace(
            go.Scatter(
                x=ex,
                y=ey,
                mode="markers",
                name="Key events (click)",
                legendgroup="events",
                marker=dict(
                    size=12,
                    symbol="diamond",
                    color=C_EVENT_BLUE,
                    line=dict(width=1.5, color=C_BG_PANEL),
                ),
                customdata=eidx,
                text=etext,
                hovertemplate="<b>%{text}</b><br>%{x|%Y-%m-%d}<br>" + _ht_tail,
            ),
            row=1,
            col=1,
            secondary_y=markers_on_spx,
        )

    wins_sorted = sorted(sel_corr_windows)
    fill_win: int | None = None
    if wins_sorted:
        fill_win = 90 if 90 in wins_sorted else wins_sorted[-1]
        y_fill = np.maximum(p[f"corr_{fill_win}"].values.astype(float), 0.0)
        fig.add_trace(
            go.Scatter(
                x=p.index,
                y=y_fill,
                mode="lines",
                line=dict(width=0),
                fill="tozeroy",
                fillcolor=C_FILL_POS,
                name="_pos_fill",
                showlegend=False,
                hoverinfo="skip",
                legendgroup="corr_fill",
            ),
            row=2,
            col=1,
        )

    for win in wins_sorted:
        col_dash = CORR_LINE_STYLE.get(win, (C_PRIMARY, "solid"))
        col, dash = col_dash
        fig.add_trace(
            go.Scatter(
                x=p.index,
                y=p[f"corr_{win}"],
                name=f"{win}-day correlation",
                legendgroup="corr",
                line=dict(color=col, width=1.25, dash=dash),
                hovertemplate=(
                    f"%{{x|%Y-%m-%d}}<br>{win}-day correlation: %{{y:.3f}}<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )

    _ytitle = dict(size=16, color=C_TEXT)
    _ytick = dict(size=14, color=C_TEXT)
    fig.update_yaxes(
        title_text="Oil (USD per barrel)",
        title_font=_ytitle,
        tickfont=_ytick,
        secondary_y=False,
        row=1,
        col=1,
        color=C_OIL_LINE,
    )
    fig.update_yaxes(
        title_text="S&P 500 (index level)",
        title_font=_ytitle,
        tickfont=_ytick,
        secondary_y=True,
        row=1,
        col=1,
        color=C_SPX_LINE,
    )
    fig.update_yaxes(
        title_text="Correlation (-1 to +1)",
        title_font=_ytitle,
        tickfont=_ytick,
        row=2,
        col=1,
        range=[-1, 1],
        color=C_PRIMARY,
    )

    fig.update_layout(
        height=820,
        template="plotly_white",
        paper_bgcolor=C_BG_PANEL,
        plot_bgcolor=C_PANEL,
        font=dict(color=C_TEXT, size=12),
        showlegend=False,
        margin=dict(l=64, r=28, t=108, b=52),
        hovermode="x unified",
        hoverdistance=-1,
        spikedistance=-1,
        uirevision="goi_intel",
    )
    fig.update_annotations(font=dict(size=15, color=C_PRIMARY))

    fig.update_xaxes(
        gridcolor=C_GRID_SOFT,
        zeroline=False,
        tickfont=dict(size=13, color=C_TEXT),
        showspikes=True,
        spikecolor=C_SPIKE,
        spikesnap="cursor",
        spikemode="across",
        spikethickness=1,
        spikedash="solid",
    )
    fig.update_yaxes(
        gridcolor=C_GRID_SOFT,
        zeroline=False,
        showspikes=False,
        secondary_y=False,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        gridcolor=C_GRID_SOFT,
        zeroline=False,
        showspikes=False,
        secondary_y=True,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        gridcolor=C_GRID_SOFT,
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor="rgba(0, 48, 135, 0.22)",
        showspikes=False,
        row=2,
        col=1,
    )

    if not show_oil and show_spx:
        fig.update_yaxes(
            visible=False,
            showticklabels=False,
            showline=False,
            secondary_y=False,
            row=1,
            col=1,
        )
    elif show_oil and not show_spx:
        fig.update_yaxes(
            visible=False,
            showticklabels=False,
            showline=False,
            secondary_y=True,
            row=1,
            col=1,
        )

    return fig


def _suite_css() -> str:
    return f"""
    <style>
      .stApp {{
        background: linear-gradient(180deg, {C_BG_DEEP} 0%, #e8ebf0 50%, {C_BG_DEEP} 100%);
        color: {C_TEXT};
      }}
      [data-testid="stHeader"] {{
        background-color: {C_BG_PANEL};
        border-bottom: 2px solid {C_CHASE};
      }}
      [data-testid="stSidebar"] {{
        background-color: {C_BG_PANEL} !important;
        border-right: 1px solid rgba(0, 48, 135, 0.15);
      }}
      [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {{
        color: {C_TEXT};
      }}
      .stMetric label {{ color: {C_MUTED} !important; }}
      .stMetric [data-testid="stMetricValue"] {{ color: {C_TEXT} !important; }}
      h1, h2, h3, h4 {{ color: {C_CHASE} !important; }}
      .goi-header {{
        background: {C_BG_PANEL};
        border-bottom: 2px solid {C_CHASE};
        padding: 1rem 0 1.1rem 0;
        margin: -1rem -4rem 1rem -4rem;
        padding-left: 2rem;
        padding-right: 2rem;
      }}
      .goi-title {{ color: {C_PRIMARY}; font-size: 2rem; font-weight: 700; margin: 0; line-height: 1.2; }}
      .goi-sub {{ color: {C_MUTED}; font-size: 0.95rem; margin-top: 0.35rem; }}
      .goi-kicker {{ color: {C_MUTED}; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; margin: 0 0 0.25rem 0; }}
      .goi-filter-panel {{
        background: {C_SOFT_CARD};
        border: 1px solid rgba(0, 48, 135, 0.14);
        border-radius: 12px;
        padding: 1rem 1rem 1.1rem 1rem;
        margin-bottom: 0.5rem;
      }}
      .goi-filter-panel h4 {{
        color: {C_PRIMARY} !important;
        font-size: 0.95rem;
        margin: 0 0 0.6rem 0;
        font-weight: 700;
      }}
      .goi-filter-panel .stMarkdown p {{ font-size: 0.82rem; color: {C_MUTED}; margin: 0.35rem 0 0 0; }}
      .goi-detail-panel {{
        background: {C_BG_PANEL};
        border: 1px solid rgba(0, 48, 135, 0.2);
        border-left: 4px solid {C_CHASE};
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-top: 0.75rem;
      }}
      .goi-detail-panel h4 {{
        color: {C_CHASE} !important;
        font-size: 0.85rem;
        margin: 0.5rem 0 0.25rem 0;
        text-transform: uppercase;
        letter-spacing: 0.04em;
      }}
      .goi-detail-panel p {{ margin: 0; color: {C_TEXT}; line-height: 1.45; }}
    </style>
    """


def main() -> None:
    try:
        st.set_page_config(
            page_title="Global Energy & Equity Monitor · GeoOil Intelligence",
            layout="wide",
        )
    except st.errors.StreamlitAPIException:
        pass

    st.markdown(_suite_css(), unsafe_allow_html=True)

    st.markdown(
        """
        <div class="goi-header">
          <p class="goi-kicker">GeoOil Intelligence</p>
          <p class="goi-title">Global Energy &amp; Equity Monitor</p>
          <p class="goi-sub">West Texas oil, the U.S. stock market, and major world events · 2003–2026</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Crude series: Brent continuous front month (BZ=F) — standard liquid proxy aligned with WTI dynamics."
    )

    try:
        prices = _merge_prices()
    except Exception as e:
        st.error(f"Could not load price data: {e}")
        return

    events_df = pd.DataFrame(GEO_EVENTS)
    events_df["date"] = pd.to_datetime(events_df["date"])

    y_data_min = int(prices.index.min().year)
    y_data_max = int(prices.index.max().year)
    y_lo_bound = min(2003, y_data_min)
    y_hi_bound = max(2026, y_data_max)

    st.sidebar.markdown("**Controls**")
    st.sidebar.markdown(f"Year range (data {y_data_min}–{y_data_max})")
    yr_lo, yr_hi = st.sidebar.slider(
        "Range",
        min_value=y_lo_bound,
        max_value=y_hi_bound,
        value=(max(2003, y_data_min), min(2026, y_data_max)),
    )

    cats = sorted(events_df["category"].unique())
    sel_cats = st.sidebar.multiselect("Event types", cats, default=cats)

    news_path = ROOT / "data/processed/news_signals.csv"
    news = pd.read_csv(news_path) if news_path.is_file() else pd.DataFrame()

    ev_f = events_df[events_df["category"].isin(sel_cats)].copy()
    ev_f = ev_f[(ev_f["date"].dt.year >= yr_lo) & (ev_f["date"].dt.year <= yr_hi)]

    p = prices[(prices.index.year >= yr_lo) & (prices.index.year <= yr_hi)].copy()
    if p.empty:
        st.warning("No overlapping price data for this range.")
        return

    ro = p["oil"].pct_change()
    rs = p["spx"].pct_change()
    for w in (30, 60, 90, 180):
        p[f"corr_{w}"] = ro.rolling(w).corr(rs)

    chart_key = "geo_intel_plotly"
    n_ev = len(ev_f)
    if n_ev > 0:
        if "event_idx" not in st.session_state:
            st.session_state.event_idx = 0
        st.session_state.event_idx = int(np.clip(st.session_state.event_idx, 0, n_ev - 1))
        picked = _plotly_selected_customdata(chart_key)
        if picked is not None and 0 <= picked < n_ev:
            st.session_state.event_idx = picked

    col_chart, col_filt = st.columns([3.15, 1.05], gap="large")

    with col_filt:
        st.markdown(
            '<div class="goi-filter-panel"><h4>Chart filters</h4>',
            unsafe_allow_html=True,
        )
        sel_wins = st.multiselect(
            "Rolling correlation windows",
            options=[30, 60, 90, 180],
            default=[30, 60, 90, 180],
            format_func=lambda x: f"{x} days",
            key="goi_corr_windows",
        )
        show_oil = st.checkbox("Oil price line", value=True, key="goi_show_oil")
        show_spx = st.checkbox("S&P 500 line", value=True, key="goi_show_spx")
        show_events = st.checkbox("Event markers & guide lines", value=True, key="goi_show_events")
        st.caption(
            "Colors: oil · gold · S&P 500 · navy · correlation · blue scale · events · blue diamonds."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    if not show_oil and not show_spx:
        st.warning("Select at least one price line (oil and/or S&P 500) to show the top chart.")
        st.stop()

    if not sel_wins:
        st.info("Pick at least one rolling correlation window to plot the lower panel.")

    fig = _make_geo_intel_figure(
        p,
        ev_f,
        sel_corr_windows=list(sel_wins),
        show_oil=show_oil,
        show_spx=show_spx,
        show_events=show_events,
    )

    with col_chart:
        try:
            _chart_box = st.container(border=True)
        except TypeError:
            _chart_box = st.container()
        with _chart_box:
            st.plotly_chart(
                fig,
                use_container_width=True,
                key=chart_key,
                on_select="rerun" if (n_ev > 0 and show_events) else "ignore",
                selection_mode="points",
            )

    # Analytics strip — latest rolling correlations (matches filter selection)
    last_i = -1
    _wins = sorted(sel_wins) if sel_wins else []
    if _wins:
        mcols = st.columns(len(_wins))
        for mi, w in enumerate(_wins):
            v = float(p[f"corr_{w}"].iloc[last_i]) if f"corr_{w}" in p.columns else float("nan")
            mcols[mi].metric(f"{w}-day correlation", f"{v:.3f}" if np.isfinite(v) else "—")

    if n_ev == 0:
        st.info("No events match the current filters — widen the year range or re-enable event types.")
        return

    labels = [f"{r['date'].strftime('%Y-%m-%d')} — {r['label']}" for _, r in ev_f.iterrows()]

    choice = st.selectbox(
        "Browse events (or click a diamond on the chart)",
        options=list(range(n_ev)),
        format_func=lambda i: labels[i],
        key="event_idx",
    )

    erow = ev_f.iloc[choice]
    ed = erow["date"]

    st.markdown("#### Event detail")
    row_news = _nearest_news_row(ed, news)

    d1, d2, d3, d4 = st.columns(4)
    if row_news is not None and not news.empty:
        d1.metric("Oil sentiment (nearest Mon.)", f"{row_news.get('oil_sentiment', float('nan')):.0f}")
        d2.metric("Geo risk", f"{row_news.get('geo_risk', float('nan')):.0f}")
        d3.metric("Market sentiment", f"{row_news.get('market_sentiment', float('nan')):.0f}")
        d4.metric("Fed signal", f"{row_news.get('fed_signal', float('nan')):.0f}")

    st.markdown(
        f"""
        <div class="goi-detail-panel">
          <h4>Oil impact signal</h4>
          <p>{erow["oil_impact"]}</p>
          <h4>Geopolitical context</h4>
          <p>{erow["geo_context"]}</p>
          <h4>Market reaction</h4>
          <p>{erow["market_reaction"]}</p>
          <h4>Federal Reserve stance</h4>
          <p>{erow["fed_stance"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if row_news is not None and not news.empty:
        summ = row_news.get("summary", "")
        if isinstance(summ, str) and summ.strip():
            snap_dt = pd.Timestamp(row_news["date"]).date()
            st.caption(f"Nearest Monday snapshot ({snap_dt}): {summ}")


if __name__ == "__main__":
    main()
