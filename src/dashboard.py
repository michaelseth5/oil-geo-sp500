"""
Monday Alpha · Oil-Geo Signal Engine — production-grade fintech dashboard (Streamlit + Plotly).

Run: streamlit run src/dashboard.py  |  streamlit run src/app.py (Monday Alpha page)
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from predict_monday import (  # noqa: E402
    classifier_predict_proba,
    load_classifier_bundle,
    load_regressor_bundle,
)

# ---------------------------------------------------------------------------
# Design tokens — Webull × Chime (semantic colors only)
# ---------------------------------------------------------------------------
C_BG = "#1a1d27"
C_PANEL = "#131722"
C_BORDER = "#2a2d3a"
C_POS = "#02c076"
C_NEG = "#f6465d"
C_INFO = "#00bcd4"
C_WARN = "#f0b90b"
C_MUTED = "#787b86"
C_TEXT = "#d1d4dc"
C_NEUTRAL_CELL = "#5a6578"
C_TITLE = "#ffffff"

# Vertical rhythm (px-equivalent via rem/spacing)
SPACE_SECTION = "1.35rem"
SPACE_BLOCK = "1.1rem"
SPACE_GRID = "1.25rem"

# Plotly shared geometry
PLOT_MARGIN_DEFAULT = dict(l=44, r=24, t=36, b=44)
PLOT_FONT = dict(family="Arial, Helvetica, sans-serif", color=C_TEXT, size=12)

# Chart heights (explicit — prevents squished layouts)
H_DONUT = 360
H_METER = 360
H_FEATURE = 400
H_HEAT = 320
H_PRED = 320
H_GAUGE = 290

# Spec snapshot (KPI / donut / heatmaps)
KPI_DATE = "2026-03-23"
KPI_P_UP = 0.639
KPI_P_DOWN = 0.361
KPI_CONF_PCT = 63.9
KPI_THRESH_PCT = 66.0
KPI_PRED_RET = -0.0152
KPI_DIRECTION_UP = True
KPI_SIGNAL = "MODERATE"

FI_ROWS: list[tuple[str, float]] = [
    ("news_market_sentiment", 0.123),
    ("oil_sp500_corr_x_sp500_wk", 0.111),
    ("gold_oil_corr_90d_lag2m", 0.109),
    ("vix_level", 0.108),
    ("brent_sp500_rollcov20", 0.102),
    ("sp500_1m_momentum", 0.094),
    ("news_confidence", 0.085),
    ("news_oil_sentiment", 0.075),
    ("oil_sp500_corr_90d", 0.067),
    ("sp500_realvol_60d", 0.059),
]

CORR_ROW = [(-0.18, "30d"), (0.12, "60d"), (0.34, "90d"), (0.51, "180d")]

DEFAULT_NEWS_SUMMARY = (
    "Escalating regional tensions and energy market volatility overshadowed stabilizing Fed signals, "
    "driving risk-off sentiment ahead of Monday open."
)
DISPLAY_MODEL_NAME = "llama-3.3-70b-versatile"


def _layout_base(*, height: int, margin: dict | None = None) -> dict:
    m = {**PLOT_MARGIN_DEFAULT, **(margin or {})}
    return dict(
        template="plotly_dark",
        paper_bgcolor=C_PANEL,
        plot_bgcolor=C_BG,
        font=PLOT_FONT,
        margin=m,
        height=height,
        autosize=True,
    )


def _section_title(text: str) -> str:
    return (
        f'<p class="ma-section">{text}</p>'
        f'<div class="ma-rule" aria-hidden="true"></div>'
    )


def _blend_proba_batch(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    cols = bundle["feature_columns"]
    m = bundle["model"]
    Xc = X[cols]
    if isinstance(m, tuple) and m[0] == "blend":
        _, xgb_m, rf_m, w = m
        px = xgb_m.predict_proba(Xc)
        pr = rf_m.predict_proba(Xc)
        b = float(w) * px + (1.0 - float(w)) * pr
        s = b.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        return b / s
    return m.predict_proba(Xc)


def _global_css() -> str:
    return f"""
    <style>
      .stApp {{ background-color: {C_BG} !important; }}
      .block-container {{ padding-top: 1.25rem; max-width: 1400px; }}

      .ma-h1 {{
        color: {C_TITLE};
        font-size: 1.65rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        margin: 0 0 0.4rem 0;
        line-height: 1.2;
      }}
      .ma-sub {{
        color: {C_MUTED};
        font-size: 0.95rem;
        font-weight: 400;
        margin: 0 0 {SPACE_BLOCK};
        letter-spacing: 0.01em;
      }}
      .ma-section {{
        color: {C_MUTED};
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 600;
        margin: {SPACE_SECTION} 0 0.5rem 0;
      }}
      .ma-rule {{
        height: 1px;
        background: {C_BORDER};
        margin-bottom: 0.85rem;
        opacity: 0.85;
      }}
      .ma-spacer {{ height: 1.15rem; }}

      .kpi-wrap {{
        background: {C_PANEL};
        border: 1px solid {C_BORDER};
        border-radius: 10px;
        padding: 1rem 1.15rem;
        min-height: 96px;
        box-shadow: 0 1px 0 rgba(255,255,255,0.04) inset, 0 8px 24px rgba(0,0,0,0.22);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        align-items: flex-start;
        transition: border-color 0.15s ease, box-shadow 0.15s ease;
      }}
      .kpi-wrap:hover {{
        border-color: rgba(0, 188, 212, 0.35);
        box-shadow: 0 1px 0 rgba(255,255,255,0.05) inset, 0 10px 28px rgba(0,0,0,0.28);
      }}
      .kpi-label {{
        color: {C_MUTED};
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
        margin: 0;
      }}
      .kpi-val {{
        font-size: 1.28rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        line-height: 1.15;
        margin: auto 0 0 0;
        width: 100%;
      }}

      .news-card {{
        background: {C_PANEL};
        border: 1px solid {C_BORDER};
        border-left: 3px solid {C_WARN};
        border-radius: 10px;
        padding: 1.35rem 1.5rem;
        color: {C_MUTED};
        font-style: italic;
        font-size: 0.98rem;
        line-height: 1.65;
        margin-bottom: 1rem;
      }}
      .pill-row {{ display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem 0.85rem; margin-bottom: 0.35rem; }}
      .pill-label {{ color: {C_MUTED}; font-size: 0.78rem; font-weight: 500; min-width: 2.5rem; }}
      .pill {{
        display: inline-flex;
        align-items: center;
        padding: 0.35rem 0.85rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.82rem;
        border: 1px solid transparent;
      }}
      .pill-up {{ background: rgba(2, 192, 118, 0.12); color: {C_POS}; border-color: rgba(2, 192, 118, 0.25); }}
      .pill-down {{ background: rgba(246, 70, 93, 0.1); color: {C_NEG}; border-color: rgba(246, 70, 93, 0.22); }}
      .pill-zero {{ background: rgba(120, 123, 134, 0.12); color: {C_MUTED}; border-color: {C_BORDER}; }}

      .ma-news-head {{
        color: {C_TEXT};
        font-size: 0.95rem;
        font-weight: 600;
        letter-spacing: -0.01em;
        margin: 0 0 0.65rem 0;
      }}
      .meta-grid {{
        display: grid;
        grid-template-columns: auto 1fr;
        gap: 0.35rem 1rem;
        margin-top: 0.9rem;
        font-size: 0.82rem;
        color: {C_MUTED};
      }}
      .meta-k {{ color: {C_MUTED}; }}
      .meta-v {{ color: {C_TEXT}; }}
      .meta-v-cyan {{ color: {C_INFO}; font-weight: 600; }}

      div[data-testid="stVerticalBlock"] > div button[kind="secondary"] {{
        background: {C_PANEL} !important;
        color: {C_TEXT} !important;
        border: 1px solid {C_BORDER} !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        padding: 0.45rem 1.1rem !important;
        border-radius: 8px !important;
      }}
      div[data-testid="stVerticalBlock"] > div button[kind="secondary"]:hover {{
        border-color: rgba(0, 188, 212, 0.45) !important;
        color: {C_INFO} !important;
      }}
    </style>
    """


def _fig_donut() -> go.Figure:
    fig = go.Figure(
        data=[
            go.Pie(
                labels=["P(UP)", "P(DOWN)"],
                values=[KPI_P_UP, KPI_P_DOWN],
                hole=0.62,
                marker=dict(
                    colors=[
                        "rgba(2, 192, 118, 0.92)",
                        "rgba(246, 70, 93, 0.88)",
                    ],
                    line=dict(color=C_PANEL, width=2),
                ),
                textinfo="percent",
                textposition="outside",
                textfont=dict(color=C_MUTED, size=12),
                insidetextfont=dict(color=C_TEXT, size=11),
                sort=False,
                direction="clockwise",
                hovertemplate="<b>%{label}</b><br>%{percent:.1%}<extra></extra>",
            )
        ]
    )
    fig.add_annotation(
        text=(
            f"<span style='font-size:32px;font-weight:700;color:{C_TITLE};letter-spacing:-0.03em'>"
            f"{KPI_CONF_PCT:.1f}%</span>"
        ),
        x=0.5,
        y=0.54,
        xref="paper",
        yref="paper",
        showarrow=False,
    )
    fig.add_annotation(
        text=f"<span style='font-size:11px;color:{C_NEG};font-weight:500'>Below threshold</span>",
        x=0.5,
        y=0.36,
        xref="paper",
        yref="paper",
        showarrow=False,
    )
    fig.update_layout(
        **_layout_base(height=H_DONUT, margin=dict(l=48, r=48, t=40, b=52)),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.08,
            x=0.5,
            xanchor="center",
            font=dict(color=C_MUTED, size=11),
            bgcolor="rgba(0,0,0,0)",
            itemwidth=30,
        ),
        title=dict(
            text="Class probability",
            font=dict(size=13, color=C_TEXT, family=PLOT_FONT["family"]),
            x=0,
            xanchor="left",
            pad=dict(t=0, b=8),
        ),
    )
    return fig


def _fig_confidence_meter() -> go.Figure:
    z_red = "rgba(246, 70, 93, 0.42)"
    z_yel = "rgba(240, 185, 11, 0.38)"
    z_grn = "rgba(2, 192, 118, 0.4)"
    w_bar = 0.52
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=[""],
            x=[50],
            orientation="h",
            marker=dict(color=z_red),
            showlegend=False,
            hoverinfo="skip",
            width=w_bar,
        )
    )
    fig.add_trace(
        go.Bar(
            y=[""],
            x=[16],
            orientation="h",
            base=[50],
            marker=dict(color=z_yel),
            showlegend=False,
            hoverinfo="skip",
            width=w_bar,
        )
    )
    fig.add_trace(
        go.Bar(
            y=[""],
            x=[34],
            orientation="h",
            base=[66],
            marker=dict(color=z_grn),
            showlegend=False,
            hoverinfo="skip",
            width=w_bar,
        )
    )
    fig.update_layout(
        barmode="stack",
        yaxis=dict(visible=False, range=[-0.55, 0.55], fixedrange=True),
        xaxis=dict(visible=True, zeroline=False),
    )
    cx, tx = KPI_CONF_PCT, KPI_THRESH_PCT
    fig.add_shape(
        type="line",
        x0=cx,
        x1=cx,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="#ffffff", width=2),
        layer="above",
    )
    fig.add_shape(
        type="line",
        x0=tx,
        x1=tx,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color=C_WARN, width=2, dash="4 3"),
        layer="above",
    )
    fig.add_annotation(
        x=cx,
        y=1.06,
        yref="paper",
        text=f"Current · {cx:.1f}%",
        showarrow=False,
        font=dict(color=C_TEXT, size=11, family=PLOT_FONT["family"]),
    )
    fig.add_annotation(
        x=tx,
        y=-0.1,
        yref="paper",
        text=f"Threshold · {tx:.0f}%",
        showarrow=False,
        font=dict(color=C_WARN, size=11, family=PLOT_FONT["family"]),
    )
    fig.update_xaxes(
        range=[0, 100],
        dtick=10,
        tickfont=dict(color=C_MUTED, size=11),
        gridcolor="rgba(42,45,58,0.5)",
        showgrid=True,
        title="",
    )
    fig.update_layout(
        **_layout_base(height=H_METER, margin=dict(l=44, r=44, t=48, b=56)),
        title=dict(
            text="Confidence meter",
            font=dict(size=13, color=C_TEXT, family=PLOT_FONT["family"]),
            x=0,
            xanchor="left",
            pad=dict(t=0, b=10),
        ),
    )
    return fig


def _fi_gradient_color(t: float) -> str:
    """Low → high: muted cyan → muted green (slightly desaturated for less noise)."""
    lo = np.array([26, 152, 172], dtype=float)
    hi = np.array([24, 168, 108], dtype=float)
    rgb = lo + t * (hi - lo)
    return f"rgb({int(rgb[0])},{int(rgb[1])},{int(rgb[2])})"


def _fig_feature_importance() -> go.Figure:
    rows = sorted(FI_ROWS, key=lambda x: x[1], reverse=True)
    names = [r[0] for r in rows]
    vals = [r[1] * 100 for r in rows]
    vmin, vmax = min(vals), max(vals)
    span = max(vmax - vmin, 1e-9)
    colors = [_fi_gradient_color((v - vmin) / span) for v in vals]
    fig = go.Figure(
        go.Bar(
            x=vals,
            y=names,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v:.1f}%" for v in vals],
            textposition="outside",
            textfont=dict(color=C_TEXT, size=11),
            width=0.72,
        )
    )
    fig.update_yaxes(
        autorange="reversed",
        tickfont=dict(color=C_MUTED, size=11),
        dtick=1,
        showgrid=False,
    )
    fig.update_xaxes(
        title=dict(text="Importance (%)", font=dict(color=C_MUTED, size=11)),
        gridcolor="rgba(42,45,58,0.45)",
        zeroline=False,
        tickfont=dict(color=C_MUTED, size=10),
    )
    fig.update_layout(
        **_layout_base(height=H_FEATURE, margin=dict(l=8, r=56, t=40, b=48)),
        bargap=0.35,
        title=dict(
            text="Classifier feature importance",
            font=dict(size=13, color=C_TEXT, family=PLOT_FONT["family"]),
            x=0,
            xanchor="left",
            pad=dict(t=0, b=8),
        ),
    )
    return fig


def _fig_news_heatmap() -> go.Figure:
    z = [[1, 1], [-1, 0]]
    text = [["Oil\n+1", "Geo Risk\n+1"], ["Market\n−1", "Fed\n0"]]
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            colorscale=[
                [0.0, C_NEG],
                [0.5, C_NEUTRAL_CELL],
                [1.0, C_POS],
            ],
            zmin=-1,
            zmax=1,
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=20, color=C_TEXT, family=PLOT_FONT["family"]),
            showscale=False,
            hovertemplate="%{text}<extra></extra>",
            xgap=10,
            ygap=10,
        )
    )
    fig.update_xaxes(visible=False, showgrid=False, zeroline=False)
    fig.update_yaxes(visible=False, autorange="reversed", showgrid=False, zeroline=False)
    fig.update_layout(
        **_layout_base(height=H_HEAT, margin=dict(l=28, r=28, t=44, b=28)),
        title=dict(
            text=f"News sentiment signals · {KPI_DATE}",
            font=dict(size=13, color=C_TEXT, family=PLOT_FONT["family"]),
            x=0,
            xanchor="left",
            pad=dict(t=0, b=12),
        ),
    )
    return fig


def _fig_corr_heatmap() -> go.Figure:
    vals = [v for v, _ in CORR_ROW]
    labs = [lb for _, lb in CORR_ROW]
    z = [vals]
    text = [[f"{v:+.2f}" for v in vals]]
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=labs,
            y=[""],
            colorscale=[
                [0.0, C_NEG],
                [0.5, C_NEUTRAL_CELL],
                [1.0, C_POS],
            ],
            zmin=-1,
            zmax=1,
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=22, color=C_TEXT, family=PLOT_FONT["family"]),
            showscale=False,
            hovertemplate="%{x}<br>ρ = %{z:.2f}<extra></extra>",
            xgap=12,
            ygap=2,
        )
    )
    fig.update_xaxes(
        side="bottom",
        tickfont=dict(color=C_MUTED, size=11),
        showgrid=False,
        showline=False,
    )
    fig.update_yaxes(visible=False, showgrid=False)
    fig.update_layout(
        **_layout_base(height=H_HEAT, margin=dict(l=32, r=32, t=44, b=40)),
        title=dict(
            text="Oil · SP500 rolling correlation",
            font=dict(size=13, color=C_TEXT, family=PLOT_FONT["family"]),
            x=0,
            xanchor="left",
            pad=dict(t=0, b=12),
        ),
    )
    return fig


def _fig_pred_vs_actual(
    dates: pd.DatetimeIndex,
    actual_pct: np.ndarray,
    pred_pct: np.ndarray,
    confident: np.ndarray,
) -> go.Figure:
    pos_rgba = "rgba(2, 192, 118, 0.72)"
    neg_rgba = "rgba(246, 70, 93, 0.72)"
    colors = [pos_rgba if a >= 0 else neg_rgba for a in actual_pct]
    pred_arr = np.asarray(pred_pct, dtype=float)
    conf_mask = np.asarray(confident, dtype=bool)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=dates,
            y=actual_pct,
            name="Actual",
            marker=dict(color=colors, line=dict(width=0)),
            width=0.65,
            hovertemplate="%{x|%Y-%m-%d}<br>actual %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=pred_arr,
            name="Predicted",
            mode="lines",
            line=dict(color="rgba(240, 185, 11, 0.22)", width=7),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=pred_arr,
            name="Predicted",
            mode="lines+markers",
            line=dict(color=C_WARN, width=2.5),
            marker=dict(size=4, color=C_WARN, line=dict(width=0)),
            hovertemplate="%{x|%Y-%m-%d}<br>pred %{y:.2f}%<extra></extra>",
        )
    )
    if conf_mask.any():
        fig.add_trace(
            go.Scatter(
                x=dates[conf_mask],
                y=pred_arr[conf_mask],
                mode="markers",
                name="High confidence",
                marker=dict(
                    size=9,
                    color="rgba(0,0,0,0)",
                    symbol="circle",
                    line=dict(width=2, color=C_INFO),
                ),
                hovertemplate="%{x|%Y-%m-%d}<br>high confidence · pred %{y:.2f}%<extra></extra>",
            )
        )
    fig.update_xaxes(
        tickangle=32,
        gridcolor="rgba(42,45,58,0.4)",
        tickfont=dict(color=C_MUTED, size=10),
        showline=False,
    )
    fig.update_yaxes(
        title=dict(text="Return (%)", font=dict(color=C_MUTED, size=11)),
        gridcolor="rgba(42,45,58,0.4)",
        zerolinecolor=C_BORDER,
        showline=False,
    )
    fig.update_layout(
        **_layout_base(height=H_PRED, margin=dict(l=48, r=28, t=40, b=48)),
        barmode="overlay",
        bargap=0.3,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            x=0,
            font=dict(size=11, color=C_MUTED),
            bgcolor="rgba(0,0,0,0)",
        ),
        title=dict(
            text="Predicted vs Actual Monday returns · Test set 2022–2024",
            font=dict(size=13, color=C_TEXT, family=PLOT_FONT["family"]),
            x=0,
            xanchor="left",
            pad=dict(t=0, b=8),
        ),
    )
    return fig


def _gauge_spec(color: str, *, is_frac: bool = False) -> dict:
    axis = (
        dict(range=[0, 1], tickformat=".2f", tickwidth=0, tickcolor=C_MUTED, nticks=5)
        if is_frac
        else dict(range=[0, 100], tickwidth=0, tickcolor=C_MUTED, nticks=5)
    )
    return dict(
        axis=axis,
        bar=dict(color=color, thickness=0.55, line=dict(width=0)),
        bgcolor=C_BG,
        borderwidth=0,
    )


def _gauges_row() -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=4,
        specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]],
        horizontal_spacing=0.12,
    )
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=47.17,
            number=dict(suffix="%", font=dict(color=C_TEXT, size=20)),
            title=dict(text="Overall accuracy", font=dict(size=11, color=C_MUTED)),
            gauge=_gauge_spec(C_NEG),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=69.23,
            number=dict(suffix="%", font=dict(color=C_TEXT, size=20)),
            title=dict(text="Confident accuracy", font=dict(size=11, color=C_MUTED)),
            gauge=_gauge_spec(C_POS),
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=24.5,
            number=dict(suffix="%", font=dict(color=C_INFO, size=20)),
            title=dict(text="Coverage (26 / 106)", font=dict(size=11, color=C_MUTED)),
            gauge=_gauge_spec(C_INFO),
        ),
        row=1,
        col=3,
    )
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=0.66,
            number=dict(valueformat=".3f", font=dict(color=C_WARN, size=20)),
            title=dict(text="Threshold", font=dict(size=11, color=C_MUTED)),
            gauge=_gauge_spec(C_WARN, is_frac=True),
        ),
        row=1,
        col=4,
    )
    fig.update_layout(
        **_layout_base(height=H_GAUGE, margin=dict(l=24, r=24, t=48, b=36)),
    )
    return fig


def _load_pred_actual_series(
    n: int = 20,
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray, float] | None:
    feat_path = ROOT / "data/processed/features.csv"
    if not feat_path.is_file():
        return None
    df = pd.read_csv(feat_path, index_col="date", parse_dates=True)
    if df.empty or "next_monday_return" not in df.columns:
        return None
    try:
        cb = load_classifier_bundle()
        rb = load_regressor_bundle()
    except Exception:
        return None
    test_m = (df.index.year >= 2022) & (df.index.year <= 2024)
    df_t = df.loc[test_m].copy()
    if len(df_t) < 2:
        return None
    sub = df_t.iloc[-n:].copy()
    pb = _blend_proba_batch(cb, sub)
    prob_max = np.max(pb, axis=1)
    if hasattr(prob_max, "values"):
        conf = prob_max.values
    else:
        conf = np.asarray(prob_max)
    conf = np.asarray(conf, dtype=float).ravel()
    thresh = float(cb.get("threshold", 0.66))
    reg = rb["model"]
    rcols = rb["feature_columns"]
    pred = reg.predict(sub[rcols]) * 100.0
    act = sub["next_monday_return"].values * 100.0
    high_conf = conf >= thresh
    return sub.index, act, pred, high_conf, thresh


def main() -> None:
    try:
        st.set_page_config(
            page_title="Monday Alpha · Oil-Geo Signal Engine",
            layout="wide",
        )
    except st.errors.StreamlitAPIException:
        pass

    st.markdown(_global_css(), unsafe_allow_html=True)

    st.markdown(
        f"""
        <p class="ma-h1">Monday Alpha · Oil-Geo Signal Engine</p>
        <p class="ma-sub">XGBoost + Llama Intelligence · Monday Return Forecasting</p>
        <div class="ma-spacer"></div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(_section_title("Latest signal snapshot"), unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6 = st.columns(6, gap="medium")
    kpi_specs = [
        (c1, "Date", KPI_DATE, C_TEXT),
        (c2, "Direction", "▲ UP" if KPI_DIRECTION_UP else "▼ DOWN", C_POS if KPI_DIRECTION_UP else C_NEG),
        (c3, "Confidence", f"{KPI_CONF_PCT:.1f}%", C_NEG),
        (c4, "Predicted return", f"{KPI_PRED_RET * 100:.2f}%", C_NEG),
        (c5, "Threshold", f"{KPI_THRESH_PCT:.1f}%", C_WARN),
        (c6, "Signal", KPI_SIGNAL, C_WARN),
    ]
    for col, label, val, colr in kpi_specs:
        col.markdown(
            f"""
            <div class="kpi-wrap">
              <div class="kpi-label">{label}</div>
              <div class="kpi-val" style="color:{colr};">{val}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="ma-spacer"></div>', unsafe_allow_html=True)

    st.markdown(_section_title("Probability & confidence"), unsafe_allow_html=True)
    left, right = st.columns([0.4, 0.6], gap="large")
    with left:
        st.plotly_chart(_fig_donut(), use_container_width=True, key="ma_donut")
    with right:
        st.plotly_chart(_fig_confidence_meter(), use_container_width=True, key="ma_meter")

    st.markdown(_section_title("Model diagnostics"), unsafe_allow_html=True)
    st.plotly_chart(_fig_feature_importance(), use_container_width=True, key="ma_fi")

    st.markdown(_section_title("Signal heatmaps"), unsafe_allow_html=True)
    h1, h2 = st.columns(2, gap="large")
    with h1:
        st.plotly_chart(_fig_news_heatmap(), use_container_width=True, key="ma_hm_news")
    with h2:
        st.plotly_chart(_fig_corr_heatmap(), use_container_width=True, key="ma_hm_corr")

    st.markdown(_section_title("Out-of-sample performance"), unsafe_allow_html=True)
    pa = _load_pred_actual_series(20)
    if pa is not None:
        dts, act, pred, hconf, _ = pa
        st.plotly_chart(
            _fig_pred_vs_actual(dts, act, pred, hconf),
            use_container_width=True,
            key="ma_pv",
        )
    else:
        st.warning("Could not load test-set predictions (check features.csv and models).")

    st.markdown(_section_title("Backtest summary"), unsafe_allow_html=True)
    st.plotly_chart(_gauges_row(), use_container_width=True, key="ma_gauges")

    st.markdown(_section_title("News intelligence"), unsafe_allow_html=True)
    st.markdown('<p class="ma-news-head">Llama News Intelligence · Latest Monday</p>', unsafe_allow_html=True)

    if "ma_news_summary" not in st.session_state:
        st.session_state.ma_news_summary = DEFAULT_NEWS_SUMMARY
    if "ma_news_oil" not in st.session_state:
        st.session_state.ma_news_oil = 1
    if "ma_news_geo" not in st.session_state:
        st.session_state.ma_news_geo = 1
    if "ma_news_mkt" not in st.session_state:
        st.session_state.ma_news_mkt = -1
    if "ma_news_fed" not in st.session_state:
        st.session_state.ma_news_fed = 0
    if "ma_news_conf" not in st.session_state:
        st.session_state.ma_news_conf = 0.70
    if "ma_news_model" not in st.session_state:
        st.session_state.ma_news_model = DISPLAY_MODEL_NAME

    st.markdown(
        f'<div class="news-card">{st.session_state.ma_news_summary}</div>',
        unsafe_allow_html=True,
    )

    def pill(v: int) -> str:
        cls = "pill-up" if v > 0 else "pill-down" if v < 0 else "pill-zero"
        disp = f"+{v}" if v > 0 else str(v)
        return f'<span class="pill {cls}">{disp}</span>'

    st.markdown(
        f"""
        <div class="pill-row">
          <span class="pill-label">Oil</span>{pill(st.session_state.ma_news_oil)}
          <span class="pill-label">Geo</span>{pill(st.session_state.ma_news_geo)}
          <span class="pill-label">Market</span>{pill(st.session_state.ma_news_mkt)}
          <span class="pill-label">Fed</span>{pill(st.session_state.ma_news_fed)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Refresh News Analysis", type="secondary", key="ma_refresh_news"):
        try:
            from news_agent import get_news_signals  # noqa: WPS433

            today = date.today().isoformat()
            out = get_news_signals(today)
            st.session_state.ma_news_summary = str(out.get("summary", DEFAULT_NEWS_SUMMARY))
            st.session_state.ma_news_oil = int(out.get("oil_sentiment", 0))
            st.session_state.ma_news_geo = int(out.get("geo_risk", 0))
            st.session_state.ma_news_mkt = int(out.get("market_sentiment", 0))
            st.session_state.ma_news_fed = int(out.get("fed_signal", 0))
            st.session_state.ma_news_conf = float(out.get("confidence", 0.7))
            import os

            st.session_state.ma_news_model = os.environ.get("OPENROUTER_MODEL", DISPLAY_MODEL_NAME)
            st.rerun()
        except Exception as e:
            st.error(f"News refresh failed: {e}")

    st.markdown(
        f"""
        <div class="meta-grid">
          <span class="meta-k">Llama confidence</span>
          <span class="meta-v-cyan">{st.session_state.ma_news_conf:.2f}</span>
          <span class="meta-k">Model</span>
          <span class="meta-v">{st.session_state.ma_news_model}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
