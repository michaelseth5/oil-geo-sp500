"""
Deprecated single-page demo. Use the branded router instead:

  streamlit run src/app.py

Pages: GeoOil Intel (timeline) + Monday Alpha (signal engine).
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from predict_monday import (  # noqa: E402
    classifier_predict_proba,
    load_classifier_bundle,
    load_regressor_bundle,
    predict_monday,
)


def main() -> None:
    st.set_page_config(page_title="Oil / SPX — Monday demo", layout="wide")
    st.title("Monday signal demo")
    st.caption("Classifier + regressor bundles · last row of `data/processed/features.csv`")

    feat_path = ROOT / "data/processed/features.csv"
    if not feat_path.is_file():
        st.error(f"Missing {feat_path}")
        return

    df = pd.read_csv(feat_path, index_col="date", parse_dates=True)
    last = df.iloc[[-1]]
    d = last.index[0]

    st.subheader("Latest row")
    st.write(f"**Date:** `{d.date()}`")

    try:
        cb = load_classifier_bundle()
        rb = load_regressor_bundle()
    except Exception as e:
        st.error(f"Could not load models: {e}")
        return

    proba = classifier_predict_proba(cb, last)[0]
    conf = float(max(proba))
    direction = "UP" if proba[1] > proba[0] else "DOWN"
    thresh = float(cb.get("threshold", 0.66))

    reg = rb["model"]
    rcols = rb["feature_columns"]
    pred_ret = float(reg.predict(last[rcols])[0])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Classifier max prob", f"{conf:.1%}")
    c2.metric("Direction (argmax)", direction)
    c3.metric("Bundle threshold", f"{thresh:.3f}")
    c4.metric("Predicted Mon return", f"{pred_ret:+.2%}")

    st.divider()
    st.subheader("Full combined output (same as `predict_monday.py`)")

    buf = io.StringIO()
    with redirect_stdout(buf):
        predict_monday(last)
    st.code(buf.getvalue() or "(no stdout)", language="text")

    buf2 = io.StringIO()
    with redirect_stdout(buf2):
        predict_monday(last, min_confidence=0.0)
    st.subheader("With confidence floor = 0 (always show direction + return)")
    st.code(buf2.getvalue(), language="text")

    with st.expander("Classifier P(down), P(up)"):
        st.json({"P(class 0)": float(proba[0]), "P(class 1)": float(proba[1])})

    with st.expander("Regressor feature snapshot"):
        st.dataframe(last[rcols].T, use_container_width=True)


if __name__ == "__main__":
    main()
