"""Streamlit research audit: predictions, conditional coverage, temporal drift, and provenance."""
from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import streamlit as st
from research_validation import (selected_metrics,feature_drift,dataset_manifest,publication_audit)

ROOT=Path(__file__).resolve().parent.parent
st.title("Research Diagnostics")
st.caption("Walk-forward research validation · not a live trading signal or a guaranteed return.")
file=ROOT/"reports/walk_forward.json"
features=ROOT/"data/processed/features.csv"
if not file.is_file():
    st.warning("No walk-forward run found. Run: python -m src.research_validation --output reports/walk_forward.json")
else:
    run=json.loads(file.read_text())
    pred=pd.DataFrame(run["predictions"])
    threshold=st.slider("Minimum confidence",0.50,0.95,float(run["spec"]["threshold"]),0.01)
    m=selected_metrics(pred.actual.to_numpy(),pred.prob_up.to_numpy(),threshold)
    a,b,c=st.columns(3)
    a.metric("Selected accuracy","N/A" if m["accuracy_selected"] is None else f'{m["accuracy_selected"]:.1%}')
    b.metric("Coverage",f'{m["coverage"]:.1%}')
    c.metric("Selected observations",f'{m["selected"]:,} / {m["observations"]:,}')
    st.caption("Changing the threshold here is exploratory; changing it after viewing holdout results is not an independent test.")
    pred["date"]=pd.to_datetime(pred["date"])
    pred["correct"]=((pred.prob_up>=0.5).astype(int)==pred.actual).astype(int)
    st.plotly_chart(px.scatter(pred,x="date",y="prob_up",color="selected",
                               title="Out-of-sample directional probability by observation date"),use_container_width=True)
    folds=pd.DataFrame(run["folds"])
    st.plotly_chart(px.line(folds,x="test_end",y=["accuracy_selected","coverage"],
                            title="Fold-level accuracy and coverage"),use_container_width=True)
    with st.expander("Model specification and fold table"):
        st.json(run["spec"])
        st.dataframe(folds,use_container_width=True)
    st.download_button("Download out-of-sample predictions",pred.to_csv(index=False),"walk_forward_predictions.csv","text/csv")
st.subheader("Data provenance")
if features.is_file():
    st.json(dataset_manifest(features))
    df=pd.read_csv(features,index_col="date",parse_dates=True).sort_index()
    numeric=[c for c in df.columns if c not in ("target","next_monday_return") and pd.api.types.is_numeric_dtype(df[c])]
    if len(df)>200:
        split=int(len(df)*0.75)
        drift=feature_drift(df.iloc[:split],df.iloc[split:],numeric)
        st.caption("Population Stability Index (PSI): reference=earliest 75%, current=latest 25%. Research-only flag ≥0.2.")
        st.dataframe(pd.DataFrame(drift).head(30),use_container_width=True)
    st.caption("Current features.csv stores Monday observations, but its target was generated from the next trading day's return (usually Tuesday). Its name is not proof of predicting Monday returns.")
st.subheader("News leakage audit")
st.info("The existing news_signals.csv does not establish article-level publication timestamps and prediction cutoffs. Historical LLM scores are therefore unverified as point-in-time features.")
with st.expander("Audit publication timestamps (upload CSV)"):
    upload=st.file_uploader("CSV with published_at and prediction_cutoff",type="csv")
    if upload is not None:
        table=pd.read_csv(upload)
        try:st.json(publication_audit(table))
        except ValueError as exc:st.error(str(exc))
