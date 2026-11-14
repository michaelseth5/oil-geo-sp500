"""Leakage-aware, expanding-window walk-forward research for Monday-labelled data.

Prediction target is whatever is present in features.csv ('target'). Existing feature
engineering labels Tuesday direction using Monday closes; do NOT call it Monday returns.
All feature choices, blend weights, and the 0.66 threshold are fixed before evaluation.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

FEATURE_EXCLUSIONS = {"target", "next_monday_return"}
DEFAULT_THRESHOLD = 0.66
DEFAULT_XGB_WEIGHT = 0.58

def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c not in FEATURE_EXCLUSIONS
            and pd.api.types.is_numeric_dtype(df[c])]
    if not cols: raise ValueError("No numeric features")
    return cols

def assert_research_frame(df: pd.DataFrame) -> pd.DataFrame:
    if "target" not in df: raise ValueError("Missing target")
    if not isinstance(df.index, pd.DatetimeIndex): raise ValueError("Expected DatetimeIndex")
    if not df.index.is_monotonic_increasing or df.index.has_duplicates:
        raise ValueError("Dates must be sorted and unique")
    if len(df)<100:raise ValueError("Need at least 100 labeled observations")
    if df["target"].isna().any() or not set(df["target"].unique()).issubset({0,1}):
        raise ValueError("Target must be binary")
    cols=feature_columns(df)
    if not np.isfinite(df[cols].to_numpy(dtype=float)).all():
        raise ValueError("Missing or non-finite features; clean inputs before testing")
    return df

def selected_metrics(y_true, proba, threshold=DEFAULT_THRESHOLD):
    if not 0.5<=threshold<=1:raise ValueError("Threshold must be between 0.5 and 1")
    y=np.asarray(y_true,dtype=int)
    p=np.asarray(proba,dtype=float)
    if len(y)!=len(p) or p.ndim!=1 or np.any(~np.isfinite(p)) or np.any((p<0)|(p>1)):
        raise ValueError("Invalid probability and target alignment")
    confidence=np.maximum(p,1-p)
    selected=confidence>=threshold
    prediction=(p>=0.5).astype(int)
    n=int(selected.sum())
    return {
        "observations":int(len(y)),
        "selected":n,
        "coverage":float(n/len(y)) if len(y) else 0.0,
        "accuracy_selected":float(np.mean(prediction[selected]==y[selected])) if n else None,
        "accuracy_all":float(np.mean(prediction==y)) if len(y) else None,
        "brier_score":float(np.mean((p-y)**2)) if len(y) else None,
        "threshold":float(threshold)
    }

def evaluate_walk_forward(df:pd.DataFrame, initial_train:int=260, test_window:int=26,
                          min_gap:int=1, threshold:float=DEFAULT_THRESHOLD,
                          xgb_weight:float=DEFAULT_XGB_WEIGHT):
    """Fixed-spec expanding folds; purge gap reduces next-day-label boundary overlap.
    Fixed input spec avoids using holdout scores to select features or thresholds.
    """
    assert_research_frame(df)
    if not 0<=xgb_weight<=1:raise ValueError("Weight must be 0..1")
    if initial_train<50 or test_window<1 or min_gap<1:
        raise ValueError("Invalid fold configuration")
    cols=feature_columns(df)
    folds=[]; predictions=[]
    start=initial_train+min_gap
    while start<len(df):
        train_end=start-min_gap
        train=df.iloc[:train_end]
        test=df.iloc[start:min(len(df),start+test_window)]
        if train["target"].nunique()!=2:raise ValueError("Training class imbalance: only one class")
        xgb=XGBClassifier(n_estimators=120,max_depth=3,learning_rate=0.05,
                          subsample=0.85,colsample_bytree=0.85,
                          random_state=42,n_jobs=1,eval_metric="logloss")
        rf=RandomForestClassifier(n_estimators=180,max_depth=6,
                                  min_samples_leaf=3,class_weight="balanced",
                                  random_state=42,n_jobs=1)
        xgb.fit(train[cols],train["target"])
        rf.fit(train[cols],train["target"])
        px=xgb.predict_proba(test[cols])[:,1]
        pr=rf.predict_proba(test[cols])[:,1]
        probability=xgb_weight*px+(1-xgb_weight)*pr
        summary=selected_metrics(test["target"],probability,threshold)
        summary.update({"fold":len(folds)+1,"train_end":str(train.index[-1].date()),
                        "test_start":str(test.index[0].date()),"test_end":str(test.index[-1].date())})
        folds.append(summary)
        for dt,actual,prob in zip(test.index,test["target"],probability):
            predictions.append({"date":dt.date().isoformat(),"actual":int(actual),
                                "prob_up":round(float(prob),8),
                                "confidence":round(float(max(prob,1-prob)),8),
                                "selected":bool(max(prob,1-prob)>=threshold),
                                "fold":len(folds)})
        start+=test_window
    if not predictions:raise ValueError("No out-of-sample folds")
    out=pd.DataFrame(predictions)
    overall=selected_metrics(out.actual.to_numpy(),out.prob_up.to_numpy(),threshold)
    return {"summary":overall,"folds":folds,"predictions":predictions,
            "spec":{"target":"next-trading-day S&P 500 direction after Monday close",
                    "weight_xgb":xgb_weight,"weight_rf":1-xgb_weight,
                    "threshold":threshold,"initial_train":initial_train,
                    "test_window":test_window,"purge_observations":min_gap,
                    "feature_count":len(cols),"feature_columns":cols}}

def feature_drift(reference:pd.DataFrame,current:pd.DataFrame,cols:list[str],bins:int=10):
    """Population Stability Index (PSI), reference-quantile bins, missingness reported."""
    if bins<2:raise ValueError("bins >= 2")
    records=[]
    for col in cols:
        if col not in reference or col not in current:continue
        r=pd.to_numeric(reference[col],errors="coerce").dropna().to_numpy(dtype=float)
        c=pd.to_numeric(current[col],errors="coerce").dropna().to_numpy(dtype=float)
        if len(r)<bins or not len(c):continue
        edges=np.unique(np.quantile(r,np.linspace(0,1,bins+1)))
        if len(edges)<3:continue
        edges[0]=-np.inf;edges[-1]=np.inf
        rh=np.histogram(r,bins=edges)[0].astype(float)
        ch=np.histogram(c,bins=edges)[0].astype(float)
        a=(rh+0.001)/(rh.sum()+0.001*len(rh))
        b=(ch+0.001)/(ch.sum()+0.001*len(ch))
        psi=float(np.sum((b-a)*np.log(b/a)))
        records.append({"feature":col,"psi":round(psi,5),
                        "reference_rows":len(r),"current_rows":len(c),
                        "flag":bool(psi>=0.2)})
    return sorted(records,key=lambda row:row["psi"],reverse=True)

def dataset_manifest(path:Path)->dict:
    raw=path.read_bytes()
    df=pd.read_csv(path)
    return {"path":str(path),"sha256":hashlib.sha256(raw).hexdigest(),
            "rows":len(df),"columns":len(df.columns),
            "earliest":str(df["date"].min()) if "date" in df else None,
            "latest":str(df["date"].max()) if "date" in df else None}

def publication_audit(news:pd.DataFrame, cutoff_col="prediction_cutoff",published_col="published_at"):
    """Only source articles published at/before the target decision cutoff may enter a signal."""
    if cutoff_col not in news or published_col not in news:
        raise ValueError("Need article publication and decision-cutoff timestamps")
    published=pd.to_datetime(news[published_col],utc=True,errors="coerce")
    cutoff=pd.to_datetime(news[cutoff_col],utc=True,errors="coerce")
    valid=published.notna() & cutoff.notna() & (published<=cutoff)
    return {"rows":len(news),"valid_asof":int(valid.sum()),"blocked":int((~valid).sum()),
            "invalid_row_indices":news.index[~valid].tolist()}

def run_cli():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--features",default="data/processed/features.csv")
    p.add_argument("--output",default="reports/walk_forward.json")
    p.add_argument("--initial-train",type=int,default=260)
    p.add_argument("--window",type=int,default=26)
    args=p.parse_args()
    path=Path(args.features)
    df=pd.read_csv(path,index_col="date",parse_dates=True).sort_index()
    result=evaluate_walk_forward(df,args.initial_train,args.window)
    result["dataset"]=dataset_manifest(path)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2))
    print(json.dumps({"summary":result["summary"],"folds":len(result["folds"]),"output":str(out)},indent=2))
if __name__=="__main__":run_cli()
