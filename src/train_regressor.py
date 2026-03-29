"""
Train XGBoost regressor for Monday return: (close_mon - close_fri) / close_fri.
Adds `next_monday_return` to data/processed/features.csv (from raw S&P 500 closes).
"""

from __future__ import annotations

import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURE_COLS = [
    "sp500_1m_momentum",
    "oil_sp500_corr_90d",
    "gold_above_50ma",
    "oil_sp500_corr_90d_lag2m",
    "sp500_realvol_60d",
    "news_oil_sentiment",
    "news_geo_risk",
    "news_market_sentiment",
    "news_fed_signal",
    "news_confidence",
]

TARGET = "next_monday_return"


def load_sp500_close() -> pd.Series:
    path = PROJECT_ROOT / "data/raw/sp500.csv"
    raw = pd.read_csv(path, skiprows=3, names=["date", "close"])
    raw["date"] = pd.to_datetime(raw["date"])
    s = raw.set_index("date")["close"].astype(float).sort_index()
    return s


def add_next_monday_return(df: pd.DataFrame, sp: pd.Series) -> pd.DataFrame:
    """Friday = previous business day before each Monday in the index."""
    idx = pd.DatetimeIndex(df.index)
    fri = idx - pd.offsets.BDay(1)
    close_mon = sp.reindex(idx)
    close_fri = sp.reindex(fri)
    y = (close_mon.values - close_fri.values) / close_fri.values
    out = df.copy()
    out[TARGET] = y
    return out


def main() -> None:
    os.chdir(PROJECT_ROOT)
    feat_path = PROJECT_ROOT / "data/processed/features.csv"

    df = pd.read_csv(feat_path, index_col="date", parse_dates=True)
    sp = load_sp500_close()

    df = add_next_monday_return(df, sp)

    missing = [c for c in FEATURE_COLS + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df.to_csv(feat_path)
    print(f"Wrote {feat_path} with column {TARGET!r} ({len(df)} rows).")

    y = df[TARGET]
    X = df[FEATURE_COLS]

    train_m = (df.index.year >= 2005) & (df.index.year <= 2021)
    test_m = (df.index.year >= 2022) & (df.index.year <= 2024)

    X_train, y_train = X[train_m], y[train_m]
    X_test, y_test = X[test_m], y[test_m]

    # Drop rows with invalid target
    tr_ok = y_train.notna()
    te_ok = y_test.notna()
    X_train, y_train = X_train[tr_ok], y_train[tr_ok]
    X_test, y_test = X_test[te_ok], y_test[te_ok]

    reg = xgb.XGBRegressor(
        n_estimators=600,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
    )
    reg.fit(X_train, y_train, verbose=False)

    pred = reg.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    dir_acc = float(np.mean(np.sign(y_test.values) == np.sign(pred)))

    print("\n--- Test set (2022-2024) ---")
    print(f"MAE:  {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"Directional accuracy: {dir_acc:.4f}")

    imp = pd.Series(reg.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\nFeature importance (gain-based, top 10):")
    for name, val in imp.head(10).items():
        print(f"  {name}: {val:.6f}")

    # Latest Monday with full features
    last_idx = df[FEATURE_COLS].dropna().index.max()
    row = df.loc[[last_idx], FEATURE_COLS]
    last_pred = float(reg.predict(row)[0])
    print(f"\nMost recent Monday in data: {last_idx.date()}")
    print(f"Predicted {TARGET}: {last_pred * 100:.4f}%")

    out_path = PROJECT_ROOT / "models/xgb_regressor_v1.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": reg,
        "feature_columns": FEATURE_COLS,
        "target": TARGET,
        "metrics_test_2022_2024": {
            "mae": mae,
            "rmse": rmse,
            "directional_accuracy": dir_acc,
        },
        "last_monday_predicted": str(last_idx.date()),
        "last_prediction": last_pred,
    }
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
