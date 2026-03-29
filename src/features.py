"""
Monday-level feature matrix from daily oil / equity / macro series.

Change log (vs. original):
- Multi-window oil-S&P correlations: 20d / 60d / 90d (90d = thesis feature `oil_sp500_corr_90d`, always kept in model.py).
- Multi-window gold-oil correlation 60d / 90d.
- Realized vol: SP500 and Brent rolling std (20/60); VIX vs 20d MA ratio.
- Short-window oil-equity alignment: rolling mean of (brent_ret * sp500_ret), 20d.
- Interaction terms: corr_90d * VIX, * brent_weekly, * sp500_weekly; brent_week * VIX; yield * VIX scale.
- Calendar cyclical encoding on the Monday date (month/quarter sin-cos).
- Monday-only lags (1, 2, 4 Mondays back) for oil-S&P corr, weekly returns, VIX, curve, vol, gold-oil corr.

Target: unchanged from your pipeline — sign of *next* trading day SP500 return (Monday row -> Tue direction).
  If you instead need same-calendar Monday return, set target to (sp500_daily_return > 0) with no shift and
  remove same-day leaky SP500 inputs from X (separate modeling choice).

`ORIGINAL_FEATURE_NAMES`: the 39 columns before the above expansion (for ablation in model.py).
"""

import pandas as pd
import numpy as np
import os

# Original feature names (39) before engineering expansion — used for ablation in model.py
ORIGINAL_FEATURE_NAMES = [
    "brent_daily_return",
    "sp500_daily_return",
    "vix_daily_change",
    "dxy_daily_return",
    "tnx_daily_change",
    "gold_daily_return",
    "irx_daily_change",
    "brent_weekly_return",
    "sp500_weekly_return",
    "vix_weekly_change",
    "dxy_weekly_return",
    "gold_weekly_return",
    "brent_1m_momentum",
    "brent_3m_momentum",
    "sp500_1m_momentum",
    "sp500_3m_momentum",
    "gold_1m_momentum",
    "brent_above_50ma",
    "brent_above_200ma",
    "sp500_above_50ma",
    "sp500_above_200ma",
    "gold_above_50ma",
    "vix_level",
    "vix_high",
    "vix_extreme",
    "tnx_level",
    "tnx_weekly_change",
    "yield_curve",
    "yield_curve_change",
    "inverted",
    "dxy_level",
    "gold_level",
    "gold_oil_ratio",
    "gold_oil_ratio_chg",
    "oil_sp500_corr_90d",
    "gold_oil_corr_90d",
    "regime_encoded",
    "oil_vix_combo",
    "gold_oil_spike",
]


def load_raw_data():

    # Load all 7 raw CSVs
    files = {
        "brent": "data/raw/brent.csv",
        "sp500": "data/raw/sp500.csv",
        "vix":   "data/raw/vix.csv",
        "tnx":   "data/raw/tnx.csv",
        "dxy":   "data/raw/dxy.csv",
        "gold":  "data/raw/gold.csv",
        "irx":   "data/raw/irx.csv",
    }

    dfs = []
    for name, path in files.items():

        # Skip the first 2 junk rows
        df = pd.read_csv(path, skiprows=2, header=0)

        # Rename columns to date and ticker name
        df.columns = ["date", name]

        # Drop leftover ticker row
        df = df[df["date"] != "date"]

        # Set date as index
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

        # Convert price to float
        df[name] = pd.to_numeric(df[name], errors="coerce")

        dfs.append(df)

    # Merge all 7 side by side
    combined = pd.concat(dfs, axis=1)

    # Sort oldest to newest
    combined = combined.sort_index()

    # Drop incomplete rows
    combined = combined.dropna()

    print(f"Loaded {len(combined)} rows from {combined.index[0].date()} to {combined.index[-1].date()}")
    return combined


def _add_monday_lags(monday_df, cols, lags):
    """In-place: lagged values from previous Monday rows only (time-safe)."""
    for col in cols:
        if col not in monday_df.columns:
            continue
        for k in lags:
            monday_df[f"{col}_lag{k}m"] = monday_df[col].shift(k)


def build_features(df):

    # Empty features dataframe
    features = pd.DataFrame(index=df.index)

    # Daily returns
    features["brent_daily_return"]  = df["brent"].pct_change()
    features["sp500_daily_return"]  = df["sp500"].pct_change()
    features["vix_daily_change"]    = df["vix"].diff()
    features["dxy_daily_return"]    = df["dxy"].pct_change()
    features["tnx_daily_change"]    = df["tnx"].diff()
    features["gold_daily_return"]   = df["gold"].pct_change()
    features["irx_daily_change"]      = df["irx"].diff()

    # Weekly returns
    features["brent_weekly_return"] = df["brent"].pct_change(5)
    features["sp500_weekly_return"] = df["sp500"].pct_change(5)
    features["vix_weekly_change"]   = df["vix"].pct_change(5)
    features["dxy_weekly_return"]   = df["dxy"].pct_change(5)
    features["gold_weekly_return"]  = df["gold"].pct_change(5)

    # Momentum
    features["brent_1m_momentum"]   = df["brent"].pct_change(21)
    features["brent_3m_momentum"]   = df["brent"].pct_change(63)
    features["sp500_1m_momentum"]   = df["sp500"].pct_change(21)
    features["sp500_3m_momentum"]   = df["sp500"].pct_change(63)
    features["gold_1m_momentum"]    = df["gold"].pct_change(21)

    # Moving averages
    features["brent_above_50ma"]    = (df["brent"] > df["brent"].rolling(50).mean()).astype(int)
    features["brent_above_200ma"]   = (df["brent"] > df["brent"].rolling(200).mean()).astype(int)
    features["sp500_above_50ma"]    = (df["sp500"] > df["sp500"].rolling(50).mean()).astype(int)
    features["sp500_above_200ma"]   = (df["sp500"] > df["sp500"].rolling(200).mean()).astype(int)
    features["gold_above_50ma"]     = (df["gold"] > df["gold"].rolling(50).mean()).astype(int)

    # VIX fear levels
    features["vix_level"]           = df["vix"]
    features["vix_high"]            = (df["vix"] > 25).astype(int)
    features["vix_extreme"]         = (df["vix"] > 40).astype(int)

    # Treasury yield
    features["tnx_level"]           = df["tnx"]
    features["tnx_weekly_change"]   = df["tnx"].pct_change(5)

    # Yield curve spread 2yr vs 10yr
    features["yield_curve"]         = df["tnx"] - df["irx"]
    features["yield_curve_change"]  = features["yield_curve"].diff(5)
    features["inverted"]            = (features["yield_curve"] < 0).astype(int)

    # Dollar index
    features["dxy_level"]           = df["dxy"]
    features["dxy_weekly_return"]   = df["dxy"].pct_change(5)

    # Gold features
    features["gold_level"]          = df["gold"]
    features["gold_weekly_return"]   = df["gold"].pct_change(5)

    # Gold oil ratio
    features["gold_oil_ratio"]      = df["gold"] / df["brent"]
    features["gold_oil_ratio_chg"]  = features["gold_oil_ratio"].pct_change(5)

    # Rolling oil–S&P correlation (thesis feature — always kept downstream)
    features["oil_sp500_corr_90d"]  = (
        features["brent_daily_return"]
        .rolling(90)
        .corr(features["sp500_daily_return"])
    )
    features["oil_sp500_corr_60d"]  = (
        features["brent_daily_return"]
        .rolling(60)
        .corr(features["sp500_daily_return"])
    )
    features["oil_sp500_corr_20d"]  = (
        features["brent_daily_return"]
        .rolling(20)
        .corr(features["sp500_daily_return"])
    )

    # Rolling gold–oil correlation (multi-window)
    features["gold_oil_corr_90d"]   = (
        features["gold_daily_return"]
        .rolling(90)
        .corr(features["brent_daily_return"])
    )
    features["gold_oil_corr_60d"]   = (
        features["gold_daily_return"]
        .rolling(60)
        .corr(features["brent_daily_return"])
    )

    # Realized vol (risk regime) — daily series, evaluated on calendar rows before Monday filter
    features["sp500_realvol_20d"]   = features["sp500_daily_return"].rolling(20).std()
    features["sp500_realvol_60d"]   = features["sp500_daily_return"].rolling(60).std()
    features["brent_realvol_20d"]   = features["brent_daily_return"].rolling(20).std()
    features["vix_ma20_ratio"]      = df["vix"] / df["vix"].rolling(20).mean()

    # Rolling covariance proxy: oil vs equity drift alignment (short window)
    features["brent_sp500_rollcov20"] = (
        features["brent_daily_return"] * features["sp500_daily_return"]
    ).rolling(20).mean()

    # Regime encoding (bins on 90d correlation)
    features["regime_encoded"]      = pd.cut(
        features["oil_sp500_corr_90d"],
        bins=[-1, -0.3, 0.3, 1],
        labels=[-1, 0, 1]
    ).astype(float)

    # Oil and VIX both spiking together
    features["oil_vix_combo"]       = (
        (features["brent_weekly_return"] > 0.03) &
        (features["vix_weekly_change"]   > 0.05)
    ).astype(int)

    # Gold and oil both spiking
    features["gold_oil_spike"]      = (
        (features["gold_weekly_return"]  > 0.02) &
        (features["brent_weekly_return"] > 0.03)
    ).astype(int)

    # Interaction terms (oil–macro thesis: corr × stress / oil move)
    features["oil_sp500_corr_x_vix"] = features["oil_sp500_corr_90d"] * features["vix_level"]
    features["oil_sp500_corr_x_brent_wk"] = features["oil_sp500_corr_90d"] * features["brent_weekly_return"]
    features["oil_sp500_corr_x_sp500_wk"] = features["oil_sp500_corr_90d"] * features["sp500_weekly_return"]
    features["brent_wk_x_vix"] = features["brent_weekly_return"] * features["vix_level"]
    features["yield_x_vix"] = features["yield_curve"] * (features["vix_level"] / 25.0)

    # Target: next trading day return sign (matches prior pipeline; Monday row predicts Tue direction)
    features["target"]              = (features["sp500_daily_return"].shift(-1) > 0).astype(int)

    # Filter to Mondays only
    features["day_of_week"]         = pd.to_datetime(features.index).dayofweek
    monday_features                 = features[features["day_of_week"] == 0].copy()
    monday_features                 = monday_features.drop(columns=["day_of_week"])

    # Calendar seasonality on Monday dates (cyclical; no lookahead)
    idx = monday_features.index
    monday_features["cal_month_sin"] = np.sin(2 * np.pi * idx.month / 12.0)
    monday_features["cal_month_cos"] = np.cos(2 * np.pi * idx.month / 12.0)
    monday_features["cal_qtr_sin"] = np.sin(2 * np.pi * idx.quarter / 4.0)
    monday_features["cal_qtr_cos"] = np.cos(2 * np.pi * idx.quarter / 4.0)

    # Monday-only lags: prior-week / prior-month Monday values for signal persistence
    lag_sources = [
        "oil_sp500_corr_90d",
        "oil_sp500_corr_60d",
        "oil_sp500_corr_20d",
        "brent_weekly_return",
        "sp500_weekly_return",
        "vix_level",
        "yield_curve",
        "sp500_realvol_20d",
        "gold_oil_corr_90d",
    ]
    _add_monday_lags(monday_features, lag_sources, (1, 2, 4))

    print(f"Built {len(monday_features)} Monday rows")
    print(f"Target distribution:\n{monday_features['target'].value_counts()}")

    return monday_features


def save_features(df):

    # Save to processed folder
    os.makedirs("data/processed", exist_ok=True)
    path = "data/processed/features.csv"
    df.to_csv(path)
    print(f"Saved -> {path}")


if __name__ == "__main__":

    # Run full pipeline
    raw   = load_raw_data()
    feats = build_features(raw)
    feats = feats.dropna()
    save_features(feats)

    # Print all feature names
    print("\nFeatures built:")
    for col in feats.columns:
        print(f"  {col}")
