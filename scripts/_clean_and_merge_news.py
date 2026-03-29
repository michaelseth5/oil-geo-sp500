"""One-off: clean news_signals.csv and merge news columns into features.csv."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    news_path = ROOT / "data/processed/news_signals.csv"
    feat_path = ROOT / "data/processed/features.csv"

    # 1. Clean news_signals.csv
    df = pd.read_csv(news_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset="date")
    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(news_path, index=False)
    print(f"Cleaned rows: {len(df)}")
    print(f"First: {df['date'].min()}")
    print(f"Last: {df['date'].max()}")

    # 2. Merge into features (rename to news_* columns)
    feat = pd.read_csv(feat_path, parse_dates=["date"])
    rename_map = {
        "oil_sentiment": "news_oil_sentiment",
        "geo_risk": "news_geo_risk",
        "market_sentiment": "news_market_sentiment",
        "fed_signal": "news_fed_signal",
        "confidence": "news_confidence",
    }
    ncols = df.rename(columns=rename_map)
    keep = ["date"] + list(rename_map.values())
    ncols = ncols[[c for c in keep if c in ncols.columns]]

    drop_cols = [c for c in rename_map.values() if c in feat.columns]
    if drop_cols:
        feat = feat.drop(columns=drop_cols)

    merged = feat.merge(ncols, on="date", how="left")
    merged.to_csv(feat_path, index=False)
    print(f"features.csv rows: {len(merged)}, cols: {merged.shape[1]}")
    miss = merged[list(rename_map.values())].isna().any(axis=1).sum()
    print(f"Rows with any missing news feature: {miss}")


if __name__ == "__main__":
    main()
