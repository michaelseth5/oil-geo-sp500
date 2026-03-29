"""Report row counts and last date for data/processed/news_signals.csv."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "data" / "processed" / "news_signals.csv"


def main() -> None:
    df = pd.read_csv(CSV)
    dates = df["date"].astype(str)
    print(f"File: {CSV}")
    print(f"Data rows (CSV body lines): {len(df)}")
    print(f"Unique dates: {dates.nunique()}")
    print(f"Last date (max): {dates.max()}")


if __name__ == "__main__":
    main()
