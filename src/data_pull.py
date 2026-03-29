import yfinance as yf
import pandas as pd
import os
from datetime import datetime


TICKERS = {
    "brent": "BZ=F",
    "sp500": "^GSPC",
    "vix":   "^VIX",
    "tnx":   "^TNX",
    "dxy":   "DX-Y.NYB",
    "gold":  "GC=F",
    "irx":   "^IRX",
}

START = "2005-01-01"
END   = datetime.today().strftime("%Y-%m-%d")


def pull_data():

    # Create raw data folder
    os.makedirs("data/raw", exist_ok=True)

    for name, ticker in TICKERS.items():
        print(f"Pulling {name} ({ticker})...")
        df = yf.download(ticker, start=START, end=END, auto_adjust=True)

        if df.empty:
            print(f"  WARNING: No data for {name}")
            continue

        # Keep only close price
        df = df[["Close"]].rename(columns={"Close": name})
        df.index.name = "date"

        # Save to raw folder
        path = f"data/raw/{name}.csv"
        df.to_csv(path)
        print(f"  Saved {len(df)} rows → {path}")


if __name__ == "__main__":
    pull_data()