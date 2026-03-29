"""
Backfill OpenRouter news sentiment signals for each Monday from START_DATE through END_DATE.

Requires OPENROUTER_API_KEY (and optional OPENROUTER_MODEL) in app/.env or environment.
Rate-limited for API safety (~24 req/min with SLEEP_SEC=2.5).

Install: pip install pandas openai
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from news_agent import get_news_signals, load_env_files  # noqa: E402

# Extended range: more training rows (~2004–2026 Mondays)
START_DATE = "2004-01-05"
# Last Monday on/before this calendar day (adjust as needed)
END_DATE = "2026-03-23"

SLEEP_SEC = 2.5  # ~24 req/min, safely under 30 RPM

OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "news_signals.csv"


def get_all_mondays(start: str, end: str) -> list[str]:
    idx = pd.date_range(start=start, end=end, freq="W-MON")
    return [d.strftime("%Y-%m-%d") for d in idx]


def load_existing(path: Path) -> set[str]:
    if path.exists():
        df = pd.read_csv(path)
        if "date" in df.columns:
            return set(df["date"].astype(str).tolist())
    return set()


def main() -> None:
    os.chdir(PROJECT_ROOT)
    load_env_files()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    mondays = get_all_mondays(START_DATE, END_DATE)
    already_done = load_existing(OUTPUT_FILE)

    remaining = [d for d in mondays if d not in already_done]
    total = len(remaining)
    print(
        f"Total Mondays: {len(mondays)} | Already done: {len(already_done)} | Remaining: {total}",
        flush=True,
    )

    for i, date_str in enumerate(remaining):
        try:
            signals = get_news_signals(date_str)
            row = {
                "date": date_str,
                "oil_sentiment": signals.get("oil_sentiment"),
                "geo_risk": signals.get("geo_risk"),
                "market_sentiment": signals.get("market_sentiment"),
                "fed_signal": signals.get("fed_signal"),
                "confidence": signals.get("confidence"),
                "summary": signals.get("summary", ""),
            }
            df_row = pd.DataFrame([row])
            write_header = not OUTPUT_FILE.exists()
            df_row.to_csv(OUTPUT_FILE, mode="a", header=write_header, index=False)

            print(
                f"[{i + 1}/{total}] {date_str} | oil={row['oil_sentiment']} "
                f"market={row['market_sentiment']} conf={row['confidence']}",
                flush=True,
            )

        except Exception as e:
            print(f"[{i + 1}/{total}] {date_str} | ERROR: {e}", flush=True)
            time.sleep(10)
            continue

        time.sleep(SLEEP_SEC)


if __name__ == "__main__":
    main()
