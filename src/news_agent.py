"""
OpenRouter LLM client for weekly macro/news sentiment signals (JSON).

Set environment variable OPENROUTER_API_KEY (never commit keys to git).
Keys are loaded from project .env, app/.env, or app/env via load_env_files().

Install: pip install openai
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_env_files() -> None:
    """
    Load KEY=value lines from .env files. Non-empty values are applied; later
    files override earlier ones. Empty values in a file are skipped.
    Order: project .env, app/.env, app/env.
    """
    for rel in (".env", Path("app") / ".env", Path("app") / "env"):
        path = _PROJECT_ROOT / rel
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8-sig")
        except OSError:
            continue
        for line in text.splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if s.lower().startswith("export "):
                s = s[7:].strip()
            if "=" not in s:
                continue
            key, _, val = s.partition("=")
            key = key.strip()
            if key.lower().startswith("$env:"):
                key = key[5:].strip()
            val = val.strip().strip("'").strip('"')
            if key and val:
                os.environ[key] = val

SYSTEM_PROMPT = """
You are a sell-side macro strategist writing for institutional clients. Your tone is professional,
concise, and market-first—similar to Bloomberg or Reuters terminal headlines: factual, no hype,
no first-person, no editorializing.

Your job is to analyze the dominant financial news narratives for a given week and return
structured sentiment signals that a quantitative model will use to predict S&P 500 Monday returns.

SUMMARY FIELD (BLOOMBERG / REUTERS STYLE):
- Exactly one sentence, under 25 words, market-focused and factual.
- No political bias; no named wars, battles, militant groups, terrorist organizations, or
  country-specific military operations.
- Do not name specific countries when describing conflict (avoid "X invaded Y"); use neutral
  professional phrasing instead.
- Prefer neutral macro/geopolitical language, for example:
  "escalating regional tensions", "heightened geopolitical risk in the Middle East",
  "global supply disruption concerns", "world conflict uncertainty", "regional instability",
  "risk-off flows on geopolitical headlines", "energy markets pricing supply risk".
- BAD (do not write like this): "Hamas attack on Israel drove oil prices higher"
- GOOD: "Escalating Middle East tensions lifted crude amid regional supply disruption concerns."

SCORING RULES:
- oil_sentiment:    -1 = bearish oil (demand fears, oversupply, price drop)
                     0 = neutral/no clear move
                     1 = bullish oil (supply cut, geopolitical risk, price spike)

- geo_risk:         -1 = risk-off easing (de-escalation, ceasefire, stability)
                     0 = neutral
                     1 = elevated risk (conflict, sanctions, political shock)

- market_sentiment: -1 = risk-off (sell-off, fear, recession talk)
                     0 = neutral/mixed
                     1 = risk-on (rally, strong data, optimism)

- fed_signal:       -1 = dovish (rate cut, pause, easy policy)
                     0 = neutral/no clear signal
                     1 = hawkish (rate hike, tightening, inflation concern)

CONFIDENCE RULES:
- 0.35–0.50 = thin evidence, mixed or contradictory signals
- 0.50–0.65 = moderate evidence, one or two themes visible
- 0.65–0.80 = strong evidence, clear dominant narrative
- 0.80+     = only when the week had a very obvious single story

IMPORTANT:
- Do NOT default everything to 0. If there was a visible lean, score it -1 or 1.
- Only use 0 when that axis was genuinely flat or absent from the news that week.
- Return ONLY a valid JSON object. No markdown, no backticks, no explanation.

{
    "oil_sentiment": -1 | 0 | 1,
    "geo_risk": -1 | 0 | 1,
    "market_sentiment": -1 | 0 | 1,
    "fed_signal": -1 | 0 | 1,
    "confidence": 0.35 to 0.85,
    "summary": "one terminal-style sentence, <25 words, neutral geo language, market-focused"
}
""".strip()


def build_user_prompt(target_date: str) -> str:
    return f"""
Analyze the dominant macroeconomic and geopolitical news narratives for the week ending {target_date}.

Focus on:
- Oil price direction and key drivers (OPEC, supply shocks, demand outlook)
- Geopolitical events affecting energy or risk sentiment
- S&P 500 tone (risk-on vs risk-off, earnings, economic data)
- Federal Reserve signals (speeches, decisions, inflation data)

Score each axis directionally. If oil moved or had a clear story that week, use -1 or 1.
If markets were clearly risk-on or risk-off, reflect that. Only use 0 if that axis was
genuinely absent or flat.

Confidence should reflect how strong and consistent the evidence is across sources.
Aim for 0.55–0.75 when themes are visible. Use 0.80+ only for very obvious weeks.

The JSON "summary" must follow the Bloomberg/Reuters-style rules in the system message (one sentence,
under 25 words, neutral geopolitical wording, no named conflicts or groups).
""".strip()


def _parse_json_response(raw: str) -> dict[str, Any]:
    """Parse model output; strip optional markdown fences."""
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fence:
        text = fence.group(1).strip()
    return json.loads(text)


def get_news_signals(
    target_date: str,
    *,
    model: str | None = None,
    temperature: float = 0.35,
    max_completion_tokens: int = 500,
) -> dict[str, Any]:
    """
    Call OpenRouter chat completion and return parsed JSON signals.

    Parameters
    ----------
    target_date
        Week-ending date string, e.g. \"2024-01-05\", for the user prompt.

    model
        OpenRouter model id; default from env OPENROUTER_MODEL or meta-llama/llama-3.3-70b-instruct.
    """
    load_env_files()
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Add it to .env (project root), app/.env, or app/env as "
            "OPENROUTER_API_KEY=your_key, or set the environment variable in your terminal."
        )

    from openai import OpenAI

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    model_id = model or os.environ.get(
        "OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct"
    )

    user_content = build_user_prompt(target_date)

    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
    )

    raw = response.choices[0].message.content
    if not raw:
        raise RuntimeError("Empty response from OpenRouter")

    return _parse_json_response(raw)


def signals_to_feature_row(signals: dict[str, Any]) -> dict[str, float]:
    """Map JSON keys to flat numeric features for merging into a DataFrame row."""
    return {
        "news_oil_sentiment": float(signals["oil_sentiment"]),
        "news_geo_risk": float(signals["geo_risk"]),
        "news_market_sentiment": float(signals["market_sentiment"]),
        "news_fed_signal": float(signals["fed_signal"]),
        "news_confidence": float(signals["confidence"]),
    }


if __name__ == "__main__":
    import sys

    load_env_files()
    date = sys.argv[1] if len(sys.argv) > 1 else "2024-01-05"
    out = get_news_signals(date)
    print(json.dumps(out, indent=2))
