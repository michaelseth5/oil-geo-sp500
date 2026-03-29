# Oil · Geo · S&P 500 — Research Stack

## Environment

From the repo root (recommended: use a virtual environment):

```bash
pip install -r requirements.txt
```

Python **3.11** matches `runtime.txt` (Streamlit Cloud) and avoids most wheel edge cases.

## Oil-Geo Research Suite (Streamlit)

Single entry point for two apps with shared corporate light styling:

```bash
streamlit run app.py
# or
streamlit run src/app.py
```

The sidebar uses **Streamlit navigation** to switch pages. Each page runs independently with its own widget state.

### GeoOil Intel

**Geopolitical timeline explorer** for oil and equity markets: dual-axis **WTI Crude (Brent proxy)** vs **S&P 500**, **18** curated world-event markers (category-colored lines and clickable diamond markers), rolling **WTI–S&P correlation** at **30 / 60 / 90 / 180** trading-day windows, and an **event detail** panel (oil impact, geopolitical context, market reaction, Fed stance) plus optional nearest-row readouts from `news_signals.csv`.

- Module: `src/geo_oil_intel.py`
- Default landing page when using `src/app.py`

### Monday Alpha

**Machine learning engine** for **Monday market return** forecasting: saved classifier (`models/xgb_v1.pkl`) and regressor (`models/xgb_regressor_v1.pkl`) on `data/processed/features.csv`, aligned with `src/predict_monday.py`. Dashboard is **Plotly-only** (gauges, heatmaps, charts).

- Module: `src/dashboard.py`
- Branding: **Monday Alpha · Oil-Geo Signal Engine**
- Subtitle: *XGBoost + Llama Intelligence · Monday Return Forecasting*

### Run a single page

```bash
streamlit run src/geo_oil_intel.py
streamlit run src/dashboard.py
```

## Data & models

- Processed features: `data/processed/features.csv`
- News signals: `data/processed/news_signals.csv`
- Raw prices: `data/raw/brent.csv`, `data/raw/sp500.csv` (Brent used as the liquid crude benchmark in charts)

## Python training / agents

- Classifier search: `python src/model.py`
- Regressor: `python src/train_regressor.py`
- News LLM: `python src/news_agent.py <YYYY-MM-DD>`
- CLI prediction: `python src/predict_monday.py`

## Deploy (Streamlit Community Cloud)

1. **Push this repo to GitHub** (Streamlit Cloud deploys from a remote Git host, not from your laptop alone).
2. In [Streamlit Community Cloud](https://share.streamlit.io/), sign in and **connect your GitHub account** if you see “code is not connected to a GitHub repository” (or similar): **Settings → Linked accounts → Connect GitHub**, and authorize the org/repo access the app needs.
3. **New app** → pick the repo → choose the correct **branch** (`main` vs `master` must match what you use on GitHub).
4. **Main file** (entry point), either:
   - **`app.py`** at the repo root (loads `src/app.py` for you), or
   - **`src/app.py`** directly.
5. **Dependencies**: `requirements.txt` at the repo root (includes `streamlit`, `pandas`, `plotly`, `xgboost`, `yfinance`, `openai`, etc.).
6. **Python version**: `runtime.txt` pins **Python 3.11** (Streamlit Cloud reads this from the repo root).
7. **API keys**: the hosted **GeoOil + Monday Alpha** UI does not require keys. The **news backfill** (`src/news_agent.py`) calls **OpenRouter** via the **`openai`** package and expects **`OPENROUTER_API_KEY`** in `.env` or Cloud Secrets. See `.streamlit/secrets.toml.example`.

Local runs (equivalent):

```bash
streamlit run app.py
# or
streamlit run src/app.py
```

If deployment still fails, open **Manage app → Logs** and check for missing files (`data/raw/*.csv`, `models/*.pkl`) or import errors.
