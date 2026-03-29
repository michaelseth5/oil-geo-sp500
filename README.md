# Oil · Geo · S&P 500 — Research Stack

## Oil-Geo Research Suite (Streamlit)

Single entry point for two apps with shared Webull-inspired dark styling:

```bash
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
