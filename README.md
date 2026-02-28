# NIFTY 50 Web Analyzer

Yes — this is now a **web-based tool**.

It analyzes NIFTY 50 (`^NSEI`) using:
- **1-year daily candles** for historical pattern and strategy context
- **5-minute intraday candles** for near-live market snapshot

and shows:
- Entry/exit strategy suggestions
- Confidence + score + reasoning
- ATR-based risk levels (entry/SL/target)
- Support/resistance zones
- Simple 1Y trend strategy backtest
- Interactive 1-year chart (Close, SMA50, SMA200)

> ⚠️ Educational use only. Not financial advice.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open: `http://localhost:8000`

## Routes

- `/` → dashboard UI
- `/api/analyze` → JSON output for integrations

## Tech

- Flask web server
- yfinance for market data
- pandas/numpy for analytics
- Plotly (CDN) for charting
