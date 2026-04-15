# AI-Powered Multi-Agent Hedge Fund Platform

A production-grade multi-agent system that analyzes financial markets, makes portfolio allocation decisions, simulates trade execution, and tracks performance with full explainability — featuring an integrated real-time dashboard.

---

## Architecture

```
                        ┌──────────────────────────┐
                        │   Frontend Dashboard     │
                        │  (served by FastAPI)     │
                        └──────────┬───────────────┘
                                   │ HTTP + WebSocket
                        ┌──────────▼───────────────┐
                        │   FastAPI API Server     │
                        │   (port 8000)            │
                        └──────────┬───────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │     Orchestrator DAG Engine     │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    │              │              │
               ┌────▼────┐  ┌─────▼────┐  ┌─────▼──────┐
               │  Macro  │  │  Quant   │  │ Sentiment  │  ← parallel
               │  Agent  │  │  Agent   │  │   Agent    │
               └────┬────┘  └─────┬────┘  └─────┬──────┘
                    │              │              │
                    └──────────────┼──────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │     Signal Aggregator           │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │       Risk Manager              │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │     Execution Agent             │
                    └──────────────┬──────────────────┘
                                   │
                          MongoDB + Response
```

## Agents

| Agent | Role | Key Output |
|-------|------|-----------|
| **Macro Agent** | Classify market regime from macro indicators | `bull/bear/neutral/crisis` + confidence |
| **Quant Agent** | Cross-sectional factor scoring (momentum, vol, RSI) | Ranked asset list |
| **Sentiment Agent** | FinBERT NLP on news headlines | Sentiment score ∈ [-1, 1] per asset |
| **Risk Manager** | VaR, Sharpe, drawdown, constraint enforcement | Approved portfolio weights |
| **Execution Agent** | Simulate fills with slippage + market impact | Trade records |

---

## Quick Start

### Prerequisites

- **Python 3.11+**
- **Docker** and **Docker Compose** (for MongoDB)
- **pip** or **Poetry**

### 1. Clone and Install

```bash
git clone https://github.com/varshitmanepalli/hedge-fund-platform.git
cd hedge-fund-platform

# Install Python dependencies
pip install -m requirements.txt

# Copy environment template
cp .env.example .env
```

### 2. Configure Environment (Optional)

Edit `.env` with your API keys. **All keys are optional** — the system gracefully falls back to mock/simulated data when keys are absent.

```bash
# .env
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB_NAME=hedgefund

# Optional — enables real data ingestion
NEWS_API_KEY=your_key_here      # NewsAPI.org
FRED_API_KEY=your_key_here      # Federal Reserve FRED
HF_API_TOKEN=your_token_here    # HuggingFace (for FinBERT + LLM)
```

### 3. Start MongoDB

```bash
docker-compose up -d mongodb
```

### 4. Seed the Database (Optional)

```bash
python scripts/seed_db.py
```

### 5. Start the Application

```bash
# Start the API server (serves both backend + frontend)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

**That's it.** Open your browser to:

| URL | Description |
|-----|-------------|
| **http://localhost:8000** | Interactive Dashboard (frontend) |
| **http://localhost:8000/docs** | Swagger API Documentation |
| **http://localhost:8000/redoc** | ReDoc API Documentation |
| **http://localhost:8000/health** | Health Check Endpoint |

The dashboard auto-detects whether the API is available and falls back to demo mode with simulated data if the backend is not running.

---

## Docker Deployment (Full Stack)

Run the entire stack (MongoDB + API + Frontend) with one command:

```bash
docker-compose up -d
```

This starts:
- **MongoDB** on port `27017`
- **API + Frontend** on port `8000`

To view logs:
```bash
docker-compose logs -f api
```

To stop:
```bash
docker-compose down
```

---

## Usage Guide

### Running a Strategy

**Via Dashboard:**
1. Open http://localhost:8000
2. Click "Run New Strategy" to expand the panel
3. Select symbols, set capital and risk tolerance
4. Click "Run Strategy"
5. Watch the real-time pipeline visualization as agents execute
6. View results in KPI cards, positions table, and reasoning chains

**Via API:**
```bash
curl -X POST http://localhost:8000/api/v1/run-strategy \
  -H "Content-Type: application/json" \
  -d '{
    "portfolio_id": "port_001",
    "symbols": ["AAPL", "MSFT", "GOOGL", "NVDA", "META"],
    "capital": 1000000,
    "risk_tolerance": "moderate"
  }'
```

**Async with WebSocket streaming:**
```bash
# 1. Start run with stream=true
curl -X POST http://localhost:8000/api/v1/run-strategy \
  -H "Content-Type: application/json" \
  -d '{
    "portfolio_id": "port_001",
    "symbols": ["AAPL", "MSFT"],
    "capital": 1000000,
    "stream": true
  }'
# Returns: {"run_id": "run_abc123", "status": "started"}

# 2. Connect WebSocket for live progress
# ws://localhost:8000/ws/pipeline/run_abc123

# 3. Check result
curl http://localhost:8000/api/v1/run-status/run_abc123
```

### Running a Backtest

**Via Dashboard:**
1. Switch to the "Backtest" tab
2. Configure date range, rebalance frequency, and symbols
3. Click "Run Backtest"
4. View equity curves, monthly returns heatmap, and performance metrics

**Via API:**
```bash
curl -X POST http://localhost:8000/api/v1/backtest \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["AAPL", "MSFT", "GOOGL", "NVDA", "META"],
    "start_date": "2023-01-01",
    "end_date": "2024-12-31",
    "initial_capital": 1000000,
    "rebalance_frequency": "monthly"
  }'
```

### Viewing Signals

**Via Dashboard:**
Switch to the "Signals" tab to see the heatmap and historical signal trends.

**Via API:**
```bash
# All recent signals
curl "http://localhost:8000/api/v1/signals?limit=50"

# Filtered by symbol and agent
curl "http://localhost:8000/api/v1/signals?symbols=AAPL&agent=quant"
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Frontend Dashboard |
| `POST` | `/api/v1/run-strategy` | Run full agent pipeline |
| `GET` | `/api/v1/run-status/{run_id}` | Check async run status |
| `GET` | `/api/v1/portfolio/{id}` | Get portfolio state |
| `GET` | `/api/v1/portfolios` | List all portfolios |
| `GET` | `/api/v1/signals` | Get recent agent signals |
| `POST` | `/api/v1/backtest` | Run historical simulation |
| `GET` | `/health` | Health check |
| `WS` | `/ws/pipeline/{run_id}` | Live pipeline progress stream |
| `GET` | `/docs` | Swagger documentation |

---

## Frontend Features

The integrated dashboard includes:

- **Tabbed Navigation** — Dashboard, Signals, Backtest, Settings
- **Live Ticker Tape** — Real-time price display
- **Run Strategy Panel** — Symbol selection, capital/risk config, agent weight sliders
- **Animated Pipeline Visualization** — Real-time step-by-step execution with WebSocket
- **KPI Cards** — Portfolio value, P&L, Sharpe ratio, drawdown with sparklines
- **Market Regime Banner** — Dynamic bull/bear/neutral display
- **Equity Curve** — Interactive chart with time range toggles and crosshair
- **Portfolio Positions** — Sortable, filterable table with P&L calculations
- **Agent Signal Dashboard** — Status cards for all 5 agents
- **Reasoning Chains** — Per-symbol step-by-step trade explanations
- **Signal Heatmap** — Symbols × signal types visualization
- **Backtest Tab** — Full configuration + equity curves + monthly returns heatmap
- **Settings Tab** — API connection status, agent toggles, risk parameters
- **Pipeline Event Log** — Expandable bottom panel with timestamped events
- **Toast Notifications** — Non-intrusive success/error/info alerts
- **Auto-Detection** — Seamlessly switches between live API and demo mode

---

## Running Tests

```bash
# All tests
make test

# Unit tests only (each agent)
make test-unit

# Integration tests
make test-integration
```

---

## Configuration

All settings loaded from `.env` (see `.env.example`):

| Variable | Description | Default |
|----------|-------------|---------|
| `MONGODB_URI` | MongoDB connection string | `mongodb://localhost:27017` |
| `MONGODB_DB_NAME` | Database name | `hedgefund` |
| `HF_API_TOKEN` | HuggingFace API token (FinBERT + LLM) | — |
| `NEWS_API_KEY` | NewsAPI key for sentiment data | — |
| `FRED_API_KEY` | FRED API key for macro indicators | — |
| `USE_LOCAL_LLM` | Load models locally instead of API | `false` |
| `ENV` | Environment (`development`/`staging`/`production`) | `development` |
| `API_HOST` | API bind address | `0.0.0.0` |
| `API_PORT` | API port | `8000` |

All API keys are **optional** — the system gracefully falls back to mock data when keys are absent.

---

## Project Structure

```
hedge_fund/
├── api/
│   ├── main.py                 # FastAPI app (API + frontend serving)
│   └── routes/
│       ├── strategy.py         # POST /api/v1/run-strategy
│       ├── portfolio.py        # GET /api/v1/portfolio/{id}
│       ├── signals.py          # GET /api/v1/signals
│       ├── backtest.py         # POST /api/v1/backtest
│       └── ws.py               # WebSocket pipeline progress
├── agents/
│   ├── base.py                 # Abstract BaseAgent
│   ├── macro/                  # Macro regime classifier
│   ├── quant/                  # Factor scoring engine
│   ├── sentiment/              # FinBERT NLP
│   ├── risk/                   # VaR, Sharpe, constraints
│   └── execution/              # Simulated trade execution
├── orchestrator/
│   ├── dag.py                  # Async DAG engine
│   ├── runner.py               # Full pipeline runner + WS integration
│   ├── signal_aggregator.py    # Signal fusion
│   └── memory.py               # MongoDB-backed pipeline memory
├── backtest/
│   ├── engine.py               # Walk-forward backtester
│   └── performance.py          # CAGR, Sharpe, alpha/beta
├── config/
│   ├── settings.py             # Pydantic settings
│   └── constants.py            # Enums and constants
├── data/
│   ├── ingestion/              # yfinance, FRED, NewsAPI fetchers
│   └── pipeline.py             # Parallel ingestion orchestrator
├── db/
│   ├── client.py               # Motor + Beanie init
│   └── models/                 # MongoDB document models
├── explainability/
│   └── chain_builder.py        # Per-trade reasoning chains
├── frontend/
│   └── index.html              # Full interactive dashboard
├── llm/
│   └── provider.py             # HuggingFace + mock LLM
├── tests/                      # Unit + integration tests
├── scripts/                    # CLI utilities
├── Dockerfile                  # Docker build
├── docker-compose.yml          # Full stack deployment
├── Makefile                    # Dev commands
├── pyproject.toml              # Python project config
├── .env.example                # Environment template
└── README.md                   # This file
```

---

## Tech Stack

- **Backend**: Python 3.11 + FastAPI + Uvicorn
- **Database**: MongoDB 7.0 + Motor (async) + Beanie (ODM)
- **AI/ML**: HuggingFace Transformers (FinBERT), scikit-learn, NumPy, Pandas
- **Data**: yfinance (prices), FRED API (macro), NewsAPI (news)
- **Orchestration**: Custom async DAG engine with WebSocket progress streaming
- **Frontend**: Single-file HTML5 dashboard with Tailwind CSS + Chart.js
- **Deployment**: Docker + Docker Compose

---

## License

MIT
