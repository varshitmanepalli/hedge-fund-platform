# AI-Powered Multi-Agent Hedge Fund Platform

A production-grade multi-agent system that analyzes financial markets, makes portfolio allocation decisions, simulates trade execution, and tracks performance with full explainability.

---

## Architecture

```
User Request → Orchestrator DAG
                │
         ┌──────┼──────┐
         │      │      │
     Macro   Quant  Sentiment  (parallel)
         │      │      │
         └──────┼──────┘
                │
           Signal Aggregator
                │
           Risk Manager
                │
           Execution Agent
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

## Quick Start

### 1. Prerequisites

- Python 3.11+
- MongoDB 7.0 (Docker recommended)
- pip

### 2. Install

```bash
cd hedge_fund
pip install -r requirements.txt
cp .env.example .env
# Fill in your API keys (FRED, NewsAPI, HuggingFace) — all optional
```

### 3. Start MongoDB

```bash
docker-compose up -d mongodb
```

### 4. Seed data

```bash
python scripts/seed_db.py
```

### 5. Run API server

```bash
make dev
# or:
uvicorn api.main:app --reload
```

Visit `http://localhost:8000/docs` for the interactive API.

### 6. Run a strategy

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

### 7. Run a backtest

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

### 8. Run tests

```bash
make test
# Or just unit tests:
make test-unit
```

---

## Configuration

All settings loaded from `.env` (see `.env.example`):

| Variable | Description | Default |
|----------|-------------|---------|
| `MONGODB_URI` | MongoDB connection string | `mongodb://localhost:27017` |
| `HF_API_TOKEN` | HuggingFace API token (for LLM + FinBERT) | — |
| `NEWS_API_KEY` | NewsAPI key for sentiment data | — |
| `FRED_API_KEY` | FRED API key for macro indicators | — |
| `USE_LOCAL_LLM` | Load models locally instead of API | `false` |

All API keys are **optional** — the system gracefully falls back to mock data when keys are absent.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/run-strategy` | Run full agent pipeline |
| `GET`  | `/api/v1/portfolio/{id}` | Get portfolio state |
| `GET`  | `/api/v1/signals` | Get recent agent signals |
| `POST` | `/api/v1/backtest` | Run historical simulation |
| `GET`  | `/health` | Health check |

---

## Tech Stack

- **Backend**: Python 3.11 + FastAPI + Uvicorn
- **Database**: MongoDB 7.0 + Motor (async driver) + Beanie (ODM)
- **AI/ML**: HuggingFace Transformers (FinBERT), scikit-learn, NumPy, Pandas
- **Data**: yfinance (prices), FRED API (macro), NewsAPI (news)
- **Orchestration**: Custom async DAG engine (LangGraph-compatible)
- **Frontend**: React dashboard with Chart.js

---

## Development Phases Completed

- [x] Phase 1: System Design & Architecture
- [x] Phase 2: Data Layer (ingestion, normalization, MongoDB)
- [x] Phase 3: Agent Implementation (all 5 agents + unit tests)
- [x] Phase 4: Orchestration (DAG runner, memory, retry)
- [x] Phase 5: Backtesting Engine (walk-forward, benchmarking)
- [x] Phase 6: API Layer (FastAPI endpoints)
- [x] Phase 7: Explainability (per-trade reasoning chains)
- [x] Phase 8: Frontend Dashboard
