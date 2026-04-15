# API Contracts

Base URL: `http://localhost:8000/api/v1`

---

## POST /run-strategy

Triggers a full pipeline run for a set of symbols.

### Request
```json
{
  "portfolio_id": "port_001",
  "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
  "capital": 1000000.0,
  "risk_tolerance": "moderate",
  "lookback_days": 252,
  "agent_weights": {
    "macro": 0.20,
    "quant": 0.50,
    "sentiment": 0.30
  }
}
```

### Response 200
```json
{
  "run_id": "run_abc123",
  "status": "success",
  "duration_ms": 4210,
  "regime": "bull",
  "trades": [...],
  "portfolio": {...},
  "risk_metrics": {...},
  "reasoning_chains": {
    "AAPL": [
      "Macro: Bullish regime detected (confidence: 72%)",
      "Quant: Strong 12m momentum (z-score: 1.2), ranked #1/5",
      "Sentiment: Positive news sentiment (score: 0.72, 12 articles)",
      "Risk: Within all constraints — VaR $12,400 < limit $20,000",
      "Execution: Bought 512 shares @ $195.69 (slippage: 5bps)"
    ]
  }
}
```

### Error 422
```json
{ "detail": [{ "loc": ["body", "capital"], "msg": "must be > 0", "type": "value_error" }] }
```

---

## GET /portfolio/{portfolio_id}

### Response 200
```json
{
  "portfolio_id": "port_001",
  "capital": 1005200.0,
  "cash": 400000.0,
  "positions": [
    {
      "symbol": "AAPL",
      "quantity": 512,
      "avg_entry_price": 195.69,
      "current_price": 197.10,
      "market_value": 100915.2,
      "weight": 0.1004,
      "unrealized_pnl": 722.0,
      "unrealized_pnl_pct": 0.0072
    }
  ],
  "total_pnl": 5200.0,
  "total_return": 0.0052,
  "updated_at": "2026-04-15T09:32:00Z"
}
```

### Error 404
```json
{ "detail": "Portfolio port_001 not found" }
```

---

## GET /signals

### Query Parameters
| Param        | Type     | Description                        |
|--------------|----------|------------------------------------|
| `symbols`    | string[] | Filter by symbols                  |
| `agent`      | string   | Filter: macro/quant/sentiment      |
| `limit`      | int      | Max results (default 50)           |
| `since`      | datetime | ISO8601 timestamp filter           |

### Response 200
```json
{
  "signals": [
    {
      "type": "quant",
      "symbol": "AAPL",
      "timestamp": "2026-04-15T09:00:00Z",
      "composite_score": 1.05,
      "rank": 1,
      "factor_scores": { "momentum_12m": 1.2, "volatility": 0.8 }
    }
  ],
  "total": 15,
  "page": 1
}
```

---

## POST /backtest

Runs a historical simulation over a date range.

### Request
```json
{
  "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
  "start_date": "2023-01-01",
  "end_date":   "2024-12-31",
  "initial_capital": 1000000.0,
  "rebalance_frequency": "monthly",
  "risk_tolerance": "moderate",
  "benchmark": "SPY",
  "agent_weights": {
    "macro": 0.20,
    "quant": 0.50,
    "sentiment": 0.30
  }
}
```

### Response 200
```json
{
  "backtest_id": "bt_xyz456",
  "period": "2023-01-01 to 2024-12-31",
  "performance": {
    "total_return": 0.324,
    "cagr": 0.161,
    "sharpe_ratio": 1.42,
    "max_drawdown": 0.112,
    "volatility_annualized": 0.187,
    "calmar_ratio": 1.44,
    "win_rate": 0.61,
    "total_trades": 84
  },
  "benchmark_performance": {
    "symbol": "SPY",
    "total_return": 0.248,
    "cagr": 0.123,
    "sharpe_ratio": 1.08,
    "max_drawdown": 0.087
  },
  "alpha": 0.076,
  "beta": 0.93,
  "equity_curve": [
    { "date": "2023-01-31", "portfolio_value": 1024500.0, "benchmark_value": 1018200.0 }
  ]
}
```

---

## WebSocket /ws/stream/{run_id}

Real-time pipeline step events (for live UI updates):

```json
{ "event": "agent_complete", "agent": "macro", "status": "success", "duration_ms": 320 }
{ "event": "agent_complete", "agent": "quant", "status": "success", "duration_ms": 890 }
{ "event": "trade_executed", "symbol": "AAPL", "side": "buy", "quantity": 512 }
{ "event": "pipeline_complete", "run_id": "run_abc123" }
```
