# Agent Interface Contracts

## 1. Macro Agent

### Input
```json
{
  "symbols": ["AAPL", "MSFT", "SPY"],
  "lookback_days": 90,
  "macro_indicators": {
    "yield_curve_slope": -0.42,
    "vix": 18.3,
    "gdp_growth_qoq": 2.1,
    "cpi_yoy": 3.4,
    "unemployment_rate": 3.9,
    "fed_funds_rate": 5.25,
    "credit_spread_hy": 320
  }
}
```

### Output → MacroSignal
```json
{
  "timestamp": "2026-04-15T09:00:00Z",
  "regime": "bull",
  "confidence": 0.72,
  "indicators": { ... },
  "llm_reasoning": "Yield curve has recovered from inversion...",
  "agent_version": "1.0.0"
}
```

### Internal Logic
1. Pull macro indicators from FRED API or DB cache
2. Normalize indicators to z-scores vs. 10-year history
3. Run rule-based regime classifier (yield curve + VIX thresholds)
4. Pass indicator summary to LLM for narrative reasoning
5. Return regime + confidence + LLM text

### Dependencies
- FRED API (or cached macro data in MongoDB)
- HuggingFace LLM (narrative generation)
- Redis cache for regime state

---

## 2. Quant Agent

### Input
```json
{
  "symbols": ["AAPL", "MSFT", "GOOGL"],
  "price_history": { "AAPL": [...], "MSFT": [...] },
  "lookback_days": 252,
  "factor_weights": {
    "momentum_12m": 0.4,
    "momentum_1m": 0.2,
    "volatility": 0.2,
    "volume_trend": 0.2
  }
}
```

### Output → List[QuantSignal]
```json
[
  {
    "symbol": "AAPL",
    "factor_scores": { "momentum_12m": 1.2, "volatility": 0.8, ... },
    "composite_score": 1.05,
    "rank": 1,
    "percentile": 95.0
  }
]
```

### Internal Logic
1. Fetch OHLCV from MongoDB (market_data collection)
2. Compute rolling returns for 1m, 3m, 12m windows
3. Compute realized vol (21-day rolling std of log returns)
4. Z-score normalize each factor cross-sectionally
5. Weight and sum factors into composite_score
6. Rank all assets in universe
7. Return sorted QuantSignal list

### Dependencies
- MongoDB market_data collection
- scikit-learn (StandardScaler, cross-sectional normalization)
- NumPy / Pandas

---

## 3. Sentiment Agent

### Input
```json
{
  "symbols": ["AAPL", "MSFT"],
  "news_items": [
    {
      "symbol": "AAPL",
      "title": "Apple beats earnings expectations",
      "body": "Apple Inc. reported Q2 earnings...",
      "source": "Reuters",
      "published": "2026-04-14T16:00:00Z"
    }
  ],
  "lookback_hours": 48
}
```

### Output → List[SentimentSignal]
```json
[
  {
    "symbol": "AAPL",
    "sentiment_score": 0.72,
    "sentiment_label": "positive",
    "news_count": 12,
    "top_headlines": [...],
    "model_used": "distilbert-base-uncased-finetuned-sst-2-english"
  }
]
```

### Internal Logic
1. Fetch recent news from MongoDB or NewsAPI
2. Filter by symbol relevance (ticker mention + company name)
3. Run FinBERT / DistilBERT on each headline+snippet
4. Aggregate per-article scores → weighted avg by recency
5. Return sentiment score ∈ [-1, 1] per asset

### Dependencies
- HuggingFace Transformers (FinBERT: ProsusAI/finbert)
- NewsAPI / GNews (ingested into MongoDB)
- MongoDB news_data collection

---

## 4. Risk Manager Agent

### Input
```json
{
  "portfolio_id": "port_001",
  "proposed_weights": { "AAPL": 0.25, "MSFT": 0.20, "GOOGL": 0.15, "cash": 0.40 },
  "portfolio_returns_history": [...],
  "constraints": {
    "max_position_weight": 0.30,
    "max_var_95_pct": 0.02,
    "min_sharpe": 0.5,
    "max_drawdown_limit": 0.15,
    "max_concentration_hhi": 0.25,
    "max_leverage": 1.0
  }
}
```

### Output → RiskMetrics + approved_weights
```json
{
  "risk_metrics": {
    "sharpe_ratio": 1.34,
    "max_drawdown": 0.08,
    "var_95": 15200.0,
    "cvar_95": 18900.0,
    "concentration_hhi": 0.18,
    "constraints_breached": [],
    "approved": true
  },
  "approved_weights": { "AAPL": 0.25, "MSFT": 0.20, ... }
}
```

### Internal Logic
1. Compute portfolio returns from historical positions
2. Calculate Sharpe (annualized), Sortino, Calmar ratios
3. Calculate max drawdown from equity curve
4. Estimate VaR using historical simulation (and parametric as fallback)
5. Check all constraints; flag breaches
6. If breached: scale down offending positions proportionally
7. Return approved_weights + full RiskMetrics

### Dependencies
- MongoDB portfolios + trades collections
- NumPy / SciPy (stats)
- PyPortfolioOpt (optional: efficient frontier)

---

## 5. Execution Agent

### Input
```json
{
  "portfolio_id": "port_001",
  "current_positions": { "AAPL": 100, "MSFT": 80 },
  "target_weights": { "AAPL": 0.25, "MSFT": 0.20, "GOOGL": 0.15, "cash": 0.40 },
  "capital": 1000000.0,
  "current_prices": { "AAPL": 195.4, "MSFT": 420.2, "GOOGL": 175.6 },
  "execution_config": {
    "slippage_model": "linear",
    "slippage_bps": 5,
    "commission_per_share": 0.005,
    "market_impact_model": "sqrt"
  }
}
```

### Output → List[Trade] + updated Portfolio
```json
{
  "trades": [
    {
      "trade_id": "trd_abc123",
      "symbol": "GOOGL",
      "side": "buy",
      "quantity": 854,
      "price": 175.69,
      "slippage_bps": 5,
      "commission": 4.27,
      "status": "filled"
    }
  ],
  "portfolio_after": { ... }
}
```

### Internal Logic
1. Compute target shares from capital × target_weight / price
2. Diff current vs. target → derive buy/sell orders
3. Apply slippage model: exec_price = price × (1 ± slippage_bps/10000)
4. Apply market impact model (sqrt of volume fraction)
5. Deduct commission and update cash
6. Write trades to MongoDB trades collection
7. Update portfolio document

### Dependencies
- MongoDB portfolios + trades collections
- Current price feed (from market_data or live API)
