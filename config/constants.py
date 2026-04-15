"""
Platform-wide constants, enumerations, and default factor weights.
These are deliberate defaults — all are overridable via API request body.
"""

from enum import Enum


# ── Asset Classes ─────────────────────────────────────────────────────────────
class AssetClass(str, Enum):
    EQUITY = "equity"
    ETF = "etf"
    CRYPTO = "crypto"
    BOND = "bond"
    COMMODITY = "commodity"


# ── Market Regimes ───────────────────────────────────────────────────────────
class MarketRegime(str, Enum):
    BULL = "bull"
    BEAR = "bear"
    NEUTRAL = "neutral"
    CRISIS = "crisis"


# ── Signal Actions ───────────────────────────────────────────────────────────
class SignalAction(str, Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


# ── Trade Side ───────────────────────────────────────────────────────────────
class TradeSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


# ── Trade Status ─────────────────────────────────────────────────────────────
class TradeStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


# ── Risk Tolerance ───────────────────────────────────────────────────────────
class RiskTolerance(str, Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


# ── Default Factor Weights (sum to 1.0) ──────────────────────────────────────
DEFAULT_FACTOR_WEIGHTS: dict[str, float] = {
    "momentum_12m": 0.40,
    "momentum_1m": 0.15,
    "volatility": 0.25,    # inverse: lower vol → higher score
    "volume_trend": 0.20,
}

# ── Default Agent Signal Weights (sum to 1.0) ─────────────────────────────────
DEFAULT_AGENT_WEIGHTS: dict[str, float] = {
    "macro": 0.20,
    "quant": 0.50,
    "sentiment": 0.30,
}

# ── Regime-Conditional Agent Weight Overrides ─────────────────────────────────
# In a crisis, macro dominates; in a bull, quant and sentiment drive allocation
REGIME_AGENT_WEIGHTS: dict[str, dict[str, float]] = {
    "bull":    {"macro": 0.15, "quant": 0.55, "sentiment": 0.30},
    "neutral": {"macro": 0.20, "quant": 0.50, "sentiment": 0.30},
    "bear":    {"macro": 0.35, "quant": 0.40, "sentiment": 0.25},
    "crisis":  {"macro": 0.60, "quant": 0.30, "sentiment": 0.10},
}

# ── Action Score Thresholds (based on fused signal score) ────────────────────
ACTION_THRESHOLDS: dict[str, float] = {
    "strong_buy":  0.70,
    "buy":         0.30,
    "hold":        -0.30,   # scores between -0.30 and 0.30 → hold
    "sell":        -0.70,
    # below -0.70 → strong_sell
}

# ── VIX Thresholds for Regime Override ───────────────────────────────────────
VIX_CRISIS_THRESHOLD = 35.0
VIX_ELEVATED_THRESHOLD = 25.0

# ── Rebalance Frequencies ────────────────────────────────────────────────────
REBALANCE_FREQUENCIES = ["daily", "weekly", "monthly", "quarterly"]

# ── MongoDB Collection Names ─────────────────────────────────────────────────
COLLECTIONS = {
    "market_data": "market_data",
    "news_data": "news_data",
    "signals": "signals",
    "portfolios": "portfolios",
    "trades": "trades",
    "risk_metrics": "risk_metrics",
    "agent_logs": "agent_logs",
    "backtest_results": "backtest_results",
}

# ── Default Universe ─────────────────────────────────────────────────────────
DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "TSLA", "JPM", "V", "UNH",
    "SPY",   # Benchmark ETF
]

# ── Trading Days per Year ─────────────────────────────────────────────────────
TRADING_DAYS_PER_YEAR = 252
ANNUALIZATION_FACTOR = TRADING_DAYS_PER_YEAR ** 0.5
