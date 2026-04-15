"""
Factor computation library for the Quant Agent.

All factor functions accept a price DataFrame (DatetimeIndex, OHLCV columns)
and return a float score. Cross-sectional z-scoring is done in the agent.

Factors:
  - momentum_12m: 12-month total return (excluding last month to avoid reversal)
  - momentum_1m:  1-month return
  - volatility:   21-day realized vol (lower = higher score → inverted)
  - volume_trend: 20d avg volume / 60d avg volume ratio (rising vol = momentum)
  - rsi_14:       14-day RSI
  - beta:         Beta to SPY (requires benchmark series)
"""

import numpy as np
import pandas as pd
from typing import Optional


def compute_momentum(close: pd.Series, lookback_days: int, skip_days: int = 0) -> Optional[float]:
    """
    Total return over lookback_days, skipping skip_days at the end.
    Returns None if insufficient data.
    """
    total_needed = lookback_days + skip_days
    if len(close) < total_needed + 1:
        return None

    end_idx = len(close) - 1 - skip_days
    start_idx = end_idx - lookback_days

    start_price = close.iloc[start_idx]
    end_price = close.iloc[end_idx]

    if start_price <= 0:
        return None

    return (end_price / start_price) - 1.0


def compute_realized_volatility(close: pd.Series, window: int = 21) -> Optional[float]:
    """
    Annualized realized volatility from log returns over trailing `window` days.
    """
    if len(close) < window + 1:
        return None

    log_returns = np.log(close / close.shift(1)).dropna()
    if len(log_returns) < window:
        return None

    recent_returns = log_returns.iloc[-window:]
    vol = recent_returns.std() * np.sqrt(252)
    return float(vol)


def compute_rsi(close: pd.Series, period: int = 14) -> Optional[float]:
    """Standard 14-day RSI."""
    if len(close) < period + 1:
        return None

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()

    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    last = rsi.iloc[-1]

    return float(last) if not np.isnan(last) else None


def compute_volume_trend(volume: pd.Series, short_window: int = 20, long_window: int = 60) -> Optional[float]:
    """
    Ratio of short-term to long-term average daily volume.
    > 1.0 means accelerating volume (bullish signal when combined with momentum).
    """
    if len(volume) < long_window:
        return None

    short_avg = volume.iloc[-short_window:].mean()
    long_avg = volume.iloc[-long_window:].mean()

    if long_avg == 0:
        return None

    return float(short_avg / long_avg)


def compute_beta(
    close: pd.Series,
    benchmark_close: pd.Series,
    window: int = 63,
) -> Optional[float]:
    """
    Rolling 63-day (≈ 1 quarter) beta to benchmark.
    """
    if len(close) < window + 1 or len(benchmark_close) < window + 1:
        return None

    asset_returns = np.log(close / close.shift(1)).dropna().iloc[-window:]
    bench_returns = np.log(benchmark_close / benchmark_close.shift(1)).dropna().iloc[-window:]

    # Align on common index
    aligned = pd.DataFrame({"asset": asset_returns, "bench": bench_returns}).dropna()
    if len(aligned) < 20:
        return None

    cov = aligned.cov().iloc[0, 1]
    var = aligned["bench"].var()
    return float(cov / var) if var > 0 else None


def cross_sectional_zscore(scores: dict[str, Optional[float]]) -> dict[str, float]:
    """
    Z-score a dict of symbol → raw_score across the cross-section.
    Symbols with None scores receive a z-score of 0 (neutral).

    Returns: dict of symbol → z_score
    """
    valid = {k: v for k, v in scores.items() if v is not None}
    if not valid:
        return {k: 0.0 for k in scores}

    values = np.array(list(valid.values()), dtype=float)
    mean = np.mean(values)
    std = np.std(values)

    result = {}
    for symbol in scores:
        if scores[symbol] is None:
            result[symbol] = 0.0
        elif std < 1e-10:
            result[symbol] = 0.0
        else:
            result[symbol] = float((scores[symbol] - mean) / std)

    return result


def compute_all_factors(
    symbol: str,
    price_df: pd.DataFrame,
    benchmark_df: Optional[pd.DataFrame] = None,
) -> dict[str, Optional[float]]:
    """
    Compute all raw factor values for a single asset.

    Args:
        symbol:       Ticker symbol (for logging)
        price_df:     DataFrame with Open/High/Low/Close/Volume columns
        benchmark_df: SPY or other benchmark for beta computation

    Returns:
        dict of factor_name → raw_value
    """
    if price_df.empty or "Close" not in price_df.columns:
        return {f: None for f in ["momentum_12m", "momentum_1m", "volatility", "volume_trend", "rsi_14", "beta"]}

    close = price_df["Close"]
    volume = price_df.get("Volume", pd.Series(dtype=float))

    factors: dict[str, Optional[float]] = {
        "momentum_12m": compute_momentum(close, lookback_days=252, skip_days=21),
        "momentum_1m":  compute_momentum(close, lookback_days=21, skip_days=0),
        "volatility":   compute_realized_volatility(close, window=21),
        "volume_trend": compute_volume_trend(volume) if not volume.empty else None,
        "rsi_14":       compute_rsi(close, period=14),
        "beta":         (
            compute_beta(close, benchmark_df["Close"], window=63)
            if benchmark_df is not None and not benchmark_df.empty
            else None
        ),
    }

    return factors
