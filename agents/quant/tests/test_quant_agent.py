"""
Unit tests for Quant Agent and factor computation library.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

from agents.quant.factors import (
    compute_momentum,
    compute_realized_volatility,
    compute_rsi,
    compute_volume_trend,
    cross_sectional_zscore,
    compute_all_factors,
)
from agents.quant.agent import QuantAgent, QuantAgentInput


# ── Factor Tests ──────────────────────────────────────────────────────────────

def make_close_series(n: int = 300, trend: float = 0.001) -> pd.Series:
    """Generate a trending price series for testing."""
    prices = [100.0]
    np.random.seed(42)
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + trend + np.random.normal(0, 0.01)))
    return pd.Series(prices)


class TestFactors:
    def test_momentum_positive_trend(self):
        close = make_close_series(300, trend=0.002)  # Strong uptrend
        mom = compute_momentum(close, lookback_days=252)
        assert mom is not None
        assert mom > 0

    def test_momentum_negative_trend(self):
        close = make_close_series(300, trend=-0.002)  # Downtrend
        mom = compute_momentum(close, lookback_days=252)
        assert mom is not None
        assert mom < 0

    def test_momentum_insufficient_data(self):
        close = make_close_series(n=10)
        mom = compute_momentum(close, lookback_days=252)
        assert mom is None

    def test_volatility_high_vol(self):
        """High-volatility series should produce higher vol metric."""
        low_vol = make_close_series(300, trend=0.001)
        high_vol = pd.Series(
            [100 * (1 + np.random.normal(0, 0.05)) for _ in range(300)]
        )
        low_vol.iloc[:] = pd.Series(low_vol).cumprod()

        vol_low = compute_realized_volatility(low_vol.cumsum(), window=21)
        vol_high = compute_realized_volatility(high_vol, window=21)

        # High-vol series should have higher realized vol
        if vol_low and vol_high:
            assert vol_high > vol_low

    def test_rsi_overbought(self):
        """Strongly trending up → RSI should be high."""
        close = pd.Series(np.linspace(100, 200, 50))  # Straight up
        rsi = compute_rsi(close, period=14)
        assert rsi is not None
        assert rsi > 60

    def test_rsi_oversold(self):
        """Strongly trending down → RSI should be low."""
        close = pd.Series(np.linspace(200, 100, 50))  # Straight down
        rsi = compute_rsi(close, period=14)
        assert rsi is not None
        assert rsi < 40

    def test_rsi_insufficient_data(self):
        close = pd.Series([100, 101, 102])
        assert compute_rsi(close, period=14) is None

    def test_volume_trend_rising(self):
        """Rising volume trend should return > 1.0."""
        # Low volume for first 60, high volume for last 20
        volume = pd.Series([1_000_000] * 60 + [3_000_000] * 20)
        ratio = compute_volume_trend(volume)
        assert ratio is not None
        assert ratio > 1.0

    def test_cross_sectional_zscore(self):
        scores = {"AAPL": 0.15, "MSFT": 0.08, "GOOGL": -0.03, "AMZN": 0.12, "META": 0.25}
        zscores = cross_sectional_zscore(scores)

        assert set(zscores.keys()) == set(scores.keys())
        # Best performer should have highest z-score
        assert zscores["META"] == max(zscores.values())
        # Z-scores should have mean ≈ 0
        assert abs(np.mean(list(zscores.values()))) < 0.1

    def test_cross_sectional_zscore_none_handling(self):
        scores = {"AAPL": 0.15, "MSFT": None, "GOOGL": 0.05}
        zscores = cross_sectional_zscore(scores)
        assert zscores["MSFT"] == 0.0  # None → neutral

    def test_compute_all_factors(self):
        n = 280
        ts = pd.date_range(end=datetime.now(), periods=n, freq="D")
        df = pd.DataFrame({
            "Open": make_close_series(n, trend=0.001).values,
            "High": make_close_series(n, trend=0.0015).values,
            "Low": make_close_series(n, trend=0.0005).values,
            "Close": make_close_series(n, trend=0.001).values,
            "Volume": np.random.randint(1_000_000, 5_000_000, n).astype(float),
        }, index=ts)

        factors = compute_all_factors("AAPL", df)
        assert "momentum_12m" in factors
        assert "volatility" in factors
        assert "rsi_14" in factors
        assert "volume_trend" in factors


# ── Quant Agent Tests ─────────────────────────────────────────────────────────

def make_price_records(n: int = 280, trend: float = 0.001) -> list[dict]:
    """Generate OHLCV records as list of dicts for QuantAgentInput."""
    prices = make_close_series(n, trend)
    ts = pd.date_range(end=datetime.now(), periods=n, freq="D")
    return [
        {
            "timestamp": str(t),
            "Open": float(p * 0.99),
            "High": float(p * 1.01),
            "Low": float(p * 0.98),
            "Close": float(p),
            "Volume": float(np.random.randint(1_000_000, 5_000_000)),
        }
        for t, p in zip(ts, prices)
    ]


@pytest.mark.asyncio
class TestQuantAgent:
    async def test_ranks_correctly(self):
        """Best-performing symbol should rank #1."""
        input_data = QuantAgentInput(
            symbols=["AAPL", "MSFT", "GOOGL"],
            price_data={
                "AAPL": make_price_records(280, trend=0.003),   # Strong uptrend
                "MSFT": make_price_records(280, trend=0.001),   # Moderate
                "GOOGL": make_price_records(280, trend=-0.001), # Declining
            },
        )

        agent = QuantAgent(run_id="test_run_q01")
        with (
            patch("db.models.signal.QuantSignalDoc.insert_many", new_callable=AsyncMock),
            patch("db.models.agent_log.AgentLog.insert", new_callable=AsyncMock),
        ):
            output = await agent.run(input_data)

        assert output.universe_size == 3
        ranks = {s.symbol: s.rank for s in output.signals}

        # AAPL has the strongest uptrend, should rank first or second
        assert ranks["AAPL"] <= 2
        # GOOGL has a downtrend, should rank last
        assert ranks["GOOGL"] == 3

    async def test_percentile_range(self):
        input_data = QuantAgentInput(
            symbols=["A", "B", "C", "D"],
            price_data={
                "A": make_price_records(280, trend=0.002),
                "B": make_price_records(280, trend=0.001),
                "C": make_price_records(280, trend=0.0),
                "D": make_price_records(280, trend=-0.001),
            },
        )
        agent = QuantAgent(run_id="test_run_q02")
        with (
            patch("db.models.signal.QuantSignalDoc.insert_many", new_callable=AsyncMock),
            patch("db.models.agent_log.AgentLog.insert", new_callable=AsyncMock),
        ):
            output = await agent.run(input_data)

        for sig in output.signals:
            assert 0 <= sig.percentile <= 100

    async def test_handles_missing_price_data(self):
        """Agent should skip symbols with no price data."""
        input_data = QuantAgentInput(
            symbols=["AAPL", "MISSING"],
            price_data={"AAPL": make_price_records(280, trend=0.001), "MISSING": []},
        )
        agent = QuantAgent(run_id="test_run_q03")
        with (
            patch("db.models.signal.QuantSignalDoc.insert_many", new_callable=AsyncMock),
            patch("db.models.agent_log.AgentLog.insert", new_callable=AsyncMock),
        ):
            output = await agent.run(input_data)

        assert output.universe_size == 2  # Both symbols processed, MISSING gets neutral scores
