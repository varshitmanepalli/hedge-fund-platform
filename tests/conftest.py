"""
Shared test fixtures for the hedge fund platform.

Key fixtures:
  - mock_db_init: patches all Beanie/Motor DB calls
  - mock_llm: returns deterministic LLM responses
  - sample_price_data: 280 days of synthetic OHLCV
  - sample_macro_indicators: reasonable indicator values
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def event_loop():
    """Shared event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_price_df():
    """Generate 280 days of synthetic OHLCV data."""
    n = 280
    np.random.seed(42)
    prices = [100.0]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + np.random.normal(0.0005, 0.012)))

    ts = pd.date_range(end=datetime.now(), periods=n, freq="D")
    return pd.DataFrame({
        "Open":   [p * 0.99 for p in prices],
        "High":   [p * 1.015 for p in prices],
        "Low":    [p * 0.985 for p in prices],
        "Close":  prices,
        "Volume": np.random.randint(1_000_000, 10_000_000, n).astype(float),
    }, index=ts)


@pytest.fixture
def sample_macro_indicators():
    return {
        "yield_curve_slope": 0.25,
        "vix": 18.5,
        "gdp_growth_qoq": 2.1,
        "cpi_yoy": 3.4,
        "unemployment_rate": 3.9,
        "fed_funds_rate": 5.25,
        "credit_spread_hy": 310.0,
    }


@pytest.fixture
def sample_returns():
    np.random.seed(42)
    return pd.Series(np.random.normal(0.0005, 0.010, 252))


@pytest.fixture
def sample_equity_curve(sample_returns):
    equity = [1_000_000.0]
    for r in sample_returns:
        equity.append(equity[-1] * (1 + r))
    return pd.Series(equity)


@pytest.fixture
def mock_db():
    """Patch all MongoDB operations for unit tests."""
    with (
        patch("db.models.signal.MacroSignalDoc.insert", new_callable=AsyncMock),
        patch("db.models.signal.QuantSignalDoc.insert_many", new_callable=AsyncMock),
        patch("db.models.signal.SentimentSignalDoc.insert_many", new_callable=AsyncMock),
        patch("db.models.signal.AggregatedSignalDoc.insert_many", new_callable=AsyncMock),
        patch("db.models.risk_metrics.RiskMetricsDoc.insert", new_callable=AsyncMock),
        patch("db.models.trade.Trade.insert_many", new_callable=AsyncMock),
        patch("db.models.agent_log.AgentLog.insert", new_callable=AsyncMock),
    ):
        yield


@pytest.fixture
def mock_llm():
    """Mock LLM that returns a deterministic response."""
    with patch("llm.provider.get_llm_provider") as mock_get:
        provider = MagicMock()
        provider.generate = AsyncMock(return_value="Market conditions are broadly constructive.")
        mock_get.return_value = provider
        yield provider
