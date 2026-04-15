"""
Beanie ODM model for OHLCV market data bars.
"""

from datetime import datetime
from typing import Optional, Literal

from beanie import Document
from pydantic import Field


class MarketBar(Document):
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: Optional[float] = None
    source: Literal["yahoo", "alpaca", "binance", "mock"] = "yahoo"
    interval: Literal["1d", "1h", "5m"] = "1d"

    class Settings:
        name = "market_data"
        indexes = [
            [("symbol", 1), ("timestamp", -1)],   # compound: symbol + time desc
            [("timestamp", -1)],                   # latest-first scans
        ]
