"""
Beanie ODM model for simulated trade execution records.
"""

import uuid
from datetime import datetime

from beanie import Document
from pydantic import Field

from config.constants import TradeSide, TradeStatus


class Trade(Document):
    trade_id: str = Field(default_factory=lambda: f"trd_{uuid.uuid4().hex[:10]}")
    portfolio_id: str
    run_id: str
    symbol: str
    side: TradeSide
    quantity: float
    price: float                    # Executed price (after slippage)
    limit_price: Optional[float] = None
    slippage_bps: float = 0.0
    commission: float = 0.0
    market_impact: float = 0.0
    gross_value: float = 0.0        # quantity × limit_price
    net_value: float = 0.0          # quantity × price + commission
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    status: TradeStatus = TradeStatus.FILLED
    reason: str = ""
    signal_id: str = ""
    reasoning_chain: list[str] = Field(default_factory=list)

    class Settings:
        name = "trades"
        indexes = [
            [("portfolio_id", 1), ("timestamp", -1)],
            [("run_id", 1)],
            [("symbol", 1), ("timestamp", -1)],
        ]


# Optional import fix for Optional type
from typing import Optional  # noqa: E402 (placed here to avoid circular)
Trade.model_rebuild()
