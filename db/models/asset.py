"""
Beanie ODM model for tradeable assets.
"""

from datetime import datetime
from typing import Optional

from beanie import Document
from pydantic import Field

from config.constants import AssetClass


class Asset(Document):
    symbol: str
    name: str = ""
    asset_class: AssetClass = AssetClass.EQUITY
    exchange: str = ""
    currency: str = "USD"
    sector: str = ""
    industry: str = ""
    market_cap: Optional[float] = None
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "assets"
        indexes = [
            [("symbol", 1)],               # unique lookup by ticker
        ]

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "AAPL",
                "name": "Apple Inc.",
                "asset_class": "equity",
                "exchange": "NASDAQ",
                "currency": "USD",
                "sector": "Technology",
            }
        }
