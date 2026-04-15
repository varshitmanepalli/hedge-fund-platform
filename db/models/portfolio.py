"""
Beanie ODM models for Portfolio and Position.
"""

from datetime import datetime
from typing import Optional

from beanie import Document
from pydantic import BaseModel, Field

from config.constants import RiskTolerance


class Position(BaseModel):
    symbol: str
    quantity: float
    avg_entry_price: float
    current_price: float = 0.0
    market_value: float = 0.0
    weight: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0

    def refresh(self, current_price: float, total_portfolio_value: float) -> None:
        """Update derived fields given a fresh price and portfolio total."""
        self.current_price = current_price
        self.market_value = self.quantity * current_price
        self.unrealized_pnl = (current_price - self.avg_entry_price) * self.quantity
        self.unrealized_pnl_pct = (
            (current_price - self.avg_entry_price) / self.avg_entry_price
            if self.avg_entry_price > 0
            else 0.0
        )
        self.weight = (
            self.market_value / total_portfolio_value if total_portfolio_value > 0 else 0.0
        )


class Portfolio(Document):
    portfolio_id: str
    name: str = "Default Portfolio"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    capital: float                  # Total portfolio value (cash + positions)
    cash: float                     # Uninvested cash
    currency: str = "USD"
    positions: list[Position] = Field(default_factory=list)
    target_weights: dict[str, float] = Field(default_factory=dict)
    total_pnl: float = 0.0
    total_return: float = 0.0
    benchmark: str = "SPY"
    risk_tolerance: RiskTolerance = RiskTolerance.MODERATE
    equity_curve: list[dict] = Field(default_factory=list)  # [{date, value}]

    class Settings:
        name = "portfolios"
        indexes = [[("portfolio_id", 1)]]

    def get_position(self, symbol: str) -> Optional[Position]:
        return next((p for p in self.positions if p.symbol == symbol), None)

    def total_invested(self) -> float:
        return sum(p.market_value for p in self.positions)
