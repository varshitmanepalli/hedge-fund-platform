"""
Beanie ODM model for portfolio risk metric snapshots.
"""

from datetime import datetime
from typing import Optional

from beanie import Document
from pydantic import Field


class RiskMetricsDoc(Document):
    run_id: str
    portfolio_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Return-based metrics
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    calmar_ratio: Optional[float] = None

    # Drawdown
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0

    # Value at Risk
    var_95: Optional[float] = None       # Dollar VaR
    var_99: Optional[float] = None
    cvar_95: Optional[float] = None      # Expected Shortfall

    # Portfolio characteristics
    beta_to_market: Optional[float] = None
    volatility_annualized: Optional[float] = None
    concentration_hhi: Optional[float] = None   # Herfindahl index
    leverage: float = 1.0

    # Constraint check results
    constraints_breached: list[str] = Field(default_factory=list)
    approved: bool = True

    class Settings:
        name = "risk_metrics"
        indexes = [
            [("portfolio_id", 1), ("timestamp", -1)],
            [("run_id", 1)],
        ]
