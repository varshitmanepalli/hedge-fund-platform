"""
Risk Manager Agent — validates portfolio weights against risk constraints
and computes comprehensive risk metrics.

Input:  RiskAgentInput (proposed weights + portfolio history)
Output: RiskAgentOutput (metrics + approved weights + constraint report)
"""

from datetime import datetime
from typing import Optional

import pandas as pd
from pydantic import BaseModel

from agents.base import BaseAgent
from agents.risk.constraints import (
    RiskConstraints,
    build_constraints_from_risk_tolerance,
    check_constraints,
    enforce_constraints,
)
from agents.risk.metrics import (
    compute_annualized_volatility,
    compute_beta_to_market,
    compute_cvar_historical,
    compute_current_drawdown,
    compute_hhi_concentration,
    compute_max_drawdown,
    compute_sharpe,
    compute_sortino,
    compute_var_historical,
)
from db.models.risk_metrics import RiskMetricsDoc


# ── I/O Models ────────────────────────────────────────────────────────────────

class RiskAgentInput(BaseModel):
    portfolio_id: str
    proposed_weights: dict[str, float]         # symbol → weight (incl. "cash")
    portfolio_returns: list[float] = []        # Daily return series
    market_returns: list[float] = []           # SPY daily returns (for beta)
    equity_curve: list[float] = []             # Portfolio value over time
    capital: float = 1_000_000.0
    risk_tolerance: str = "moderate"


class RiskAgentOutput(BaseModel):
    timestamp: datetime
    portfolio_id: str
    # Risk metrics
    sharpe_ratio: Optional[float]
    sortino_ratio: Optional[float]
    max_drawdown: float
    current_drawdown: float
    var_95: Optional[float]
    var_99: Optional[float]
    cvar_95: Optional[float]
    beta_to_market: Optional[float]
    volatility_annualized: Optional[float]
    concentration_hhi: float
    leverage: float
    # Constraint results
    constraints_breached: list[str]
    approved: bool
    # Final weights (may be adjusted)
    approved_weights: dict[str, float]
    agent_version: str = "1.0.0"


# ── Agent ─────────────────────────────────────────────────────────────────────

class RiskManagerAgent(BaseAgent[RiskAgentInput, RiskAgentOutput]):
    name = "risk_manager_agent"
    version = "1.0.0"

    async def _run(self, input_data: RiskAgentInput) -> RiskAgentOutput:
        returns = pd.Series(input_data.portfolio_returns, dtype=float)
        market_returns = pd.Series(input_data.market_returns, dtype=float)
        equity_curve = pd.Series(input_data.equity_curve, dtype=float)
        capital = input_data.capital
        constraints = build_constraints_from_risk_tolerance(input_data.risk_tolerance)

        # ── Compute all risk metrics ──────────────────────────────────────────
        sharpe = compute_sharpe(returns)
        sortino = compute_sortino(returns)
        max_dd = compute_max_drawdown(equity_curve) if not equity_curve.empty else 0.0
        current_dd = compute_current_drawdown(equity_curve) if not equity_curve.empty else 0.0
        var_95 = compute_var_historical(returns, confidence=0.95, portfolio_value=capital)
        var_99 = compute_var_historical(returns, confidence=0.99, portfolio_value=capital)
        cvar_95 = compute_cvar_historical(returns, confidence=0.95, portfolio_value=capital)
        beta = compute_beta_to_market(returns, market_returns) if not market_returns.empty else None
        vol = compute_annualized_volatility(returns)
        hhi = compute_hhi_concentration(input_data.proposed_weights)

        invested_weight = sum(
            w for k, w in input_data.proposed_weights.items() if k != "cash"
        )

        # ── Check constraints ─────────────────────────────────────────────────
        breaches = check_constraints(
            proposed_weights=input_data.proposed_weights,
            capital=capital,
            var_95=var_95,
            sharpe=sharpe,
            current_drawdown=current_dd,
            constraints=constraints,
        )

        # ── Enforce: adjust weights if breached ───────────────────────────────
        if breaches:
            self._logger.warning(f"Constraints breached: {breaches}. Enforcing adjustments.")
            approved_weights = enforce_constraints(input_data.proposed_weights, constraints)
        else:
            approved_weights = dict(input_data.proposed_weights)

        approved = not breaches or approved_weights != {}

        # ── Persist risk metrics ──────────────────────────────────────────────
        doc = RiskMetricsDoc(
            run_id=self.run_id,
            portfolio_id=input_data.portfolio_id,
            timestamp=datetime.utcnow(),
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            current_drawdown=current_dd,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            beta_to_market=beta,
            volatility_annualized=vol,
            concentration_hhi=hhi,
            leverage=invested_weight,
            constraints_breached=breaches,
            approved=approved,
        )
        await doc.insert()

        self._logger.info(
            f"Risk check complete. Breaches: {len(breaches)}. "
            f"Sharpe={sharpe}, VaR95=${var_95:,.0f}" if var_95 else
            f"Risk check complete. Breaches: {len(breaches)}."
        )

        return RiskAgentOutput(
            timestamp=doc.timestamp,
            portfolio_id=input_data.portfolio_id,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            current_drawdown=current_dd,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            beta_to_market=beta,
            volatility_annualized=vol,
            concentration_hhi=hhi,
            leverage=invested_weight,
            constraints_breached=breaches,
            approved=approved,
            approved_weights=approved_weights,
            agent_version=self.version,
        )
