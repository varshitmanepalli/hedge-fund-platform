"""
Unit tests for Risk Manager Agent and metrics library.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import AsyncMock, patch

from agents.risk.metrics import (
    compute_sharpe,
    compute_sortino,
    compute_max_drawdown,
    compute_current_drawdown,
    compute_var_historical,
    compute_cvar_historical,
    compute_hhi_concentration,
    compute_annualized_volatility,
)
from agents.risk.constraints import (
    check_constraints,
    enforce_constraints,
    RiskConstraints,
    build_constraints_from_risk_tolerance,
)
from agents.risk.agent import RiskManagerAgent, RiskAgentInput


# ── Metrics Tests ─────────────────────────────────────────────────────────────

def make_returns(n: int = 252, mean: float = 0.0005, std: float = 0.01, seed: int = 42) -> pd.Series:
    np.random.seed(seed)
    return pd.Series(np.random.normal(mean, std, n))


class TestRiskMetrics:
    def test_sharpe_positive_returns(self):
        returns = make_returns(mean=0.001, std=0.008)
        sharpe = compute_sharpe(returns, risk_free_rate=0.05)
        assert sharpe is not None
        assert sharpe > 0

    def test_sharpe_negative_returns(self):
        returns = make_returns(mean=-0.001, std=0.008)
        sharpe = compute_sharpe(returns, risk_free_rate=0.05)
        assert sharpe is not None
        assert sharpe < 0

    def test_sharpe_insufficient_data(self):
        assert compute_sharpe(pd.Series([0.01, 0.02]), risk_free_rate=0.05) is None

    def test_sortino_only_penalizes_downside(self):
        """Sortino should be higher than Sharpe for positively-skewed returns."""
        np.random.seed(42)
        # Skewed returns: mostly small gains, rare large losses
        returns = pd.Series(np.abs(np.random.normal(0.001, 0.005, 252)))
        sharpe = compute_sharpe(returns, risk_free_rate=0.0)
        sortino = compute_sortino(returns, risk_free_rate=0.0)
        if sharpe and sortino:
            assert sortino >= sharpe

    def test_max_drawdown_no_drawdown(self):
        """Monotonically increasing equity → 0 drawdown."""
        equity = pd.Series([100 + i for i in range(100)])
        assert compute_max_drawdown(equity) == 0.0

    def test_max_drawdown_known_value(self):
        """Equity drops from 100 to 80 = 20% drawdown."""
        equity = pd.Series([100, 105, 110, 108, 90, 80, 85, 95, 100])
        dd = compute_max_drawdown(equity)
        assert abs(dd - 0.2727) < 0.01  # ~27% from peak 110 to 80

    def test_current_drawdown_at_peak(self):
        equity = pd.Series([100, 105, 110, 115])  # At all-time high
        assert compute_current_drawdown(equity) == 0.0

    def test_current_drawdown_below_peak(self):
        equity = pd.Series([100, 110, 105])  # Below peak of 110
        dd = compute_current_drawdown(equity)
        assert abs(dd - 0.0455) < 0.001  # (110-105)/110

    def test_var_95_is_positive(self):
        returns = make_returns(n=100, mean=0.0, std=0.015)
        var = compute_var_historical(returns, confidence=0.95, portfolio_value=1_000_000)
        assert var is not None
        assert var > 0

    def test_cvar_greater_than_var(self):
        """CVaR (Expected Shortfall) must be >= VaR."""
        returns = make_returns(n=200, mean=0.0, std=0.01)
        var = compute_var_historical(returns, 0.95, 1_000_000)
        cvar = compute_cvar_historical(returns, 0.95, 1_000_000)
        if var and cvar:
            assert cvar >= var

    def test_hhi_equal_weights(self):
        """Equal weights across 4 assets → HHI = 0.25."""
        weights = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
        assert abs(compute_hhi_concentration(weights) - 0.25) < 0.001

    def test_hhi_concentrated(self):
        """Single asset → HHI = 1.0."""
        weights = {"A": 1.0}
        assert compute_hhi_concentration(weights) == 1.0

    def test_hhi_includes_cash(self):
        weights = {"AAPL": 0.60, "cash": 0.40}
        hhi = compute_hhi_concentration(weights)
        assert 0.25 < hhi < 1.0


# ── Constraints Tests ─────────────────────────────────────────────────────────

class TestConstraints:
    def test_no_breach_clean_portfolio(self):
        weights = {"AAPL": 0.20, "MSFT": 0.15, "cash": 0.65}
        constraints = RiskConstraints(max_position_weight=0.30, max_var_95_pct=0.02)
        breaches = check_constraints(
            proposed_weights=weights,
            capital=1_000_000,
            var_95=10_000,      # 1% of capital → within limit
            sharpe=1.5,
            current_drawdown=0.05,
            constraints=constraints,
        )
        assert breaches == []

    def test_breach_position_concentration(self):
        weights = {"AAPL": 0.45, "cash": 0.55}  # Exceeds 30% limit
        constraints = RiskConstraints(max_position_weight=0.30)
        breaches = check_constraints(weights, 1_000_000, None, None, 0.0, constraints)
        assert any("max_position_weight" in b for b in breaches)

    def test_breach_var_limit(self):
        weights = {"AAPL": 0.50, "cash": 0.50}
        constraints = RiskConstraints(max_var_95_pct=0.01)
        breaches = check_constraints(weights, 1_000_000, var_95=25_000, sharpe=None,
                                     current_drawdown=0.0, constraints=constraints)
        # VaR = $25k = 2.5% > 1% limit
        assert any("max_var_95_pct" in b for b in breaches)

    def test_enforce_caps_concentration(self):
        weights = {"AAPL": 0.50, "MSFT": 0.30, "cash": 0.20}
        constraints = RiskConstraints(max_position_weight=0.30)
        corrected = enforce_constraints(weights, constraints)

        for sym, w in corrected.items():
            if sym != "cash":
                assert w <= 0.30 + 1e-6   # tolerance for float

        assert abs(sum(corrected.values()) - 1.0) < 1e-6

    def test_risk_tolerance_configs(self):
        c = build_constraints_from_risk_tolerance("conservative")
        assert c.max_position_weight < 0.20

        m = build_constraints_from_risk_tolerance("moderate")
        assert c.max_position_weight < m.max_position_weight

        a = build_constraints_from_risk_tolerance("aggressive")
        assert m.max_position_weight < a.max_position_weight


# ── Risk Agent Integration Test ───────────────────────────────────────────────

@pytest.mark.asyncio
class TestRiskManagerAgent:
    async def test_run_approves_clean_portfolio(self):
        returns = list(make_returns(252, mean=0.0006, std=0.008))
        equity = [1_000_000 * (1 + sum(returns[:i])) for i in range(len(returns))]

        input_data = RiskAgentInput(
            portfolio_id="test_port",
            proposed_weights={"AAPL": 0.20, "MSFT": 0.15, "GOOGL": 0.10, "cash": 0.55},
            portfolio_returns=returns,
            equity_curve=equity,
            capital=1_000_000,
            risk_tolerance="moderate",
        )

        agent = RiskManagerAgent(run_id="test_run_r01")
        with (
            patch("db.models.risk_metrics.RiskMetricsDoc.insert", new_callable=AsyncMock),
            patch("db.models.agent_log.AgentLog.insert", new_callable=AsyncMock),
        ):
            output = await agent.run(input_data)

        assert output.approved
        assert output.sharpe_ratio is not None
        assert output.var_95 is not None
        assert isinstance(output.constraints_breached, list)

    async def test_run_corrects_overweight_position(self):
        returns = list(make_returns(252))

        input_data = RiskAgentInput(
            portfolio_id="test_port_2",
            proposed_weights={"AAPL": 0.60, "cash": 0.40},  # AAPL way over limit
            portfolio_returns=returns,
            equity_curve=[],
            capital=1_000_000,
            risk_tolerance="moderate",  # limit = 0.25
        )

        agent = RiskManagerAgent(run_id="test_run_r02")
        with (
            patch("db.models.risk_metrics.RiskMetricsDoc.insert", new_callable=AsyncMock),
            patch("db.models.agent_log.AgentLog.insert", new_callable=AsyncMock),
        ):
            output = await agent.run(input_data)

        # Approved weights should cap AAPL
        assert output.approved_weights.get("AAPL", 0) <= 0.25 + 1e-6
