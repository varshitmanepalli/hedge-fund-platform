"""
Unit tests for Execution Agent and slippage models.
"""

import pytest
from unittest.mock import AsyncMock, patch

from agents.execution.slippage import (
    apply_linear_slippage,
    apply_sqrt_market_impact,
    compute_commission,
    compute_execution_price,
)
from agents.execution.agent import ExecutionAgent, ExecutionAgentInput


# ── Slippage Model Tests ───────────────────────────────────────────────────────

class TestSlippageModels:
    def test_buy_slippage_increases_price(self):
        exec_price = apply_linear_slippage(100.0, side="buy", slippage_bps=5.0)
        assert exec_price > 100.0
        assert abs(exec_price - 100.05) < 0.001

    def test_sell_slippage_decreases_price(self):
        exec_price = apply_linear_slippage(100.0, side="sell", slippage_bps=5.0)
        assert exec_price < 100.0

    def test_zero_slippage(self):
        assert apply_linear_slippage(100.0, "buy", 0.0) == 100.0

    def test_sqrt_market_impact_buy(self):
        price, impact_bps = apply_sqrt_market_impact(
            price=100.0, side="buy", order_value=100_000, avg_daily_volume_value=10_000_000
        )
        assert price > 100.0
        assert impact_bps > 0

    def test_sqrt_market_impact_zero_volume(self):
        price, impact_bps = apply_sqrt_market_impact(100.0, "buy", 1000, 0)
        assert price == 100.0
        assert impact_bps == 0.0

    def test_commission_minimum(self):
        comm = compute_commission(quantity=0.1, commission_per_share=0.005, min_commission=1.0)
        assert comm == 1.0

    def test_commission_scales_with_quantity(self):
        comm = compute_commission(quantity=1000, commission_per_share=0.005)
        assert abs(comm - 5.0) < 0.001

    def test_execution_price_all_costs(self):
        result = compute_execution_price(
            mid_price=100.0, side="buy", order_quantity=1000,
            slippage_bps=5.0, avg_daily_volume=500_000
        )
        assert result["execution_price"] > 100.0
        assert result["total_cost_bps"] >= result["slippage_bps"]


# ── Execution Agent Tests ─────────────────────────────────────────────────────

BASIC_INPUT = ExecutionAgentInput(
    portfolio_id="test_port_001",
    current_positions={"AAPL": 100.0, "MSFT": 0.0},
    target_weights={"AAPL": 0.25, "MSFT": 0.20, "cash": 0.55},
    capital=1_000_000.0,
    current_prices={"AAPL": 195.0, "MSFT": 420.0},
    avg_daily_volumes={"AAPL": 50_000_000.0, "MSFT": 20_000_000.0},
    slippage_bps=5.0,
    commission_per_share=0.005,
)


@pytest.mark.asyncio
class TestExecutionAgent:
    async def test_generates_trades(self):
        agent = ExecutionAgent(run_id="test_run_e01")
        with (
            patch("db.models.trade.Trade.insert_many", new_callable=AsyncMock),
            patch("db.models.agent_log.AgentLog.insert", new_callable=AsyncMock),
        ):
            output = await agent.run(BASIC_INPUT)
        assert len(output.trades) >= 1

    async def test_buy_order_for_new_position(self):
        agent = ExecutionAgent(run_id="test_run_e02")
        with (
            patch("db.models.trade.Trade.insert_many", new_callable=AsyncMock),
            patch("db.models.agent_log.AgentLog.insert", new_callable=AsyncMock),
        ):
            output = await agent.run(BASIC_INPUT)
        msft_trades = [t for t in output.trades if t["symbol"] == "MSFT"]
        assert any(t["side"] == "buy" for t in msft_trades)

    async def test_portfolio_value_is_reasonable(self):
        agent = ExecutionAgent(run_id="test_run_e03")
        with (
            patch("db.models.trade.Trade.insert_many", new_callable=AsyncMock),
            patch("db.models.agent_log.AgentLog.insert", new_callable=AsyncMock),
        ):
            output = await agent.run(BASIC_INPUT)
        assert 900_000 < output.portfolio_value < 1_100_000

    async def test_no_short_selling(self):
        input_data = ExecutionAgentInput(
            portfolio_id="test_port_003",
            current_positions={"AAPL": 50.0},
            target_weights={"AAPL": 0.0, "cash": 1.0},
            capital=1_000_000.0,
            current_prices={"AAPL": 195.0},
        )
        agent = ExecutionAgent(run_id="test_run_e04")
        with (
            patch("db.models.trade.Trade.insert_many", new_callable=AsyncMock),
            patch("db.models.agent_log.AgentLog.insert", new_callable=AsyncMock),
        ):
            output = await agent.run(input_data)
        for sym, qty in output.positions_after.items():
            assert qty >= 0

    async def test_zero_target_generates_sell(self):
        input_data = ExecutionAgentInput(
            portfolio_id="test_port_004",
            current_positions={"AAPL": 100.0},
            target_weights={"AAPL": 0.0, "cash": 1.0},
            capital=500_000.0,
            current_prices={"AAPL": 195.0},
        )
        agent = ExecutionAgent(run_id="test_run_e05")
        with (
            patch("db.models.trade.Trade.insert_many", new_callable=AsyncMock),
            patch("db.models.agent_log.AgentLog.insert", new_callable=AsyncMock),
        ):
            output = await agent.run(input_data)
        sell_trades = [t for t in output.trades if t["side"] == "sell" and t["symbol"] == "AAPL"]
        assert len(sell_trades) >= 1
