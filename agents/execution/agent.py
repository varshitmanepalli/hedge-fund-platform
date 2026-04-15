"""
Execution Agent — converts target portfolio weights into simulated trades.

Pipeline:
  1. Compute target share quantities from weights × capital / prices
  2. Diff against current holdings → buy/sell orders
  3. Apply slippage + market impact + commission
  4. Simulate fills and update portfolio state
  5. Persist Trade documents to MongoDB
  6. Return updated portfolio + trade list

Simulation assumptions:
  - All orders fill immediately at simulated execution price
  - No partial fills (simplified)
  - No short selling (long-only mode)
"""

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from agents.base import BaseAgent
from agents.execution.slippage import compute_execution_price, compute_commission
from config.constants import TradeSide, TradeStatus
from config.settings import settings
from db.models.portfolio import Portfolio, Position
from db.models.trade import Trade


# ── I/O Models ────────────────────────────────────────────────────────────────

class ExecutionAgentInput(BaseModel):
    portfolio_id: str
    current_positions: dict[str, float]        # symbol → shares held
    target_weights: dict[str, float]           # symbol → target weight (incl. cash)
    capital: float
    current_prices: dict[str, float]           # symbol → mid price
    avg_daily_volumes: dict[str, float] = {}   # symbol → avg daily volume (shares)
    slippage_bps: float = 5.0
    commission_per_share: float = 0.005
    reasoning_chains: dict[str, list[str]] = {}  # symbol → reasoning steps


class ExecutionAgentOutput(BaseModel):
    timestamp: datetime
    portfolio_id: str
    trades: list[dict]
    portfolio_value: float
    cash_remaining: float
    positions_after: dict[str, float]   # symbol → shares
    total_commission: float
    total_slippage_cost: float
    agent_version: str = "1.0.0"


# ── Agent ─────────────────────────────────────────────────────────────────────

class ExecutionAgent(BaseAgent[ExecutionAgentInput, ExecutionAgentOutput]):
    name = "execution_agent"
    version = "1.0.0"

    async def _run(self, input_data: ExecutionAgentInput) -> ExecutionAgentOutput:
        capital = input_data.capital
        prices = input_data.current_prices
        current = dict(input_data.current_positions)
        target_weights = input_data.target_weights

        # ── Step 1: Compute target shares per asset ───────────────────────────
        target_shares: dict[str, float] = {}
        for symbol, weight in target_weights.items():
            if symbol == "cash":
                continue
            price = prices.get(symbol)
            if not price or price <= 0:
                self._logger.warning(f"No price for {symbol} — skipping")
                continue
            target_shares[symbol] = (capital * weight) / price

        # ── Step 2: Compute order deltas ──────────────────────────────────────
        all_symbols = set(list(current.keys()) + list(target_shares.keys()))
        orders = []

        for symbol in all_symbols:
            current_qty = current.get(symbol, 0.0)
            target_qty = target_shares.get(symbol, 0.0)
            delta = target_qty - current_qty

            if abs(delta) < 0.01:  # Ignore dust
                continue

            side = "buy" if delta > 0 else "sell"
            quantity = abs(delta)

            mid_price = prices.get(symbol, 0.0)
            if mid_price <= 0:
                continue

            adv = input_data.avg_daily_volumes.get(symbol)
            exec_info = compute_execution_price(
                mid_price=mid_price,
                side=side,
                order_quantity=quantity,
                slippage_bps=input_data.slippage_bps,
                avg_daily_volume=adv,
            )

            commission = compute_commission(
                quantity=quantity,
                commission_per_share=input_data.commission_per_share,
            )

            exec_price = exec_info["execution_price"]
            gross_value = quantity * mid_price
            net_value = quantity * exec_price + commission if side == "buy" else quantity * exec_price - commission

            orders.append({
                "symbol": symbol,
                "side": side,
                "quantity": round(quantity, 4),
                "mid_price": mid_price,
                "exec_price": exec_price,
                "slippage_bps": exec_info["slippage_bps"],
                "market_impact_bps": exec_info["market_impact_bps"],
                "commission": round(commission, 4),
                "gross_value": round(gross_value, 2),
                "net_value": round(net_value, 2),
            })

        # ── Step 3: Simulate fills and update positions ───────────────────────
        cash = capital * target_weights.get("cash", 0.0)
        positions_after = dict(current)
        trade_records = []
        total_commission = 0.0
        total_slippage_cost = 0.0

        for order in orders:
            sym = order["symbol"]
            qty = order["quantity"]
            side = order["side"]
            exec_price = order["exec_price"]

            # Update position
            if side == "buy":
                positions_after[sym] = positions_after.get(sym, 0.0) + qty
            else:
                positions_after[sym] = max(0.0, positions_after.get(sym, 0.0) - qty)

            # Remove zero-quantity positions
            if positions_after.get(sym, 0.0) < 0.001:
                positions_after.pop(sym, None)

            reasoning = input_data.reasoning_chains.get(sym, [])

            # Build Trade document
            trade = Trade(
                trade_id=f"trd_{uuid.uuid4().hex[:10]}",
                portfolio_id=input_data.portfolio_id,
                run_id=self.run_id,
                symbol=sym,
                side=TradeSide(side),
                quantity=qty,
                price=exec_price,
                slippage_bps=order["slippage_bps"] + order["market_impact_bps"],
                commission=order["commission"],
                market_impact=order["market_impact_bps"],
                gross_value=order["gross_value"],
                net_value=order["net_value"],
                timestamp=datetime.utcnow(),
                status=TradeStatus.FILLED,
                reason=f"Signal-driven {'buy' if side == 'buy' else 'sell'} — target weight {target_weights.get(sym, 0):.1%}",
                reasoning_chain=reasoning,
            )
            trade_records.append(trade)
            total_commission += order["commission"]
            total_slippage_cost += (order["slippage_bps"] + order["market_impact_bps"]) / 10000 * order["gross_value"]

        # Bulk insert trades
        if trade_records:
            await Trade.insert_many(trade_records)
            self._logger.info(f"Executed {len(trade_records)} trades (total commission: ${total_commission:.2f})")

        # ── Step 4: Compute final portfolio value ─────────────────────────────
        portfolio_value = cash + sum(
            positions_after.get(sym, 0.0) * prices.get(sym, 0.0)
            for sym in positions_after
        )

        return ExecutionAgentOutput(
            timestamp=datetime.utcnow(),
            portfolio_id=input_data.portfolio_id,
            trades=[t.model_dump() for t in trade_records],
            portfolio_value=portfolio_value,
            cash_remaining=cash,
            positions_after=positions_after,
            total_commission=round(total_commission, 2),
            total_slippage_cost=round(total_slippage_cost, 2),
            agent_version=self.version,
        )
