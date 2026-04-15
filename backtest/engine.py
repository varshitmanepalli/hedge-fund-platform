"""
Backtesting Engine — historical simulation of the multi-agent strategy.

Design:
  - Walk-forward simulation: for each rebalance date, run agents on
    data available up to that date (no look-ahead bias)
  - Uses only yfinance price data (offline-compatible)
  - Simplified: uses rule-based regime + quant factors (no LLM calls in backtest)
  - Tracks equity curve, trades, and drawdowns day by day

Usage:
  result = await run_backtest(request)
"""

import asyncio
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel

from agents.quant.factors import compute_all_factors, cross_sectional_zscore
from agents.risk.constraints import build_constraints_from_risk_tolerance, enforce_constraints
from agents.risk.metrics import compute_max_drawdown, compute_sharpe
from agents.execution.slippage import apply_linear_slippage, compute_commission
from backtest.performance import compute_full_performance_report


# ── Request / Result Models ───────────────────────────────────────────────────

class BacktestRequest(BaseModel):
    symbols: list[str]
    start_date: date
    end_date: date
    initial_capital: float = 1_000_000.0
    rebalance_frequency: str = "monthly"   # daily | weekly | monthly
    risk_tolerance: str = "moderate"
    benchmark: str = "SPY"
    top_n: int = 5
    slippage_bps: float = 5.0
    commission_per_share: float = 0.005
    factor_weights: dict[str, float] = {}


class BacktestResult(BaseModel):
    backtest_id: str
    period: str
    performance: dict
    equity_curve: list[dict]   # [{date, portfolio_value, benchmark_value}]
    trades: list[dict]
    total_rebalances: int
    errors: list[str]


# ── Helpers ───────────────────────────────────────────────────────────────────

FREQ_DAYS = {"daily": 1, "weekly": 5, "monthly": 21, "quarterly": 63}


def _get_rebalance_dates(
    all_dates: pd.DatetimeIndex,
    freq: str,
    warmup: int = 252,
) -> list[pd.Timestamp]:
    """Return dates at which rebalancing occurs, skipping warmup period."""
    step = FREQ_DAYS.get(freq, 21)
    eligible = all_dates[warmup:]
    return [eligible[i] for i in range(0, len(eligible), step)]


def _compute_portfolio_weights(
    close_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
    top_n: int,
    factor_weights: dict,
    risk_tolerance: str,
) -> dict[str, float]:
    """
    Compute target portfolio weights using quant factors at a given date.
    No LLM — pure factor-based allocation for backtest speed.
    """
    from config.constants import DEFAULT_FACTOR_WEIGHTS
    fw = factor_weights or DEFAULT_FACTOR_WEIGHTS

    symbols = [c for c in close_df.columns if c != "SPY"]
    scores: dict[str, Optional[float]] = {}

    for sym in symbols:
        hist = close_df.loc[:as_of_date, sym].dropna()
        if len(hist) < 60:
            scores[sym] = None
            continue

        # Use a simple price-only DataFrame for factor computation
        sym_df = pd.DataFrame({"Close": hist, "Volume": 1_000_000})
        factors = compute_all_factors(sym, sym_df)

        # Weighted composite (momentum focus)
        mom_12 = factors.get("momentum_12m") or 0.0
        mom_1 = factors.get("momentum_1m") or 0.0
        vol = -(factors.get("volatility") or 0.0)  # Invert
        scores[sym] = 0.5 * mom_12 + 0.2 * mom_1 + 0.3 * vol

    zscores = cross_sectional_zscore(scores)

    # Top N by z-score
    ranked = sorted(zscores.items(), key=lambda x: x[1], reverse=True)
    buyable = [(sym, z) for sym, z in ranked[:top_n] if z > 0]

    if not buyable:
        return {"cash": 1.0}

    total_z = sum(z for _, z in buyable)
    constraints = build_constraints_from_risk_tolerance(risk_tolerance)
    raw_weights = {sym: z / total_z * (1 - constraints.min_cash_pct) for sym, z in buyable}
    raw_weights["cash"] = constraints.min_cash_pct

    return enforce_constraints(raw_weights, constraints)


# ── Main Backtest Engine ──────────────────────────────────────────────────────

async def run_backtest(request: BacktestRequest) -> BacktestResult:
    """
    Walk-forward historical backtest.

    1. Download price history for all symbols + benchmark
    2. For each rebalance date: compute factor signals → weights → simulate trades
    3. Mark-to-market portfolio daily
    4. Compute full performance report vs benchmark
    """
    import yfinance as yf
    import uuid

    backtest_id = f"bt_{uuid.uuid4().hex[:8]}"
    errors = []
    all_symbols = list(set(request.symbols + [request.benchmark]))

    logger.info(f"Backtest {backtest_id}: {request.start_date} to {request.end_date}, {len(request.symbols)} symbols")

    # ── Download price data ────────────────────────────────────────────────────
    # Add 1 year of warmup data before start_date
    warmup_start = date(request.start_date.year - 1, request.start_date.month, request.start_date.day)

    raw = yf.download(
        tickers=all_symbols,
        start=warmup_start.isoformat(),
        end=request.end_date.isoformat(),
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )

    if raw.empty:
        raise ValueError("No price data returned from yfinance")

    # Build close price DataFrame
    close_dfs = {}
    for sym in all_symbols:
        try:
            if len(all_symbols) == 1:
                close_dfs[sym] = raw["Close"]
            else:
                close_dfs[sym] = raw[sym]["Close"] if sym in raw.columns.get_level_values(0) else pd.Series()
        except Exception:
            logger.warning(f"No data for {sym}")

    close_df = pd.DataFrame(close_dfs).dropna(how="all")
    sim_dates = close_df[request.start_date.isoformat():].index

    if sim_dates.empty:
        raise ValueError("No simulation dates in range")

    rebalance_dates = _get_rebalance_dates(close_df.index, request.rebalance_frequency, warmup=252)
    rebalance_set = set(rebalance_dates)

    # ── Simulation loop ────────────────────────────────────────────────────────
    capital = request.initial_capital
    cash = capital
    positions: dict[str, float] = {}      # symbol → shares held
    current_weights: dict[str, float] = {"cash": 1.0}

    equity_curve_records: list[dict] = []
    all_trades: list[dict] = []

    bench_initial = close_df[request.benchmark].loc[sim_dates[0]]
    bench_shares = capital / bench_initial if bench_initial > 0 else 0

    for sim_date in sim_dates:
        prices = close_df.loc[sim_date]

        # ── Rebalance if scheduled ─────────────────────────────────────────────
        if sim_date in rebalance_set:
            hist_up_to = close_df.loc[:sim_date]
            target_weights = _compute_portfolio_weights(
                close_df=hist_up_to,
                as_of_date=sim_date,
                top_n=request.top_n,
                factor_weights=request.factor_weights,
                risk_tolerance=request.risk_tolerance,
            )

            # Simulate orders
            for sym, target_w in target_weights.items():
                if sym == "cash":
                    continue
                price = prices.get(sym)
                if price is None or np.isnan(price) or price <= 0:
                    continue

                target_shares_val = (capital * target_w) / price
                current_shares = positions.get(sym, 0.0)
                delta = target_shares_val - current_shares

                if abs(delta) < 0.01:
                    continue

                side = "buy" if delta > 0 else "sell"
                qty = abs(delta)
                exec_price = apply_linear_slippage(float(price), side, request.slippage_bps)
                commission = compute_commission(qty, request.commission_per_share)
                cost = qty * exec_price + commission if side == "buy" else -(qty * exec_price - commission)

                cash -= cost
                positions[sym] = max(0.0, current_shares + (qty if side == "buy" else -qty))

                all_trades.append({
                    "date": str(sim_date.date()),
                    "symbol": sym,
                    "side": side,
                    "quantity": round(qty, 4),
                    "price": round(exec_price, 4),
                    "commission": round(commission, 4),
                })

            current_weights = target_weights

        # ── Mark-to-market ─────────────────────────────────────────────────────
        equity = cash + sum(
            positions.get(sym, 0.0) * float(prices.get(sym, 0))
            for sym in positions
            if not np.isnan(prices.get(sym, np.nan))
        )
        capital = equity  # Update for weight calculations

        bench_price = prices.get(request.benchmark, bench_initial)
        bench_value = bench_shares * float(bench_price) if not np.isnan(bench_price) else capital

        equity_curve_records.append({
            "date": str(sim_date.date()),
            "portfolio_value": round(equity, 2),
            "benchmark_value": round(bench_value, 2),
        })

    # ── Performance Report ─────────────────────────────────────────────────────
    if equity_curve_records:
        eq_series = pd.Series([r["portfolio_value"] for r in equity_curve_records])
        bench_series = pd.Series([r["benchmark_value"] for r in equity_curve_records])
        performance = compute_full_performance_report(eq_series, bench_series, all_trades)
    else:
        performance = {}

    logger.info(
        f"Backtest {backtest_id} complete. "
        f"Total return: {performance.get('strategy', {}).get('total_return', 'N/A')}"
    )

    return BacktestResult(
        backtest_id=backtest_id,
        period=f"{request.start_date} to {request.end_date}",
        performance=performance,
        equity_curve=equity_curve_records,
        trades=all_trades,
        total_rebalances=len([t for t in all_trades]),
        errors=errors,
    )
