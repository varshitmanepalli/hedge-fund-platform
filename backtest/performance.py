"""
Portfolio performance analytics for backtesting.

All functions are pure — take numpy arrays/DataFrames, return floats or dicts.
"""

import numpy as np
import pandas as pd
from typing import Optional

TRADING_DAYS = 252


def compute_cagr(equity_curve: pd.Series) -> Optional[float]:
    """Compound Annual Growth Rate."""
    if len(equity_curve) < 2 or equity_curve.iloc[0] <= 0:
        return None
    years = len(equity_curve) / TRADING_DAYS
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    return round(float(total_return ** (1 / years) - 1), 4)


def compute_total_return(equity_curve: pd.Series) -> float:
    if equity_curve.iloc[0] <= 0:
        return 0.0
    return round(float((equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1), 4)


def compute_win_rate(trades: list[dict]) -> float:
    """Fraction of trades with positive PnL."""
    closed = [t for t in trades if t.get("pnl") is not None]
    if not closed:
        return 0.0
    winners = sum(1 for t in closed if t.get("pnl", 0) > 0)
    return round(winners / len(closed), 4)


def compute_alpha_beta(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.05,
) -> tuple[Optional[float], Optional[float]]:
    """OLS regression of portfolio excess returns on benchmark excess returns."""
    rf_daily = (1 + risk_free_rate) ** (1 / TRADING_DAYS) - 1
    aligned = pd.DataFrame({
        "port": portfolio_returns - rf_daily,
        "bench": benchmark_returns - rf_daily,
    }).dropna()

    if len(aligned) < 20:
        return None, None

    from scipy import stats
    slope, intercept, r, p, se = stats.linregress(aligned["bench"], aligned["port"])

    # Alpha is annualized intercept
    alpha = float(intercept) * TRADING_DAYS
    beta = float(slope)
    return round(alpha, 4), round(beta, 4)


def compute_information_ratio(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> Optional[float]:
    """IR = mean(active returns) / tracking error."""
    active = portfolio_returns - benchmark_returns
    te = active.std()
    if te == 0:
        return None
    return round(float(active.mean() / te * np.sqrt(TRADING_DAYS)), 4)


def compute_full_performance_report(
    equity_curve: pd.Series,
    benchmark_curve: pd.Series,
    trades: list[dict],
    risk_free_rate: float = 0.05,
) -> dict:
    """
    Generate a complete performance report comparing strategy vs benchmark.
    """
    from agents.risk.metrics import (
        compute_sharpe, compute_sortino, compute_max_drawdown,
        compute_var_historical, compute_annualized_volatility
    )

    port_returns = equity_curve.pct_change().dropna()
    bench_returns = benchmark_curve.pct_change().dropna()

    sharpe = compute_sharpe(port_returns, risk_free_rate)
    sortino = compute_sortino(port_returns, risk_free_rate)
    max_dd = compute_max_drawdown(equity_curve)
    vol = compute_annualized_volatility(port_returns)
    cagr = compute_cagr(equity_curve)
    total_ret = compute_total_return(equity_curve)
    calmar = round(cagr / max_dd, 4) if cagr and max_dd > 0 else None
    win_rate = compute_win_rate(trades)
    alpha, beta = compute_alpha_beta(port_returns, bench_returns, risk_free_rate)
    ir = compute_information_ratio(port_returns, bench_returns)
    var_95 = compute_var_historical(port_returns, 0.95, equity_curve.iloc[-1])

    bench_total = compute_total_return(benchmark_curve)
    bench_cagr = compute_cagr(benchmark_curve)
    bench_sharpe = compute_sharpe(bench_returns, risk_free_rate)
    bench_max_dd = compute_max_drawdown(benchmark_curve)

    return {
        "strategy": {
            "total_return": total_ret,
            "cagr": cagr,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "max_drawdown": max_dd,
            "volatility_annualized": vol,
            "var_95": var_95,
            "win_rate": win_rate,
            "total_trades": len(trades),
            "alpha": alpha,
            "beta": beta,
            "information_ratio": ir,
        },
        "benchmark": {
            "total_return": bench_total,
            "cagr": bench_cagr,
            "sharpe_ratio": bench_sharpe,
            "max_drawdown": bench_max_dd,
        },
        "excess_return": round((total_ret or 0) - (bench_total or 0), 4),
    }
