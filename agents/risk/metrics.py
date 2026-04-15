"""
Risk metric computation library.

All functions are pure — they take numpy arrays or pandas Series
and return floats. No side effects, fully testable in isolation.
"""

import numpy as np
import pandas as pd
from typing import Optional


TRADING_DAYS = 252


def compute_sharpe(
    returns: pd.Series,
    risk_free_rate: float = 0.05,
    annualize: bool = True,
) -> Optional[float]:
    """
    Annualized Sharpe ratio.
    risk_free_rate: annual, e.g. 0.05 for 5%
    """
    if len(returns) < 5:
        return None

    rf_daily = (1 + risk_free_rate) ** (1 / TRADING_DAYS) - 1
    excess = returns - rf_daily
    std = excess.std()

    if std == 0 or np.isnan(std):
        return None

    sharpe = excess.mean() / std
    if annualize:
        sharpe *= np.sqrt(TRADING_DAYS)

    return round(float(sharpe), 4)


def compute_sortino(
    returns: pd.Series,
    risk_free_rate: float = 0.05,
    annualize: bool = True,
) -> Optional[float]:
    """
    Sortino ratio (penalizes only downside volatility).
    """
    if len(returns) < 5:
        return None

    rf_daily = (1 + risk_free_rate) ** (1 / TRADING_DAYS) - 1
    excess = returns - rf_daily
    downside = returns[returns < rf_daily]

    if len(downside) < 2:
        return None

    downside_std = downside.std()
    if downside_std == 0:
        return None

    sortino = excess.mean() / downside_std
    if annualize:
        sortino *= np.sqrt(TRADING_DAYS)

    return round(float(sortino), 4)


def compute_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Maximum peak-to-trough drawdown as a positive fraction.
    e.g. 0.15 = 15% max drawdown
    """
    if len(equity_curve) < 2:
        return 0.0

    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_dd = abs(drawdown.min())
    return round(float(max_dd), 4)


def compute_current_drawdown(equity_curve: pd.Series) -> float:
    """
    Current drawdown from the most recent peak.
    """
    if len(equity_curve) < 2:
        return 0.0

    peak = equity_curve.max()
    current = equity_curve.iloc[-1]

    if peak <= 0:
        return 0.0

    return round(float(abs((current - peak) / peak)), 4)


def compute_var_historical(
    returns: pd.Series,
    confidence: float = 0.95,
    portfolio_value: float = 1_000_000,
) -> Optional[float]:
    """
    Historical simulation VaR in dollar terms.
    confidence=0.95 → 95% 1-day VaR
    """
    if len(returns) < 30:
        return None

    cutoff = np.percentile(returns, (1 - confidence) * 100)
    var_dollar = abs(cutoff * portfolio_value)
    return round(float(var_dollar), 2)


def compute_cvar_historical(
    returns: pd.Series,
    confidence: float = 0.95,
    portfolio_value: float = 1_000_000,
) -> Optional[float]:
    """
    Conditional VaR (Expected Shortfall) in dollar terms.
    Average of losses beyond the VaR threshold.
    """
    if len(returns) < 30:
        return None

    cutoff = np.percentile(returns, (1 - confidence) * 100)
    tail_losses = returns[returns <= cutoff]

    if tail_losses.empty:
        return None

    cvar_dollar = abs(tail_losses.mean() * portfolio_value)
    return round(float(cvar_dollar), 2)


def compute_beta_to_market(
    portfolio_returns: pd.Series,
    market_returns: pd.Series,
) -> Optional[float]:
    """OLS beta of portfolio to market benchmark."""
    aligned = pd.DataFrame({"port": portfolio_returns, "mkt": market_returns}).dropna()
    if len(aligned) < 20:
        return None

    cov = aligned.cov().iloc[0, 1]
    var = aligned["mkt"].var()

    return round(float(cov / var), 4) if var > 0 else None


def compute_hhi_concentration(weights: dict[str, float]) -> float:
    """
    Herfindahl-Hirschman Index — measures portfolio concentration.
    HHI = sum(w_i^2). Range: [1/n, 1]. Higher = more concentrated.
    """
    total = sum(weights.values())
    if total == 0:
        return 0.0

    normalized = [w / total for w in weights.values()]
    hhi = sum(w ** 2 for w in normalized)
    return round(float(hhi), 4)


def compute_annualized_volatility(returns: pd.Series) -> Optional[float]:
    """Annualized portfolio return volatility."""
    if len(returns) < 5:
        return None
    return round(float(returns.std() * np.sqrt(TRADING_DAYS)), 4)


def compute_calmar(
    total_return_annualized: float,
    max_drawdown: float,
) -> Optional[float]:
    """Calmar ratio = CAGR / max drawdown."""
    if max_drawdown == 0:
        return None
    return round(total_return_annualized / max_drawdown, 4)
