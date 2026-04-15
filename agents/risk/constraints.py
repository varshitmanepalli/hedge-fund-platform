"""
Risk constraint checking and enforcement.

Constraints are checked in priority order. If a constraint is breached,
position weights are adjusted proportionally to restore compliance.
"""

from dataclasses import dataclass
from typing import Optional

from loguru import logger


@dataclass
class RiskConstraints:
    max_position_weight: float = 0.30       # Max single-asset weight
    max_var_95_pct: float = 0.02            # Max VaR as % of capital
    min_sharpe: float = 0.0                 # Min acceptable Sharpe (historical)
    max_drawdown_limit: float = 0.15        # Stop if drawdown exceeds this
    max_concentration_hhi: float = 0.30     # Max HHI
    max_leverage: float = 1.0              # No leverage by default
    min_cash_pct: float = 0.05             # Always hold at least 5% cash


def check_constraints(
    proposed_weights: dict[str, float],
    capital: float,
    var_95: Optional[float],
    sharpe: Optional[float],
    current_drawdown: float,
    constraints: RiskConstraints,
) -> list[str]:
    """
    Check all risk constraints against proposed portfolio state.

    Returns a list of breached constraint names (empty → all clear).
    """
    breaches = []

    # 1. Single-position concentration
    for symbol, weight in proposed_weights.items():
        if symbol == "cash":
            continue
        if weight > constraints.max_position_weight:
            breaches.append(f"max_position_weight:{symbol}:{weight:.2%}>{constraints.max_position_weight:.2%}")

    # 2. VaR limit
    if var_95 is not None and capital > 0:
        var_pct = var_95 / capital
        if var_pct > constraints.max_var_95_pct:
            breaches.append(f"max_var_95_pct:{var_pct:.2%}>{constraints.max_var_95_pct:.2%}")

    # 3. Minimum Sharpe (only enforce if we have enough history)
    if sharpe is not None and sharpe < constraints.min_sharpe:
        breaches.append(f"min_sharpe:{sharpe:.2f}<{constraints.min_sharpe:.2f}")

    # 4. Drawdown kill-switch
    if current_drawdown > constraints.max_drawdown_limit:
        breaches.append(
            f"max_drawdown:{current_drawdown:.2%}>{constraints.max_drawdown_limit:.2%}"
        )

    # 5. Leverage check (sum of non-cash weights should not exceed 1)
    total_invested = sum(w for k, w in proposed_weights.items() if k != "cash")
    if total_invested > constraints.max_leverage:
        breaches.append(f"max_leverage:{total_invested:.2f}>{constraints.max_leverage:.2f}")

    # 6. Minimum cash requirement
    cash_weight = proposed_weights.get("cash", 0.0)
    if cash_weight < constraints.min_cash_pct:
        breaches.append(f"min_cash_pct:{cash_weight:.2%}<{constraints.min_cash_pct:.2%}")

    return breaches


def enforce_constraints(
    proposed_weights: dict[str, float],
    constraints: RiskConstraints,
) -> dict[str, float]:
    """
    Proportionally scale down positions that breach constraints.
    Excess weight is redistributed to cash.

    Returns corrected weights that sum to 1.0.
    """
    weights = dict(proposed_weights)

    # Cap individual positions
    excess = 0.0
    for symbol in list(weights.keys()):
        if symbol == "cash":
            continue
        if weights[symbol] > constraints.max_position_weight:
            excess += weights[symbol] - constraints.max_position_weight
            weights[symbol] = constraints.max_position_weight
            logger.info(f"Capped {symbol} weight to {constraints.max_position_weight:.0%}")

    # Move excess to cash
    weights["cash"] = weights.get("cash", 0.0) + excess

    # Enforce min cash
    if weights.get("cash", 0.0) < constraints.min_cash_pct:
        deficit = constraints.min_cash_pct - weights.get("cash", 0.0)
        # Scale down all non-cash positions proportionally
        invested_symbols = [k for k in weights if k != "cash" and weights[k] > 0]
        total_invested = sum(weights[s] for s in invested_symbols)

        if total_invested > 0:
            for sym in invested_symbols:
                weights[sym] -= weights[sym] / total_invested * deficit

        weights["cash"] = weights.get("cash", 0.0) + deficit

    # Normalize to sum to 1.0
    total = sum(weights.values())
    if total > 0:
        weights = {k: round(v / total, 6) for k, v in weights.items()}

    return weights


def build_constraints_from_risk_tolerance(tolerance: str) -> RiskConstraints:
    """Map risk tolerance label → RiskConstraints config."""
    configs = {
        "conservative": RiskConstraints(
            max_position_weight=0.15,
            max_var_95_pct=0.01,
            min_sharpe=0.5,
            max_drawdown_limit=0.08,
            max_concentration_hhi=0.20,
            min_cash_pct=0.20,
        ),
        "moderate": RiskConstraints(
            max_position_weight=0.25,
            max_var_95_pct=0.02,
            min_sharpe=0.3,
            max_drawdown_limit=0.15,
            max_concentration_hhi=0.25,
            min_cash_pct=0.10,
        ),
        "aggressive": RiskConstraints(
            max_position_weight=0.40,
            max_var_95_pct=0.04,
            min_sharpe=0.0,
            max_drawdown_limit=0.25,
            max_concentration_hhi=0.40,
            min_cash_pct=0.05,
        ),
    }
    return configs.get(tolerance, RiskConstraints())
