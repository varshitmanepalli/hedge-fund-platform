"""
Slippage and market impact models for trade execution simulation.

Models:
  - Linear slippage: fixed bps applied to all orders
  - SQRT market impact: impact proportional to sqrt(order_size / avg_daily_volume)
  - Combined: linear slippage + sqrt market impact
"""

import math
from typing import Optional


def apply_linear_slippage(
    price: float,
    side: str,
    slippage_bps: float = 5.0,
) -> float:
    """
    Apply fixed basis-point slippage to execution price.
    Buy orders: price increases. Sell orders: price decreases.
    """
    slippage_factor = slippage_bps / 10_000
    if side == "buy":
        return price * (1 + slippage_factor)
    else:
        return price * (1 - slippage_factor)


def apply_sqrt_market_impact(
    price: float,
    side: str,
    order_value: float,
    avg_daily_volume_value: float,
    impact_coefficient: float = 0.1,
) -> tuple[float, float]:
    """
    Square-root market impact model.
    Impact = coefficient × sqrt(participation_rate)
    where participation_rate = order_value / avg_daily_volume_value

    Returns: (adjusted_price, impact_bps)
    """
    if avg_daily_volume_value <= 0:
        return price, 0.0

    participation = order_value / avg_daily_volume_value
    impact_fraction = impact_coefficient * math.sqrt(participation)

    if side == "buy":
        adjusted = price * (1 + impact_fraction)
    else:
        adjusted = price * (1 - impact_fraction)

    impact_bps = impact_fraction * 10_000
    return adjusted, round(impact_bps, 2)


def compute_commission(
    quantity: float,
    commission_per_share: float = 0.005,
    min_commission: float = 1.0,
) -> float:
    """Commission with a minimum ticket fee."""
    return max(min_commission, quantity * commission_per_share)


def compute_execution_price(
    mid_price: float,
    side: str,
    order_quantity: float,
    slippage_bps: float = 5.0,
    avg_daily_volume: Optional[float] = None,
    impact_coefficient: float = 0.1,
) -> dict:
    """
    Full execution price computation combining all cost components.

    Returns dict with:
      - execution_price: final price after all costs
      - slippage_bps: linear slippage applied
      - market_impact_bps: sqrt impact applied
      - total_cost_bps: combined cost in basis points
    """
    # 1. Linear slippage
    price_after_slippage = apply_linear_slippage(mid_price, side, slippage_bps)

    # 2. Market impact
    impact_bps = 0.0
    if avg_daily_volume is not None and avg_daily_volume > 0:
        order_value = order_quantity * mid_price
        adv_value = avg_daily_volume * mid_price
        price_after_slippage, impact_bps = apply_sqrt_market_impact(
            price=price_after_slippage,
            side=side,
            order_value=order_value,
            avg_daily_volume_value=adv_value,
            impact_coefficient=impact_coefficient,
        )

    total_cost_bps = slippage_bps + impact_bps

    return {
        "execution_price": round(price_after_slippage, 4),
        "slippage_bps": slippage_bps,
        "market_impact_bps": round(impact_bps, 2),
        "total_cost_bps": round(total_cost_bps, 2),
    }
