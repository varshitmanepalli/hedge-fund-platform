"""
Rule-based + heuristic market regime classifier.

Regime determination logic:
  1. Crisis: VIX > 35 OR credit spread > 700bps → override everything
  2. Bear: yield curve inverted (<-50bps) AND VIX elevated (>25) AND GDP declining
  3. Bull: yield curve positive AND VIX low (<20) AND GDP growing
  4. Neutral: everything else

Confidence scoring uses a weighted voting system across indicators.
"""

from config.constants import MarketRegime, VIX_CRISIS_THRESHOLD, VIX_ELEVATED_THRESHOLD
from typing import Optional


def classify_regime(
    indicators: dict[str, Optional[float]],
) -> tuple[MarketRegime, float]:
    """
    Determine market regime from macro indicators.

    Returns:
        (regime, confidence) where confidence ∈ [0, 1]
    """
    vix = indicators.get("vix") or 20.0
    yield_slope = indicators.get("yield_curve_slope") or 0.0
    gdp_growth = indicators.get("gdp_growth_qoq") or 2.0
    credit_spread = indicators.get("credit_spread_hy") or 400.0
    unemployment = indicators.get("unemployment_rate") or 4.0
    cpi = indicators.get("cpi_yoy") or 3.0
    fed_rate = indicators.get("fed_funds_rate") or 4.0

    votes: dict[str, float] = {"bull": 0.0, "bear": 0.0, "neutral": 0.0, "crisis": 0.0}

    # ── Crisis signals (high weight) ──────────────────────────────────────────
    if vix > VIX_CRISIS_THRESHOLD:
        votes["crisis"] += 3.0
    if credit_spread > 700:
        votes["crisis"] += 2.0
    if unemployment > 8.0:
        votes["crisis"] += 1.5

    # ── Yield curve ───────────────────────────────────────────────────────────
    if yield_slope > 0.5:
        votes["bull"] += 2.0
    elif 0.0 < yield_slope <= 0.5:
        votes["bull"] += 1.0
        votes["neutral"] += 0.5
    elif -0.5 <= yield_slope <= 0.0:
        votes["neutral"] += 1.0
        votes["bear"] += 0.5
    else:  # < -0.5 (inverted)
        votes["bear"] += 2.0

    # ── VIX ───────────────────────────────────────────────────────────────────
    if vix < 15:
        votes["bull"] += 2.0
    elif vix < VIX_ELEVATED_THRESHOLD:
        votes["bull"] += 1.0
    elif vix < VIX_CRISIS_THRESHOLD:
        votes["bear"] += 1.5
        votes["neutral"] += 0.5

    # ── GDP growth ────────────────────────────────────────────────────────────
    if gdp_growth > 2.5:
        votes["bull"] += 1.5
    elif gdp_growth > 0:
        votes["neutral"] += 1.0
    else:
        votes["bear"] += 2.0

    # ── Credit spreads ────────────────────────────────────────────────────────
    if credit_spread < 300:
        votes["bull"] += 1.0
    elif credit_spread < 500:
        votes["neutral"] += 0.5
    elif credit_spread < 700:
        votes["bear"] += 1.0

    # ── Fed policy stance (high rate + declining growth → restrictive) ────────
    real_rate = fed_rate - cpi
    if real_rate > 2.0:    # Very restrictive
        votes["bear"] += 1.0
    elif real_rate > 0:    # Mildly restrictive
        votes["neutral"] += 0.5
    else:                  # Accommodative
        votes["bull"] += 0.5

    # ── Determine winner ──────────────────────────────────────────────────────
    total_votes = sum(votes.values())
    if total_votes == 0:
        return MarketRegime.NEUTRAL, 0.5

    best_regime = max(votes, key=lambda k: votes[k])
    confidence = votes[best_regime] / total_votes

    # Clamp confidence to reasonable bounds
    confidence = max(0.40, min(0.95, confidence))

    return MarketRegime(best_regime), round(confidence, 3)


def build_regime_summary(
    regime: MarketRegime,
    confidence: float,
    indicators: dict[str, Optional[float]],
) -> str:
    """Build a human-readable summary of the regime assessment."""
    vix = indicators.get("vix", "N/A")
    yield_slope = indicators.get("yield_curve_slope", "N/A")
    gdp = indicators.get("gdp_growth_qoq", "N/A")

    return (
        f"Regime: {regime.value.upper()} (confidence: {confidence:.0%}). "
        f"Key drivers: VIX={vix}, yield_curve={yield_slope}bps, GDP={gdp}%QoQ. "
        f"{'Elevated volatility warrants defensive positioning.' if regime in [MarketRegime.BEAR, MarketRegime.CRISIS] else 'Conditions support risk-on allocation.'}"
    )
