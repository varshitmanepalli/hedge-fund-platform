"""
Signal Aggregator — fuses outputs from Macro, Quant, and Sentiment agents
into a single per-asset action signal.

Fusion logic:
  1. Regime-conditional agent weight selection
  2. Normalize quant z-score and sentiment score to [-1, 1]
  3. Weighted sum → final_score ∈ [-1, 1]
  4. Threshold mapping → SignalAction
  5. Propose portfolio weights based on scores (long-only, top-N selection)
"""

from datetime import datetime
from typing import Optional

from loguru import logger
from pydantic import BaseModel

from agents.macro.agent import MacroAgentOutput
from agents.quant.agent import QuantAgentOutput
from agents.sentiment.agent import SentimentAgentOutput
from config.constants import (
    ACTION_THRESHOLDS,
    DEFAULT_AGENT_WEIGHTS,
    REGIME_AGENT_WEIGHTS,
    SignalAction,
)
from db.models.signal import AggregatedSignalDoc


class AggregatedSignal(BaseModel):
    symbol: str
    macro_regime: str
    quant_score: float
    sentiment_score: float
    final_score: float
    action: SignalAction
    weights_used: dict[str, float]
    proposed_weight: float = 0.0       # Portfolio allocation weight


class AggregationResult(BaseModel):
    timestamp: datetime
    signals: list[AggregatedSignal]
    proposed_weights: dict[str, float]  # symbol → portfolio weight (incl. cash)
    regime: str
    agent_weights_used: dict[str, float]


def _score_to_action(score: float) -> SignalAction:
    """Convert normalized score to discrete action."""
    if score >= ACTION_THRESHOLDS["strong_buy"]:
        return SignalAction.STRONG_BUY
    elif score >= ACTION_THRESHOLDS["buy"]:
        return SignalAction.BUY
    elif score >= ACTION_THRESHOLDS["hold"]:
        return SignalAction.HOLD
    elif score >= ACTION_THRESHOLDS["sell"]:
        return SignalAction.SELL
    else:
        return SignalAction.STRONG_SELL


def _normalize_quant_score(raw: float, max_expected: float = 2.0) -> float:
    """Clamp quant composite z-score to [-1, 1]."""
    return max(-1.0, min(1.0, raw / max_expected))


def _compute_proposed_weights(
    signals: list[AggregatedSignal],
    regime: str,
    top_n: int = 5,
    min_cash: float = 0.10,
) -> dict[str, float]:
    """
    Convert final scores into portfolio allocation weights.

    Strategy:
      - Select top_n assets with action in [buy, strong_buy]
      - Weight proportional to final_score (positive only)
      - In bear/crisis regime: reduce overall equity exposure
      - Remainder to cash
    """
    # Regime-based equity allocation cap
    max_equity = {
        "bull": 0.90,
        "neutral": 0.75,
        "bear": 0.55,
        "crisis": 0.30,
    }.get(regime, 0.75)

    buyable = [
        s for s in signals
        if s.action in [SignalAction.STRONG_BUY, SignalAction.BUY]
        and s.final_score > 0
    ]

    if not buyable:
        return {"cash": 1.0}

    # Take top N
    buyable = sorted(buyable, key=lambda x: x.final_score, reverse=True)[:top_n]

    # Proportional weights from scores
    total_score = sum(s.final_score for s in buyable)
    if total_score <= 0:
        return {"cash": 1.0}

    raw_weights = {s.symbol: (s.final_score / total_score) * max_equity for s in buyable}

    # Compute cash residual
    total_equity = sum(raw_weights.values())
    cash_weight = max(min_cash, 1.0 - total_equity)

    # Rescale equity to fit
    scale = (1.0 - cash_weight) / total_equity if total_equity > 0 else 1.0
    weights = {sym: round(w * scale, 4) for sym, w in raw_weights.items()}
    weights["cash"] = round(cash_weight, 4)

    # Normalize
    total = sum(weights.values())
    return {k: round(v / total, 4) for k, v in weights.items()}


async def aggregate_signals(
    macro_output: MacroAgentOutput,
    quant_output: QuantAgentOutput,
    sentiment_output: SentimentAgentOutput,
    run_id: str,
    top_n: int = 5,
) -> AggregationResult:
    """
    Fuse all agent outputs into per-asset action signals and portfolio weights.
    """
    regime = macro_output.regime.value

    # Pick regime-conditional agent weights
    agent_weights = REGIME_AGENT_WEIGHTS.get(regime, DEFAULT_AGENT_WEIGHTS)
    w_macro = agent_weights["macro"]
    w_quant = agent_weights["quant"]
    w_sentiment = agent_weights["sentiment"]

    # Index quant and sentiment outputs by symbol
    quant_by_symbol = {s.symbol: s for s in quant_output.signals}
    sentiment_by_symbol = {s.symbol: s for s in sentiment_output.signals}

    all_symbols = set(quant_by_symbol.keys()) | set(sentiment_by_symbol.keys())
    signals: list[AggregatedSignal] = []

    # Macro score: +1 for bull, -1 for crisis, 0 for neutral, -0.5 for bear
    macro_score_map = {"bull": 1.0, "neutral": 0.0, "bear": -0.5, "crisis": -1.0}
    macro_score = macro_score_map.get(regime, 0.0)

    for symbol in all_symbols:
        quant_sig = quant_by_symbol.get(symbol)
        sentiment_sig = sentiment_by_symbol.get(symbol)

        quant_z = _normalize_quant_score(quant_sig.composite_score if quant_sig else 0.0)
        sentiment_s = sentiment_sig.sentiment_score if sentiment_sig else 0.0

        final_score = (
            w_macro * macro_score
            + w_quant * quant_z
            + w_sentiment * sentiment_s
        )
        final_score = round(max(-1.0, min(1.0, final_score)), 4)

        action = _score_to_action(final_score)

        sig = AggregatedSignal(
            symbol=symbol,
            macro_regime=regime,
            quant_score=round(quant_z, 4),
            sentiment_score=round(sentiment_s, 4),
            final_score=final_score,
            action=action,
            weights_used={"macro": w_macro, "quant": w_quant, "sentiment": w_sentiment},
        )
        signals.append(sig)

    # Compute proposed portfolio weights
    proposed_weights = _compute_proposed_weights(signals, regime, top_n=top_n)

    # Update proposed_weight field on each signal
    for sig in signals:
        sig.proposed_weight = proposed_weights.get(sig.symbol, 0.0)

    # Persist to MongoDB
    docs = [
        AggregatedSignalDoc(
            run_id=run_id,
            symbol=sig.symbol,
            macro_regime=sig.macro_regime,
            quant_score=sig.quant_score,
            sentiment_score=sig.sentiment_score,
            final_score=sig.final_score,
            action=sig.action,
            weights_used=sig.weights_used,
        )
        for sig in signals
    ]
    if docs:
        await AggregatedSignalDoc.insert_many(docs)

    logger.info(
        f"Signal aggregation complete. Regime: {regime}. "
        f"Buy signals: {sum(1 for s in signals if s.action in [SignalAction.BUY, SignalAction.STRONG_BUY])}/"
        f"{len(signals)}"
    )

    return AggregationResult(
        timestamp=datetime.utcnow(),
        signals=signals,
        proposed_weights=proposed_weights,
        regime=regime,
        agent_weights_used=agent_weights,
    )
