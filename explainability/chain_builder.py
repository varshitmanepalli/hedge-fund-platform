"""
Reasoning Chain Builder — assembles a human-readable explanation
for every trade decision, tracing back through all agents.

Output format (per symbol):
  [
    "Macro: Bullish regime detected (confidence: 72%)",
    "Quant: Strong 12m momentum (z-score: 1.2), ranked #1/5",
    "Sentiment: Positive news sentiment (score: 0.72, 12 articles)",
    "Signal: final_score=0.81 → STRONG_BUY",
    "Risk: All constraints passed. VaR(95%)=$12,400",
    "Execution: Bought 512 shares @ $195.69 (total cost bps: 5.8)"
  ]
"""

from typing import Optional


def build_reasoning_chains(
    macro_output,
    quant_output,
    sentiment_output,
    risk_output,
    aggregation_result,
) -> dict[str, list[str]]:
    """
    Build per-symbol reasoning chains from all agent outputs.

    Returns dict: symbol → list of reasoning step strings
    """
    chains: dict[str, list[str]] = {}

    # Index quant + sentiment signals by symbol
    quant_by_sym = {s.symbol: s for s in (quant_output.signals if quant_output else [])}
    sentiment_by_sym = {s.symbol: s for s in (sentiment_output.signals if sentiment_output else [])}
    agg_by_sym = {s.symbol: s for s in (aggregation_result.signals if aggregation_result else [])}

    all_symbols = set(quant_by_sym.keys()) | set(sentiment_by_sym.keys()) | set(agg_by_sym.keys())

    for symbol in all_symbols:
        chain: list[str] = []

        # ── Step 1: Macro context ──────────────────────────────────────────────
        if macro_output:
            chain.append(
                f"Macro: {macro_output.regime.value.capitalize()} regime "
                f"(confidence: {macro_output.confidence:.0%}). "
                f"VIX={macro_output.indicators.get('vix', 'N/A')}, "
                f"YieldCurve={macro_output.indicators.get('yield_curve_slope', 'N/A')}bps"
            )

        # ── Step 2: Quant factors ─────────────────────────────────────────────
        if q := quant_by_sym.get(symbol):
            top_factors = sorted(q.factor_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            factor_str = ", ".join(f"{k}={v:+.2f}" for k, v in top_factors)
            chain.append(
                f"Quant: Composite score={q.composite_score:+.2f}, "
                f"Rank #{q.rank}/{quant_output.universe_size} ({q.percentile:.0f}th pct). "
                f"Top factors: {factor_str}"
            )

        # ── Step 3: Sentiment ─────────────────────────────────────────────────
        if s := sentiment_by_sym.get(symbol):
            chain.append(
                f"Sentiment: {s.sentiment_label.capitalize()} "
                f"(score={s.sentiment_score:+.2f}, {s.news_count} articles analyzed)"
            )
            if s.top_headlines:
                top_headline = s.top_headlines[0].get("title", "")[:80]
                chain.append(f"  → Top headline: \"{top_headline}\"")

        # ── Step 4: Signal fusion ─────────────────────────────────────────────
        if agg := agg_by_sym.get(symbol):
            weights = agg.weights_used
            chain.append(
                f"Signal Fusion: macro({weights.get('macro', 0):.0%})×{agg.macro_regime} + "
                f"quant({weights.get('quant', 0):.0%})×{agg.quant_score:+.2f} + "
                f"sentiment({weights.get('sentiment', 0):.0%})×{agg.sentiment_score:+.2f} "
                f"→ final_score={agg.final_score:+.2f} → {agg.action.value.upper()}"
            )

        # ── Step 5: Risk assessment ───────────────────────────────────────────
        if risk_output:
            if risk_output.constraints_breached:
                breaches_str = ", ".join(risk_output.constraints_breached[:2])
                chain.append(
                    f"Risk: ⚠ Constraints adjusted ({breaches_str}). "
                    f"Sharpe={risk_output.sharpe_ratio or 'N/A'}, "
                    f"MaxDD={risk_output.max_drawdown:.1%}"
                )
            else:
                chain.append(
                    f"Risk: ✓ All constraints passed. "
                    f"Sharpe={risk_output.sharpe_ratio or 'N/A'}, "
                    f"VaR(95%)=${risk_output.var_95 or 0:,.0f}, "
                    f"MaxDD={risk_output.max_drawdown:.1%}"
                )

        chains[symbol] = chain

    return chains


def format_reasoning_chain(symbol: str, chain: list[str]) -> str:
    """Format a reasoning chain for display in API responses or logs."""
    header = f"Decision rationale for {symbol}:"
    lines = "\n".join(f"  [{i+1}] {step}" for i, step in enumerate(chain))
    return f"{header}\n{lines}"


def build_trade_explanation(
    trade: dict,
    reasoning_chain: list[str],
) -> str:
    """
    Combine a trade record with its reasoning chain into a human-readable explanation.
    Used for the explainability endpoint and audit logs.
    """
    side_verb = "Bought" if trade.get("side") == "buy" else "Sold"
    symbol = trade.get("symbol", "")
    qty = trade.get("quantity", 0)
    price = trade.get("price", 0)

    explanation = (
        f"{side_verb} {qty:,.0f} shares of {symbol} @ ${price:.2f}\n"
        f"Reasoning:\n"
    )
    for i, step in enumerate(reasoning_chain):
        explanation += f"  [{i+1}] {step}\n"

    return explanation.strip()
