"""
Quant Agent — computes cross-sectional factor scores and ranks assets.

Pipeline:
  1. Compute raw factors for each asset (momentum, vol, volume trend, RSI, beta)
  2. Cross-sectionally z-score each factor across the universe
  3. Invert volatility z-score (lower vol → better score)
  4. Weighted sum → composite_score
  5. Rank assets, compute percentiles
  6. Persist QuantSignalDoc to MongoDB
"""

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel

from agents.base import BaseAgent
from agents.quant.factors import compute_all_factors, cross_sectional_zscore
from config.constants import DEFAULT_FACTOR_WEIGHTS
from db.models.signal import QuantSignalDoc


# ── I/O Models ────────────────────────────────────────────────────────────────

class QuantAgentInput(BaseModel):
    symbols: list[str]
    price_data: dict[str, list[dict]]   # symbol → list of OHLCV dicts
    benchmark_data: list[dict] = []     # SPY OHLCV (for beta)
    factor_weights: dict[str, float] = DEFAULT_FACTOR_WEIGHTS

    model_config = {"arbitrary_types_allowed": True}


class AssetQuantSignal(BaseModel):
    symbol: str
    factor_scores: dict[str, float]
    raw_factors: dict[str, Optional[float]]
    composite_score: float
    rank: int
    percentile: float


class QuantAgentOutput(BaseModel):
    timestamp: datetime
    signals: list[AssetQuantSignal]
    factor_weights_used: dict[str, float]
    universe_size: int
    agent_version: str = "1.0.0"


# ── Agent ─────────────────────────────────────────────────────────────────────

class QuantAgent(BaseAgent[QuantAgentInput, QuantAgentOutput]):
    name = "quant_agent"
    version = "1.0.0"

    async def _run(self, input_data: QuantAgentInput) -> QuantAgentOutput:
        factor_weights = input_data.factor_weights
        symbols = input_data.symbols

        # Reconstruct DataFrames from dict records
        price_dfs: dict[str, pd.DataFrame] = {
            sym: pd.DataFrame(records).set_index("timestamp")
            for sym, records in input_data.price_data.items()
            if records
        }

        benchmark_df: Optional[pd.DataFrame] = None
        if input_data.benchmark_data:
            benchmark_df = pd.DataFrame(input_data.benchmark_data).set_index("timestamp")

        # ── Step 1: Compute raw factors per asset ─────────────────────────────
        raw_factors_by_symbol: dict[str, dict] = {}
        for symbol in symbols:
            df = price_dfs.get(symbol, pd.DataFrame())
            raw_factors_by_symbol[symbol] = compute_all_factors(
                symbol=symbol,
                price_df=df,
                benchmark_df=benchmark_df,
            )

        # ── Step 2: Cross-sectional z-score each factor ───────────────────────
        factor_names = list(factor_weights.keys())
        zscored: dict[str, dict[str, float]] = {}  # factor → {symbol: zscore}

        for factor in factor_names:
            raw_by_symbol = {sym: raw_factors_by_symbol[sym].get(factor) for sym in symbols}
            z = cross_sectional_zscore(raw_by_symbol)

            # Invert volatility: lower vol = higher rank
            if factor == "volatility":
                z = {sym: -val for sym, val in z.items()}

            zscored[factor] = z

        # ── Step 3: Weighted composite score ──────────────────────────────────
        total_weight = sum(factor_weights.get(f, 0) for f in factor_names)
        composite: dict[str, float] = {}

        for symbol in symbols:
            score = sum(
                factor_weights.get(f, 0) * zscored[f].get(symbol, 0.0)
                for f in factor_names
            )
            composite[symbol] = score / total_weight if total_weight > 0 else 0.0

        # ── Step 4: Rank and percentile ───────────────────────────────────────
        sorted_symbols = sorted(symbols, key=lambda s: composite[s], reverse=True)
        n = len(sorted_symbols)

        signals: list[AssetQuantSignal] = []
        docs_to_insert: list[QuantSignalDoc] = []

        for rank_idx, symbol in enumerate(sorted_symbols):
            rank = rank_idx + 1
            percentile = ((n - rank_idx) / n) * 100

            factor_zscores = {f: round(zscored[f].get(symbol, 0.0), 4) for f in factor_names}

            sig = AssetQuantSignal(
                symbol=symbol,
                factor_scores=factor_zscores,
                raw_factors=raw_factors_by_symbol[symbol],
                composite_score=round(composite[symbol], 4),
                rank=rank,
                percentile=round(percentile, 1),
            )
            signals.append(sig)

            docs_to_insert.append(
                QuantSignalDoc(
                    run_id=self.run_id,
                    timestamp=datetime.utcnow(),
                    symbol=symbol,
                    factor_scores=factor_zscores,
                    composite_score=sig.composite_score,
                    rank=rank,
                    percentile=sig.percentile,
                    agent_version=self.version,
                )
            )

        # Bulk insert
        if docs_to_insert:
            await QuantSignalDoc.insert_many(docs_to_insert)

        self._logger.info(
            f"Ranked {n} assets. Top 3: "
            + ", ".join(f"{s.symbol}({s.composite_score:.2f})" for s in signals[:3])
        )

        return QuantAgentOutput(
            timestamp=datetime.utcnow(),
            signals=signals,
            factor_weights_used=factor_weights,
            universe_size=n,
            agent_version=self.version,
        )
