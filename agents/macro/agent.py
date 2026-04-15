"""
Macro Agent — identifies the current market regime using macro indicators
and generates an LLM-backed reasoning narrative.

Input:  MacroAgentInput
Output: MacroAgentOutput (wraps MacroSignalDoc)
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from agents.base import BaseAgent
from agents.macro.regime_classifier import classify_regime, build_regime_summary
from config.constants import MarketRegime
from db.models.signal import MacroSignalDoc
from llm.provider import get_llm_provider


# ── I/O Models ────────────────────────────────────────────────────────────────

class MacroAgentInput(BaseModel):
    symbols: list[str]
    macro_indicators: dict[str, Optional[float]]
    lookback_days: int = 90


class MacroAgentOutput(BaseModel):
    timestamp: datetime
    regime: MarketRegime
    confidence: float
    indicators: dict[str, Optional[float]]
    llm_reasoning: str
    agent_version: str = "1.0.0"


# ── Agent ─────────────────────────────────────────────────────────────────────

class MacroAgent(BaseAgent[MacroAgentInput, MacroAgentOutput]):
    name = "macro_agent"
    version = "1.0.0"

    def __init__(self, run_id: Optional[str] = None):
        super().__init__(run_id=run_id)
        self._llm = get_llm_provider()

    async def _run(self, input_data: MacroAgentInput) -> MacroAgentOutput:
        indicators = input_data.macro_indicators

        # Step 1: Rule-based regime classification
        regime, confidence = classify_regime(indicators)
        self._logger.info(f"Regime classified as {regime.value} ({confidence:.0%})")

        # Step 2: Build rule-based summary
        rule_summary = build_regime_summary(regime, confidence, indicators)

        # Step 3: Augment with LLM narrative reasoning
        llm_reasoning = await self._generate_reasoning(indicators, regime, confidence)
        full_reasoning = f"{rule_summary}\n\n{llm_reasoning}".strip()

        # Step 4: Persist signal to MongoDB
        signal_doc = MacroSignalDoc(
            run_id=self.run_id,
            timestamp=datetime.utcnow(),
            regime=regime,
            confidence=confidence,
            indicators={k: v for k, v in indicators.items() if v is not None},
            llm_reasoning=full_reasoning,
            agent_version=self.version,
        )
        await signal_doc.insert()
        self._logger.info(f"MacroSignal persisted (id={signal_doc.id})")

        return MacroAgentOutput(
            timestamp=signal_doc.timestamp,
            regime=regime,
            confidence=confidence,
            indicators=indicators,
            llm_reasoning=full_reasoning,
            agent_version=self.version,
        )

    async def _generate_reasoning(
        self,
        indicators: dict[str, Optional[float]],
        regime: MarketRegime,
        confidence: float,
    ) -> str:
        """Call LLM to generate a 2-3 sentence macro narrative."""
        prompt = self._build_prompt(indicators, regime, confidence)
        try:
            response = await self._llm.generate(prompt, max_tokens=200)
            return response.strip()
        except Exception as e:
            self._logger.warning(f"LLM reasoning failed, using fallback: {e}")
            return (
                f"Based on current macro conditions — VIX at {indicators.get('vix', 'N/A')}, "
                f"yield curve at {indicators.get('yield_curve_slope', 'N/A')}bps, "
                f"and GDP growth of {indicators.get('gdp_growth_qoq', 'N/A')}% — "
                f"the market regime is assessed as {regime.value} with {confidence:.0%} confidence."
            )

    @staticmethod
    def _build_prompt(
        indicators: dict[str, Optional[float]],
        regime: MarketRegime,
        confidence: float,
    ) -> str:
        ind_str = "\n".join(
            f"  - {k}: {v}" for k, v in indicators.items() if v is not None
        )
        return f"""You are a senior macro analyst at a hedge fund. Based on the following macro indicators, 
provide a concise 2-3 sentence investment narrative explaining the current market environment 
and its implications for equity allocation. Be specific and data-driven.

Macro Indicators:
{ind_str}

Preliminary regime assessment: {regime.value.upper()} (confidence: {confidence:.0%})

Narrative:"""
