"""
Unit tests for Macro Agent.
Uses MockLLMProvider — no API key required.
"""

import pytest
from unittest.mock import AsyncMock, patch
from datetime import datetime

from agents.macro.agent import MacroAgent, MacroAgentInput
from agents.macro.regime_classifier import classify_regime, build_regime_summary
from config.constants import MarketRegime


# ── Regime Classifier Tests ───────────────────────────────────────────────────

class TestRegimeClassifier:
    def test_bull_regime(self):
        indicators = {
            "vix": 14.0,
            "yield_curve_slope": 0.8,
            "gdp_growth_qoq": 3.2,
            "cpi_yoy": 2.5,
            "unemployment_rate": 3.5,
            "fed_funds_rate": 3.0,
            "credit_spread_hy": 280.0,
        }
        regime, confidence = classify_regime(indicators)
        assert regime == MarketRegime.BULL
        assert confidence > 0.5

    def test_bear_regime(self):
        indicators = {
            "vix": 28.0,
            "yield_curve_slope": -0.8,
            "gdp_growth_qoq": -0.5,
            "cpi_yoy": 7.0,
            "unemployment_rate": 5.5,
            "fed_funds_rate": 5.5,
            "credit_spread_hy": 600.0,
        }
        regime, confidence = classify_regime(indicators)
        assert regime == MarketRegime.BEAR

    def test_crisis_regime_from_vix(self):
        indicators = {
            "vix": 42.0,         # > 35 → crisis
            "yield_curve_slope": -0.3,
            "gdp_growth_qoq": 0.5,
            "cpi_yoy": 4.0,
            "unemployment_rate": 5.0,
            "fed_funds_rate": 4.5,
            "credit_spread_hy": 450.0,
        }
        regime, confidence = classify_regime(indicators)
        assert regime == MarketRegime.CRISIS

    def test_confidence_is_bounded(self):
        """Confidence must always be in [0.40, 0.95]."""
        indicators = {
            "vix": 16.0, "yield_curve_slope": 0.4, "gdp_growth_qoq": 2.0,
            "cpi_yoy": 3.0, "unemployment_rate": 4.0, "fed_funds_rate": 4.0,
            "credit_spread_hy": 350.0,
        }
        _, confidence = classify_regime(indicators)
        assert 0.40 <= confidence <= 0.95

    def test_neutral_regime_mixed_signals(self):
        """Conflicting signals → neutral."""
        indicators = {
            "vix": 22.0,           # elevated but not crisis
            "yield_curve_slope": 0.1,   # slightly positive
            "gdp_growth_qoq": 1.0,      # slow but positive
            "cpi_yoy": 4.5,
            "unemployment_rate": 4.2,
            "fed_funds_rate": 5.0,
            "credit_spread_hy": 420.0,
        }
        regime, _ = classify_regime(indicators)
        # Should be neutral or bear — not strong bull
        assert regime in [MarketRegime.NEUTRAL, MarketRegime.BEAR]

    def test_none_indicators_handled(self):
        """Missing indicators should not crash the classifier."""
        indicators = {"vix": 18.0, "yield_curve_slope": 0.3}
        regime, confidence = classify_regime(indicators)
        assert isinstance(regime, MarketRegime)
        assert 0 <= confidence <= 1

    def test_build_regime_summary(self):
        indicators = {"vix": 16.0, "yield_curve_slope": 0.5, "gdp_growth_qoq": 2.5}
        summary = build_regime_summary(MarketRegime.BULL, 0.72, indicators)
        assert "BULL" in summary
        assert "72%" in summary


# ── Macro Agent Tests ─────────────────────────────────────────────────────────

BULL_INPUT = MacroAgentInput(
    symbols=["AAPL", "MSFT"],
    macro_indicators={
        "vix": 14.0,
        "yield_curve_slope": 0.8,
        "gdp_growth_qoq": 3.2,
        "cpi_yoy": 2.5,
        "unemployment_rate": 3.5,
        "fed_funds_rate": 3.0,
        "credit_spread_hy": 280.0,
    },
)


@pytest.mark.asyncio
class TestMacroAgent:
    async def test_run_returns_correct_regime(self):
        agent = MacroAgent(run_id="test_run_001")

        # Patch out DB write and LLM call
        with (
            patch.object(agent._llm, "generate", new_callable=AsyncMock,
                         return_value="Bullish macro conditions."),
            patch("db.models.signal.MacroSignalDoc.insert", new_callable=AsyncMock),
            patch("db.models.agent_log.AgentLog.insert", new_callable=AsyncMock),
        ):
            output = await agent.run(BULL_INPUT)

        assert output.regime == MarketRegime.BULL
        assert output.confidence > 0.5
        assert isinstance(output.llm_reasoning, str)
        assert len(output.llm_reasoning) > 10
        assert output.agent_version == "1.0.0"

    async def test_run_uses_mock_llm_on_failure(self):
        """Agent should not crash if LLM call fails — uses fallback reasoning."""
        agent = MacroAgent(run_id="test_run_002")

        with (
            patch.object(agent._llm, "generate", new_callable=AsyncMock,
                         side_effect=Exception("LLM timeout")),
            patch("db.models.signal.MacroSignalDoc.insert", new_callable=AsyncMock),
            patch("db.models.agent_log.AgentLog.insert", new_callable=AsyncMock),
        ):
            output = await agent.run(BULL_INPUT)

        # Should succeed with fallback reasoning
        assert output.regime == MarketRegime.BULL
        assert "VIX" in output.llm_reasoning or "confidence" in output.llm_reasoning

    async def test_output_is_timestamped(self):
        agent = MacroAgent(run_id="test_run_003")
        with (
            patch.object(agent._llm, "generate", new_callable=AsyncMock,
                         return_value="Test response"),
            patch("db.models.signal.MacroSignalDoc.insert", new_callable=AsyncMock),
            patch("db.models.agent_log.AgentLog.insert", new_callable=AsyncMock),
        ):
            output = await agent.run(BULL_INPUT)

        assert isinstance(output.timestamp, datetime)
