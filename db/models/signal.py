"""
Beanie ODM models for all signal types produced by agents.
"""

from datetime import datetime
from typing import Optional

from beanie import Document
from pydantic import Field

from config.constants import MarketRegime, SignalAction


class MacroSignalDoc(Document):
    run_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    regime: MarketRegime
    confidence: float
    indicators: dict = Field(default_factory=dict)
    llm_reasoning: str = ""
    agent_version: str = "1.0.0"

    class Settings:
        name = "macro_signals"
        indexes = [[("run_id", 1)], [("timestamp", -1)]]


class QuantSignalDoc(Document):
    run_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    symbol: str
    factor_scores: dict[str, float] = Field(default_factory=dict)
    composite_score: float
    rank: int
    percentile: float
    agent_version: str = "1.0.0"

    class Settings:
        name = "quant_signals"
        indexes = [
            [("run_id", 1), ("symbol", 1)],
            [("timestamp", -1)],
        ]


class SentimentSignalDoc(Document):
    run_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    symbol: str
    sentiment_score: float          # ∈ [-1, 1]
    sentiment_label: str
    news_count: int = 0
    top_headlines: list[dict] = Field(default_factory=list)
    model_used: str = ""
    agent_version: str = "1.0.0"

    class Settings:
        name = "sentiment_signals"
        indexes = [
            [("run_id", 1), ("symbol", 1)],
            [("timestamp", -1)],
        ]


class AggregatedSignalDoc(Document):
    run_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    symbol: str
    macro_regime: str
    quant_score: float
    sentiment_score: float
    final_score: float
    action: SignalAction
    weights_used: dict[str, float] = Field(default_factory=dict)

    class Settings:
        name = "aggregated_signals"
        indexes = [
            [("run_id", 1), ("symbol", 1)],
            [("timestamp", -1)],
        ]
