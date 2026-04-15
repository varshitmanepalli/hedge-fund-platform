"""
Sentiment Agent — scores news sentiment per asset using FinBERT.

Pipeline:
  1. Receive news articles per symbol (pre-fetched by data pipeline)
  2. Score each article headline + body snippet with FinBERT
  3. Aggregate scores with recency weighting (exponential decay)
  4. Return SentimentSignal per asset
  5. Persist to MongoDB
"""

import math
from datetime import datetime, timedelta
from typing import Optional

from pydantic import BaseModel

from agents.base import BaseAgent
from agents.sentiment.nlp_model import get_sentiment_model
from db.models.signal import SentimentSignalDoc


# ── I/O Models ────────────────────────────────────────────────────────────────

class NewsArticle(BaseModel):
    symbol: str
    title: str
    body: str = ""
    source: str = ""
    url: str = ""
    published_at: datetime


class SentimentAgentInput(BaseModel):
    symbols: list[str]
    news_articles: dict[str, list[NewsArticle]]   # symbol → articles
    lookback_hours: int = 48
    recency_halflife_hours: float = 12.0          # Exp decay halflife


class AssetSentimentSignal(BaseModel):
    symbol: str
    sentiment_score: float          # ∈ [-1, 1]
    sentiment_label: str
    news_count: int
    top_headlines: list[dict]       # [{title, score, source, url}]
    model_used: str


class SentimentAgentOutput(BaseModel):
    timestamp: datetime
    signals: list[AssetSentimentSignal]
    model_used: str
    agent_version: str = "1.0.0"


# ── Agent ─────────────────────────────────────────────────────────────────────

class SentimentAgent(BaseAgent[SentimentAgentInput, SentimentAgentOutput]):
    name = "sentiment_agent"
    version = "1.0.0"

    def __init__(self, run_id: Optional[str] = None):
        super().__init__(run_id=run_id)
        self._model = get_sentiment_model()

    async def _run(self, input_data: SentimentAgentInput) -> SentimentAgentOutput:
        now = datetime.utcnow()
        signals: list[AssetSentimentSignal] = []
        docs_to_insert: list[SentimentSignalDoc] = []

        for symbol in input_data.symbols:
            articles = input_data.news_articles.get(symbol, [])

            if not articles:
                # No news → neutral signal
                sig = AssetSentimentSignal(
                    symbol=symbol,
                    sentiment_score=0.0,
                    sentiment_label="neutral",
                    news_count=0,
                    top_headlines=[],
                    model_used=self._model_name(),
                )
            else:
                sig = self._score_symbol(
                    symbol=symbol,
                    articles=articles,
                    now=now,
                    halflife_hours=input_data.recency_halflife_hours,
                )

            signals.append(sig)
            docs_to_insert.append(
                SentimentSignalDoc(
                    run_id=self.run_id,
                    timestamp=now,
                    symbol=symbol,
                    sentiment_score=sig.sentiment_score,
                    sentiment_label=sig.sentiment_label,
                    news_count=sig.news_count,
                    top_headlines=sig.top_headlines,
                    model_used=sig.model_used,
                    agent_version=self.version,
                )
            )

        if docs_to_insert:
            await SentimentSignalDoc.insert_many(docs_to_insert)

        self._logger.info(
            f"Scored sentiment for {len(signals)} symbols. "
            + " | ".join(f"{s.symbol}:{s.sentiment_score:.2f}" for s in signals)
        )

        return SentimentAgentOutput(
            timestamp=now,
            signals=signals,
            model_used=self._model_name(),
            agent_version=self.version,
        )

    def _score_symbol(
        self,
        symbol: str,
        articles: list[NewsArticle],
        now: datetime,
        halflife_hours: float,
    ) -> AssetSentimentSignal:
        """Score and aggregate all articles for one symbol."""
        decay_lambda = math.log(2) / halflife_hours

        scored_articles = []
        for art in articles:
            text = f"{art.title}. {art.body[:200]}".strip()
            score_obj = self._model.score_text(text)

            # Exponential recency weight
            age_hours = max(0, (now - art.published_at).total_seconds() / 3600)
            weight = math.exp(-decay_lambda * age_hours)

            scored_articles.append({
                "title": art.title,
                "score": score_obj.compound,
                "source": art.source,
                "url": art.url,
                "weight": weight,
            })

        # Weighted average compound score
        total_weight = sum(a["weight"] for a in scored_articles)
        if total_weight > 0:
            weighted_score = sum(
                a["score"] * a["weight"] for a in scored_articles
            ) / total_weight
        else:
            weighted_score = 0.0

        weighted_score = round(max(-1.0, min(1.0, weighted_score)), 4)

        label = (
            "positive" if weighted_score > 0.1
            else "negative" if weighted_score < -0.1
            else "neutral"
        )

        # Top 3 headlines sorted by absolute score
        top = sorted(scored_articles, key=lambda x: abs(x["score"]), reverse=True)[:3]
        top_headlines = [{k: v for k, v in h.items() if k != "weight"} for h in top]

        return AssetSentimentSignal(
            symbol=symbol,
            sentiment_score=weighted_score,
            sentiment_label=label,
            news_count=len(articles),
            top_headlines=top_headlines,
            model_used=self._model_name(),
        )

    def _model_name(self) -> str:
        from config.settings import settings
        return settings.sentiment_model
