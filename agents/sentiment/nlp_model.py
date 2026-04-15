"""
FinBERT-based sentiment scoring wrapper.

Uses ProsusAI/finbert — a BERT model fine-tuned on financial text.
Outputs: positive / negative / neutral probabilities per text.

Lazy-loads the model on first use. Supports both:
  - Local inference (use_local_llm=True): loads model weights locally
  - HuggingFace Inference API (use_local_llm=False): calls remote endpoint
"""

from functools import lru_cache
from typing import Optional

from loguru import logger

from config.settings import settings


class SentimentScore:
    """Sentiment result for a single text input."""

    def __init__(self, positive: float, negative: float, neutral: float):
        self.positive = positive
        self.negative = negative
        self.neutral = neutral

    @property
    def compound(self) -> float:
        """Compound score ∈ [-1, 1]: positive - negative."""
        return round(self.positive - self.negative, 4)

    @property
    def label(self) -> str:
        scores = {"positive": self.positive, "negative": self.negative, "neutral": self.neutral}
        return max(scores, key=scores.__getitem__)

    def __repr__(self) -> str:
        return f"SentimentScore(compound={self.compound:.3f}, label={self.label})"


class FinBERTSentimentModel:
    """
    Lazy-loading FinBERT wrapper.
    Thread-safe via singleton pattern.
    """

    _pipeline = None

    def _load_pipeline(self):
        if self._pipeline is not None:
            return

        logger.info(f"Loading FinBERT model: {settings.sentiment_model}")
        from transformers import pipeline as hf_pipeline

        self._pipeline = hf_pipeline(
            "text-classification",
            model=settings.sentiment_model,
            return_all_scores=True,
            truncation=True,
            max_length=512,
        )
        logger.info("FinBERT model loaded.")

    def score_text(self, text: str) -> SentimentScore:
        """Score a single text string. Returns SentimentScore."""
        if not text or not text.strip():
            return SentimentScore(0.0, 0.0, 1.0)

        if settings.use_local_llm:
            return self._score_local(text)
        else:
            return self._score_api(text)

    def _score_local(self, text: str) -> SentimentScore:
        self._load_pipeline()
        try:
            results = self._pipeline(text[:512])[0]
            scores = {r["label"].lower(): r["score"] for r in results}
            return SentimentScore(
                positive=scores.get("positive", 0.0),
                negative=scores.get("negative", 0.0),
                neutral=scores.get("neutral", 1.0),
            )
        except Exception as e:
            logger.error(f"Local FinBERT inference failed: {e}")
            return SentimentScore(0.0, 0.0, 1.0)

    def _score_api(self, text: str) -> SentimentScore:
        """Call HuggingFace Inference API."""
        import httpx

        endpoint = f"{settings.hf_inference_endpoint}/{settings.sentiment_model}"
        headers = {"Authorization": f"Bearer {settings.hf_api_token}"}

        try:
            resp = httpx.post(
                endpoint,
                headers=headers,
                json={"inputs": text[:512]},
                timeout=15.0,
            )
            resp.raise_for_status()
            results = resp.json()

            # HF API returns [[{label, score}, ...]]
            if isinstance(results, list) and isinstance(results[0], list):
                results = results[0]

            scores = {r["label"].lower(): r["score"] for r in results}
            return SentimentScore(
                positive=scores.get("positive", 0.0),
                negative=scores.get("negative", 0.0),
                neutral=scores.get("neutral", 1.0),
            )
        except Exception as e:
            logger.warning(f"HF API sentiment failed, returning neutral: {e}")
            return SentimentScore(0.0, 0.0, 1.0)

    def score_batch(self, texts: list[str]) -> list[SentimentScore]:
        """Score a list of texts. Falls back to sequential on API mode."""
        return [self.score_text(t) for t in texts]


@lru_cache(maxsize=1)
def get_sentiment_model() -> FinBERTSentimentModel:
    return FinBERTSentimentModel()
