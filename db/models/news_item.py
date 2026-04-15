"""
Beanie ODM model for news articles used by the Sentiment Agent.
"""

from datetime import datetime
from typing import Optional

from beanie import Document
from pydantic import Field


class NewsItem(Document):
    symbol: str                         # Primary ticker this article relates to
    related_symbols: list[str] = Field(default_factory=list)
    title: str
    body: str = ""
    source: str = ""
    url: str = ""
    author: str = ""
    published_at: datetime
    fetched_at: datetime = Field(default_factory=datetime.utcnow)
    sentiment_score: Optional[float] = None   # Populated after Sentiment Agent runs
    sentiment_label: Optional[str] = None

    class Settings:
        name = "news_data"
        indexes = [
            [("symbol", 1), ("published_at", -1)],
            [("published_at", -1)],
        ]
