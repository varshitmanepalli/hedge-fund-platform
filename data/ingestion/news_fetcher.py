"""
News/sentiment data ingestion from NewsAPI.
Fetches recent headlines for each symbol and persists to MongoDB.

Falls back to mock news articles if API key is not set.
"""

from datetime import datetime, timedelta
from typing import Optional

import httpx
from loguru import logger

from config.settings import settings
from db.models.news_item import NewsItem

NEWSAPI_BASE = "https://newsapi.org/v2/everything"


async def fetch_news(
    symbols: list[str],
    lookback_hours: int = 48,
    persist: bool = True,
) -> dict[str, list[NewsItem]]:
    """
    Fetch recent news articles for each symbol.

    Args:
        symbols:        List of ticker symbols
        lookback_hours: How many hours back to search
        persist:        Whether to upsert into MongoDB

    Returns:
        Dict of symbol → list of NewsItem objects
    """
    if not settings.news_api_key:
        logger.warning("NEWS_API_KEY not set — using mock news articles")
        return {s: _mock_news(s) for s in symbols}

    result: dict[str, list[NewsItem]] = {}
    since = datetime.utcnow() - timedelta(hours=lookback_hours)

    async with httpx.AsyncClient(timeout=30.0) as client:
        for symbol in symbols:
            try:
                items = await _fetch_symbol_news(client, symbol, since)
                result[symbol] = items

                if persist and items:
                    await _upsert_news(items)

                logger.info(f"Fetched {len(items)} news items for {symbol}")

            except Exception as e:
                logger.error(f"News fetch failed for {symbol}: {e}")
                result[symbol] = []

    return result


async def _fetch_symbol_news(
    client: httpx.AsyncClient,
    symbol: str,
    since: datetime,
) -> list[NewsItem]:
    """Query NewsAPI for articles mentioning a ticker symbol."""
    resp = await client.get(
        NEWSAPI_BASE,
        params={
            "q": f'"{symbol}" stock OR "{symbol}" earnings OR "{symbol}" shares',
            "from": since.isoformat() + "Z",
            "sortBy": "publishedAt",
            "language": "en",
            "pageSize": 20,
            "apiKey": settings.news_api_key,
        },
    )
    resp.raise_for_status()
    data = resp.json()

    items = []
    for art in data.get("articles", []):
        try:
            published = datetime.fromisoformat(
                art["publishedAt"].replace("Z", "+00:00")
            ).replace(tzinfo=None)

            item = NewsItem(
                symbol=symbol,
                title=art.get("title") or "",
                body=art.get("description") or "",
                source=art.get("source", {}).get("name", ""),
                url=art.get("url") or "",
                author=art.get("author") or "",
                published_at=published,
            )
            items.append(item)
        except Exception:
            continue

    return items


async def _upsert_news(items: list[NewsItem]) -> None:
    """Upsert news items by (symbol, url) to avoid duplicates."""
    from db.client import get_client
    from config.settings import settings as cfg
    from config.constants import COLLECTIONS
    from pymongo import UpdateOne

    client = get_client()
    collection = client[cfg.mongodb_db_name][COLLECTIONS["news_data"]]

    ops = [
        UpdateOne(
            {"symbol": item.symbol, "url": item.url},
            {"$set": item.model_dump(exclude={"id"})},
            upsert=True,
        )
        for item in items
        if item.url
    ]

    if ops:
        await collection.bulk_write(ops, ordered=False)


async def get_recent_news(
    symbol: str,
    lookback_hours: int = 48,
) -> list[NewsItem]:
    """Retrieve recent news for a symbol from MongoDB."""
    since = datetime.utcnow() - timedelta(hours=lookback_hours)
    docs = (
        await NewsItem.find(
            NewsItem.symbol == symbol,
            NewsItem.published_at >= since,
        )
        .sort(-NewsItem.published_at)
        .limit(50)
        .to_list()
    )
    return docs


def _mock_news(symbol: str) -> list[NewsItem]:
    """Return deterministic mock news for testing."""
    now = datetime.utcnow()
    return [
        NewsItem(
            symbol=symbol,
            title=f"{symbol} reports strong quarterly earnings, beats estimates",
            body=f"{symbol} Inc. reported Q1 results that exceeded analyst expectations.",
            source="MockNews",
            url=f"https://mock.news/{symbol}/q1-earnings",
            published_at=now - timedelta(hours=2),
        ),
        NewsItem(
            symbol=symbol,
            title=f"Analysts upgrade {symbol} to Buy on strong fundamentals",
            body=f"Three major banks raised their {symbol} price targets today.",
            source="MockFinance",
            url=f"https://mock.news/{symbol}/analyst-upgrade",
            published_at=now - timedelta(hours=8),
        ),
        NewsItem(
            symbol=symbol,
            title=f"{symbol} announces new product line amid competitive pressure",
            body=f"{symbol} unveiled new products but faces growing competition.",
            source="MockTech",
            url=f"https://mock.news/{symbol}/product-launch",
            published_at=now - timedelta(hours=24),
        ),
    ]
