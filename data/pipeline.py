"""
Data ingestion pipeline orchestrator.
Runs price, macro, and news ingestion in parallel.
Called at pipeline startup to ensure fresh data before agents run.
"""

import asyncio
from datetime import date, timedelta
from typing import Optional

import pandas as pd
from loguru import logger

from data.ingestion.price_fetcher import fetch_prices
from data.ingestion.macro_fetcher import fetch_latest_macro_indicators
from data.ingestion.news_fetcher import fetch_news
from db.models.news_item import NewsItem


class DataIngestionResult:
    """Typed container for all ingested data passed to agents."""

    def __init__(
        self,
        price_data: dict[str, pd.DataFrame],
        macro_indicators: dict[str, Optional[float]],
        news_data: dict[str, list[NewsItem]],
    ):
        self.price_data = price_data
        self.macro_indicators = macro_indicators
        self.news_data = news_data

    def get_close_prices(self) -> pd.DataFrame:
        """Return a single DataFrame of close prices, symbols as columns."""
        closes = {}
        for symbol, df in self.price_data.items():
            if not df.empty and "Close" in df.columns:
                closes[symbol] = df["Close"]
        if not closes:
            return pd.DataFrame()
        return pd.DataFrame(closes).sort_index()

    def symbols_with_data(self) -> list[str]:
        return [s for s, df in self.price_data.items() if not df.empty]


async def run_ingestion(
    symbols: list[str],
    lookback_days: int = 252,
    news_lookback_hours: int = 48,
    persist: bool = True,
) -> DataIngestionResult:
    """
    Run all data ingestion pipelines in parallel.

    Args:
        symbols:             Ticker symbols to fetch
        lookback_days:       Historical price window
        news_lookback_hours: Recent news window for sentiment
        persist:             Whether to write to MongoDB

    Returns:
        DataIngestionResult with prices, macro, and news
    """
    start_date = date.today() - timedelta(days=lookback_days)

    logger.info(
        f"Starting data ingestion for {len(symbols)} symbols "
        f"({lookback_days}d prices, {news_lookback_hours}h news)"
    )

    # Run all ingestion tasks concurrently
    price_task = asyncio.create_task(
        fetch_prices(symbols, start=start_date, persist=persist)
    )
    macro_task = asyncio.create_task(fetch_latest_macro_indicators())
    news_task = asyncio.create_task(
        fetch_news(symbols, lookback_hours=news_lookback_hours, persist=persist)
    )

    price_data, macro_indicators, news_data = await asyncio.gather(
        price_task, macro_task, news_task, return_exceptions=False
    )

    result = DataIngestionResult(
        price_data=price_data,
        macro_indicators=macro_indicators,
        news_data=news_data,
    )

    valid_symbols = result.symbols_with_data()
    logger.info(
        f"Ingestion complete: {len(valid_symbols)} symbols with price data, "
        f"{len(macro_indicators)} macro indicators, "
        f"{sum(len(v) for v in news_data.values())} news articles"
    )

    return result
