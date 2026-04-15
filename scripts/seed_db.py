"""
Seed script — loads sample market data into MongoDB for development.
Run: python scripts/seed_db.py

Downloads 2 years of daily price data for DEFAULT_UNIVERSE via yfinance
and stores it in the market_data collection.
"""

import asyncio
from datetime import date, timedelta

from loguru import logger

from config.constants import DEFAULT_UNIVERSE
from db.client import init_db
from data.ingestion.price_fetcher import fetch_prices
from data.ingestion.macro_fetcher import fetch_latest_macro_indicators
from data.ingestion.news_fetcher import fetch_news


async def main():
    logger.info("Initializing database...")
    await init_db()

    symbols = DEFAULT_UNIVERSE
    start = date.today() - timedelta(days=730)

    logger.info(f"Seeding price data for {symbols}...")
    price_data = await fetch_prices(symbols, start=start, persist=True)
    logger.info(f"Seeded price data for {len(price_data)} symbols.")

    logger.info("Seeding macro indicators...")
    indicators = await fetch_latest_macro_indicators()
    logger.info(f"Macro indicators: {indicators}")

    logger.info("Seeding news articles (mock)...")
    news = await fetch_news(symbols[:5], lookback_hours=48, persist=True)
    total_articles = sum(len(v) for v in news.values())
    logger.info(f"Seeded {total_articles} news articles.")

    logger.info("Seed complete.")


if __name__ == "__main__":
    asyncio.run(main())
