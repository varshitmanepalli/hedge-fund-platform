"""
MongoDB async client initialization using Motor + Beanie.
Call `init_db()` once at application startup.
"""

import motor.motor_asyncio
from beanie import init_beanie
from loguru import logger

from config.settings import settings
from db.models.agent_log import AgentLog
from db.models.asset import Asset
from db.models.market_bar import MarketBar
from db.models.news_item import NewsItem
from db.models.portfolio import Portfolio
from db.models.risk_metrics import RiskMetricsDoc
from db.models.signal import (
    AggregatedSignalDoc,
    MacroSignalDoc,
    QuantSignalDoc,
    SentimentSignalDoc,
)
from db.models.trade import Trade

# All Beanie document models — order does not matter
DOCUMENT_MODELS = [
    Asset,
    MarketBar,
    NewsItem,
    MacroSignalDoc,
    QuantSignalDoc,
    SentimentSignalDoc,
    AggregatedSignalDoc,
    Portfolio,
    Trade,
    RiskMetricsDoc,
    AgentLog,
]

_client: motor.motor_asyncio.AsyncIOMotorClient | None = None


async def init_db() -> None:
    """
    Initialize Motor client and Beanie ODM.
    Must be called once before any DB operations (FastAPI lifespan or test setup).
    """
    global _client
    _client = motor.motor_asyncio.AsyncIOMotorClient(settings.mongodb_uri)
    db = _client[settings.mongodb_db_name]

    await init_beanie(database=db, document_models=DOCUMENT_MODELS)
    logger.info(
        f"MongoDB connected: {settings.mongodb_uri} / {settings.mongodb_db_name}"
    )


async def close_db() -> None:
    global _client
    if _client:
        _client.close()
        logger.info("MongoDB connection closed.")


def get_client() -> motor.motor_asyncio.AsyncIOMotorClient:
    if _client is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _client
