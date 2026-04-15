"""
Price data ingestion from Yahoo Finance via yfinance.
Fetches OHLCV daily bars and persists to MongoDB market_data collection.

Design decisions:
- yfinance as primary source (free, reliable for equities)
- Bulk upsert on (symbol, timestamp) to avoid duplicates on re-runs
- Returns normalized list of MarketBar documents
"""

from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from db.models.market_bar import MarketBar


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def fetch_prices(
    symbols: list[str],
    start: date,
    end: Optional[date] = None,
    interval: str = "1d",
    persist: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Download price history for a list of symbols.

    Args:
        symbols:  List of ticker symbols e.g. ["AAPL", "MSFT"]
        start:    Start date (inclusive)
        end:      End date (exclusive), defaults to today
        interval: Bar interval — "1d", "1h", "5m"
        persist:  Whether to upsert bars into MongoDB

    Returns:
        Dict of symbol → OHLCV DataFrame with DatetimeIndex
    """
    end = end or date.today()
    logger.info(f"Fetching prices for {symbols} from {start} to {end}")

    # yfinance handles bulk downloads efficiently
    raw = yf.download(
        tickers=symbols,
        start=start.isoformat(),
        end=end.isoformat(),
        interval=interval,
        group_by="ticker",
        auto_adjust=True,      # Adjust for splits/dividends
        progress=False,
        threads=True,
    )

    result: dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        try:
            # Handle single vs multi-ticker yfinance output format
            if len(symbols) == 1:
                df = raw.copy()
            else:
                df = raw[symbol].copy() if symbol in raw.columns.get_level_values(0) else pd.DataFrame()

            if df.empty:
                logger.warning(f"No price data returned for {symbol}")
                continue

            df = df.dropna(subset=["Close"])
            df.index = pd.to_datetime(df.index)
            result[symbol] = df

            if persist:
                await _upsert_bars(symbol, df, interval)

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")

    logger.info(f"Fetched price data for {len(result)}/{len(symbols)} symbols")
    return result


async def _upsert_bars(symbol: str, df: pd.DataFrame, interval: str) -> int:
    """Upsert OHLCV bars into MongoDB. Returns count of inserted documents."""
    bars_to_insert = []
    col_map = {
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    }

    for ts, row in df.iterrows():
        bar = MarketBar(
            symbol=symbol,
            timestamp=pd.Timestamp(ts).to_pydatetime(),
            open=float(row.get("Open", 0)),
            high=float(row.get("High", 0)),
            low=float(row.get("Low", 0)),
            close=float(row.get("Close", 0)),
            volume=float(row.get("Volume", 0)),
            source="yahoo",
            interval=interval,
        )
        bars_to_insert.append(bar)

    if not bars_to_insert:
        return 0

    # Bulk upsert using replace_one with upsert=True (Beanie doesn't expose bulk_write directly)
    # We use pymongo underneath for efficiency
    from db.client import get_client
    from config.settings import settings
    from config.constants import COLLECTIONS

    client = get_client()
    collection = client[settings.mongodb_db_name][COLLECTIONS["market_data"]]

    ops = []
    from pymongo import UpdateOne
    for bar in bars_to_insert:
        ops.append(
            UpdateOne(
                {"symbol": bar.symbol, "timestamp": bar.timestamp},
                {"$set": bar.model_dump(exclude={"id"})},
                upsert=True,
            )
        )

    if ops:
        result = await collection.bulk_write(ops, ordered=False)
        logger.debug(
            f"Upserted {result.upserted_count + result.modified_count} bars for {symbol}"
        )
        return result.upserted_count

    return 0


async def get_price_history(
    symbol: str,
    lookback_days: int = 252,
    end_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Retrieve price history from MongoDB for a single symbol.
    Returns a DataFrame indexed by timestamp with OHLCV columns.
    """
    from db.client import get_client
    from config.settings import settings
    from config.constants import COLLECTIONS

    end_date = end_date or datetime.utcnow()
    start_date = end_date - timedelta(days=lookback_days)

    client = get_client()
    collection = client[settings.mongodb_db_name][COLLECTIONS["market_data"]]

    cursor = collection.find(
        {"symbol": symbol, "timestamp": {"$gte": start_date, "$lte": end_date}},
        sort=[("timestamp", 1)],
    )

    records = await cursor.to_list(length=10000)
    if not records:
        logger.warning(f"No price history in DB for {symbol}. Consider running ingestion.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = df.set_index("timestamp")[["open", "high", "low", "close", "volume"]]
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    return df
