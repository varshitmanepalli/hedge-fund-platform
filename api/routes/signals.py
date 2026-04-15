"""
GET /api/v1/signals
Returns recent aggregated signals filtered by symbol/agent/time.
"""

from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Query
from db.models.signal import AggregatedSignalDoc, MacroSignalDoc, QuantSignalDoc, SentimentSignalDoc

router = APIRouter(prefix="/api/v1", tags=["signals"])


@router.get("/signals")
async def get_signals(
    symbols: Optional[list[str]] = Query(None),
    agent: Optional[str] = Query(None, pattern="^(macro|quant|sentiment|aggregated)?$"),
    limit: int = Query(50, ge=1, le=500),
    since: Optional[datetime] = Query(None),
) -> dict:
    """
    Retrieve recent signals. Filter by symbol, agent type, and lookback window.
    Returns aggregated signals by default; use `agent` param for raw agent signals.
    """
    since = since or datetime.utcnow() - timedelta(days=1)

    if agent == "macro":
        docs = await MacroSignalDoc.find(MacroSignalDoc.timestamp >= since).sort(-MacroSignalDoc.timestamp).limit(limit).to_list()
        items = [{"type": "macro", **d.model_dump()} for d in docs]

    elif agent == "quant":
        query = QuantSignalDoc.timestamp >= since
        docs = await QuantSignalDoc.find(query).sort(-QuantSignalDoc.timestamp).limit(limit).to_list()
        if symbols:
            docs = [d for d in docs if d.symbol in symbols]
        items = [{"type": "quant", **d.model_dump()} for d in docs]

    elif agent == "sentiment":
        query = SentimentSignalDoc.timestamp >= since
        docs = await SentimentSignalDoc.find(query).sort(-SentimentSignalDoc.timestamp).limit(limit).to_list()
        if symbols:
            docs = [d for d in docs if d.symbol in symbols]
        items = [{"type": "sentiment", **d.model_dump()} for d in docs]

    else:
        # Default: aggregated signals
        query = AggregatedSignalDoc.timestamp >= since
        docs = await AggregatedSignalDoc.find(query).sort(-AggregatedSignalDoc.timestamp).limit(limit).to_list()
        if symbols:
            docs = [d for d in docs if d.symbol in symbols]
        items = [{"type": "aggregated", **d.model_dump()} for d in docs]

    return {
        "signals": items,
        "total": len(items),
        "since": since.isoformat(),
    }
