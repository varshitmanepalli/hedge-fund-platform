"""
Orchestrator Memory — stores past pipeline decisions and agent outputs
to enable context-aware reasoning in subsequent runs.

MongoDB-backed, indexed by portfolio_id + run_id.
"""

from datetime import datetime, timedelta
from typing import Optional

from loguru import logger


class PipelineMemory:
    """
    Lightweight key-value store backed by MongoDB for pipeline state.
    Retrieves:
      - Past regime assessments (for trend detection)
      - Past trades (for position tracking)
      - Past risk metrics (for drawdown monitoring)
    """

    def __init__(self, portfolio_id: str):
        self.portfolio_id = portfolio_id

    async def get_recent_regimes(self, n: int = 5) -> list[dict]:
        """Return the last N macro regime assessments."""
        from db.models.signal import MacroSignalDoc

        docs = (
            await MacroSignalDoc.find()
            .sort(-MacroSignalDoc.timestamp)
            .limit(n)
            .to_list()
        )
        return [
            {"timestamp": str(d.timestamp), "regime": d.regime.value, "confidence": d.confidence}
            for d in docs
        ]

    async def get_portfolio_equity_curve(
        self,
        lookback_days: int = 252,
    ) -> list[float]:
        """
        Reconstruct portfolio equity curve from trade history.
        Returns list of daily portfolio values.
        """
        from db.client import get_client
        from config.settings import settings
        from config.constants import COLLECTIONS

        client = get_client()
        collection = client[settings.mongodb_db_name][COLLECTIONS["portfolios"]]

        doc = await collection.find_one({"portfolio_id": self.portfolio_id})
        if doc and "equity_curve" in doc:
            return [entry["value"] for entry in doc["equity_curve"][-lookback_days:]]
        return []

    async def get_portfolio_returns(self, lookback_days: int = 252) -> list[float]:
        """Compute daily returns from equity curve."""
        equity = await self.get_portfolio_equity_curve(lookback_days)
        if len(equity) < 2:
            return []
        returns = []
        for i in range(1, len(equity)):
            if equity[i - 1] > 0:
                returns.append((equity[i] - equity[i - 1]) / equity[i - 1])
        return returns

    async def get_current_positions(self) -> dict[str, float]:
        """Get current holdings from portfolio document."""
        from db.client import get_client
        from config.settings import settings
        from config.constants import COLLECTIONS

        client = get_client()
        collection = client[settings.mongodb_db_name][COLLECTIONS["portfolios"]]

        doc = await collection.find_one({"portfolio_id": self.portfolio_id})
        if not doc:
            return {}

        return {
            pos["symbol"]: pos["quantity"]
            for pos in doc.get("positions", [])
        }

    async def update_portfolio_value(self, value: float) -> None:
        """Append current portfolio value to equity curve."""
        from db.client import get_client
        from config.settings import settings
        from config.constants import COLLECTIONS

        client = get_client()
        collection = client[settings.mongodb_db_name][COLLECTIONS["portfolios"]]

        await collection.update_one(
            {"portfolio_id": self.portfolio_id},
            {
                "$push": {
                    "equity_curve": {
                        "date": datetime.utcnow().isoformat(),
                        "value": value,
                    }
                }
            },
            upsert=True,
        )

    async def save_run_summary(self, run_id: str, summary: dict) -> None:
        """Store a lightweight summary of the pipeline run for future context."""
        from db.client import get_client
        from config.settings import settings

        client = get_client()
        collection = client[settings.mongodb_db_name]["pipeline_runs"]

        await collection.update_one(
            {"run_id": run_id},
            {"$set": {"portfolio_id": self.portfolio_id, "timestamp": datetime.utcnow(), **summary}},
            upsert=True,
        )
        logger.debug(f"Saved pipeline run summary for run_id={run_id}")
