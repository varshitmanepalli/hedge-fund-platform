"""
POST /api/v1/run-strategy
Triggers a full pipeline run and returns trades, portfolio state, and reasoning.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional

from orchestrator.runner import StrategyRequest, StrategyResult, run_strategy

router = APIRouter(prefix="/api/v1", tags=["strategy"])


class RunStrategyRequest(BaseModel):
    portfolio_id: str = Field(..., example="port_001")
    symbols: list[str] = Field(..., min_length=1, example=["AAPL", "MSFT", "GOOGL"])
    capital: float = Field(1_000_000.0, gt=0, example=1_000_000.0)
    risk_tolerance: str = Field("moderate", pattern="^(conservative|moderate|aggressive)$")
    lookback_days: int = Field(252, ge=30, le=1000)
    news_lookback_hours: int = Field(48, ge=1, le=168)
    top_n: int = Field(5, ge=1, le=20)


@router.post("/run-strategy", response_model=StrategyResult)
async def run_strategy_endpoint(body: RunStrategyRequest) -> StrategyResult:
    """
    Run the full multi-agent pipeline:
    Data Ingestion → Macro → (Quant || Sentiment) → Signal Aggregation → Risk → Execution

    Returns trades, portfolio weights, risk metrics, and per-symbol reasoning chains.
    """
    try:
        request = StrategyRequest(
            portfolio_id=body.portfolio_id,
            symbols=body.symbols,
            capital=body.capital,
            risk_tolerance=body.risk_tolerance,
            lookback_days=body.lookback_days,
            news_lookback_hours=body.news_lookback_hours,
            top_n=body.top_n,
        )
        result = await run_strategy(request)
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")
