"""
POST /api/v1/run-strategy
Triggers a full pipeline run and returns trades, portfolio state, and reasoning.

Supports two modes:
  - Synchronous (default): Runs pipeline and returns result directly.
  - Async (stream=true): Returns run_id immediately, streams progress via WebSocket.
"""

import asyncio
import uuid
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional

from orchestrator.runner import StrategyRequest, StrategyResult, run_strategy
from api.routes.ws import progress_manager

router = APIRouter(prefix="/api/v1", tags=["strategy"])


class RunStrategyRequest(BaseModel):
    portfolio_id: str = Field(..., example="port_001")
    symbols: list[str] = Field(..., min_length=1, example=["AAPL", "MSFT", "GOOGL"])
    capital: float = Field(1_000_000.0, gt=0, example=1_000_000.0)
    risk_tolerance: str = Field("moderate", pattern="^(conservative|moderate|aggressive)$")
    lookback_days: int = Field(252, ge=30, le=1000)
    news_lookback_hours: int = Field(48, ge=1, le=168)
    top_n: int = Field(5, ge=1, le=20)
    stream: bool = Field(False, description="If true, return run_id immediately and stream progress via WebSocket")


# In-memory store for async pipeline results
_async_results: dict[str, Optional[StrategyResult]] = {}


async def _run_pipeline_async(run_id: str, request: StrategyRequest):
    """Background task that runs the pipeline and emits WebSocket events."""
    try:
        result = await run_strategy(request, progress_callback=progress_manager)
        _async_results[run_id] = result
        await progress_manager.emit_pipeline_complete(run_id, result.duration_ms, {
            "status": result.status,
            "trades_count": len(result.trades),
            "portfolio_value": result.portfolio_value,
        })
    except Exception as e:
        await progress_manager.emit_pipeline_error(run_id, str(e))
        _async_results[run_id] = None


@router.post("/run-strategy", response_model=StrategyResult)
async def run_strategy_endpoint(
    body: RunStrategyRequest,
    background_tasks: BackgroundTasks,
) -> StrategyResult | dict:
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

        if body.stream:
            # Async mode: return run_id, run pipeline in background
            run_id = f"run_{uuid.uuid4().hex[:10]}"
            _async_results[run_id] = None
            background_tasks.add_task(_run_pipeline_async, run_id, request)
            return {"run_id": run_id, "status": "started", "message": "Connect to /ws/pipeline/{run_id} for live progress"}

        # Synchronous mode
        result = await run_strategy(request)
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")


@router.get("/run-status/{run_id}")
async def get_run_status(run_id: str) -> dict:
    """Check status of an async pipeline run."""
    if run_id not in _async_results:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")

    result = _async_results[run_id]
    if result is None:
        return {"run_id": run_id, "status": "running"}

    return {
        "run_id": run_id,
        "status": result.status,
        "result": result.model_dump(),
    }
