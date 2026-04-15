"""
POST /api/v1/backtest
Runs a historical walk-forward simulation.
"""

from datetime import date
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backtest.engine import BacktestRequest, BacktestResult, run_backtest

router = APIRouter(prefix="/api/v1", tags=["backtest"])


class BacktestBody(BaseModel):
    symbols: list[str] = Field(..., min_length=1)
    start_date: date
    end_date: date
    initial_capital: float = Field(1_000_000.0, gt=0)
    rebalance_frequency: str = Field("monthly", pattern="^(daily|weekly|monthly|quarterly)$")
    risk_tolerance: str = Field("moderate", pattern="^(conservative|moderate|aggressive)$")
    benchmark: str = "SPY"
    top_n: int = Field(5, ge=1, le=20)
    slippage_bps: float = Field(5.0, ge=0)
    commission_per_share: float = Field(0.005, ge=0)


@router.post("/backtest", response_model=BacktestResult)
async def run_backtest_endpoint(body: BacktestBody) -> BacktestResult:
    """
    Run a historical simulation of the multi-agent strategy.

    Uses walk-forward factor signals (no look-ahead bias).
    Compares performance against the specified benchmark (default: SPY).
    """
    if body.end_date <= body.start_date:
        raise HTTPException(status_code=422, detail="end_date must be after start_date")

    min_days = 252 + 30  # Warmup + at least 30 days of simulation
    delta_days = (body.end_date - body.start_date).days
    if delta_days < 60:
        raise HTTPException(status_code=422, detail="Backtest period must be at least 60 days")

    try:
        result = await run_backtest(
            BacktestRequest(
                symbols=body.symbols,
                start_date=body.start_date,
                end_date=body.end_date,
                initial_capital=body.initial_capital,
                rebalance_frequency=body.rebalance_frequency,
                risk_tolerance=body.risk_tolerance,
                benchmark=body.benchmark,
                top_n=body.top_n,
                slippage_bps=body.slippage_bps,
                commission_per_share=body.commission_per_share,
            )
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest failed: {str(e)}")
