"""
GET /api/v1/portfolio/{portfolio_id}
Returns current portfolio state, positions, and PnL.
"""

from fastapi import APIRouter, HTTPException
from db.models.portfolio import Portfolio

router = APIRouter(prefix="/api/v1", tags=["portfolio"])


@router.get("/portfolio/{portfolio_id}")
async def get_portfolio(portfolio_id: str) -> dict:
    """Retrieve current portfolio state by ID."""
    doc = await Portfolio.find_one(Portfolio.portfolio_id == portfolio_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Portfolio {portfolio_id!r} not found")

    return doc.model_dump()


@router.get("/portfolios")
async def list_portfolios(limit: int = 10) -> list[dict]:
    """List all portfolios (most recently updated first)."""
    docs = await Portfolio.find().sort(-Portfolio.updated_at).limit(limit).to_list()
    return [d.model_dump() for d in docs]
