"""
CLI runner for a single pipeline run.
Run: python scripts/run_pipeline.py

Override defaults via environment variables or modify the StrategyRequest below.
"""

import asyncio
import json
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from loguru import logger

from db.client import init_db
from orchestrator.runner import StrategyRequest, run_strategy


async def main():
    await init_db()

    request = StrategyRequest(
        portfolio_id="cli_portfolio",
        symbols=["AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMZN", "JPM", "V"],
        capital=1_000_000.0,
        risk_tolerance="moderate",
        lookback_days=252,
        news_lookback_hours=48,
        top_n=5,
    )

    logger.info(f"Running pipeline for {request.symbols}")
    result = await run_strategy(request)

    print(f"\n{'='*60}")
    print(f"Run ID: {result.run_id}")
    print(f"Status: {result.status}")
    print(f"Duration: {result.duration_ms:.0f}ms")
    print(f"Regime: {result.regime}")
    print(f"Trades: {len(result.trades)}")
    print(f"Portfolio value: ${result.portfolio_value:,.2f}" if result.portfolio_value else "")
    print(f"\nStep durations: {result.step_durations}")
    print(f"\nProposed weights: {json.dumps(result.proposed_weights, indent=2)}")

    if result.reasoning_chains:
        print(f"\nSample reasoning chain ({list(result.reasoning_chains.keys())[0]}):")
        for step in list(result.reasoning_chains.values())[0]:
            print(f"  • {step}")

    if result.errors:
        print(f"\nErrors: {result.errors}")


if __name__ == "__main__":
    asyncio.run(main())
