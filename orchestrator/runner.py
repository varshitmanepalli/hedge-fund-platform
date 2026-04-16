"""
Pipeline Runner — wires all agents into the DAG and runs a full strategy cycle.

Pipeline topology (node: depends_on):
  ingest           : []
  macro + quant    : [ingest]          ← parallel
  sentiment        : [ingest]          ← parallel with macro/quant
  aggregate        : [macro, quant, sentiment]
  risk             : [aggregate]
  execute          : [risk]
"""

import uuid
from datetime import datetime
from typing import Optional

from loguru import logger
from pydantic import BaseModel

from orchestrator.dag import DAGNode, PipelineDAG, PipelineState
from orchestrator.memory import PipelineMemory
from orchestrator.signal_aggregator import aggregate_signals
from agents.macro.agent import MacroAgent, MacroAgentInput
from agents.quant.agent import QuantAgent, QuantAgentInput
from agents.sentiment.agent import SentimentAgent, SentimentAgentInput, NewsArticle
from agents.risk.agent import RiskManagerAgent, RiskAgentInput
from agents.execution.agent import ExecutionAgent, ExecutionAgentInput
from data.pipeline import run_ingestion
from explainability.chain_builder import build_reasoning_chains


class StrategyRequest(BaseModel):
    portfolio_id: str
    symbols: list[str]
    capital: float = 1_000_000.0
    risk_tolerance: str = "moderate"
    lookback_days: int = 252
    news_lookback_hours: int = 48
    top_n: int = 5
    persist: bool = True


class StrategyResult(BaseModel):
    run_id: str
    status: str
    duration_ms: float
    regime: Optional[str]
    trades: list[dict]
    portfolio_value: Optional[float]
    risk_metrics: Optional[dict]
    proposed_weights: Optional[dict]
    reasoning_chains: dict[str, list[str]]
    step_durations: dict[str, float]
    errors: list[str]


async def run_strategy(request: StrategyRequest, progress_callback=None) -> StrategyResult:
    """
    Execute the full multi-agent pipeline for a strategy request.
    Returns a StrategyResult with all agent outputs and reasoning chains.
    """
    run_id = f"run_{uuid.uuid4().hex[:10]}"
    start = datetime.utcnow()

    # WebSocket progress emitter (optional)
    async def emit(event_type: str, step: str = "", duration_ms: float = 0, data: dict = None):
        if progress_callback is None:
            return
        if event_type == "step_start":
            await progress_callback.emit_step_start(run_id, step)
        elif event_type == "step_complete":
            await progress_callback.emit_step_complete(run_id, step, duration_ms, data)

    state = PipelineState(
        run_id=run_id,
        portfolio_id=request.portfolio_id,
        symbols=request.symbols,
        capital=request.capital,
        risk_tolerance=request.risk_tolerance,
    )

    memory = PipelineMemory(portfolio_id=request.portfolio_id)

    # ── Define pipeline nodes ──────────────────────────────────────────────────

    async def node_ingest(s: PipelineState) -> PipelineState:
        await emit("step_start", "ingest")
        _t = datetime.utcnow()
        s.ingestion_result = await run_ingestion(
            symbols=s.symbols,
            lookback_days=request.lookback_days,
            news_lookback_hours=request.news_lookback_hours,
            persist=request.persist,
        )
        await emit("step_complete", "ingest", (datetime.utcnow() - _t).total_seconds() * 1000)
        return s

    async def node_macro(s: PipelineState) -> PipelineState:
        await emit("step_start", "macro")
        _t = datetime.utcnow()
        agent = MacroAgent(run_id=s.run_id)
        s.macro_output = await agent.run(
            MacroAgentInput(
                symbols=s.symbols,
                macro_indicators=s.ingestion_result.macro_indicators,
                lookback_days=request.lookback_days,
            )
        )
        await emit("step_complete", "macro", (datetime.utcnow() - _t).total_seconds() * 1000)
        return s

    async def node_quant(s: PipelineState) -> PipelineState:
        await emit("step_start", "quant")
        _t = datetime.utcnow()
        agent = QuantAgent(run_id=s.run_id)
        price_data = {
            sym: df.reset_index().rename(
                columns={df.index.name or "index": "timestamp"}
            ).to_dict("records")
            for sym, df in s.ingestion_result.price_data.items()
        }
        s.quant_output = await agent.run(
            QuantAgentInput(
                symbols=s.symbols,
                price_data=price_data,
            )
        )
        await emit("step_complete", "quant", (datetime.utcnow() - _t).total_seconds() * 1000)
        return s

    async def node_sentiment(s: PipelineState) -> PipelineState:
        await emit("step_start", "sentiment")
        _t = datetime.utcnow()
        agent = SentimentAgent(run_id=s.run_id)
        news_articles = {
            sym: [
                NewsArticle(
                    symbol=sym,
                    title=item.title,
                    body=item.body,
                    source=item.source,
                    url=item.url,
                    published_at=item.published_at,
                )
                for item in items
            ]
            for sym, items in s.ingestion_result.news_data.items()
        }
        s.sentiment_output = await agent.run(
            SentimentAgentInput(
                symbols=s.symbols,
                news_articles=news_articles,
                lookback_hours=request.news_lookback_hours,
            )
        )
        await emit("step_complete", "sentiment", (datetime.utcnow() - _t).total_seconds() * 1000)
        return s

    async def node_aggregate(s: PipelineState) -> PipelineState:
        await emit("step_start", "aggregate")
        _t = datetime.utcnow()
        s.aggregation_result = await aggregate_signals(
            macro_output=s.macro_output,
            quant_output=s.quant_output,
            sentiment_output=s.sentiment_output,
            run_id=s.run_id,
            top_n=request.top_n,
        )
        await emit("step_complete", "aggregate", (datetime.utcnow() - _t).total_seconds() * 1000)
        return s

    async def node_risk(s: PipelineState) -> PipelineState:
        await emit("step_start", "risk")
        _t = datetime.utcnow()
        portfolio_returns = await memory.get_portfolio_returns()
        equity_curve = await memory.get_portfolio_equity_curve()

        agent = RiskManagerAgent(run_id=s.run_id)
        s.risk_output = await agent.run(
            RiskAgentInput(
                portfolio_id=s.portfolio_id,
                proposed_weights=s.aggregation_result.proposed_weights,
                portfolio_returns=portfolio_returns,
                equity_curve=equity_curve,
                capital=s.capital,
                risk_tolerance=s.risk_tolerance,
            )
        )
        await emit("step_complete", "risk", (datetime.utcnow() - _t).total_seconds() * 1000)
        return s

    async def node_execute(s: PipelineState) -> PipelineState:
        await emit("step_start", "execute")
        _t = datetime.utcnow()
        # Only execute if risk manager approved
        if not s.risk_output.approved:
            logger.warning(f"Risk manager rejected allocation. Skipping execution.")
            await emit("step_complete", "execute", (datetime.utcnow() - _t).total_seconds() * 1000, {"skipped": True})
            return s

        current_positions = await memory.get_current_positions()
        current_prices = {
            sym: float(df["Close"].iloc[-1])
            for sym, df in s.ingestion_result.price_data.items()
            if not df.empty and "Close" in df.columns
        }

        # Build reasoning chains for each symbol
        reasoning_chains = build_reasoning_chains(
            macro_output=s.macro_output,
            quant_output=s.quant_output,
            sentiment_output=s.sentiment_output,
            risk_output=s.risk_output,
            aggregation_result=s.aggregation_result,
        )

        agent = ExecutionAgent(run_id=s.run_id)
        s.execution_output = await agent.run(
            ExecutionAgentInput(
                portfolio_id=s.portfolio_id,
                current_positions=current_positions,
                target_weights=s.risk_output.approved_weights,
                capital=s.capital,
                current_prices=current_prices,
                slippage_bps=5.0,
                commission_per_share=0.005,
                reasoning_chains=reasoning_chains,
            )
        )

        # Update memory with new portfolio value
        await memory.update_portfolio_value(s.execution_output.portfolio_value)
        await emit("step_complete", "execute", (datetime.utcnow() - _t).total_seconds() * 1000)
        return s

    # ── Build and run DAG ──────────────────────────────────────────────────────

    dag = PipelineDAG([
        DAGNode("ingest",    node_ingest,    dependencies=[],                              max_retries=2),
        DAGNode("macro",     node_macro,     dependencies=["ingest"],                      max_retries=1),
        DAGNode("quant",     node_quant,     dependencies=["ingest"],                      max_retries=1),
        DAGNode("sentiment", node_sentiment, dependencies=["ingest"],                      max_retries=1),
        DAGNode("aggregate", node_aggregate, dependencies=["macro", "quant", "sentiment"], max_retries=0),
        DAGNode("risk",      node_risk,      dependencies=["aggregate"],                   max_retries=0),
        DAGNode("execute",   node_execute,   dependencies=["risk"],                        max_retries=1),
    ])

    final_state = await dag.run(state)

    # ── Build final result ─────────────────────────────────────────────────────
    duration_ms = (datetime.utcnow() - start).total_seconds() * 1000

    # Save run summary to memory
    await memory.save_run_summary(run_id, {
        "regime": final_state.macro_output.regime.value if final_state.macro_output else None,
        "trades_count": len(final_state.execution_output.trades) if final_state.execution_output else 0,
    })

    reasoning_chains = {}
    if final_state.execution_output and final_state.macro_output and final_state.aggregation_result:
        reasoning_chains = build_reasoning_chains(
            macro_output=final_state.macro_output,
            quant_output=final_state.quant_output,
            sentiment_output=final_state.sentiment_output,
            risk_output=final_state.risk_output,
            aggregation_result=final_state.aggregation_result,
        )

    return StrategyResult(
        run_id=run_id,
        status="success" if not final_state.errors else "partial",
        duration_ms=round(duration_ms, 2),
        regime=final_state.macro_output.regime.value if final_state.macro_output else None,
        trades=final_state.execution_output.trades if final_state.execution_output else [],
        portfolio_value=final_state.execution_output.portfolio_value if final_state.execution_output else None,
        risk_metrics=final_state.risk_output.model_dump() if final_state.risk_output else None,
        proposed_weights=final_state.risk_output.approved_weights if final_state.risk_output else None,
        reasoning_chains=reasoning_chains,
        step_durations=final_state.step_durations,
        errors=final_state.errors,
    )
