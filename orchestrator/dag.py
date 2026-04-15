"""
Custom DAG Orchestrator for the hedge fund agent pipeline.

Supports:
  - Sequential and parallel node execution
  - Per-node retry with exponential backoff (via tenacity)
  - Pipeline state propagation between nodes
  - Step-level timing and logging
  - Pluggable node definitions

The pipeline graph is defined as an adjacency dict.
Nodes with no dependencies in a given step run in parallel (asyncio.gather).
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Coroutine, Optional

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass
class PipelineState:
    """
    Mutable state object passed through the entire pipeline.
    Agents append their outputs here; downstream agents read from it.
    """
    run_id: str = field(default_factory=lambda: f"run_{uuid.uuid4().hex[:10]}")
    portfolio_id: str = ""
    symbols: list[str] = field(default_factory=list)
    capital: float = 1_000_000.0
    risk_tolerance: str = "moderate"

    # Agent outputs (populated in sequence)
    ingestion_result: Optional[Any] = None
    macro_output: Optional[Any] = None
    quant_output: Optional[Any] = None
    sentiment_output: Optional[Any] = None
    aggregation_result: Optional[Any] = None
    risk_output: Optional[Any] = None
    execution_output: Optional[Any] = None

    # Metadata
    start_time: datetime = field(default_factory=datetime.utcnow)
    step_durations: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    completed: bool = False


NodeFn = Callable[[PipelineState], Coroutine[Any, Any, PipelineState]]


@dataclass
class DAGNode:
    name: str
    fn: NodeFn
    dependencies: list[str] = field(default_factory=list)
    max_retries: int = 2
    timeout_seconds: Optional[float] = None


class PipelineDAG:
    """
    Lightweight async DAG executor.
    Nodes are executed in topological order; independent nodes run in parallel.
    """

    def __init__(self, nodes: list[DAGNode]):
        self.nodes = {n.name: n for n in nodes}
        self._validate()

    def _validate(self) -> None:
        """Ensure all dependency references exist."""
        for node in self.nodes.values():
            for dep in node.dependencies:
                if dep not in self.nodes:
                    raise ValueError(f"Node '{node.name}' has unknown dependency '{dep}'")

    def _topological_levels(self) -> list[list[str]]:
        """
        Return nodes grouped by execution level.
        Level 0 = no deps. Level N = all deps in levels 0..N-1.
        Nodes at the same level can run in parallel.
        """
        in_degree = {n: len(self.nodes[n].dependencies) for n in self.nodes}
        completed: set[str] = set()
        levels = []

        while len(completed) < len(self.nodes):
            ready = [
                n for n in self.nodes
                if n not in completed and all(dep in completed for dep in self.nodes[n].dependencies)
            ]
            if not ready:
                raise RuntimeError("Cycle detected in pipeline DAG")
            levels.append(ready)
            completed.update(ready)

        return levels

    async def run(self, state: PipelineState) -> PipelineState:
        """Execute the full DAG, returning the final state."""
        levels = self._topological_levels()
        logger.info(f"Pipeline {state.run_id} starting. Nodes: {list(self.nodes.keys())}")

        for level_idx, node_names in enumerate(levels):
            logger.info(f"Executing level {level_idx}: {node_names}")

            # Run all nodes in this level concurrently
            tasks = [self._run_node(self.nodes[name], state) for name in node_names]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for name, result in zip(node_names, results):
                if isinstance(result, Exception):
                    logger.error(f"Node '{name}' failed: {result}")
                    state.errors.append(f"{name}: {str(result)}")
                    # Non-fatal: continue pipeline; agent output will be None

        state.completed = True
        total_duration = (datetime.utcnow() - state.start_time).total_seconds()
        logger.info(f"Pipeline {state.run_id} complete in {total_duration:.2f}s. Errors: {state.errors}")
        return state

    async def _run_node(self, node: DAGNode, state: PipelineState) -> PipelineState:
        """Execute a single node with retry and timeout."""
        import time

        start = time.monotonic()
        logger.info(f"[{state.run_id}] Starting node: {node.name}")

        attempt = 0
        last_error: Optional[Exception] = None

        for attempt in range(node.max_retries + 1):
            try:
                if node.timeout_seconds:
                    await asyncio.wait_for(node.fn(state), timeout=node.timeout_seconds)
                else:
                    await node.fn(state)

                duration = time.monotonic() - start
                state.step_durations[node.name] = round(duration * 1000, 2)  # ms
                logger.info(f"[{state.run_id}] Node '{node.name}' succeeded ({duration:.2f}s)")
                return state

            except Exception as e:
                last_error = e
                if attempt < node.max_retries:
                    wait = 2 ** attempt
                    logger.warning(
                        f"[{state.run_id}] Node '{node.name}' failed (attempt {attempt+1}), "
                        f"retrying in {wait}s: {e}"
                    )
                    await asyncio.sleep(wait)

        duration = time.monotonic() - start
        state.step_durations[node.name] = round(duration * 1000, 2)
        raise last_error or RuntimeError(f"Node {node.name} failed")
