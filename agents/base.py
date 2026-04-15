"""
Abstract base class for all agents.

Every agent gets:
  - Structured logging (loguru + MongoDB AgentLog)
  - Execution timing
  - Standardized error handling with optional retry
  - run_id tracking for pipeline correlation
"""

import hashlib
import json
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Generic, Optional, TypeVar

from loguru import logger
from pydantic import BaseModel

from db.models.agent_log import AgentLog

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


class BaseAgent(ABC, Generic[InputT, OutputT]):
    """
    Abstract base agent.

    Subclasses must implement:
      - name: str class attribute
      - version: str class attribute
      - _run(input_data: InputT) -> OutputT
    """

    name: str = "base_agent"
    version: str = "1.0.0"

    def __init__(self, run_id: Optional[str] = None):
        self.run_id = run_id or f"run_{uuid.uuid4().hex[:10]}"
        self._logger = logger.bind(agent=self.name, run_id=self.run_id)

    async def run(self, input_data: InputT) -> OutputT:
        """
        Public entry point. Wraps _run() with timing, logging, and error handling.
        """
        start_ms = time.monotonic() * 1000
        input_hash = self._hash_input(input_data)
        self._logger.info(f"[{self.name}] Starting (input_hash={input_hash})")

        error_msg: Optional[str] = None
        output: Optional[OutputT] = None
        tokens_used = 0

        try:
            output = await self._run(input_data)
            self._logger.info(
                f"[{self.name}] Completed successfully in "
                f"{time.monotonic() * 1000 - start_ms:.0f}ms"
            )
        except Exception as e:
            error_msg = str(e)
            self._logger.exception(f"[{self.name}] Failed: {e}")
            raise
        finally:
            duration_ms = time.monotonic() * 1000 - start_ms
            await self._write_log(
                input_hash=input_hash,
                output=output,
                duration_ms=duration_ms,
                error=error_msg,
                tokens_used=tokens_used,
            )

        return output

    @abstractmethod
    async def _run(self, input_data: InputT) -> OutputT:
        """Core agent logic. Must be implemented by subclasses."""
        ...

    async def _write_log(
        self,
        input_hash: str,
        output: Optional[Any],
        duration_ms: float,
        error: Optional[str],
        tokens_used: int = 0,
    ) -> None:
        """Persist structured agent execution log to MongoDB."""
        try:
            summary = ""
            if output is not None:
                try:
                    summary = output.model_dump_json()[:500]
                except Exception:
                    summary = str(output)[:500]

            log = AgentLog(
                agent_name=self.name,
                run_id=self.run_id,
                timestamp=datetime.utcnow(),
                duration_ms=duration_ms,
                input_hash=input_hash,
                output_summary=summary,
                llm_tokens_used=tokens_used,
                error=error,
            )
            await log.insert()
        except Exception as e:
            # Log failure must never crash the agent
            self._logger.warning(f"Failed to write agent log: {e}")

    @staticmethod
    def _hash_input(input_data: BaseModel) -> str:
        """Deterministic hash of agent input for deduplication."""
        raw = input_data.model_dump_json(sort_keys=True)
        return hashlib.md5(raw.encode()).hexdigest()[:12]
