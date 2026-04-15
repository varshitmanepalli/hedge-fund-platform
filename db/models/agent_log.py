"""
Beanie ODM model for structured agent execution logs.
"""

import uuid
from datetime import datetime
from typing import Optional

from beanie import Document
from pydantic import Field


class AgentLog(Document):
    log_id: str = Field(default_factory=lambda: f"log_{uuid.uuid4().hex[:10]}")
    agent_name: str
    run_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    duration_ms: float = 0.0
    input_hash: str = ""
    output_summary: str = ""
    llm_tokens_used: int = 0
    error: Optional[str] = None
    metadata: dict = Field(default_factory=dict)

    class Settings:
        name = "agent_logs"
        indexes = [
            [("run_id", 1)],
            [("agent_name", 1), ("timestamp", -1)],
        ]
