"""
WebSocket endpoint for live pipeline progress streaming.

Clients connect to /ws/pipeline/{run_id} and receive JSON messages
as each pipeline step starts and completes.

Message format:
{
    "type": "step_start" | "step_complete" | "pipeline_complete" | "pipeline_error",
    "run_id": "run_abc123",
    "step": "ingest",
    "timestamp": "2026-04-15T19:32:00Z",
    "duration_ms": 320,       # only on step_complete
    "data": { ... }            # optional payload
}
"""

import asyncio
import json
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

router = APIRouter(tags=["websocket"])


class PipelineProgressManager:
    """
    In-memory pub/sub for pipeline progress events.
    Each run_id has a set of connected WebSocket clients.
    """

    def __init__(self):
        self._subscribers: dict[str, list[WebSocket]] = {}

    async def subscribe(self, run_id: str, ws: WebSocket):
        await ws.accept()
        if run_id not in self._subscribers:
            self._subscribers[run_id] = []
        self._subscribers[run_id].append(ws)
        logger.info(f"WebSocket subscribed to pipeline {run_id}")

    def unsubscribe(self, run_id: str, ws: WebSocket):
        if run_id in self._subscribers:
            self._subscribers[run_id] = [
                w for w in self._subscribers[run_id] if w != ws
            ]
            if not self._subscribers[run_id]:
                del self._subscribers[run_id]

    async def broadcast(self, run_id: str, message: dict):
        """Send a message to all subscribers of a given run_id."""
        if run_id not in self._subscribers:
            return

        dead = []
        for ws in self._subscribers[run_id]:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)

        for ws in dead:
            self.unsubscribe(run_id, ws)

    async def emit_step_start(self, run_id: str, step: str):
        await self.broadcast(run_id, {
            "type": "step_start",
            "run_id": run_id,
            "step": step,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    async def emit_step_complete(self, run_id: str, step: str, duration_ms: float, data: Optional[dict] = None):
        await self.broadcast(run_id, {
            "type": "step_complete",
            "run_id": run_id,
            "step": step,
            "duration_ms": round(duration_ms, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data or {},
        })

    async def emit_pipeline_complete(self, run_id: str, duration_ms: float, summary: Optional[dict] = None):
        await self.broadcast(run_id, {
            "type": "pipeline_complete",
            "run_id": run_id,
            "duration_ms": round(duration_ms, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": summary or {},
        })

    async def emit_pipeline_error(self, run_id: str, error: str):
        await self.broadcast(run_id, {
            "type": "pipeline_error",
            "run_id": run_id,
            "error": error,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })


# Singleton instance — importable by the strategy runner
progress_manager = PipelineProgressManager()


@router.websocket("/ws/pipeline/{run_id}")
async def pipeline_progress_ws(websocket: WebSocket, run_id: str):
    """
    WebSocket endpoint for real-time pipeline progress.

    Connect to /ws/pipeline/{run_id} after triggering a strategy run.
    The server pushes step_start, step_complete, and pipeline_complete events.
    """
    await progress_manager.subscribe(run_id, websocket)
    try:
        # Keep connection alive — wait for client messages (ping/close)
        while True:
            data = await websocket.receive_text()
            # Client can send "ping" to keep alive
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected from pipeline {run_id}")
    finally:
        progress_manager.unsubscribe(run_id, websocket)
