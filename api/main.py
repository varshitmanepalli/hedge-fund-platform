"""
FastAPI application entrypoint.

Startup sequence:
  1. Initialize MongoDB (Beanie ODM)
  2. Mount all route modules
  3. Add CORS + logging middleware
  4. Serve frontend static files
  5. WebSocket endpoint for live pipeline progress

Run: uvicorn api.main:app --reload
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pathlib import Path

from config.settings import settings
from db.client import close_db, init_db
from api.routes import strategy, portfolio, signals, backtest
from api.routes.ws import router as ws_router


FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Handle startup and shutdown events."""
    logger.info(f"Starting Hedge Fund Platform API ({settings.env})")
    await init_db()
    logger.info("Database initialized.")
    yield
    await close_db()
    logger.info("Shutdown complete.")


app = FastAPI(
    title="AI Hedge Fund Platform",
    description="Multi-agent AI-powered hedge fund simulation system",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Global Exception Handler ──────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(f"Unhandled exception on {request.method} {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__},
    )


# ── Request Logging Middleware ─────────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    import time
    start = time.monotonic()
    response = await call_next(request)
    duration = (time.monotonic() - start) * 1000
    # Don't log static asset requests
    path = request.url.path
    if not path.startswith("/static") and not path.endswith((".js", ".css", ".ico", ".png")):
        logger.info(
            f"{request.method} {path} → {response.status_code} ({duration:.0f}ms)"
        )
    return response


# ── Mount API Routes ──────────────────────────────────────────────────────────
app.include_router(strategy.router)
app.include_router(portfolio.router)
app.include_router(signals.router)
app.include_router(backtest.router)
app.include_router(ws_router)


# ── Health Check ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["system"])
async def health_check() -> dict:
    return {
        "status": "ok",
        "env": settings.env,
        "version": "2.0.0",
    }


# ── Frontend Serving ──────────────────────────────────────────────────────────
# Serve frontend at root — must be AFTER API routes so /api/v1/* takes priority

@app.get("/", tags=["frontend"])
async def serve_frontend():
    """Serve the main frontend dashboard."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return JSONResponse(
        content={"message": "AI Hedge Fund Platform", "docs": "/docs", "health": "/health"},
    )


# Mount static assets if the frontend directory has any static files
if (FRONTEND_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="assets")
