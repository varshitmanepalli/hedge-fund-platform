"""
FastAPI application entrypoint.

Startup sequence:
  1. Initialize MongoDB (Beanie ODM)
  2. Mount all route modules
  3. Add CORS + logging middleware

Run: uvicorn api.main:app --reload
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from config.settings import settings
from db.client import close_db, init_db
from api.routes import strategy, portfolio, signals, backtest


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
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
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
    logger.info(
        f"{request.method} {request.url.path} → {response.status_code} ({duration:.0f}ms)"
    )
    return response


# ── Mount Routes ──────────────────────────────────────────────────────────────
app.include_router(strategy.router)
app.include_router(portfolio.router)
app.include_router(signals.router)
app.include_router(backtest.router)


# ── Health Check ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["system"])
async def health_check() -> dict:
    return {
        "status": "ok",
        "env": settings.env,
        "version": "1.0.0",
    }


@app.get("/", tags=["system"])
async def root() -> dict:
    return {
        "message": "AI Hedge Fund Platform",
        "docs": "/docs",
        "health": "/health",
    }
