# ─────────────────────────────────────────────────────────────────────────────
# AI Hedge Fund Platform — Multi-stage Docker build
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim AS base

# Prevent Python from writing .pyc and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# ── Python deps (cached layer) ───────────────────────────────────────────────
COPY pyproject.toml ./
RUN pip install --no-cache-dir \
    fastapi[standard] \
    uvicorn[standard] \
    motor \
    beanie \
    pydantic \
    pydantic-settings \
    loguru \
    httpx \
    numpy \
    pandas \
    scikit-learn \
    yfinance \
    transformers \
    torch --index-url https://download.pytorch.org/whl/cpu \
    websockets

# ── Application code ─────────────────────────────────────────────────────────
COPY . .

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Expose port ───────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Start server ──────────────────────────────────────────────────────────────
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
