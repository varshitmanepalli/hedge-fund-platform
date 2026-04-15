"""
Macro indicator ingestion from FRED (Federal Reserve Economic Data).

Fetched series:
  - T10Y2Y   : 10Y-2Y Treasury spread (yield curve)
  - VIXCLS   : CBOE VIX
  - GDP       : Real GDP (quarterly, interpolated to monthly)
  - CPIAUCSL  : CPI All Urban Consumers (YoY computed here)
  - UNRATE    : Unemployment rate
  - FEDFUNDS  : Fed Funds Effective Rate
  - BAMLH0A0HYM2 : HY OAS credit spread

Falls back to mock data if FRED API key is not configured.
"""

from datetime import date, datetime, timedelta
from typing import Optional

import httpx
from loguru import logger

from config.settings import settings

# FRED series IDs → our field names
FRED_SERIES = {
    "T10Y2Y":       "yield_curve_slope",
    "VIXCLS":       "vix",
    "GDP":          "gdp_growth_qoq",
    "CPIAUCSL":     "cpi_yoy",
    "UNRATE":       "unemployment_rate",
    "FEDFUNDS":     "fed_funds_rate",
    "BAMLH0A0HYM2": "credit_spread_hy",
}

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"


async def fetch_latest_macro_indicators() -> dict[str, Optional[float]]:
    """
    Fetch the most recent value for each macro indicator.
    Returns a dict of field_name → float value.
    Falls back to safe mock values if FRED key is absent.
    """
    if not settings.fred_api_key:
        logger.warning("FRED API key not set — using mock macro indicators")
        return _mock_indicators()

    indicators: dict[str, Optional[float]] = {}
    async with httpx.AsyncClient(timeout=30.0) as client:
        for series_id, field_name in FRED_SERIES.items():
            try:
                value = await _fetch_series_latest(client, series_id)
                # GDP: convert to QoQ growth rate (series is level; we want % change)
                if series_id == "GDP" and value is not None:
                    prev = await _fetch_series_at_offset(client, series_id, offset=1)
                    value = ((value / prev) - 1) * 100 if prev else None
                # CPI: compute YoY rate
                elif series_id == "CPIAUCSL" and value is not None:
                    prev_year = await _fetch_series_at_offset(client, series_id, offset=12)
                    value = ((value / prev_year) - 1) * 100 if prev_year else None
                # HY spread: FRED returns in percent, convert to bps
                elif series_id == "BAMLH0A0HYM2" and value is not None:
                    value = value * 100

                indicators[field_name] = value
                logger.debug(f"Fetched {series_id} ({field_name}): {value}")

            except Exception as e:
                logger.error(f"Failed to fetch FRED series {series_id}: {e}")
                indicators[field_name] = None

    return indicators


async def _fetch_series_latest(
    client: httpx.AsyncClient, series_id: str
) -> Optional[float]:
    """Fetch the most recent non-null observation for a FRED series."""
    resp = await client.get(
        FRED_BASE,
        params={
            "series_id": series_id,
            "api_key": settings.fred_api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 1,
            "observation_start": (date.today() - timedelta(days=90)).isoformat(),
        },
    )
    resp.raise_for_status()
    obs = resp.json().get("observations", [])
    if not obs:
        return None
    val = obs[0].get("value", ".")
    return float(val) if val != "." else None


async def _fetch_series_at_offset(
    client: httpx.AsyncClient, series_id: str, offset: int
) -> Optional[float]:
    """
    Fetch the value `offset` periods back.
    Used for computing period-over-period changes.
    offset=1 → 1 quarter back for GDP, offset=12 → 1 year back for CPI
    """
    resp = await client.get(
        FRED_BASE,
        params={
            "series_id": series_id,
            "api_key": settings.fred_api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": offset + 1,
        },
    )
    resp.raise_for_status()
    obs = resp.json().get("observations", [])
    if len(obs) <= offset:
        return None
    val = obs[offset].get("value", ".")
    return float(val) if val != "." else None


def _mock_indicators() -> dict[str, float]:
    """Safe mock indicators for development / testing without a FRED key."""
    return {
        "yield_curve_slope": 0.25,     # Mildly positive — not inverted
        "vix": 17.5,                   # Low vol environment
        "gdp_growth_qoq": 2.1,
        "cpi_yoy": 3.2,
        "unemployment_rate": 3.9,
        "fed_funds_rate": 5.25,
        "credit_spread_hy": 310.0,     # In bps
    }
