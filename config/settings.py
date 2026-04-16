"""
Global application settings loaded from environment variables.
All agents and services import from here — never import os.environ directly.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Environment ──────────────────────────────────────────────────────────
    env: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"

    # ── API Server ───────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ── MongoDB ──────────────────────────────────────────────────────────────
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db_name: str = "hedgefund"

    # ── External Data ────────────────────────────────────────────────────────
    news_api_key: str = ""
    fred_api_key: str = ""
    alpha_vantage_key: str = ""

    # ── LLM / HuggingFace ────────────────────────────────────────────────────
    hf_api_token: str = ""
    hf_inference_endpoint: str = "https://router.huggingface.co/hf-inference/models"
    sentiment_model: str = "ProsusAI/finbert"
    macro_llm_model: str = "mistralai/Mistral-7B-Instruct-v0.3"
    use_local_llm: bool = False

    # ── Risk Defaults ────────────────────────────────────────────────────────
    default_max_position_weight: float = 0.30
    default_max_var_95_pct: float = 0.02
    default_min_sharpe: float = 0.5
    default_max_drawdown: float = 0.15
    default_max_leverage: float = 1.0

    # ── Execution Defaults ───────────────────────────────────────────────────
    default_slippage_bps: float = 5.0
    default_commission_per_share: float = 0.005

    @property
    def is_production(self) -> bool:
        return self.env == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton settings — cached after first load."""
    return Settings()


# Module-level convenience alias
settings = get_settings()
