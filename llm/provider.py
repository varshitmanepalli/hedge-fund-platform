"""
Abstract LLM provider interface.

Agents call get_llm_provider() to get the configured backend.
Backends: HuggingFace Inference API (default), local HF model, mock (testing).

Adding a new backend: subclass BaseLLMProvider and implement generate().
"""

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Optional

from loguru import logger

from config.settings import settings


class BaseLLMProvider(ABC):
    @abstractmethod
    async def generate(self, prompt: str, max_tokens: int = 256) -> str: ...


class HuggingFaceAPIProvider(BaseLLMProvider):
    """Calls HuggingFace Inference API (text-generation endpoint)."""

    async def generate(self, prompt: str, max_tokens: int = 256) -> str:
        import httpx

        model = settings.macro_llm_model
        endpoint = f"{settings.hf_inference_endpoint}/{model}"
        headers = {"Authorization": f"Bearer {settings.hf_api_token}"}

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": 0.3,
                "do_sample": True,
                "return_full_text": False,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(endpoint, headers=headers, json=payload)
                resp.raise_for_status()
                result = resp.json()

                if isinstance(result, list) and result:
                    return result[0].get("generated_text", "").strip()
                return str(result)
        except Exception as e:
            logger.warning(f"HF API generate failed: {e}")
            raise


class MockLLMProvider(BaseLLMProvider):
    """
    Deterministic mock provider for testing.
    Returns a template response so tests don't need an API key.
    """

    async def generate(self, prompt: str, max_tokens: int = 256) -> str:
        # Extract key information from prompt for a semi-realistic mock
        if "macro" in prompt.lower() or "regime" in prompt.lower():
            return (
                "Current macro conditions present a mixed but broadly constructive environment. "
                "The combination of moderating inflation and resilient employment data supports "
                "a continued risk-on posture with selective sector exposure."
            )
        elif "signal" in prompt.lower() or "trade" in prompt.lower():
            return (
                "The convergence of positive momentum, supportive macro regime, and bullish "
                "sentiment creates a compelling entry point. Risk-adjusted return expectations "
                "are favorable given current volatility levels."
            )
        return "Analysis complete. Market conditions support the proposed allocation."


@lru_cache(maxsize=1)
def get_llm_provider() -> BaseLLMProvider:
    """
    Returns the appropriate LLM provider based on settings.
    Cached singleton — one provider instance per process.
    """
    if not settings.hf_api_token:
        logger.warning("HF_API_TOKEN not set — using MockLLMProvider")
        return MockLLMProvider()

    logger.info(f"Using HuggingFace API provider ({settings.macro_llm_model})")
    return HuggingFaceAPIProvider()
