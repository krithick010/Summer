"""
core/llm_engine.py
------------------
Async client for Ollama's native tool-calling `/api/chat` endpoint.
No hand-written prompt tokens or JSON extractors.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

import config

log = logging.getLogger(__name__)


def _ns_to_ms(value: Any) -> float | None:
    """Convert an Ollama nanosecond duration field to milliseconds."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return round(float(value) / 1_000_000.0, 1)
    return None


def log_ollama_timing(
    logger: logging.Logger,
    kind: str,
    data: dict[str, Any],
    model_fallback: str,
    elapsed_ms: float,
) -> None:
    """Emit a single INFO timing line from an Ollama /api/chat JSON response."""
    model = data.get("model") or model_fallback
    load_ms = _ns_to_ms(data.get("load_duration"))
    prompt_eval_count = data.get("prompt_eval_count")
    eval_count = data.get("eval_count")
    total_ms = _ns_to_ms(data.get("total_duration"))
    if total_ms is None:
        total_ms = round(elapsed_ms, 1)

    logger.info(
        "Ollama %s | model=%s load_duration_ms=%s prompt_eval_count=%s eval_count=%s total_ms=%s",
        kind,
        model,
        load_ms if load_ms is not None else "na",
        prompt_eval_count if prompt_eval_count is not None else "na",
        eval_count if eval_count is not None else "na",
        total_ms,
    )


class LLMEngine:
    def __init__(self) -> None:
        self._url = f"{config.OLLAMA_BASE_URL}/api/chat"
        log.info("LLMEngine ready model=%s", config.OLLAMA_MODEL)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": config.OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "keep_alive": config.OLLAMA_KEEP_ALIVE,
            "options": {
                "temperature": config.LLM_TEMPERATURE if temperature is None else temperature,
                "num_ctx": config.OLLAMA_NUM_CTX,
                "num_predict": 300,
            },
        }
        if tools:
            payload["tools"] = tools

        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=120) as c:
            r = await c.post(self._url, json=payload)
            r.raise_for_status()
        data = r.json()
        log_ollama_timing(
            log,
            "chat",
            data,
            config.OLLAMA_MODEL,
            (time.perf_counter() - t0) * 1000.0,
        )
        return data["message"]  # {"role", "content", "tool_calls"?}
