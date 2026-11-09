"""
core/llm_engine.py
------------------
Async client for Ollama's native tool-calling `/api/chat` endpoint.
No hand-written prompt tokens or JSON extractors.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from config import LLM_TEMPERATURE, OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_NUM_CTX

log = logging.getLogger(__name__)


class LLMEngine:
    def __init__(self) -> None:
        self._url = f"{OLLAMA_BASE_URL}/api/chat"
        log.info("LLMEngine ready model=%s", OLLAMA_MODEL)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "keep_alive": "30m",
            "options": {
                "temperature": LLM_TEMPERATURE if temperature is None else temperature,
                "num_ctx": OLLAMA_NUM_CTX,
                "num_predict": 300,
            },
        }
        if tools:
            payload["tools"] = tools

        async with httpx.AsyncClient(timeout=120) as c:
            r = await c.post(self._url, json=payload)
            r.raise_for_status()

        return r.json()["message"]  # {"role", "content", "tool_calls"?}
