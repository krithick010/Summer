"""
core/choices.py
---------------
In-memory store for interactive choice buttons.
"""

from __future__ import annotations

import time
import uuid
from typing import Any


class ChoiceStore:
    """Stores temporary user choices for interactive Telegram inline buttons."""

    def __init__(self) -> None:
        self._stores: dict[str, dict[str, Any]] = {}

    def register(self, user_id: int, choices: list[dict[str, Any]], ttl: int = 600) -> str:
        """Register a list of choices for a specific user with a TTL (seconds)."""
        now = time.time()
        # Clean expired tokens
        expired = [k for k, v in self._stores.items() if now > v.get("expires_at", 0)]
        for k in expired:
            self._stores.pop(k, None)

        token = uuid.uuid4().hex[:8]
        self._stores[token] = {
            "user_id": user_id,
            "choices": choices,
            "expires_at": now + ttl,
        }
        return token

    def has_pending(self, user_id: int) -> bool:
        """Return True if user has any active unexpired choices."""
        now = time.time()
        for v in self._stores.values():
            if v.get("user_id") == user_id and now <= v.get("expires_at", 0):
                return True
        return False

    def get(self, token: str, user_id: int, index: int) -> dict[str, Any] | None:
        """Retrieve a specific choice if token is valid, unexpired, and matches user_id."""
        now = time.time()
        record = self._stores.get(token)
        if record is None:
            return None
        if now > record.get("expires_at", 0):
            self._stores.pop(token, None)
            return None
        if record.get("user_id") != user_id:
            return None
        choices = record.get("choices", [])
        if index < 0 or index >= len(choices):
            return None
        return choices[index]

    def clear(self) -> None:
        """Clear all stored choices."""
        self._stores.clear()


choice_store = ChoiceStore()
