"""
plugins/p_remember.py
---------------------
Memory persistence plugin that allows Summer to remember user facts, preferences,
contacts, and notes permanently in ChromaDB.
"""

from __future__ import annotations

import logging
from typing import Any

from core.memory import MemoryManager
from plugins.base import PluginBase, PluginResult

log = logging.getLogger(__name__)


class RememberPlugin(PluginBase):
    name = "save_memory"
    description = (
        "Save a fact about the user (contacts, numbers, emails, preferences, plans) "
        "as one full standalone sentence."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "fact": {
                "type": "string",
                "description": "Full standalone sentence, e.g. 'Mom's phone number is 555-0123'",
            }
        },
        "required": ["fact"],
    }
    requires_approval = False

    def __init__(self, memory_manager: MemoryManager | None = None) -> None:
        self.memory_manager = memory_manager

    async def execute(self, **kwargs: Any) -> PluginResult:
        fact: str = str(kwargs.get("fact", "")).strip()

        if not fact:
            return PluginResult(
                success=False,
                output="No fact was provided to remember.",
                error="Missing required 'fact' parameter in arguments.",
            )

        manager = self.memory_manager or MemoryManager()
        success = await manager.aadd_memory(fact)

        if success:
            log.info("Successfully remembered fact: %r", fact)
            return PluginResult(
                success=True,
                output="Got it, saved.",
                data={"fact": fact},
            )

        return PluginResult(
            success=False,
            output="Failed to store fact in memory.",
            error="MemoryManager failed to embed or write fact to ChromaDB.",
        )
