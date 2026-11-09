"""
core/state_manager.py
---------------------
In-memory store for Human-in-the-Loop (HITL) pending approvals.

Flow
~~~~
1. Orchestrator calls ``create_action()`` → gets back a ``PendingAction``.
2. ``main.py`` sends the approval prompt to Telegram and stores the
   ``approval_id`` in the inline-button callback data.
3. When the user taps Approve/Cancel, ``main.py`` calls ``resolve()``.
4. The orchestrator's ``await action.event.wait()`` unblocks and checks
   ``action.approved``.

Thread safety
~~~~~~~~~~~~~
All mutations happen on the asyncio event loop — no extra locks needed for
the asyncio.Event.  ``cleanup_expired()`` is designed to be called from an
asyncio periodic task (see main.py lifespan).
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Optional

from config import HITL_TIMEOUT_SECONDS


# ---------------------------------------------------------------------------
# Data container for one pending approval
# ---------------------------------------------------------------------------


class PendingAction:
    """Represents one action waiting for user approval via Telegram."""

    def __init__(self, tool_name: str, kwargs: dict, description: str, user_id: int):
        self.approval_id: str = str(uuid.uuid4())[:8]
        self.tool_name: str = tool_name          # was plugin_name — aligned with main.py
        self.kwargs: dict = kwargs
        self.description: str = description
        self.user_id: int = user_id              # tracks which Telegram user owns this
        self.timestamp: float = time.monotonic()
        self.event: asyncio.Event = asyncio.Event()
        self.approved: bool = False

    def is_expired(self) -> bool:
        return (time.monotonic() - self.timestamp) > HITL_TIMEOUT_SECONDS


# ---------------------------------------------------------------------------
# State manager (singleton instance exported at module level)
# ---------------------------------------------------------------------------


class StateManager:
    """Manages all in-flight HITL approval requests."""

    def __init__(self) -> None:
        # approval_id → PendingAction
        self._pending: dict[str, PendingAction] = {}
        # user_id → approval_id  (one active approval per user at a time)
        self._user_index: dict[int, str] = {}

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    def create_action(
        self,
        tool_name: str,
        kwargs: dict,
        description: str,
        user_id: int,
    ) -> PendingAction:
        """Register a new pending action and return it.

        If the same user already has a pending action it is auto-cancelled
        before the new one is registered.
        """
        # Cancel any pre-existing approval for this user
        self.cancel_pending(user_id)

        action = PendingAction(tool_name, kwargs, description, user_id)
        self._pending[action.approval_id] = action
        self._user_index[user_id] = action.approval_id
        return action

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_action(self, approval_id: str) -> Optional[PendingAction]:
        """Look up by approval_id (from inline-button callback data)."""
        return self._pending.get(approval_id)

    # ------------------------------------------------------------------
    # Resolve (called by Telegram callback handler in main.py)
    # ------------------------------------------------------------------

    def resolve(self, approval_id: str) -> Optional[dict]:
        """Resolve an approval by id.

        Sets the asyncio Event so the waiting orchestrator unblocks.

        Returns the action's dict payload so ``main.py`` can execute the
        plugin, or ``None`` if the id is unknown / already resolved.
        """
        action = self._pending.pop(approval_id, None)
        if action is None:
            return None

        # Clean up user index
        self._user_index.pop(action.user_id, None)

        # Unblock the orchestrator's ``await action.event.wait()``
        action.approved = True
        action.event.set()

        return {"tool_name": action.tool_name, "kwargs": action.kwargs}

    # ------------------------------------------------------------------
    # Cancel helpers
    # ------------------------------------------------------------------

    def cancel_pending(self, user_id: int) -> bool:
        """Cancel any pending approval for a given Telegram user_id.

        Called by:
        - ``/cancel`` command handler in main.py
        - ``create_action()`` (auto-cancel before creating a new one)

        Returns True if something was cancelled.
        """
        approval_id = self._user_index.pop(user_id, None)
        if approval_id is None:
            return False

        action = self._pending.pop(approval_id, None)
        if action:
            action.approved = False
            action.event.set()   # unblock any waiting coroutine
        return True

    def cancel_by_id(self, approval_id: str) -> bool:
        """Cancel a specific approval by id (used by the ❌ Cancel button)."""
        action = self._pending.pop(approval_id, None)
        if action is None:
            return False
        self._user_index.pop(action.user_id, None)
        action.approved = False
        action.event.set()
        return True

    # ------------------------------------------------------------------
    # Periodic cleanup
    # ------------------------------------------------------------------

    def cleanup_expired(self) -> int:
        """Cancel all approvals that have exceeded ``HITL_TIMEOUT_SECONDS``.

        Returns the number of actions cleaned up.
        """
        expired = [
            a_id
            for a_id, action in list(self._pending.items())
            if action.is_expired()
        ]
        for a_id in expired:
            self.cancel_by_id(a_id)
        return len(expired)


# ---------------------------------------------------------------------------
# Module-level singleton — imported everywhere as:
#   from core.state_manager import state_manager
# ---------------------------------------------------------------------------
state_manager = StateManager()
