from __future__ import annotations

import json
import sqlite3
import threading
from typing import Any

from config import HISTORY_DB_PATH, HISTORY_TURNS


class History:
    """Per-user conversation history, persisted in SQLite."""

    def __init__(self, max_turns: int = HISTORY_TURNS, db_path: str = HISTORY_DB_PATH) -> None:
        self.max_turns = max_turns
        self._limit = max_turns * 2
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS messages ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, uid INTEGER NOT NULL, payload TEXT NOT NULL)"
        )
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_messages_uid ON messages(uid, id)")
        self._db.commit()
        self._repair()

    def _repair(self) -> None:
        """Startup only: a pending approval does not survive a restart, so drop dangling tool turns."""
        with self._lock:
            uids = [r[0] for r in self._db.execute("SELECT DISTINCT uid FROM messages")]
            for uid in uids:
                rows = self._db.execute(
                    "SELECT id, payload FROM messages WHERE uid=? ORDER BY id", (uid,)
                ).fetchall()
                items = [(i, json.loads(p)) for i, p in rows]
                while items and (
                    items[-1][1].get("role") == "tool"
                    or (items[-1][1].get("role") == "assistant" and items[-1][1].get("tool_calls"))
                ):
                    self._db.execute("DELETE FROM messages WHERE id=?", (items.pop()[0],))
                while items and items[0][1].get("role") != "user":
                    self._db.execute("DELETE FROM messages WHERE id=?", (items.pop(0)[0],))
            self._db.commit()

    def get(self, uid: int) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT payload FROM messages WHERE uid=? ORDER BY id DESC LIMIT ?",
                (uid, self._limit),
            ).fetchall()
        msgs = [json.loads(r[0]) for r in reversed(rows)]
        while msgs and msgs[0].get("role") != "user":   # window must start on a user turn
            msgs.pop(0)
        return msgs

    def add(self, uid: int, role_or_msg: str | dict[str, Any], content: str | None = None) -> None:
        if isinstance(role_or_msg, dict):
            msg = dict(role_or_msg)
        else:
            msg = {"role": str(role_or_msg), "content": "" if content is None else str(content)}
        with self._lock:
            self._db.execute(
                "INSERT INTO messages (uid, payload) VALUES (?, ?)", (uid, json.dumps(msg, default=str))
            )
            self._db.execute(
                "DELETE FROM messages WHERE uid=? AND id NOT IN "
                "(SELECT id FROM messages WHERE uid=? ORDER BY id DESC LIMIT ?)",
                (uid, uid, self._limit),
            )
            self._db.commit()

    def add_message(self, uid: int, msg: dict[str, Any]) -> None:
        self.add(uid, msg)

    def clear(self, uid: int) -> None:
        with self._lock:
            self._db.execute("DELETE FROM messages WHERE uid=?", (uid,))
            self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()
