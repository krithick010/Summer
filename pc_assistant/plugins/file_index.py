"""
plugins/file_index.py
----------------------
Background file indexer with SQLite and FTS5 / LIKE fallback search.

Maintains a lightweight background index of local files across configured roots
with generational cleanup, periodic refresh, and merge-search with live scanning.
Does NOT start with `p_` so Orchestrator ignores it during plugin auto-discovery.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import re
import sqlite3
import threading
import time
from typing import Any

import config
from plugins.file_search import (
    FileMatch,
    _match_candidate,
    _should_skip_dir,
    is_excluded_path,
    is_reparse_point,
    is_sensitive_path,
    roots_for_mode,
    split_stem_tokens,
)

log = logging.getLogger(__name__)


def _get_default_db_path() -> Path:
    data_dir = Path(getattr(config, "DATA_DIR", "data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "file_index.db"


class FileIndexer:
    """Manages SQLite file index, background crawler thread, and fast search."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path) if db_path else _get_default_db_path()
        self.has_fts5: bool = False
        self.is_running: bool = False
        self.total_indexed: int = 0
        self.last_completed: float | None = None
        self.last_error: str | None = None

        self._stop_event = threading.Event()
        self._crawler_thread: threading.Thread | None = None
        self._lock = threading.Lock()

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create a configured SQLite connection."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
        except Exception:
            pass
        return conn

    def _init_db(self) -> None:
        """Initialize database tables, indexes, and test for FTS5."""
        conn = self._get_connection()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS files(
                    id INTEGER PRIMARY KEY,
                    path TEXT UNIQUE,
                    name TEXT,
                    name_norm TEXT,
                    ext TEXT,
                    size INTEGER,
                    mtime REAL,
                    is_dir INTEGER,
                    gen INTEGER
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_files_ext ON files(ext);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_files_is_dir ON files(is_dir);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_files_mtime ON files(mtime);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_files_gen ON files(gen);")

            # Check FTS5 availability
            self.has_fts5 = self._check_fts5(conn)
            if self.has_fts5:
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS files_fts USING fts5(
                        name_norm,
                        content='files',
                        content_rowid='id'
                    );
                    """
                )
                conn.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS files_ai AFTER INSERT ON files BEGIN
                        INSERT INTO files_fts(rowid, name_norm) VALUES (new.id, new.name_norm);
                    END;
                    """
                )
                conn.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS files_ad AFTER DELETE ON files BEGIN
                        INSERT INTO files_fts(files_fts, rowid, name_norm) VALUES('delete', old.id, old.name_norm);
                    END;
                    """
                )
                conn.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS files_au AFTER UPDATE ON files BEGIN
                        INSERT INTO files_fts(files_fts, rowid, name_norm) VALUES('delete', old.id, old.name_norm);
                        INSERT INTO files_fts(rowid, name_norm) VALUES (new.id, new.name_norm);
                    END;
                    """
                )
            conn.commit()

            # Load initial count
            row = conn.execute("SELECT count(*) FROM files").fetchone()
            if row:
                self.total_indexed = row[0]
                if self.total_indexed > 0:
                    self.last_completed = time.time()
        except Exception as exc:
            log.warning("FileIndexer DB init warning: %s", exc)
        finally:
            conn.close()

    def _check_fts5(self, conn: sqlite3.Connection) -> bool:
        try:
            conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS _fts_probe USING fts5(x)")
            conn.execute("DROP TABLE IF EXISTS _fts_probe")
            return True
        except Exception:
            return False

    def is_ready(self) -> bool:
        """Return True if the index contains indexed files and has completed a crawl."""
        return self.total_indexed > 0 and self.last_completed is not None

    def crawl_sync(self, roots: list[Path] | None = None, batch_size: int = 5000) -> int:
        """Crawl target roots synchronously, batch insert, and clean up unvisited files."""
        with self._lock:
            self.is_running = True
            self.last_error = None

        conn = self._get_connection()
        try:
            target_roots = roots if roots is not None else roots_for_mode()
            row = conn.execute("SELECT coalesce(max(gen), 0) FROM files").fetchone()
            current_gen = (row[0] if row else 0) + 1

            batch: list[tuple[Any, ...]] = []
            items_indexed = 0
            items_seen = 0
            visited_dirs: set[Path] = set()

            for root_dir in target_roots:
                if self._stop_event.is_set():
                    break
                if not root_dir.exists() or not root_dir.is_dir():
                    continue
                if is_sensitive_path(root_dir) or is_excluded_path(root_dir):
                    continue

                stack = [root_dir]
                while stack and not self._stop_event.is_set():
                    curr_dir = stack.pop()
                    try:
                        curr_res = curr_dir.resolve()
                        if curr_res in visited_dirs:
                            continue
                        visited_dirs.add(curr_res)
                    except Exception:
                        pass

                    try:
                        with os.scandir(str(curr_dir)) as it:
                            for entry in it:
                                if self._stop_event.is_set():
                                    break
                                items_seen += 1
                                if items_seen % 2000 == 0:
                                    time.sleep(0)  # yield CPU to event loop/other threads

                                name = entry.name
                                if name.startswith(".") and name not in (".", ".."):
                                    continue

                                full_p = Path(entry.path)
                                if is_reparse_point(full_p):
                                    continue
                                if is_sensitive_path(full_p):
                                    continue
                                if is_excluded_path(full_p):
                                    continue

                                try:
                                    is_directory = entry.is_dir(follow_symlinks=False)
                                except OSError:
                                    continue

                                if is_directory:
                                    if _should_skip_dir(name, full_p):
                                        continue
                                    stack.append(full_p)
                                    try:
                                        st = entry.stat(follow_symlinks=False)
                                        norm_stem = " ".join(split_stem_tokens(full_p.stem))
                                        batch.append((
                                            str(full_p.resolve()),
                                            name,
                                            norm_stem,
                                            "",
                                            0,
                                            st.st_mtime,
                                            1,
                                            current_gen,
                                        ))
                                        items_indexed += 1
                                    except Exception:
                                        pass
                                else:
                                    try:
                                        st = entry.stat(follow_symlinks=False)
                                        norm_stem = " ".join(split_stem_tokens(full_p.stem))
                                        ext = full_p.suffix.lower()
                                        batch.append((
                                            str(full_p.resolve()),
                                            name,
                                            norm_stem,
                                            ext,
                                            st.st_size,
                                            st.st_mtime,
                                            0,
                                            current_gen,
                                        ))
                                        items_indexed += 1
                                    except Exception:
                                        pass

                                if len(batch) >= batch_size:
                                    self._insert_batch(conn, batch)
                                    batch.clear()

                    except (PermissionError, OSError):
                        continue

            if batch and not self._stop_event.is_set():
                self._insert_batch(conn, batch)
                batch.clear()

            if not self._stop_event.is_set():
                # Generational cleanup: purge any files not visited in this crawl
                conn.execute("DELETE FROM files WHERE gen < ?", (current_gen,))
                conn.commit()

                count_row = conn.execute("SELECT count(*) FROM files").fetchone()
                self.total_indexed = count_row[0] if count_row else items_indexed
                self.last_completed = time.time()
                log.info("File crawl completed. Total indexed: %d", self.total_indexed)

            return items_indexed

        except Exception as exc:
            self.last_error = str(exc)
            log.exception("Error during file indexing crawl: %s", exc)
            return 0
        finally:
            conn.close()
            with self._lock:
                self.is_running = False

    def _insert_batch(self, conn: sqlite3.Connection, batch: list[tuple[Any, ...]]) -> None:
        """Insert or update batch of files."""
        sql = """
            INSERT INTO files(path, name, name_norm, ext, size, mtime, is_dir, gen)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                name=excluded.name,
                name_norm=excluded.name_norm,
                ext=excluded.ext,
                size=excluded.size,
                mtime=excluded.mtime,
                is_dir=excluded.is_dir,
                gen=excluded.gen;
        """
        conn.executemany(sql, batch)
        conn.commit()

    def query(
        self,
        tokens: list[str],
        ext_filters: list[str] | None = None,
        limit: int = 50,
        force_like: bool = False,
    ) -> list[dict[str, Any]]:
        """Query index using FTS5 or LIKE fallback."""
        conn = self._get_connection()
        exts = [e.lower() for e in (ext_filters or [])]
        rows: list[sqlite3.Row] = []

        try:
            if not force_like and self.has_fts5 and tokens:
                # FTS5 path: build match query
                clean_tokens = [re.sub(r"[^\w]", "", t) for t in tokens if re.sub(r"[^\w]", "", t)]
                if clean_tokens:
                    fts_expr = " AND ".join(f'"{t}"*' for t in clean_tokens)
                    params: list[Any] = [fts_expr]
                    sql = """
                        SELECT f.id, f.path, f.name, f.name_norm, f.ext, f.size, f.mtime, f.is_dir
                        FROM files_fts(?) fts
                        JOIN files f ON f.id = fts.rowid
                    """
                    if exts:
                        sql += f" WHERE f.ext IN ({','.join('?' for _ in exts)})"
                        params.extend(exts)
                    sql += f" ORDER BY f.mtime DESC LIMIT {max(10, limit * 2)}"
                    try:
                        rows = conn.execute(sql, params).fetchall()
                    except Exception as fts_exc:
                        log.warning("FTS5 query failed (%s), falling back to LIKE", fts_exc)
                        rows = []

            # Fallback to LIKE if FTS5 returned no rows or force_like is True
            if not rows:
                where_clauses: list[str] = []
                params = []
                for t in tokens:
                    where_clauses.append("name_norm LIKE ?")
                    params.append(f"%{t.lower()}%")
                if exts:
                    where_clauses.append(f"ext IN ({','.join('?' for _ in exts)})")
                    params.extend(exts)

                where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
                sql = f"""
                    SELECT id, path, name, name_norm, ext, size, mtime, is_dir
                    FROM files
                    {where_sql}
                    ORDER BY mtime DESC LIMIT {max(10, limit * 2)}
                """
                rows = conn.execute(sql, params).fetchall()

            return [dict(r) for r in rows]
        except Exception as exc:
            log.warning("Error querying file index: %s", exc)
            return []
        finally:
            conn.close()

    def search(
        self,
        name_tokens: list[str],
        ext_filters: list[str] | None = None,
        max_results: int = 5,
        force_like: bool = False,
    ) -> list[FileMatch]:
        """Search the index and return verified, ranked FileMatch objects."""
        raw_rows = self.query(
            tokens=name_tokens,
            ext_filters=ext_filters,
            limit=max_results * 5,
            force_like=force_like,
        )
        scored: list[tuple[int, float, FileMatch]] = []

        for row in raw_rows:
            p = Path(row["path"])
            # Re-validate existence and security boundaries
            if not p.exists() or is_sensitive_path(p) or is_excluded_path(p):
                continue

            match_res = _match_candidate(p, name_tokens, ext_filters or [])
            if match_res is not None:
                score, approx = match_res
                scored.append((
                    score,
                    row["mtime"],
                    FileMatch(path=p, size=row["size"], mtime=row["mtime"], approximate=approx),
                ))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [m for _, _, m in scored[:max_results]]

    def reindex(self) -> None:
        """Trigger a background crawl immediately."""
        with self._lock:
            if self.is_running:
                log.info("Crawl already in progress.")
                return
        thread = threading.Thread(target=self.crawl_sync, daemon=True, name="FileIndexCrawl")
        thread.start()

    def rebuild(self) -> None:
        """Alias for reindex."""
        self.reindex()

    def start_background_worker(self) -> None:
        """Start standing background crawler thread with refresh loop."""
        with self._lock:
            if self._crawler_thread is not None and self._crawler_thread.is_alive():
                return
            self._stop_event.clear()
            self._crawler_thread = threading.Thread(target=self._worker_loop, daemon=True, name="FileIndexWorker")
            self._crawler_thread.start()

    def _worker_loop(self) -> None:
        """Periodic background refresh loop."""
        # Initial crawl
        self.crawl_sync()
        refresh_hours = getattr(config, "FILE_INDEX_REFRESH_HOURS", 6.0)
        refresh_seconds = max(60.0, float(refresh_hours) * 3600.0)

        while not self._stop_event.is_set():
            # Sleep in 5-second intervals to allow fast shutdown
            slept = 0.0
            while slept < refresh_seconds and not self._stop_event.is_set():
                time.sleep(min(5.0, refresh_seconds - slept))
                slept += 5.0
            if not self._stop_event.is_set():
                self.crawl_sync()

    def stop(self) -> None:
        """Signal background crawler worker to stop."""
        self._stop_event.set()

    def get_status(self) -> dict[str, Any]:
        """Return status information for inspection."""
        roots = roots_for_mode()
        return {
            "is_running": self.is_running,
            "total_indexed": self.total_indexed,
            "last_completed": self.last_completed,
            "last_error": self.last_error,
            "mode": getattr(config, "FILE_SEARCH_MODE", "home"),
            "roots": [str(r) for r in roots],
        }

    def format_status_message(self) -> str:
        """Format index status for the /index Telegram command."""
        status = self.get_status()
        count = status["total_indexed"]
        mode = status["mode"]
        roots = roots_for_mode()

        # Format roots list
        root_labels = []
        for r in roots:
            lbl = r.name or str(r)
            root_labels.append(lbl)
        roots_str = ", ".join(root_labels) if root_labels else "none"

        # Format last crawl time
        if status["last_completed"]:
            delta = max(0.0, time.time() - status["last_completed"])
            if delta < 60:
                time_str = f"{int(delta)}s ago"
            elif delta < 3600:
                time_str = f"{int(delta / 60)}m ago"
            else:
                time_str = f"{delta / 3600:.1f}h ago"
        else:
            time_str = "in progress" if status["is_running"] else "never"

        return f"Indexed {count:,} files. Last crawl: {time_str}. Mode: {mode} ({roots_str})."


# Singleton instance
file_indexer = FileIndexer()

