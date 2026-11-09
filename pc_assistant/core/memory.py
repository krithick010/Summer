"""
core/memory.py
--------------
Persistent RAG-based Long-Term Memory for Summer using ChromaDB and
local Ollama embeddings (nomic-embed-text via langchain_ollama).

Uses nomic-embed-text search prefixes:
- 'search_document: <fact>' when indexing
- 'search_query: <query>' when searching
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import time
from pathlib import Path
from typing import Any

import chromadb
from langchain_ollama import OllamaEmbeddings

from config import (
    DATA_DIR,
    EMBEDDING_MODEL,
    MEMORY_DB_PATH,
    MEMORY_MAX_DISTANCE,
    MEMORY_TOP_K,
    OLLAMA_BASE_URL,
)

log = logging.getLogger(__name__)


class MemoryManager:
    """Manages persistent long-term facts using ChromaDB and nomic-embed-text."""

    def __init__(
        self,
        path: str = MEMORY_DB_PATH,
        collection_name: str = "summer_long_term",
        model: str = EMBEDDING_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        max_distance: float = MEMORY_MAX_DISTANCE,
    ) -> None:
        self.path = path
        self.collection_name = collection_name
        self.model = model
        self.base_url = base_url
        self.max_distance = max_distance

        legacy_mem = Path("./memory_db")
        if not legacy_mem.exists():
            legacy_mem = Path(__file__).resolve().parent.parent / "memory_db"
        target_mem = DATA_DIR / "memory_db"
        if legacy_mem.exists() and not target_mem.exists() and Path(self.path).resolve() != legacy_mem.resolve():
            log.warning(
                "Legacy memory_db folder found at '%s' but '%s' does not exist. "
                "Please move '%s' into '%s' manually.",
                legacy_mem,
                target_mem,
                legacy_mem,
                DATA_DIR,
            )

        Path(self.path).mkdir(parents=True, exist_ok=True)
        log.info("Initialising ChromaDB PersistentClient at: %s", self.path)
        self.client = chromadb.PersistentClient(path=self.path)

        log.info("Initialising OllamaEmbeddings (model=%s, base_url=%s)", self.model, self.base_url)
        # keep_alive in langchain_ollama expects int seconds (1800s = 30m)
        self.embeddings = OllamaEmbeddings(
            model=self.model,
            base_url=self.base_url,
            keep_alive=1800,
        )

        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        self._check_collection_compatibility()

    def _check_collection_compatibility(self) -> None:
        """Warn user if existing collection lacks timestamp metadata from previous versions."""
        try:
            count = self.collection.count()
            if count > 0:
                peek_data = self.collection.peek(limit=min(5, count))
                metas = peek_data.get("metadatas") or []
                if not metas or any(m is None or "ts" not in m for m in metas):
                    log.warning(
                        "⚠️ Legacy ChromaDB collection detected without 'ts' metadata. "
                        "Please delete '%s' to reset and avoid incompatible embeddings.",
                        self.path,
                    )
            log.info(
                "Collection '%s' ready with %d items (max_distance=%.2f).",
                self.collection_name,
                count,
                self.max_distance,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not check collection compatibility: %s", exc)

    def _embed(self, text: str) -> list[float]:
        """Generate a unit-normalized embedding vector for consistent distance calculation."""
        raw_vec = self.embeddings.embed_query(text)
        norm = math.sqrt(sum(x * x for x in raw_vec))
        if norm > 0:
            return [x / norm for x in raw_vec]
        return raw_vec

    def add_memory(self, fact: str) -> bool:
        """Embeds 'search_document: {fact}' and adds clean fact + ts to ChromaDB.

        Uses SHA-256 of lowercased, stripped fact as ID for exact deduplication.
        """
        fact = fact.strip()
        if not fact:
            log.warning("Cannot add empty fact to memory.")
            return False

        doc_id = hashlib.sha256(fact.lower().encode("utf-8")).hexdigest()
        ts = time.time()

        try:
            vector = self._embed(f"search_document: {fact}")
            self.collection.upsert(
                ids=[doc_id],
                documents=[fact],
                embeddings=[vector],
                metadatas=[{"fact": fact, "ts": ts}],
            )
            log.info("Memory saved [id=%s]: %r", doc_id[:8], fact)
            return True
        except Exception as exc:
            log.exception("Failed to add memory: %s", exc)
            return False

    def search_memory(self, query: str, top_k: int = MEMORY_TOP_K) -> str:
        """Embeds 'search_query: {query}', retrieves top-k facts, filters by max_distance,
        and returns results sorted newest-first (ts descending).
        """
        query = query.strip()
        if not query:
            return ""

        try:
            count = self.collection.count()
            if count == 0:
                return ""

            k = min(top_k, count)
            query_vector = self._embed(f"search_query: {query}")
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=k,
                include=["documents", "metadatas", "distances"],
            )

            docs = results.get("documents", [[]])
            distances = results.get("distances", [[]])
            metadatas = results.get("metadatas", [[]])

            if not docs or not docs[0]:
                return ""

            doc_list = docs[0]
            dist_list = distances[0] if distances and distances[0] else [0.0] * len(doc_list)
            meta_list = metadatas[0] if metadatas and metadatas[0] else [{}] * len(doc_list)

            kept: list[dict[str, Any]] = []
            for doc, dist, meta in zip(doc_list, dist_list, meta_list):
                log.debug("Memory candidate doc=%r dist=%.4f (max=%.2f)", doc, dist, self.max_distance)
                if dist is not None and dist <= self.max_distance:
                    fact_text = (meta.get("fact") if meta else "") or (doc.strip() if doc else "")
                    ts = float(meta.get("ts", 0.0)) if meta else 0.0
                    if fact_text:
                        kept.append({"fact": fact_text, "ts": ts, "dist": dist})
                else:
                    log.debug("Dropped memory candidate exceeding max_distance: %r (dist=%.4f)", doc, dist)

            if not kept:
                return ""

            # Sort newest first so recent corrections override older facts
            kept.sort(key=lambda x: x["ts"], reverse=True)

            formatted = "\n".join(f"- {item['fact']}" for item in kept)
            log.info("Retrieved %d relevant memories for query %r", len(kept), query[:40])
            return formatted

        except Exception as exc:
            log.exception("Failed to search memory: %s", exc)
            return ""

    async def aadd_memory(self, fact: str) -> bool:
        """Asynchronously embed and store a memory on a worker thread."""
        return await asyncio.to_thread(self.add_memory, fact)

    # ------------------------------------------------------------------
    # Subject+attribute keyed upsert (used by background extractor)
    # ------------------------------------------------------------------

    @staticmethod
    def _fact_id(subject: str, attribute: str) -> str:
        """Deterministic ID from subject+attribute so corrections overwrite."""
        key = f"{subject.lower().strip()}|{attribute.lower().strip()}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    @staticmethod
    def _fact_text(subject: str, attribute: str, value: str) -> str:
        """Build a readable sentence from structured fact fields."""
        s = "User" if subject.lower().strip() in ("user", "me", "i") else subject.strip().capitalize()
        return f"{s}'s {attribute.strip()} is {value.strip()}"

    def upsert(self, subject: str, attribute: str, value: str) -> bool:
        """Upsert a fact keyed by subject+attribute so corrections overwrite.

        Returns True on success.
        """
        fact = self._fact_text(subject, attribute, value)
        doc_id = self._fact_id(subject, attribute)
        ts = time.time()

        try:
            vector = self._embed(f"search_document: {fact}")
            self.collection.upsert(
                ids=[doc_id],
                documents=[fact],
                embeddings=[vector],
                metadatas=[{
                    "fact": fact,
                    "subject": subject.lower().strip(),
                    "attribute": attribute.lower().strip(),
                    "ts": ts,
                }],
            )
            log.info("Memory upsert [id=%s]: %r", doc_id[:8], fact)
            return True
        except Exception as exc:
            log.exception("Failed to upsert memory: %s", exc)
            return False

    async def aupsert(self, subject: str, attribute: str, value: str) -> bool:
        """Asynchronously upsert a fact on a worker thread."""
        return await asyncio.to_thread(self.upsert, subject, attribute, value)

    async def asearch_memory(self, query: str, k: int = MEMORY_TOP_K) -> str:
        """Asynchronously embed and query memories on a worker thread."""
        return await asyncio.to_thread(self.search_memory, query, k)
