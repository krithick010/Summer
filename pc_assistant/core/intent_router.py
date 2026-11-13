"""
core/intent_router.py
---------------------
Deterministic intent router for file requests (R1–R7).
Evaluates incoming messages before calling LLM, eliminating model hallucination for file operations.
"""

from __future__ import annotations

import difflib
import logging
import os
from pathlib import Path
import re
import time
from typing import Any

from plugins.base import IntentContext, PluginBase, RoutedCall
from plugins.file_search import (
    COMMON_EXTS,
    KNOWN_USER_FOLDERS,
    resolve_user_path,
)

log = logging.getLogger(__name__)

# File command verbs matched within first 3 tokens
FILE_VERBS = {
    "send", "sent", "share", "give", "get", "fetch",
    "mail", "email", "find", "search", "locate", "show",
    "look", "need", "want", "where", "open", "browse", "list",
}

FIND_VERBS = {"find", "search", "locate", "where", "look"}
LIST_VERBS = {"list", "browse"}
PRONOUNS = {"it", "that", "this", "them", "that again", "it again"}

# Common extension typos
TYPO_EXTS = {
    "rxt": "txt", "tzt": "txt", "txr": "txt", "tt": "txt",
    "pfd": "pdf", "pd": "pdf",
    "dock": "docx", "dox": "docx", "docz": "docx",
    "pnj": "png", "jpj": "jpg", "peg": "jpeg",
    "zipp": "zip", "zpi": "zip", "pyt": "py",
}

FILE_TYPE_WORDS = {
    "file", "files", "folder", "folders", "directory", "directories",
    "pdf", "doc", "docx", "document", "documents", "sheet", "sheets",
    "image", "images", "photo", "photos", "picture", "pictures",
    "video", "videos", "movie", "movies", "audio", "song", "songs",
    "zip", "archive", "txt", "text", "paper", "report", "notes",
}

NON_FILE_KEYWORDS = {
    "weather", "forecast", "temperature", "news", "headline", "headlines",
    "ai news", "system info", "cpu", "ram", "disk", "memory", "wikipedia",
    "wiki", "joke", "jokes", "hello", "hi", "hey", "help", "thanks", "thank",
    "bye", "goodbye", "yo", "sup",
}


def _is_non_file_message(raw_lower: str) -> bool:
    """Check if message is clearly non-file related."""
    # Exact single/short greetings or conversational words
    if raw_lower in ("hi", "yo", "hello", "hey", "sup", "bye", "goodbye", "thanks", "thank you", "help"):
        return True
    if raw_lower in ("weather", "news", "ai news", "wikipedia"):
        return True
    if raw_lower.startswith(("hi ", "yo ", "hello ", "hey ", "how's the weather", "how is the weather", "what's the weather", "what is the weather")):
        return True
    if re.search(r"\b(weather|forecast)\b", raw_lower) and not any(ext in raw_lower for ext in (".txt", ".csv", ".json", ".pdf")):
        return True
    if re.search(r"\b(ai\s+news|latest\s+news|any\s+news)\b", raw_lower):
        return True
    if re.search(r"\b(on\s+wikipedia|wikipedia\s+page|wikipedia\s+summary)\b", raw_lower):
        return True
    if raw_lower.startswith(("who is ", "who was ", "tell me a joke", "tell me about ", "explain ")):
        return True
    return False


def _check_email_delivery(raw_lower: str) -> tuple[bool, str | None]:
    """Check if email delivery is requested, and ensure no other recipient pattern exists.

    Returns (is_email, error_or_drop_reason).
    If another recipient is present (e.g. "to mom", "to <email>"), returns (False, "other_recipient").
    """
    has_mail = bool(re.search(r"\b(mail|email)\b", raw_lower))
    if not has_mail:
        return False, None

    # Check for other recipient patterns like "to mom", "to alice", "to <email>"
    to_other = re.search(r"\bto\s+(?!me\b|myself\b)[a-zA-Z0-9_.-]+", raw_lower)
    if to_other:
        return False, "other_recipient"

    email_addr = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", raw_lower)
    if email_addr:
        return False, "other_recipient"

    to_someone = re.search(r"\b(email|mail)\s+(my\s+\w+|\w+)\s+(saying|that|about|hi)\b", raw_lower)
    if to_someone:
        return False, "other_recipient"

    return True, None


def _fuzzy_find_verb(tokens: list[str]) -> tuple[str | None, int]:
    """Find a file verb in the first 3 tokens, matching fuzzily."""
    for idx, tok in enumerate(tokens[:3]):
        t_clean = tok.lower().strip(".,;:\"'!?")
        if t_clean in FILE_VERBS:
            return t_clean, idx
        close = difflib.get_close_matches(t_clean, FILE_VERBS, n=1, cutoff=0.75)
        if close:
            return close[0], idx
    return None, -1


def match_file_intent(message: str, ctx: IntentContext | None = None) -> RoutedCall | None:
    """Evaluate message against file rules R1–R7."""
    raw = (message or "").strip().strip("'\"").strip()
    if not raw:
        return None

    raw_lower = raw.lower()

    # Guard against clear non-file queries
    if _is_non_file_message(raw_lower):
        return None

    # Email delivery check
    is_email, drop_reason = _check_email_delivery(raw_lower)
    if drop_reason == "other_recipient":
        # Leave for model / send_email
        return None

    delivery = "email" if is_email else "telegram"

    # Context values
    if ctx is None:
        ctx = IntentContext()
    now = time.time()

    # -------------------------------------------------------------------------
    # R7. Folder listing ("what's in / list / show me / contents of <folder>")
    # -------------------------------------------------------------------------
    list_m = re.match(
        r"^(?:can\s+you\s+|could\s+you\s+|please\s+)?(?:list\s+files\s+in|list\s+folder|list\s+directory|list|browse\s+folder|browse\s+directory|browse|show\s+files\s+in|show\s+folder|show\s+directory|show\s+me\s+contents\s+of|show\s+contents\s+of|contents\s+of|what\s+is\s+in|what's\s+in|whats\s+in)\s+(.+)$",
        raw,
        flags=re.IGNORECASE,
    )
    if list_m:
        cand_folder = list_m.group(1).strip().strip("'\"").strip()
        # Strip leading "my " or "the "
        cand_folder = re.sub(r"^(?:my|the)\s+", "", cand_folder, flags=re.IGNORECASE).strip()
        if cand_folder and not _is_non_file_message(cand_folder.lower()):
            # Resolve folder by alias first, then index / path
            cand_lower = cand_folder.lower()
            if cand_lower in KNOWN_USER_FOLDERS:
                return RoutedCall(tool_name="list_folder", kwargs={"path": cand_folder}, rule="R7")
            res_path = resolve_user_path(cand_folder)
            if res_path and res_path.is_dir():
                return RoutedCall(tool_name="list_folder", kwargs={"path": str(res_path.resolve())}, rule="R7")
            try:
                p_cand = Path(os.path.expandvars(os.path.expanduser(cand_folder)))
                if p_cand.exists() and p_cand.is_dir():
                    return RoutedCall(tool_name="list_folder", kwargs={"path": str(p_cand.resolve())}, rule="R7")
            except Exception:
                pass
            return RoutedCall(tool_name="list_folder", kwargs={"path": cand_folder}, rule="R7")

    # -------------------------------------------------------------------------
    # R5. Pronoun follow-ups ("mail it", "send it", "share that again")
    # -------------------------------------------------------------------------
    pronoun_m = re.match(
        r"^(?:can\s+you\s+|could\s+you\s+|please\s+)?(?:send|sent|share|give|get|fetch|mail|email|open)\s+(?:it|that|this|them)(?:\s+again)?$",
        raw,
        flags=re.IGNORECASE,
    )
    if pronoun_m:
        last_file = ctx.last_file_path
        last_ts = ctx.timestamps.get("last_file_path", 0)
        if last_file and (now - last_ts < 1800):  # 30 min
            kw: dict[str, Any] = {"path": last_file}
            if delivery == "email":
                kw["delivery"] = "email"
            return RoutedCall(tool_name="share_file", kwargs=kw, rule="R5")

    # -------------------------------------------------------------------------
    # R6. Location-only follow-up ("it's in desktop", "actually Documents", etc.)
    # -------------------------------------------------------------------------
    last_q = ctx.last_query
    last_q_ts = ctx.timestamps.get("last_query", 0)
    if last_q and (now - last_q_ts < 600):  # 10 min
        loc_m = re.match(
            r"^(?:it's\s+in|its\s+in|actually|it\s+is\s+in|in|inside)\s+(.+)$",
            raw,
            flags=re.IGNORECASE,
        )
        if loc_m:
            loc_term = loc_m.group(1).strip()
            combined = f"{last_q} in {loc_term}"
            kw = {"path": combined}
            if delivery == "email":
                kw["delivery"] = "email"
            return RoutedCall(tool_name="share_file", kwargs=kw, rule="R6")
        if raw_lower in KNOWN_USER_FOLDERS:
            combined = f"{last_q} in {raw}"
            kw = {"path": combined}
            if delivery == "email":
                kw["delivery"] = "email"
            return RoutedCall(tool_name="share_file", kwargs=kw, rule="R6")

    # -------------------------------------------------------------------------
    # R1. Existing absolute or alias path in message
    # -------------------------------------------------------------------------
    cand_path = resolve_user_path(raw)
    if cand_path is not None and cand_path.exists():
        kw = {"path": str(cand_path.resolve())}
        if delivery == "email":
            kw["delivery"] = "email"
        return RoutedCall(tool_name="share_file", kwargs=kw, rule="R1")

    try:
        norm_raw = raw.replace("/", os.sep).replace("\\", os.sep)
        p_direct = Path(os.path.expandvars(os.path.expanduser(norm_raw)))
        if p_direct.exists():
            kw = {"path": str(p_direct.resolve())}
            if delivery == "email":
                kw["delivery"] = "email"
            return RoutedCall(tool_name="share_file", kwargs=kw, rule="R1")
    except Exception:
        pass

    # Tokenize message
    raw_tokens = raw.split()
    verb, verb_idx = _fuzzy_find_verb(raw_tokens)
    has_verb = verb is not None

    # Remaining text after verb
    if has_verb:
        rem_tokens = raw_tokens[:verb_idx] + raw_tokens[verb_idx + 1:]
        # Strip leading filler words: "me", "the", "a", "an", "for", "please", "is", "are"
        while rem_tokens and rem_tokens[0].lower().strip(".,;:\"'!?") in ("me", "the", "a", "an", "for", "please", "with", "is", "are"):
            rem_tokens.pop(0)
        target_text = " ".join(rem_tokens).strip()
    else:
        target_text = raw

    # -------------------------------------------------------------------------
    # R2. Token with a known file extension (e.g. "Tree.txt", "tree.txt ig", "Ahre me tree.txt")
    # -------------------------------------------------------------------------
    ext_token_found = False
    for t in raw_tokens:
        t_clean = t.strip(".,;:\"'!?")
        if "." in t_clean and not t_clean.lower().startswith(("http://", "https://", "www.")):
            parts = t_clean.rsplit(".", 1)
            if len(parts) == 2:
                ext = parts[1].lower()
                if ext in COMMON_EXTS or ext in TYPO_EXTS or (2 <= len(ext) <= 4 and ext.isalnum()):
                    ext_token_found = True
                    break

    if ext_token_found:
        # Determine tool: find vs share
        if has_verb and verb in FIND_VERBS:
            return RoutedCall(tool_name="find_files", kwargs={"query": target_text or raw}, rule="R2")
        kw = {"path": target_text or raw}
        if delivery == "email":
            kw["delivery"] = "email"
        return RoutedCall(tool_name="share_file", kwargs=kw, rule="R2")

    # -------------------------------------------------------------------------
    # R3. Command verb + remaining tokens
    # -------------------------------------------------------------------------
    if has_verb and target_text:
        # Check if type word or "file"/"folder" is present
        target_tokens = set(re.findall(r"[A-Za-z0-9]+", target_text.lower()))
        has_type_word = bool(target_tokens.intersection(FILE_TYPE_WORDS))

        has_index_hit = False
        try:
            from plugins.file_search import search_files  # noqa: PLC0415
            matches, _ = search_files(target_text, limit=3)
            if any(m.tier in ("A", "B") or m.score >= 40 for m in matches):
                has_index_hit = True
        except Exception:
            pass

        if has_type_word or has_index_hit:
            if verb in FIND_VERBS:
                return RoutedCall(tool_name="find_files", kwargs={"query": target_text}, rule="R3")
            kw = {"path": target_text}
            if delivery == "email":
                kw["delivery"] = "email"
            return RoutedCall(tool_name="share_file", kwargs=kw, rule="R3")

    # -------------------------------------------------------------------------
    # R4. Bare message of 1-6 tokens with NO verb
    # -------------------------------------------------------------------------
    if not has_verb:
        name_tokens = [w for w in re.findall(r"[A-Za-z0-9]+", raw.lower()) if w not in ("file", "folder", "directory", "files", "folders")]
        if not name_tokens and "." not in raw:
            return None
        # Never route a single bare word without an extension
        if len(name_tokens) == 1 and "." not in raw:
            return None
        if (1 <= len(name_tokens) <= 6) or ("." in raw and len(raw_tokens) <= 6):
            # Check search_files
            try:
                from plugins.file_search import search_files  # noqa: PLC0415
                matches, _ = search_files(raw, limit=5)
                tier_a_matches = [m for m in matches if m.tier == "A" or m.score >= 60]
                if tier_a_matches:
                    query_tokens = set(name_tokens)
                    for m in tier_a_matches:
                        hit_stem = m.path.stem.lower()
                        hit_tokens = set(re.findall(r"[A-Za-z0-9]+", hit_stem))
                        if hit_tokens:
                            coverage = len(query_tokens.intersection(hit_tokens)) / len(hit_tokens)
                            if coverage >= 0.5 and ("." in raw or len(name_tokens) >= 2):
                                kw = {"path": raw}
                                if delivery == "email":
                                    kw["delivery"] = "email"
                                return RoutedCall(tool_name="share_file", kwargs=kw, rule="R4")
            except Exception:
                pass

            # Also check index vocabulary if index search returned nothing
            try:
                from plugins.file_index import file_indexer  # noqa: PLC0415
                vocab = file_indexer.get_vocabulary()
                if vocab:
                    if len(name_tokens) >= 2 and all(w in vocab or difflib.get_close_matches(w, vocab, n=1, cutoff=0.8) for w in name_tokens):
                        kw = {"path": raw}
                        if delivery == "email":
                            kw["delivery"] = "email"
                        return RoutedCall(tool_name="share_file", kwargs=kw, rule="R4")
            except Exception:
                pass

    return None


def classify_file_intent(text: str, context: Any = None) -> tuple[str, dict[str, Any]] | None:
    """Backwards-compatible helper returning (tool_name, kwargs) or None."""
    ctx: IntentContext | None = None
    if isinstance(context, IntentContext):
        ctx = context
    elif isinstance(context, dict):
        ctx = IntentContext(
            user_id=context.get("user_id", 0),
            last_file_path=context.get("last_file_path", ""),
            last_query=context.get("last_query", ""),
        )
    res = match_file_intent(text, ctx)
    if res:
        return res.tool_name, res.kwargs
    return None


class IntentRouter:
    """Deterministic intent router that queries plugins in registry order."""

    def __init__(self, registry: dict[str, PluginBase] | None = None):
        self.registry = registry or {}
        self.contexts: dict[int | str, IntentContext] = {}

    def get_context(self, user_id: int | str) -> IntentContext:
        if user_id not in self.contexts:
            self.contexts[user_id] = IntentContext(user_id=user_id)
        return self.contexts[user_id]

    def update_context(
        self,
        user_id: int | str,
        *,
        last_file_path: str | None = None,
        last_query: str | None = None,
    ) -> None:
        ctx = self.get_context(user_id)
        now = time.time()
        if last_file_path is not None:
            ctx.last_file_path = last_file_path
            ctx.timestamps["last_file_path"] = now
        if last_query is not None:
            ctx.last_query = last_query
            ctx.timestamps["last_query"] = now

    def route(self, user_id: int | str, message: str) -> RoutedCall | None:
        ctx = self.get_context(user_id)
        for plugin in self.registry.values():
            if hasattr(plugin, "match_intent"):
                try:
                    res = plugin.match_intent(message, ctx)
                    if isinstance(res, RoutedCall):
                        return res
                    if isinstance(res, tuple) and len(res) == 2 and res[0]:
                        return RoutedCall(tool_name=plugin.name, kwargs=res[1], rule="R0")
                except Exception:
                    pass
        return None
