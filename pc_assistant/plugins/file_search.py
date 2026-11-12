"""
plugins/file_search.py
----------------------
Two-tier fast file search, security boundary checking, and compression engine.

Helper module for file discovery and smart sharing; does NOT start with `p_`
so Orchestrator ignores it during plugin auto-discovery.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import re
import time
from typing import Sequence
import zipfile

import config

log = logging.getLogger(__name__)

# Security denylist: sensitive files, keys, credentials, and state databases
SENSITIVE_NAMES = {
    "gmail_token.json",
    "gmail_oauth.json",
    "whatsapp_session",
    "memory_db",
    "history.db",
    ".env",
}
SENSITIVE_EXTS = {".key", ".pem"}

# System, package manager, and build artifact directories pruned in-place
SKIP_DIRS = {
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    "appdata",
    "application data",
    "$recycle.bin",
    "windows",
    "program files",
    "program files (x86)",
    "ms-playwright",
    "tmp_shares",
}

COMMON_EXTS = {
    "pdf", "png", "jpg", "jpeg", "webp", "gif", "docx", "doc", "xlsx",
    "xls", "txt", "csv", "zip", "tar", "gz", "mp4", "mp3", "json", "py"
}

QUERY_STOP_WORDS = {
    "send", "me", "find", "locate", "search", "for", "my", "the", "a",
    "an", "file", "folder", "please", "and", "it", "to", "give"
}


def is_sensitive_path(path: Path | str) -> bool:
    """Return True if ``path`` matches sensitive patterns, credentials, or keys."""
    try:
        p = Path(path).resolve()
    except Exception:
        p = Path(path)

    parts_lower = [part.lower() for part in p.parts]
    name_lower = p.name.lower()

    if p.suffix.lower() in SENSITIVE_EXTS:
        return True

    for part in parts_lower:
        if part in SENSITIVE_NAMES:
            return True
        if "credentials" in part:
            return True
        if part == "id_rsa" or part.startswith("id_rsa"):
            return True
        if part == ".env" or part.startswith(".env") or part.endswith(".env"):
            return True
        if any(part.endswith(ext) for ext in SENSITIVE_EXTS):
            return True
        if "memory_db" in part or "history.db" in part:
            return True

    if name_lower in SENSITIVE_NAMES:
        return True
    if name_lower.startswith("id_rsa") or name_lower.startswith(".env") or name_lower.endswith(".env"):
        return True
    if "credentials" in name_lower:
        return True

    return False


def _parse_query(query: str, file_type: str = "") -> tuple[list[str], str | None]:
    """Extract search tokens and optional extension filter from query."""
    ext_filter: str | None = None
    if file_type:
        clean_ext = file_type.strip().lower()
        ext_filter = clean_ext if clean_ext.startswith(".") else f".{clean_ext}"

    cleaned = query.strip().strip('"\'')
    raw_tokens = [t.lower() for t in re.split(r"[\s\-_,;:]+", cleaned) if t.strip()]

    search_tokens: list[str] = []
    for token in raw_tokens:
        clean_tok = token.lstrip(".")
        if token.startswith(".") and clean_tok in COMMON_EXTS:
            if not ext_filter:
                ext_filter = f".{clean_tok}"
        elif clean_tok in COMMON_EXTS and len(raw_tokens) > 1 and not ext_filter:
            ext_filter = f".{clean_tok}"
        elif clean_tok not in QUERY_STOP_WORDS:
            search_tokens.append(clean_tok)

    if not search_tokens:
        search_tokens = [t.lstrip(".") for t in raw_tokens if t.lstrip(".") not in QUERY_STOP_WORDS] or raw_tokens

    return search_tokens, ext_filter


def _score_candidate(
    candidate: Path,
    search_tokens: list[str],
    ext_filter: str | None,
) -> int:
    """Score candidate file from 0 to 100 based on token match quality."""
    if is_sensitive_path(candidate):
        return 0
    if ext_filter and candidate.suffix.lower() != ext_filter:
        return 0

    name_lower = candidate.name.lower()
    stem_lower = candidate.stem.lower()
    joined_query = " ".join(search_tokens)

    # 1. Exact match on stem or full name
    if joined_query == stem_lower or joined_query == name_lower:
        return 100
    # 2. Starts with query
    if stem_lower.startswith(joined_query):
        return 80
    # 3. All tokens in stem
    if all(t in stem_lower for t in search_tokens):
        return 60
    # 4. Partial tokens in stem
    matching_tokens = sum(1 for t in search_tokens if t in stem_lower)
    if matching_tokens > 0:
        return 40 + min(matching_tokens * 2, 18)
    # 5. Token match in name or immediate parent
    if any(t in name_lower for t in search_tokens):
        return 20

    return 0


def find_files(query: str, file_type: str = "", max_results: int = 5) -> list[Path]:
    """Find files or folders matching ``query`` across two search tiers."""
    cleaned_query = query.strip().strip('"\'')
    if not cleaned_query:
        return []

    # 1. Direct path resolution
    try:
        expanded = os.path.expandvars(os.path.expanduser(cleaned_query))
        candidate = Path(expanded)
        if candidate.exists():
            resolved = candidate.resolve()
            if not is_sensitive_path(resolved):
                return [resolved]
            return []
    except Exception:
        pass

    search_tokens, ext_filter = _parse_query(cleaned_query, file_type=file_type)
    if not search_tokens and not ext_filter:
        return []

    # Determine search roots
    override_roots: list[Path] = []
    if config.FILE_SEARCH_ROOTS:
        for r in config.FILE_SEARCH_ROOTS.split(","):
            r_str = r.strip()
            if r_str:
                p = Path(os.path.expandvars(os.path.expanduser(r_str)))
                if p.exists() and p.is_dir() and not is_sensitive_path(p):
                    override_roots.append(p)

    home = Path.home()
    priority_subdirs = ["Desktop", "Downloads", "Documents", "Pictures", "Videos", "Music"]
    tier1_roots = [
        home / d for d in priority_subdirs
        if (home / d).is_dir() and not is_sensitive_path(home / d)
    ]

    searched_tier1_roots: set[str] = set()
    scored_candidates: list[tuple[int, float, Path]] = []

    def _scan_root(root_dir: Path, timeout_sec: float | None = None, start_time: float | None = None) -> bool:
        """Walk root_dir and collect scored candidates. Returns True if timeout occurred."""
        for current_root, dirs, files in os.walk(str(root_dir), onerror=lambda _: None):
            if timeout_sec and start_time and (time.perf_counter() - start_time) > timeout_sec:
                return True

            # Prune noisy or sensitive directories in-place
            dirs[:] = [
                d for d in dirs
                if d.lower() not in SKIP_DIRS and not is_sensitive_path(Path(current_root) / d)
            ]

            # Check directory names themselves as potential matches
            for d in list(dirs):
                dir_path = Path(current_root) / d
                score = _score_candidate(dir_path, search_tokens, ext_filter)
                if score > 0:
                    try:
                        mtime = dir_path.stat().st_mtime
                    except Exception:
                        mtime = 0.0
                    scored_candidates.append((score, mtime, dir_path))

            # Check files
            for f in files:
                file_path = Path(current_root) / f
                score = _score_candidate(file_path, search_tokens, ext_filter)
                if score > 0:
                    try:
                        mtime = file_path.stat().st_mtime
                    except Exception:
                        mtime = 0.0
                    scored_candidates.append((score, mtime, file_path))

        return False

    # Tier 1 scan: User priority folders or override roots
    scan_targets = override_roots if override_roots else tier1_roots
    for root in scan_targets:
        searched_tier1_roots.add(str(root.resolve()).lower())
        _scan_root(root)

    # Check for strong matches (score >= 60)
    has_strong_match = any(score >= 60 for score, _, _ in scored_candidates)
    if has_strong_match or override_roots:
        # Sort by score descending, then mtime descending
        scored_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        # Deduplicate while preserving order
        seen: set[Path] = set()
        results: list[Path] = []
        for _, _, p in scored_candidates:
            if p not in seen:
                seen.add(p)
                results.append(p)
                if len(results) >= max_results:
                    break
        return results

    # Tier 2 scan: Walk rest of Path.home() with 6.0 second timeout guard
    t0 = time.perf_counter()
    if home.exists() and home.is_dir() and not is_sensitive_path(home):
        for entry in home.iterdir():
            if time.perf_counter() - t0 > 6.0:
                break
            try:
                if not entry.is_dir():
                    # Check top-level home files
                    score = _score_candidate(entry, search_tokens, ext_filter)
                    if score > 0:
                        scored_candidates.append((score, entry.stat().st_mtime, entry))
                    continue

                if entry.name.lower() in SKIP_DIRS or is_sensitive_path(entry):
                    continue

                resolved_entry = str(entry.resolve()).lower()
                if resolved_entry in searched_tier1_roots:
                    continue

                timed_out = _scan_root(entry, timeout_sec=6.0, start_time=t0)
                if timed_out:
                    break
            except Exception:
                continue

    scored_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    seen_paths: set[Path] = set()
    final_results: list[Path] = []
    for _, _, p in scored_candidates:
        if p not in seen_paths:
            seen_paths.add(p)
            final_results.append(p)
            if len(final_results) >= max_results:
                break

    return final_results


def compress_to_zip(target: Path, dest_dir: Path) -> Path:
    """Compress a single file or directory into ``dest_dir / f'{target.name}.zip'``.

    Excludes noisy, sensitive, or system files and uses ZIP_DEFLATED with
    maximum compression (compresslevel=9).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / f"{target.name}.zip"

    if target.is_file():
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            zf.write(target, arcname=target.name)
    elif target.is_dir():
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            for root, dirs, files in os.walk(target):
                dirs[:] = [
                    d for d in dirs
                    if d.lower() not in SKIP_DIRS and not is_sensitive_path(Path(root) / d)
                ]
                for f in files:
                    fp = Path(root) / f
                    if not is_sensitive_path(fp):
                        arcname = fp.relative_to(target.parent)
                        zf.write(fp, arcname=str(arcname))
    else:
        raise FileNotFoundError(f"Target path does not exist: {target}")

    return zip_path


def split_file_chunks(source_zip: Path, chunk_size_mb: float = 45.0) -> list[Path]:
    """Split ``source_zip`` into sequential binary chunks (.part01, .part02, ...) and remove source."""
    if not source_zip.exists():
        raise FileNotFoundError(f"Source file does not exist: {source_zip}")

    chunk_bytes = max(1024, int(chunk_size_mb * 1024 * 1024))
    dest_dir = source_zip.parent
    parts: list[Path] = []
    part_idx = 1

    with open(source_zip, "rb") as f:
        while True:
            chunk = f.read(chunk_bytes)
            if not chunk:
                break
            part_path = dest_dir / f"{source_zip.name}.part{part_idx:02d}"
            part_path.write_bytes(chunk)
            parts.append(part_path)
            part_idx += 1

    source_zip.unlink(missing_ok=True)
    return parts

