"""
plugins/file_search.py
----------------------
Search v2: Two-tier fast file search, security boundary checking, and compression engine.

Helper module for file discovery and smart sharing; does NOT start with `p_`
so Orchestrator ignores it during plugin auto-discovery.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
import time
from typing import Sequence
import zipfile

import config

log = logging.getLogger(__name__)

ALREADY_COMPRESSED_EXTENSIONS: set[str] = {
    ".docx", ".xlsx", ".pptx", ".zip", ".rar", ".7z",
    ".mp4", ".mkv", ".mov", ".jpg", ".jpeg", ".png",
    ".pdf", ".mp3",
}

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

# Directory names pruned in-place
ALWAYS_EXCLUDED_NAMES: set[str] = {
    "windows",
    "program files",
    "program files (x86)",
    "programdata",
    "$recycle.bin",
    "system volume information",
    "recovery",
    "perflogs",
    "appdata",
    "application data",
    "node_modules",
    "venv",
    ".venv",
    "__pycache__",
    "site-packages",
    "ms-playwright",
    "tmp_shares",
}
SKIP_DIRS = ALWAYS_EXCLUDED_NAMES

POSIX_EXCLUDED_PREFIXES: tuple[str, ...] = (
    "/proc",
    "/sys",
    "/dev",
    "/run",
    "/snap",
    "/var/lib",
    "/System",
    "/Library",
    "/private/var",
    "/usr",
    "/bin",
    "/sbin",
    "/etc",
    "/opt/homebrew",
)

# Type words mapped to corresponding extension filters
TYPE_EXTENSIONS: dict[str, list[str]] = {
    "doc": [".doc", ".docx"],
    "docs": [".doc", ".docx"],
    "docx": [".doc", ".docx"],
    "word": [".doc", ".docx"],
    "document": [".doc", ".docx"],
    "pdf": [".pdf"],
    "excel": [".xls", ".xlsx", ".csv"],
    "sheet": [".xls", ".xlsx", ".csv"],
    "spreadsheet": [".xls", ".xlsx", ".csv"],
    "ppt": [".ppt", ".pptx"],
    "slides": [".ppt", ".pptx"],
    "presentation": [".ppt", ".pptx"],
    "movie": [".mp4", ".mkv", ".avi", ".mov", ".webm"],
    "video": [".mp4", ".mkv", ".avi", ".mov", ".webm"],
    "film": [".mp4", ".mkv", ".avi", ".mov", ".webm"],
    "photo": [".jpg", ".jpeg", ".png", ".gif", ".webp"],
    "image": [".jpg", ".jpeg", ".png", ".gif", ".webp"],
    "picture": [".jpg", ".jpeg", ".png", ".gif", ".webp"],
    "pic": [".jpg", ".jpeg", ".png", ".gif", ".webp"],
    "zip": [".zip", ".rar", ".7z"],
    "archive": [".zip", ".rar", ".7z"],
    "song": [".mp3", ".wav", ".flac", ".m4a"],
    "music": [".mp3", ".wav", ".flac", ".m4a"],
    "audio": [".mp3", ".wav", ".flac", ".m4a"],
}

LOCATION_WORDS: dict[str, str] = {
    "desktop": "Desktop",
    "downloads": "Downloads",
    "documents": "Documents",
    "pictures": "Pictures",
    "videos": "Videos",
    "music": "Music",
}

FILLER_TOKENS: set[str] = {
    "send", "sent", "share", "give", "get", "find", "show",
    "me", "my", "the", "a", "an", "file", "files", "folder",
    "please", "pls", "bro", "man", "it", "this", "that", "one",
    "first", "second", "1st", "2nd", "again", "then", "now",
    "in", "at", "for", "to", "from", "with", "on", "of",
}

NUMERAL_MAP: dict[str, str] = {
    "ii": "2", "iii": "3", "iv": "4", "v": "5",
    "two": "2", "three": "3", "four": "4", "five": "5",
}


@dataclass(frozen=True)
class FileMatch:
    path: Path
    size: int
    mtime: float
    approximate: bool = False


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


def is_reparse_point(path: Path | str) -> bool:
    """Return True if path is a symlink or Windows junction / reparse point."""
    try:
        p = Path(path)
        if p.is_symlink():
            return True
        st = os.lstat(str(p))
        attrs = getattr(st, "st_file_attributes", 0)
        reparse_attr = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x0400)
        if bool(attrs & reparse_attr):
            return True
    except Exception:
        pass
    return False


def is_excluded_path(path: Path | str) -> bool:
    """Return True if path matches system exclusions, dot-directories, or FILE_SEARCH_EXCLUDE."""
    try:
        p = Path(path)
    except Exception:
        return True

    for part in p.parts:
        if part.startswith(".") and part not in (".", ".."):
            return True

    is_under_temp = False
    try:
        tmp_dir = Path(tempfile.gettempdir()).resolve()
        p_res = p.resolve()
        if p_res == tmp_dir or tmp_dir in p_res.parents:
            is_under_temp = True
    except Exception:
        pass

    parts_lower = [part.lower() for part in p.parts]
    p_str = str(p).replace("\\", "/")
    p_lower = p_str.lower()

    for part in parts_lower:
        if is_under_temp and part in ("appdata", "application data", "local", "roaming", "locallow"):
            continue
        if part in ALWAYS_EXCLUDED_NAMES:
            return True
        if not is_under_temp and "appdata" in part:
            return True

    windir = os.environ.get("WINDIR") or os.environ.get("SystemRoot")
    if windir:
        try:
            windir_p = Path(windir).resolve()
            p_res = p.resolve()
            if p_res == windir_p or windir_p in p_res.parents:
                return True
        except Exception:
            pass

    for prefix in POSIX_EXCLUDED_PREFIXES:
        pref_clean = prefix.rstrip("/")
        if p_str == pref_clean or p_str.startswith(f"{pref_clean}/") or p_lower == pref_clean.lower() or p_lower.startswith(f"{pref_clean.lower()}/"):
            return True

    try:
        data_dir_cfg = getattr(config, "DATA_DIR", "data")
        if data_dir_cfg:
            data_dir = Path(data_dir_cfg).resolve()
            p_res = p.resolve()
            if p_res == data_dir or data_dir in p_res.parents:
                return True
    except Exception:
        pass

    custom_exclude = getattr(config, "FILE_SEARCH_EXCLUDE", "")
    if custom_exclude:
        for ex in custom_exclude.split(","):
            ex_str = ex.strip()
            if not ex_str:
                continue
            if "/" in ex_str or "\\" in ex_str:
                try:
                    ex_p = Path(os.path.expandvars(os.path.expanduser(ex_str))).resolve()
                    p_res = p.resolve()
                    if p_res == ex_p or ex_p in p_res.parents:
                        return True
                except Exception:
                    pass
            else:
                if ex_str.lower() in parts_lower or ex_str.lower() == p.name.lower():
                    return True

    return False


def roots_for_mode(mode: str | None = None) -> list[Path]:
    """Return root search paths based on mode ('home', 'all_drives', or 'custom')."""
    active_mode = (mode or getattr(config, "FILE_SEARCH_MODE", "home")).strip().lower()

    if active_mode == "custom":
        custom_str = getattr(config, "FILE_SEARCH_ROOTS", "")
        roots: list[Path] = []
        for r in custom_str.split(","):
            r_str = r.strip()
            if r_str:
                p = Path(os.path.expandvars(os.path.expanduser(r_str)))
                if p.exists() and p.is_dir() and not is_reparse_point(p):
                    try:
                        resolved = p.resolve()
                        if not is_sensitive_path(resolved) and not is_excluded_path(resolved) and resolved not in roots:
                            roots.append(resolved)
                    except Exception:
                        pass
        return roots or [Path.home()]

    if active_mode == "all_drives":
        roots: list[Path] = [Path.home()]
        allow_removable = getattr(config, "FILE_INDEX_REMOVABLE", False)
        try:
            import psutil
            for part in psutil.disk_partitions(all=False):
                opts = [o.strip() for o in part.opts.lower().split(",")]
                fstype = part.fstype.lower()
                if "cdrom" in opts or fstype in ("iso9660", "udf"):
                    continue
                if "network" in opts or fstype in ("nfs", "smbfs", "cifs", "afs"):
                    continue
                if "removable" in opts and not allow_removable:
                    continue
                mount = Path(part.mountpoint)
                if mount.exists() and mount.is_dir():
                    try:
                        resolved = mount.resolve()
                        if (
                            resolved not in roots
                            and mount not in roots
                            and not is_sensitive_path(resolved)
                            and not is_excluded_path(resolved)
                        ):
                            roots.append(mount)
                    except Exception:
                        pass
        except Exception as exc:
            log.warning("Failed enumerating disk partitions for all_drives: %s", exc)
        return roots

    # default: "home"
    return [Path.home()]


def _should_skip_dir(name: str, full_path: Path | None = None) -> bool:
    """Return True if directory should be pruned from walk."""
    if name.startswith(".") and name not in (".", ".."):
        return True
    name_l = name.lower()
    is_under_temp = False
    if full_path is not None:
        try:
            tmp_dir = Path(tempfile.gettempdir()).resolve()
            p_res = full_path.resolve()
            if p_res == tmp_dir or tmp_dir in p_res.parents:
                is_under_temp = True
        except Exception:
            pass

    if name_l in ALWAYS_EXCLUDED_NAMES or name_l in SKIP_DIRS:
        if not (is_under_temp and name_l in ("appdata", "application data", "local", "roaming", "locallow")):
            return True
    if "appdata" in name_l and not is_under_temp:
        return True
    if full_path is not None:
        if is_reparse_point(full_path):
            return True
        if is_excluded_path(full_path):
            return True
        if is_sensitive_path(full_path):
            return True
    return False


def split_stem_tokens(stem: str) -> list[str]:
    """Split a filename stem into normalized lowercase word tokens on non-alphanumerics & camelCase."""
    # Split camelCase
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", stem)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    raw_tokens = [t for t in re.split(r"[^\w]+|_", s, flags=re.UNICODE) if t]
    tokens: list[str] = []
    for t in raw_tokens:
        tok = t.lower().strip()
        if not tok:
            continue
        tok = NUMERAL_MAP.get(tok, tok)
        tokens.append(tok)
    return tokens


def parse_query(q: str) -> tuple[list[str], list[str], Path | None]:
    """Parse query into (name_tokens, ext_filter, root_hint).

    - Strips quotes and punctuation, treating / and \\ as separators.
    - Drops filler tokens.
    - Converts type words or explicit .ext into ext_filter list.
    - Identifies location words (e.g. desktop, downloads) as root_hint.
    - Normalizes numerals (ii->2, two->2).
    """
    raw = q.strip().strip('"\'')
    if not raw:
        return [], [], None

    root_hint: Path | None = None
    ext_filters: list[str] = []

    # First inspect raw string for path fragments containing location words (e.g. "krith\\downloads")
    # Split by whitespace into chunks
    chunks = raw.split()
    remaining_words: list[str] = []

    for chunk in chunks:
        # Check if chunk contains slashes/backslashes
        if "\\" in chunk or "/" in chunk:
            segments = [s.strip().lower() for s in re.split(r"[\\/]+", chunk) if s.strip()]
            loc_found = False
            for seg in segments:
                if seg in LOCATION_WORDS:
                    target_folder = LOCATION_WORDS[seg]
                    root_hint = _resolve_location_folder(target_folder)
                    loc_found = True
                    break
            if loc_found:
                # Discard path prefix/fragment from name tokens
                continue

        remaining_words.append(chunk)

    normalized_text = " ".join(remaining_words).lower()
    # Replace punctuation except dots and alphanumerics
    cleaned = re.sub(r"[,;:\'\"`\(\)\[\]\{\}\?!#\*\+=]+", " ", normalized_text)
    tokens = [t.strip() for t in re.split(r"\s+", cleaned) if t.strip()]

    name_tokens: list[str] = []

    for token in tokens:
        # Check for explicit extension: e.g. ".pdf" or "name.pdf"
        if "." in token:
            parts = token.rsplit(".", 1)
            stem_part, ext_part = parts[0], f".{parts[1]}"
            if len(parts[1]) >= 1 and parts[1].isalnum():
                if ext_part not in ext_filters:
                    ext_filters.append(ext_part)
                if stem_part:
                    for sw in split_stem_tokens(stem_part):
                        if sw not in FILLER_TOKENS:
                            name_tokens.append(sw)
                continue

        # Check for location words
        if token in LOCATION_WORDS:
            if root_hint is None:
                root_hint = _resolve_location_folder(LOCATION_WORDS[token])
            continue

        # Check for type words
        if token in TYPE_EXTENSIONS:
            for ext in TYPE_EXTENSIONS[token]:
                if ext not in ext_filters:
                    ext_filters.append(ext)
            continue

        # Check for filler tokens
        if token in FILLER_TOKENS:
            continue

        # Regular name token - normalize numerals and split camelCase / underscores
        for sw in split_stem_tokens(token):
            if sw not in FILLER_TOKENS:
                name_tokens.append(sw)

    return name_tokens, ext_filters, root_hint


def _resolve_location_folder(folder_name: str) -> Path:
    """Resolve location folder checking config.FILE_SEARCH_ROOTS override first."""
    if config.FILE_SEARCH_ROOTS:
        for r in config.FILE_SEARCH_ROOTS.split(","):
            r_str = r.strip()
            if r_str:
                p = Path(os.path.expandvars(os.path.expanduser(r_str)))
                if (p / folder_name).is_dir():
                    return (p / folder_name).resolve()
                if p.name.lower() == folder_name.lower() and p.is_dir():
                    return p.resolve()

    return (Path.home() / folder_name).resolve()


def _match_candidate(
    candidate: Path,
    name_tokens: list[str],
    ext_filters: list[str],
) -> tuple[int, bool] | None:
    """Evaluate candidate file.

    Returns tuple (tier_score, approximate) or None if no match.
    Tier A returns (100, False), Tier B returns (60, True).
    """
    if is_sensitive_path(candidate):
        return None
    if candidate.name.startswith("."):
        return None

    if ext_filters:
        if candidate.suffix.lower() not in ext_filters:
            return None

    if not name_tokens:
        # Only extension or location filter was specified
        return (100, False)

    cand_tokens = split_stem_tokens(candidate.stem)
    if not cand_tokens:
        return None

    # Check exact stem match
    if cand_tokens == name_tokens:
        return (110, False)

    # Match each query token against candidate tokens
    matched_q_indices: list[int] = []
    matched_c_indices: list[int] = []

    for q_idx, q_tok in enumerate(name_tokens):
        for c_idx, c_tok in enumerate(cand_tokens):
            if c_idx in matched_c_indices:
                continue
            # Equal match or candidate token starts with query token (length >= 3)
            if c_tok == q_tok or (len(q_tok) >= 3 and c_tok.startswith(q_tok)):
                matched_q_indices.append(q_idx)
                matched_c_indices.append(c_idx)
                break

    matched_count = len(matched_q_indices)
    total_q = len(name_tokens)

    # Tier A: All query tokens matched
    if matched_count == total_q:
        # Check token order
        is_in_order = matched_c_indices == sorted(matched_c_indices)
        score = 105 if is_in_order else 100
        return (score, False)

    # Tier B: at least ceil(60%) of tokens matched, query >= 2 tokens, and matched >= 2
    threshold = math.ceil(0.6 * total_q)
    if total_q >= 2 and matched_count >= threshold and matched_count >= 2:
        return (60, True)

    return None


def get_search_roots() -> list[Path]:
    """Return configured or priority search roots based on mode."""
    mode = getattr(config, "FILE_SEARCH_MODE", "home").strip().lower()
    if mode == "home":
        if config.FILE_SEARCH_ROOTS:
            override_roots: list[Path] = []
            for r in config.FILE_SEARCH_ROOTS.split(","):
                r_str = r.strip()
                if r_str:
                    p = Path(os.path.expandvars(os.path.expanduser(r_str)))
                    if p.exists() and p.is_dir() and not is_sensitive_path(p) and not is_excluded_path(p):
                        override_roots.append(p.resolve())
            if override_roots:
                return override_roots

        home = Path.home()
        priority_subdirs = ["Desktop", "Downloads", "Documents", "Pictures", "Videos", "Music"]
        roots: list[Path] = []
        for d in priority_subdirs:
            p = home / d
            if p.exists() and p.is_dir() and not is_sensitive_path(p) and not is_excluded_path(p):
                roots.append(p)
        return roots or [home]

    return roots_for_mode(mode)


def search_files(query: str, file_type: str = "", max_results: int = 5) -> list[FileMatch]:
    """Find files or folders matching ``query`` returning list of FileMatch objects."""
    cleaned_query = query.strip().strip('"\'')
    if not cleaned_query:
        return []

    # 1. Direct path resolution (supports / and \\)
    try:
        norm_path = cleaned_query.replace("/", os.sep).replace("\\", os.sep)
        expanded = os.path.expandvars(os.path.expanduser(norm_path))
        candidate = Path(expanded)
        if candidate.exists():
            resolved = candidate.resolve()
            if not is_sensitive_path(resolved) and not is_excluded_path(resolved):
                st = resolved.stat()
                return [FileMatch(path=resolved, size=st.st_size, mtime=st.st_mtime, approximate=False)]
            return []
    except Exception:
        pass

    name_tokens, ext_filters, root_hint = parse_query(cleaned_query)

    if file_type:
        clean_ext = file_type.strip().lower()
        clean_ext = clean_ext if clean_ext.startswith(".") else f".{clean_ext}"
        if clean_ext not in ext_filters:
            ext_filters.append(clean_ext)

    if not name_tokens and not ext_filters and not root_hint:
        return []

    # Determine search roots
    mode = getattr(config, "FILE_SEARCH_MODE", "home").strip().lower()
    override_roots: list[Path] = []
    if mode == "custom" or (mode == "home" and config.FILE_SEARCH_ROOTS):
        for r in (config.FILE_SEARCH_ROOTS or "").split(","):
            r_str = r.strip()
            if r_str:
                p = Path(os.path.expandvars(os.path.expanduser(r_str)))
                if p.exists() and p.is_dir() and not is_sensitive_path(p) and not is_excluded_path(p):
                    override_roots.append(p.resolve())
    elif mode == "all_drives":
        override_roots = roots_for_mode("all_drives")

    home = Path.home()
    priority_subdirs = ["Desktop", "Downloads", "Documents", "Pictures", "Videos", "Music"]

    scored_matches: list[tuple[int, float, FileMatch]] = []

    def _scan_root(root_dir: Path, timeout_sec: float | None = None, start_time: float | None = None) -> bool:
        """Walk root_dir and collect matches. Returns True if timed out."""
        visited_dirs: set[Path] = set()
        for current_root, dirs, files in os.walk(str(root_dir), onerror=lambda _: None):
            if timeout_sec and start_time and (time.perf_counter() - start_time) > timeout_sec:
                return True

            curr_p = Path(current_root)
            try:
                curr_res = curr_p.resolve()
                if curr_res in visited_dirs:
                    dirs[:] = []
                    continue
                visited_dirs.add(curr_res)
            except Exception:
                pass

            # Prune noisy, hidden, or sensitive directories in-place
            dirs[:] = [
                d for d in dirs
                if not _should_skip_dir(d, curr_p / d)
            ]

            # Check files
            for f in files:
                if f.startswith("."):
                    continue
                file_path = curr_p / f
                if is_excluded_path(file_path) or is_sensitive_path(file_path):
                    continue
                res = _match_candidate(file_path, name_tokens, ext_filters)
                if res is not None:
                    score, approx = res
                    try:
                        st = file_path.stat()
                        scored_matches.append((score, st.st_mtime, FileMatch(path=file_path, size=st.st_size, mtime=st.st_mtime, approximate=approx)))
                    except Exception:
                        pass

        return False

    # If root_hint is set, search only that folder
    if root_hint is not None and root_hint.exists() and root_hint.is_dir():
        _scan_root(root_hint)
    elif override_roots:
        for r in override_roots:
            _scan_root(r)
    else:
        # Tier 1 roots: Standard user folders + OneDrive mirror folders
        tier1_roots: list[Path] = []
        for d in priority_subdirs:
            p = home / d
            if p.is_dir() and not is_sensitive_path(p):
                tier1_roots.append(p.resolve())

        # OneDrive folders
        try:
            for p in home.iterdir():
                if p.is_dir() and p.name.lower().startswith("onedrive") and not is_sensitive_path(p):
                    for d in priority_subdirs:
                        od_sub = p / d
                        if od_sub.is_dir() and not is_sensitive_path(od_sub):
                            tier1_roots.append(od_sub.resolve())
        except Exception:
            pass

        searched_tier1_paths: set[str] = set()
        for root in tier1_roots:
            searched_tier1_paths.add(str(root).lower())
            _scan_root(root)

        # If any Tier A match found, return top results immediately
        has_tier_a = any(score >= 100 for score, _, _ in scored_matches)
        if not has_tier_a:
            # Tier 2 scan: Rest of Path.home() with 6.0s timeout
            t0 = time.perf_counter()
            if home.exists() and home.is_dir() and not is_sensitive_path(home):
                try:
                    for entry in home.iterdir():
                        if time.perf_counter() - t0 > 6.0:
                            break
                        if not entry.is_dir():
                            if not entry.name.startswith("."):
                                res = _match_candidate(entry, name_tokens, ext_filters)
                                if res is not None:
                                    score, approx = res
                                    st = entry.stat()
                                    scored_matches.append((score, st.st_mtime, FileMatch(path=entry, size=st.st_size, mtime=st.st_mtime, approximate=approx)))
                            continue

                        if _should_skip_dir(entry.name, entry) or is_sensitive_path(entry) or is_excluded_path(entry):
                            continue

                        if str(entry.resolve()).lower() in searched_tier1_paths:
                            continue

                        timed_out = _scan_root(entry, timeout_sec=6.0, start_time=t0)
                        if timed_out:
                            break
                except Exception:
                    pass

    # Sort matches by score descending, then mtime descending
    scored_matches.sort(key=lambda item: (item[0], item[1]), reverse=True)

    seen: set[Path] = set()
    results: list[FileMatch] = []
    for _, _, match in scored_matches:
        if match.path not in seen:
            seen.add(match.path)
            results.append(match)
            if len(results) >= max_results:
                break

    return results


def find_files(query: str, file_type: str = "", max_results: int = 5) -> list[Path]:
    """Backward-compatible file search returning list of Path objects."""
    matches = search_files(query, file_type=file_type, max_results=max_results)
    return [m.path for m in matches]


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
                    if not _should_skip_dir(d, Path(root) / d) and not is_sensitive_path(Path(root) / d)
                ]
                for f in files:
                    if f.startswith("."):
                        continue
                    fp = Path(root) / f
                    if not is_sensitive_path(fp):
                        arcname = fp.relative_to(target.parent)
                        zf.write(fp, arcname=str(arcname))
    else:
        raise FileNotFoundError(f"Target path does not exist: {target}")

    return zip_path


def split_file_chunks(
    source_file: Path,
    chunk_size_mb: float = 45.0,
    dest_dir: Path | None = None,
    remove_source: bool | None = None,
) -> list[Path]:
    """Split ``source_file`` into sequential binary chunks (.part01, .part02, ...)."""
    if not source_file.exists():
        raise FileNotFoundError(f"Source file does not exist: {source_file}")

    chunk_bytes = max(1024, int(chunk_size_mb * 1024 * 1024))
    target_dir = dest_dir or source_file.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    parts: list[Path] = []
    part_idx = 1

    with open(source_file, "rb") as f:
        while True:
            chunk = f.read(chunk_bytes)
            if not chunk:
                break
            part_path = target_dir / f"{source_file.name}.part{part_idx:02d}"
            part_path.write_bytes(chunk)
            parts.append(part_path)
            part_idx += 1

    if remove_source is None:
        remove_source = source_file.suffix.lower() == ".zip"
    if remove_source:
        source_file.unlink(missing_ok=True)
    return parts


def cleanup_stale_tmp_shares(max_age_s: float = 86400.0) -> int:
    """Remove leftover files in DATA_DIR / tmp_shares older than max_age_s (default 24h)."""
    tmp_shares_dir = Path(config.DATA_DIR) / "tmp_shares"
    if not tmp_shares_dir.exists():
        return 0
    now = time.time()
    removed = 0
    for item in tmp_shares_dir.iterdir():
        try:
            if (now - item.stat().st_mtime) > max_age_s:
                if item.is_file():
                    item.unlink(missing_ok=True)
                    removed += 1
                elif item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                    removed += 1
        except Exception as exc:
            log.warning("Failed to clean up stale tmp_share %s: %s", item, exc)
    return removed

