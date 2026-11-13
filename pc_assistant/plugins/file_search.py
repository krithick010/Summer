"""
plugins/file_search.py
----------------------
Search v2: Two-tier fast file search, security boundary checking, and compression engine.

Helper module for file discovery and smart sharing; does NOT start with `p_`
so Orchestrator ignores it during plugin auto-discovery.
"""

from __future__ import annotations

from dataclasses import dataclass
import difflib
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

SENSITIVE_EXTS: set[str] = {
    ".key", ".pem", ".pfx", ".p12", ".ovpn", ".kdbx", ".keystore",
}

SENSITIVE_FILES_EXACT: set[str] = {
    "login data",
    "cookies",
    "web data",
    "local state",
    "key4.db",
    "logins.json",
    "ntuser.dat",
    "sam",
    "system",
    "security",
    "wallet.dat",
    ".git-credentials",
    ".netrc",
    ".npmrc",
    ".pypirc",
    "file_index.db",
    "gmail_token.json",
    "gmail_oauth.json",
    "whatsapp_session",
    "history.db",
    "shadow",
    "sudoers",
}

SENSITIVE_DIR_SEGMENTS: set[str] = {
    ".ssh",
    ".gnupg",
    ".aws",
    ".azure",
    ".kube",
    ".docker",
    "bravesoftware",
    "credentials",
    "memory_db",
}

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


@dataclass
class ParsedQuery:
    tokens: list[str]
    ext_filter: list[str]
    want_dir: bool
    location_terms: list[str]
    corrections: dict[str, str]
    raw: str

    def __iter__(self):
        return iter((self.tokens, self.ext_filter, self.location_terms))


@dataclass(frozen=True)
class FileMatch:
    path: Path
    size: int
    mtime: float
    approximate: bool = False


DENYLIST_KEYWORDS: tuple[str, ...] = (
    "password", "passwords", "passwd", "shadow", ".env", "id_rsa", "id_ed25519",
    "wallet.dat", "secret_key", "keystore",
)


def check_denylist_rejection(query: str) -> str | None:
    """Return denial message if query matches any denylist keyword, else None."""
    q_lower = query.lower()
    for kw in DENYLIST_KEYWORDS:
        if kw in q_lower:
            return f"I cannot share protected or sensitive files ({kw})."
    return None


def format_size(size_bytes: int) -> str:
    """Format bytes into human-readable B, KB, MB, or GB string.

    < 1024 -> "{n} B"
    < 1 MB -> "{n/1024:.1f} KB"
    < 1 GB -> "{n/1024/1024:.1f} MB"
    >= 1 GB -> "{n/1024/1024/1024:.2f} GB"
    Never show "0.0 MB" for a small file.
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    if size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def format_miss_reply(clean_query: str) -> str:
    """Format honest plain-text message when file search yields 0 results."""
    from plugins.file_index import file_indexer  # noqa: PLC0415
    status = file_indexer.get_status()
    count = status.get("total_indexed", 0)
    roots = roots_for_mode()
    roots_str = ", ".join(r.name or str(r) for r in roots) if roots else "configured roots"

    vocab = file_indexer.get_vocabulary()
    closest = difflib.get_close_matches(clean_query.lower(), vocab, n=3, cutoff=0.4) if vocab else []
    if closest:
        return (
            f"No matching files found for '{clean_query}' (searched {count:,} indexed files across {roots_str}).\n"
            f"Closest names: {', '.join(closest)}."
        )
    return f"No matching files found for '{clean_query}' (searched {count:,} indexed files across {roots_str})."


def is_sensitive_path(path: Path | str) -> bool:
    """Return True if ``path`` matches sensitive patterns, credentials, or keys."""
    try:
        p = Path(path)
    except Exception:
        return True

    name_lower = p.name.lower()
    suffix_lower = p.suffix.lower()
    p_str = str(p).replace("\\", "/").lower()
    parts_lower = [part.lower() for part in p.parts]

    # 1. Private keys & certificates by extension
    if suffix_lower in SENSITIVE_EXTS:
        return True

    # 2. Key prefixes and crypto wallets
    if (
        name_lower.startswith("id_rsa")
        or name_lower.startswith("id_ed25519")
        or "keystore" in name_lower
        or name_lower == "wallet.dat"
    ):
        return True

    # 3. Exact sensitive file names
    if name_lower in SENSITIVE_FILES_EXACT:
        return True

    # 4. Summer's own secrets: .env, .env.*, *.env
    if name_lower == ".env" or name_lower.startswith(".env.") or name_lower.endswith(".env") or suffix_lower == ".env":
        return True

    # 5. Gmail token patterns
    if "gmail_token" in name_lower or "gmail_oauth" in name_lower:
        return True

    # 6. Sensitive directory segments
    for part in parts_lower:
        if part in SENSITIVE_DIR_SEGMENTS:
            return True
        if part in SENSITIVE_FILES_EXACT:
            return True
        if part == ".env" or part.startswith(".env."):
            return True
        if part.startswith("id_rsa") or part.startswith("id_ed25519"):
            return True
        if any(part.endswith(ext) for ext in SENSITIVE_EXTS):
            return True

    # 7. Specific credential dirs: .config/gcloud
    if "gcloud" in parts_lower:
        return True

    # 8. Browser folders: Chrome/User Data, Edge/User Data, Firefox/Profiles, BraveSoftware
    if "user data" in parts_lower:
        if any(b in p_str for b in ("chrome", "edge", "google", "microsoft")):
            return True
    if "profiles" in parts_lower:
        if any(b in p_str for b in ("firefox", "mozilla")):
            return True
    if "bravesoftware" in p_str or "brave-browser" in p_str:
        return True

    # 9. OS credentials: /etc/shadow, /etc/sudoers
    if "/etc/shadow" in p_str or "/etc/sudoers" in p_str or p_str.endswith("/shadow") or p_str.endswith("/sudoers"):
        return True

    # 10. Summer's data directory & memory DB
    try:
        data_dir_cfg = getattr(config, "DATA_DIR", "data")
        if data_dir_cfg:
            data_dir = Path(data_dir_cfg).resolve()
            p_res = p.resolve()
            if p_res == data_dir or data_dir in p_res.parents:
                if name_lower in ("history.db", "memory_db", "file_index.db", "file_audit.log") or "memory_db" in parts_lower:
                    return True
    except Exception:
        pass

    return False


SOFT_SENSITIVE_WORDS = ("password", "passwd", "secret", "token", "private")
USUAL_FOLDER_NAMES = {"desktop", "documents", "downloads", "pictures", "videos", "music"}


def is_soft_sensitive(path: Path | str) -> bool:
    """Return True if filename contains soft-sensitive indicator words."""
    p = Path(path)
    name_l = p.name.lower()
    return any(w in name_l for w in SOFT_SENSITIVE_WORDS)


def is_outside_usual_folders(path: Path | str) -> bool:
    """Return True if path is not within usual user folders (Desktop, Documents, Downloads, Pictures, Videos, Music)."""
    try:
        p = Path(path).resolve()
    except Exception:
        p = Path(path)

    for parent in [p] + list(p.parents):
        if parent.name.lower() in USUAL_FOLDER_NAMES:
            return False
    return True


_SHARE_RATE_LIMITS: dict[int, list[float]] = {}


def check_and_increment_share_limit(user_id: int | None = None) -> tuple[bool, str]:
    """Check rolling 3600-second window for user_id against config.FILE_MAX_SHARES_PER_HOUR.

    Returns (allowed: bool, message: str).
    """
    if not user_id:
        uid = 0
    else:
        try:
            uid = int(user_id)
        except (ValueError, TypeError):
            uid = 0

    max_shares = getattr(config, "FILE_MAX_SHARES_PER_HOUR", 20)
    now = time.time()
    cutoff = now - 3600.0

    timestamps = _SHARE_RATE_LIMITS.setdefault(uid, [])
    # Prune timestamps older than 1 hour
    _SHARE_RATE_LIMITS[uid] = [t for t in timestamps if t > cutoff]

    if len(_SHARE_RATE_LIMITS[uid]) >= max_shares:
        return False, f"Rate limit reached ({max_shares} files/hr)"

    _SHARE_RATE_LIMITS[uid].append(now)
    return True, ""


def reset_share_rate_limits() -> None:
    """Clear in-memory share rate limits (useful for test isolation)."""
    _SHARE_RATE_LIMITS.clear()


def log_file_audit(
    user_id: int | str,
    action: str,
    path: Path | str,
    size: float | int | str,
    route: str,
    status: str,
) -> None:
    """Append one line to DATA_DIR/file_audit.log."""
    if not getattr(config, "FILE_AUDIT_LOG", True):
        return
    try:
        data_dir = Path(getattr(config, "DATA_DIR", "data"))
        data_dir.mkdir(parents=True, exist_ok=True)
        log_file = data_dir / "file_audit.log"
        iso_ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        line = f"{iso_ts} | user_id={user_id} | action={action} | path={path} | size={size} | route={route} | status={status}\n"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception as exc:
        log.warning("Failed writing to file_audit.log: %s", exc)


def is_reparse_point(path: Path | str) -> bool:
    """Return True if path is a symlink or Windows junction / mount point.

    Reparse points that are cloud sync placeholders (e.g. OneDrive) are NOT skipped.
    """
    try:
        p = Path(path)
        st = os.lstat(str(p))
        attrs = getattr(st, "st_file_attributes", None)
        if attrs is not None:
            reparse_attr = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x0400)
            if bool(attrs & reparse_attr):
                tag = getattr(st, "st_reparse_tag", 0)
                symlink_tag = getattr(stat, "IO_REPARSE_TAG_SYMLINK", 0xA000000C)
                mount_tag = getattr(stat, "IO_REPARSE_TAG_MOUNT_POINT", 0xA0000003)
                if tag in (symlink_tag, mount_tag):
                    return True
                if tag == 0:
                    return p.is_symlink()
                return False
        return p.is_symlink()
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


KNOWN_USER_FOLDERS: dict[str, str] = {
    "downloads": "Downloads",
    "desktop": "Desktop",
    "documents": "Documents",
    "pictures": "Pictures",
    "videos": "Videos",
    "music": "Music",
}


def resolve_user_path(raw_path: str) -> Path | None:
    """Resolve user-supplied or LLM-hallucinated paths into an existing local Path.

    Handles:
    - Quoted paths, ~, environment variables
    - Known folder aliases ("Downloads", "desktop", "friend/downloads", "user/documents")
    - Hallucinated username prefixes (e.g. "C:\\Users\\friend\\Downloads\\report.txt")
    - Relative paths under known folders ("Downloads/tree.txt")
    Returns resolved Path if it exists and is not sensitive, else None.
    """
    if not raw_path or not str(raw_path).strip():
        return None

    raw = str(raw_path).strip().strip("'\"").strip()
    if not raw:
        return None

    expanded = os.path.expandvars(os.path.expanduser(raw))
    norm = expanded.replace("\\", "/")

    home = Path.home()

    def _find_onedrive_folder(folder_name: str, tail: Path | None = None) -> Path | None:
        cands = [home / "OneDrive" / folder_name]
        try:
            for d in home.iterdir():
                if d.is_dir() and d.name.lower().startswith("onedrive"):
                    cands.append(d / folder_name)
        except Exception:
            pass
        for c in cands:
            cand = c / tail if tail else c
            if cand.exists() and not is_sensitive_path(cand):
                return cand.resolve()
        return None

    # 1. Directly a known folder alias or ends with /<folder_name>
    clean_tail = norm.strip("/").lower()
    parts = [p for p in clean_tail.split("/") if p]
    if parts:
        last_seg = parts[-1]
        if last_seg in KNOWN_USER_FOLDERS:
            folder_name = KNOWN_USER_FOLDERS[last_seg]
            cand = home / folder_name
            if cand.exists() and not is_sensitive_path(cand):
                return cand.resolve()
            od_cand = _find_onedrive_folder(folder_name)
            if od_cand is not None:
                return od_cand

    # 2. Any segment is a known user folder with a relative tail
    if parts:
        for idx, seg in enumerate(parts):
            if seg in KNOWN_USER_FOLDERS:
                folder_name = KNOWN_USER_FOLDERS[seg]
                rel_tail_parts = parts[idx + 1:]
                if rel_tail_parts:
                    orig_parts = [p for p in norm.strip("/").split("/") if p]
                    tail = Path(*orig_parts[idx + 1:])
                    cand = home / folder_name / tail
                    if cand.exists() and not is_sensitive_path(cand):
                        return cand.resolve()
                    od_cand = _find_onedrive_folder(folder_name, tail)
                    if od_cand is not None:
                        return od_cand

    # 3. Rewrite hallucinated Users\<wrong_user>\... to Path.home()
    m = re.search(r"(?:^|[/\\])[uU]sers[/\\][^/\\]+[/\\](?P<rest>.+)$", norm)
    if m:
        rest = m.group("rest")
        cand = home / Path(rest)
        if cand.exists() and not is_sensitive_path(cand):
            return cand.resolve()
        rest_parts = [p for p in rest.split("/") if p]
        if rest_parts and rest_parts[0].lower() in KNOWN_USER_FOLDERS:
            folder_name = KNOWN_USER_FOLDERS[rest_parts[0].lower()]
            tail = Path(*rest_parts[1:]) if len(rest_parts) > 1 else None
            od_cand = _find_onedrive_folder(folder_name, tail)
            if od_cand is not None:
                return od_cand

    # 4. Direct path check
    try:
        p = Path(expanded)
        if p.exists() and not is_sensitive_path(p):
            return p.resolve()
        p_home = home / Path(norm.lstrip("/"))
        if p_home.exists() and not is_sensitive_path(p_home):
            return p_home.resolve()
    except Exception:
        pass

    return None


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


CANONICAL_VOCAB: list[str] = [
    "folder", "directory", "dir",
    "send", "sent", "share", "give", "get", "find", "show", "search",
    "file", "files", "document", "image", "picture", "photo", "video", "movie",
    "pdf", "word", "docx", "doc", "excel", "sheet", "spreadsheet", "xlsx", "xls",
    "powerpoint", "slides", "presentation", "pptx", "ppt", "archive", "zip",
    "music", "audio", "song", "text", "txt",
]

COMMON_EXTS: list[str] = [
    "txt", "pdf", "docx", "doc", "xlsx", "xls", "pptx", "ppt",
    "png", "jpg", "jpeg", "zip", "mp4", "mp3", "csv", "json", "py",
]

STANDALONE_ALIASES: set[str] = {
    "desktop", "downloads", "documents", "pictures", "videos", "music", "onedrive",
}

DIR_WORDS: set[str] = {"folder", "directory", "dir"}


def _path_matches_location(path: Path, location_terms: list[str]) -> bool:
    """Return True if path's folder components match location_terms IN ORDER (exact or word-prefix)."""
    if not location_terms:
        return True

    parent_folders = [p.name.lower() for p in reversed(path.parents) if p.name]
    if path.is_dir() and path.name:
        parent_folders.append(path.name.lower())

    term_idx = 0
    total_terms = len(location_terms)

    for folder in parent_folders:
        folder_words = re.split(r"[^\w]+", folder)
        while term_idx < total_terms:
            t = location_terms[term_idx].lower()
            matched = False
            if folder == t or folder.startswith(t):
                matched = True
            else:
                for fw in folder_words:
                    if fw == t or (len(t) >= 3 and fw.startswith(t)):
                        matched = True
                        break
            if matched:
                term_idx += 1
            else:
                break
        if term_idx >= total_terms:
            return True

    return term_idx >= total_terms


def parse_query(q: str, vocab: set[str] | None = None) -> ParsedQuery:
    """Parse raw query into ParsedQuery with location phrases, directory flags, and typo corrections."""
    raw_clean = q.strip().strip("'\"").strip()
    if not raw_clean:
        return ParsedQuery(tokens=[], ext_filter=[], want_dir=False, location_terms=[], corrections={}, raw=q)

    corrections: dict[str, str] = {}
    ext_filters: list[str] = []
    want_dir = False

    # 1. Location phrase extraction: "in / inside / under / from / within / at / on <folder>"
    loc_match = re.search(r"\b(in|inside|under|from|within|at|on)\s+(.+)$", raw_clean, flags=re.IGNORECASE)
    if loc_match:
        loc_part = loc_match.group(2).strip()
        query_text = raw_clean[:loc_match.start()].strip()
        loc_terms = [t.lower().strip() for t in re.split(r"[\s/\\]+", loc_part) if t.strip()]
        loc_terms = [re.sub(r"^[^\w]+|[^\w]+$", "", t) for t in loc_terms if re.sub(r"^[^\w]+|[^\w]+$", "", t)]
    else:
        query_text = raw_clean
        loc_terms = []

    # 2. Extract standalone location aliases if there are other tokens in query_text
    chunks = query_text.split()
    if len(chunks) > 1:
        kept_chunks = []
        for ch in chunks:
            ch_clean = ch.lower().strip(",;:")
            if ch_clean in STANDALONE_ALIASES and "." not in ch:
                if ch_clean not in loc_terms:
                    loc_terms.append(ch_clean)
            else:
                kept_chunks.append(ch)
        query_text = " ".join(kept_chunks)

    # 3. Tokenize on spaces, commas, semicolons, underscores, etc.
    raw_tokens = [t.strip() for t in re.split(r"[\s,;:\"'()\[\]{}]+", query_text) if t.strip()]

    # Load index vocabulary if not provided
    if vocab is None:
        try:
            from plugins.file_index import file_indexer  # noqa: PLC0415
            vocab = file_indexer.get_vocabulary()
        except Exception:
            vocab = set()

    name_tokens: list[str] = []

    for tok in raw_tokens:
        tok_l = tok.lower()

        # Check explicit dot-extension: e.g. "tree.txt" or "tree.rxt"
        if "." in tok:
            parts = tok.rsplit(".", 1)
            stem_part, ext_part = parts[0], parts[1].lower()
            if ext_part.isalnum():
                if ext_part in COMMON_EXTS or f".{ext_part}" in TYPE_EXTENSIONS:
                    target_ext = f".{ext_part}"
                else:
                    close_ext = difflib.get_close_matches(ext_part, COMMON_EXTS, n=1, cutoff=0.6)
                    if close_ext:
                        target_ext = f".{close_ext[0]}"
                        corrections[f".{ext_part}"] = target_ext
                    else:
                        target_ext = f".{ext_part}"
                if target_ext not in ext_filters:
                    ext_filters.append(target_ext)

                if stem_part:
                    for sub_w in split_stem_tokens(stem_part):
                        if sub_w not in FILLER_TOKENS:
                            name_tokens.append(sub_w)
                continue

        # Split on underscores or non-alphanumerics inside token
        sub_tokens = split_stem_tokens(tok_l)
        for w in sub_tokens:
            if not w:
                continue

            # Check fuzzy vocabulary for commands/types/fillers (len >= 4)
            if len(w) >= 4 and w not in CANONICAL_VOCAB and w not in FILLER_TOKENS:
                close = difflib.get_close_matches(w, CANONICAL_VOCAB, n=1, cutoff=0.82)
                if close:
                    canonical = close[0]
                    corrections[w] = canonical
                    w = canonical

            # Check if directory keyword
            if w in DIR_WORDS:
                want_dir = True
                continue

            # Check if type keyword
            if w in TYPE_EXTENSIONS:
                for e in TYPE_EXTENSIONS[w]:
                    if e not in ext_filters:
                        ext_filters.append(e)
                continue

            # Check if filler token
            if w in FILLER_TOKENS or w in ("send", "sent", "share", "give", "get", "find", "show", "search", "file", "files"):
                continue

            # Name typo tolerance using index vocabulary (len >= 4)
            if len(w) >= 4 and vocab and w not in vocab:
                close_name = difflib.get_close_matches(w, vocab, n=1, cutoff=0.8)
                if close_name:
                    corrections[w] = close_name[0]
                    w = close_name[0]

            name_tokens.append(w)

    return ParsedQuery(
        tokens=name_tokens,
        ext_filter=ext_filters,
        want_dir=want_dir,
        location_terms=loc_terms,
        corrections=corrections,
        raw=q,
    )


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


def _score_candidate(candidate: Path, pq: ParsedQuery) -> tuple[int, float, float, int] | None:
    """Evaluate candidate file against ParsedQuery.

    Returns tuple (tier, matched_ratio, mtime, neg_path_len) or None if no match.
    Tier 1 is Tier A (100% matched), Tier 2 is Tier B (>= 60% matched).
    """
    if is_sensitive_path(candidate) or is_excluded_path(candidate):
        return None
    if candidate.name.startswith("."):
        return None

    try:
        is_dir = candidate.is_dir()
    except (OSError, PermissionError):
        return None

    if pq.want_dir and not is_dir:
        return None

    if pq.location_terms and not _path_matches_location(candidate, pq.location_terms):
        return None

    if pq.ext_filter and not pq.want_dir:
        if candidate.suffix.lower() not in pq.ext_filter:
            return None

    try:
        st = candidate.stat()
        mtime = st.st_mtime
    except Exception:
        mtime = 0.0

    neg_len = -len(str(candidate))

    if not pq.tokens:
        # No specific name tokens: matched by ext or want_dir or location
        return (1, 1.0, mtime, neg_len)

    cand_tokens = split_stem_tokens(candidate.stem)
    parent_tokens: list[str] = []
    if not pq.location_terms:
        for p in candidate.parents:
            if p.name:
                parent_tokens.extend(split_stem_tokens(p.name))

    matched_tokens = 0
    total_q = len(pq.tokens)

    for q_tok in pq.tokens:
        # Match candidate stem
        stem_match = any(c == q_tok or (len(q_tok) >= 3 and c.startswith(q_tok)) for c in cand_tokens)
        if stem_match:
            matched_tokens += 1
            continue
        # Match extension as a token
        if candidate.suffix.lower().lstrip(".") == q_tok:
            matched_tokens += 1
            continue
        # Match parent folder if location_terms was empty
        if not pq.location_terms:
            parent_match = any(p == q_tok or (len(q_tok) >= 3 and p.startswith(q_tok)) for p in parent_tokens)
            if parent_match:
                matched_tokens += 1
                continue

    ratio = matched_tokens / total_q

    if matched_tokens == total_q:
        return (1, 1.0, mtime, neg_len)

    if total_q >= 2 and ratio >= 0.6:
        return (2, ratio, mtime, neg_len)

    return None


def _match_candidate(
    candidate: Path,
    name_tokens: list[str],
    ext_filters: list[str],
) -> tuple[int, bool] | None:
    """Evaluate candidate file for backward-compatibility.

    Returns tuple (tier_score, approximate) or None if no match.
    Tier A returns (100, False), Tier B returns (60, True).
    """
    pq = ParsedQuery(
        tokens=name_tokens,
        ext_filter=ext_filters,
        want_dir=False,
        location_terms=[],
        corrections={},
        raw="",
    )
    res = _score_candidate(candidate, pq)
    if res is None:
        return None
    tier, ratio, _, _ = res
    if tier == 1:
        # Check exact stem match bonus
        cand_tokens = split_stem_tokens(candidate.stem)
        score = 110 if cand_tokens == name_tokens else 100
        return (score, False)
    return (60, True)


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


def _scan_tier1_fast(
    name_tokens: list[str],
    ext_filters: list[str],
    timeout_sec: float = 1.0,
) -> list[FileMatch]:
    """Perform a fast live walk of standard user folders capped at timeout_sec."""
    home = Path.home()
    priority_subdirs = ["Desktop", "Downloads", "Documents", "Pictures", "Videos", "Music"]
    roots: list[Path] = []
    for d in priority_subdirs:
        p = home / d
        if p.is_dir() and not is_sensitive_path(p) and not is_excluded_path(p):
            roots.append(p.resolve())

    # OneDrive folders
    try:
        for p in home.iterdir():
            if p.is_dir() and p.name.lower().startswith("onedrive") and not is_sensitive_path(p):
                for d in priority_subdirs:
                    od_sub = p / d
                    if od_sub.is_dir() and not is_sensitive_path(od_sub) and not is_excluded_path(od_sub):
                        roots.append(od_sub.resolve())
    except Exception:
        pass

    matches: list[FileMatch] = []
    t0 = time.perf_counter()

    for root_dir in roots:
        if (time.perf_counter() - t0) > timeout_sec:
            break
        for current_root, dirs, files in os.walk(str(root_dir), onerror=lambda _: None):
            if (time.perf_counter() - t0) > timeout_sec:
                break
            curr_p = Path(current_root)
            dirs[:] = [d for d in dirs if not _should_skip_dir(d, curr_p / d)]
            for f in files:
                if f.startswith("."):
                    continue
                fp = curr_p / f
                if is_sensitive_path(fp) or is_excluded_path(fp):
                    continue
                res = _match_candidate(fp, name_tokens, ext_filters)
                if res is not None:
                    _, approx = res
                    try:
                        st = fp.stat()
                        matches.append(FileMatch(path=fp, size=st.st_size, mtime=st.st_mtime, approximate=approx))
                    except Exception:
                        pass
    return matches


def _scan_root(
    root_dir: Path,
    name_tokens: list[str],
    ext_filters: list[str],
    scored_matches: list[tuple[int, float, FileMatch]],
    timeout_sec: float | None = None,
    start_time: float | None = None,
    pq: ParsedQuery | None = None,
) -> bool:
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

        # Check directories
        for d in dirs:
            if d.startswith("."):
                continue
            dir_path = curr_p / d
            if is_excluded_path(dir_path) or is_sensitive_path(dir_path):
                continue
            res = _score_candidate(dir_path, pq) if pq else _match_candidate(dir_path, name_tokens, ext_filters)
            if res is not None:
                if pq:
                    tier, _, mtime, _ = res
                    approx = (tier == 2)
                    score = 100 if tier == 1 else 60
                else:
                    score, approx = res
                try:
                    st = dir_path.stat()
                    scored_matches.append((score, st.st_mtime, FileMatch(path=dir_path, size=st.st_size, mtime=st.st_mtime, approximate=approx)))
                except Exception:
                    pass

        # Check files
        for f in files:
            if f.startswith("."):
                continue
            file_path = curr_p / f
            if is_excluded_path(file_path) or is_sensitive_path(file_path):
                continue
            res = _score_candidate(file_path, pq) if pq else _match_candidate(file_path, name_tokens, ext_filters)
            if res is not None:
                if pq:
                    tier, _, mtime, _ = res
                    approx = (tier == 2)
                    score = 100 if tier == 1 else 60
                else:
                    score, approx = res
                try:
                    st = file_path.stat()
                    scored_matches.append((score, st.st_mtime, FileMatch(path=file_path, size=st.st_size, mtime=st.st_mtime, approximate=approx)))
                except Exception:
                    pass

    return False


def _scan_home_wide(
    home: Path,
    name_tokens: list[str],
    ext_filters: list[str],
    searched_tier1_paths: set[str],
    scored_matches: list[tuple[int, float, FileMatch]],
    timeout_sec: float = 6.0,
    pq: ParsedQuery | None = None,
) -> None:
    """Tier 2 scan: Walk rest of Path.home() capped at timeout_sec."""
    t0 = time.perf_counter()
    if home.exists() and home.is_dir() and not is_sensitive_path(home):
        try:
            for entry in home.iterdir():
                if time.perf_counter() - t0 > timeout_sec:
                    break
                if not entry.is_dir():
                    if not entry.name.startswith("."):
                        res = _score_candidate(entry, pq) if pq else _match_candidate(entry, name_tokens, ext_filters)
                        if res is not None:
                            if pq:
                                tier, _, mtime, _ = res
                                approx = (tier == 2)
                                score = 100 if tier == 1 else 60
                            else:
                                score, approx = res
                            try:
                                st = entry.stat()
                                scored_matches.append((score, st.st_mtime, FileMatch(path=entry, size=st.st_size, mtime=st.st_mtime, approximate=approx)))
                            except Exception:
                                pass
                    continue

                if _should_skip_dir(entry.name, entry) or is_sensitive_path(entry) or is_excluded_path(entry):
                    continue

                try:
                    res_p = entry.resolve()
                except Exception:
                    res_p = entry
                if str(res_p).lower() in searched_tier1_paths:
                    continue

                timed_out = _scan_root(
                    entry,
                    name_tokens=name_tokens,
                    ext_filters=ext_filters,
                    scored_matches=scored_matches,
                    timeout_sec=timeout_sec,
                    start_time=t0,
                    pq=pq,
                )
                if timed_out:
                    break
        except Exception:
            pass


def search_files(query: str, file_type: str = "", max_results: int = 5) -> list[FileMatch]:
    """Find files or folders matching ``query`` returning list of FileMatch objects."""
    t_start = time.perf_counter()
    cleaned_query = query.strip().strip('"\'')
    if not cleaned_query:
        return []

    # Early rejection for denylist keywords
    if check_denylist_rejection(cleaned_query):
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

    pq = parse_query(cleaned_query)

    if file_type:
        clean_ext = file_type.strip().lower()
        clean_ext = clean_ext if clean_ext.startswith(".") else f".{clean_ext}"
        if clean_ext not in pq.ext_filter:
            pq.ext_filter.append(clean_ext)

    if not pq.tokens and not pq.ext_filter and not pq.location_terms and not pq.want_dir:
        return []

    # Determine search roots
    mode = getattr(config, "FILE_SEARCH_MODE", "home").strip().lower()
    override_roots: list[Path] = []
    if config.FILE_SEARCH_ROOTS or mode == "custom":
        for r in (config.FILE_SEARCH_ROOTS or "").split(","):
            r_str = r.strip()
            if r_str:
                p = Path(os.path.expandvars(os.path.expanduser(r_str)))
                if p.exists() and p.is_dir() and not is_sensitive_path(p) and not is_excluded_path(p):
                    override_roots.append(p.resolve())
    elif mode == "all_drives":
        override_roots = roots_for_mode("all_drives")

    from plugins.file_index import file_indexer  # noqa: PLC0415
    index_ready = bool(file_indexer.has_completed_crawl() or file_indexer.is_ready())

    raw_candidates: list[Path] = []
    index_ms = 0.0
    live_ms = 0.0

    if index_ready:
        t0 = time.perf_counter()
        index_matches = file_indexer.search(
            name_tokens=pq.tokens,
            ext_filters=pq.ext_filter,
            max_results=max_results * 8,
        )
        for m in index_matches:
            raw_candidates.append(m.path)
        index_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        live_matches = _scan_tier1_fast(pq.tokens, pq.ext_filter, timeout_sec=1.0)
        for m in live_matches:
            raw_candidates.append(m.path)
        live_ms = (time.perf_counter() - t1) * 1000.0
    else:
        t1 = time.perf_counter()
        scored_matches: list[tuple[int, float, FileMatch]] = []
        if override_roots:
            for r in override_roots:
                _scan_root(r, pq.tokens, pq.ext_filter, scored_matches, pq=pq)
        else:
            home = Path.home()
            priority_subdirs = ["Desktop", "Downloads", "Documents", "Pictures", "Videos", "Music"]
            tier1_roots: list[Path] = []
            for d in priority_subdirs:
                p = home / d
                if p.is_dir() and not is_sensitive_path(p):
                    tier1_roots.append(p.resolve())
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
                _scan_root(root, pq.tokens, pq.ext_filter, scored_matches, pq=pq)

            # Check if any candidate is Tier A
            has_tier_a = False
            for _, _, m in scored_matches:
                res = _score_candidate(m.path, pq)
                if res and res[0] == 1:
                    has_tier_a = True

            if not has_tier_a:
                _scan_home_wide(
                    home=home,
                    name_tokens=pq.tokens,
                    ext_filters=pq.ext_filter,
                    searched_tier1_paths=searched_tier1_paths,
                    scored_matches=scored_matches,
                    timeout_sec=6.0,
                    pq=pq,
                )

        for _, _, m in scored_matches:
            raw_candidates.append(m.path)
        live_ms = (time.perf_counter() - t1) * 1000.0

    # Two-tier ranking
    seen_paths: set[Path] = set()
    tier_a_matches: list[tuple[float, float, int, FileMatch]] = []
    tier_b_matches: list[tuple[float, float, int, FileMatch]] = []

    for cand in raw_candidates:
        try:
            cand_res = cand.resolve()
        except Exception:
            cand_res = cand
        if cand_res in seen_paths:
            continue
        seen_paths.add(cand_res)

        score_res = _score_candidate(cand_res, pq)
        if score_res is not None:
            tier, ratio, mtime, neg_len = score_res
            try:
                st = cand_res.stat()
                sz = st.st_size
            except Exception:
                sz = 0
            match_obj = FileMatch(path=cand_res, size=sz, mtime=mtime, approximate=(tier == 2))
            if tier == 1:
                tier_a_matches.append((ratio, mtime, neg_len, match_obj))
            else:
                tier_b_matches.append((ratio, mtime, neg_len, match_obj))

    tier_a_matches.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    tier_b_matches.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)

    if tier_a_matches:
        results = [m for _, _, _, m in tier_a_matches[:max_results]]
    else:
        results = [m for _, _, _, m in tier_b_matches[:min(5, max_results)]]

    total_ms = (time.perf_counter() - t_start) * 1000.0
    log.info("search timing: index_ms=%.1f live_ms=%.1f total_ms=%.1f results=%d", index_ms, live_ms, total_ms, len(results))
    return results


def find_files(query: str, file_type: str = "", max_results: int = 5) -> list[Path]:
    """Backward-compatible file search returning list of Path objects."""
    matches = search_files(query, file_type=file_type, max_results=max_results)
    return [m.path for m in matches]


def resolve_share_target(raw_input: str) -> tuple[Path | None, list[FileMatch], str | None]:
    """Resolve raw input (path, filename, keywords, or alias) to a local file or folder.

    Returns:
        (resolved_path, matches, error_message)
        - If resolved to a single valid path: (Path, matches, None)
        - If ambiguous (multiple matches): (None, matches, None)
        - If not found or access blocked: (None, [], error_str)
    """
    if not raw_input or not str(raw_input).strip():
        return None, [], "Please specify the file or folder path to share."

    clean_input = str(raw_input).strip().strip("'\"").strip()
    if not clean_input:
        return None, [], "Please specify the file or folder path to share."

    # Early rejection for denylist keywords
    denial = check_denylist_rejection(clean_input)
    if denial:
        return None, [], denial

    # 1. Try resolve_user_path (handles aliases like "Downloads", "~", hallucinated users, relative tails)
    cand = resolve_user_path(clean_input)
    if cand is not None and cand.exists():
        if is_sensitive_path(cand):
            return None, [], f"I cannot share protected or sensitive files ({cand.name})."
        return cand, [], None

    # 2. Try direct path on disk
    try:
        direct_p = Path(os.path.expandvars(os.path.expanduser(clean_input))).resolve()
        if direct_p.exists():
            if is_sensitive_path(direct_p):
                return None, [], f"I cannot share protected or sensitive files ({direct_p.name})."
            return direct_p, [], None
    except Exception:
        pass

    # 3. Strip fake/hallucinated directory prefix or search with clean input
    matches = search_files(clean_input)
    if not matches and ("/" in clean_input or "\\" in clean_input):
        leaf = Path(clean_input).name
        if leaf and leaf != clean_input:
            matches = search_files(leaf)

    if not matches:
        miss_msg = format_miss_reply(clean_input)
        return None, [], f"No file found matching '{clean_input}'. {miss_msg}"

    # 4. Check single match vs multiple matches
    if len(matches) == 1:
        target = matches[0].path
        if is_sensitive_path(target):
            return None, [], f"I cannot share protected or sensitive files ({target.name})."
        return target, matches, None

    # Check for 1 exact Tier-A stem or filename match among multiple matches
    q_name = Path(clean_input).name.lower()
    q_stem = Path(clean_input).stem.lower()

    exact_names = [m for m in matches if m.path.name.lower() == q_name]
    if len(exact_names) == 1:
        target = exact_names[0].path
        if is_sensitive_path(target):
            return None, [], f"⚠️ Access denied: '{target.name}' is a protected or sensitive file."
        return target, matches, None

    exact_stems = [m for m in matches if m.path.stem.lower() == q_stem]
    if len(exact_stems) == 1:
        target = exact_stems[0].path
        if is_sensitive_path(target):
            return None, [], f"⚠️ Access denied: '{target.name}' is a protected or sensitive file."
        return target, matches, None

    return None, matches, None


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

