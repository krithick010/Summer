"""
plugins/p_files.py
------------------
Local file search, smart compression, and multi-channel file sharing plugins.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
import logging
import os
from pathlib import Path
import time
from typing import Any

import config
from plugins.base import PluginBase, PluginResult
from plugins.file_search import (
    ALREADY_COMPRESSED_EXTENSIONS,
    check_and_increment_share_limit,
    compress_to_zip,
    find_files,
    get_search_roots,
    is_outside_usual_folders,
    is_sensitive_path,
    is_soft_sensitive,
    log_file_audit,
    search_files,
    split_file_chunks,
)

log = logging.getLogger(__name__)


def _format_size(size_bytes: int) -> str:
    """Format bytes into human-readable B, KB, or MB."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.2f} MB"


class FindFilesPlugin(PluginBase):
    """Searches for files or folders on the local PC matching a name, keywords, or path."""

    name = "find_files"
    description = (
        "Searches for files or folders on the local PC matching a name, keywords, or path."
    )
    usage_hint = "when they ask to find, locate or send a file or folder (always search first)"
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Name, keywords, or path of the file or folder to find",
            },
        },
        "required": ["query"],
    }
    requires_approval = False

    async def execute(self, **kwargs: Any) -> PluginResult:
        query = str(kwargs.get("query", "")).strip()

        if not query:
            return PluginResult(
                success=False,
                output="⚠️ Please specify what file or folder you want to find.",
                error="Missing required query parameter",
            )

        matches = await asyncio.to_thread(search_files, query)
        if not matches:
            roots = get_search_roots()
            folders = ", ".join(p.name or str(p) for p in roots)
            no_match_msg = f"No match for '{query}'. Looked in: {folders}."
            return PluginResult(
                success=False,
                output=no_match_msg,
                data={"direct_reply": no_match_msg, "matches": []},
            )

        top_matches = matches[:5]
        lines = []
        choices = []
        for idx, m in enumerate(top_matches, start=1):
            size_mb = m.size / (1024 * 1024)
            size_str = f"{size_mb:.1f} MB"
            date_str = datetime.fromtimestamp(m.mtime).strftime("%Y-%m-%d")
            folder_name = m.path.parent.name
            lines.append(f"{idx}. {m.path.name} - {size_str} - {date_str} - in {folder_name}")
            choices.append({
                "label": f"{idx}. {m.path.name} ({size_str})",
                "tool": "share_file",
                "kwargs": {"path": str(m.path.resolve())},
            })

        list_text = "\n".join(lines)
        direct_reply = f"{list_text}\nTap one to send it."
        return PluginResult(
            success=True,
            output=direct_reply,
            data={
                "matches": [str(m.path.resolve()) for m in top_matches],
                "direct_reply": direct_reply,
                "choices": choices,
            },
        )


class ShareFilePlugin(PluginBase):
    """Finds, prepares, and shares a local file or folder via Telegram or email."""

    name = "share_file"
    description = (
        "Finds, prepares, and shares a local file or folder via Telegram or email."
    )
    usage_hint = "only with an exact path from find_files or typed by the user"
    parameters_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute path of the file or folder to share",
            },
            "delivery": {
                "type": "string",
                "enum": ["telegram", "email"],
                "description": "Delivery channel: 'telegram' (default) or 'email'",
            },
            "to_email": {
                "type": "string",
                "description": "Optional recipient email address when delivery is email",
            },
        },
        "required": ["path"],
    }
    requires_approval = True

    async def prepare_preview(self, kwargs: dict[str, Any]) -> str | None:
        """Resolve target file and build a rich HITL approval preview."""
        path_str = str(kwargs.get("path") or kwargs.get("query") or "").strip()
        delivery = str(kwargs.get("delivery", "telegram")).lower()
        is_email = (delivery == "email") or ("email" in delivery) or ("mail" in delivery)
        route_str = "Email" if is_email else "Telegram"

        if not path_str:
            return None

        target = Path(path_str)
        if not target.is_absolute() and not target.exists():
            matches = await asyncio.to_thread(find_files, path_str)
            if matches:
                target = matches[0]

        if is_sensitive_path(target):
            return f"• Path: {target}\n• File: {target.name}\n• Security: Blocked (sensitive file)"

        if not target.exists():
            return f"• Path: {target}\n• File: {target.name}\n• Status: File not found"

        try:
            if target.is_dir():
                size_bytes = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
            else:
                size_bytes = target.stat().st_size
            size_mb = size_bytes / (1024 * 1024)
        except Exception:
            size_mb = 0.0

        preview_lines = [
            f"• File: {target.name}",
            f"• Path: {target.resolve()}",
            f"• Size: {size_mb:.1f} MB",
            f"• Route: {route_str}",
            f"• Action: Send via {route_str}",
        ]

        if is_outside_usual_folders(target):
            preview_lines.append("⚠️ Note: file is outside your usual folders")
        if is_soft_sensitive(target):
            preview_lines.append("⚠️ Note: filename looks sensitive")

        return "\n".join(preview_lines)

    async def execute(self, **kwargs: Any) -> PluginResult:
        path_str = str(kwargs.get("path") or kwargs.get("query") or "").strip()
        delivery = str(kwargs.get("delivery", "telegram")).lower()
        to_email = str(kwargs.get("to_email", "")).strip()
        user_id = kwargs.get("user_id")

        if not path_str:
            return PluginResult(
                success=False,
                output="⚠️ Please specify the file or folder path to share.",
                error="Missing required path parameter",
            )

        target = Path(path_str)
        if not target.is_absolute() and not target.exists():
            matches = await asyncio.to_thread(find_files, path_str)
            if matches:
                target = matches[0]
            else:
                return PluginResult(
                    success=False,
                    output=f"Could not find any file or folder for '{path_str}'.",
                    error=f"File not found: {path_str}",
                )

        if not target.exists():
            return PluginResult(
                success=False,
                output=f"⚠️ Target path does not exist: `{target}`",
                error=f"Target path not found: {target}",
            )

        if is_sensitive_path(target):
            return PluginResult(
                success=False,
                output=f"⚠️ Access denied: '{target.name}' is a protected or sensitive file.",
                error="Sensitive file access blocked",
            )

        # Rate limit check
        allowed, rl_msg = check_and_increment_share_limit(user_id)
        if not allowed:
            log_file_audit(
                user_id=user_id or 0,
                action="share",
                path=target,
                size=0,
                route=delivery,
                status="rate_limited",
            )
            return PluginResult(
                success=False,
                output=f"⚠️ {rl_msg}",
                error=rl_msg,
            )

        tmp_shares_dir = Path(config.DATA_DIR) / "tmp_shares"
        tmp_shares_dir.mkdir(parents=True, exist_ok=True)

        is_temp_zip = False
        send_path = target

        # Handle directory compression
        if target.is_dir():
            try:
                t0 = time.perf_counter()
                send_path = await asyncio.to_thread(compress_to_zip, target, tmp_shares_dir)
                is_temp_zip = True
                log.info("Compressed directory '%s' in %.1f ms", target.name, (time.perf_counter() - t0) * 1000)
            except Exception as exc:
                return PluginResult(
                    success=False,
                    output=f"⚠️ Failed to compress directory '{target.name}': {exc}",
                    error=str(exc),
                )

        try:
            if send_path.is_file():
                size_mb = os.path.getsize(str(send_path)) / (1024 * 1024)
            else:
                size_mb = sum(os.path.getsize(str(f)) for f in send_path.rglob("*") if f.is_file()) / (1024 * 1024)
        except Exception as exc:
            if is_temp_zip:
                send_path.unlink(missing_ok=True)
            return PluginResult(
                success=False,
                output=f"⚠️ Could not check file size for '{send_path.name}': {exc}",
                error=str(exc),
            )

        is_email = (delivery == "email") or ("email" in delivery) or ("mail" in delivery)

        if is_email:
            from plugins.p_gmail import GmailPlugin
            gmail_plugin = GmailPlugin()
            resolved_recipient = to_email or config.USER_DEFAULT_EMAIL or config.GMAIL_SENDER_EMAIL
            if not resolved_recipient:
                if is_temp_zip:
                    send_path.unlink(missing_ok=True)
                return PluginResult(
                    success=False,
                    output="⚠️ No recipient email address specified and USER_DEFAULT_EMAIL is not set.",
                    error="Missing recipient email",
                )

            is_already_compressed = (not target.is_dir()) and (target.suffix.lower() in ALREADY_COMPRESSED_EXTENSIONS)
            if size_mb > 25.0 and not is_temp_zip and not is_already_compressed:
                try:
                    t0 = time.perf_counter()
                    send_path = await asyncio.to_thread(compress_to_zip, send_path, tmp_shares_dir)
                    is_temp_zip = True
                    size_mb = os.path.getsize(str(send_path)) / (1024 * 1024)
                    log.info("Compressed file for email in %.1f ms", (time.perf_counter() - t0) * 1000)
                except Exception as exc:
                    return PluginResult(
                        success=False,
                        output=f"⚠️ Compression failed for '{send_path.name}': {exc}",
                        error=str(exc),
                    )

            if size_mb > 25.0:
                if is_temp_zip:
                    send_path.unlink(missing_ok=True)
                return PluginResult(
                    success=False,
                    output=(
                        f"⚠️ Cannot email '{target.name}': file is {size_mb:.1f} MB "
                        "(Gmail attachment limit is 25 MB)."
                    ),
                    error="File exceeds Gmail 25 MB limit",
                )

            try:
                res = await gmail_plugin.execute(
                    to_email=resolved_recipient,
                    subject=f"Requested File: {target.name}",
                    body=f"Sending requested file: {target.name} ({size_mb:.2f} MB)",
                    attachment_path=str(send_path),
                )
                direct_reply = f"Sent: {target.name} ({size_mb:.1f} MB) via Email."
                res.data = {"direct_reply": direct_reply}
                res.output = direct_reply
                log_file_audit(
                    user_id=user_id or 0,
                    action="share",
                    path=target.resolve() if target.exists() else target,
                    size=f"{size_mb:.1f} MB",
                    route="Email",
                    status="approved",
                )
                return res
            finally:
                if is_temp_zip:
                    send_path.unlink(missing_ok=True)

        # Telegram delivery route (default up to FILE_TELEGRAM_MAX_MB, default 50 MB)
        if size_mb <= config.FILE_TELEGRAM_MAX_MB:
            is_photo = (
                send_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
                and size_mb <= 10.0
                and not is_temp_zip
            )
            kind = "photo" if is_photo else "document"
            direct_reply = f"Sent: {send_path.name} ({size_mb:.1f} MB) via Telegram."
            log_file_audit(
                user_id=user_id or 0,
                action="share",
                path=target.resolve() if target.exists() else target,
                size=f"{size_mb:.1f} MB",
                route="Telegram",
                status="approved",
            )
            return PluginResult(
                success=True,
                output=direct_reply,
                data={
                    "direct_reply": direct_reply,
                    "send_files": [
                        {
                            "path": str(send_path),
                            "filename": send_path.name,
                            "kind": kind,
                            "cleanup": is_temp_zip,
                        }
                    ],
                },
            )

        # File is > 50 MB. Skip zipping if already compressed; zip only text-like files and directories.
        is_already_compressed = (not target.is_dir()) and (target.suffix.lower() in ALREADY_COMPRESSED_EXTENSIONS)
        if not is_already_compressed and not is_temp_zip:
            try:
                t0 = time.perf_counter()
                send_path = await asyncio.to_thread(compress_to_zip, send_path, tmp_shares_dir)
                is_temp_zip = True
                size_mb = os.path.getsize(str(send_path)) / (1024 * 1024)
                log.info("Compressed text-like file '%s' in %.1f ms", target.name, (time.perf_counter() - t0) * 1000)
            except Exception as exc:
                return PluginResult(
                    success=False,
                    output=f"⚠️ Failed to compress '{send_path.name}': {exc}",
                    error=str(exc),
                )

        # If compression brought it under 50 MB:
        if size_mb <= config.FILE_TELEGRAM_MAX_MB:
            direct_reply = f"Sent: {send_path.name} ({size_mb:.1f} MB) via Telegram."
            log_file_audit(
                user_id=user_id or 0,
                action="share",
                path=target.resolve() if target.exists() else target,
                size=f"{size_mb:.1f} MB",
                route="Telegram",
                status="approved",
            )
            return PluginResult(
                success=True,
                output=direct_reply,
                data={
                    "direct_reply": direct_reply,
                    "send_files": [
                        {
                            "path": str(send_path),
                            "filename": send_path.name,
                            "kind": "document",
                            "cleanup": is_temp_zip,
                        }
                    ],
                },
            )

        # Over FILE_MAX_SPLIT_MB (default 150 MB)
        if size_mb > config.FILE_MAX_SPLIT_MB:
            if is_temp_zip:
                send_path.unlink(missing_ok=True)
            log_file_audit(
                user_id=user_id or 0,
                action="share",
                path=target.resolve() if target.exists() else target,
                size=f"{size_mb:.1f} MB",
                route=delivery,
                status="error",
            )
            return PluginResult(
                success=False,
                output=(
                    f"⚠️ File '{target.name}' is too large ({size_mb:.1f} MB). "
                    f"Telegram bots can only upload files up to 50 MB (or split up to {config.FILE_MAX_SPLIT_MB:.0f} MB). "
                    "Consider using email for files under 25 MB, or choose a smaller file."
                ),
                error=f"File exceeds maximum split limit of {config.FILE_MAX_SPLIT_MB} MB",
            )

        # Multi-part split Telegram route (50 MB - FILE_MAX_SPLIT_MB)
        try:
            t0 = time.perf_counter()
            parts = await asyncio.to_thread(
                split_file_chunks,
                send_path,
                chunk_size_mb=45.0,
                dest_dir=tmp_shares_dir,
                remove_source=is_temp_zip,
            )
            log.info("Split '%s' into %d parts in %.1f ms", send_path.name, len(parts), (time.perf_counter() - t0) * 1000)
        except Exception as exc:
            if is_temp_zip:
                send_path.unlink(missing_ok=True)
            return PluginResult(
                success=False,
                output=f"⚠️ Failed to split '{send_path.name}': {exc}",
                error=str(exc),
            )

        send_files = [
            {
                "path": str(part),
                "filename": part.name,
                "kind": "document",
                "cleanup": True,
            }
            for part in parts
        ]
        win_join = "+".join(f'"{p.name}"' for p in parts)
        join_hint = f'To join parts (Windows: copy /b {win_join} "{target.name}" ; Linux/macOS: cat "{target.name}.part"* > "{target.name}").'
        direct_reply = (
            f"Sent: {target.name} ({size_mb:.1f} MB) in {len(parts)} parts via Telegram.\n"
            f"{join_hint}"
        )
        log_file_audit(
            user_id=user_id or 0,
            action="share",
            path=target.resolve() if target.exists() else target,
            size=f"{size_mb:.1f} MB",
            route="Telegram",
            status="approved",
        )
        return PluginResult(
            success=True,
            output=direct_reply,
            data={"direct_reply": direct_reply, "send_files": send_files},
        )
