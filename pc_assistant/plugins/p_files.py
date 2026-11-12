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
from typing import Any

import config
from plugins.base import PluginBase, PluginResult
from plugins.file_search import (
    compress_to_zip,
    find_files,
    is_sensitive_path,
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
    usage_hint = "when they ask to search, find, or locate a file or folder on the PC"
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Name, keywords, or path of the file or folder to find",
            },
            "file_type": {
                "type": "string",
                "description": "Optional file extension filter (e.g. 'pdf', 'png')",
            },
        },
        "required": ["query"],
    }
    requires_approval = False

    async def execute(self, **kwargs: Any) -> PluginResult:
        query = str(kwargs.get("query", "")).strip()
        file_type = str(kwargs.get("file_type", "")).strip()

        if not query:
            return PluginResult(
                success=False,
                output="⚠️ Please specify what file or folder you want to find.",
                error="Missing required query parameter",
            )

        matches = await asyncio.to_thread(find_files, query, file_type=file_type)
        if not matches:
            return PluginResult(
                success=True,
                output=f"No files or folders found matching '{query}'.",
                data={"query": query, "files": []},
            )

        lines = [f"Found {len(matches)} matching file(s):"]
        for idx, path in enumerate(matches, start=1):
            try:
                st = path.stat()
                size_str = _format_size(st.st_size)
                mtime_str = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")
                item_type = "folder" if path.is_dir() else "file"
                lines.append(f"{idx}. {path.name} ({size_str}, {item_type}) - `{path}` [Modified: {mtime_str}]")
            except Exception:
                lines.append(f"{idx}. {path.name} - `{path}`")

        return PluginResult(
            success=True,
            output="\n".join(lines),
            data={"query": query, "files": [str(p) for p in matches]},
        )


class ShareFilePlugin(PluginBase):
    """Finds, prepares, and shares a local file or folder via Telegram or email."""

    name = "share_file"
    description = (
        "Finds, prepares, and shares a local file or folder via Telegram or email."
    )
    usage_hint = "when they ask to send, share, or email a file or folder"
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Exact/partial path or blunt/fuzzy name of the file or folder to send",
            },
            "delivery": {
                "type": "string",
                "enum": ["auto", "telegram", "email"],
                "description": "Delivery channel: 'auto' (default), 'telegram', or 'email'",
            },
            "to_email": {
                "type": "string",
                "description": "Optional recipient email address when delivery is email",
            },
        },
        "required": ["query"],
    }
    requires_approval = True

    async def prepare_preview(self, kwargs: dict[str, Any]) -> str | None:
        """Resolve top matching file and build a rich HITL approval preview."""
        query = str(kwargs.get("query", "")).strip()
        delivery = str(kwargs.get("delivery", "auto")).lower()
        if not query:
            return None

        matches = await asyncio.to_thread(find_files, query)
        if not matches:
            return f"• File: No matching file found for '{query}'\n• Action: Search will report not found"

        target = matches[0]
        if is_sensitive_path(target):
            return f"• File: {target}\n• Security: Blocked (sensitive file)"

        try:
            if target.is_dir():
                size_bytes = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
                size_mb = size_bytes / (1024 * 1024)
                action = "Auto-Zip & Send via Telegram" if delivery != "email" else "Auto-Zip & Email"
            else:
                size_mb = target.stat().st_size / (1024 * 1024)
                if delivery == "email":
                    action = "Email Attachment" if size_mb <= 25.0 else "Auto-Zip & Email"
                elif delivery == "auto" and config.FILE_TELEGRAM_MAX_MB < size_mb <= 25.0:
                    action = "Email Attachment (Gmail)"
                elif size_mb <= 50.0:
                    action = "Send via Telegram"
                elif size_mb <= config.FILE_MAX_SPLIT_MB:
                    action = "Auto-Zip & Split via Telegram"
                else:
                    action = "Too Large (exceeds limit)"
        except Exception:
            size_mb = 0.0
            action = "Send via Telegram"

        return f"• File: {target}\n• Size: {size_mb:.2f} MB\n• Action: {action}"

    async def execute(self, **kwargs: Any) -> PluginResult:
        query = str(kwargs.get("query", "")).strip()
        delivery = str(kwargs.get("delivery", "auto")).lower()
        to_email = str(kwargs.get("to_email", "")).strip()

        if not query:
            return PluginResult(
                success=False,
                output="⚠️ Please specify the file or folder to share.",
                error="Missing required query parameter",
            )

        matches = await asyncio.to_thread(find_files, query)
        if not matches:
            return PluginResult(
                success=False,
                output=f"Could not find any file or folder matching '{query}'.",
                error=f"File not found for query '{query}'",
            )

        target = matches[0]
        if is_sensitive_path(target):
            return PluginResult(
                success=False,
                output=f"⚠️ Access denied: '{target.name}' is a protected or sensitive file.",
                error="Sensitive file access blocked",
            )

        if not target.exists():
            return PluginResult(
                success=False,
                output=f"⚠️ Target path does not exist: `{target}`",
                error=f"Target path not found: {target}",
            )

        tmp_shares_dir = Path(config.DATA_DIR) / "tmp_shares"
        tmp_shares_dir.mkdir(parents=True, exist_ok=True)

        is_temp_zip = False
        send_path = target

        # Handle directory compression
        if target.is_dir():
            try:
                send_path = compress_to_zip(target, tmp_shares_dir)
                is_temp_zip = True
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

        # Determine if email routing should be used
        from plugins.p_gmail import GmailPlugin
        gmail_plugin = GmailPlugin()
        gmail_available = gmail_plugin.is_available()

        resolved_recipient = to_email or config.USER_DEFAULT_EMAIL or config.GMAIL_SENDER_EMAIL

        should_email = False
        if delivery == "email":
            should_email = True
        elif delivery == "auto" and (config.FILE_TELEGRAM_MAX_MB < size_mb <= 25.0) and gmail_available and resolved_recipient:
            should_email = True

        if should_email:
            if not resolved_recipient:
                if is_temp_zip:
                    send_path.unlink(missing_ok=True)
                return PluginResult(
                    success=False,
                    output="⚠️ No recipient email address specified and USER_DEFAULT_EMAIL is not set.",
                    error="Missing recipient email",
                )

            # If file > 25 MB and not zipped, attempt compression
            if size_mb > 25.0 and not is_temp_zip:
                try:
                    send_path = compress_to_zip(send_path, tmp_shares_dir)
                    is_temp_zip = True
                    size_mb = os.path.getsize(str(send_path)) / (1024 * 1024)
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
                return res
            finally:
                if is_temp_zip:
                    send_path.unlink(missing_ok=True)

        # Telegram delivery routes
        # If file > 50 MB and not zipped yet, compress first
        if size_mb > 50.0 and not is_temp_zip:
            try:
                send_path = compress_to_zip(send_path, tmp_shares_dir)
                is_temp_zip = True
                size_mb = os.path.getsize(str(send_path)) / (1024 * 1024)
            except Exception as exc:
                return PluginResult(
                    success=False,
                    output=f"⚠️ Failed to compress '{send_path.name}': {exc}",
                    error=str(exc),
                )

        # Telegram single file (<= 50 MB)
        if size_mb <= 50.0:
            is_photo = (
                send_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
                and size_mb <= 10.0
                and not is_temp_zip
            )
            kind = "photo" if is_photo else "document"
            return PluginResult(
                success=True,
                output=f"Sending '{send_path.name}' ({size_mb:.2f} MB) via Telegram.",
                data={
                    "send_files": [
                        {
                            "path": str(send_path),
                            "filename": send_path.name,
                            "kind": kind,
                            "cleanup": is_temp_zip,
                        }
                    ]
                },
            )

        # Multi-part split Telegram route (50 MB - FILE_MAX_SPLIT_MB)
        if size_mb <= config.FILE_MAX_SPLIT_MB:
            try:
                parts = split_file_chunks(send_path, chunk_size_mb=45.0)
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
            return PluginResult(
                success=True,
                output=(
                    f"'{target.name}' is {size_mb:.1f} MB (exceeds Telegram 50 MB single-file limit). "
                    f"Compressed and split into {len(parts)} parts for delivery."
                ),
                data={"send_files": send_files},
            )

        # Exceeds maximum split limit
        if is_temp_zip:
            send_path.unlink(missing_ok=True)

        return PluginResult(
            success=False,
            output=(
                f"⚠️ File '{target.name}' is too large ({size_mb:.1f} MB). "
                f"Maximum allowed size for sharing is {config.FILE_MAX_SPLIT_MB} MB."
            ),
            error=f"File exceeds maximum split limit of {config.FILE_MAX_SPLIT_MB} MB",
        )

