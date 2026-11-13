"""
plugins/p_folder.py
-------------------
Interactive folder and drive browsing plugin.
Returns directory contents paginated with Telegram inline button choices:
- 📁 Folders can be tapped to navigate deeper (inline)
- 📄 Files can be tapped to trigger share_file (with approval)
- ⬆ Up navigates to parent directory
- ➡ More paginates next 8 items
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import re
from typing import Any

from plugins.base import PluginBase, PluginResult
from plugins.file_search import (
    is_excluded_path,
    is_reparse_point,
    is_sensitive_path,
    resolve_user_path,
    roots_for_mode,
)

log = logging.getLogger(__name__)

PAGE_SIZE = 8


def _format_size(size_bytes: int) -> str:
    """Format file size in B, KB, or MB."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


class ListFolderPlugin(PluginBase):
    """Lists folders and files in a directory for interactive browsing."""

    name = "list_folder"
    description = "Lists folders and files in a directory for interactive browsing."
    usage_hint = "when they ask what is in a folder or drive"
    requires_approval = False

    parameters_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to folder or drive to list. Omit or empty to list root drives / folders.",
            },
            "offset": {
                "type": "integer",
                "description": "Paging offset (0-based, increments by 8). Default is 0.",
            },
        },
        "required": [],
    }

    async def execute(self, **kwargs: Any) -> PluginResult:
        raw_path = str(kwargs.get("path") or "").strip()
        try:
            offset = max(0, int(kwargs.get("offset") or 0))
        except (ValueError, TypeError):
            offset = 0

        # Case 1: Root / Drives listing
        if not raw_path:
            roots = roots_for_mode()
            allowed_roots = [
                r for r in roots
                if not is_sensitive_path(r) and not is_excluded_path(r)
            ]
            if not allowed_roots:
                msg = "No accessible drives or folders found."
                return PluginResult(
                    success=False,
                    output=msg,
                    data={"direct_reply": msg, "choices": []},
                )

            lines = ["📂 Drives and Roots:"]
            choices = []
            for r in allowed_roots:
                name = r.name or str(r)
                lines.append(f"📁 {name}")
                choices.append({
                    "label": f"📁 {name}",
                    "tool": "list_folder",
                    "kwargs": {"path": str(r.resolve())},
                })

            direct_reply = "\n".join(lines) + "\n\nTap a folder to browse."
            return PluginResult(
                success=True,
                output=direct_reply,
                data={
                    "path": "",
                    "total": len(allowed_roots),
                    "offset": 0,
                    "choices": choices,
                    "direct_reply": direct_reply,
                },
            )

        # Case 2: Target folder listing
        clean_input = raw_path.strip()

        # Check if user asked for a drive root that does not exist (e.g. "D:/", "D:\", "D:", "Z:/", "D drive")
        drive_match = re.match(r"^([a-zA-Z]):[/\\]*$", clean_input) or re.match(r"^([a-zA-Z])\s*drive$", clean_input, re.I)
        if drive_match:
            drive_letter = drive_match.group(1).upper()
            target_drive = Path(f"{drive_letter}:/")
            if not target_drive.exists():
                available = ", ".join(r.name or str(r) for r in roots_for_mode())
                msg = f"Drive {drive_letter}: was not found. Available roots: {available}."
                return PluginResult(
                    success=False,
                    output=msg,
                    data={"direct_reply": msg, "choices": []},
                )

        resolved = resolve_user_path(clean_input)
        if resolved is not None and resolved.exists():
            target = resolved
        else:
            try:
                target = Path(clean_input).resolve()
            except Exception as exc:
                err_msg = f"Invalid path: {clean_input}"
                return PluginResult(
                    success=False,
                    output=err_msg,
                    data={"direct_reply": err_msg, "choices": []},
                    error=str(exc),
                )

        if not target.exists():
            msg = f"Folder not found: {raw_path}"
            return PluginResult(
                success=False,
                output=msg,
                data={"direct_reply": msg, "choices": []},
            )

        if not target.is_dir():
            msg = f"Not a directory: {raw_path}"
            return PluginResult(
                success=False,
                output=msg,
                data={"direct_reply": msg, "choices": []},
            )

        if is_sensitive_path(target) or is_excluded_path(target):
            msg = "⚠️ Access to this folder is restricted."
            return PluginResult(
                success=False,
                output=msg,
                data={"direct_reply": msg, "choices": []},
            )

        dirs: list[Path] = []
        files: list[tuple[Path, int, float]] = []

        try:
            with os.scandir(target) as it:
                for entry in it:
                    try:
                        epath = Path(entry.path)
                        if is_reparse_point(epath):
                            continue
                        if is_sensitive_path(epath) or is_excluded_path(epath):
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            dirs.append(epath)
                        elif entry.is_file(follow_symlinks=False):
                            st = entry.stat(follow_symlinks=False)
                            files.append((epath, st.st_size, st.st_mtime))
                    except (OSError, PermissionError):
                        continue
        except (OSError, PermissionError) as exc:
            msg = f"⚠️ Cannot open folder {target.name}: {exc}"
            return PluginResult(
                success=False,
                output=msg,
                data={"direct_reply": msg, "choices": []},
                error=str(exc),
            )

        # Sort: directories alphabetically, files newest first
        dirs.sort(key=lambda p: p.name.lower())
        files.sort(key=lambda item: item[2], reverse=True)

        combined: list[tuple[str, Path, int]] = []
        for d in dirs:
            combined.append(("dir", d, 0))
        for f, sz, _ in files:
            combined.append(("file", f, sz))

        total = len(combined)
        page_items = combined[offset : offset + PAGE_SIZE]
        has_more = (offset + PAGE_SIZE) < total

        # Check parent navigation
        has_parent = False
        parent_kwarg_path = ""
        if target.parent != target:
            p = target.parent.resolve()
            if not is_sensitive_path(p) and not is_excluded_path(p):
                has_parent = True
                parent_kwarg_path = str(p)
        else:
            # Drive root (e.g. C:\) -> Up goes back to roots listing
            has_parent = True
            parent_kwarg_path = ""

        choices: list[dict[str, Any]] = []
        if has_parent:
            choices.append({
                "label": "⬆ Up",
                "tool": "list_folder",
                "kwargs": {"path": parent_kwarg_path},
            })

        for kind, item_path, sz in page_items:
            if kind == "dir":
                choices.append({
                    "label": f"📁 {item_path.name}",
                    "tool": "list_folder",
                    "kwargs": {"path": str(item_path.resolve())},
                })
            else:
                choices.append({
                    "label": f"📄 {item_path.name}",
                    "tool": "share_file",
                    "kwargs": {"path": str(item_path.resolve())},
                })

        if has_more:
            choices.append({
                "label": "➡ More",
                "tool": "list_folder",
                "kwargs": {"path": str(target.resolve()), "offset": offset + PAGE_SIZE},
            })

        # Format output text
        lines = [f"📂 {target}"]
        if total == 0:
            lines.append("(empty folder)")
        else:
            for kind, item_path, sz in page_items:
                if kind == "dir":
                    lines.append(f"📁 {item_path.name}/")
                else:
                    lines.append(f"📄 {item_path.name} ({_format_size(sz)})")

        output_text = "\n".join(lines)
        direct_reply = f"{output_text}\n\nTap a folder to browse or a file to share."
        return PluginResult(
            success=True,
            output=direct_reply,
            data={
                "path": str(target),
                "total": total,
                "offset": offset,
                "choices": choices,
                "direct_reply": direct_reply,
            },
        )
