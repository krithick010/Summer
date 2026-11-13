"""
plugins/p_gmail.py
------------------
Gmail send plugin with mandatory HITL approval gate.

Authentication
~~~~~~~~~~~~~~
Uses OAuth 2.0 "installed app" flow. On first run the user must complete a
browser consent screen on the PC; the resulting token is cached in the path
set by ``GMAIL_TOKEN_PATH`` in config.py and refreshed automatically on
subsequent runs.

HITL
~~~~
``requires_approval = True`` means the orchestrator will NEVER call
``execute()`` without the Telegram user first tapping ✅ Approve.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from email.message import EmailMessage
import os
from pathlib import Path
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

import config
from config import GMAIL_CREDENTIALS_PATH, GMAIL_SENDER_EMAIL, GMAIL_TOKEN_PATH
from plugins.base import PluginBase, PluginResult
from plugins.file_search import (
    compress_to_zip,
    is_outside_usual_folders,
    is_sensitive_path,
    is_soft_sensitive,
    resolve_share_target,
)

log = logging.getLogger(__name__)

# Grants send-only access. If you change scopes, delete the token file and
# re-authenticate.
_SCOPES = ["https://www.googleapis.com/auth/gmail.send"]


def resolve_recipient_email(raw_email: str) -> str:
    """Resolve recipient email address, replacing 'me', 'self', or mock placeholders with user's configured email."""
    val = (raw_email or "").strip()
    clean = val.lower().strip("<>\"'")
    default_email = getattr(config, "USER_DEFAULT_EMAIL", "") or getattr(config, "GMAIL_SENDER_EMAIL", "")
    if not clean or clean in ("me", "self", "myself"):
        return default_email or val

    if default_email:
        if clean in ("user@example.com", "friend@example.com", "me@example.com", "my_email@example.com"):
            return default_email
        if clean.endswith("@example.com") and not clean.startswith("partner@") and not clean.startswith("test@") and not clean.startswith("boss@"):
            return default_email

    return val


def is_self_email(email_str: str) -> bool:
    """Check if the recipient email corresponds to the user themselves."""
    val = (email_str or "").strip().lower()
    if not val or val in ("me", "self", "myself"):
        return True
    default = (getattr(config, "USER_DEFAULT_EMAIL", "") or getattr(config, "GMAIL_SENDER_EMAIL", "")).lower()
    if default and val == default:
        return True
    if val in ("friend@example.com", "user@example.com", "me@example.com"):
        return True
    return False


class GmailPlugin(PluginBase):
    name = "send_email"
    description = (
        "Sends an email via Gmail to a specified recipient with a subject and body."
    )
    usage_hint = "only when they clearly ask to send an email"
    parameters_schema = {
        "type": "object",
        "properties": {
            "to_email": {
                "type": "string",
                "description": "Recipient's full email address (e.g. boss@example.com)",
            },
            "subject": {
                "type": "string",
                "description": "Subject line of the email",
            },
            "body": {
                "type": "string",
                "description": "Plain-text body content of the email",
            },
            "attachment_path": {
                "type": "string",
                "description": "Optional local file path to attach to the email",
            },
        },
        "required": ["to_email", "subject", "body"],
    }
    requires_approval = True   # ← always pauses for Telegram ✅/❌

    def is_available(self) -> bool:
        """Available only if OAuth credentials or a cached token file exists."""
        creds_exists = Path(GMAIL_CREDENTIALS_PATH).exists()
        token_exists = Path(GMAIL_TOKEN_PATH).exists()
        if creds_exists or token_exists:
            return True
        self.unavailable_reason = (
            f"Gmail credentials not found at '{GMAIL_CREDENTIALS_PATH}' or '{GMAIL_TOKEN_PATH}'"
        )
        return False

    # ------------------------------------------------------------------
    # OAuth credential helper (blocking — called via asyncio.to_thread)
    # ------------------------------------------------------------------

    def _get_service(self):
        """Build and return an authenticated Gmail API service object.

        Credentials are loaded from / saved to paths defined in config.py.
        On first run this opens a local browser window on the PC for OAuth
        consent (InstalledAppFlow.run_local_server).
        """
        token_path = Path(GMAIL_TOKEN_PATH)
        creds_path = Path(GMAIL_CREDENTIALS_PATH)

        # Ensure parent directories exist
        token_path.parent.mkdir(parents=True, exist_ok=True)

        creds: Credentials | None = None

        if token_path.exists():
            creds = Credentials.from_authorized_user_file(str(token_path), _SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                log.info("Refreshing expired Gmail OAuth token…")
                creds.refresh(Request())
            else:
                if not creds_path.exists():
                    raise FileNotFoundError(
                        f"Gmail OAuth credentials not found at '{creds_path}'. "
                        "Download 'credentials.json' from Google Cloud Console → "
                        "APIs & Services → Credentials."
                    )
                log.info("Starting Gmail OAuth consent flow (browser will open)…")
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(creds_path), _SCOPES
                )
                # Opens a local browser on the PC for first-time consent
                creds = flow.run_local_server(port=0)

            # Persist token for future runs
            token_path.write_text(creds.to_json())
            log.info("Gmail token saved to %s", token_path)

        return build("gmail", "v1", credentials=creds)

    async def prepare_preview(self, kwargs: dict[str, Any]) -> str | None:
        """Build HITL preview card with resolved recipient and attachment details."""
        to_email: str = (kwargs.get("to_email") or kwargs.get("to") or "").strip()
        resolved_recipient = resolve_recipient_email(to_email)
        subject: str = kwargs.get("subject", "").strip()
        body: str = kwargs.get("body", "").strip()
        attachment_path: str | None = kwargs.get("attachment_path")

        preview_lines = [
            f"• To: {resolved_recipient or '(no recipient)'}",
            f"• Subject: {subject or '(no subject)'}",
        ]
        if body:
            preview_lines.append(f"• Body: {body[:100]}{'...' if len(body) > 100 else ''}")

        if attachment_path:
            att = Path(attachment_path) if Path(attachment_path).exists() else None
            if att is None:
                resolved_target, matches, err = await asyncio.to_thread(resolve_share_target, attachment_path)
                att = resolved_target

            if not att or not att.exists():
                preview_lines.append(f"• Attachment: {attachment_path} (not found)")
            elif is_sensitive_path(att):
                preview_lines.append(f"• Attachment: {att.name} (BLOCKED - sensitive file)")
            else:
                try:
                    if att.is_dir():
                        size_bytes = sum(f.stat().st_size for f in att.rglob("*") if f.is_file())
                    else:
                        size_bytes = att.stat().st_size
                    size_mb = size_bytes / (1024 * 1024)
                except Exception:
                    size_mb = 0.0
                preview_lines.append(f"• Attachment: {att.name} ({size_mb:.1f} MB)")
                if is_outside_usual_folders(att):
                    preview_lines.append("⚠️ Note: attachment is outside your usual folders")
                if is_soft_sensitive(att):
                    preview_lines.append("⚠️ Note: attachment name looks sensitive")

        return "\n".join(preview_lines)

    # ------------------------------------------------------------------
    # Plugin execution (called ONLY after Telegram approval)
    # ------------------------------------------------------------------

    async def execute(self, **kwargs) -> PluginResult:
        to_email: str = (kwargs.get("to_email") or kwargs.get("to") or "").strip()
        to_email = resolve_recipient_email(to_email)
        subject: str  = kwargs.get("subject",  "").strip()
        body: str     = kwargs.get("body",     "").strip()
        attachment_path: str | None = kwargs.get("attachment_path")

        # Basic validation — the LLM should always pass these, but be safe
        missing = [f for f, v in [("to_email", to_email), ("subject", subject), ("body", body)] if not v]
        if missing:
            return PluginResult(
                success=False,
                output=f"⚠️ Missing required fields: {', '.join(missing)}",
                error=f"Missing fields: {missing}",
            )

        send_path: Path | None = None
        is_temp_zip = False
        original_name = ""
        size_mb = 0.0

        if attachment_path:
            att = Path(attachment_path) if Path(attachment_path).exists() else None
            if att is None:
                resolved_target, matches, err = await asyncio.to_thread(resolve_share_target, attachment_path)
                att = resolved_target

            if not att or not att.exists():
                return PluginResult(
                    success=False,
                    output=f"⚠️ Attachment file not found: `{attachment_path}`",
                    error=f"Attachment not found: {attachment_path}",
                )

            if is_sensitive_path(att):
                return PluginResult(
                    success=False,
                    output=f"⚠️ Access denied: '{att.name}' is a protected or sensitive file.",
                    error="Sensitive file access blocked",
                )

            original_name = att.name
            send_path = att

            if send_path.is_dir():
                tmp_shares_dir = Path(config.DATA_DIR) / "tmp_shares"
                tmp_shares_dir.mkdir(parents=True, exist_ok=True)
                try:
                    send_path = await asyncio.to_thread(compress_to_zip, send_path, tmp_shares_dir)
                    is_temp_zip = True
                except Exception as exc:
                    return PluginResult(
                        success=False,
                        output=f"⚠️ Failed to compress directory '{original_name}': {exc}",
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
                    output=f"⚠️ Could not check attachment size: {exc}",
                    error=str(exc),
                )

            # Check size limits
            if size_mb > 25.0:
                is_self = is_self_email(to_email)
                if is_self and size_mb <= 50.0:
                    direct_reply = f"📧 '{original_name}' ({size_mb:.1f} MB) exceeds Gmail's 25 MB limit, so I sent it here on Telegram instead."
                    return PluginResult(
                        success=True,
                        output=direct_reply,
                        data={
                            "send_files": [{
                                "path": str(send_path),
                                "filename": send_path.name,
                                "kind": "document",
                                "cleanup": is_temp_zip,
                            }],
                            "direct_reply": direct_reply,
                        },
                    )

                if is_temp_zip:
                    send_path.unlink(missing_ok=True)
                limit_msg = f"⚠️ Attachment '{original_name}' is {size_mb:.1f} MB (Gmail attachment limit is 25 MB)."
                return PluginResult(
                    success=False,
                    output=limit_msg,
                    error="Attachment exceeds Gmail 25 MB limit",
                    data={"direct_reply": limit_msg},
                )

        try:
            # Build + send in a thread (googleapiclient is blocking I/O)
            result = await asyncio.to_thread(
                self._build_and_send,
                to_email,
                subject,
                body,
                str(send_path) if send_path else None,
            )
            return result
        except FileNotFoundError as exc:
            log.error("Gmail credentials missing: %s", exc)
            return PluginResult(
                success=False,
                output=(
                    "⚠️ *Gmail is not configured yet.*\n\n"
                    f"`{exc}`\n\n"
                    "Place your Google Cloud OAuth client secret at `credentials/gmail_oauth.json`."
                ),
                error=str(exc),
            )
        except Exception as exc:   # noqa: BLE001
            log.exception("Gmail API error: %s", exc)
            return PluginResult(
                success=False,
                output=f"⚠️ Gmail API error: `{exc}`",
                error=str(exc),
            )
        finally:
            if is_temp_zip and send_path:
                send_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Blocking helper (runs inside asyncio.to_thread)
    # ------------------------------------------------------------------

    def _build_and_send(
        self,
        to_email: str,
        subject: str,
        body: str,
        attachment_path: str | None = None,
    ) -> PluginResult:
        if attachment_path:
            import mimetypes

            att = Path(attachment_path)
            if not att.exists():
                return PluginResult(
                    success=False,
                    output=f"⚠️ Attachment file not found: `{attachment_path}`",
                    error=f"Attachment not found: {attachment_path}",
                )
            size_mb = os.path.getsize(str(att)) / (1024 * 1024)
            if size_mb > 25.0:
                return PluginResult(
                    success=False,
                    output=f"⚠️ Attachment '{att.name}' is {size_mb:.1f} MB (Gmail attachment limit is 25 MB).",
                    error="Attachment exceeds Gmail 25 MB limit",
                )

        service = self._get_service()

        msg = EmailMessage()
        msg.set_content(body)
        msg["To"] = to_email
        msg["Subject"] = subject
        # From header is required by Gmail API
        if GMAIL_SENDER_EMAIL:
            msg["From"] = GMAIL_SENDER_EMAIL

        if attachment_path:
            ctype, encoding = mimetypes.guess_type(str(att))
            if ctype is None or encoding is not None:
                ctype = "application/octet-stream"
            maintype, subtype = ctype.split("/", 1)
            msg.add_attachment(
                att.read_bytes(),
                maintype=maintype,
                subtype=subtype,
                filename=att.name,
            )

        encoded = base64.urlsafe_b64encode(msg.as_bytes()).decode()
        service.users().messages().send(
            userId="me", body={"raw": encoded}
        ).execute()

        log.info("Email sent to %s | subject: %r (attachment: %s)", to_email, subject, attachment_path)
        att_line = f"\n📎 *Attachment:* `{Path(attachment_path).name}`" if attachment_path else ""
        direct_reply = f"📧 Email sent to {to_email} (Subject: '{subject}')."
        data: dict[str, Any] = {
            "to": to_email,
            "subject": subject,
            "direct_reply": direct_reply,
        }
        if attachment_path:
            data["attachment"] = str(attachment_path)
        return PluginResult(
            success=True,
            output=(
                f"✅ Email sent successfully!\n\n"
                f"📧 *To:* `{to_email}`\n"
                f"📌 *Subject:* {subject}{att_line}"
            ),
            data=data,
        )
