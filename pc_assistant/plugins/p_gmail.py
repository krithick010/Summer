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
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from config import GMAIL_CREDENTIALS_PATH, GMAIL_SENDER_EMAIL, GMAIL_TOKEN_PATH
from plugins.base import PluginBase, PluginResult

log = logging.getLogger(__name__)

# Grants send-only access. If you change scopes, delete the token file and
# re-authenticate.
_SCOPES = ["https://www.googleapis.com/auth/gmail.send"]


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

    # ------------------------------------------------------------------
    # Plugin execution (called ONLY after Telegram approval)
    # ------------------------------------------------------------------

    async def execute(self, **kwargs) -> PluginResult:
        to_email: str = (kwargs.get("to_email") or kwargs.get("to") or "").strip()
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

        try:
            # Build + send in a thread (googleapiclient is blocking I/O)
            result = await asyncio.to_thread(
                self._build_and_send, to_email, subject, body, attachment_path
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
            import os

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
        data: dict[str, Any] = {"to": to_email, "subject": subject}
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
