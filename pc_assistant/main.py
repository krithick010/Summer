"""
main.py
-------
FastAPI application and Telegram Bot entry point.

Startup modes
~~~~~~~~~~~~~
* **Polling (default / development):** python-telegram-bot's
  ``Application.run_polling()`` runs in a background thread; FastAPI is
  served by uvicorn in the main thread.
* **Webhook (production):**  Set ``WEBHOOK_URL`` in the environment.
  FastAPI exposes ``POST /webhook/{token}`` and feeds updates to the
  bot application.

Security
~~~~~~~~
* Only Telegram user IDs listed in ``ALLOWED_USER_IDS`` can interact with
  the bot.  Every handler short-circuits if the sender is not in the allow
  list, or if the list is empty (open-to-anyone mode is not supported in
  production — set the env var).
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    LinkPreviewOptions,
    Update,
)
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

import config

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
# Silence httpx request logging so URLs containing the bot token aren't printed
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
log = logging.getLogger("pc_assistant")

# ---------------------------------------------------------------------------
# Lazy imports (populated at startup so the orchestrator is available)
# ---------------------------------------------------------------------------
# We import orchestrator here rather than at module level to avoid circular
# imports during plugin auto-discovery that happens at import time.
_orchestrator: Any = None  # set in lifespan()


# ===========================================================================
# Access-control helper
# ===========================================================================


def _is_allowed(user_id: int) -> bool:
    """Return True if the user is authorised to use the assistant."""
    if not config.ALLOWED_USER_IDS:
        return False
    return user_id in config.ALLOWED_USER_IDS


def _deny_message() -> str:
    return (
        "⛔ You are not authorised to use this assistant.\n"
        "Ask the PC owner to add your Telegram user ID to `ALLOWED_USER_IDS`."
    )


_NO_LINK_PREVIEW = LinkPreviewOptions(is_disabled=True)


async def _safe_send_message(
    bot: Any,
    chat_id: int,
    text: str,
    reply_markup: Any = None,
) -> Any:
    """Send a message using Markdown, falling back to plain text if Telegram fails parsing entities."""
    try:
        return await bot.send_message(
            chat_id=chat_id,
            text=text,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=reply_markup,
            link_preview_options=_NO_LINK_PREVIEW,
        )
    except Exception:
        return await bot.send_message(
            chat_id=chat_id,
            text=text,
            reply_markup=reply_markup,
            link_preview_options=_NO_LINK_PREVIEW,
        )


async def _safe_reply_text(
    message: Any,
    text: str,
    reply_markup: Any = None,
) -> Any:
    """Reply using Markdown, falling back to plain text if Telegram fails parsing entities."""
    try:
        return await message.reply_text(
            text=text,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=reply_markup,
            link_preview_options=_NO_LINK_PREVIEW,
        )
    except Exception:
        return await message.reply_text(
            text=text,
            reply_markup=reply_markup,
            link_preview_options=_NO_LINK_PREVIEW,
        )


# ===========================================================================
# Telegram command & message handlers
# ===========================================================================


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/start — Greeting message."""
    user = update.effective_user
    if not _is_allowed(user.id):
        await update.message.reply_text(_deny_message())
        return

    log.info("/start from user %s (%s)", user.id, user.full_name)
    await update.message.reply_text(
        f"👋 Hello *{user.first_name}*!\n\n"
        "I'm your *Local PC Assistant* — running fully offline on your machine.\n\n"
        "Just send me a message like:\n"
        "• _Show me the system stats_\n"
        "• _Draft an email to boss@example.com about the quarterly report_\n\n"
        "Use /help to see all commands.",
        parse_mode=ParseMode.MARKDOWN,
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/help — Available commands."""
    if not _is_allowed(update.effective_user.id):
        await update.message.reply_text(_deny_message())
        return

    await update.message.reply_text(
        "*Available commands*\n\n"
        "/start  — Welcome message\n"
        "/help   — This help text\n"
        "/clear  — Reset conversation history\n"
        "/status — Quick PC health check\n"
        "/cancel — Cancel any pending approval\n\n"
        "Or just *type a natural-language request* and I'll figure out what to do! 🤖",
        parse_mode=ParseMode.MARKDOWN,
    )


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/clear — Reset conversation history."""
    user_id = update.effective_user.id
    if not _is_allowed(user_id):
        await update.message.reply_text(_deny_message())
        return

    if _orchestrator:
        _orchestrator.history.clear(user_id)
    log.info("/clear history for user %s", user_id)
    await update.message.reply_text("Fresh start.")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/status — Shortcut: immediately run the system-telemetry plugin."""
    if not _is_allowed(update.effective_user.id):
        await update.message.reply_text(_deny_message())
        return

    log.info("/status from user %s", update.effective_user.id)
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action=ChatAction.TYPING,
    )

    result = await _orchestrator.run_tool(
        tool_name="system_info",
        kwargs={},
        chat_id=update.effective_chat.id,
    )
    await update.message.reply_text(result.output, parse_mode=ParseMode.MARKDOWN)


async def cmd_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/cancel — Cancel the user's current pending HITL approval."""
    user_id = update.effective_user.id
    if not _is_allowed(user_id):
        await update.message.reply_text(_deny_message())
        return

    from core.state_manager import state_manager  # noqa: PLC0415

    cancelled = state_manager.cancel_pending(user_id)
    if cancelled:
        await update.message.reply_text("✅ Pending action cancelled.")
    else:
        await update.message.reply_text("ℹ️ No pending action to cancel.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Route every plain text message through the LLM orchestrator."""
    user = update.effective_user
    if not _is_allowed(user.id):
        await update.message.reply_text(_deny_message())
        return

    user_text = update.message.text.strip()
    if not user_text:
        return

    log.info("Message from %s: %r", user.id, user_text[:80])

    # Show a typing action in Telegram status while the LLM thinks
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action=ChatAction.TYPING,
    )

    try:
        response = await _orchestrator.handle_user_message(
            user_id=user.id,
            chat_id=update.effective_chat.id,
            message=user_text,
        )
    except httpx.HTTPError:
        await update.message.reply_text("Can't reach Ollama - is `ollama serve` running?")
        return
    except Exception as exc:  # noqa: BLE001
        log.exception("Orchestrator error: %s", exc)
        await update.message.reply_text(
            f"⚠️ An internal error occurred. Details: `{exc}`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    # ``response`` can be:
    #   • A plain string  → send directly
    #   • A dict with ``text`` + ``reply_markup`` keys → HITL approval prompt
    if isinstance(response, str):
        await _safe_reply_text(update.message, response)
    elif isinstance(response, dict):
        await _safe_reply_text(
            update.message,
            response["text"],
            reply_markup=response.get("reply_markup"),
        )


async def handle_callback_query(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle Approve / Cancel button presses from HITL prompts."""
    query = update.callback_query
    await query.answer()  # stop the spinner on the button

    user_id = query.from_user.id
    if not _is_allowed(user_id):
        await query.edit_message_text(_deny_message())
        return

    # Callback data format:  "hitl:<action>:<approval_id>"
    # e.g.  "hitl:approve:abc123"  or  "hitl:cancel:abc123"
    data: str = query.data or ""
    if not data.startswith("hitl:"):
        log.warning("Unexpected callback data: %r", data)
        return

    _, action, approval_id = data.split(":", 2)
    log.info("HITL callback: action=%s approval_id=%s user=%s", action, approval_id, user_id)

    from core.state_manager import state_manager  # noqa: PLC0415

    sm = _orchestrator.state_manager if _orchestrator else state_manager
    pending = sm.resolve(approval_id)
    if pending is None:
        await query.edit_message_text(
            "⏰ This action has already been resolved or timed out."
        )
        return

    if action == "approve":
        await query.edit_message_text("✅ Approved! Executing…")
        try:
            # 1. Execute plugin directly
            plugin = _orchestrator.registry.get(pending["tool_name"]) if _orchestrator else None
            if plugin is not None:
                result = await plugin.execute(**pending["kwargs"])
            elif _orchestrator:
                result = await _orchestrator.run_tool(
                    tool_name=pending["tool_name"],
                    kwargs=pending["kwargs"],
                    chat_id=query.message.chat_id,
                )
            else:
                result = None

            result_output = result.output if result else "Action completed."

            # 2. Format into standard native tool response message
            tool_msg: dict[str, Any] = {
                "role": "tool",
                "content": result_output,
            }

            # Map tool_call_id if present in previous assistant turn
            if _orchestrator:
                hist = _orchestrator.history.get(user_id)
                last_assistant = next(
                    (m for m in reversed(hist) if m.get("role") == "assistant"),
                    None,
                )
                if last_assistant and "tool_calls" in last_assistant:
                    calls = last_assistant.get("tool_calls", [])
                    if calls and isinstance(calls, list):
                        call_id = calls[0].get("id")
                        if call_id:
                            tool_msg["tool_call_id"] = call_id

                # 3. Append tool response message to user history
                _orchestrator.history.add(user_id, tool_msg)

                # 4. Resume the LLM conversational loop
                final_text = await _orchestrator.resume_tool_loop(user_id)
            else:
                final_text = result_output

            # 5. Send final conversational response
            await _safe_send_message(
                context.bot,
                chat_id=query.message.chat_id,
                text=final_text,
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("Error executing approved tool: %s", exc)
            await _safe_send_message(
                context.bot,
                chat_id=query.message.chat_id,
                text=f"⚠️ Execution failed: {exc}",
            )
    else:
        sm.cancel_by_id(approval_id)
        await query.edit_message_text("❌ Action cancelled.")


# ===========================================================================
# HITL inline-keyboard helper (called by orchestrator)
# ===========================================================================


def build_hitl_prompt(
    approval_id: str,
    preview_text: str,
    tool_name: str,
) -> dict[str, Any]:
    """Return a dict with ``text`` and ``reply_markup`` for a HITL prompt."""
    keyboard = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "✅ Approve", callback_data=f"hitl:approve:{approval_id}"
                ),
                InlineKeyboardButton(
                    "❌ Cancel", callback_data=f"hitl:cancel:{approval_id}"
                ),
            ]
        ]
    )
    return {
        "text": (
            f"🔐 *Action requires your approval*\n\n"
            f"*Tool:* `{tool_name}`\n\n"
            f"{preview_text}\n\n"
            "_Tap Approve or Cancel below._"
        ),
        "reply_markup": keyboard,
    }


# ===========================================================================
# Telegram Application (singleton, built once)
# ===========================================================================

_telegram_app: Application | None = None


def _build_telegram_app() -> Application:
    app = (
        Application.builder()
        .token(config.TELEGRAM_BOT_TOKEN)
        .build()
    )

    # Register handlers
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("cancel", cmd_cancel))
    app.add_handler(CallbackQueryHandler(handle_callback_query))
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )

    return app


# ===========================================================================
# FastAPI lifespan — initialise orchestrator and start polling/webhook
# ===========================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN001
    """Startup / shutdown lifecycle for FastAPI + Telegram."""
    global _orchestrator, _telegram_app  # noqa: PLW0603

    # -- Startup -----------------------------------------------------------
    log.info("Starting Local PC Assistant…")
    if not config.ALLOWED_USER_IDS:
        log.error(
            "ALLOWED_USER_IDS is empty: the bot will ignore everyone. "
            "Add your Telegram user id to .env (message @userinfobot to find it)."
        )

    # Import orchestrator (triggers plugin auto-discovery)
    from core.orchestrator import Orchestrator  # noqa: PLC0415

    _orchestrator = Orchestrator()
    _orchestrator.set_hitl_builder(build_hitl_prompt)
    log.info("Orchestrator ready. Loaded plugins: %s", list(_orchestrator.registry.keys()))

    _telegram_app = _build_telegram_app()
    await _telegram_app.initialize()

    if config.WEBHOOK_URL:
        # Production webhook mode
        webhook_path = f"/webhook/{config.TELEGRAM_BOT_TOKEN}"
        webhook_url = f"{config.WEBHOOK_URL}{webhook_path}"
        await _telegram_app.bot.set_webhook(webhook_url)
        await _telegram_app.start()
        log.info("Webhook registered at %s/webhook/<token-hidden>", config.WEBHOOK_URL)
    else:
        # Development polling mode — run directly on FastAPI's asyncio event loop.
        log.info("No WEBHOOK_URL set — using long-polling mode.")
        await _telegram_app.bot.delete_webhook(drop_pending_updates=True)
        await _telegram_app.start()
        await _telegram_app.updater.start_polling()

    log.info("PC Assistant is live 🚀")
    yield

    # -- Shutdown ----------------------------------------------------------
    log.info("Shutting down…")
    from plugins.web_scraper import close_browser

    await close_browser()
    if _telegram_app:
        if _telegram_app.updater and _telegram_app.updater.running:
            await _telegram_app.updater.stop()
        if _telegram_app.running:
            await _telegram_app.stop()
        await _telegram_app.shutdown()
    log.info("Goodbye.")


# ===========================================================================
# FastAPI Application
# ===========================================================================

app = FastAPI(
    title="Local PC Assistant",
    description="Phone-to-PC control via Telegram + local LLM (Ollama).",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/health", tags=["meta"])
async def health() -> dict[str, str]:
    """Simple liveness probe."""
    return {"status": "ok", "version": "1.0.0"}


# ---------------------------------------------------------------------------
# Webhook endpoint (only used when WEBHOOK_URL is set)
# ---------------------------------------------------------------------------


@app.post("/webhook/{token}", tags=["telegram"])
async def telegram_webhook(token: str, request: Request) -> Response:
    """Receive Telegram updates via webhook."""
    if token != config.TELEGRAM_BOT_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")

    if _telegram_app is None:
        raise HTTPException(status_code=503, detail="Bot not ready")

    body = await request.json()
    update = Update.de_json(body, _telegram_app.bot)
    await _telegram_app.process_update(update)
    return Response(status_code=200)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
        reload=False,
        log_level=config.LOG_LEVEL.lower(),
    )

