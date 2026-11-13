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
import html
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import telegram.error
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
    parse_mode: Any = None,
) -> Any:
    """Send a message, defaulting to plain text (no parse_mode) to protect underscores."""
    send_kwargs: dict[str, Any] = {
        "chat_id": chat_id,
        "text": text,
        "reply_markup": reply_markup,
        "link_preview_options": _NO_LINK_PREVIEW,
    }
    if parse_mode is not None:
        send_kwargs["parse_mode"] = parse_mode

    try:
        coro = bot.send_message(**send_kwargs)
        if asyncio.iscoroutine(coro):
            return await coro
        return coro
    except Exception:
        # Fall back to plain text if formatting failed
        send_kwargs.pop("parse_mode", None)
        try:
            coro = bot.send_message(**send_kwargs)
            if asyncio.iscoroutine(coro):
                return await coro
            return coro
        except Exception:
            return None


async def _safe_reply_text(
    message: Any,
    text: str,
    reply_markup: Any = None,
    parse_mode: Any = None,
) -> Any:
    """Reply using specified parse_mode (or plain text by default)."""
    reply_kwargs: dict[str, Any] = {
        "text": text,
        "reply_markup": reply_markup,
        "link_preview_options": _NO_LINK_PREVIEW,
    }
    if parse_mode is not None:
        reply_kwargs["parse_mode"] = parse_mode

    try:
        coro = message.reply_text(**reply_kwargs)
        if asyncio.iscoroutine(coro):
            return await coro
        return coro
    except Exception:
        reply_kwargs.pop("parse_mode", None)
        try:
            coro = message.reply_text(**reply_kwargs)
            if asyncio.iscoroutine(coro):
                return await coro
            return coro
        except Exception:
            return None


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


async def cmd_index(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/index — Report status of the background file index."""
    user_id = update.effective_user.id
    if not _is_allowed(user_id):
        await update.message.reply_text(_deny_message())
        return

    from plugins.file_index import file_indexer  # noqa: PLC0415
    msg = file_indexer.format_status_message()
    await update.message.reply_text(msg)


async def cmd_reindex(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/reindex — Trigger an immediate background file crawl."""
    user_id = update.effective_user.id
    if not _is_allowed(user_id):
        await update.message.reply_text(_deny_message())
        return

    from plugins.file_index import file_indexer  # noqa: PLC0415
    file_indexer.reindex()
    await update.message.reply_text("Reindex started.")


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
    #   • ChoicesReply → inline keyboard choices + plain text
    #   • A plain string  → send directly as plain text
    #   • A dict with ``text`` + ``reply_markup`` keys → HITL approval prompt
    from core.orchestrator import ChoicesReply  # noqa: PLC0415

    if isinstance(response, ChoicesReply):
        keyboard = InlineKeyboardMarkup(
            [
                [InlineKeyboardButton(label, callback_data=f"pick:{response.token}:{idx}")]
                for idx, label in enumerate(response.labels)
            ]
        )
        await update.message.reply_text(
            text=response.text,
            reply_markup=keyboard,
        )
    elif isinstance(response, str):
        await update.message.reply_text(text=response)
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

    # Callback data formats:
    #   "pick:<token>:<index>" -> user picked a file/choice option
    #   "hitl:<action>:<approval_id>" -> e.g. "hitl:approve:abc123" or "hitl:cancel:abc123"
    data: str = query.data or ""
    if data.startswith("pick:"):
        _, token, idx_str = data.split(":", 2)
        try:
            index = int(idx_str)
        except ValueError:
            await query.edit_message_text("Invalid selection.")
            return

        from core.choices import choice_store  # noqa: PLC0415

        choice = choice_store.get(token, user_id, index)
        if choice is None:
            await query.edit_message_text("⏰ This choice has expired or is invalid.")
            return

        tool_name = choice.get("tool", "")
        kwargs = choice.get("kwargs", {})
        if not _orchestrator:
            await query.edit_message_text("Assistant is not ready.")
            return

        plugin = _orchestrator.registry.get(tool_name) if _orchestrator else None
        if plugin is not None and not getattr(plugin, "requires_approval", True):
            # Tool does not require HITL approval (e.g. list_folder) -> execute inline!
            result = await plugin.execute(**kwargs)
            choices_list = (result.data or {}).get("choices") if result.data else None
            if choices_list:
                new_token = choice_store.register(user_id, choices_list)
                keyboard = InlineKeyboardMarkup(
                    [
                        [InlineKeyboardButton(c["label"], callback_data=f"pick:{new_token}:{i}")]
                        for i, c in enumerate(choices_list)
                    ]
                )
                try:
                    await query.edit_message_text(result.output, reply_markup=keyboard)
                except Exception:
                    pass
            else:
                try:
                    await query.edit_message_text(result.output)
                except Exception:
                    pass
            return

        approval_payload = await _orchestrator.request_tool_approval(user_id, tool_name, kwargs)
        if isinstance(approval_payload, dict) and "reply_markup" in approval_payload:
            await query.edit_message_text(
                approval_payload["text"],
                reply_markup=approval_payload["reply_markup"],
                parse_mode=approval_payload.get("parse_mode", ParseMode.HTML),
            )
        else:
            txt = (
                approval_payload.get("text", "Action pending.")
                if isinstance(approval_payload, dict)
                else str(approval_payload)
            )
            await query.edit_message_text(txt)
        return

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
            exec_kwargs = dict(pending.get("kwargs") or {})
            if pending.get("tool_name") == "share_file":
                exec_kwargs["user_id"] = user_id
            plugin = _orchestrator.registry.get(pending["tool_name"]) if _orchestrator else None
            if plugin is not None:
                result = await plugin.execute(**exec_kwargs)
            elif _orchestrator:
                result = await _orchestrator.run_tool(
                    tool_name=pending["tool_name"],
                    kwargs=exec_kwargs,
                    chat_id=query.message.chat_id,
                )
            else:
                result = None

            # 2. Upload files if the approved plugin returned send_files
            timed_out_part: int | None = None
            if (
                result
                and result.success
                and isinstance(result.data, dict)
                and "send_files" in result.data
            ):
                send_files = result.data.get("send_files") or []
                total_parts = len(send_files)
                upload_timeout = config.FILE_UPLOAD_TIMEOUT_S

                for idx, item in enumerate(send_files, start=1):
                    f_path = Path(item["path"])
                    kind = item.get("kind", "document")
                    filename = item.get("filename") or f_path.name
                    try:
                        if f_path.exists():
                            # Send chat action before each upload
                            action_type = (
                                ChatAction.UPLOAD_PHOTO if kind == "photo" else ChatAction.UPLOAD_DOCUMENT
                            )
                            try:
                                action_res = context.bot.send_chat_action(
                                    chat_id=query.message.chat_id,
                                    action=action_type,
                                )
                                if asyncio.iscoroutine(action_res):
                                    await action_res
                            except Exception:
                                pass

                            if total_parts > 1:
                                await _safe_send_message(
                                    context.bot,
                                    chat_id=query.message.chat_id,
                                    text=f"Sending part {idx} of {total_parts}...",
                                )

                            with open(f_path, "rb") as f:
                                if kind == "photo":
                                    await context.bot.send_photo(
                                        chat_id=query.message.chat_id,
                                        photo=f,
                                        caption=filename,
                                        read_timeout=upload_timeout,
                                        write_timeout=upload_timeout,
                                        connect_timeout=upload_timeout,
                                        pool_timeout=upload_timeout,
                                    )
                                else:
                                    await context.bot.send_document(
                                        chat_id=query.message.chat_id,
                                        document=f,
                                        filename=filename,
                                        read_timeout=upload_timeout,
                                        write_timeout=upload_timeout,
                                        connect_timeout=upload_timeout,
                                        pool_timeout=upload_timeout,
                                    )
                    except telegram.error.TimedOut:
                        timed_out_part = idx
                        log.warning("Telegram upload timed out on part %d (%s)", idx, filename)
                        await _safe_send_message(
                            context.bot,
                            chat_id=query.message.chat_id,
                            text=f"Telegram timed out on part {idx} - it may still arrive. Check before asking again.",
                        )
                        break
                    finally:
                        if item.get("cleanup"):
                            f_path.unlink(missing_ok=True)

                if timed_out_part is not None:
                    # Clean up remaining un-sent parts
                    for rem_item in send_files[timed_out_part:]:
                        if rem_item.get("cleanup"):
                            Path(rem_item["path"]).unlink(missing_ok=True)
                    return

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

                # 4. Use direct_reply if present, else resume the LLM conversational loop
                if (
                    result
                    and result.success
                    and isinstance(result.data, dict)
                    and "direct_reply" in result.data
                ):
                    final_text = str(result.data["direct_reply"])
                    _orchestrator.history.add(user_id, "assistant", final_text)
                else:
                    final_text = await _orchestrator.resume_tool_loop(user_id)
            else:
                final_text = (
                    str(result.data["direct_reply"])
                    if result and isinstance(result.data, dict) and "direct_reply" in result.data
                    else result_output
                )

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
        if pending and pending.get("tool_name") == "share_file":
            kw = pending.get("kwargs") or {}
            target_p = kw.get("path") or kw.get("query") or "unknown"
            deliv = kw.get("delivery", "telegram")
            from plugins.file_search import log_file_audit  # noqa: PLC0415
            log_file_audit(
                user_id=user_id,
                action="share",
                path=target_p,
                size="0",
                route=deliv,
                status="cancelled",
            )
        await query.edit_message_text("❌ Action cancelled.")


# ===========================================================================
# HITL inline-keyboard helper (called by orchestrator)
# ===========================================================================


def build_hitl_prompt(
    approval_id: str,
    preview_text: str,
    tool_name: str,
) -> dict[str, Any]:
    """Return a dict with ``text``, ``reply_markup``, and ``parse_mode=HTML`` for a HITL prompt."""
    escaped_tool = html.escape(tool_name)
    escaped_preview = html.escape(preview_text)
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
            f"🔐 <b>Action requires your approval</b>\n\n"
            f"<b>Tool:</b> <code>{escaped_tool}</code>\n\n"
            f"{escaped_preview}\n\n"
            "<i>Tap Approve or Cancel below.</i>"
        ),
        "reply_markup": keyboard,
        "parse_mode": ParseMode.HTML,
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
    app.add_handler(CommandHandler("index", cmd_index))
    app.add_handler(CommandHandler("reindex", cmd_reindex))
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
    from plugins.file_search import cleanup_stale_tmp_shares  # noqa: PLC0415

    cleaned_count = cleanup_stale_tmp_shares(max_age_s=86400.0)
    if cleaned_count:
        log.info("Startup cleanup: removed %d stale temp share file(s)", cleaned_count)

    _orchestrator = Orchestrator()
    _orchestrator.set_hitl_builder(build_hitl_prompt)
    log.info("Orchestrator ready. Loaded plugins: %s", list(_orchestrator.registry.keys()))

    from plugins.file_index import file_indexer  # noqa: PLC0415
    file_indexer.start_background_worker()

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
    from plugins.file_index import file_indexer  # noqa: PLC0415
    file_indexer.stop()

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

