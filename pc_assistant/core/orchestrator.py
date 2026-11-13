"""
core/orchestrator.py
--------------------
Central routing hub for PC Assistant ("Summer").
Uses single-call conversational flow with Ollama's native tool calling.
No classifier layers, no keyword guardrails, no fast-paths.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
import importlib
import inspect
import logging
import pkgutil
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Callable

import config
from config import ASSISTANT_NAME, EXTRACTOR_ENABLED, MEMORY_TOP_K, USER_NAME
from core.choices import ChoiceStore, choice_store as default_choice_store
from core.history import History
from core.llm_engine import LLMEngine
from core.memory import MemoryManager
from core.state_manager import StateManager, state_manager as default_state_manager
import plugins
from plugins.base import PluginBase, PluginResult
from plugins.file_search import resolve_share_target

log = logging.getLogger(__name__)


@dataclass
class ChoicesReply:
    text: str
    token: str
    labels: list[str]


SYSTEM_TEMPLATE = """You are {assistant_name}, {user_name}'s personal assistant running privately on their own PC.
You are chatting with {user_name} directly inside Telegram right now.
Talk like a close friend texting: casual, short (1-2 sentences), natural. No corporate phrases, no "As an AI".
- When they ask for something you know, just say it: "Her number's 555-0123." No lead-in.
- When they tell you a fact, a quick ack is enough: "Noted.", "Saved that.", "Got it, new email's in."
- Never say "do you need anything else", "what do you want to do with that" or "let me know if...". Don't end with a question unless you truly need an answer.
- If you don't know something, say so in a few words: "Nope, don't have that. Wanna give it to me?"

{tools_section}

Facts they tell you are saved automatically, so just acknowledge them. Don't repeat the whole fact back unless it was a correction.

WHAT YOU KNOW ABOUT {user_name} (their own private data, share it freely when asked; newest first):
{memories}

If something isn't listed above or in the chat, say you don't know. Never make things up.{location_line}"""


def _format_tools_section(plugins_arg: Iterable[PluginBase] | Mapping[str, PluginBase] | None) -> str:
    """Render the TOOLS prompt section from registered, available chat plugins."""
    plugin_lines: list[str] = []
    if plugins_arg is not None:
        plugin_iter = plugins_arg.values() if isinstance(plugins_arg, Mapping) else plugins_arg
        for p in plugin_iter:
            if p.name == "save_memory":
                continue
            is_avail = getattr(p, "is_available", None)
            if callable(is_avail) and not is_avail():
                continue
            hint = (getattr(p, "usage_hint", "") or getattr(p, "description", "") or "").strip()
            plugin_lines.append(f"- {p.name}: use for {hint}")

    lines = [
        "TOOLS (use them, never guess):",
        *plugin_lines,
        "- When they ask to share, send, or get a file or folder, call share_file (or find_files if searching). The file will be sent right here in this Telegram chat.",
        "- Never answer weather, news, lookups or PC stats from your own knowledge. If you did not call the tool, you do not have the answer.",
        "- Never say you found, opened, sent or emailed anything unless a tool result says so. You cannot open files on this PC.",
        "- Call the tool again every time, even if you answered something similar earlier.",
        "- After a tool returns, report its result faithfully. If it failed, say so in one short line. Don't suggest other places or topics.",
        "- For greetings, small talk and anything about facts he told you, don't call tools.",
        "- When a tool returns a list (like headlines), pass it along as a short list instead of summarizing it away.",
    ]
    return "\n".join(lines)


def build_system_prompt(
    memories: str = "",
    assistant_name: str | None = None,
    user_name: str | None = None,
    user_location: str | None = None,
    plugins: Iterable[PluginBase] | Mapping[str, PluginBase] | None = None,
) -> str:
    """Build the system prompt formatting names, memories, active tools, and location."""
    resolved_user_name = user_name or USER_NAME
    resolved_location = (
        user_location if user_location is not None else config.USER_LOCATION
    ).strip()
    location_line = (
        f"\n{resolved_user_name} lives in {resolved_location}. "
        "For weather, news or anything local, assume this place unless they name another."
        if resolved_location
        else ""
    )
    return SYSTEM_TEMPLATE.format(
        assistant_name=assistant_name or ASSISTANT_NAME,
        user_name=resolved_user_name,
        memories=memories or "(nothing yet)",
        tools_section=_format_tools_section(plugins),
        location_line=location_line,
    )


# Backward compatibility alias
SYSTEM = SYSTEM_TEMPLATE


class Orchestrator:
    """Dispatches messages through LLM /api/chat with native tool calling."""

    def __init__(
        self,
        llm: LLMEngine | None = None,
        state_manager: StateManager | None = None,
        memory: MemoryManager | None = None,
        choice_store: ChoiceStore | None = None,
    ) -> None:
        self.llm = llm or LLMEngine()
        self.state_manager = state_manager or default_state_manager
        self.memory = memory or MemoryManager()
        self.choice_store = choice_store or default_choice_store
        self.history = History()
        self.last_file_path: dict[int, str] = {}
        self.registry: dict[str, PluginBase] = {}
        self._hitl_builder: Callable[..., dict[str, Any]] | None = None
        self._bg_tasks: set[asyncio.Task[Any]] = set()
        self._load_plugins()

    def set_hitl_builder(self, fn: Callable[..., dict[str, Any]]) -> None:
        """Register the callable from main.py that builds Telegram inline-keyboard dicts."""
        self._hitl_builder = fn

    def _load_plugins(self) -> None:
        """Scan pc_assistant/plugins/ for all subclasses of PluginBase."""
        for _, module_name, is_pkg in pkgutil.iter_modules(plugins.__path__):
            if is_pkg or not module_name.startswith("p_"):
                continue

            full_name = f"plugins.{module_name}"
            try:
                module = importlib.import_module(full_name)
            except Exception as exc:  # noqa: BLE001
                log.error("Failed to import plugin module %s: %s", module_name, exc)
                continue

            for _, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, PluginBase) and obj is not PluginBase:
                    try:
                        instance = obj()
                        if not instance.is_available():
                            reason = instance.unavailable_reason or "unavailable in current environment"
                            log.info("Plugin '%s' disabled: %s", instance.name, reason)
                            continue
                        self.registry[instance.name] = instance
                        log.info("Registered plugin: %r", instance)
                    except Exception as exc:  # noqa: BLE001
                        log.error("Failed to instantiate plugin %s: %s", obj.__name__, exc)

    def _chat_tools(self) -> list[dict[str, Any]]:
        """Build Ollama tool schemas for all active non-memory plugins."""
        return [
            {
                "type": "function",
                "function": {
                    "name": p.name,
                    "description": p.description,
                    "parameters": p.parameters_schema,
                },
            }
            for p in self.registry.values()
            if p.name != "save_memory"
        ]

    async def request_tool_approval(
        self,
        user_id: int,
        tool_name: str,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Build HITL approval action and return approval prompt payload."""
        plugin = self.registry.get(tool_name)
        if plugin is None:
            return {"text": f"Unknown tool: {tool_name}"}
        allowed = plugin.parameters_schema.get("properties", {})
        filtered_args = {k: v for k, v in kwargs.items() if k in allowed} if allowed else {}
        try:
            custom_preview = await plugin.prepare_preview(filtered_args)
        except Exception as exc:  # noqa: BLE001
            log.warning("prepare_preview failed for %s: %s", tool_name, exc)
            custom_preview = None
        preview = custom_preview or "\n".join(f"• {k}: {v}" for k, v in filtered_args.items())
        action = self.state_manager.create_action(
            tool_name=tool_name,
            kwargs=filtered_args,
            description=preview,
            user_id=user_id,
        )
        if self._hitl_builder is None:
            log.error("HITL builder not set — cannot pause for approval.")
            return {"text": "⚠️ Internal configuration error: HITL builder not registered."}
        return self._hitl_builder(
            approval_id=action.approval_id,
            preview_text=preview,
            tool_name=tool_name,
        )

    async def run_tool(
        self,
        tool_name: str,
        kwargs: dict[str, Any],
        chat_id: int,
    ) -> PluginResult:
        """Execute a plugin directly, bypassing HITL checks."""
        plugin = self.registry.get(tool_name)
        if plugin is None:
            return PluginResult(
                success=False,
                output=f"⚠️ Unknown tool: `{tool_name}`",
                error=f"Tool '{tool_name}' not found in registry.",
            )
        try:
            return await plugin.execute(**kwargs)
        except Exception as exc:  # noqa: BLE001
            log.exception("Plugin '%s' raised an exception", tool_name)
            return PluginResult(
                success=False,
                output=f"⚠️ Plugin error: `{exc}`",
                error=str(exc),
            )

    async def handle_user_message(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> str | dict[str, Any] | ChoicesReply:
        """Process incoming user message through History + RAG + Ollama tool loop."""
        if len(args) == 1:
            user_id, chat_id, message = 0, 0, str(args[0])
        elif len(args) >= 3:
            user_id, chat_id, message = int(args[0]), int(args[1]), str(args[2])
        else:
            user_id = int(kwargs.get("user_id", 0))
            chat_id = int(kwargs.get("chat_id", 0))
            message = str(kwargs.get("message", ""))

        message = message.strip()
        if not message:
            return ""

        t_msg = time.perf_counter()
        retrieval_ms = 0.0
        try:
            hist = self.history.get(user_id)

            # Split-query retrieval: short follow-ups get two queries for context
            t_ret = time.perf_counter()
            last_user = next((m.get("content", "") for m in reversed(hist) if m.get("role") == "user"), "")
            queries = [message]
            if last_user and len(message.split()) < 6:
                queries.append(last_user)

            seen: list[str] = []
            for q in queries:
                result = await self.memory.asearch_memory(q, k=MEMORY_TOP_K)
                for line in result.split("\n"):
                    if line and line not in seen:
                        seen.append(line)
                if len(seen) >= MEMORY_TOP_K:
                    break
            memories = "\n".join(seen[:MEMORY_TOP_K])
            retrieval_ms = (time.perf_counter() - t_ret) * 1000.0

            msgs: list[dict[str, Any]] = [
                {
                    "role": "system",
                    "content": build_system_prompt(memories, plugins=self.registry.values()),
                }
            ]
            # Normalise backslashes in user message sent to the LLM to prevent JSON escape sequence issues
            model_content = message.replace("\\", "/")
            msgs += hist + [{"role": "user", "content": model_content}]

            tools = self._chat_tools()

            reply = await self.llm.chat(msgs, tools=tools)
            log.info("tool_calls=%s", reply.get("tool_calls"))

            tool_trace: list[dict] = []
            for _ in range(3):  # tool loop
                calls = reply.get("tool_calls")
                if not calls:
                    break
                msgs.append(reply)
                assistant_turn = dict(reply)
                assistant_turn.setdefault("role", "assistant")
                tool_trace.append(assistant_turn)
                for call in calls:
                    name = call["function"]["name"]
                    call_args = call["function"]["arguments"] or {}
                    plugin = self.registry.get(name)
                    if plugin is None:
                        tool_msg = {"role": "tool", "tool_name": name, "content": "unknown tool"}
                        msgs.append(tool_msg)
                        tool_trace.append(tool_msg)
                        continue

                    allowed = plugin.parameters_schema.get("properties", {})
                    filtered_args = {k: v for k, v in call_args.items() if k in allowed} if allowed else {}
                    if name == "find_files" and "query" not in filtered_args and "path" in call_args:
                        filtered_args["query"] = call_args["path"]

                    if name == "share_file":
                        if "path" not in filtered_args and "query" in call_args:
                            filtered_args["path"] = call_args["query"]
                        raw_target = str(
                            filtered_args.get("path")
                            or filtered_args.get("query")
                            or call_args.get("path")
                            or call_args.get("query")
                            or ""
                        ).strip()
                        if raw_target.lower() in ("it", "this", "that", "the file", "the folder") and user_id in self.last_file_path:
                            raw_target = self.last_file_path[user_id]
                            filtered_args["path"] = raw_target

                        msg_lower = message.lower()
                        if ("mail" in msg_lower or "email" in msg_lower) and "delivery" not in filtered_args:
                            filtered_args["delivery"] = "email"

                        # Pre-approval validation
                        target_path, matches, err_msg = await asyncio.to_thread(resolve_share_target, raw_target)
                        if err_msg:
                            reply_text = str(err_msg)
                            self.history.add(user_id, "user", message)
                            self.history.add(user_id, assistant_turn)
                            self.history.add(user_id, {"role": "tool", "tool_name": name, "content": reply_text})
                            self.history.add(user_id, "assistant", reply_text)
                            if EXTRACTOR_ENABLED:
                                task = asyncio.create_task(self._remember(hist, message))
                                self._bg_tasks.add(task)
                                task.add_done_callback(self._bg_tasks.discard)
                            return reply_text

                        if not target_path and matches:
                            top_matches = matches[:5]
                            lines = []
                            choices = []
                            is_email = filtered_args.get("delivery") == "email"
                            to_email = filtered_args.get("to_email", "")
                            for idx, m in enumerate(top_matches, start=1):
                                size_mb = m.size / (1024 * 1024)
                                size_str = f"{size_mb:.1f} MB"
                                date_str = datetime.fromtimestamp(m.mtime).strftime("%Y-%m-%d")
                                folder_name = m.path.parent.name
                                lines.append(f"{idx}. {m.path.name} - {size_str} - {date_str} - in {folder_name}")
                                kw = {"path": str(m.path.resolve())}
                                if is_email:
                                    kw["delivery"] = "email"
                                    if to_email:
                                        kw["to_email"] = to_email
                                choices.append({
                                    "label": f"{idx}. {m.path.name} ({size_str})",
                                    "tool": "share_file",
                                    "kwargs": kw,
                                })
                            list_text = "\n".join(lines)
                            reply_text = f"Found multiple files matching '{raw_target}':\n{list_text}\nTap one to send it."
                            token = self.choice_store.register(user_id, choices)
                            labels = [str(c.get("label", "")) for c in choices]
                            self.history.add(user_id, "user", message)
                            self.history.add(user_id, assistant_turn)
                            self.history.add(user_id, {"role": "tool", "tool_name": name, "content": reply_text})
                            self.history.add(user_id, "assistant", reply_text)
                            if EXTRACTOR_ENABLED:
                                task = asyncio.create_task(self._remember(hist, message))
                                self._bg_tasks.add(task)
                                task.add_done_callback(self._bg_tasks.discard)
                            return ChoicesReply(text=reply_text, token=token, labels=labels)

                        if target_path:
                            filtered_args["path"] = str(target_path.resolve())
                            self.last_file_path[user_id] = str(target_path.resolve())

                    if plugin.requires_approval:
                        self.history.add(user_id, "user", message)
                        self.history.add(user_id, assistant_turn)
                        return await self.request_tool_approval(user_id, name, filtered_args)

                    res = await self.run_tool(name, filtered_args, chat_id)
                    msgs.append({"role": "tool", "tool_name": name, "content": res.output})
                    tool_trace.append({
                        "role": "tool",
                        "tool_name": name,
                        "content": (res.output or "")[:500],
                    })

                    if len(calls) == 1 and isinstance(res.data, dict):
                        choices = res.data.get("choices")
                        if choices and isinstance(choices, list):
                            if "mail" in message.lower() or "email" in message.lower():
                                for c in choices:
                                    if isinstance(c, dict) and "kwargs" in c:
                                        c["kwargs"]["delivery"] = "email"
                            if res.data.get("matches"):
                                self.last_file_path[user_id] = str(res.data["matches"][0])
                            token = self.choice_store.register(user_id, choices)
                            labels = [str(c.get("label", "")) for c in choices]
                            reply_text = str(res.data.get("direct_reply") or res.output or "")
                            self.history.add(user_id, "user", message)
                            self.history.add(user_id, assistant_turn)
                            self.history.add(user_id, {"role": "tool", "tool_name": name, "content": res.output})
                            self.history.add(user_id, "assistant", reply_text)
                            if EXTRACTOR_ENABLED:
                                task = asyncio.create_task(self._remember(hist, message))
                                self._bg_tasks.add(task)
                                task.add_done_callback(self._bg_tasks.discard)
                            return ChoicesReply(text=reply_text, token=token, labels=labels)

                        if "direct_reply" in res.data and res.data["direct_reply"]:
                            reply_text = str(res.data["direct_reply"])
                            self.history.add(user_id, "user", message)
                            self.history.add(user_id, assistant_turn)
                            self.history.add(user_id, {"role": "tool", "tool_name": name, "content": res.output})
                            self.history.add(user_id, "assistant", reply_text)
                            if EXTRACTOR_ENABLED:
                                task = asyncio.create_task(self._remember(hist, message))
                                self._bg_tasks.add(task)
                                task.add_done_callback(self._bg_tasks.discard)
                            return reply_text

                reply = await self.llm.chat(msgs, tools=tools)
                log.info("tool_calls=%s", reply.get("tool_calls"))

            content = (reply.get("content") or "").strip()
            calls = reply.get("tool_calls")
            if not content and not calls and not tool_trace:
                done_reason = reply.get("done_reason") or reply.get("finish_reason") or "unknown"
                log.warning("Empty model reply: done_reason=%s raw=%r", done_reason, reply)
                text = "I didn't catch that. Can you rephrase?"
            else:
                text = content or "I didn't catch that. Can you rephrase?"
            self.history.add(user_id, "user", message)
            for entry in tool_trace:
                self.history.add(user_id, entry)
            self.history.add(user_id, "assistant", text)

            # Background memory extraction — never affects the reply
            if EXTRACTOR_ENABLED:
                task = asyncio.create_task(self._remember(hist, message))
                self._bg_tasks.add(task)
                task.add_done_callback(self._bg_tasks.discard)

            return text
        finally:
            total_ms = (time.perf_counter() - t_msg) * 1000.0
            log.info(
                "Orchestrator timing | retrieval_ms=%.1f handle_user_message_ms=%.1f",
                retrieval_ms,
                total_ms,
            )

    async def _remember(self, hist: list[dict[str, Any]], message: str) -> None:
        """Background task: extract facts from the conversation and upsert them."""
        try:
            from core.extractor import extract_facts  # noqa: PLC0415

            for f in await extract_facts(hist, message):
                await self.memory.aupsert(f["subject"], f["attribute"], f["value"])
                log.info("Memory upsert: %s", f)
        except Exception:
            log.exception("memory extraction failed")

    async def resume_tool_loop(self, user_id: int) -> str:
        """Resume LLM generation after user approval and tool execution.

        Fetches user history (which now includes the user's approval and the tool's execution result),
        sends this updated history back to self.llm.chat(), appends the final conversational response
        to history, and returns the text string.
        """
        hist = self.history.get(user_id)

        # Retrieve relevant memories using the last user prompt in history
        last_user = next(
            (m.get("content", "") for m in reversed(hist) if m.get("role") == "user"),
            "",
        )
        memories = await self.memory.asearch_memory(last_user, k=MEMORY_TOP_K) if last_user else ""

        msgs: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": build_system_prompt(memories, plugins=self.registry.values()),
            }
        ]
        msgs += hist

        tools = self._chat_tools()

        reply = await self.llm.chat(msgs, tools=tools)
        log.info("tool_calls=%s", reply.get("tool_calls"))

        for _ in range(3):  # tool loop
            calls = reply.get("tool_calls")
            if not calls:
                break
            msgs.append(reply)
            for call in calls:
                name = call.get("function", {}).get("name")
                args = call.get("function", {}).get("arguments") or {}
                plugin = self.registry.get(name)
                if plugin is None:
                    msgs.append({"role": "tool", "tool_name": name, "content": "unknown tool"})
                    continue

                allowed = plugin.parameters_schema.get("properties", {})
                filtered_args = {k: v for k, v in args.items() if k in allowed} if allowed else {}

                if name == "save_memory" and not filtered_args.get("fact"):
                    msgs.append({"role": "tool", "tool_name": name, "content": "missing fact"})
                    continue

                if plugin.requires_approval:
                    break

                res = await self.run_tool(name, filtered_args, 0)
                msgs.append({"role": "tool", "tool_name": name, "content": res.output})

            reply = await self.llm.chat(msgs, tools=tools)
            log.info("tool_calls=%s", reply.get("tool_calls"))

        text = (reply.get("content") or "").strip() or "Done!"
        self.history.add(user_id, "assistant", text)
        return text
