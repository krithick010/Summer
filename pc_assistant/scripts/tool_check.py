"""
scripts/tool_check.py
---------------------
Manual tool-routing evaluation harness for Summer against a live Ollama server.

Not part of the unit test suite and not run in CI. Replaces every plugin's
``execute()`` with a lightweight stub (no Chromium browser, no Gmail, no
external network) and runs canned prompts through the real ``Orchestrator``.

Usage:
    cd pc_assistant
    python scripts/tool_check.py
    python scripts/tool_check.py --model qwen3:4b
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Callable


@dataclass(frozen=True)
class EvalCase:
    message: str
    expected_tool: str | None
    arg_check: Callable[[list[tuple[str, dict[str, Any]]]], bool] | None = None
    note: str = ""


def _build_single_turn_cases() -> list[EvalCase]:
    return [
        EvalCase("hi", None),
        EvalCase("how's the weather?", "get_weather"),
        EvalCase(
            "weather in Paris",
            "get_weather",
            lambda calls: any(
                "paris" in str(args).lower()
                for name, args in calls
                if name == "get_weather"
            ),
            note="args mention Paris",
        ),
        EvalCase("any ai news?", "get_ai_news"),
        EvalCase("latest AI headlines", "get_ai_news"),
        EvalCase("what's new in AI?", "get_ai_news"),
        EvalCase("is it going to rain today?", "get_weather"),
        EvalCase("who is APJ Abdul Kalam on wikipedia", "search_wikipedia"),
        EvalCase("look up Alan Turing", "search_wikipedia"),
        EvalCase("how's my PC doing?", "system_info"),
        EvalCase("email test@example.com saying hi", "send_email"),
        EvalCase("what's my mom's number?", None),
        EvalCase("my grandma's number is 555-0123", None),
        EvalCase("thanks", None),
    ]


def _setup_hermetic_env(tmpdir: str, model_override: str | None) -> None:
    """Configure temp storage env vars before importing ``config``."""
    pc_assistant_dir = str(Path(__file__).resolve().parent.parent)
    if pc_assistant_dir not in sys.path:
        sys.path.insert(0, pc_assistant_dir)

    if not os.environ.get("TELEGRAM_BOT_TOKEN"):
        os.environ["TELEGRAM_BOT_TOKEN"] = "tool_check_dummy:ABCdefGHIjklMNOpqrsTUVwxyz"
    os.environ.setdefault("ALLOWED_USER_IDS", "1")
    os.environ["DATA_DIR"] = tmpdir
    os.environ["MEMORY_DB_PATH"] = str(Path(tmpdir) / "memory_db")
    os.environ["HISTORY_DB_PATH"] = ":memory:"
    os.environ["EXTRACTOR_ENABLED"] = "false"
    if model_override:
        os.environ["OLLAMA_MODEL"] = model_override


def _stub_orchestrator_plugins(orch: Any, recorded_calls: list[tuple[str, dict[str, Any]]]) -> None:
    """Stub all plugin executions and capture HITL tool triggers without real side effects."""
    from plugins.base import PluginResult

    if "send_email" not in orch.registry:
        try:
            from plugins.p_gmail import GmailPlugin

            orch.registry["send_email"] = GmailPlugin()
        except Exception:
            pass

    for tool_name, plugin in orch.registry.items():
        async def _stub_execute(
            _tool: str = tool_name, **kwargs: Any
        ) -> PluginResult:
            recorded_calls.append((_tool, dict(kwargs)))
            if _tool == "get_ai_news":
                out = (
                    "1. Stub AI headline one - https://example.org/1\n"
                    "2. Stub AI headline two - https://example.org/2\n"
                    "3. Stub AI headline three - https://example.org/3\n"
                    "Source: example.org"
                )
            elif _tool == "get_weather":
                loc = kwargs.get("location") or "Paris"
                out = f"{loc}: Clear, +18C (feels +17C), humidity 60%, wind 8km/h"
            elif _tool == "search_wikipedia":
                out = f"Stub summary for {kwargs.get('query', 'topic')}."
            elif _tool == "system_info":
                out = "CPU: 12% | RAM: 45% | Disk: 50%"
            else:
                out = f"Stub {_tool} result."
            return PluginResult(success=True, output=out)

        plugin.execute = _stub_execute  # type: ignore[method-assign]

    orig_create_action = orch.state_manager.create_action

    def _wrapped_create_action(
        tool_name: str,
        kwargs: dict[str, Any],
        description: str,
        user_id: int,
    ) -> Any:
        recorded_calls.append((tool_name, dict(kwargs)))
        return orig_create_action(
            tool_name=tool_name,
            kwargs=kwargs,
            description=description,
            user_id=user_id,
        )

    orch.state_manager.create_action = _wrapped_create_action
    orch.set_hitl_builder(lambda **kw: {"hitl": True, **kw})


def _matches_case(case: EvalCase, calls: list[tuple[str, dict[str, Any]]]) -> bool:
    called_names = [name for name, _ in calls]
    if case.expected_tool is None:
        return len(called_names) == 0
    if case.expected_tool not in called_names:
        return False
    if case.arg_check is not None:
        return bool(case.arg_check(calls))
    return True


def _print_table(rows: list[tuple[str, str, str, str, bool]], label: str) -> None:
    print(f"\n=== {label} ===")
    header = f"{'message':<38} | {'expected tool':<18} | {'tools actually called':<28} | {'seconds':>7}"
    print(header)
    print("-" * len(header))
    for msg, exp, actual, secs, ok in rows:
        mark = "" if ok else " [MISS]"
        print(f"{msg[:38]:<38} | {exp:<18} | {actual[:28]:<28} | {secs:>7}{mark}")


async def _run_eval(model_override: str | None) -> int:
    with tempfile.TemporaryDirectory(prefix="summer_tool_check_") as tmpdir:
        _setup_hermetic_env(tmpdir, model_override)

        import config

        if model_override:
            config.OLLAMA_MODEL = model_override

        from core.orchestrator import Orchestrator

        orch = Orchestrator()
        recorded_calls: list[tuple[str, dict[str, Any]]] = []
        _stub_orchestrator_plugins(orch, recorded_calls)

        uid = 1
        cases = _build_single_turn_cases()
        rows: list[tuple[str, str, str, str, bool]] = []
        correct = 0

        for case in cases:
            orch.history.clear(uid)
            recorded_calls.clear()
            t0 = time.perf_counter()
            await orch.handle_user_message(uid, uid, case.message)
            elapsed = time.perf_counter() - t0
            passed = _matches_case(case, recorded_calls)
            if passed:
                correct += 1
            exp_label = case.expected_tool or "none"
            if case.note:
                exp_label = f"{exp_label} ({case.note})"
            actual_label = ", ".join(f"{n}({a})" if a else n for n, a in recorded_calls) or "none"
            rows.append((case.message, exp_label, actual_label, f"{elapsed:.2f}", passed))

        _print_table(rows, f"Single-Turn Eval (model={config.OLLAMA_MODEL})")
        print(f"\n{correct}/{len(cases)} correct")
        orch.history.close()
        return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Summer's Ollama tool calling accuracy.")
    parser.add_argument("--model", help="Override OLLAMA_MODEL for this evaluation run.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return asyncio.run(_run_eval(args.model))


if __name__ == "__main__":
    raise SystemExit(main())

