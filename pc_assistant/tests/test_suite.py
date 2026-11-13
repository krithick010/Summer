"""
tests/test_suite.py
-------------------
Layered test suite for the Local PC Assistant.

Layer 1 — Import & contract tests  (no external services needed)
Layer 2 — Unit tests               (History, StateManager, LLMEngine, Orchestrator, Memory)
Layer 3 — Integration smoke tests  (real psutil, no Telegram/Ollama)

Run with:
    cd pc_assistant
    python tests/test_suite.py
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import tempfile
import time
import unittest
import warnings
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

# ---------------------------------------------------------------------------
# Hermetic environment settings for tests (must be set BEFORE project imports)
# ---------------------------------------------------------------------------
if not os.environ.get("TELEGRAM_BOT_TOKEN"):
    os.environ["TELEGRAM_BOT_TOKEN"] = "dummy_token_123456:ABCdefGHIjklMNOpqrsTUVwxyz"
os.environ["ALLOWED_USER_IDS"] = "1"
os.environ["HISTORY_DB_PATH"] = ":memory:"
_TEST_TMP_DIR = tempfile.mkdtemp(prefix="summer_test_")
os.environ["DATA_DIR"] = _TEST_TMP_DIR
os.environ["MEMORY_DB_PATH"] = str(Path(_TEST_TMP_DIR) / "memory_db")
os.environ["USER_LOCATION"] = ""
os.environ["BROWSER_HEADLESS"] = "true"
os.environ["AI_NEWS_URLS"] = (
    "https://techcrunch.com/category/artificial-intelligence/,"
    "https://venturebeat.com/category/ai/"
)
os.environ["LLM_THINK"] = ""
os.environ["FILE_SEARCH_MODE"] = "home"
os.environ["FILE_SEARCH_ROOTS"] = ""
os.environ["FILE_SEARCH_EXCLUDE"] = ""

# ---------------------------------------------------------------------------
# Silence noisy third-party warnings so test output stays readable
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore", category=FutureWarning, module="google")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---------------------------------------------------------------------------
# Make sure imports resolve from pc_assistant/ root
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def _run(coro):
    return asyncio.run(coro)


# ============================================================================
# LAYER 1 — Import & Contract Tests
# ============================================================================

class TestImports(unittest.TestCase):
    """Every module must import cleanly."""

    def test_import_config(self):
        import config
        self.assertIsNotNone(config.TELEGRAM_BOT_TOKEN)

    def test_import_history(self):
        from core.history import History
        self.assertTrue(callable(History))

    def test_import_plugin_base(self):
        from plugins.base import PluginBase, PluginResult
        self.assertTrue(callable(PluginBase))
        self.assertTrue(callable(PluginResult))

    def test_import_state_manager(self):
        from core.state_manager import StateManager, state_manager
        self.assertIsNotNone(state_manager)

    def test_import_llm_engine(self):
        from core.llm_engine import LLMEngine
        self.assertIsNotNone(LLMEngine())

    def test_import_p_system(self):
        from plugins.p_system import SystemInfoPlugin
        self.assertEqual(SystemInfoPlugin().name, "system_info")

    def test_import_p_gmail(self):
        from plugins.p_gmail import GmailPlugin
        self.assertEqual(GmailPlugin().name, "send_email")

    def test_import_p_remember(self):
        from plugins.p_remember import RememberPlugin
        self.assertEqual(RememberPlugin().name, "save_memory")

    def test_import_p_wikipedia(self):
        from plugins.p_wikipedia import WikipediaPlugin
        self.assertEqual(WikipediaPlugin().name, "search_wikipedia")

    def test_import_p_weather(self):
        from plugins.p_weather import WeatherPlugin
        self.assertEqual(WeatherPlugin().name, "get_weather")

    def test_import_p_ai_news(self):
        from plugins.p_ai_news import AINewsPlugin
        self.assertEqual(AINewsPlugin().name, "get_ai_news")

    def test_import_file_search(self):
        from plugins.file_search import find_files, is_sensitive_path
        self.assertTrue(callable(find_files))
        self.assertTrue(callable(is_sensitive_path))

    def test_import_p_files(self):
        from plugins.p_files import FindFilesPlugin, ShareFilePlugin
        self.assertEqual(FindFilesPlugin().name, "find_files")
        self.assertEqual(ShareFilePlugin().name, "share_file")

    def test_import_p_folder(self):
        from plugins.p_folder import ListFolderPlugin
        self.assertEqual(ListFolderPlugin().name, "list_folder")

    def test_import_memory(self):
        from core.memory import MemoryManager
        self.assertTrue(callable(MemoryManager))

    def test_import_extractor(self):
        from core.extractor import extract_facts
        self.assertTrue(callable(extract_facts))


class TestPluginBaseContract(unittest.TestCase):
    """PluginBase must enforce its contract at class-definition time."""

    def test_valid_plugin_accepted(self):
        from plugins.base import PluginBase, PluginResult

        class GoodPlugin(PluginBase):
            name = "good_plugin"
            description = "A well-formed test plugin."
            parameters_schema = {"type": "object", "properties": {}, "required": []}
            requires_approval = False

            async def execute(self, **kwargs):
                return PluginResult(success=True, output="ok")

        self.assertEqual(GoodPlugin().name, "good_plugin")

    def test_missing_name_raises(self):
        from plugins.base import PluginBase, PluginResult

        with self.assertRaises(TypeError):
            class BadPlugin(PluginBase):
                name = ""
                description = "desc"
                parameters_schema = {"type": "object", "properties": {}}

                async def execute(self, **kwargs):
                    return PluginResult(success=True, output="ok")

    def test_non_snakecase_name_raises(self):
        from plugins.base import PluginBase, PluginResult

        with self.assertRaises(TypeError):
            class BadPlugin(PluginBase):
                name = "Bad Plugin!"
                description = "desc"
                parameters_schema = {"type": "object", "properties": {}}

                async def execute(self, **kwargs):
                    return PluginResult(success=True, output="ok")

    def test_to_tool_spec_shape(self):
        from plugins.p_system import SystemInfoPlugin
        spec = SystemInfoPlugin().to_tool_spec()
        for key in ("name", "description", "parameters"):
            self.assertIn(key, spec)


# ============================================================================
# LAYER 2 — Unit Tests (mocked dependencies)
# ============================================================================

class TestHistory(unittest.TestCase):
    """Test SQLite-persisted rolling conversation history."""

    def test_history_maxlen(self):
        """(b) Window limit and order."""
        from core.history import History
        # max_turns=2 -> maxlen = 2 * 2 = 4 messages
        h = History(max_turns=2, db_path=":memory:")
        for i in range(6):
            h.add(1, "user" if i % 2 == 0 else "assistant", f"msg {i}")

        items = h.get(1)
        self.assertEqual(len(items), 4)
        self.assertEqual(items[0]["content"], "msg 2")
        self.assertEqual(items[-1]["content"], "msg 5")

    def test_history_clear(self):
        """(c) Clear history."""
        from core.history import History
        h = History(db_path=":memory:")
        h.add(1, "user", "hi")
        h.add(1, "assistant", "hello")
        self.assertEqual(len(h.get(1)), 2)

        h.clear(1)
        self.assertEqual(h.get(1), [])

    def test_history_isolation(self):
        """(c) Per-user isolation."""
        from core.history import History
        h = History(db_path=":memory:")
        h.add(100, "user", "secret of user 100")
        self.assertEqual(h.get(200), [])
        self.assertEqual(len(h.get(100)), 1)

    def test_history_supports_dict_messages(self):
        """(d) Tool calls and tool dict round-trip exactly."""
        from core.history import History
        h = History(db_path=":memory:")
        h.add(1, "user", "Send an email")
        tool_call = {
            "id": "call_123",
            "type": "function",
            "function": {"name": "send_email", "arguments": '{"to": "boss@example.com"}'},
        }
        assistant_turn = {
            "role": "assistant",
            "content": "",
            "tool_calls": [tool_call],
        }
        tool_turn = {"role": "tool", "tool_call_id": "call_123", "content": "Email sent"}

        h.add(1, assistant_turn)
        h.add(1, tool_turn)
        items = h.get(1)
        self.assertEqual(len(items), 3)
        self.assertEqual(items[1], assistant_turn)
        self.assertEqual(items[2], tool_turn)

    def test_history_persistence_across_instances(self):
        """(a) Persistence across two History instances on one file."""
        from core.history import History
        db_file = Path(_TEST_TMP_DIR) / f"test_hist_{time.time_ns()}.db"
        h1 = History(db_path=str(db_file))
        h1.add(42, "user", "Remember this message")
        h1.add(42, "assistant", "Understood")
        h1.close()

        h2 = History(db_path=str(db_file))
        items = h2.get(42)
        h2.close()

        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["role"], "user")
        self.assertEqual(items[0]["content"], "Remember this message")
        self.assertEqual(items[1]["role"], "assistant")
        self.assertEqual(items[1]["content"], "Understood")

    def test_history_startup_repair(self):
        """(e) Startup repair removes orphaned trailing assistant tool_calls & trims leading non-user."""
        from core.history import History
        db_file = Path(_TEST_TMP_DIR) / f"test_repair_{time.time_ns()}.db"
        h1 = History(db_path=str(db_file))
        # 1. Leading non-user messages
        h1.add(10, "assistant", "Leading orphan assistant message")
        # 2. Valid user message
        h1.add(10, "user", "User query")
        # 3. Trailing assistant message with unexecuted tool_calls (crash before resolution)
        h1.add(10, {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "orphan_call", "function": {"name": "send_email", "arguments": "{}"}}],
        })
        h1.close()

        # h2 startup triggers self._repair()
        h2 = History(db_path=str(db_file))
        items = h2.get(10)
        h2.close()

        # Leading assistant dropped, trailing assistant with tool_calls dropped, only user query left
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["role"], "user")
        self.assertEqual(items[0]["content"], "User query")

    def test_history_get_during_pending_approval(self):
        """(f) get() on a pending-approval history still returns the assistant tool_calls turn."""
        from core.history import History
        h = History(db_path=":memory:")
        h.add(1, "user", "Send email to team")
        assistant_turn = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "pending_call_1", "function": {"name": "send_email", "arguments": "{}"}}],
        }
        h.add(1, assistant_turn)

        # Calling get() during runtime (e.g. inside main.py callback) must preserve the pending turn
        items = h.get(1)
        self.assertEqual(len(items), 2)
        self.assertEqual(items[-1]["role"], "assistant")
        self.assertEqual(items[-1]["tool_calls"][0]["id"], "pending_call_1")


class TestStateManager(unittest.TestCase):
    """Test HITL state machine logic without touching Telegram."""

    def setUp(self):
        from core.state_manager import StateManager
        self.sm = StateManager()

    def test_create_and_get_action(self):
        action = self.sm.create_action("send_email", {"to": "a@b.com"}, "desc", user_id=1)
        self.assertIsNotNone(self.sm.get_action(action.approval_id))
        self.assertEqual(self.sm.get_action(action.approval_id).tool_name, "send_email")

    def test_resolve_returns_payload(self):
        action = self.sm.create_action("send_email", {"to": "a@b.com"}, "desc", user_id=1)
        payload = self.sm.resolve(action.approval_id)
        self.assertIsNotNone(payload)
        self.assertEqual(payload["tool_name"], "send_email")
        self.assertEqual(payload["kwargs"]["to"], "a@b.com")

    def test_resolve_twice_returns_none(self):
        action = self.sm.create_action("send_email", {}, "desc", user_id=1)
        self.sm.resolve(action.approval_id)
        self.assertIsNone(self.sm.resolve(action.approval_id))

    def test_cancel_pending_by_user_id(self):
        action = self.sm.create_action("send_email", {}, "desc", user_id=42)
        self.assertTrue(self.sm.cancel_pending(user_id=42))
        self.assertIsNone(self.sm.get_action(action.approval_id))

    def test_cancel_nonexistent_user(self):
        self.assertFalse(self.sm.cancel_pending(user_id=9999))

    def test_cancel_by_id(self):
        action = self.sm.create_action("send_email", {}, "desc", user_id=1)
        self.assertTrue(self.sm.cancel_by_id(action.approval_id))
        self.assertIsNone(self.sm.get_action(action.approval_id))

    def test_new_action_cancels_old_one_for_same_user(self):
        a1 = self.sm.create_action("tool_a", {}, "desc", user_id=7)
        a2 = self.sm.create_action("tool_b", {}, "desc", user_id=7)
        self.assertIsNone(self.sm.get_action(a1.approval_id))
        self.assertIsNotNone(self.sm.get_action(a2.approval_id))

    def test_cleanup_expired(self):
        action = self.sm.create_action("tool", {}, "desc", user_id=1)
        action.timestamp = time.monotonic() - 999
        self.assertEqual(self.sm.cleanup_expired(), 1)
        self.assertIsNone(self.sm.get_action(action.approval_id))

    def test_resolve_sets_event(self):
        action = self.sm.create_action("tool", {}, "desc", user_id=1)
        self.assertFalse(action.event.is_set())
        self.sm.resolve(action.approval_id)
        self.assertTrue(action.event.is_set())

    def test_cancel_sets_event_and_not_approved(self):
        action = self.sm.create_action("tool", {}, "desc", user_id=1)
        self.sm.cancel_pending(user_id=1)
        self.assertTrue(action.event.is_set())
        self.assertFalse(action.approved)


class TestLLMEngine(unittest.TestCase):
    """Test LLMEngine /api/chat payload and error propagation."""

    @patch("httpx.AsyncClient.post")
    def test_chat_payload(self, mock_post):
        from core.llm_engine import LLMEngine
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "message": {"role": "assistant", "content": "Hello!"}
        }
        mock_post.return_value = mock_resp

        engine = LLMEngine()
        msgs = [{"role": "user", "content": "Hi"}]
        tools = [{"type": "function", "function": {"name": "test_tool"}}]

        res = _run(engine.chat(messages=msgs, tools=tools))
        self.assertEqual(res["content"], "Hello!")

        mock_post.assert_called_once()
        sent_payload = mock_post.call_args[1]["json"]
        self.assertEqual(sent_payload["messages"], msgs)
        self.assertEqual(sent_payload["tools"], tools)
        self.assertEqual(sent_payload["keep_alive"], "30m")
        self.assertEqual(sent_payload["options"]["num_ctx"], 4096)

    @patch("httpx.AsyncClient.post")
    def test_chat_propagates_httpx_error(self, mock_post):
        import httpx
        from core.llm_engine import LLMEngine
        mock_post.side_effect = httpx.ConnectError("connection refused")

        engine = LLMEngine()
        with self.assertRaises(httpx.HTTPError):
            _run(engine.chat(messages=[{"role": "user", "content": "Hi"}]))

    @patch("httpx.AsyncClient.post")
    def test_chat_logs_ollama_timing_and_tolerates_missing_fields(self, mock_post):
        from core.llm_engine import LLMEngine

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "model": "qwen2.5:3b",
            "load_duration": 125_000_000,
            "prompt_eval_count": 42,
            "eval_count": 11,
            "total_duration": 850_000_000,
            "message": {"role": "assistant", "content": "Hi!"},
        }
        mock_post.return_value = mock_resp

        engine = LLMEngine()
        with self.assertLogs("core.llm_engine", level="INFO") as cm:
            _run(engine.chat(messages=[{"role": "user", "content": "Hi"}]))

        timing_lines = [line for line in cm.output if "Ollama chat |" in line]
        self.assertEqual(len(timing_lines), 1)
        line = timing_lines[0]
        self.assertIn("model=qwen2.5:3b", line)
        self.assertIn("load_duration_ms=125.0", line)
        self.assertIn("prompt_eval_count=42", line)
        self.assertIn("eval_count=11", line)
        self.assertIn("total_ms=850.0", line)

        # Missing duration/count fields are tolerated gracefully
        mock_resp.json.return_value = {"message": {"role": "assistant", "content": "Ok"}}
        with self.assertLogs("core.llm_engine", level="INFO") as cm2:
            _run(engine.chat(messages=[{"role": "user", "content": "Ok"}]))
        line2 = [line for line in cm2.output if "Ollama chat |" in line][0]
        self.assertIn("load_duration_ms=na", line2)
        self.assertIn("total_ms=", line2)

    @patch("httpx.AsyncClient.post")
    def test_extractor_logs_ollama_timing(self, mock_post):
        from core.extractor import extract_facts

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "model": "qwen2.5:3b",
            "load_duration": 10_000_000,
            "prompt_eval_count": 88,
            "eval_count": 9,
            "total_duration": 210_000_000,
            "message": {"content": '{"facts": []}'},
        }
        mock_post.return_value = mock_resp

        with self.assertLogs("core.extractor", level="INFO") as cm:
            facts = _run(extract_facts([], "hello there"))

        self.assertEqual(facts, [])
        timing_lines = [line for line in cm.output if "Ollama extractor |" in line]
        self.assertEqual(len(timing_lines), 1)
        self.assertIn("load_duration_ms=10.0", timing_lines[0])
        self.assertIn("prompt_eval_count=88", timing_lines[0])
        self.assertIn("eval_count=9", timing_lines[0])
        self.assertIn("total_ms=210.0", timing_lines[0])

    def test_orchestrator_logs_retrieval_and_total_timing(self):
        from core.orchestrator import Orchestrator

        with patch.object(Orchestrator, "_load_plugins", lambda self: None):
            orch = Orchestrator()
        orch.memory = MagicMock()
        orch.memory.asearch_memory = AsyncMock(return_value="")
        orch.llm = MagicMock()
        orch.llm.chat = AsyncMock(return_value={"role": "assistant", "content": "Hi!"})

        with patch("core.orchestrator.EXTRACTOR_ENABLED", False), \
             self.assertLogs("core.orchestrator", level="INFO") as cm:
            reply = _run(orch.handle_user_message(1, 1, "hello"))

        self.assertEqual(reply, "Hi!")
        timing_lines = [line for line in cm.output if "Orchestrator timing |" in line]
        self.assertEqual(len(timing_lines), 1)
        self.assertIn("retrieval_ms=", timing_lines[0])
        self.assertIn("handle_user_message_ms=", timing_lines[0])

    def test_import_scripts_tool_check_without_side_effects(self):
        import importlib

        before_data = os.environ.get("DATA_DIR")
        before_hist = os.environ.get("HISTORY_DB_PATH")
        mod = importlib.import_module("scripts.tool_check")
        self.assertTrue(hasattr(mod, "main"))
        self.assertEqual(len(mod._build_single_turn_cases()), 14)
        self.assertEqual(os.environ.get("DATA_DIR"), before_data)
        self.assertEqual(os.environ.get("HISTORY_DB_PATH"), before_hist)

    @patch("httpx.AsyncClient.post")
    def test_extractor_shares_num_ctx_and_keep_alive_with_chat(self, mock_post):
        import config
        from core.extractor import extract_facts
        from core.llm_engine import LLMEngine

        chat_resp = MagicMock()
        chat_resp.raise_for_status = MagicMock()
        chat_resp.json.return_value = {"message": {"role": "assistant", "content": "Hi"}}

        ext_resp = MagicMock()
        ext_resp.raise_for_status = MagicMock()
        ext_resp.json.return_value = {"message": {"content": '{"facts": []}'}}

        mock_post.side_effect = [chat_resp, ext_resp]

        with patch.object(config, "OLLAMA_NUM_CTX", 8192), \
             patch.object(config, "OLLAMA_KEEP_ALIVE", "45m"):
            engine = LLMEngine()
            _run(engine.chat(messages=[{"role": "user", "content": "Hi"}]))
            _run(extract_facts([], "Hi"))

        chat_payload = mock_post.call_args_list[0][1]["json"]
        ext_payload = mock_post.call_args_list[1][1]["json"]
        self.assertEqual(ext_payload["options"]["num_ctx"], chat_payload["options"]["num_ctx"])
        self.assertEqual(ext_payload["options"]["num_ctx"], 8192)
        self.assertEqual(ext_payload["keep_alive"], chat_payload["keep_alive"])
        self.assertEqual(ext_payload["keep_alive"], "45m")
        self.assertEqual(ext_payload["options"]["temperature"], 0)
        self.assertEqual(ext_payload["options"]["num_predict"], 200)
        self.assertIn("format", ext_payload)

    @patch("httpx.AsyncClient.post")
    def test_llm_think_flag_in_chat_and_extractor(self, mock_post):
        import config
        from core.extractor import extract_facts
        from core.llm_engine import LLMEngine

        chat_resp = MagicMock()
        chat_resp.raise_for_status = MagicMock()
        chat_resp.json.return_value = {"message": {"role": "assistant", "content": "Hi"}}

        ext_resp = MagicMock()
        ext_resp.raise_for_status = MagicMock()
        ext_resp.json.return_value = {"message": {"content": '{"facts": []}'}}

        engine = LLMEngine()

        # Empty LLM_THINK -> omit "think"
        mock_post.reset_mock()
        mock_post.side_effect = [chat_resp, ext_resp]
        with patch.object(config, "LLM_THINK", ""):
            _run(engine.chat(messages=[{"role": "user", "content": "Hi"}]))
            _run(extract_facts([], "Hi"))
        self.assertNotIn("think", mock_post.call_args_list[0][1]["json"])
        self.assertNotIn("think", mock_post.call_args_list[1][1]["json"])

        # Explicit false -> "think": False
        mock_post.reset_mock()
        mock_post.side_effect = [chat_resp, ext_resp]
        with patch.object(config, "LLM_THINK", "false"):
            _run(engine.chat(messages=[{"role": "user", "content": "Hi"}]))
            _run(extract_facts([], "Hi"))
        self.assertIs(mock_post.call_args_list[0][1]["json"]["think"], False)
        self.assertIs(mock_post.call_args_list[1][1]["json"]["think"], False)

        # Explicit true -> "think": True
        mock_post.reset_mock()
        mock_post.side_effect = [chat_resp, ext_resp]
        with patch.object(config, "LLM_THINK", "true"):
            _run(engine.chat(messages=[{"role": "user", "content": "Hi"}]))
            _run(extract_facts([], "Hi"))
        self.assertIs(mock_post.call_args_list[0][1]["json"]["think"], True)
        self.assertIs(mock_post.call_args_list[1][1]["json"]["think"], True)


class TestOrchestrator(unittest.TestCase):
    """Test single-call conversational Orchestrator with native tool calling."""

    def _make_orch(self):
        from core.orchestrator import Orchestrator
        with patch.object(Orchestrator, "_load_plugins", lambda self: None):
            return Orchestrator()

    def test_plain_reply_path(self):
        orch = self._make_orch()
        orch.memory = MagicMock()
        orch.memory.asearch_memory = AsyncMock(return_value="")

        orch.llm = MagicMock()
        orch.llm.chat = AsyncMock(return_value={"role": "assistant", "content": "Hey bro, what's up?"})

        result = _run(orch.handle_user_message(user_id=1, chat_id=1, message="wassup"))
        self.assertEqual(result, "Hey bro, what's up?")

        # Check history updated with user and assistant turns
        hist = orch.history.get(1)
        self.assertEqual(len(hist), 2)
        self.assertEqual(hist[0], {"role": "user", "content": "wassup"})
        self.assertEqual(hist[1], {"role": "assistant", "content": "Hey bro, what's up?"})

    def test_tool_calls_save_memory(self):
        from plugins.base import PluginResult
        orch = self._make_orch()
        orch.memory = MagicMock()
        orch.memory.asearch_memory = AsyncMock(return_value="")

        fake_plugin = MagicMock()
        fake_plugin.name = "save_memory"
        fake_plugin.requires_approval = False
        fake_plugin.parameters_schema = {
            "type": "object",
            "properties": {"fact": {"type": "string"}},
            "required": ["fact"],
        }
        fake_plugin.execute = AsyncMock(return_value=PluginResult(success=True, output="Got it, saved."))
        orch.registry["save_memory"] = fake_plugin

        orch.llm = MagicMock()
        orch.llm.chat = AsyncMock(side_effect=[
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "save_memory",
                            "arguments": {"fact": "Mom's number is 555-0123"},
                        }
                    }
                ],
            },
            {
                "role": "assistant",
                "content": "Got it, I've noted down your mom's number.",
            },
        ])

        result = _run(orch.handle_user_message(user_id=1, chat_id=1, message="my mom's number is 555-0123"))
        self.assertEqual(result, "Got it, I've noted down your mom's number.")
        fake_plugin.execute.assert_called_once_with(fact="Mom's number is 555-0123")

    def test_tool_trace_stored_truncated_and_seen_on_next_turn(self):
        from plugins.base import PluginResult
        orch = self._make_orch()
        orch.memory = MagicMock()
        orch.memory.asearch_memory = AsyncMock(return_value="")

        fake_plugin = MagicMock()
        fake_plugin.name = "get_weather"
        fake_plugin.description = "weather"
        fake_plugin.usage_hint = ""
        fake_plugin.requires_approval = False
        fake_plugin.parameters_schema = {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": [],
        }
        long_output = "Paris forecast: " + ("rainy " * 120)
        self.assertGreater(len(long_output), 500)
        fake_plugin.execute = AsyncMock(return_value=PluginResult(success=True, output=long_output))
        orch.registry["get_weather"] = fake_plugin

        captured_msgs = []

        async def _fake_chat(msgs, tools=None):
            captured_msgs.append([dict(m) for m in msgs])
            if len(captured_msgs) == 1:
                return {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "get_weather", "arguments": {"location": "Paris"}}}
                    ],
                }
            if len(captured_msgs) == 2:
                return {"role": "assistant", "content": "It is rainy in Paris."}
            return {"role": "assistant", "content": "Still rainy."}

        orch.llm = MagicMock()
        orch.llm.chat = AsyncMock(side_effect=_fake_chat)

        first = _run(orch.handle_user_message(user_id=7, chat_id=7, message="weather in Paris"))
        self.assertEqual(first, "It is rainy in Paris.")
        # Live turn saw the full >500-char tool output
        self.assertEqual(captured_msgs[1][-1]["role"], "tool")
        self.assertEqual(captured_msgs[1][-1]["content"], long_output)

        hist = orch.history.get(7)
        self.assertEqual([m["role"] for m in hist], ["user", "assistant", "tool", "assistant"])
        self.assertIn("tool_calls", hist[1])
        self.assertEqual(hist[2]["tool_name"], "get_weather")
        self.assertEqual(len(hist[2]["content"]), 500)
        self.assertEqual(hist[2]["content"], long_output[:500])

        second = _run(orch.handle_user_message(user_id=7, chat_id=7, message="what about now?"))
        self.assertEqual(second, "Still rainy.")
        turn2_roles = [m["role"] for m in captured_msgs[2]]
        self.assertEqual(turn2_roles, ["system", "user", "assistant", "tool", "assistant", "user"])

    def test_hitl_send_email_returns_dict_and_does_not_execute(self):
        from plugins.base import PluginResult
        orch = self._make_orch()
        orch.memory = MagicMock()
        orch.memory.asearch_memory = AsyncMock(return_value="")

        orch.set_hitl_builder(lambda approval_id, preview_text, tool_name: {
            "text": f"Approve {tool_name}?", "reply_markup": "keyboard"
        })

        fake_gmail = MagicMock()
        fake_gmail.name = "send_email"
        fake_gmail.requires_approval = True
        fake_gmail.parameters_schema = {
            "type": "object",
            "properties": {"to_email": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}},
        }
        fake_gmail.execute = AsyncMock(return_value=PluginResult(success=True, output="sent"))
        orch.registry["send_email"] = fake_gmail

        orch.llm = MagicMock()
        orch.llm.chat = AsyncMock(return_value={
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "send_email",
                        "arguments": {"to_email": "boss@example.com", "subject": "Update", "body": "Hello"},
                    }
                }
            ],
        })

        result = _run(orch.handle_user_message(user_id=42, chat_id=42, message="email boss@example.com saying Hello"))
        self.assertIsInstance(result, dict)
        self.assertEqual(result["text"], "Approve send_email?")
        fake_gmail.execute.assert_not_called()

        # Check both user message and assistant tool_calls recorded in history for context resumption
        hist = orch.history.get(42)
        self.assertEqual(len(hist), 2)
        self.assertEqual(hist[0]["role"], "user")
        self.assertEqual(hist[1]["role"], "assistant")
        self.assertEqual(hist[1]["tool_calls"][0]["function"]["name"], "send_email")

    def test_resume_tool_loop(self):
        orch = self._make_orch()
        orch.memory = MagicMock()
        orch.memory.asearch_memory = AsyncMock(return_value="")

        # Preload history with user prompt, assistant tool call, and tool execution response
        orch.history.add(42, "user", "email boss@example.com saying Hello")
        orch.history.add(42, {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "send_email",
                        "arguments": {"to_email": "boss@example.com", "subject": "Update", "body": "Hello"},
                    }
                }
            ],
        })
        orch.history.add(42, {"role": "tool", "content": "Email sent successfully."})

        orch.llm = MagicMock()
        orch.llm.chat = AsyncMock(return_value={
            "role": "assistant",
            "content": "I've sent that email to your boss!",
        })

        res = _run(orch.resume_tool_loop(user_id=42))
        self.assertEqual(res, "I've sent that email to your boss!")

        hist = orch.history.get(42)
        self.assertEqual(len(hist), 4)
        self.assertEqual(hist[2]["role"], "tool")
        self.assertEqual(hist[3]["role"], "assistant")
        self.assertEqual(hist[3]["content"], "I've sent that email to your boss!")

    def test_retrieval_query_includes_previous_user_turn(self):
        orch = self._make_orch()
        orch.memory = MagicMock()
        orch.memory.asearch_memory = AsyncMock(return_value="")
        orch.llm = MagicMock()
        orch.llm.chat = AsyncMock(return_value={"role": "assistant", "content": "Her number is 555-0123."})

        # Preload history with previous user turn
        orch.history.add(1, "user", "Tell me about my mom")
        orch.history.add(1, "assistant", "Sure, what would you like to know?")

        _run(orch.handle_user_message(user_id=1, chat_id=1, message="phone number?"))
        self.assertEqual(orch.memory.asearch_memory.call_count, 2)
        called_queries = [call[0][0] for call in orch.memory.asearch_memory.call_args_list]
        self.assertEqual(called_queries, ["phone number?", "Tell me about my mom"])

    def test_split_query_retrieval(self):
        orch = self._make_orch()
        orch.memory = MagicMock()
        orch.memory.asearch_memory = AsyncMock(return_value="")
        orch.llm = MagicMock()
        orch.llm.chat = AsyncMock(return_value={"role": "assistant", "content": "Sure."})

        orch.history.add(1, "user", "Tell me about my mom")
        orch.history.add(1, "assistant", "Sure, what would you like to know?")

        # Short message (< 6 words) -> 2 queries
        _run(orch.handle_user_message(user_id=1, chat_id=1, message="phone number?"))
        self.assertEqual(orch.memory.asearch_memory.call_count, 2)

        # Long message (>= 6 words) -> 1 query
        orch.memory.asearch_memory.reset_mock()
        _run(orch.handle_user_message(user_id=1, chat_id=1, message="can you please tell me how the computer is doing right now"))
        self.assertEqual(orch.memory.asearch_memory.call_count, 1)
        self.assertEqual(orch.memory.asearch_memory.call_args[0][0], "can you please tell me how the computer is doing right now")

    def test_tools_exclude_save_memory(self):
        from core.orchestrator import Orchestrator
        with patch.object(Orchestrator, "_load_plugins"):
            orch = Orchestrator()
            p_sys = MagicMock()
            p_sys.name = "system_info"
            p_sys.description = "System stats"
            p_sys.parameters_schema = {"type": "object"}

            p_email = MagicMock()
            p_email.name = "send_email"
            p_email.description = "Send email"
            p_email.parameters_schema = {"type": "object"}

            p_save = MagicMock()
            p_save.name = "save_memory"
            p_save.description = "Save memory"
            p_save.parameters_schema = {"type": "object"}

            orch.registry = {
                "system_info": p_sys,
                "send_email": p_email,
                "save_memory": p_save,
            }
            orch.memory = MagicMock()
            orch.memory.asearch_memory = AsyncMock(return_value="")
            orch.llm = MagicMock()
            orch.llm.chat = AsyncMock(return_value={"role": "assistant", "content": "ok"})

            _run(orch.handle_user_message(user_id=1, chat_id=1, message="hello"))

            tools_arg = orch.llm.chat.call_args[1]["tools"]
            tool_names = [t["function"]["name"] for t in tools_arg]
            self.assertNotIn("save_memory", tool_names)
            self.assertIn("system_info", tool_names)
            self.assertIn("send_email", tool_names)

    @patch("core.extractor.extract_facts")
    def test_background_extraction_called_on_plain_reply(self, mock_extract):
        mock_extract.return_value = [{"subject": "mom", "attribute": "phone", "value": "555-0123"}]
        orch = self._make_orch()
        orch.memory = MagicMock()
        orch.memory.asearch_memory = AsyncMock(return_value="")
        orch.memory.aupsert = AsyncMock(return_value=True)
        orch.llm = MagicMock()
        orch.llm.chat = AsyncMock(return_value={"role": "assistant", "content": "Got it!"})

        orch.history.add(1, "user", "hi")
        orch.history.add(1, "assistant", "hello")

        res = _run(orch.handle_user_message(user_id=1, chat_id=1, message="My mom's phone is 555-0123"))
        self.assertEqual(res, "Got it!")
        _run(asyncio.sleep(0.05))
        mock_extract.assert_called_once()
        pre_turn_hist = mock_extract.call_args[0][0]
        self.assertEqual(len(pre_turn_hist), 2)
        self.assertEqual(mock_extract.call_args[0][1], "My mom's phone is 555-0123")
        orch.memory.aupsert.assert_called_once_with("mom", "phone", "555-0123")

    @patch("core.extractor.extract_facts")
    def test_background_extraction_failure_does_not_affect_reply(self, mock_extract):
        mock_extract.side_effect = RuntimeError("extractor crashed")
        orch = self._make_orch()
        orch.memory = MagicMock()
        orch.memory.asearch_memory = AsyncMock(return_value="")
        orch.llm = MagicMock()
        orch.llm.chat = AsyncMock(return_value={"role": "assistant", "content": "Hello there!"})

        res = _run(orch.handle_user_message(user_id=1, chat_id=1, message="hi"))
        self.assertEqual(res, "Hello there!")
        _run(asyncio.sleep(0.05))

    @patch("core.extractor.extract_facts")
    def test_hitl_send_email_does_not_run_extractor(self, mock_extract):
        from plugins.base import PluginResult
        orch = self._make_orch()
        orch.memory = MagicMock()
        orch.memory.asearch_memory = AsyncMock(return_value="")

        orch.set_hitl_builder(lambda approval_id, preview_text, tool_name: {
            "text": f"Approve {tool_name}?", "reply_markup": "keyboard"
        })

        fake_gmail = MagicMock()
        fake_gmail.name = "send_email"
        fake_gmail.requires_approval = True
        fake_gmail.parameters_schema = {
            "type": "object",
            "properties": {"to_email": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}},
        }
        fake_gmail.execute = AsyncMock(return_value=PluginResult(success=True, output="sent"))
        orch.registry["send_email"] = fake_gmail

        orch.llm = MagicMock()
        orch.llm.chat = AsyncMock(return_value={
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "send_email",
                        "arguments": {"to_email": "boss@example.com", "subject": "Update", "body": "Hello"},
                    }
                }
            ],
        })

        result = _run(orch.handle_user_message(user_id=42, chat_id=42, message="email boss@example.com saying Hello"))
        self.assertIsInstance(result, dict)
        self.assertEqual(result["text"], "Approve send_email?")
        _run(asyncio.sleep(0.05))
        mock_extract.assert_not_called()

    def test_dynamic_plugin_loading(self):
        from core.orchestrator import Orchestrator
        with patch("core.orchestrator.MemoryManager"), patch("plugins.p_gmail.GmailPlugin.is_available", return_value=True):
            orch = Orchestrator()
            self.assertIn("system_info", orch.registry)
            self.assertIn("send_email", orch.registry)
            self.assertIn("save_memory", orch.registry)
            self.assertIn("search_wikipedia", orch.registry)
            self.assertIn("get_weather", orch.registry)
            self.assertIn("get_ai_news", orch.registry)


class TestPluginAvailability(unittest.TestCase):
    """Test optional plugin availability (e.g. Gmail when credentials missing)."""

    def test_plugin_base_default_is_available(self):
        from plugins.base import PluginBase
        class DummyPlugin(PluginBase):
            name = "dummy"
            description = "Dummy description"
            parameters_schema = {"type": "object"}
            async def execute(self, **kwargs):
                return None

        p = DummyPlugin()
        self.assertTrue(p.is_available())
        self.assertEqual(p.unavailable_reason, "")

    def test_gmail_unavailable_when_no_credentials(self):
        from plugins.p_gmail import GmailPlugin
        with patch("pathlib.Path.exists", return_value=False):
            p = GmailPlugin()
            self.assertFalse(p.is_available())
            self.assertIn("Gmail credentials not found", p.unavailable_reason)

    def test_gmail_available_when_credentials_exist(self):
        from plugins.p_gmail import GmailPlugin
        with patch("pathlib.Path.exists", return_value=True):
            p = GmailPlugin()
            self.assertTrue(p.is_available())

    def test_unavailable_plugin_not_registered_and_not_in_tools(self):
        from core.orchestrator import Orchestrator
        with patch("core.orchestrator.MemoryManager"), patch("plugins.p_gmail.GmailPlugin.is_available", return_value=False):
            orch = Orchestrator()
            self.assertNotIn("send_email", orch.registry)
            self.assertIn("system_info", orch.registry)
            self.assertIn("save_memory", orch.registry)

            orch.llm = MagicMock()
            orch.llm.chat = AsyncMock(return_value={"role": "assistant", "content": "Hello"})
            orch.memory.asearch_memory = AsyncMock(return_value="")

            _run(orch.handle_user_message(user_id=1, chat_id=1, message="hello"))
            tools_arg = orch.llm.chat.call_args[1]["tools"]
            tool_names = [t["function"]["name"] for t in tools_arg]
            self.assertNotIn("send_email", tool_names)
            self.assertIn("system_info", tool_names)

    def test_available_plugin_registered_with_hitl(self):
        from core.orchestrator import Orchestrator
        with patch("core.orchestrator.MemoryManager"), patch("plugins.p_gmail.GmailPlugin.is_available", return_value=True):
            orch = Orchestrator()
            self.assertIn("send_email", orch.registry)
            plugin = orch.registry["send_email"]
            self.assertTrue(plugin.requires_approval)


class TestRememberPlugin(unittest.TestCase):
    """Test RememberPlugin schema and execution."""

    def test_remember_schema(self):
        from plugins.p_remember import RememberPlugin
        p = RememberPlugin()
        self.assertEqual(p.name, "save_memory")
        self.assertFalse(p.requires_approval)
        self.assertIn("fact", p.parameters_schema["required"])
        self.assertIn("standalone", p.description.lower())

    def test_execute_empty_fact_fails(self):
        from plugins.p_remember import RememberPlugin
        p = RememberPlugin()
        res = _run(p.execute(fact="   "))
        self.assertFalse(res.success)
        self.assertIn("No fact", res.output)

    def test_execute_success(self):
        from plugins.p_remember import RememberPlugin
        mock_mem = MagicMock()
        mock_mem.aadd_memory = AsyncMock(return_value=True)
        p = RememberPlugin(memory_manager=mock_mem)

        res = _run(p.execute(fact="Mom's phone number is 555-0123"))
        self.assertTrue(res.success)
        self.assertEqual(res.output, "Got it, saved.")
        mock_mem.aadd_memory.assert_called_once_with("Mom's phone number is 555-0123")


class TestMemoryManager(unittest.TestCase):
    """Test MemoryManager prefixes, hashing, newest-first sorting, and max_distance."""

    @patch("chromadb.PersistentClient")
    @patch("core.memory.OllamaEmbeddings")
    def test_add_memory_stores_clean_fact_and_search_document_prefix(self, mock_emb_cls, mock_client_cls):
        from core.memory import MemoryManager
        mock_coll = MagicMock()
        mock_coll.count.return_value = 0
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_coll
        mock_client_cls.return_value = mock_client

        mock_emb = MagicMock()
        mock_emb.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_emb_cls.return_value = mock_emb

        mem = MemoryManager(path=str(Path(_TEST_TMP_DIR) / "fake_path"))
        fact = " Mom's email is mom@example.com "
        expected_id = hashlib.sha256("mom's email is mom@example.com".encode("utf-8")).hexdigest()

        res = mem.add_memory(fact)
        self.assertTrue(res)

        # Must embed with search_document prefix
        mock_emb.embed_query.assert_called_with("search_document: Mom's email is mom@example.com")

        # Must store clean fact
        call_kwargs = mock_coll.upsert.call_args[1]
        self.assertEqual(call_kwargs["ids"], [expected_id])
        self.assertEqual(call_kwargs["documents"], ["Mom's email is mom@example.com"])
        self.assertEqual(call_kwargs["metadatas"][0]["fact"], "Mom's email is mom@example.com")
        self.assertIn("ts", call_kwargs["metadatas"][0])

    @patch("chromadb.PersistentClient")
    @patch("core.memory.OllamaEmbeddings")
    def test_search_memory_search_query_prefix_and_sorting(self, mock_emb_cls, mock_client_cls):
        from core.memory import MemoryManager
        mock_coll = MagicMock()
        mock_coll.count.return_value = 2
        mock_coll.query.return_value = {
            "documents": [["Old fact", "Newest correction"]],
            "distances": [[0.2, 0.3]],
            "metadatas": [[{"fact": "Old fact", "ts": 1000.0}, {"fact": "Newest correction", "ts": 2000.0}]],
        }
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_coll
        mock_client_cls.return_value = mock_client

        mock_emb = MagicMock()
        mock_emb.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_emb_cls.return_value = mock_emb

        mem = MemoryManager(path=str(Path(_TEST_TMP_DIR) / "fake_path"), max_distance=1.3)
        res = mem.search_memory("mom email")

        # Must embed with search_query prefix
        mock_emb.embed_query.assert_called_with("search_query: mom email")

        # Newest fact (ts=2000.0) must appear before old fact (ts=1000.0)
        lines = res.split("\n")
        self.assertEqual(lines[0], "- Newest correction")
        self.assertEqual(lines[1], "- Old fact")

    @patch("chromadb.PersistentClient")
    @patch("core.memory.OllamaEmbeddings")
    def test_search_memory_drops_above_max_distance(self, mock_emb_cls, mock_client_cls):
        from core.memory import MemoryManager
        mock_coll = MagicMock()
        mock_coll.count.return_value = 1
        mock_coll.query.return_value = {
            "documents": [["Far away fact"]],
            "distances": [[1.55]],  # > 1.3
            "metadatas": [[{"fact": "Far away fact", "ts": 1000.0}]],
        }
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_coll
        mock_client_cls.return_value = mock_client

        mock_emb = MagicMock()
        mock_emb.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_emb_cls.return_value = mock_emb

        mem = MemoryManager(path=str(Path(_TEST_TMP_DIR) / "fake_path"), max_distance=1.3)
        res = mem.search_memory("unrelated query")
        self.assertEqual(res, "")

    @patch("chromadb.PersistentClient")
    @patch("core.memory.OllamaEmbeddings")
    def test_embed_on_cpu_passes_or_omits_num_gpu(self, mock_emb_cls, mock_client_cls):
        import config
        from core.memory import MemoryManager

        mock_coll = MagicMock()
        mock_coll.count.return_value = 0
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_coll
        mock_client_cls.return_value = mock_client

        with patch.object(config, "EMBED_ON_CPU", True):
            MemoryManager(path=str(Path(_TEST_TMP_DIR) / "fake_cpu_true"))
        self.assertEqual(mock_emb_cls.call_args[1].get("num_gpu"), 0)

        mock_emb_cls.reset_mock()
        with patch.object(config, "EMBED_ON_CPU", False):
            MemoryManager(path=str(Path(_TEST_TMP_DIR) / "fake_cpu_false"))
        self.assertNotIn("num_gpu", mock_emb_cls.call_args[1])


class TestExtractor(unittest.TestCase):
    """Test core/extractor.py structured fact extraction and grounding rules."""

    @patch("httpx.AsyncClient.post")
    def test_extract_facts_valid_json(self, mock_post):
        from core.extractor import extract_facts
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "message": {
                "content": json.dumps({
                    "facts": [
                        {"subject": "grandma", "attribute": "phone", "value": "555-0199"},
                        {"subject": "grandma", "attribute": "email", "value": "x@y.com"},
                    ]
                })
            }
        }
        mock_post.return_value = mock_resp

        history = []
        msg = "my grandma's email is x@y.com and phone is 555-0199"
        facts = _run(extract_facts(history, msg))
        self.assertEqual(len(facts), 2)
        self.assertEqual(facts[0], {"subject": "grandma", "attribute": "phone", "value": "555-0199"})
        self.assertEqual(facts[1], {"subject": "grandma", "attribute": "email", "value": "x@y.com"})

    @patch("httpx.AsyncClient.post")
    def test_extract_facts_empty(self, mock_post):
        from core.extractor import extract_facts
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "message": {"content": json.dumps({"facts": []})}
        }
        mock_post.return_value = mock_resp

        facts = _run(extract_facts([], "do you know my mom's number?"))
        self.assertEqual(facts, [])

    @patch("httpx.AsyncClient.post")
    def test_ungrounded_values_dropped(self, mock_post):
        from core.extractor import extract_facts
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        # Model hallucinates email and phone not in conversation
        mock_resp.json.return_value = {
            "message": {
                "content": json.dumps({
                    "facts": [
                        {"subject": "mom", "attribute": "email", "value": "mom@example.com"},
                        {"subject": "mom", "attribute": "phone", "value": "1234567890"},
                    ]
                })
            }
        }
        mock_post.return_value = mock_resp

        # Conversation has NO email and NO phone
        facts = _run(extract_facts([], "do you know my mom's number?"))
        self.assertEqual(facts, [])

    @patch("httpx.AsyncClient.post")
    def test_remember_it_from_earlier_history(self, mock_post):
        from core.extractor import extract_facts
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "message": {
                "content": json.dumps({
                    "facts": [
                        {"subject": "mom", "attribute": "phone", "value": "555-0123"}
                    ]
                })
            }
        }
        mock_post.return_value = mock_resp

        # User gave number in previous turn, now says "remember it"
        # Under new strict rules, values not present in the latest message must be dropped
        history = [
            {"role": "user", "content": "My mom's phone is 555-0123"},
            {"role": "assistant", "content": "Got it."},
        ]
        facts = _run(extract_facts(history, "remember it"))
        self.assertEqual(facts, [])

    @patch("httpx.AsyncClient.post")
    def test_relation_prefix_and_attribute_normalization(self, mock_post):
        from core.extractor import extract_facts
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "message": {
                "content": json.dumps({
                    "facts": [
                        {"subject": "user", "attribute": "grandma_phone_number", "value": "555-0199"},
                        {"subject": "user", "attribute": "grandma_email", "value": "g@x.com"},
                    ]
                })
            }
        }
        mock_post.return_value = mock_resp

        msg = "my grandma's number is 555-0199 and email is g@x.com"
        facts = _run(extract_facts([], msg))
        self.assertEqual(len(facts), 2)
        self.assertEqual(facts[0], {"subject": "grandma", "attribute": "phone", "value": "555-0199"})
        self.assertEqual(facts[1], {"subject": "grandma", "attribute": "email", "value": "g@x.com"})

    @patch("httpx.AsyncClient.post")
    def test_history_only_value_in_question_dropped(self, mock_post):
        from core.extractor import extract_facts
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "message": {
                "content": json.dumps({
                    "facts": [
                        {"subject": "grandma", "attribute": "phone", "value": "555-0199"}
                    ]
                })
            }
        }
        mock_post.return_value = mock_resp

        history = [{"role": "user", "content": "my grandma's number is 555-0199"}]
        facts = _run(extract_facts(history, "what's my grandma's number?"))
        self.assertEqual(facts, [])

    @patch("httpx.AsyncClient.post")
    def test_placeholder_values_dropped(self, mock_post):
        from core.extractor import extract_facts
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "message": {
                "content": json.dumps({
                    "facts": [
                        {"subject": "user", "attribute": "email", "value": "user_email_not_provided"},
                        {"subject": "mom", "attribute": "phone", "value": "not provided"},
                        {"subject": "user", "attribute": "address", "value": "unknown"},
                        {"subject": "user", "attribute": "allergy", "value": "n/a"},
                        {"subject": "user", "attribute": "job", "value": "none"},
                        {"subject": "user", "attribute": "notes", "value": "null"},
                        {"subject": "user", "attribute": "status", "value": "unspecified"},
                        {"subject": "user", "attribute": "location", "value": "not available"},
                    ]
                })
            }
        }
        mock_post.return_value = mock_resp

        msg = "my details are unknown, not provided, n/a, none, null, unspecified, not available"
        facts = _run(extract_facts([], msg))
        self.assertEqual(facts, [])

    @patch("httpx.AsyncClient.post")
    def test_squash_formatting_variations(self, mock_post):
        from core.extractor import extract_facts
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "message": {
                "content": json.dumps({
                    "facts": [
                        {"subject": "mom", "attribute": "phone", "value": "555-0199"},
                        {"subject": "dad", "attribute": "phone", "value": "123-456-7890"},
                        {"subject": "user", "attribute": "email", "value": "USER@EXAMPLE.COM"},
                    ]
                })
            }
        }
        mock_post.return_value = mock_resp

        msg = "Mom's number is 555 0199, Dad is (123) 456-7890, email user@example.com"
        facts = _run(extract_facts([], msg))
        self.assertEqual(len(facts), 3)
        self.assertEqual(facts[0], {"subject": "mom", "attribute": "phone", "value": "555-0199"})
        self.assertEqual(facts[1], {"subject": "dad", "attribute": "phone", "value": "123-456-7890"})
        self.assertEqual(facts[2], {"subject": "user", "attribute": "email", "value": "USER@EXAMPLE.COM"})

    @patch("httpx.AsyncClient.post")
    def test_subject_and_attribute_aliases(self, mock_post):
        from core.extractor import extract_facts
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "message": {
                "content": json.dumps({
                    "facts": [
                        {"subject": "Mother", "attribute": "mobile", "value": "1112223333"},
                        {"subject": "papa", "attribute": "mail", "value": "p@x.com"},
                        {"subject": "granny", "attribute": "email_address", "value": "g@x.com"},
                        {"subject": "grandfather", "attribute": "number", "value": "4445556666"},
                        {"subject": "myself", "attribute": "gmail", "value": "me@x.com"},
                    ]
                })
            }
        }
        mock_post.return_value = mock_resp

        msg = "Mother mobile 1112223333 papa mail p@x.com granny email_address g@x.com grandfather number 4445556666 myself gmail me@x.com"
        facts = _run(extract_facts([], msg))
        self.assertEqual(len(facts), 5)
        self.assertEqual(facts[0], {"subject": "mom", "attribute": "phone", "value": "1112223333"})
        self.assertEqual(facts[1], {"subject": "dad", "attribute": "email", "value": "p@x.com"})
        self.assertEqual(facts[2], {"subject": "grandma", "attribute": "email", "value": "g@x.com"})
        self.assertEqual(facts[3], {"subject": "grandpa", "attribute": "phone", "value": "4445556666"})
        self.assertEqual(facts[4], {"subject": "user", "attribute": "email", "value": "me@x.com"})

    @patch("httpx.AsyncClient.post")
    def test_empty_fields_dropped(self, mock_post):
        from core.extractor import extract_facts
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "message": {
                "content": json.dumps({
                    "facts": [
                        {"subject": "", "attribute": "phone", "value": "9876543210"},
                        {"subject": "mom", "attribute": "", "value": "9876543210"},
                        {"subject": "mom", "attribute": "phone", "value": ""},
                        {"subject": "mom", "attribute": "phone", "value": "   "},
                    ]
                })
            }
        }
        mock_post.return_value = mock_resp

        msg = "phone is 9876543210"
        facts = _run(extract_facts([], msg))
        self.assertEqual(facts, [])

    @patch("httpx.AsyncClient.post")
    def test_extract_facts_skips_tool_and_tool_calls_messages(self, mock_post):
        from core.extractor import extract_facts
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"message": {"content": json.dumps({"facts": []})}}
        mock_post.return_value = mock_resp

        history = [
            {"role": "user", "content": "any ai news?"},
            {
                "role": "assistant",
                "content": "checking_tool_marker",
                "tool_calls": [{"function": {"name": "get_ai_news", "arguments": {}}}],
            },
            {"role": "tool", "tool_name": "get_ai_news", "content": "tool_output_marker"},
            {"role": "assistant", "content": "Here are the headlines."},
        ]
        _run(extract_facts(history, "thanks"))
        sent_prompt = mock_post.call_args.kwargs["json"]["messages"][0]["content"]
        self.assertIn("user: any ai news?", sent_prompt)
        self.assertIn("assistant: Here are the headlines.", sent_prompt)
        self.assertNotIn("checking_tool_marker", sent_prompt)
        self.assertNotIn("tool_output_marker", sent_prompt)

    @patch("httpx.AsyncClient.post")
    def test_malformed_json_raises_json_decode_error(self, mock_post):
        import json
        from core.extractor import extract_facts
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "message": {"content": "not valid json {{"}
        }
        mock_post.return_value = mock_resp

        with self.assertRaises(json.JSONDecodeError):
            _run(extract_facts([], "hello"))

    def test_clean_direct_cases(self):
        from core.extractor import _clean, RELATIONS, SUBJECT_ALIASES, ATTR_ALIASES
        self.assertIn("mom", RELATIONS)
        self.assertIn("mother", SUBJECT_ALIASES)
        self.assertIn("phone_number", ATTR_ALIASES)

        # Test relation prefix splitting
        cleaned = _clean({"subject": "user", "attribute": "mom_phone_number", "value": "12345"}, "mom is 12345")
        self.assertEqual(cleaned, {"subject": "mom", "attribute": "phone", "value": "12345"})

        # Test placeholder rejection
        self.assertIsNone(_clean({"subject": "user", "attribute": "email", "value": "unknown"}, "unknown"))
        self.assertIsNone(_clean({"subject": "user", "attribute": "email", "value": "user_email_not_provided"}, "email"))

        # Test ungrounded rejection
        self.assertIsNone(_clean({"subject": "mom", "attribute": "phone", "value": "99999"}, "no number here"))


class TestMemoryUpsert(unittest.TestCase):
    """Test MemoryManager subject+attribute upsert logic."""

    @patch("chromadb.PersistentClient")
    @patch("core.memory.OllamaEmbeddings")
    def test_upsert_same_attribute_overwrites(self, mock_emb_cls, mock_client_cls):
        from core.memory import MemoryManager
        mock_coll = MagicMock()
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_coll
        mock_client_cls.return_value = mock_client

        mock_emb = MagicMock()
        mock_emb.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_emb_cls.return_value = mock_emb

        mem = MemoryManager(path=str(Path(_TEST_TMP_DIR) / "fake_path"))

        # 1st upsert
        _run(mem.aupsert("grandma", "email", "old@y.com"))
        first_call_id = mock_coll.upsert.call_args_list[0][1]["ids"][0]
        first_doc = mock_coll.upsert.call_args_list[0][1]["documents"][0]
        self.assertEqual(first_doc, "Grandma's email is old@y.com")

        # 2nd upsert with same subject + attribute
        _run(mem.aupsert("grandma", "email", "new@y.com"))
        second_call_id = mock_coll.upsert.call_args_list[1][1]["ids"][0]
        second_doc = mock_coll.upsert.call_args_list[1][1]["documents"][0]
        self.assertEqual(second_doc, "Grandma's email is new@y.com")

        # Must share identical doc_id so Chroma overwrites
        self.assertEqual(first_call_id, second_call_id)

    @patch("chromadb.PersistentClient")
    @patch("core.memory.OllamaEmbeddings")
    def test_upsert_different_attributes_separate_records(self, mock_emb_cls, mock_client_cls):
        from core.memory import MemoryManager
        mock_coll = MagicMock()
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_coll
        mock_client_cls.return_value = mock_client

        mock_emb = MagicMock()
        mock_emb.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_emb_cls.return_value = mock_emb

        mem = MemoryManager(path=str(Path(_TEST_TMP_DIR) / "fake_path"))
        _run(mem.aupsert("mom", "email", "x@y.com"))
        _run(mem.aupsert("mom", "phone", "555-0199"))

        id1 = mock_coll.upsert.call_args_list[0][1]["ids"][0]
        id2 = mock_coll.upsert.call_args_list[1][1]["ids"][0]
        self.assertNotEqual(id1, id2)

    @patch("chromadb.PersistentClient")
    @patch("core.memory.OllamaEmbeddings")
    def test_upsert_user_subject_formatting(self, mock_emb_cls, mock_client_cls):
        from core.memory import MemoryManager
        mock_coll = MagicMock()
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_coll
        mock_client_cls.return_value = mock_client

        mock_emb = MagicMock()
        mock_emb.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_emb_cls.return_value = mock_emb

        mem = MemoryManager(path=str(Path(_TEST_TMP_DIR) / "fake_path"))
        _run(mem.aupsert("user", "allergy", "peanuts"))
        doc = mock_coll.upsert.call_args[1]["documents"][0]
        self.assertEqual(doc, "User's allergy is peanuts")


# ============================================================================
# LAYER 3 — Integration Smoke Tests (real psutil, no network)
# ============================================================================

class TestSystemPlugin(unittest.TestCase):
    """Run the system plugin against the real OS."""

    def setUp(self):
        from plugins.p_system import SystemInfoPlugin
        self.plugin = SystemInfoPlugin()

    def test_execute_returns_success(self):
        result = _run(self.plugin.execute())
        self.assertTrue(result.success)
        self.assertIn("System Status", result.output)

    def test_data_keys_present(self):
        result = _run(self.plugin.execute())
        data = result.data
        self.assertIn("cpu_percent", data)
        self.assertIn("ram_percent", data)
        self.assertIn("disk_free_gb", data)

    def test_cpu_in_valid_range(self):
        result = _run(self.plugin.execute())
        cpu = result.data["cpu_percent"]
        self.assertGreaterEqual(cpu, 0.0)
        self.assertLessEqual(cpu, 100.0)

    def test_ram_values_sane(self):
        result = _run(self.plugin.execute())
        d = result.data
        self.assertGreater(d["ram_total_gb"], 0)
        self.assertGreaterEqual(d["ram_used_gb"], 0)
        self.assertLessEqual(d["ram_used_gb"], d["ram_total_gb"])
        self.assertGreaterEqual(d["ram_percent"], 0.0)
        self.assertLessEqual(d["ram_percent"], 100.0)

    def test_disk_values_sane(self):
        result = _run(self.plugin.execute())
        d = result.data
        self.assertGreater(d["disk_total_gb"], 0)
        self.assertGreaterEqual(d["disk_free_gb"], 0)
        self.assertLessEqual(d["disk_free_gb"], d["disk_total_gb"])


class TestGmailPluginValidation(unittest.TestCase):
    """Test Gmail plugin input validation without sending real emails."""

    def setUp(self):
        from plugins.p_gmail import GmailPlugin
        self.plugin = GmailPlugin()

    def test_requires_approval_flag_set(self):
        self.assertTrue(self.plugin.requires_approval)

    def test_missing_fields_returns_failure(self):
        result = _run(self.plugin.execute(to_email="", subject="Sub", body="Body"))
        self.assertFalse(result.success)

    def test_all_required_params_in_schema(self):
        props = self.plugin.parameters_schema["properties"]
        for field in ("to_email", "subject", "body", "attachment_path"):
            self.assertIn(field, props)


class TestWikipediaPlugin(unittest.TestCase):
    """Test WikipediaPlugin metadata, redirects, disambiguation, and sentence extraction."""

    def setUp(self):
        from plugins.p_wikipedia import WikipediaPlugin
        self.plugin = WikipediaPlugin()

    def tearDown(self):
        from plugins.web_scraper import close_browser
        _run(close_browser())

    @staticmethod
    def _make_playwright_mock(
        pages_by_url=None,
        paragraphs=None,
        goto_side_effect=None,
        wait_side_effect=None,
        status=200,
    ):
        pages_by_url = pages_by_url or {}
        default_spec = {
            "final_url": "https://en.wikipedia.org/wiki/Python_(programming_language)",
            "title": "Python (programming language) - Wikipedia",
            "paragraphs": paragraphs if paragraphs is not None else [],
            "search_links": [],
            "disambig": False,
            "disambig_links": [],
            "status": status,
        }
        current_spec = dict(default_spec)

        mock_page = MagicMock()
        mock_page.url = default_spec["final_url"]
        mock_page.title = AsyncMock(side_effect=lambda: current_spec.get("title", "Wikipedia"))
        mock_page.route = AsyncMock()
        mock_page.screenshot = AsyncMock()

        async def _goto(url, **_kwargs):
            if goto_side_effect is not None:
                raise goto_side_effect
            if url in pages_by_url:
                current_spec.clear()
                current_spec.update({**default_spec, **pages_by_url[url]})
            mock_page.url = current_spec["final_url"]
            resp = MagicMock()
            resp.status = current_spec.get("status", 200)
            return resp

        def _make_list_locator(items, is_links=False):
            loc = MagicMock()
            loc.count = AsyncMock(return_value=len(items))

            def _nth(idx):
                el = MagicMock()
                if is_links:
                    text, href = items[idx]
                    el.inner_text = AsyncMock(return_value=text)
                    el.get_attribute = AsyncMock(return_value=href)
                else:
                    el.inner_text = AsyncMock(return_value=items[idx])
                return el

            loc.nth = MagicMock(side_effect=_nth)
            return loc

        def _locator(sel):
            if sel == ".mw-search-result-heading a":
                return _make_list_locator(current_spec.get("search_links", []), is_links=True)
            if sel == "#disambigbox, .dmbox":
                loc = MagicMock()
                loc.count = AsyncMock(return_value=1 if current_spec.get("disambig") else 0)
                return loc
            if sel == "#mw-content-text .mw-parser-output ul li a":
                return _make_list_locator(current_spec.get("disambig_links", []), is_links=True)
            return _make_list_locator(current_spec.get("paragraphs", []), is_links=False)

        mock_page.goto = AsyncMock(side_effect=_goto)
        if wait_side_effect is not None:
            mock_page.wait_for_selector = AsyncMock(side_effect=wait_side_effect)
        else:
            mock_page.wait_for_selector = AsyncMock(return_value=None)
        mock_page.locator = MagicMock(side_effect=_locator)

        mock_browser_ctx = MagicMock()
        mock_browser_ctx.new_page = AsyncMock(return_value=mock_page)
        mock_browser_ctx.close = AsyncMock()

        mock_browser = MagicMock()
        mock_browser.new_context = AsyncMock(return_value=mock_browser_ctx)
        mock_browser.close = AsyncMock()
        mock_browser.is_connected = MagicMock(return_value=True)

        mock_pw_obj = MagicMock()
        mock_pw_obj.chromium.launch = AsyncMock(return_value=mock_browser)
        mock_pw_obj.stop = AsyncMock()

        mock_ctx = MagicMock()
        mock_ctx.start = AsyncMock(return_value=mock_pw_obj)
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_pw_obj)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_factory = MagicMock(return_value=mock_ctx)
        return mock_factory, mock_browser, mock_browser_ctx, mock_page

    def test_schema_and_flags(self):
        self.assertEqual(self.plugin.name, "search_wikipedia")
        self.assertFalse(self.plugin.requires_approval)
        self.assertIn("query", self.plugin.parameters_schema["required"])
        self.assertEqual(self.plugin.parameters_schema["properties"]["query"]["type"], "string")

    def test_empty_query_fails_without_launching_browser(self):
        with patch("playwright.async_api.async_playwright") as mock_pw:
            result = _run(self.plugin.execute(query="   "))
            self.assertFalse(result.success)
            mock_pw.assert_not_called()

    def test_successful_wikipedia_lookup_extracts_summary_and_closes_browser(self):
        mock_factory, mock_browser, mock_browser_ctx, mock_page = self._make_playwright_mock(
            paragraphs=[
                "",
                "Python is a high-level programming language[1]. Its design philosophy emphasizes readability[2]. It supports multiple paradigms.",
                "Fourth sentence should not be included.",
            ]
        )
        with patch("playwright.async_api.async_playwright", mock_factory):
            result = _run(self.plugin.execute(query="Python programming"))

        self.assertTrue(result.success)
        self.assertIn("Python is a high-level programming language.", result.output)
        self.assertIn("Its design philosophy emphasizes readability.", result.output)
        self.assertIn("It supports multiple paradigms.", result.output)
        self.assertIn("https://en.wikipedia.org/wiki/Python_(programming_language)", result.output)
        self.assertNotIn("Fourth sentence", result.output)
        self.assertNotIn("[1]", result.output)
        self.assertNotIn("[2]", result.output)
        self.assertLessEqual(len(result.output), 800)
        goto_url = mock_page.goto.call_args[0][0]
        self.assertEqual(
            goto_url,
            "https://en.wikipedia.org/w/index.php?search=Python%20programming&go=Go",
        )
        mock_browser_ctx.close.assert_awaited_once()

    def test_search_results_page_follows_first_result(self):
        search_url = "https://en.wikipedia.org/w/index.php?search=Alan%20Turing&go=Go"
        article_url = "https://en.wikipedia.org/wiki/Alan_Turing"
        mock_factory, _mock_browser, mock_browser_ctx, _ = self._make_playwright_mock(
            pages_by_url={
                search_url: {
                    "final_url": search_url,
                    "search_links": [("Alan Turing", "/wiki/Alan_Turing")],
                    "paragraphs": [],
                },
                article_url: {
                    "final_url": article_url,
                    "paragraphs": [
                        "Alan Mathison Turing was an English mathematician[6]. He was influential in theoretical computer science.",
                    ],
                },
            }
        )
        with patch("playwright.async_api.async_playwright", mock_factory):
            result = _run(self.plugin.execute(query="Alan Turing"))

        self.assertTrue(result.success)
        self.assertIn("Alan Mathison Turing was an English mathematician.", result.output)
        self.assertIn(article_url, result.output)
        self.assertNotIn("[6]", result.output)
        self.assertEqual(result.data["url"], article_url)
        mock_browser_ctx.close.assert_awaited_once()

    def test_disambiguation_page_follows_first_target(self):
        disambig_url = "https://en.wikipedia.org/w/index.php?search=Mercury&go=Go"
        target_url = "https://en.wikipedia.org/wiki/Mercury_(planet)"
        mock_factory, _mock_browser, mock_browser_ctx, _ = self._make_playwright_mock(
            pages_by_url={
                disambig_url: {
                    "final_url": "https://en.wikipedia.org/wiki/Mercury",
                    "disambig": True,
                    "disambig_links": [
                        ("Help:Disambiguation", "/wiki/Help:Disambiguation"),
                        ("Mercury (planet)", "/wiki/Mercury_(planet)"),
                    ],
                    "paragraphs": ["Mercury may refer to:"],
                },
                target_url: {
                    "final_url": target_url,
                    "paragraphs": ["Mercury is the first planet from the Sun."],
                },
            }
        )
        with patch("playwright.async_api.async_playwright", mock_factory):
            result = _run(self.plugin.execute(query="Mercury"))

        self.assertTrue(result.success)
        self.assertIn("Mercury is the first planet from the Sun.", result.output)
        self.assertIn(target_url, result.output)
        self.assertEqual(result.data["url"], target_url)
        mock_browser_ctx.close.assert_awaited_once()

    def test_timeout_returns_error_and_closes_browser(self):
        from playwright.async_api import TimeoutError as PlaywrightTimeoutError

        mock_factory, _mock_browser, mock_browser_ctx, mock_page = self._make_playwright_mock(
            goto_side_effect=PlaywrightTimeoutError("Navigation timeout")
        )
        with patch("playwright.async_api.async_playwright", mock_factory):
            result = _run(self.plugin.execute(query="Alan Turing"))

        self.assertFalse(result.success)
        self.assertIn("timed out", result.output.lower())
        self.assertNotIn("\n", result.output)
        mock_browser_ctx.close.assert_awaited_once()
        mock_page.screenshot.assert_awaited_once()

    def test_missing_summary_returns_error_and_closes_browser(self):
        mock_factory, _mock_browser, mock_browser_ctx, mock_page = self._make_playwright_mock(paragraphs=[])
        with patch("playwright.async_api.async_playwright", mock_factory):
            result = _run(self.plugin.execute(query="NonExistentTopicXYZ123"))

        self.assertFalse(result.success)
        self.assertIn("No Wikipedia", result.output)
        self.assertNotIn("\n", result.output)
        mock_browser_ctx.close.assert_awaited_once()
        mock_page.screenshot.assert_awaited_once()


def _make_body_playwright_mock(
    body_text="",
    headlines=None,
    goto_side_effect=None,
    status=200,
    url="https://example.com",
    title="Example Page",
):
    """Create mocked async_playwright context/browser/page returning ``body_text`` and optional headlines."""
    headlines = headlines or []

    mock_body = MagicMock()
    mock_body.inner_text = AsyncMock(return_value=body_text)

    mock_headlines = MagicMock()
    mock_headlines.count = AsyncMock(return_value=len(headlines))

    def _nth_headline(idx):
        elem = MagicMock()
        elem.inner_text = AsyncMock(return_value=headlines[idx])
        return elem

    mock_headlines.nth = MagicMock(side_effect=_nth_headline)

    def _locator(sel):
        return mock_body if sel == "body" else mock_headlines

    mock_resp = MagicMock()
    mock_resp.status = status

    mock_page = MagicMock()
    mock_page.url = url
    mock_page.title = AsyncMock(return_value=title)
    mock_page.route = AsyncMock()
    mock_page.screenshot = AsyncMock()
    if goto_side_effect is not None:
        mock_page.goto = AsyncMock(side_effect=goto_side_effect)
    else:
        mock_page.goto = AsyncMock(return_value=mock_resp)
    mock_page.locator = MagicMock(side_effect=_locator)

    mock_browser_ctx = MagicMock()
    mock_browser_ctx.new_page = AsyncMock(return_value=mock_page)
    mock_browser_ctx.close = AsyncMock()

    mock_browser = MagicMock()
    mock_browser.new_context = AsyncMock(return_value=mock_browser_ctx)
    mock_browser.close = AsyncMock()
    mock_browser.is_connected = MagicMock(return_value=True)

    mock_pw_obj = MagicMock()
    mock_pw_obj.chromium.launch = AsyncMock(return_value=mock_browser)
    mock_pw_obj.stop = AsyncMock()

    mock_ctx = MagicMock()
    mock_ctx.start = AsyncMock(return_value=mock_pw_obj)
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_pw_obj)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)

    return MagicMock(return_value=mock_ctx), mock_browser, mock_browser_ctx, mock_page


class TestWebScraper(unittest.TestCase):
    """Test shared Playwright web_scraper layer."""

    def tearDown(self):
        from plugins.web_scraper import close_browser
        _run(close_browser())

    def test_captcha_redirect_returns_blocked_result(self):
        from plugins.web_scraper import scrape_page_body

        for blocked_url in (
            "https://www.google.com/sorry/index?continue=...",
            "https://example.com/captcha-verify",
        ):
            mock_factory, mock_browser, mock_bctx, mock_page = _make_body_playwright_mock(
                body_text="captcha challenge",
                url=blocked_url,
            )
            with patch("playwright.async_api.async_playwright", mock_factory):
                result = _run(
                    scrape_page_body("https://example.com", max_chars=1000, label="Test")
                )

            self.assertFalse(result.success)
            self.assertEqual(result.output, "⚠️ Blocked by CAPTCHA.")
            self.assertEqual(result.error, "CAPTCHA redirect detected")
            mock_bctx.close.assert_awaited_once()
            mock_page.screenshot.assert_awaited_once()
            mock_browser.close.assert_not_awaited()
            _run(self._reset_shared_browser())

    async def _reset_shared_browser(self):
        from plugins.web_scraper import close_browser
        await close_browser()

    def test_shared_browser_started_once_and_contexts_closed(self):
        from plugins.web_scraper import scrape_page_body

        mock_factory, mock_browser, mock_bctx, _ = _make_body_playwright_mock(
            body_text="hello world",
            url="https://example.com/1",
        )
        pw_instance = mock_factory.return_value.start.return_value

        async def _two_calls():
            r1 = await scrape_page_body("https://example.com/1", max_chars=100, label="One")
            r2 = await scrape_page_body("https://example.com/2", max_chars=100, label="Two")
            return r1, r2

        with patch("playwright.async_api.async_playwright", mock_factory):
            r1, r2 = _run(_two_calls())

        self.assertTrue(r1.success)
        self.assertTrue(r2.success)
        self.assertEqual(pw_instance.chromium.launch.await_count, 1)
        self.assertEqual(mock_browser.new_context.await_count, 2)
        self.assertEqual(mock_bctx.close.await_count, 2)

    def test_failure_saves_debug_screenshot_and_prunes_to_10(self):
        import config
        from plugins.web_scraper import scrape_page_body

        with tempfile.TemporaryDirectory() as tmpdir:
            debug_dir = Path(tmpdir) / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            # Seed 11 older screenshots
            for i in range(11):
                p = debug_dir / f"old-{i:02d}.png"
                p.write_bytes(b"png")
                os.utime(p, (1000 + i, 1000 + i))

            mock_factory, _, mock_bctx, mock_page = _make_body_playwright_mock(
                status=500,
                url="https://example.com/err",
            )

            async def _fake_screenshot(path):
                Path(path).write_bytes(b"new-png")

            mock_page.screenshot = AsyncMock(side_effect=_fake_screenshot)

            with patch.object(config, "DATA_DIR", Path(tmpdir)), \
                 patch("playwright.async_api.async_playwright", mock_factory):
                result = _run(
                    scrape_page_body("https://example.com/err", max_chars=100, label="FailTest")
                )

            self.assertFalse(result.success)
            self.assertNotIn("debug", result.output.lower())
            mock_bctx.close.assert_awaited_once()
            mock_page.screenshot.assert_awaited_once()
            remaining = list(debug_dir.glob("*.png"))
            self.assertEqual(len(remaining), 10)
            self.assertFalse((debug_dir / "old-00.png").exists())
            self.assertFalse((debug_dir / "old-01.png").exists())

    def test_browser_headless_false_passed_to_launch(self):
        import config
        from plugins.web_scraper import scrape_page_body

        mock_factory, _, _, _ = _make_body_playwright_mock(body_text="ok")
        pw_instance = mock_factory.return_value.start.return_value

        with patch.object(config, "BROWSER_HEADLESS", False), \
             patch("playwright.async_api.async_playwright", mock_factory):
            result = _run(
                scrape_page_body("https://example.com", max_chars=100, label="Headful")
            )

        self.assertTrue(result.success)
        pw_instance.chromium.launch.assert_awaited_once_with(
            headless=False,
            args=["--disable-gpu"],
        )

    def test_chromium_launch_includes_disable_gpu(self):
        from plugins.web_scraper import scrape_page_body

        mock_factory, _, _, _ = _make_body_playwright_mock(body_text="ok")
        pw_instance = mock_factory.return_value.start.return_value

        with patch("playwright.async_api.async_playwright", mock_factory):
            result = _run(
                scrape_page_body("https://example.com", max_chars=100, label="GpuFlag")
            )

        self.assertTrue(result.success)
        launch_kwargs = pw_instance.chromium.launch.call_args[1]
        self.assertIn("--disable-gpu", launch_kwargs.get("args", []))

    def test_blocked_resource_types_are_aborted(self):
        from plugins.web_scraper import _abort_heavy_resources

        for blocked in ("image", "media", "font"):
            route = MagicMock()
            route.request.resource_type = blocked
            route.abort = AsyncMock()
            route.continue_ = AsyncMock()
            _run(_abort_heavy_resources(route))
            route.abort.assert_awaited_once()
            route.continue_.assert_not_awaited()

        for allowed in ("document", "script", "xhr"):
            route = MagicMock()
            route.request.resource_type = allowed
            route.abort = AsyncMock()
            route.continue_ = AsyncMock()
            _run(_abort_heavy_resources(route))
            route.continue_.assert_awaited_once()
            route.abort.assert_not_awaited()


class TestWeatherPlugin(unittest.TestCase):
    """Test WeatherPlugin metadata, location resolution, and wttr.in scraping."""

    def setUp(self):
        from plugins.p_weather import WeatherPlugin
        self.plugin = WeatherPlugin()

    def tearDown(self):
        from plugins.web_scraper import close_browser
        _run(close_browser())

    def test_schema_and_flags(self):
        self.assertEqual(self.plugin.name, "get_weather")
        self.assertFalse(self.plugin.requires_approval)
        self.assertEqual(self.plugin.parameters_schema["required"], [])
        loc_schema = self.plugin.parameters_schema["properties"]["location"]
        self.assertEqual(loc_schema["type"], "string")
        self.assertEqual(
            loc_schema["description"],
            "City to get the weather for. Leave empty to use the user's home location.",
        )
        for city in ("London", "Paris", "Berlin", "Chennai", "New York", "Tokyo"):
            self.assertNotIn(city.lower(), loc_schema["description"].lower())

    def test_explicit_location_is_url_quoted_and_capped_to_600(self):
        long_body = "New York:\n\nPartly cloudy,   +15C\n" + ("forecast " * 120)
        expected_url = (
            "https://wttr.in/New%20York?"
            "format=%25l:+%25C,+%25t+(feels+%25f),+humidity+%25h,+wind+%25w"
        )
        mock_factory, mock_browser, mock_bctx, mock_page = _make_body_playwright_mock(
            body_text=long_body,
            url=expected_url,
        )
        with patch("playwright.async_api.async_playwright", mock_factory):
            result = _run(self.plugin.execute(location="New York"))

        self.assertTrue(result.success)
        self.assertNotIn("\n", result.output)
        self.assertEqual(len(result.output), 600)
        self.assertTrue(result.output.startswith("New York: Partly cloudy, +15C"))
        self.assertEqual(mock_page.goto.call_args[0][0], expected_url)
        self.assertEqual(result.data["location"], "New York")
        mock_bctx.close.assert_awaited_once()
        mock_browser.close.assert_not_awaited()

    def test_empty_location_uses_user_location_config(self):
        import config

        expected_url = (
            "https://wttr.in/Paris?"
            "format=%25l:+%25C,+%25t+(feels+%25f),+humidity+%25h,+wind+%25w"
        )
        for kwargs in ({}, {"location": "   "}):
            mock_factory, mock_browser, mock_bctx, mock_page = _make_body_playwright_mock(
                body_text="Paris: Clear, +18C (feels +17C), humidity 60%, wind 10km/h",
                url=expected_url,
            )
            with patch.object(config, "USER_LOCATION", "Paris"), \
                 patch("playwright.async_api.async_playwright", mock_factory):
                result = _run(self.plugin.execute(**kwargs))

            self.assertTrue(result.success)
            self.assertEqual(result.data["location"], "Paris")
            self.assertEqual(mock_page.goto.call_args[0][0], expected_url)
            mock_bctx.close.assert_awaited_once()
            mock_browser.close.assert_not_awaited()
            _run(self._reset_shared_browser())

    async def _reset_shared_browser(self):
        from plugins.web_scraper import close_browser
        await close_browser()

    def test_both_empty_asks_for_city_without_launching_browser(self):
        import config

        mock_factory, _, _, _ = _make_body_playwright_mock()
        with patch.object(config, "USER_LOCATION", "   "), \
             patch("playwright.async_api.async_playwright", mock_factory):
            result = _run(self.plugin.execute(location=""))

        self.assertFalse(result.success)
        self.assertEqual(result.output, "Which city should I check?")
        mock_factory.assert_not_called()

    def test_weather_empty_body_returns_error(self):
        mock_factory, mock_browser, mock_bctx, _ = _make_body_playwright_mock(body_text="   \n\n  ")
        with patch("playwright.async_api.async_playwright", mock_factory):
            result = _run(self.plugin.execute(location="Berlin"))

        self.assertFalse(result.success)
        self.assertIn("empty", result.output.lower())
        mock_bctx.close.assert_awaited_once()
        mock_browser.close.assert_not_awaited()

    def test_weather_timeout_returns_error(self):
        from playwright.async_api import TimeoutError as PlaywrightTimeoutError

        mock_factory, mock_browser, mock_bctx, _ = _make_body_playwright_mock(
            goto_side_effect=PlaywrightTimeoutError("Timed out")
        )
        with patch("playwright.async_api.async_playwright", mock_factory):
            result = _run(self.plugin.execute(location="Chennai"))

        self.assertFalse(result.success)
        self.assertIn("timed out", result.output.lower())
        mock_bctx.close.assert_awaited_once()
        mock_browser.close.assert_not_awaited()


def _make_news_playwright_mock(pages_by_url):
    """Create mocked async_playwright supporting per-URL headlines/errors for AINewsPlugin."""

    async def _new_page_factory():
        mock_page = MagicMock()
        mock_page.url = ""
        mock_page.route = AsyncMock()
        mock_page.screenshot = AsyncMock()
        mock_page.title = AsyncMock(return_value="News Page")

        current_items = []

        async def _goto(url, **_kwargs):
            spec = pages_by_url[url]
            if isinstance(spec, Exception):
                raise spec
            mock_page.url = spec.get("final_url", url)
            current_items[:] = spec.get("items", [])
            resp = MagicMock()
            resp.status = spec.get("status", 200)
            return resp

        def _locator(_sel):
            loc = MagicMock()
            loc.count = AsyncMock(return_value=len(current_items))

            def _nth(idx):
                text, href = current_items[idx]
                el = MagicMock()
                el.inner_text = AsyncMock(return_value=text)
                el.get_attribute = AsyncMock(return_value=href)
                return el

            loc.nth = MagicMock(side_effect=_nth)
            return loc

        mock_page.goto = AsyncMock(side_effect=_goto)
        mock_page.locator = MagicMock(side_effect=_locator)
        return mock_page

    mock_browser_ctx = MagicMock()
    mock_browser_ctx.new_page = AsyncMock(side_effect=_new_page_factory)
    mock_browser_ctx.close = AsyncMock()

    mock_browser = MagicMock()
    mock_browser.new_context = AsyncMock(return_value=mock_browser_ctx)
    mock_browser.close = AsyncMock()
    mock_browser.is_connected = MagicMock(return_value=True)

    mock_pw_obj = MagicMock()
    mock_pw_obj.chromium.launch = AsyncMock(return_value=mock_browser)
    mock_pw_obj.stop = AsyncMock()

    mock_ctx = MagicMock()
    mock_ctx.start = AsyncMock(return_value=mock_pw_obj)

    return MagicMock(return_value=mock_ctx), mock_browser, mock_browser_ctx


class TestAINewsPlugin(unittest.TestCase):
    """Test AINewsPlugin metadata, multi-source fallback, and untrusted text cleaning."""

    def setUp(self):
        from plugins.p_ai_news import AINewsPlugin
        self.plugin = AINewsPlugin()

    def tearDown(self):
        from plugins.web_scraper import close_browser
        _run(close_browser())

    def test_schema_and_flags(self):
        self.assertEqual(self.plugin.name, "get_ai_news")
        self.assertFalse(self.plugin.requires_approval)
        self.assertEqual(self.plugin.parameters_schema["type"], "object")
        self.assertEqual(self.plugin.parameters_schema["properties"], {})
        self.assertEqual(self.plugin.parameters_schema["required"], [])

    def test_first_source_works_and_limits_to_5_with_source_line(self):
        import config

        items = [
            (f"Artificial intelligence milestone story number {i}", f"https://techcrunch.com/story-{i}")
            for i in range(1, 8)
        ]
        mock_factory, mock_browser, mock_bctx = _make_news_playwright_mock({
            "https://techcrunch.com/category/artificial-intelligence/": {"items": items},
        })

        with patch.object(
            config,
            "AI_NEWS_URLS",
            [
                "https://techcrunch.com/category/artificial-intelligence/",
                "https://venturebeat.com/category/ai/",
            ],
        ), patch("playwright.async_api.async_playwright", mock_factory):
            result = _run(self.plugin.execute())

        self.assertTrue(result.success)
        lines = result.output.split("\n")
        self.assertEqual(len(lines), 6)
        self.assertEqual(
            lines[0],
            "1. Artificial intelligence milestone story number 1 - https://techcrunch.com/story-1",
        )
        self.assertEqual(
            lines[4],
            "5. Artificial intelligence milestone story number 5 - https://techcrunch.com/story-5",
        )
        self.assertEqual(lines[5], "Source: techcrunch.com")
        mock_bctx.close.assert_awaited_once()
        mock_browser.close.assert_not_awaited()

    def test_first_source_fails_and_second_source_works(self):
        import config
        from playwright.async_api import TimeoutError as PlaywrightTimeoutError

        vb_items = [
            ("Enterprise AI agents orchestration update one", "/ai/story-1"),
            ("Enterprise AI governance data layer story two", "/ai/story-2"),
            ("Enterprise AI inference benchmark story three", "/ai/story-3"),
        ]
        mock_factory, mock_browser, mock_bctx = _make_news_playwright_mock({
            "https://techcrunch.com/category/artificial-intelligence/": PlaywrightTimeoutError("Timeout"),
            "https://venturebeat.com/category/ai/": {
                "final_url": "https://venturebeat.com/category/ai",
                "items": vb_items,
            },
        })

        with patch.object(
            config,
            "AI_NEWS_URLS",
            [
                "https://techcrunch.com/category/artificial-intelligence/",
                "https://venturebeat.com/category/ai/",
            ],
        ), patch("playwright.async_api.async_playwright", mock_factory):
            result = _run(self.plugin.execute())

        self.assertTrue(result.success)
        self.assertIn("https://venturebeat.com/ai/story-1", result.output)
        self.assertIn("Source: venturebeat.com", result.output)
        self.assertEqual(mock_bctx.close.await_count, 2)
        mock_browser.close.assert_not_awaited()

    def test_all_sources_fail_returns_friendly_one_line_error(self):
        import config

        mock_factory, _, mock_bctx = _make_news_playwright_mock({
            "https://example.org/ai-1": {"status": 503, "items": []},
            "https://example.org/ai-2": {
                "items": [("Only one valid headline long enough", "https://example.org/only-1")]
            },
        })

        with patch.object(
            config,
            "AI_NEWS_URLS",
            ["https://example.org/ai-1", "https://example.org/ai-2"],
        ), patch("playwright.async_api.async_playwright", mock_factory):
            result = _run(self.plugin.execute())

        self.assertFalse(result.success)
        self.assertNotIn("\n", result.output)
        self.assertIn("couldn't fetch ai news", result.output.lower())
        self.assertEqual(mock_bctx.close.await_count, 2)

    def test_drops_short_duplicates_and_hash_links_and_cleans_untrusted_text(self):
        import config

        long_title = ("A" * 180) + "\x00\x07\x1b" + ("B" * 80)
        items = [
            ("Too short", "https://example.org/short"),
            ("Headline with hash link that is long enough", "#"),
            ("Headline with empty link that is long enough", ""),
            ("Valid AI headline number one for testing", "https://example.org/1"),
            ("Valid AI headline number one for testing", "https://example.org/1-dup"),
            ("Second\n\t  valid\x07 AI headline for testing", "https://example.org/2"),
            (long_title, "https://example.org/3"),
        ]
        mock_factory, _, _ = _make_news_playwright_mock({
            "https://example.org/ai": {"items": items},
        })

        with patch.object(config, "AI_NEWS_URLS", ["https://example.org/ai"]), \
             patch("playwright.async_api.async_playwright", mock_factory):
            result = _run(self.plugin.execute())

        self.assertTrue(result.success)
        self.assertLessEqual(len(result.output), 1500)
        self.assertNotIn("Too short", result.output)
        self.assertNotIn("#", result.output)
        self.assertNotIn("1-dup", result.output)
        self.assertNotIn("\x00", result.output)
        self.assertNotIn("\x07", result.output)
        self.assertNotIn("\x1b", result.output)
        self.assertIn("2. Second valid AI headline for testing - https://example.org/2", result.output)
        lines = result.output.split("\n")
        third_title = lines[2].split(" - https://example.org/3")[0].removeprefix("3. ")
        self.assertLessEqual(len(third_title), 200)


class TestTelegramHITLCallback(unittest.TestCase):
    """Test callback query handling in main.py for HITL resumption."""

    def test_callback_approve_executes_tool_and_resumes_loop(self):
        import main
        from plugins.base import PluginResult

        mock_orch = MagicMock()
        fake_plugin = MagicMock()
        fake_plugin.execute = AsyncMock(return_value=PluginResult(success=True, output="Email sent."))
        mock_orch.registry = {"send_email": fake_plugin}
        mock_orch.run_tool = AsyncMock(return_value=PluginResult(success=True, output="Email sent."))
        mock_orch.history = MagicMock()
        mock_orch.history.get.return_value = [
            {"role": "user", "content": "email boss"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call_123", "function": {"name": "send_email"}}]},
        ]
        mock_orch.resume_tool_loop = AsyncMock(return_value="Email has been sent!")

        from core.state_manager import state_manager
        mock_orch.state_manager = state_manager
        action = state_manager.create_action(
            tool_name="send_email",
            kwargs={"to_email": "boss@example.com"},
            description="to: boss",
            user_id=999,
        )

        update = MagicMock()
        update.effective_user.id = 999
        query = MagicMock()
        query.from_user.id = 999
        query.data = f"hitl:approve:{action.approval_id}"
        query.message.chat_id = 999
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()
        update.callback_query = query

        context = MagicMock()
        context.bot.send_message = AsyncMock()

        orig_orch = main._orchestrator
        orig_allowed = main.config.ALLOWED_USER_IDS
        main._orchestrator = mock_orch
        main.config.ALLOWED_USER_IDS = [999]
        try:
            _run(main.handle_callback_query(update, context))
            fake_plugin.execute.assert_called_once_with(to_email="boss@example.com")
            mock_orch.history.add.assert_called_once()
            added_msg = mock_orch.history.add.call_args[0][1]
            self.assertEqual(added_msg["role"], "tool")
            self.assertEqual(added_msg["content"], "Email sent.")
            self.assertEqual(added_msg["tool_call_id"], "call_123")
            mock_orch.resume_tool_loop.assert_called_once_with(999)
            context.bot.send_message.assert_called_once()
        finally:
            main._orchestrator = orig_orch
            main.config.ALLOWED_USER_IDS = orig_allowed


class TestAllowlist(unittest.TestCase):
    """Test fail-closed allowlist behavior."""

    def test_empty_allowlist_denies_all(self):
        import main
        orig_allowed = main.config.ALLOWED_USER_IDS
        try:
            main.config.ALLOWED_USER_IDS = []
            self.assertFalse(main._is_allowed(1))
            self.assertFalse(main._is_allowed(999))
            self.assertFalse(main._is_allowed(123456789))
        finally:
            main.config.ALLOWED_USER_IDS = orig_allowed

    def test_listed_id_allowed(self):
        import main
        orig_allowed = main.config.ALLOWED_USER_IDS
        try:
            main.config.ALLOWED_USER_IDS = [100, 200]
            self.assertTrue(main._is_allowed(100))
            self.assertTrue(main._is_allowed(200))
        finally:
            main.config.ALLOWED_USER_IDS = orig_allowed

    def test_unlisted_id_denied(self):
        import main
        orig_allowed = main.config.ALLOWED_USER_IDS
        try:
            main.config.ALLOWED_USER_IDS = [100, 200]
            self.assertFalse(main._is_allowed(300))
            self.assertFalse(main._is_allowed(0))
        finally:
            main.config.ALLOWED_USER_IDS = orig_allowed

    def test_callback_query_from_unlisted_id_denied(self):
        import main
        update = MagicMock()
        query = MagicMock()
        query.from_user.id = 777  # unlisted
        query.data = "hitl:approve:fake_action"
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()
        update.callback_query = query
        context = MagicMock()

        orig_orch = main._orchestrator
        orig_allowed = main.config.ALLOWED_USER_IDS
        mock_orch = MagicMock()
        main._orchestrator = mock_orch
        main.config.ALLOWED_USER_IDS = [999]  # 777 is unlisted
        try:
            _run(main.handle_callback_query(update, context))
            query.answer.assert_called_once()
            query.edit_message_text.assert_called_once_with(main._deny_message())
            # Orchestrator must not be touched
            mock_orch.state_manager.get_action.assert_not_called()
            mock_orch.resume_tool_loop.assert_not_called()
        finally:
            main._orchestrator = orig_orch
            main.config.ALLOWED_USER_IDS = orig_allowed


class TestSystemPrompt(unittest.TestCase):
    """Test system prompt building and de-personalization helper."""

    def test_prompt_helper_substitutes_names_and_memories(self):
        import re
        from core.orchestrator import build_system_prompt
        prompt = build_system_prompt(
            memories="Alice's phone is 555-0123",
            assistant_name="Jarvis",
            user_name="Tony",
        )
        self.assertIn("You are Jarvis, Tony's personal assistant", prompt)
        self.assertIn("WHAT YOU KNOW ABOUT Tony", prompt)
        self.assertIn("Alice's phone is 555-0123", prompt)
        unformatted = re.findall(r"\{[a-zA-Z0-9_]+\}", prompt)
        self.assertEqual(unformatted, [])

    def test_prompt_helper_defaults(self):
        import re
        from core.orchestrator import build_system_prompt
        prompt = build_system_prompt()
        self.assertIn("You are Summer, friend's personal assistant", prompt)
        self.assertIn("WHAT YOU KNOW ABOUT friend", prompt)
        self.assertIn("(nothing yet)", prompt)
        self.assertIn(
            "When a tool returns a list (like headlines), pass it along as a short list instead of summarizing it away.",
            prompt,
        )
        self.assertNotIn("lives in", prompt)
        unformatted = re.findall(r"\{[a-zA-Z0-9_]+\}", prompt)
        self.assertEqual(unformatted, [])

    def test_prompt_helper_includes_user_location_when_set(self):
        import config
        from core.orchestrator import build_system_prompt

        with patch.object(config, "USER_LOCATION", "Paris, France"):
            prompt = build_system_prompt(user_name="Alex")

        self.assertIn(
            "Alex lives in Paris, France. For weather, news or anything local, assume this place unless they name another.",
            prompt,
        )

    def test_prompt_helper_omits_user_location_when_unset(self):
        import config
        from core.orchestrator import build_system_prompt

        with patch.object(config, "USER_LOCATION", "   "):
            prompt = build_system_prompt(user_name="Alex")

        self.assertNotIn("lives in", prompt)
        self.assertNotIn("assume this place", prompt)

    def test_prompt_helper_braces_in_values_are_safe(self):
        import config
        from core.orchestrator import build_system_prompt
        from plugins.base import PluginBase, PluginResult

        class BracedPlugin(PluginBase):
            name = "braced_tool"
            description = "Desc with {curls}"
            usage_hint = "queries containing {json_braces}"
            parameters_schema = {"type": "object", "properties": {}}
            async def execute(self, **kwargs):
                return PluginResult(success=True, output="ok")

        with patch.object(config, "USER_LOCATION", "Berlin {Mitte}"):
            prompt = build_system_prompt(
                memories="Note with {braces}",
                user_name="Sam {K}",
                plugins=[BracedPlugin()],
            )

        self.assertIn("Sam {K} lives in Berlin {Mitte}.", prompt)
        self.assertIn("Note with {braces}", prompt)
        self.assertIn("- braced_tool: use for queries containing {json_braces}", prompt)

    def test_prompt_includes_registered_plugins_and_excludes_disabled(self):
        from core.orchestrator import Orchestrator, build_system_prompt
        from plugins.base import PluginBase, PluginResult

        class FallbackDescPlugin(PluginBase):
            name = "no_hint_tool"
            description = "clean description fallback"
            parameters_schema = {"type": "object", "properties": {}}
            async def execute(self, **kwargs):
                return PluginResult(success=True, output="ok")

        with patch("core.orchestrator.MemoryManager"), \
             patch("plugins.p_gmail.GmailPlugin.is_available", return_value=False):
            orch = Orchestrator()
            orch.registry["no_hint_tool"] = FallbackDescPlugin()
            prompt = build_system_prompt(plugins=orch.registry.values())

        self.assertIn("TOOLS (use them, never guess):", prompt)
        self.assertIn(
            "- get_weather: use for any question about weather, temperature, rain or the forecast",
            prompt,
        )
        self.assertIn(
            "- get_ai_news: use for any request for AI news, headlines or what's new in AI",
            prompt,
        )
        self.assertIn(
            "- search_wikipedia: use for when they ask you to look something up, or mention Wikipedia",
            prompt,
        )
        self.assertIn(
            "- system_info: use for only for questions about this PC's CPU, RAM or disk",
            prompt,
        )
        self.assertIn("- no_hint_tool: use for clean description fallback", prompt)
        self.assertNotIn("- send_email:", prompt)
        self.assertNotIn("- save_memory:", prompt)
        self.assertNotIn("Never call tools for greetings, questions or small talk.", prompt)
        self.assertIn(
            "Never answer weather, news, lookups or PC stats from your own knowledge.",
            prompt,
        )
        self.assertIn(
            "Call the tool again every time, even if you answered something similar earlier.",
            prompt,
        )


class TestConfigEnvExampleSanity(unittest.TestCase):
    """Ensure every environment variable read by config.py is documented in .env.example."""

    def test_env_example_covers_all_config_vars(self):
        import re
        config_path = ROOT / "config.py"
        example_path = ROOT / ".env.example"
        self.assertTrue(config_path.exists())
        self.assertTrue(example_path.exists())

        config_text = config_path.read_text(encoding="utf-8")
        example_text = example_path.read_text(encoding="utf-8")

        # Find variables referenced like os.getenv("FOO"), os.environ.get("FOO"), os.environ["FOO"]
        pattern = r'os\.(?:environ\.get|getenv)\(\s*["\']([A-Z0-9_]+)["\']|os\.environ\[\s*["\']([A-Z0-9_]+)["\']\]'
        vars_in_config = {m[0] or m[1] for m in re.findall(pattern, config_text)}

        # Find variables in .env.example defined as KEY=...
        vars_in_example = set(re.findall(r"^([A-Z0-9_]+)=", example_text, flags=re.MULTILINE))

        missing = vars_in_config - vars_in_example
        self.assertEqual(missing, set(), f"Variables in config.py missing from .env.example: {missing}")

        for expected_web_var in ("USER_LOCATION", "BROWSER_HEADLESS", "AI_NEWS_URLS"):
            self.assertIn(expected_web_var, vars_in_example)

    def test_web_plugin_docs_cover_settings_and_debugging(self):
        repo_root = ROOT.parent
        readme_text = (repo_root / "README.md").read_text(encoding="utf-8")
        plugins_doc_text = (repo_root / "docs" / "PLUGINS.md").read_text(encoding="utf-8")
        example_text = (ROOT / ".env.example").read_text(encoding="utf-8")

        for doc_text in (readme_text, plugins_doc_text, example_text):
            for required in (
                "USER_LOCATION",
                "BROWSER_HEADLESS",
                "AI_NEWS_URLS",
                "playwright install chromium",
                "data/debug",
            ):
                self.assertIn(required, doc_text)


class TestFastAPIApp(unittest.TestCase):
    """Test FastAPI endpoints using TestClient with mocked Telegram network initialization."""

    def _make_mock_app(self):
        mock_app = MagicMock()
        mock_app.initialize = AsyncMock()
        mock_app.start = AsyncMock()
        mock_app.stop = AsyncMock()
        mock_app.shutdown = AsyncMock()
        mock_app.updater.start_polling = AsyncMock()
        mock_app.updater.stop = AsyncMock()
        mock_app.updater.running = False
        mock_app.running = False
        mock_app.bot.delete_webhook = AsyncMock()
        return mock_app

    def test_health_endpoint(self):
        from fastapi.testclient import TestClient
        import main
        with patch.object(main, "_build_telegram_app", return_value=self._make_mock_app()):
            with TestClient(main.app) as client:
                resp = client.get("/health")
                self.assertEqual(resp.status_code, 200)
                self.assertEqual(resp.json()["status"], "ok")

    def test_webhook_unauthorized_token(self):
        from fastapi.testclient import TestClient
        import main
        with patch.object(main, "_build_telegram_app", return_value=self._make_mock_app()):
            with TestClient(main.app) as client:
                resp = client.post("/webhook/wrong-token", json={})
                self.assertEqual(resp.status_code, 403)

    def test_telegram_link_previews_disabled(self):
        from telegram import LinkPreviewOptions
        import main

        mock_bot = MagicMock()
        mock_bot.send_message = AsyncMock(side_effect=[ValueError("bad markdown"), "ok"])
        _run(main._safe_send_message(mock_bot, chat_id=123, text="See https://example.com"))
        self.assertEqual(mock_bot.send_message.await_count, 2)
        for call in mock_bot.send_message.await_args_list:
            opts = call.kwargs["link_preview_options"]
            self.assertIsInstance(opts, LinkPreviewOptions)
            self.assertTrue(opts.is_disabled)

        mock_msg = MagicMock()
        mock_msg.reply_text = AsyncMock(return_value="ok")
        _run(main._safe_reply_text(mock_msg, text="See https://example.com"))
        opts = mock_msg.reply_text.await_args.kwargs["link_preview_options"]
        self.assertIsInstance(opts, LinkPreviewOptions)
        self.assertTrue(opts.is_disabled)


# ============================================================================
# LAYER 4 — File Search, Compression, and Sharing Tests
# ============================================================================

class TestFileSearch(unittest.TestCase):
    """Test plugins/file_search.py security boundaries, two-tier search, and zip/split helpers."""

    def test_is_sensitive_path(self):
        from plugins.file_search import is_sensitive_path
        sensitive = [
            ".env",
            ".env.local",
            ".env.production",
            "project/.env",
            "production.env",
            "server.key",
            "cert.pem",
            "cert.pfx",
            "cert.p12",
            "vpn.ovpn",
            "passwords.kdbx",
            "app.keystore",
            "my_wallet.keystore",
            "id_rsa",
            "id_rsa.pub",
            "id_ed25519",
            "id_ed25519.pub",
            ".ssh/id_rsa",
            ".ssh/config",
            ".gnupg/pubring.kbx",
            ".aws/credentials",
            ".azure/tokens.json",
            ".kube/config",
            ".docker/config.json",
            ".config/gcloud/credentials.db",
            ".git-credentials",
            ".netrc",
            ".npmrc",
            ".pypirc",
            "Login Data",
            "Cookies",
            "Web Data",
            "Local State",
            "key4.db",
            "logins.json",
            "Google/Chrome/User Data/Default/History",
            "Microsoft/Edge/User Data/Default/Cookies",
            "Mozilla/Firefox/Profiles/abc.default/key4.db",
            "BraveSoftware/Brave-Browser/User Data",
            "NTUSER.DAT",
            "C:/Windows/System32/config/SAM",
            "C:/Windows/System32/config/SYSTEM",
            "C:/Windows/System32/config/SECURITY",
            "/etc/shadow",
            "/etc/sudoers",
            "wallet.dat",
            "keystore.json",
            "pc_assistant/credentials/gmail_oauth.json",
            "credentials/gmail_token.json",
            "data/history.db",
            "data/file_index.db",
            "data/memory_db/chroma.sqlite3",
            "whatsapp_session/client.json",
        ]
        for s in sensitive:
            self.assertTrue(is_sensitive_path(s), f"Expected {s} to be sensitive")

        safe = [
            "resume.pdf",
            "quarterly_report.docx",
            "screenshot.png",
            "dataset.csv",
            "notes.txt",
            "src/main.py",
            "security_brief.docx",
            "system_metrics.json",
            "sam_intro.pptx",
        ]
        for s in safe:
            self.assertFalse(is_sensitive_path(s), f"Expected {s} to be safe")

    def test_find_files_direct_path(self):
        from plugins.file_search import find_files
        with tempfile.TemporaryDirectory() as td:
            safe_file = Path(td) / "safe_document.pdf"
            safe_file.write_text("content")
            res = find_files(str(safe_file))
            self.assertEqual(len(res), 1)
            self.assertEqual(res[0].resolve(), safe_file.resolve())

            secret_file = Path(td) / ".env"
            secret_file.write_text("SECRET=1")
            res_blocked = find_files(str(secret_file))
            self.assertEqual(res_blocked, [])

    def test_find_files_fuzzy_tokens(self):
        import config
        from plugins.file_search import find_files
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "annual_financial_report_2026.pdf"
            target.write_text("report")
            other = Path(td) / "other_notes.txt"
            other.write_text("notes")

            with patch.object(config, "FILE_SEARCH_ROOTS", td):
                res = find_files("financial report pdf")
                self.assertGreaterEqual(len(res), 1)
                self.assertEqual(res[0].resolve(), target.resolve())

    def test_find_files_skips_noisy_directories(self):
        import config
        from plugins.file_search import find_files
        with tempfile.TemporaryDirectory() as td:
            git_dir = Path(td) / ".git"
            git_dir.mkdir()
            (git_dir / "invoice.pdf").write_text("hidden")

            node_dir = Path(td) / "node_modules"
            node_dir.mkdir()
            (node_dir / "invoice.pdf").write_text("hidden")

            valid_dir = Path(td) / "Documents"
            valid_dir.mkdir()
            valid_file = valid_dir / "invoice.pdf"
            valid_file.write_text("visible")

            with patch.object(config, "FILE_SEARCH_ROOTS", td):
                res = find_files("invoice pdf")
                self.assertEqual(len(res), 1)
                self.assertEqual(res[0].resolve(), valid_file.resolve())

    def test_compress_to_zip_and_split_chunks(self):
        import os
        import zipfile
        from plugins.file_search import compress_to_zip, split_file_chunks
        with tempfile.TemporaryDirectory() as td:
            test_file = Path(td) / "sample_data.bin"
            test_file.write_bytes(os.urandom(20_000))

            zips_dir = Path(td) / "zips"
            zip_path = compress_to_zip(test_file, zips_dir)
            self.assertTrue(zip_path.exists())
            self.assertTrue(zipfile.is_zipfile(zip_path))

            # Split into 5 KB parts
            parts = split_file_chunks(zip_path, chunk_size_mb=0.005)
            self.assertFalse(zip_path.exists())
            self.assertGreater(len(parts), 1)
            for p in parts:
                self.assertTrue(p.exists())
                self.assertIn(".part", p.name)

    def test_search_aicte_final_avoids_single_token_match(self):
        import config
        from plugins.file_search import find_files
        with tempfile.TemporaryDirectory() as td:
            f1 = Path(td) / "aicte final.docx"
            f1.write_text("aicte final")
            f2 = Path(td) / "AICTE FINAL FRONT.docx"
            f2.write_text("aicte final front")
            f3 = Path(td) / "SPS-Instructions-Final.pdf"
            f3.write_text("instructions")

            with patch.object(config, "FILE_SEARCH_ROOTS", td):
                res = find_files("aicte final file")
                res_names = [p.name for p in res]
                self.assertIn("aicte final.docx", res_names)
                self.assertIn("AICTE FINAL FRONT.docx", res_names)
                self.assertNotIn("SPS-Instructions-Final.pdf", res_names)

    def test_search_hangover_part_2_and_vscode_pruning(self):
        import config
        from plugins.file_search import find_files
        with tempfile.TemporaryDirectory() as td:
            movie = Path(td) / "The Hangover Part II.mp4"
            movie.write_text("movie")

            mp_root = Path(td) / "multipart.py"
            mp_root.write_text("code")

            vscode = Path(td) / ".vscode"
            vscode.mkdir()
            mp_hidden = vscode / "multipart.py"
            mp_hidden.write_text("code")

            with patch.object(config, "FILE_SEARCH_ROOTS", td):
                res = find_files("hangover part 2")
                res_names = [p.name for p in res]
                self.assertIn("The Hangover Part II.mp4", res_names)
                self.assertNotIn("multipart.py", res_names)

    def test_resume_pdf_uses_ext_filter(self):
        import config
        from plugins.file_search import find_files
        with tempfile.TemporaryDirectory() as td:
            pdf = Path(td) / "My_Resume.pdf"
            pdf.write_text("pdf")
            docx = Path(td) / "My_Resume.docx"
            docx.write_text("docx")

            with patch.object(config, "FILE_SEARCH_ROOTS", td):
                res = find_files("resume pdf")
                self.assertEqual(len(res), 1)
                self.assertEqual(res[0].name, "My_Resume.pdf")

    def test_docs_file_in_downloads_applies_root_hint_and_ext_filter(self):
        import config
        from plugins.file_search import find_files
        with tempfile.TemporaryDirectory() as td:
            dl_dir = Path(td) / "Downloads"
            dl_dir.mkdir()
            doc_target = dl_dir / "annual_review.docx"
            doc_target.write_text("doc")
            doc_wrong_ext = dl_dir / "annual_review.pdf"
            doc_wrong_ext.write_text("pdf")

            other_dir = Path(td) / "Documents"
            other_dir.mkdir()
            doc_other_loc = other_dir / "annual_review.docx"
            doc_other_loc.write_text("doc in documents")

            with patch.object(config, "FILE_SEARCH_ROOTS", td):
                res = find_files("docs file in downloads")
                self.assertEqual(len(res), 1)
                self.assertEqual(res[0].resolve(), doc_target.resolve())

    def test_direct_path_with_backslashes_resolves(self):
        from plugins.file_search import find_files
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "sample_doc.txt"
            f.write_text("hello")
            backslash_path = str(f).replace("/", "\\")
            res = find_files(backslash_path)
            self.assertEqual(len(res), 1)
            self.assertEqual(res[0].resolve(), f.resolve())

    def test_tier_b_marks_approximate(self):
        import config
        from plugins.file_search import search_files
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "annual_financial_report.pdf"
            f.write_text("report")

            with patch.object(config, "FILE_SEARCH_ROOTS", td):
                # 3 tokens: annual, financial, budget. 2 match out of 3 = 66% >= 60%
                matches = search_files("annual financial budget")
                self.assertEqual(len(matches), 1)
                self.assertTrue(matches[0].approximate)

    def test_unicode_names_work(self):
        import config
        from plugins.file_search import find_files
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "Übung_Bericht_2.docx"
            f.write_text("bericht")

            with patch.object(config, "FILE_SEARCH_ROOTS", td):
                res = find_files("übung bericht 2")
                self.assertEqual(len(res), 1)
                self.assertEqual(res[0].name, "Übung_Bericht_2.docx")

    def test_roots_per_mode(self):
        import config
        from collections import namedtuple
        from plugins.file_search import roots_for_mode

        # home mode
        home_roots = roots_for_mode("home")
        self.assertEqual(home_roots, [Path.home()])

        # custom mode
        with tempfile.TemporaryDirectory() as td1, tempfile.TemporaryDirectory() as td2:
            custom_roots_str = f"{td1},{td2},/nonexistent/fake/dir/xyz"
            with patch.object(config, "FILE_SEARCH_ROOTS", custom_roots_str):
                custom_roots = roots_for_mode("custom")
                self.assertEqual(len(custom_roots), 2)
                self.assertIn(Path(td1).resolve(), custom_roots)
                self.assertIn(Path(td2).resolve(), custom_roots)

        # all_drives mode
        Partition = namedtuple("Partition", ["device", "mountpoint", "fstype", "opts"])
        with tempfile.TemporaryDirectory() as fake_d:
            fake_partitions = [
                Partition("D:\\", fake_d, "NTFS", "rw,fixed"),
                Partition("E:\\", "E:\\", "iso9660", "ro,cdrom"),
                Partition("Z:\\", "Z:\\", "cifs", "rw,network"),
                Partition("F:\\", fake_d, "FAT32", "rw,removable"),
            ]
            with patch("psutil.disk_partitions", return_value=fake_partitions):
                with patch.object(config, "FILE_INDEX_REMOVABLE", False):
                    roots = roots_for_mode("all_drives")
                    self.assertIn(Path.home(), roots)
                    self.assertIn(Path(fake_d), roots)
                    self.assertNotIn(Path("E:\\"), roots)
                    self.assertNotIn(Path("Z:\\"), roots)

                with patch.object(config, "FILE_INDEX_REMOVABLE", True):
                    roots_rem = roots_for_mode("all_drives")
                    self.assertIn(Path.home(), roots_rem)

    def test_exclusions_applied(self):
        import config
        from plugins.file_search import is_excluded_path

        # Windows system paths & always excluded names
        self.assertTrue(is_excluded_path("C:\\Windows\\System32\\cmd.exe"))
        self.assertTrue(is_excluded_path("C:\\Program Files\\App\\run.exe"))
        self.assertTrue(is_excluded_path("C:\\ProgramData\\file.dat"))
        self.assertTrue(is_excluded_path("D:\\$Recycle.Bin\\deleted.txt"))
        self.assertTrue(is_excluded_path("D:\\System Volume Information\\wps.db"))
        self.assertTrue(is_excluded_path("C:\\Users\\krith\\AppData\\Local\\temp.txt"))
        self.assertTrue(is_excluded_path("project/node_modules/pkg/index.js"))
        self.assertTrue(is_excluded_path("project/.venv/bin/activate"))
        self.assertTrue(is_excluded_path("project/__pycache__/mod.pyc"))

        # Dot-directories
        self.assertTrue(is_excluded_path("home/.config/settings.json"))
        self.assertTrue(is_excluded_path(".hidden_folder/file.txt"))

        # POSIX mounts
        self.assertTrue(is_excluded_path("/proc/cpuinfo"))
        self.assertTrue(is_excluded_path("/sys/devices"))
        self.assertTrue(is_excluded_path("/dev/null"))
        self.assertTrue(is_excluded_path("/etc/passwd"))

        # Custom exclude
        with patch.object(config, "FILE_SEARCH_EXCLUDE", "secret_project,junk_folder"):
            self.assertTrue(is_excluded_path("C:\\Users\\krith\\secret_project\\doc.txt"))
            self.assertTrue(is_excluded_path("/home/user/junk_folder/test.dat"))
            self.assertFalse(is_excluded_path("C:\\Users\\testuser\\other_project\\doc.txt"))

    def test_symlink_or_junction_loop_avoidance(self):
        import config
        from plugins.file_search import is_reparse_point, search_files

        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            sub = base / "subfolder"
            sub.mkdir()
            target_file = sub / "my_secret_report.pdf"
            target_file.write_text("dummy")

            loop_dir = sub / "loop_dir"
            can_symlink = False
            try:
                loop_dir.symlink_to(sub, target_is_directory=True)
                can_symlink = True
            except (OSError, NotImplementedError):
                pass

            if can_symlink:
                self.assertTrue(is_reparse_point(loop_dir))
                with patch.object(config, "FILE_SEARCH_ROOTS", td):
                    with patch.object(config, "FILE_SEARCH_MODE", "custom"):
                        matches = search_files("secret report")
                        self.assertEqual(len(matches), 1)
                        self.assertEqual(matches[0].path.resolve(), target_file.resolve())
            else:
                with patch("plugins.file_search.is_reparse_point", side_effect=lambda p: Path(p).name == "fake_loop"):
                    fake_loop = sub / "fake_loop"
                    fake_loop.mkdir()
                    (fake_loop / "inner.pdf").write_text("inner")
                    with patch.object(config, "FILE_SEARCH_ROOTS", td):
                        with patch.object(config, "FILE_SEARCH_MODE", "custom"):
                            matches = search_files("secret report")
                            self.assertEqual(len(matches), 1)
                            self.assertEqual(matches[0].path.resolve(), target_file.resolve())


class TestFilesPlugins(unittest.TestCase):
    """Test FindFilesPlugin and ShareFilePlugin execution, previews, and routing."""

    def test_find_files_plugin_metadata_and_execute(self):
        import config
        from plugins.p_files import FindFilesPlugin
        plugin = FindFilesPlugin()
        self.assertEqual(plugin.name, "find_files")
        self.assertFalse(plugin.requires_approval)
        self.assertIn("query", plugin.parameters_schema["required"])

        with tempfile.TemporaryDirectory() as td:
            doc = Path(td) / "project_brief.pdf"
            doc.write_text("brief content")

            with patch.object(config, "FILE_SEARCH_ROOTS", td):
                res = _run(plugin.execute(query="project brief"))
                self.assertTrue(res.success)
                self.assertIn("project_brief.pdf", res.output)
                self.assertEqual(len(res.data["matches"]), 1)

    def test_share_file_plugin_metadata(self):
        from plugins.p_files import ShareFilePlugin
        plugin = ShareFilePlugin()
        self.assertEqual(plugin.name, "share_file")
        self.assertTrue(plugin.requires_approval)
        self.assertIn("path", plugin.parameters_schema["required"])
        self.assertIn("query", plugin.parameters_schema.get("properties", {}))

    def test_share_file_prepare_preview(self):
        import config
        from plugins.p_files import ShareFilePlugin
        plugin = ShareFilePlugin()
        with tempfile.TemporaryDirectory() as td:
            doc = Path(td) / "statement.pdf"
            doc.write_text("statement")

            with patch.object(config, "FILE_SEARCH_ROOTS", td):
                preview = _run(plugin.prepare_preview({"query": "statement"}))
                self.assertIsNotNone(preview)
                self.assertIn("• File:", preview)
                self.assertIn("statement.pdf", preview)
                self.assertIn("• Size:", preview)
                self.assertIn("• Action:", preview)

    def test_share_file_telegram_photo_route(self):
        import config
        from plugins.p_files import ShareFilePlugin
        plugin = ShareFilePlugin()
        with tempfile.TemporaryDirectory() as td:
            img = Path(td) / "photo.jpg"
            img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

            with patch.object(config, "FILE_SEARCH_ROOTS", td):
                res = _run(plugin.execute(query="photo.jpg"))
                self.assertTrue(res.success)
                self.assertEqual(len(res.data["send_files"]), 1)
                item = res.data["send_files"][0]
                self.assertEqual(item["kind"], "photo")
                self.assertFalse(item["cleanup"])

    def test_share_file_telegram_document_route(self):
        import config
        from plugins.p_files import ShareFilePlugin
        plugin = ShareFilePlugin()
        with tempfile.TemporaryDirectory() as td:
            doc = Path(td) / "document.pdf"
            doc.write_text("pdf document content")

            with patch.object(config, "FILE_SEARCH_ROOTS", td):
                res = _run(plugin.execute(query="document.pdf"))
                self.assertTrue(res.success)
                self.assertEqual(len(res.data["send_files"]), 1)
                item = res.data["send_files"][0]
                self.assertEqual(item["kind"], "document")
                self.assertFalse(item["cleanup"])

    def test_share_file_folder_auto_zip(self):
        import config
        from plugins.p_files import ShareFilePlugin
        plugin = ShareFilePlugin()
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td) / "project_folder"
            folder.mkdir()
            (folder / "file.txt").write_text("hello")

            with patch.object(config, "FILE_SEARCH_ROOTS", td):
                res = _run(plugin.execute(query=str(folder)))
                self.assertTrue(res.success)
                item = res.data["send_files"][0]
                self.assertEqual(item["kind"], "document")
                self.assertTrue(item["cleanup"])
                self.assertTrue(item["path"].endswith(".zip"))
                self.assertTrue(Path(item["path"]).exists())
                # Clean up test temp zip
                Path(item["path"]).unlink(missing_ok=True)

    def test_share_file_email_route(self):
        import config
        from plugins.p_files import ShareFilePlugin
        from plugins.base import PluginResult
        plugin = ShareFilePlugin()
        with tempfile.TemporaryDirectory() as td:
            doc = Path(td) / "contract.pdf"
            doc.write_text("contract")

            mock_gmail = MagicMock()
            mock_gmail.is_available.return_value = True
            mock_gmail.execute = AsyncMock(return_value=PluginResult(success=True, output="Email sent!"))

            with patch("plugins.p_gmail.GmailPlugin", return_value=mock_gmail):
                with patch.object(config, "FILE_SEARCH_ROOTS", td):
                    res = _run(plugin.execute(query="contract.pdf", delivery="email", to_email="partner@example.com"))
                    self.assertTrue(res.success)
                    mock_gmail.execute.assert_awaited_once()
                    self.assertEqual(mock_gmail.execute.call_args.kwargs["to_email"], "partner@example.com")
                    self.assertIn("contract.pdf", mock_gmail.execute.call_args.kwargs["attachment_path"])

    def test_share_file_default_route_telegram_for_20mb(self):
        import config
        from plugins.p_files import ShareFilePlugin
        from plugins.base import PluginResult
        plugin = ShareFilePlugin()
        with tempfile.TemporaryDirectory() as td:
            doc = Path(td) / "heavy_report.pdf"
            doc.write_bytes(b"0" * 100)

            mock_gmail = MagicMock()
            mock_gmail.is_available.return_value = True
            mock_gmail.execute = AsyncMock(return_value=PluginResult(success=True, output="Email sent!"))

            with patch("plugins.p_gmail.GmailPlugin", return_value=mock_gmail):
                with patch.object(config, "FILE_SEARCH_ROOTS", td):
                    with patch.object(config, "USER_DEFAULT_EMAIL", "default@example.com"):
                        with patch("os.path.getsize", return_value=20 * 1024 * 1024):
                            res = _run(plugin.execute(path=str(doc)))
                            self.assertTrue(res.success)
                            self.assertEqual(len(res.data["send_files"]), 1)
                            mock_gmail.execute.assert_not_called()

    def test_share_file_email_only_when_asked(self):
        import config
        from plugins.p_files import ShareFilePlugin
        from plugins.base import PluginResult
        plugin = ShareFilePlugin()
        with tempfile.TemporaryDirectory() as td:
            doc = Path(td) / "heavy_report.pdf"
            doc.write_bytes(b"0" * 100)

            mock_gmail = MagicMock()
            mock_gmail.is_available.return_value = True
            mock_gmail.execute = AsyncMock(return_value=PluginResult(success=True, output="Email sent!"))

            with patch("plugins.p_gmail.GmailPlugin", return_value=mock_gmail):
                with patch.object(config, "FILE_SEARCH_ROOTS", td):
                    with patch.object(config, "USER_DEFAULT_EMAIL", "default@example.com"):
                        with patch("os.path.getsize", return_value=20 * 1024 * 1024):
                            res = _run(plugin.execute(path=str(doc), delivery="email"))
                            self.assertTrue(res.success)
                            mock_gmail.execute.assert_awaited_once()

    def test_share_file_oversized_split_telegram(self):
        import config
        from plugins.p_files import ShareFilePlugin
        plugin = ShareFilePlugin()
        with tempfile.TemporaryDirectory() as td:
            doc = Path(td) / "large_archive.dat"
            doc.write_bytes(b"0" * 100)

            with patch.object(config, "FILE_SEARCH_ROOTS", td):
                with patch("plugins.p_files.compress_to_zip") as mock_zip:
                    zip_mock = Path(td) / "large_archive.dat.zip"
                    zip_mock.write_bytes(b"123")
                    mock_zip.return_value = zip_mock

                    with patch("plugins.p_files.split_file_chunks") as mock_split:
                        part1 = Path(td) / "part.01"
                        part2 = Path(td) / "part.02"
                        part1.write_bytes(b"1")
                        part2.write_bytes(b"2")
                        mock_split.return_value = [part1, part2]

                        with patch("os.path.getsize", return_value=60 * 1024 * 1024):
                            res = _run(plugin.execute(query="large_archive.dat"))
                            self.assertTrue(res.success)
                            self.assertEqual(len(res.data["send_files"]), 2)
                            self.assertTrue(all(item["cleanup"] for item in res.data["send_files"]))

    def test_share_file_too_large_fails(self):
        import config
        from plugins.p_files import ShareFilePlugin
        plugin = ShareFilePlugin()
        with tempfile.TemporaryDirectory() as td:
            doc = Path(td) / "massive.iso"
            doc.write_bytes(b"0" * 100)

            with patch.object(config, "FILE_SEARCH_ROOTS", td):
                with patch("plugins.p_files.compress_to_zip") as mock_zip:
                    zip_mock = Path(td) / "massive.iso.zip"
                    zip_mock.write_bytes(b"123")
                    mock_zip.return_value = zip_mock

                    with patch("os.path.getsize", return_value=200 * 1024 * 1024):
                        res = _run(plugin.execute(query="massive.iso"))
                        self.assertFalse(res.success)
                        self.assertIn("too large", res.output.lower())

    def test_soft_warning_in_preview(self):
        from plugins.p_files import ShareFilePlugin
        plugin = ShareFilePlugin()

        with tempfile.TemporaryDirectory() as td:
            # File outside usual folders and with sensitive word
            sec_file = Path(td) / "my_secret_token_report.pdf"
            sec_file.write_text("dummy")

            preview = _run(plugin.prepare_preview({"path": str(sec_file)}))
            self.assertIsNotNone(preview)
            self.assertIn("⚠️ Note: file is outside your usual folders", preview)
            self.assertIn("⚠️ Note: filename looks sensitive", preview)

            # Normal file inside fake Documents folder
            docs_dir = Path(td) / "Documents"
            docs_dir.mkdir()
            normal_file = docs_dir / "simple_presentation.pptx"
            normal_file.write_text("presentation")

            preview_normal = _run(plugin.prepare_preview({"path": str(normal_file)}))
            self.assertIsNotNone(preview_normal)
            self.assertNotIn("⚠️ Note: file is outside your usual folders", preview_normal)
            self.assertNotIn("⚠️ Note: filename looks sensitive", preview_normal)

    def test_share_rate_limiter(self):
        import config
        from plugins.file_search import check_and_increment_share_limit, reset_share_rate_limits
        from plugins.p_files import ShareFilePlugin

        reset_share_rate_limits()
        plugin = ShareFilePlugin()
        user_id = 777

        # First 20 shares succeed
        for i in range(20):
            allowed, msg = check_and_increment_share_limit(user_id=user_id)
            self.assertTrue(allowed, f"Share {i+1} should be allowed")
            self.assertEqual(msg, "")

        # 21st share trips the rate limiter
        allowed_21, msg_21 = check_and_increment_share_limit(user_id=user_id)
        self.assertFalse(allowed_21)
        self.assertIn("Rate limit reached (20 files/hr)", msg_21)

        # Execution also returns failure
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "sample.txt"
            f.write_text("data")
            res = _run(plugin.execute(path=str(f), user_id=user_id))
            self.assertFalse(res.success)
            self.assertIn("Rate limit reached", res.output)

        reset_share_rate_limits()

    def test_audit_log_appends_entry(self):
        import config
        from plugins.file_search import log_file_audit

        with tempfile.TemporaryDirectory() as td:
            with patch.object(config, "DATA_DIR", td):
                log_file_audit(
                    user_id=456,
                    action="share",
                    path="test_contract.pdf",
                    size="2.4 MB",
                    route="Telegram",
                    status="approved",
                )
                audit_log = Path(td) / "file_audit.log"
                self.assertTrue(audit_log.exists())
                content = audit_log.read_text(encoding="utf-8")
                self.assertIn("user_id=456", content)
                self.assertIn("action=share", content)
                self.assertIn("test_contract.pdf", content)
                self.assertIn("size=2.4 MB", content)
                self.assertIn("route=Telegram", content)
                self.assertIn("status=approved", content)


class TestGmailPluginAttachment(unittest.TestCase):
    """Test Gmail plugin attachments and 25 MB ceiling."""

    def test_gmail_plugin_with_attachment(self):
        from plugins.p_gmail import GmailPlugin
        plugin = GmailPlugin()

        with tempfile.TemporaryDirectory() as td:
            att = Path(td) / "document.pdf"
            att.write_bytes(b"%PDF-1.4 sample content")

            with patch.object(plugin, "_get_service") as mock_svc:
                mock_send = MagicMock()
                mock_svc.return_value.users.return_value.messages.return_value.send.return_value = mock_send

                res = _run(plugin.execute(
                    to_email="test@example.com",
                    subject="Doc",
                    body="Hello",
                    attachment_path=str(att),
                ))
                self.assertTrue(res.success)
                self.assertIn("Attachment:", res.output)
                self.assertEqual(res.data["attachment"], str(att))
                mock_send.execute.assert_called_once()

    def test_gmail_plugin_attachment_over_25mb_fails(self):
        from plugins.p_gmail import GmailPlugin
        plugin = GmailPlugin()

        with tempfile.TemporaryDirectory() as td:
            att = Path(td) / "huge.pdf"
            att.write_bytes(b"0" * 100)

            with patch("os.path.getsize", return_value=26 * 1024 * 1024):
                res = _run(plugin.execute(
                    to_email="test@example.com",
                    subject="Doc",
                    body="Hello",
                    attachment_path=str(att),
                ))
                self.assertFalse(res.success)
                self.assertIn("25 MB", res.output)


class TestTelegramCallbackFileDelivery(unittest.TestCase):
    """Test handle_callback_query uploading send_files and cleaning up temp archives in finally."""

    def test_callback_sends_files_and_cleans_up(self):
        import main
        from plugins.base import PluginResult

        with tempfile.TemporaryDirectory() as td:
            photo_file = Path(td) / "pic.png"
            photo_file.write_bytes(b"fake photo")

            temp_zip = Path(td) / "archive.zip"
            temp_zip.write_bytes(b"fake zip")

            fake_result = PluginResult(
                success=True,
                output="Files ready.",
                data={
                    "send_files": [
                        {"path": str(photo_file), "filename": "pic.png", "kind": "photo", "cleanup": False},
                        {"path": str(temp_zip), "filename": "archive.zip", "kind": "document", "cleanup": True},
                    ]
                },
            )

            mock_plugin = MagicMock()
            mock_plugin.execute = AsyncMock(return_value=fake_result)

            mock_sm = MagicMock()
            mock_sm.resolve.return_value = {
                "tool_name": "share_file",
                "kwargs": {"query": "test"},
            }

            mock_orch = MagicMock()
            mock_orch.registry = {"share_file": mock_plugin}
            mock_orch.state_manager = mock_sm
            mock_orch.history.get.return_value = []
            mock_orch.resume_tool_loop = AsyncMock(return_value="Here are your files!")

            mock_query = MagicMock()
            mock_query.from_user.id = 1
            mock_query.data = "hitl:approve:action123"
            mock_query.message.chat_id = 123
            mock_query.answer = AsyncMock()
            mock_query.edit_message_text = AsyncMock()

            mock_context = MagicMock()
            mock_context.bot.send_photo = AsyncMock()
            mock_context.bot.send_document = AsyncMock()
            mock_context.bot.send_message = AsyncMock()

            mock_update = MagicMock()
            mock_update.callback_query = mock_query

            with patch.object(main, "_orchestrator", mock_orch):
                with patch.object(main, "_is_allowed", return_value=True):
                    with patch("core.state_manager.state_manager", mock_sm):
                        _run(main.handle_callback_query(mock_update, mock_context))

            # Verified bot sent photo and document
            mock_context.bot.send_photo.assert_awaited_once()
            mock_context.bot.send_document.assert_awaited_once()

            # Verified photo (cleanup=False) remains, while temp_zip (cleanup=True) was deleted
            self.assertTrue(photo_file.exists())
            self.assertFalse(temp_zip.exists())


class TestPhase2DeterministicFileTools(unittest.TestCase):
    """Hermetic unit tests for Phase 2 deterministic file tools, buttons, and direct replies."""

    def test_choices_roundtrip(self):
        from core.choices import ChoiceStore
        cs = ChoiceStore()
        choices = [
            {"label": "1. Report.pdf (1.2 MB)", "tool": "share_file", "kwargs": {"path": "/fake/report.pdf"}},
            {"label": "2. Notes.txt (0.1 MB)", "tool": "share_file", "kwargs": {"path": "/fake/notes.txt"}},
        ]
        token = cs.register(user_id=123, choices=choices, ttl=300)
        self.assertIsInstance(token, str)
        c0 = cs.get(token, user_id=123, index=0)
        self.assertIsNotNone(c0)
        self.assertEqual(c0["kwargs"]["path"], "/fake/report.pdf")
        c1 = cs.get(token, user_id=123, index=1)
        self.assertIsNotNone(c1)
        self.assertEqual(c1["kwargs"]["path"], "/fake/notes.txt")

    def test_choices_expired_or_foreign_token_rejected(self):
        from core.choices import ChoiceStore
        cs = ChoiceStore()
        token = cs.register(user_id=100, choices=[{"label": "A", "tool": "foo", "kwargs": {}}], ttl=-1)
        # Expired
        self.assertIsNone(cs.get(token, user_id=100, index=0))

        # Foreign user
        token2 = cs.register(user_id=100, choices=[{"label": "A", "tool": "foo", "kwargs": {}}], ttl=300)
        self.assertIsNone(cs.get(token2, user_id=999, index=0))

        # Bad index
        self.assertIsNone(cs.get(token2, user_id=100, index=5))
        self.assertIsNone(cs.get(token2, user_id=100, index=-1))

    def test_find_files_schema_has_only_query(self):
        from plugins.p_files import FindFilesPlugin
        p = FindFilesPlugin()
        self.assertEqual(list(p.parameters_schema["properties"].keys()), ["query"])
        self.assertEqual(p.parameters_schema["required"], ["query"])

    def test_share_file_schema_has_path_required(self):
        from plugins.p_files import ShareFilePlugin
        p = ShareFilePlugin()
        self.assertEqual(p.parameters_schema["required"], ["path"])
        self.assertIn("query", p.parameters_schema["properties"])

    def test_orchestrator_returns_choices_reply(self):
        from core.orchestrator import ChoicesReply, Orchestrator
        from plugins.base import PluginResult
        orch = Orchestrator(llm=MagicMock())
        fake_result = PluginResult(
            success=True,
            output="Found files",
            data={
                "direct_reply": "1. Report.pdf (1.2 MB)\nTap one to send it.",
                "choices": [{"label": "1. Report.pdf (1.2 MB)", "tool": "share_file", "kwargs": {"path": "/fake/path.pdf"}}],
            },
        )
        fake_plugin = MagicMock()
        fake_plugin.name = "find_files"
        fake_plugin.requires_approval = False
        fake_plugin.parameters_schema = {"type": "object", "properties": {"query": {"type": "string"}}}
        fake_plugin.execute = AsyncMock(return_value=fake_result)
        orch.registry = {"find_files": fake_plugin}

        orch.llm.chat = AsyncMock(return_value={
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "find_files", "arguments": {"query": "report"}}}],
        })

        reply = _run(orch.handle_user_message(user_id=1, chat_id=1, message="find report"))
        self.assertIsInstance(reply, ChoicesReply)
        self.assertIn("Tap one to send it.", reply.text)
        self.assertEqual(len(reply.labels), 1)
        self.assertIn("1. Report.pdf", reply.labels[0])
        # Checked choice registered in choice_store
        stored = orch.choice_store.get(reply.token, user_id=1, index=0)
        self.assertIsNotNone(stored)
        self.assertEqual(stored["kwargs"]["path"], "/fake/path.pdf")

    def test_direct_reply_skips_second_model_call_and_lands_in_history(self):
        from core.orchestrator import Orchestrator
        from plugins.base import PluginResult
        orch = Orchestrator(llm=MagicMock())
        fake_result = PluginResult(
            success=True,
            output="Searched",
            data={"direct_reply": "No match for 'xyz'. Looked in: Downloads."},
        )
        fake_plugin = MagicMock()
        fake_plugin.name = "find_files"
        fake_plugin.requires_approval = False
        fake_plugin.parameters_schema = {"type": "object", "properties": {"query": {"type": "string"}}}
        fake_plugin.execute = AsyncMock(return_value=fake_result)
        orch.registry = {"find_files": fake_plugin}

        orch.llm.chat = AsyncMock(return_value={
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "find_files", "arguments": {"query": "xyz"}}}],
        })

        reply = _run(orch.handle_user_message(user_id=1, chat_id=1, message="find xyz"))
        self.assertEqual(reply, "No match for 'xyz'. Looked in: Downloads.")
        # Only one chat call made, follow-up call skipped!
        orch.llm.chat.assert_awaited_once()

        # Check history contains user, assistant tool_call, tool result, and assistant direct_reply
        hist = orch.history.get(1)
        self.assertEqual(hist[-1]["role"], "assistant")
        self.assertEqual(hist[-1]["content"], "No match for 'xyz'. Looked in: Downloads.")

    def test_pick_callback_to_approval_card_with_exact_path(self):
        import main
        from core.choices import choice_store
        token = choice_store.register(
            user_id=42,
            choices=[{"label": "1. Report.pdf", "tool": "share_file", "kwargs": {"path": "/exact/path/report.pdf"}}],
        )

        mock_query = MagicMock()
        mock_query.from_user.id = 42
        mock_query.data = f"pick:{token}:0"
        mock_query.answer = AsyncMock()
        mock_query.edit_message_text = AsyncMock()

        mock_update = MagicMock()
        mock_update.callback_query = mock_query

        mock_orch = MagicMock()
        mock_orch.request_tool_approval = AsyncMock(return_value={
            "text": "Action requires approval for /exact/path/report.pdf",
            "reply_markup": MagicMock(),
        })

        with patch.object(main, "_orchestrator", mock_orch):
            with patch.object(main, "_is_allowed", return_value=True):
                _run(main.handle_callback_query(mock_update, MagicMock()))

        mock_orch.request_tool_approval.assert_awaited_once_with(42, "share_file", {"path": "/exact/path/report.pdf"})
        mock_query.edit_message_text.assert_awaited_once()
        call_kwargs = mock_query.edit_message_text.call_args[0]
        self.assertIn("/exact/path/report.pdf", call_kwargs[0])

    def test_orchestrator_file_has_no_telegram_import(self):
        orch_path = Path(__file__).resolve().parent.parent / "core" / "orchestrator.py"
        with open(orch_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("import ") or line.startswith("from "):
                    self.assertNotIn("telegram", line, f"Orchestrator must not import telegram: {line}")


class TestPhase3ReliableDelivery(unittest.TestCase):
    """Hermetic unit tests for Phase 3 timeouts, splitting, and delivery."""

    def test_upload_timeouts_passed_to_telegram(self):
        import main
        import config
        from plugins.base import PluginResult
        fake_plugin = MagicMock()
        fake_result = PluginResult(
            success=True,
            output="Sent",
            data={
                "send_files": [
                    {"path": "dummy.pdf", "filename": "dummy.pdf", "kind": "document", "cleanup": False}
                ]
            },
        )
        fake_plugin.execute = AsyncMock(return_value=fake_result)
        mock_sm = MagicMock()
        mock_sm.resolve.return_value = {"tool_name": "share_file", "kwargs": {}}
        mock_orch = MagicMock()
        mock_orch.registry = {"share_file": fake_plugin}
        mock_orch.state_manager = mock_sm
        mock_orch.history.get.return_value = []
        mock_orch.resume_tool_loop = AsyncMock(return_value="Done")

        mock_query = MagicMock()
        mock_query.from_user.id = 1
        mock_query.data = "hitl:approve:123"
        mock_query.message.chat_id = 123
        mock_query.answer = AsyncMock()
        mock_query.edit_message_text = AsyncMock()

        mock_context = MagicMock()
        mock_context.bot.send_chat_action = AsyncMock()
        mock_context.bot.send_document = AsyncMock()
        mock_context.bot.send_message = AsyncMock()

        mock_update = MagicMock()
        mock_update.callback_query = mock_query

        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=b"data")):
                with patch.object(main, "_orchestrator", mock_orch):
                    with patch.object(main, "_is_allowed", return_value=True):
                        with patch.object(config, "FILE_UPLOAD_TIMEOUT_S", 720.0):
                            _run(main.handle_callback_query(mock_update, mock_context))

        mock_context.bot.send_chat_action.assert_awaited()
        mock_context.bot.send_document.assert_awaited_once()
        kwargs = mock_context.bot.send_document.call_args[1]
        self.assertEqual(kwargs.get("read_timeout"), 720.0)
        self.assertEqual(kwargs.get("write_timeout"), 720.0)
        self.assertEqual(kwargs.get("connect_timeout"), 720.0)
        self.assertEqual(kwargs.get("pool_timeout"), 720.0)

    def test_timedout_stops_later_parts_and_reports(self):
        import main
        import telegram.error
        from plugins.base import PluginResult
        fake_plugin = MagicMock()
        fake_result = PluginResult(
            success=True,
            output="Sent",
            data={
                "send_files": [
                    {"path": "part1.dat", "filename": "part1.dat", "kind": "document", "cleanup": True},
                    {"path": "part2.dat", "filename": "part2.dat", "kind": "document", "cleanup": True},
                    {"path": "part3.dat", "filename": "part3.dat", "kind": "document", "cleanup": True},
                ]
            },
        )
        fake_plugin.execute = AsyncMock(return_value=fake_result)
        mock_sm = MagicMock()
        mock_sm.resolve.return_value = {"tool_name": "share_file", "kwargs": {}}
        mock_orch = MagicMock()
        mock_orch.registry = {"share_file": fake_plugin}
        mock_orch.state_manager = mock_sm
        mock_orch.history.get.return_value = []
        mock_orch.resume_tool_loop = AsyncMock(return_value="Done")

        mock_query = MagicMock()
        mock_query.from_user.id = 1
        mock_query.data = "hitl:approve:123"
        mock_query.message.chat_id = 123
        mock_query.answer = AsyncMock()
        mock_query.edit_message_text = AsyncMock()

        mock_context = MagicMock()
        mock_context.bot.send_chat_action = AsyncMock()
        # Raise TimedOut on first part
        mock_context.bot.send_document = AsyncMock(side_effect=telegram.error.TimedOut("Connection timed out"))
        mock_context.bot.send_message = AsyncMock()

        mock_update = MagicMock()
        mock_update.callback_query = mock_query

        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=b"data")):
                with patch.object(main, "_orchestrator", mock_orch):
                    with patch.object(main, "_is_allowed", return_value=True):
                        _run(main.handle_callback_query(mock_update, mock_context))

        # Stopped after part 1: send_document called only once, later parts not attempted
        self.assertEqual(mock_context.bot.send_document.await_count, 1)
        # Sent timeout message
        reported = False
        for call in mock_context.bot.send_message.call_args_list:
            if "Telegram timed out on part 1" in str(call):
                reported = True
                break
        self.assertTrue(reported)

    def test_docx_over_50mb_split_raw_no_zip(self):
        import config
        from plugins.p_files import ShareFilePlugin
        plugin = ShareFilePlugin()
        with tempfile.TemporaryDirectory() as td:
            docx = Path(td) / "big_report.docx"
            docx.write_bytes(b"x" * 100)

            with patch.object(config, "FILE_SEARCH_ROOTS", td):
                with patch("plugins.p_files.compress_to_zip") as mock_zip:
                    with patch("plugins.p_files.split_file_chunks") as mock_split:
                        mock_split.return_value = [Path(td) / "big_report.docx.part01", Path(td) / "big_report.docx.part02"]
                        with patch("os.path.getsize", return_value=60 * 1024 * 1024):
                            res = _run(plugin.execute(path=str(docx)))
                            self.assertTrue(res.success)
                            # Zipping was SKIPPED for .docx
                            mock_zip.assert_not_called()
                            # split_file_chunks was called directly
                            mock_split.assert_called_once()
                            self.assertEqual(len(res.data["send_files"]), 2)
                            self.assertIn("copy /b", res.output)

    def test_text_directory_is_zipped(self):
        import config
        from plugins.p_files import ShareFilePlugin
        plugin = ShareFilePlugin()
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td) / "text_project"
            folder.mkdir()
            (folder / "notes.txt").write_text("notes")

            with patch.object(config, "FILE_SEARCH_ROOTS", td):
                with patch("plugins.p_files.compress_to_zip") as mock_zip:
                    zip_out = Path(td) / "text_project.zip"
                    zip_out.write_bytes(b"zip")
                    mock_zip.return_value = zip_out
                    with patch("os.path.getsize", return_value=5 * 1024 * 1024):
                        res = _run(plugin.execute(path=str(folder)))
                        self.assertTrue(res.success)
                        mock_zip.assert_called_once()
                        self.assertEqual(res.data["send_files"][0]["path"], str(zip_out))

    def test_startup_cleanup_removes_old_temp_files(self):
        import config
        from plugins.file_search import cleanup_stale_tmp_shares
        with tempfile.TemporaryDirectory() as td:
            tmp_shares = Path(td) / "tmp_shares"
            tmp_shares.mkdir()
            old_file = tmp_shares / "old_part.01"
            old_file.write_text("old")
            new_file = tmp_shares / "new_part.01"
            new_file.write_text("new")

            # Set mtime of old_file to 48 hours ago
            old_mtime = time.time() - (48 * 3600)
            os.utime(str(old_file), (old_mtime, old_mtime))

            with patch.object(config, "DATA_DIR", td):
                removed = cleanup_stale_tmp_shares(max_age_s=86400.0)
                self.assertEqual(removed, 1)
                self.assertFalse(old_file.exists())
                self.assertTrue(new_file.exists())


class TestPhase4HonestyAndFormatting(unittest.TestCase):
    """Hermetic unit tests for Phase 4 tool honesty and plain text / HTML formatting."""

    def test_system_prompt_includes_honesty_rule_and_tool_hints(self):
        from core.orchestrator import build_system_prompt
        from plugins.p_files import FindFilesPlugin, ShareFilePlugin
        prompt = build_system_prompt(plugins=[FindFilesPlugin(), ShareFilePlugin()])
        self.assertIn(
            "Never say you found, opened, sent or emailed anything unless a tool result says so. You cannot open files on this PC.",
            prompt,
        )
        self.assertIn(
            "find_files: use for when they ask to search, find, or locate files by keyword",
            prompt,
        )
        self.assertIn(
            "share_file: use for when they ask to send, share, upload, or email a specific file or folder (pass the path or filename)",
            prompt,
        )

    def test_empty_model_reply_logs_warning_and_returns_friendly_rephrase(self):
        from core.orchestrator import Orchestrator
        orch = Orchestrator(llm=MagicMock())
        orch.llm.chat = AsyncMock(return_value={
            "role": "assistant",
            "content": "",
            "tool_calls": None,
            "done_reason": "stop",
        })
        with self.assertLogs("core.orchestrator", level="WARNING") as cm:
            reply = _run(orch.handle_user_message(user_id=1, chat_id=1, message="hello"))
            self.assertEqual(reply, "I didn't catch that. Can you rephrase?")
            self.assertTrue(any("Empty model reply" in log_msg for log_msg in cm.output))
            self.assertTrue(any("done_reason=stop" in log_msg for log_msg in cm.output))

    def test_backslash_normalized_in_text_sent_to_model(self):
        from core.orchestrator import Orchestrator
        orch = Orchestrator(llm=MagicMock())
        orch.llm.chat = AsyncMock(return_value={"role": "assistant", "content": "Found it!"})
        _run(orch.handle_user_message(user_id=1, chat_id=1, message=r"Aicte final, docs file in krith\downloads"))
        sent_msgs = orch.llm.chat.call_args[0][0]
        user_msg = next(m["content"] for m in sent_msgs if m.get("role") == "user")
        self.assertNotIn(r"krith\downloads", user_msg)
        self.assertIn("krith/downloads", user_msg)

    def test_approval_card_html_escaped_preserves_underscores(self):
        import main
        from telegram.constants import ParseMode
        filename = "Ashish_Resume_final.pdf"
        preview = f"• File: {filename}\n• Size: 1.5 MB\n• Route: Telegram"
        prompt_dict = main.build_hitl_prompt(
            approval_id="test1234",
            preview_text=preview,
            tool_name="share_file",
        )
        self.assertEqual(prompt_dict["parse_mode"], ParseMode.HTML)
        # Underscores in filename are preserved intact without Markdown italics delimiters
        self.assertIn("Ashish_Resume_final.pdf", prompt_dict["text"])
        self.assertNotIn("<em>", prompt_dict["text"])
        self.assertNotIn("<i>Ashish", prompt_dict["text"])
        self.assertIn("<b>Tool:</b> <code>share_file</code>", prompt_dict["text"])


class TestFileIndex(unittest.TestCase):
    """Hermetic unit tests for background file indexer, generational cleanup, and Telegram commands."""

    def test_crawl_populates_db_and_ignores_sensitive_and_exclusions(self):
        import sqlite3
        from plugins.file_index import FileIndexer

        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            db_path = base / "index.db"
            indexer = FileIndexer(db_path=db_path)

            # Valid files
            (base / "annual_report_2026.pdf").write_text("report")
            (base / "user_notes.txt").write_text("notes")
            sub = base / "projects"
            sub.mkdir()
            (sub / "analysis_sheet.xlsx").write_text("sheet")

            # Sensitive and excluded files
            (base / ".env").write_text("SECRET=1")
            (base / "server.key").write_text("KEY")
            ssh_dir = base / ".ssh"
            ssh_dir.mkdir()
            (ssh_dir / "id_rsa").write_text("RSA")
            node_dir = base / "node_modules"
            node_dir.mkdir()
            (node_dir / "pkg.js").write_text("code")

            indexed_count = indexer.crawl_sync(roots=[base])
            self.assertGreaterEqual(indexed_count, 3)

            conn = sqlite3.connect(str(db_path))
            rows = conn.execute("SELECT name, ext, is_dir FROM files").fetchall()
            names = [r[0] for r in rows]
            conn.close()

            # Valid items are indexed
            self.assertIn("annual_report_2026.pdf", names)
            self.assertIn("user_notes.txt", names)
            self.assertIn("analysis_sheet.xlsx", names)

            # Sensitive / excluded items are NOT indexed
            self.assertNotIn(".env", names)
            self.assertNotIn("server.key", names)
            self.assertNotIn("id_rsa", names)
            self.assertNotIn("pkg.js", names)

    def test_deleted_file_removed_on_next_crawl_gen(self):
        import sqlite3
        from plugins.file_index import FileIndexer

        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            db_path = base / "index.db"
            indexer = FileIndexer(db_path=db_path)

            file_a = base / "keep_forever.pdf"
            file_a.write_text("keep")
            file_b = base / "temporary_draft.docx"
            file_b.write_text("draft")

            indexer.crawl_sync(roots=[base])

            conn = sqlite3.connect(str(db_path))
            names1 = [r[0] for r in conn.execute("SELECT name FROM files").fetchall()]
            conn.close()
            self.assertIn("keep_forever.pdf", names1)
            self.assertIn("temporary_draft.docx", names1)

            # Delete file_b from disk
            file_b.unlink()

            # Second crawl should purge file_b via generational cleanup
            indexer.crawl_sync(roots=[base])

            conn = sqlite3.connect(str(db_path))
            names2 = [r[0] for r in conn.execute("SELECT name FROM files").fetchall()]
            conn.close()
            self.assertIn("keep_forever.pdf", names2)
            self.assertNotIn("temporary_draft.docx", names2)

    def test_query_fts5_and_like_fallback(self):
        from plugins.file_index import FileIndexer

        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            db_path = base / "index.db"
            indexer = FileIndexer(db_path=db_path)

            doc = base / "annual_budget_summary.pdf"
            doc.write_text("budget")

            indexer.crawl_sync(roots=[base])

            # Query via FTS5
            res_fts = indexer.search(["budget", "annual"], ext_filters=[".pdf"], force_like=False)
            self.assertEqual(len(res_fts), 1)
            self.assertEqual(res_fts[0].path.name, "annual_budget_summary.pdf")

            # Query via LIKE fallback
            res_like = indexer.search(["budget", "annual"], ext_filters=[".pdf"], force_like=True)
            self.assertEqual(len(res_like), 1)
            self.assertEqual(res_like[0].path.name, "annual_budget_summary.pdf")

    def test_fast_livescan_finds_file_created_after_index(self):
        import config
        from plugins.file_index import FileIndexer
        from plugins.file_search import FileMatch, search_files

        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            db_path = base / "index.db"
            indexer = FileIndexer(db_path=db_path)
            downloads = base / "Downloads"
            downloads.mkdir()

            initial_file = downloads / "existing_doc.pdf"
            initial_file.write_text("existing")

            # Run initial crawl
            indexer.crawl_sync(roots=[base])

            # Create brand new file AFTER crawl has completed
            fresh_file = downloads / "urgent_invoice_new.pdf"
            fresh_file.write_text("invoice")

            with patch("plugins.file_index.file_indexer", indexer):
                with patch("plugins.file_search._scan_tier1_fast", return_value=[
                    FileMatch(path=fresh_file, size=10, mtime=time.time(), approximate=False)
                ]):
                    matches = search_files("urgent invoice new")
                    self.assertGreaterEqual(len(matches), 1)
                    self.assertEqual(matches[0].path.resolve(), fresh_file.resolve())

    def test_index_command_and_reindex_command(self):
        import config
        import main
        from plugins.file_index import FileIndexer

        with tempfile.TemporaryDirectory() as td:
            indexer = FileIndexer(db_path=Path(td) / "test.db")
            (Path(td) / "sample.pdf").write_text("sample")
            indexer.crawl_sync(roots=[Path(td)])

            status_msg = indexer.format_status_message()
            self.assertIn("Indexed", status_msg)
            self.assertIn("files.", status_msg)
            self.assertIn("Last crawl:", status_msg)
            self.assertIn("Mode:", status_msg)

            # Test command handlers
            mock_update = MagicMock()
            mock_update.effective_user.id = 123456789
            mock_update.message.reply_text = AsyncMock()
            mock_context = MagicMock()

            with patch.object(config, "ALLOWED_USER_IDS", [123456789]):
                with patch("plugins.file_index.file_indexer", indexer):
                    _run(main.cmd_index(mock_update, mock_context))
                    mock_update.message.reply_text.assert_called()
                    last_call_text = mock_update.message.reply_text.call_args[0][0]
                    self.assertIn("Indexed", last_call_text)

                    with patch.object(indexer, "reindex") as mock_reindex:
                        _run(main.cmd_reindex(mock_update, mock_context))
                        mock_reindex.assert_called_once()
                        mock_update.message.reply_text.assert_called_with("Reindex started.")


class TestPhase1FileIndex(unittest.TestCase):
    """Hermetic unit tests for Phase 1: cloud placeholders, diagnostics, skip counters, no slow walks."""

    def test_reparse_junction_skipped_and_cloud_placeholder_indexed(self):
        from plugins.file_index import FileIndexer, is_link_or_junction

        fake_stat_junction = MagicMock()
        fake_stat_junction.st_file_attributes = 0x400
        fake_stat_junction.st_reparse_tag = 0xA0000003  # Mount point / junction
        entry_junction = MagicMock()
        entry_junction.is_symlink.return_value = False
        entry_junction.stat.return_value = fake_stat_junction

        fake_stat_cloud = MagicMock()
        fake_stat_cloud.st_file_attributes = 0x400
        fake_stat_cloud.st_reparse_tag = 0x9000001A  # Cloud placeholder (OneDrive)
        fake_stat_cloud.st_size = 1234
        fake_stat_cloud.st_mtime = 1700000000.0
        entry_cloud = MagicMock()
        entry_cloud.is_symlink.return_value = False
        entry_cloud.stat.return_value = fake_stat_cloud

        self.assertTrue(is_link_or_junction(entry_junction, fake_stat_junction))
        self.assertFalse(is_link_or_junction(entry_cloud, fake_stat_cloud))

        # Test during crawl_sync: junction skipped, cloud placeholder indexed
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            db_path = base / "index.db"
            indexer = FileIndexer(db_path=db_path)

            file_normal = base / "cloud_file.docx"
            file_normal.write_text("dummy")

            class MockScandirContext:
                def __init__(self, entries):
                    self._entries = entries
                def __enter__(self):
                    return iter(self._entries)
                def __exit__(self, *args):
                    pass

            def mock_scandir(path_str):
                fake_cloud_entry = MagicMock()
                fake_cloud_entry.name = "cloud_file.docx"
                fake_cloud_entry.path = str(file_normal)
                fake_cloud_entry.is_symlink.return_value = False
                fake_cloud_entry.stat.return_value = fake_stat_cloud
                fake_cloud_entry.is_dir.return_value = False

                fake_junction_entry = MagicMock()
                fake_junction_entry.name = "fake_junction"
                fake_junction_entry.path = str(base / "fake_junction")
                fake_junction_entry.is_symlink.return_value = False
                fake_junction_entry.stat.return_value = fake_stat_junction
                fake_junction_entry.is_dir.return_value = True

                return MockScandirContext([fake_cloud_entry, fake_junction_entry])

            with patch("os.scandir", side_effect=mock_scandir):
                indexed_count = indexer.crawl_sync(roots=[base])

            self.assertEqual(indexed_count, 1)
            self.assertEqual(indexer.skip_counts["link_or_junction"], 1)

    def test_skip_counters_increment_for_each_reason(self):
        from plugins.file_index import FileIndexer

        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            db_path = base / "index.db"
            indexer = FileIndexer(db_path=db_path)

            # Excluded: .hidden_dotfile and node_modules folder
            (base / ".hidden_dotfile").write_text("hidden")
            nm = base / "node_modules"
            nm.mkdir()
            (nm / "dep.js").write_text("js")

            # Sensitive: server.key
            (base / "server.key").write_text("secret_key")

            # Link or junction
            fake_stat_junction = MagicMock()
            fake_stat_junction.st_file_attributes = 0x400
            fake_stat_junction.st_reparse_tag = 0xA0000003

            # Permission error: simulated subdirectory
            perm_dir = base / "locked_dir"
            perm_dir.mkdir()

            orig_scandir = os.scandir

            class MockScandirContext:
                def __init__(self, entries):
                    self._entries = entries
                def __enter__(self):
                    return iter(self._entries)
                def __exit__(self, *args):
                    pass

            def mock_scandir(path_str):
                p = Path(path_str)
                if p.name == "locked_dir":
                    raise PermissionError("Access denied")
                with orig_scandir(path_str) as it:
                    entries = list(it)
                if p == base:
                    fake_j = MagicMock()
                    fake_j.name = "symlink_dir"
                    fake_j.path = str(base / "symlink_dir")
                    fake_j.is_symlink.return_value = False
                    fake_j.stat.return_value = fake_stat_junction
                    fake_j.is_dir.return_value = True
                    entries.append(fake_j)
                return MockScandirContext(entries)

            with patch("os.scandir", side_effect=mock_scandir):
                indexer.crawl_sync(roots=[base])

            self.assertGreaterEqual(indexer.skip_counts["excluded"], 1)
            self.assertGreaterEqual(indexer.skip_counts["sensitive"], 1)
            self.assertGreaterEqual(indexer.skip_counts["link_or_junction"], 1)
            self.assertGreaterEqual(indexer.skip_counts["permission_error"], 1)

    def test_index_reply_text_contains_per_root_counts_and_skip_counts(self):
        from plugins.file_index import FileIndexer

        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            db_path = base / "index.db"
            indexer = FileIndexer(db_path=db_path)

            doc_dir = base / "Documents"
            doc_dir.mkdir()
            (doc_dir / "paper1.pdf").write_text("p1")
            (doc_dir / "paper2.pdf").write_text("p2")

            dl_dir = base / "Downloads"
            dl_dir.mkdir()
            (dl_dir / "archive.zip").write_text("zip")

            indexer.crawl_sync(roots=[base])
            msg = indexer.format_status_message()

            self.assertIn("Documents: 2", msg)
            self.assertIn("Downloads: 1", msg)
            self.assertIn("duration:", msg)
            self.assertIn("Skipped: excluded: 0, sensitive: 0, link_or_junction: 0, permission_error: 0", msg)

    def test_tier2_walk_not_called_when_index_is_ready(self):
        from plugins.file_index import CallableBool, FileIndexer
        from plugins.file_search import search_files

        mock_indexer = MagicMock(spec=FileIndexer)
        mock_indexer.has_completed_crawl = CallableBool(True)
        mock_indexer.is_ready.return_value = True
        mock_indexer.search.return_value = []

        with patch("plugins.file_index.file_indexer", mock_indexer):
            with patch("plugins.file_search._scan_tier1_fast", return_value=[]):
                with patch("plugins.file_search._scan_home_wide") as mock_home_wide:
                    matches = search_files("nonexistent_rare_file_query")
                    self.assertEqual(matches, [])
                    # Verify Tier 2 walk is NOT called
                    mock_home_wide.assert_not_called()

        # Conversely, verify Tier 2 walk IS called when index has not completed
        mock_indexer_unready = MagicMock(spec=FileIndexer)
        mock_indexer_unready.has_completed_crawl = CallableBool(False)
        mock_indexer_unready.is_ready.return_value = False

        with patch("plugins.file_index.file_indexer", mock_indexer_unready):
            with patch("plugins.file_search._scan_root"):
                with patch("plugins.file_search._scan_home_wide") as mock_home_wide:
                    search_files("nonexistent_rare_file_query")
                    mock_home_wide.assert_called_once()


class TestPhase2SearchV3(unittest.TestCase):
    """Hermetic unit tests for Phase 2: Search v3 (locations, folders, typos, honest misses, denylist, format_size)."""

    def test_parse_query_locations_and_aliases(self):
        from plugins.file_search import parse_query

        # Location preposition and deep path
        pq = parse_query("tree.txt in documents/Coding Research/dsa")
        self.assertEqual(pq.tokens, ["tree"])
        self.assertEqual(pq.ext_filter, [".txt"])
        self.assertFalse(pq.want_dir)
        self.assertEqual(pq.location_terms, ["documents", "coding", "research", "dsa"])

        # Folder keyword sets want_dir
        pq_dir = parse_query("potential wallpapers folder")
        self.assertEqual(pq_dir.tokens, ["potential", "wallpapers"])
        self.assertTrue(pq_dir.want_dir)
        self.assertNotIn("folder", pq_dir.tokens)

        # Standalone alias extraction with filler tokens removed
        pq_fill = parse_query("send me aicte final file in downloads")
        self.assertEqual(pq_fill.tokens, ["aicte", "final"])
        self.assertEqual(pq_fill.location_terms, ["downloads"])

        # Underscore splitting
        pq_under = parse_query("aicte_final")
        self.assertEqual(pq_under.tokens, ["aicte", "final"])

    def test_parse_query_fuzzy_vocab_and_typos(self):
        from plugins.file_search import parse_query

        # "filder" fuzzy matches "folder" -> want_dir=True
        pq_filder = parse_query("filder")
        self.assertTrue(pq_filder.want_dir)
        self.assertEqual(pq_filder.corrections.get("filder"), "folder")

        # "shre" fuzzy matches "share" -> dropped as filler
        pq_shre = parse_query("shre my notes.txt")
        self.assertEqual(pq_shre.corrections.get("shre"), "share")
        self.assertEqual(pq_shre.tokens, ["notes"])
        self.assertEqual(pq_shre.ext_filter, [".txt"])

        # "tree.rxt" extension typo -> .txt
        pq_ext = parse_query("tree.rxt")
        self.assertIn(".txt", pq_ext.ext_filter)
        self.assertEqual(pq_ext.corrections.get(".rxt"), ".txt")

        # Name typo tolerance with index vocab
        custom_vocab = {"wallpaper", "financial", "report"}
        pq_typo = parse_query("wallpapr", vocab=custom_vocab)
        self.assertEqual(pq_typo.tokens, ["wallpaper"])
        self.assertEqual(pq_typo.corrections.get("wallpapr"), "wallpaper")

    def test_location_terms_in_order_matching(self):
        import config
        from plugins.file_search import search_files

        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            # Create subfolder chain: Documents / Coding Research / dsa / tree.txt
            target_dir = base / "Documents" / "Coding Research" / "dsa"
            target_dir.mkdir(parents=True)
            target_file = target_dir / "tree.txt"
            target_file.write_text("correct target")

            # Create other folder: Desktop / tree.txt
            other_dir = base / "Desktop"
            other_dir.mkdir(parents=True)
            other_file = other_dir / "tree.txt"
            other_file.write_text("wrong target")

            with patch.object(config, "FILE_SEARCH_ROOTS", str(base)), \
                 patch.object(config, "FILE_SEARCH_MODE", "custom"):
                matches = search_files("tree.txt in documents/Coding Research/dsa")
                self.assertEqual(len(matches), 1)
                self.assertEqual(matches[0].path.resolve(), target_file.resolve())

    def test_two_tier_ranking_tier_a_vs_tier_b(self):
        import config
        from plugins.file_search import search_files

        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            f_all = base / "annual_financial_report_2026.pdf"
            f_all.write_text("all tokens")
            f_partial = base / "annual_summary_2026.pdf"
            f_partial.write_text("partial tokens")

            with patch.object(config, "FILE_SEARCH_ROOTS", str(base)), \
                 patch.object(config, "FILE_SEARCH_MODE", "custom"):
                # Tier A: all tokens matched (annual, financial, 2026)
                matches_a = search_files("annual financial 2026 pdf")
                self.assertEqual(len(matches_a), 1)
                self.assertEqual(matches_a[0].path.resolve(), f_all.resolve())
                self.assertFalse(matches_a[0].approximate)

                # Tier B: 2 of 3 tokens matched (annual, 2026, audit) -> Tier A has 0, Tier B returns matches
                matches_b = search_files("annual audit 2026 pdf")
                self.assertGreaterEqual(len(matches_b), 1)
                self.assertTrue(any(m.approximate for m in matches_b))

    def test_denylist_early_rejection(self):
        from plugins.file_search import check_denylist_rejection, search_files
        from plugins.p_files import FindFilesPlugin

        denial_pw = check_denylist_rejection("send me password.txt")
        self.assertIsNotNone(denial_pw)
        self.assertIn("protected or sensitive files", denial_pw)

        denial_key = check_denylist_rejection("id_rsa key")
        self.assertIsNotNone(denial_key)
        self.assertIn("id_rsa", denial_key)

        denial_env = check_denylist_rejection("find .env file")
        self.assertIsNotNone(denial_env)
        self.assertIn(".env", denial_env)

        # search_files rejects denylist
        self.assertEqual(search_files("my password.txt"), [])

        # FindFilesPlugin execute rejects early
        plugin = FindFilesPlugin()
        res = _run(plugin.execute(query="find id_rsa"))
        self.assertFalse(res.success)
        self.assertIn("protected or sensitive files", res.output)

    def test_honest_direct_reply_on_miss(self):
        from plugins.file_search import format_miss_reply
        from plugins.p_files import FindFilesPlugin

        reply = format_miss_reply("nonexistent_rare_artifact_xyz")
        self.assertIn("No matching files found for 'nonexistent_rare_artifact_xyz'", reply)
        self.assertIn("searched", reply)
        self.assertIn("indexed files across", reply)

        plugin = FindFilesPlugin()
        res = _run(plugin.execute(query="nonexistent_rare_artifact_xyz"))
        self.assertFalse(res.success)
        self.assertIn("No matching files found for 'nonexistent_rare_artifact_xyz'", res.output)

    def test_format_size_never_zero_mb(self):
        from plugins.file_search import format_size

        self.assertEqual(format_size(500), "500 B")
        self.assertEqual(format_size(3200), "3.1 KB")
        self.assertNotIn("0.0 MB", format_size(3200))
        self.assertEqual(format_size(15 * 1024 * 1024), "15.0 MB")
        self.assertEqual(format_size(2 * 1024 * 1024 * 1024), "2.00 GB")


class TestListFolder(unittest.TestCase):
    """Test ListFolderPlugin directory browsing, paging, security filtering, and inline callback handling."""

    def setUp(self):
        from plugins.p_folder import ListFolderPlugin
        self.plugin = ListFolderPlugin()

    def test_schema_and_flags(self):
        self.assertEqual(self.plugin.name, "list_folder")
        self.assertFalse(self.plugin.requires_approval)
        self.assertEqual(self.plugin.usage_hint, "when they ask what is in a folder or drive")
        self.assertEqual(self.plugin.parameters_schema["type"], "object")

    def test_list_folder_paging_and_navigation(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "work"
            base.mkdir()
            # 2 directories
            (base / "alpha_dir").mkdir()
            (base / "beta_dir").mkdir()
            # 10 files
            for i in range(1, 11):
                p = base / f"doc_{i:02d}.txt"
                p.write_text(f"content {i}")
                # stagger mtimes
                os.utime(p, (1000 + i * 10, 1000 + i * 10))

            # Page 1: offset=0
            res1 = _run(self.plugin.execute(path=str(base), offset=0))
            self.assertTrue(res1.success)
            choices1 = res1.data["choices"]
            labels1 = [c["label"] for c in choices1]

            # Up navigation choice
            self.assertEqual(labels1[0], "⬆ Up")
            self.assertEqual(choices1[0]["tool"], "list_folder")
            self.assertEqual(choices1[0]["kwargs"]["path"], str(base.parent.resolve()))

            # Dirs first, alphabetical
            self.assertEqual(labels1[1], "📁 alpha_dir")
            self.assertEqual(choices1[1]["tool"], "list_folder")
            self.assertEqual(labels1[2], "📁 beta_dir")
            self.assertEqual(choices1[2]["tool"], "list_folder")

            # Followed by files (newest first: doc_10 down to doc_05)
            self.assertEqual(labels1[3], "📄 doc_10.txt")
            self.assertEqual(choices1[3]["tool"], "share_file")

            # More navigation choice at the end
            self.assertEqual(labels1[-1], "➡ More")
            self.assertEqual(choices1[-1]["tool"], "list_folder")
            self.assertEqual(choices1[-1]["kwargs"]["offset"], 8)

            # Total items on page 1: 1 Up + 8 items + 1 More = 10 choices
            self.assertEqual(len(choices1), 10)

            # Page 2: offset=8
            res2 = _run(self.plugin.execute(path=str(base), offset=8))
            self.assertTrue(res2.success)
            choices2 = res2.data["choices"]
            labels2 = [c["label"] for c in choices2]

            self.assertEqual(labels2[0], "⬆ Up")
            # 4 remaining files
            self.assertEqual(len(choices2), 1 + 4)  # Up + 4 files, no More
            self.assertNotIn("➡ More", labels2)

    def test_sensitive_and_excluded_hidden_from_browse(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td) / "projects"
            folder.mkdir()
            (folder / "allowed_dir").mkdir()
            (folder / "normal_file.txt").write_text("ok")

            # Excluded / sensitive items
            (folder / ".git").mkdir()
            (folder / "node_modules").mkdir()
            (folder / "id_rsa").write_text("secret")
            (folder / ".env").write_text("secret")

            res = _run(self.plugin.execute(path=str(folder)))
            self.assertTrue(res.success)
            choices = res.data["choices"]
            labels = [c["label"] for c in choices]

            self.assertIn("📁 allowed_dir", labels)
            self.assertIn("📄 normal_file.txt", labels)
            self.assertNotIn("📁 .git", labels)
            self.assertNotIn("📁 node_modules", labels)
            self.assertNotIn("📄 id_rsa", labels)
            self.assertNotIn("📄 .env", labels)

    def test_sensitive_or_excluded_folder_access_blocked(self):
        with tempfile.TemporaryDirectory() as td:
            secret_folder = Path(td) / ".ssh"
            secret_folder.mkdir()

            res = _run(self.plugin.execute(path=str(secret_folder)))
            self.assertFalse(res.success)
            self.assertIn("restricted", res.output.lower())

    def test_empty_path_lists_roots(self):
        with tempfile.TemporaryDirectory() as td:
            r1 = Path(td) / "Disk1"
            r2 = Path(td) / "Disk2"
            r1.mkdir()
            r2.mkdir()

            with patch("plugins.p_folder.roots_for_mode", return_value=[r1, r2]):
                res = _run(self.plugin.execute(path=""))
                self.assertTrue(res.success)
                choices = res.data["choices"]
                labels = [c["label"] for c in choices]
                self.assertIn(f"📁 {r1.name}", labels)
                self.assertIn(f"📁 {r2.name}", labels)

    def test_callback_query_inline_execution_for_non_approval(self):
        import config
        import main
        from core.choices import choice_store

        with tempfile.TemporaryDirectory() as td:
            test_dir = Path(td) / "browse_me"
            test_dir.mkdir()
            (test_dir / "sample.txt").write_text("hello")

            token = choice_store.register(
                123456,
                [{"label": "📁 browse_me", "tool": "list_folder", "kwargs": {"path": str(test_dir)}}],
            )

            mock_update = MagicMock()
            mock_query = MagicMock()
            mock_query.from_user.id = 123456
            mock_query.data = f"pick:{token}:0"
            mock_query.answer = AsyncMock()
            mock_query.edit_message_text = AsyncMock()
            mock_update.callback_query = mock_query
            mock_context = MagicMock()

            orig_orch = main._orchestrator
            mock_orch = MagicMock()
            mock_orch.registry = {"list_folder": self.plugin}
            main._orchestrator = mock_orch

            with patch.object(config, "ALLOWED_USER_IDS", [123456]):
                try:
                    _run(main.handle_callback_query(mock_update, mock_context))
                    mock_query.answer.assert_called_once()
                    mock_query.edit_message_text.assert_called_once()
                    args, kwargs = mock_query.edit_message_text.call_args
                    self.assertIn("📂", args[0])
                    self.assertIn("sample.txt", args[0])
                    # Reply markup must have inline buttons
                    self.assertIn("reply_markup", kwargs)
                    # request_tool_approval must NOT have been called
                    mock_orch.request_tool_approval.assert_not_called()
                finally:
                    main._orchestrator = orig_orch


class TestPhase1FolderAliases(unittest.TestCase):
    """Test resolve_user_path and list_folder drive/alias resolution."""

    def test_folder_aliases_and_hallucinated_user_paths(self):
        from plugins.file_search import resolve_user_path

        with tempfile.TemporaryDirectory() as td:
            fake_home = Path(td) / "fake_home"
            fake_home.mkdir()
            dl = fake_home / "Downloads"
            dl.mkdir()
            report = dl / "report.txt"
            report.write_text("my report")

            with patch("pathlib.Path.home", return_value=fake_home):
                # 1. Direct folder alias "Downloads"
                p1 = resolve_user_path("Downloads")
                self.assertIsNotNone(p1)
                self.assertEqual(p1.resolve(), dl.resolve())

                # 2. Lowercase / user-prefixed "user/downloads"
                p2 = resolve_user_path("user/downloads")
                self.assertIsNotNone(p2)
                self.assertEqual(p2.resolve(), dl.resolve())

                # 3. "friend/downloads"
                p3 = resolve_user_path("friend/downloads")
                self.assertIsNotNone(p3)
                self.assertEqual(p3.resolve(), dl.resolve())

                # 4. Relative path "Downloads/report.txt"
                p4 = resolve_user_path("Downloads/report.txt")
                self.assertIsNotNone(p4)
                self.assertEqual(p4.resolve(), report.resolve())

                # 5. Hallucinated Windows path "C:\\Users\\friend\\Downloads\\report.txt"
                p5 = resolve_user_path(r"C:\Users\friend\Downloads\report.txt")
                self.assertIsNotNone(p5)
                self.assertEqual(p5.resolve(), report.resolve())

                # 6. Non-existent path returns None
                self.assertIsNone(resolve_user_path("NonExistentFolder12345"))

    def test_missing_drive_reports_available_roots(self):
        from plugins.p_folder import ListFolderPlugin

        plugin = ListFolderPlugin()
        with tempfile.TemporaryDirectory() as td:
            r1 = Path(td) / "DiskC"
            r1.mkdir()
            with patch("plugins.p_folder.roots_for_mode", return_value=[r1]):
                res = _run(plugin.execute(path="Z:/"))
                self.assertFalse(res.success)
                self.assertIn("Drive Z: was not found", res.output)
                self.assertIn("Available roots:", res.output)
                self.assertIn(r1.name, res.output)
                self.assertEqual(res.data["choices"], [])

    def test_find_files_accepts_path_alias(self):
        import config
        from plugins.p_files import FindFilesPlugin

        plugin = FindFilesPlugin()
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "target.pdf"
            f.write_text("data")
            with patch.object(config, "FILE_SEARCH_ROOTS", td):
                res = _run(plugin.execute(path="target.pdf"))
                self.assertTrue(res.success)
                self.assertIn("target.pdf", res.output)


class TestPhase2ShareFileResolution(unittest.TestCase):
    """Test smart share_file query resolution, pre-approval validation, and oversized limits."""

    def test_resolve_share_target_variations(self):
        import config
        from plugins.file_search import resolve_share_target

        with tempfile.TemporaryDirectory() as td:
            fake_home = Path(td) / "home"
            fake_home.mkdir()
            dl = fake_home / "Downloads"
            dl.mkdir()
            doc = dl / "resume.pdf"
            doc.write_text("my resume")

            # Ambiguous files in td
            a1 = dl / "report_2025.txt"
            a2 = dl / "report_2026.txt"
            a1.write_text("2025")
            a2.write_text("2026")

            with patch("pathlib.Path.home", return_value=fake_home):
                with patch.object(config, "FILE_SEARCH_ROOTS", str(dl)):
                    # 1. Non-existent file
                    path, matches, err = resolve_share_target("ghost_file_xyz.doc")
                    self.assertIsNone(path)
                    self.assertEqual(matches, [])
                    self.assertIn("No file found matching", err)

                    # 2. Existing via alias / hallucinated path
                    path, matches, err = resolve_share_target(r"C:\Users\friend\Downloads\resume.pdf")
                    self.assertIsNone(err)
                    self.assertIsNotNone(path)
                    self.assertEqual(path.resolve(), doc.resolve())

                    # 3. Stem match
                    path, matches, err = resolve_share_target("resume")
                    self.assertIsNone(err)
                    self.assertIsNotNone(path)
                    self.assertEqual(path.resolve(), doc.resolve())

                    # 4. Ambiguous match returns multiple matches, no path
                    path, matches, err = resolve_share_target("report")
                    self.assertIsNone(path)
                    self.assertIsNone(err)
                    self.assertGreaterEqual(len(matches), 2)

    def test_orchestrator_share_file_hallucinated_path_resolves_before_approval(self):
        import config
        from core.orchestrator import Orchestrator
        from core.state_manager import StateManager

        with tempfile.TemporaryDirectory() as td:
            fake_home = Path(td) / "home"
            fake_home.mkdir()
            dl = fake_home / "Downloads"
            dl.mkdir()
            real_file = dl / "thesis.pdf"
            real_file.write_text("my thesis")

            mock_llm = MagicMock()
            mock_llm.chat = AsyncMock(return_value={
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_1",
                    "function": {
                        "name": "share_file",
                        "arguments": {"path": r"C:\Users\wronguser\Downloads\thesis.pdf"},
                    },
                }],
            })

            sm = StateManager()
            orch = Orchestrator(llm=mock_llm, state_manager=sm)
            orch.set_hitl_builder(lambda approval_id, preview_text, tool_name: {
                "text": preview_text,
                "reply_markup": MagicMock(),
            })

            with patch("pathlib.Path.home", return_value=fake_home):
                with patch.object(config, "FILE_SEARCH_ROOTS", str(dl)):
                    res = _run(orch.handle_user_message(10, 10, "send my thesis"))
                    self.assertIsInstance(res, dict)
                    self.assertIn("thesis.pdf", res["text"])
                    # State manager should have recorded the REAL resolved path
                    approval_id = sm._user_index.get(10)
                    pending = sm.get_action(approval_id) if approval_id else None
                    self.assertIsNotNone(pending)
                    self.assertEqual(pending.kwargs["path"], str(real_file.resolve()))
                    self.assertEqual(orch.last_file_path[10], str(real_file.resolve()))

    def test_orchestrator_share_file_nonexistent_returns_direct_reply_without_hitl(self):
        from core.orchestrator import Orchestrator
        from core.state_manager import StateManager

        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value={
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_1",
                "function": {
                    "name": "share_file",
                    "arguments": {"path": "ghost_document.pdf"},
                },
            }],
        })

        sm = StateManager()
        orch = Orchestrator(llm=mock_llm, state_manager=sm)
        orch.set_hitl_builder(lambda approval_id, preview_text, tool_name: {
            "text": preview_text,
            "reply_markup": MagicMock(),
        })

        res = _run(orch.handle_user_message(20, 20, "send ghost document"))
        self.assertIsInstance(res, str)
        self.assertIn("No file found matching", res)
        # Verify NO pending approval was created
        self.assertNotIn(20, sm._user_index)

    def test_orchestrator_share_file_ambiguous_returns_choices(self):
        import config
        from core.orchestrator import ChoicesReply, Orchestrator
        from core.state_manager import StateManager

        with tempfile.TemporaryDirectory() as td:
            f1 = Path(td) / "contract_v1.docx"
            f2 = Path(td) / "contract_v2.docx"
            f1.write_text("v1")
            f2.write_text("v2")

            mock_llm = MagicMock()
            mock_llm.chat = AsyncMock(return_value={
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_1",
                    "function": {
                        "name": "share_file",
                        "arguments": {"path": "contract"},
                    },
                }],
            })

            sm = StateManager()
            orch = Orchestrator(llm=mock_llm, state_manager=sm)
            with patch.object(config, "FILE_SEARCH_ROOTS", td):
                res = _run(orch.handle_user_message(30, 30, "send contract"))
                self.assertIsInstance(res, ChoicesReply)
                self.assertIn("contract_v1.docx", res.text)
                self.assertIn("contract_v2.docx", res.text)
                self.assertNotIn(30, sm._user_index)

    def test_oversized_file_error_message_no_email_suggestion(self):
        import config
        from plugins.p_files import ShareFilePlugin

        plugin = ShareFilePlugin()
        with tempfile.TemporaryDirectory() as td:
            doc = Path(td) / "big_dump.iso"
            doc.write_bytes(b"0" * 100)

            with patch.object(config, "FILE_SEARCH_ROOTS", td):
                with patch("plugins.p_files.compress_to_zip") as mock_zip:
                    zip_mock = Path(td) / "big_dump.iso.zip"
                    zip_mock.write_bytes(b"123")
                    mock_zip.return_value = zip_mock

                    with patch("os.path.getsize", return_value=500 * 1024 * 1024):
                        res = _run(plugin.execute(query="big_dump.iso"))
                        self.assertFalse(res.success)
                        self.assertIn("exceeds the maximum transfer limit", res.output)
                        self.assertNotIn("consider using email for files under 25 mb", res.output.lower())


class TestPhase3SmartEmailResolution(unittest.TestCase):
    """Test smart recipient and attachment resolution for send_email and share_file."""

    def test_resolve_recipient_email(self):
        import config
        from plugins.p_gmail import resolve_recipient_email

        with patch.object(config, "USER_DEFAULT_EMAIL", "default@company.org"):
            # Empty / self
            self.assertEqual(resolve_recipient_email(""), "default@company.org")
            self.assertEqual(resolve_recipient_email("me"), "default@company.org")
            self.assertEqual(resolve_recipient_email("self"), "default@company.org")

            # Placeholders
            self.assertEqual(resolve_recipient_email("friend@example.com"), "default@company.org")
            self.assertEqual(resolve_recipient_email("user@example.com"), "default@company.org")

            # Explicit 3rd party addresses preserved
            self.assertEqual(resolve_recipient_email("partner@example.com"), "partner@example.com")
            self.assertEqual(resolve_recipient_email("test@example.com"), "test@example.com")
            self.assertEqual(resolve_recipient_email("colleague@somewhere.com"), "colleague@somewhere.com")

    def test_send_email_resolves_relative_attachment_path(self):
        import config
        from plugins.p_gmail import GmailPlugin

        plugin = GmailPlugin()
        with tempfile.TemporaryDirectory() as td:
            fake_home = Path(td) / "fake_home"
            fake_home.mkdir()
            dl = fake_home / "Downloads"
            dl.mkdir()
            f = dl / "tree.txt"
            f.write_text("oak and maple")

            with patch("pathlib.Path.home", return_value=fake_home):
                with patch.object(plugin, "_get_service") as mock_svc:
                    mock_send = MagicMock()
                    mock_svc.return_value.users.return_value.messages.return_value.send.return_value = mock_send

                    res = _run(plugin.execute(
                        to_email="test@example.com",
                        subject="Tree File",
                        body="Here is tree",
                        attachment_path="Downloads/tree.txt",
                    ))
                    self.assertTrue(res.success)
                    self.assertIn("Attachment:", res.output)
                    self.assertIn("tree.txt", res.data["attachment"])
                    self.assertIn("direct_reply", res.data)
                    mock_send.execute.assert_called_once()

    def test_send_email_directory_auto_zips(self):
        import config
        from plugins.p_gmail import GmailPlugin

        plugin = GmailPlugin()
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td) / "wallpapers"
            folder.mkdir()
            (folder / "mountain.png").write_text("img")

            with patch.object(plugin, "_get_service") as mock_svc:
                mock_send = MagicMock()
                mock_svc.return_value.users.return_value.messages.return_value.send.return_value = mock_send

                res = _run(plugin.execute(
                    to_email="test@example.com",
                    subject="Wallpapers",
                    body="Here are wallpapers",
                    attachment_path=str(folder),
                ))
                self.assertTrue(res.success)
                self.assertIn("wallpapers.zip", res.output)

    def test_send_email_oversized_self_falls_back_to_telegram(self):
        import config
        from plugins.p_gmail import GmailPlugin

        plugin = GmailPlugin()
        with tempfile.TemporaryDirectory() as td:
            doc = Path(td) / "heavy_project.dat"
            doc.write_bytes(b"0" * 100)

            with patch.object(config, "USER_DEFAULT_EMAIL", "default@company.org"):
                with patch("os.path.getsize", return_value=35 * 1024 * 1024):
                    res = _run(plugin.execute(
                        to_email="me",
                        subject="Big File",
                        body="File body",
                        attachment_path=str(doc),
                    ))
                    self.assertTrue(res.success)
                    self.assertIn("exceeds Gmail's 25 MB limit, so I sent it here on Telegram instead", res.output)
                    self.assertIn("send_files", res.data)
                    self.assertEqual(len(res.data["send_files"]), 1)
                    self.assertEqual(res.data["send_files"][0]["path"], str(doc))

    def test_send_email_blocks_sensitive_file(self):
        from plugins.p_gmail import GmailPlugin

        plugin = GmailPlugin()
        with tempfile.TemporaryDirectory() as td:
            sec = Path(td) / "id_rsa.key"
            sec.write_text("private")

            res = _run(plugin.execute(
                to_email="test@example.com",
                subject="Key",
                body="Secret",
                attachment_path=str(sec),
            ))
            self.assertFalse(res.success)
            self.assertIn("Access denied", res.output)


class TestPhase4TelegramAwarenessAndNoDoubleReply(unittest.TestCase):
    """Test Telegram self-awareness in prompt and no double-reply on HITL approval."""

    def test_system_prompt_contains_telegram_awareness(self):
        from core.orchestrator import build_system_prompt
        prompt = build_system_prompt()
        self.assertIn("directly inside Telegram right now", prompt)
        self.assertIn("The file will be sent right here in this Telegram chat", prompt)

    def test_hitl_approval_with_direct_reply_skips_resume_tool_loop(self):
        import main
        from core.orchestrator import Orchestrator
        from core.state_manager import StateManager
        from plugins.base import PluginBase, PluginResult

        class DummySharePlugin(PluginBase):
            name = "dummy_share"
            description = "Dummy share plugin"
            requires_approval = True
            async def execute(self, **kwargs):
                return PluginResult(
                    success=True,
                    output="File sent successfully",
                    data={"direct_reply": "Sent: photo.jpg via Telegram."},
                )

        sm = StateManager()
        orch = Orchestrator(llm=MagicMock(), state_manager=sm)
        orch.registry["dummy_share"] = DummySharePlugin()
        orch.resume_tool_loop = AsyncMock()

        orig_orch = main._orchestrator
        main._orchestrator = orch

        try:
            # Setup realistic history before approval
            orch.history.add(99, "user", "send photo.jpg")
            orch.history.add(99, {"role": "assistant", "tool_calls": [{"id": "call_1", "function": {"name": "dummy_share"}}]})

            action = sm.create_action("dummy_share", {}, "desc", user_id=99)
            approval_id = action.approval_id

            mock_update = MagicMock()
            mock_update.callback_query.data = f"hitl:approve:{approval_id}"
            mock_update.callback_query.from_user.id = 99
            mock_update.callback_query.answer = AsyncMock()
            mock_update.callback_query.edit_message_text = AsyncMock()
            mock_update.callback_query.message.chat_id = 12345

            mock_context = MagicMock()
            mock_context.bot.send_message = AsyncMock()

            with patch("main._is_allowed", return_value=True):
                _run(main.handle_callback_query(mock_update, mock_context))

            # 1. resume_tool_loop must NOT be called
            orch.resume_tool_loop.assert_not_called()

            # 2. History must contain the direct_reply
            hist = orch.history.get(99)
            assistant_replies = [m.get("content") for m in hist if m.get("role") == "assistant"]
            self.assertIn("Sent: photo.jpg via Telegram.", assistant_replies)

            # 3. Message sent to user
            mock_context.bot.send_message.assert_called_once()
            self.assertEqual(mock_context.bot.send_message.call_args.kwargs["text"], "Sent: photo.jpg via Telegram.")
        finally:
            main._orchestrator = orig_orch


if __name__ == "__main__":
    unittest.main(verbosity=2)
