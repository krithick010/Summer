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
from unittest.mock import AsyncMock, MagicMock, patch

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
        for field in ("to_email", "subject", "body"):
            self.assertIn(field, props)


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
        unformatted = re.findall(r"\{[a-zA-Z0-9_]+\}", prompt)
        self.assertEqual(unformatted, [])


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
