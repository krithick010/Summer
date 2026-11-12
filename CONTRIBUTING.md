# Contributing to Summer PC Assistant

Thank you for your interest in contributing! Here are quick guidelines to get started.

## Running Tests

All changes must pass the test suite before submitting a pull request:

```bash
cd pc_assistant
python -u tests/test_suite.py
```

## Hermetic Test Guidelines

- Tests must be strictly hermetic.
- Never make real network calls or reach out to actual external services.
- Always mock external APIs (Ollama, ChromaDB, Telegram, Google/Gmail).
- Use temporary directories (`tempfile.mkdtemp()`) or in-memory databases (`HISTORY_DB_PATH=":memory:"`) so local files and user data are never touched.

## Evaluating Tool Calling Locally (Dev)

When iterating on prompts, tool schemas, or local models with a running Ollama instance, use the standalone tool-evaluation harness (not run in CI):

```bash
cd pc_assistant
python scripts/tool_check.py
python scripts/tool_check.py --model qwen3:4b
```

`scripts/tool_check.py` runs canned prompts through the real `Orchestrator` against your live Ollama server while stubbing plugin `execute()` calls (no browser, Gmail, or network requests) and writing only to a temporary `DATA_DIR` with `:memory:` history.

## Adding a Plugin

1. Create a new module inside `pc_assistant/plugins/` prefixed with `p_` (e.g., `p_weather.py`).
2. Subclass `PluginBase` from `plugins.base`.
3. Define the required class attributes:
   - `name`: unique snake_case string identifier.
   - `description`: clear explanation of when the model should invoke the tool.
   - `parameters_schema`: JSON schema dictionary defining accepted keyword arguments.
   - `requires_approval`: set to `True` for sensitive/destructive operations to enforce Human-in-the-Loop confirmation.
4. Implement `is_available(self) -> bool` to gracefully disable the plugin when optional dependencies or API keys are missing.
5. Implement `async def execute(self, **kwargs) -> PluginResult`.
6. Add unit tests for your plugin in `pc_assistant/tests/test_suite.py`.

