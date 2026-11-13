# Writing Custom Plugins for Summer

Summer uses an auto-discovering plugin system. Any Python module in `pc_assistant/plugins/` named `p_*.py` that subclasses `PluginBase` is discovered and registered when the application starts.

---

## 1. Minimal Plugin Example

Below is a complete, minimal plugin demonstrating all core attributes and methods:

```python
from typing import Any
from plugins.base import PluginBase, PluginResult


class NoteTakerPlugin(PluginBase):
    # Unique snake_case identifier used in LLM tool calling
    name = "take_note"

    # Injected into the model's tool definitions so it knows when to invoke it
    description = "Append a short note or reminder to the user's daily scratchpad."

    # JSON Schema definition of kwargs passed to execute()
    parameters_schema = {
        "type": "object",
        "properties": {
            "note": {
                "type": "string",
                "description": "The note content to record",
            }
        },
        "required": ["note"],
    }

    # When True, the user must approve the action in Telegram before execution
    requires_approval = False

    def is_available(self) -> bool:
        """Check whether prerequisites, libraries, or credentials exist."""
        # Optional: return False and set self.unavailable_reason if unavailable
        return True

    async def execute(self, **kwargs: Any) -> PluginResult:
        note = kwargs.get("note", "").strip()
        if not note:
            return PluginResult(success=False, output="Empty note provided.", error="missing_note")

        # Perform the actual action here
        return PluginResult(success=True, output=f"Saved note: {note}")
```

---

## 2. Auto-Discovery

- Place your plugin file in `pc_assistant/plugins/` with the prefix `p_` (e.g., `pc_assistant/plugins/p_notes.py`).
- On startup, `Orchestrator._load_plugins` inspects all modules in `plugins/`.
- If `is_available()` returns `False`, the plugin is skipped and logged as disabled, preventing it from appearing in the tool schema sent to Ollama.
- If `is_available()` returns `True`, the tool schema is registered and exposed to the LLM.

---

## 3. Human-in-the-Loop (HITL) Approvals

When `requires_approval = True`:
1. The orchestrator intercepts the tool call before execution.
2. An action card is registered in `StateManager`, and an inline Telegram confirmation prompt (`Approve` / `Cancel`) is sent to the authorized user.
3. The conversation state pauses.
4. When the user taps **Approve**, the Telegram callback triggers execution via `run_tool()`.
5. The result is appended to SQLite history as a `role: tool` message, and `resume_tool_loop()` is invoked to deliver the final assistant response.
6. When the user taps **Cancel**, the action is cancelled and no changes are executed.

---

## 4. Built-in Web Plugins (Playwright)

Read-only web plugins (`get_weather`, `get_ai_news`, `search_wikipedia`) use a real Chromium browser powered by Playwright (`plugins/web_scraper.py`):

- **One-Time Setup**: Install the Chromium browser binary once via `playwright install chromium`.
- **Shared Browser Layer**: One Chromium browser starts lazily per process and stays open until shutdown; each plugin request creates an isolated browser context, aborts heavy assets (`image`, `media`, `font`), and always closes its context afterward.
- **Watch the Browser (`BROWSER_HEADLESS`)**: Defaults to `true` (background headless). Set `BROWSER_HEADLESS=false` in `.env` so you can watch the visible Chromium window while debugging.
- **Home Location (`USER_LOCATION`)**: `get_weather` uses an explicit city argument if given, or falls back to `USER_LOCATION` (and asks which city to check if neither is set).
- **Multi-Source AI News (`AI_NEWS_URLS`)**: `get_ai_news` tries each comma-separated URL in `AI_NEWS_URLS` in order until one yields at least 3 clean headlines. Because live web pages change over time, you can swap or reorder `AI_NEWS_URLS` in `.env` anytime without touching code.
- **Debug Screenshots (`data/debug/`)**: When a web lookup fails (timeout, HTTP `>= 400`, CAPTCHA redirect, empty page, or missing headlines), a screenshot is saved under `data/debug/` (the newest 10 files are kept; `data/` is git-ignored) and logged for inspection.

---

## 5. Generic `PluginResult.data` Keys

Plugins can pass structured instructions back to the orchestrator and UI via the `PluginResult.data` dictionary:

- **`direct_reply`** (`str`): If a turn's only tool call returns this, the orchestrator uses it verbatim as the reply, records it as the assistant message in history, and does **NOT** make the follow-up model call.
- **`choices`** (`list[dict]`): A list of interactive options structured as `[{"label": str, "tool": str, "kwargs": dict}]`. When present, the orchestrator registers them with `ChoiceStore` and returns a `ChoicesReply`, rendered in Telegram as an inline keyboard. Tapping an option triggers the corresponding tool (or its HITL approval flow if required).

---

## 6. File Access & Interactive Browser (`plugins/p_folder.py`, `plugins/p_files.py`)

Summer provides controlled, fast, read-only file search and interactive folder browsing:

- **Interactive Folder Browser (`list_folder`)**:
  - `requires_approval = False` (read-only browsing).
  - Paginates folder contents at 8 items per page with directories first (alphabetical) and files (newest first).
  - Navigation controls: `⬆ Up` navigates to parent directory; `➡ More` loads the next 8 items inline by editing the Telegram message.
  - Tapping a directory button executes `list_folder` inline without approval; tapping a file button routes to `share_file` which triggers HITL approval.
- **Fast Search & Background Index (`find_files`, `file_index.py`)**:
  - Background crawler indexes filenames and metadata into SQLite (`data/file_index.db`) using an FTS5 full-text virtual table with automatic sync triggers.
  - Combines instant FTS5 match results with a fast live scan (capped at 1.5s) over Tier 1 directories (Desktop, Downloads, Documents) for files created between crawl cycles.
  - Manageable via Telegram commands: `/index` shows current status (indexed count, mode, roots, last crawl); `/reindex` triggers an immediate background recrawl.
- **Delivery & Security Boundaries (`share_file`, `file_search.py`)**:
  - `requires_approval = True` (HITL gated).
  - **Sensitive-Path Denylist v2**: Hard-blocks SSH keys, cloud credentials (`.aws`, `gcloud`), cryptocurrency wallets (`wallet.dat`), browser profiles (Chrome, Edge, Firefox, Brave), OS secrets (`/etc/shadow`), and Summer's own secrets (`.env*`, `credentials/`, `data/`).
  - **Soft Warnings**: If a requested file has sensitive keywords (e.g. `salary`, `tax`, `statement`, `aadhaar`) or is outside standard user folders, preview shows a warning before approval.
  - **Audit Log & Rate Limiting**: All file shares and cancellations append to `data/file_audit.log`. A rolling rate limiter enforces a default cap of 20 shares per hour (`FILE_MAX_SHARES_PER_HOUR`).

---

## 7. Deterministic Intent Routing (`PluginBase.match_intent`)

Plugins can bypass the LLM for high-frequency or deterministic actions by implementing the optional `match_intent` hook:

```python
from plugins.base import PluginBase, IntentContext, RoutedCall

class MyPlugin(PluginBase):
    name = "my_tool"

    def match_intent(self, message: str, ctx: IntentContext) -> RoutedCall | None:
        """Inspect message and context to route deterministically without an LLM turn."""
        if message.lower().startswith("do custom "):
            return RoutedCall(tool_name=self.name, kwargs={"param": message[10:].strip()}, rule="MY_CUSTOM_RULE")
        return None
```

When `FILE_ROUTER=true` (enabled by default in `.env`), the `Orchestrator` calls registered plugins' `match_intent` hooks in registry order before invoking the LLM or memory extractor. If any hook returns a `RoutedCall`, the orchestrator executes the tool directly, logs the routing event, records the turn in history, and skips the model call. See [docs/FILES.md](FILES.md) for full details on the built-in file intent router and rules R1–R7.




