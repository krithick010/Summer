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

