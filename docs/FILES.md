# File Management, Deterministic Intent Routing, and Search Architecture

This document describes Summer's deterministic file routing layer, search modes, indexing, preflight checks, delivery options, and extensibility.

---

## 1. Deterministic File Intent Router (`FILE_ROUTER`)

To eliminate tool-selection hallucinations and latency from small local models (e.g., `qwen2.5:3b`), Summer provides a lightweight, deterministic intent-routing layer.

When `FILE_ROUTER=true` (the default in `.env`), incoming user messages are evaluated by the router **before** invoking the LLM or memory extractor. If a message matches any of the rules R1–R7, the corresponding plugin tool is invoked directly, skipping the model turn completely while recording user requests, tool executions, and replies in conversation history.

To disable deterministic routing and route all requests through the LLM tool loop, set:
```bash
FILE_ROUTER=false
```

---

## 2. Intent Routing Rules (R1–R7)

The router identifies file commands and parameters using fuzzy matching on the first 3 tokens (`send`, `sent`, `share`, `give`, `get`, `fetch`, `mail`, `email`, `find`, `search`, `locate`, `show`, `look`, `need`, `want`, `where`, `open`, `browse`, `list`) and evaluates rules in order:

| Rule | Description | Examples | Target Tool & Action |
|------|-------------|----------|----------------------|
| **R1** | Existing path or alias | `C:\Users\jdoe\report.pdf`, `Downloads`, `/tmp/data` | `share_file(path=...)` |
| **R2** | Token with known file extension | `Tree.txt`, `tree.txt ig`, `tree.rxt`, `Find me gmail_oauth.json` | `find_files` if find verb; else `share_file` |
| **R3** | Command verb + remaining tokens | `Sent me report final file`, `Shre me the wallpapers folder` | Lookup via `find_files` or `share_file` |
| **R4** | Bare query (1–6 tokens, no verb) | `report_final`, `Aicte final`, `WhatsApp image` | `share_file` if Tier A index/vocab match |
| **R5** | Pronoun follow-ups (< 30 min) | `mail it`, `send it`, `share that again`, `get it` | `share_file(path=last_file_path)` |
| **R6** | Location follow-up (< 10 min) | `It's in desktop`, `actually Documents`, `inside dsa` | Rerun `last_query` in new location terms |
| **R7** | Folder listing / browsing | `list downloads`, `What is in my Downloads`, `browse /data` | `list_folder(path=...)` |

### Non-File Messages (Fall Through to LLM)
The router deliberately skips:
- Greetings (`hi`, `yo`, `hello`, `hey`)
- Weather requests (`how's the weather?`, `weather in Paris`)
- News queries (`any ai news?`)
- Wikipedia queries (`who is Ada Lovelace on wikipedia`)
- Emails to third parties (`email my mom saying hi`, `Mail it to boss@example.com`)
- Factual and conversational questions

### Delivery Routing
- If the request contains `mail` or `email` without third-party recipients, delivery is routed to `email`.
- If an external recipient is specified (e.g. `to mom` or `user@example.com`), routing is skipped so the LLM and `send_email` handle it.

---

## 3. Dedicated Telegram Commands

Summer provides allowlisted Telegram commands that run directly without calling the LLM:

- `/find <keywords>`: Searches files and folders matching keywords. Shows interactive choice buttons if multiple candidates match.
- `/browse [path]`: Lists files and subfolders for interactive directory navigation. Passing no path lists drive roots.
- `/index`: Shows background crawl status, indexed file count, database size, and elapsed time since last crawl.
- `/reindex`: Triggers an immediate background filesystem re-crawl.
- `/cancel`: Cancels any active pending human-in-the-loop (HITL) approval.

---

## 4. Preflight Checks & Delivery Approval Cards

Before presenting an approval card to the user:
1. **Target Verification**: Resolves user queries, aliases (`Desktop`, `Downloads`, etc.), and fuzzy matches. If missing, reports failure with no card.
2. **Security & Sensitivity**: Denies access to system folders (`Windows`, `Program Files`, `.ssh`, `.gnupg`, `.git`, `.venv`) and credentials (`.env`, `id_rsa`, `*token*.json`).
3. **Preflight Size Check**:
   - If total size > `FILE_MAX_SPLIT_MB` (default `150.0` MB), immediately rejects:
     `'<filename>' is 459.9 MB. Telegram bots can upload 50 MB (150 MB in parts); Gmail allows 25 MB.`
   - No approval card is generated when limits are exceeded.

### Share Approval Buttons
For valid files, Summer presents an inline keyboard:
- `[Send here]` (`hitl:approve`): Delivers file directly in Telegram.
- `[Email me]` (`hitl:email`): Displayed **only** when Gmail is configured and size $\le$ 25 MB. Forces delivery to `USER_DEFAULT_EMAIL` or `GMAIL_SENDER_EMAIL`.
- `[Cancel]` (`hitl:cancel`): Cancels the operation.

---

## 5. Indexing, Search Modes & Privacy

Configure search behavior via `.env`:
```bash
# Search mode: home (default), all_drives, or custom
FILE_SEARCH_MODE=home

# Custom roots (used when FILE_SEARCH_MODE=custom)
FILE_SEARCH_ROOTS=

# Exclude extra paths
FILE_SEARCH_EXCLUDE=node_modules,.git,.venv

# Crawl refresh schedule in hours
FILE_INDEX_REFRESH_HOURS=6

# Removable drive indexing (disabled by default)
FILE_INDEX_REMOVABLE=false
```

### Privacy & Auditing
- **Read-Only**: Summer cannot modify, delete, or execute files.
- **Audit Logging**: When `FILE_AUDIT_LOG=true`, every file search, choice pick, and delivery action is logged to `data/file_audit.log`.

---

## 6. Extending with `PluginBase.match_intent`

Plugins can register their own deterministic matching logic by implementing `match_intent`:

```python
from plugins.base import PluginBase, IntentContext, RoutedCall

class MyCustomPlugin(PluginBase):
    name = "my_custom_tool"

    def match_intent(self, message: str, ctx: IntentContext) -> RoutedCall | None:
        if message.lower().startswith("do custom "):
            arg = message[10:].strip()
            return RoutedCall(tool_name=self.name, kwargs={"param": arg}, rule="CUSTOM_1")
        return None
```

