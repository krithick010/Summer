# Summer Architecture Overview

Summer is a lightweight, local-first PC assistant controlled over Telegram. It uses local LLMs and vector embeddings to converse, remember facts across sessions, and execute tools on the local machine with approval safety gates.

---

## High-Level System Architecture

```mermaid
graph TD
    User([Telegram User]) <--> Bot[python-telegram-bot / FastAPI]
    Bot --> Security{Fail-Closed Allowlist}
    Security -- Rejected --> DenyMsg[403 / Deny Message]
    Security -- Allowed --> Orch[Orchestrator]
    
    subgraph Storage & Context
        SQLite[(SQLite History: history.db)]
        Chroma[(ChromaDB: memory_db)]
    end

    Orch <--> SQLite
    Orch -- "Query (top_k, max_distance)" --> Chroma

    subgraph LLM & Inference [Ollama Local Runtime]
        ChatModel[Chat LLM: qwen2.5:3b]
        ExtractorModel[Extractor LLM: qwen2.5:3b]
        EmbedModel[Embeddings: nomic-embed-text]
    end

    Orch <--> ChatModel
    Chroma <--> EmbedModel

    subgraph Plugins & Execution
        SystemPlugin[System Info]
        GmailPlugin[Gmail API]
        StateMgr{State Manager HITL}
    end

    Orch --> SystemPlugin
    Orch --> StateMgr
    StateMgr -- "Requires Approval" --> User
    User -- "Approve Button" --> GmailPlugin

    subgraph Background Memory
        Extractor[Extractor: extract_facts]
        GroundCheck{Grounding & Placeholder Check}
    end

    Orch -. "Async fire-and-forget" .-> Extractor
    Extractor <--> ExtractorModel
    Extractor --> GroundCheck
    GroundCheck -- "Valid new fact" --> Chroma
```

---

## Component Breakdown

### 1. Gateway & Security
- **Entry points**: FastAPI webhook (`POST /webhook/{token}`) or long-polling mode.
- **Fail-closed allowlist**: Every incoming command, message, and callback query verifies `update.effective_user.id in ALLOWED_USER_IDS`. If `ALLOWED_USER_IDS` is unset or empty, all interaction is denied.

### 2. Conversation History & Persistence
- **Storage**: SQLite database (`data/history.db` or ephemeral `:memory:`).
- **Format**: Full message schema compliant with Ollama `/api/chat` (including `assistant` tool calls and `tool` responses).
- **Startup self-repair**: Drops unexecuted trailing assistant tool calls that did not survive a process restart and trims leading non-user messages.

### 3. Long-Term Memory (RAG)
- **Vector DB**: ChromaDB (`data/memory_db`) with `nomic-embed-text` embeddings.
- **Retrieval**: Multi-turn query expansion retrieves facts relevant to current and previous user messages, filtered by strict semantic distance threshold (`max_distance`).
- **Extraction**: Runs asynchronously in the background. The model extracts structured facts (`subject`, `attribute`, `value`).
- **Grounding validation**: Facts are rejected if the extracted value is not grounded directly in the user's latest turn or matches placeholder patterns (`"unknown"`, `"not provided"`).

### 4. Human-in-the-Loop (HITL) Safety
- Plugins flagged with `requires_approval = True` (e.g., sending emails) trigger an approval action in `StateManager`.
- Control returns to the Telegram interface with inline action buttons (`Approve` / `Cancel`).
- When approved, the action executes, appends the result to history, and resumes generation.

