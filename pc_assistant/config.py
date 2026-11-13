"""
config.py
---------
Centralised configuration loaded from environment variables or a .env file.
All secrets and runtime settings are defined here — never hard-coded elsewhere.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Load .env from the project root (pc_assistant/../.env  OR  pc_assistant/.env)
# ---------------------------------------------------------------------------
_ENV_PATHS = [
    BASE_DIR / ".env",
    BASE_DIR.parent / ".env",
]
for _p in _ENV_PATHS:
    if _p.exists():
        load_dotenv(_p)
        break

# ---------------------------------------------------------------------------
# Data and Persistence Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
_h = os.getenv("HISTORY_DB_PATH", "")
HISTORY_DB_PATH = _h if _h == ":memory:" else (_h or str(DATA_DIR / "history.db"))
MEMORY_DB_PATH = os.getenv("MEMORY_DB_PATH", str(DATA_DIR / "memory_db"))

# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN: str = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN is missing or empty in .env")
if " " in TELEGRAM_BOT_TOKEN:
    raise ValueError(
        f"TELEGRAM_BOT_TOKEN contains accidental spaces: '{TELEGRAM_BOT_TOKEN}'. "
        "Telegram bot tokens must never have spaces."
    )
# Comma-separated list of numeric Telegram user IDs allowed to control the PC.
# Example:  ALLOWED_USER_IDS=123456789,987654321
_raw_ids = os.getenv("ALLOWED_USER_IDS", "")
ALLOWED_USER_IDS: list[int] = (
    [int(uid.strip()) for uid in _raw_ids.split(",") if uid.strip()]
    if _raw_ids
    else []
)

# Webhook URL (leave empty to use long-polling in development)
WEBHOOK_URL: str = os.getenv("WEBHOOK_URL", "")
# Port the FastAPI server listens on
SERVER_PORT: int = int(os.getenv("SERVER_PORT", "8000"))
# Host the FastAPI server binds to
SERVER_HOST: str = os.getenv("SERVER_HOST", "0.0.0.0")

# ---------------------------------------------------------------------------
# Ollama / LLM
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_NUM_CTX: int = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
# Shared keep-alive duration for chat and extractor so Ollama keeps the model warm
OLLAMA_KEEP_ALIVE: str = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
# Max tokens the LLM is allowed to generate per response
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "512"))
# Temperature for LLM generation
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.6"))
# Optional top-level "think" flag for qwen3-family reasoning models ("true", "false", or empty)
LLM_THINK: str = os.getenv("LLM_THINK", "").strip().lower()
# Model used by the background memory extractor (defaults to chat model)
EXTRACTOR_MODEL: str = os.getenv("EXTRACTOR_MODEL", OLLAMA_MODEL)
# Enable/disable background memory extraction
EXTRACTOR_ENABLED: bool = os.getenv("EXTRACTOR_ENABLED", "true").strip().lower() in ("true", "1", "yes")

# User and assistant identity
USER_NAME: str = os.getenv("USER_NAME", "friend")
ASSISTANT_NAME: str = os.getenv("ASSISTANT_NAME", "Summer")
# Optional home location used as context for local queries (e.g. "City, Region, Country")
USER_LOCATION: str = os.getenv("USER_LOCATION", "").strip()

# ---------------------------------------------------------------------------
# Web Plugins (Playwright)
# ---------------------------------------------------------------------------
# Run headless Chromium by default; set to false to watch the browser window
BROWSER_HEADLESS: bool = os.getenv("BROWSER_HEADLESS", "true").strip().lower() in ("true", "1", "yes")
# Comma-separated AI news URLs tried in order until one yields enough headlines
_DEFAULT_AI_NEWS_URLS = (
    "https://techcrunch.com/category/artificial-intelligence/,"
    "https://venturebeat.com/category/ai/"
)
AI_NEWS_URLS: list[str] = [
    u.strip()
    for u in os.getenv("AI_NEWS_URLS", _DEFAULT_AI_NEWS_URLS).split(",")
    if u.strip()
]

# ---------------------------------------------------------------------------
# Conversation History & Context
# ---------------------------------------------------------------------------
HISTORY_TURNS: int = int(os.getenv("HISTORY_TURNS", "10"))

# ---------------------------------------------------------------------------
# Google / Gmail
# ---------------------------------------------------------------------------
# Path to the OAuth2 client-secret JSON downloaded from Google Cloud Console
GMAIL_CREDENTIALS_PATH: str = os.getenv(
    "GMAIL_CREDENTIALS_PATH",
    str(Path(__file__).parent / "credentials" / "gmail_oauth.json"),
)
# Token cache created automatically after first OAuth consent
GMAIL_TOKEN_PATH: str = os.getenv(
    "GMAIL_TOKEN_PATH",
    str(Path(__file__).parent / "credentials" / "gmail_token.json"),
)
# Sender address used when drafting / sending mail
GMAIL_SENDER_EMAIL: str = os.getenv("GMAIL_SENDER_EMAIL", "")

# ---------------------------------------------------------------------------
# Human-in-the-Loop (HITL)
# ---------------------------------------------------------------------------
# Seconds to keep a pending approval before auto-cancelling
HITL_TIMEOUT_SECONDS: int = int(os.getenv("HITL_TIMEOUT_SECONDS", "120"))

# ---------------------------------------------------------------------------
# File Sharing & Search
# ---------------------------------------------------------------------------
# Optional comma-separated root paths to search (empty = priority user folders then home)
FILE_SEARCH_ROOTS: str = os.getenv("FILE_SEARCH_ROOTS", "")
# Maximum file size (MB) sent directly over Telegram without compression or email routing
FILE_TELEGRAM_MAX_MB: float = float(os.getenv("FILE_TELEGRAM_MAX_MB", "50.0"))
# Maximum post-zip file size (MB) eligible for multi-part 45 MB chunks over Telegram
FILE_MAX_SPLIT_MB: float = float(os.getenv("FILE_MAX_SPLIT_MB", "150.0"))
# File upload timeout in seconds for Telegram send_document and send_photo
FILE_UPLOAD_TIMEOUT_S: float = float(os.getenv("FILE_UPLOAD_TIMEOUT_S", "600.0"))
# Default email address to send files to when delivery is email
USER_DEFAULT_EMAIL: str = os.getenv("USER_DEFAULT_EMAIL", "")

# ---------------------------------------------------------------------------
# Vector Memory (ChromaDB + nomic-embed-text)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
# Run embeddings on CPU (num_gpu=0) by default so chat LLM keeps GPU VRAM
EMBED_ON_CPU: bool = os.getenv("EMBED_ON_CPU", "true").strip().lower() in ("true", "1", "yes")
MEMORY_TOP_K: int = int(os.getenv("MEMORY_TOP_K", "5"))
MEMORY_MAX_DISTANCE: float = float(os.getenv("MEMORY_MAX_DISTANCE", "1.3"))
# Backward-compatibility alias
MEMORY_DISTANCE_THRESHOLD: float = float(os.getenv("MEMORY_DISTANCE_THRESHOLD", str(MEMORY_MAX_DISTANCE)))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

