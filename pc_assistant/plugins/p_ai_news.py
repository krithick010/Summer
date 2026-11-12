"""
plugins/p_ai_news.py
--------------------
Read-only AI news plugin using Playwright headless Chromium and TechCrunch's AI feed.

Before using this plugin, install the Chromium browser binary once:
    playwright install chromium
"""

from __future__ import annotations

from typing import Any

from plugins.base import PluginBase, PluginResult
from plugins.web_scraper import scrape_page_body

_AI_NEWS_URL = "https://techcrunch.com/category/artificial-intelligence/"
_AI_NEWS_MAX_CHARS = 2_500
_HEADLINE_SELECTOR = "h2, h3"
_MAX_HEADLINES = 5


class AINewsPlugin(PluginBase):
    """Scrapes TechCrunch's Artificial Intelligence section for the latest AI headlines."""

    name = "get_ai_news"
    description = (
        "Scrapes the web for the top latest news headlines regarding "
        "Artificial Intelligence."
    )
    # PluginBase requires a non-empty parameters_schema dict; empty-object JSON Schema
    # represents zero parameters cleanly for Ollama tool calling.
    parameters_schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }
    requires_approval = False

    async def execute(self, **kwargs: Any) -> PluginResult:  # noqa: ARG002
        """Navigate to TechCrunch AI and return the top 5 headlines (or body fallback)."""
        return await scrape_page_body(
            _AI_NEWS_URL,
            max_chars=_AI_NEWS_MAX_CHARS,
            label="AI news lookup",
            headline_selector=_HEADLINE_SELECTOR,
            max_headlines=_MAX_HEADLINES,
        )
