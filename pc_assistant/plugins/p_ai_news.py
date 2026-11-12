"""
plugins/p_ai_news.py
--------------------
Read-only AI news plugin using Playwright headless Chromium via the shared browser layer.

Tries each URL in ``config.AI_NEWS_URLS`` in order until one yields at least
three clean headlines.

Before using this plugin, install the Chromium browser binary once:
    playwright install chromium
"""

from __future__ import annotations

import re
from typing import Any
import urllib.parse

import config
from plugins.base import PluginBase, PluginResult
from plugins.web_scraper import run_in_page

_HEADLINE_SELECTOR = "h2 a, h3 a"
_MIN_HEADLINE_CHARS = 25
_MIN_HEADLINES_REQUIRED = 3
_MAX_HEADLINES = 5
_MAX_TITLE_CHARS = 200
_MAX_TOTAL_CHARS = 1_500
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]+")


def _clean_untrusted_text(raw: str) -> str:
    """Strip control characters and collapse whitespace into single spaces."""
    without_ctrl = _CONTROL_CHARS_RE.sub(" ", raw or "")
    return " ".join(without_ctrl.split())


def _source_label(url: str) -> str:
    """Return a clean hostname label (without leading www.) for a source URL."""
    host = urllib.parse.urlparse(url).netloc.lower()
    return host.removeprefix("www.") or url


async def _extract_clean_headlines(page: Any, base_url: str) -> list[tuple[str, str]]:
    """Extract deduped (title, link) pairs matching the Phase 1 strategy."""
    locator = page.locator(_HEADLINE_SELECTOR)
    count = await locator.count()
    seen: set[str] = set()
    items: list[tuple[str, str]] = []

    for i in range(count):
        el = locator.nth(i)
        raw_href = await el.get_attribute("href")
        href = _CONTROL_CHARS_RE.sub("", (raw_href or "")).strip()
        if not href or href == "#":
            continue

        title = _clean_untrusted_text(await el.inner_text())
        if len(title) < _MIN_HEADLINE_CHARS:
            continue
        if title in seen:
            continue
        seen.add(title)

        if len(title) > _MAX_TITLE_CHARS:
            title = title[:_MAX_TITLE_CHARS].rstrip()

        full_link = urllib.parse.urljoin(base_url, href)
        items.append((title, full_link))
        if len(items) >= _MAX_HEADLINES:
            break

    return items


class AINewsPlugin(PluginBase):
    """Scrapes configured AI news sources in order for top AI headlines."""

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
        """Try each URL in config.AI_NEWS_URLS until one yields >= 3 clean headlines."""
        urls = [u.strip() for u in config.AI_NEWS_URLS if u.strip()]
        if not urls:
            return PluginResult(
                success=False,
                output="⚠️ No AI news sources are configured.",
                error="AI_NEWS_URLS is empty",
            )

        for idx, source_url in enumerate(urls, 1):
            label = f"AI news source {idx} ({_source_label(source_url)})"

            async def _handler(page: Any, _response: Any, url: str = source_url) -> PluginResult:
                final_url = page.url or url
                headlines = await _extract_clean_headlines(page, final_url)
                if len(headlines) < _MIN_HEADLINES_REQUIRED:
                    return PluginResult(
                        success=False,
                        output=f"⚠️ Not enough headlines found at {_source_label(final_url)}.",
                        error=f"Only {len(headlines)} clean headlines found at {final_url}",
                    )

                lines = [
                    f"{i}. {title} - {link}"
                    for i, (title, link) in enumerate(headlines, 1)
                ]
                lines.append(f"Source: {_source_label(final_url)}")
                output = "\n".join(lines)[:_MAX_TOTAL_CHARS]
                return PluginResult(
                    success=True,
                    output=output,
                    data={
                        "url": final_url,
                        "source": _source_label(final_url),
                        "count": len(headlines),
                    },
                )

            result = await run_in_page(source_url, label=label, handler=_handler)
            if result.success:
                return result

        return PluginResult(
            success=False,
            output="⚠️ Couldn't fetch AI news right now. Please try again later.",
            error="All configured AI news sources failed",
        )
