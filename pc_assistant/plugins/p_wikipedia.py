"""
plugins/p_wikipedia.py
----------------------
Read-only Wikipedia search plugin using the shared Playwright browser layer.

Before using this plugin, install the Chromium browser binary once:
    playwright install chromium
"""

from __future__ import annotations

import logging
import re
import urllib.parse
from typing import Any

from plugins.base import PluginBase, PluginResult
from plugins.web_scraper import run_in_page

log = logging.getLogger(__name__)

_NAVIGATION_TIMEOUT_MS = 15_000
_SELECTOR_TIMEOUT_MS = 10_000
_MAX_SCAN_ELEMENTS = 12
_MAX_SENTENCES = 3
_MAX_OUTPUT_CHARS = 800
_CITATION_RE = re.compile(r"\s*\[(?:\d+|[a-zA-Z]|citation needed|note\s+\d+|clarification needed)\]")


def _strip_citations(text: str) -> str:
    """Remove Wikipedia footnote markers like [1] or [a] and collapse whitespace."""
    cleaned = _CITATION_RE.sub("", text or "")
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    return " ".join(cleaned.split())


def _first_sentences(paragraphs: list[str], max_sentences: int = _MAX_SENTENCES, max_chars: int = _MAX_OUTPUT_CHARS) -> str:
    """Extract the first 2-3 clean sentences from lead paragraphs, capped at ``max_chars``."""
    sentences: list[str] = []
    for para in paragraphs:
        parts = re.split(r"(?<=[.!?])\s+", para.strip())
        for part in parts:
            s = part.strip()
            if s:
                sentences.append(s)
            if len(sentences) >= max_sentences:
                break
        if len(sentences) >= max_sentences:
            break

    joined = " ".join(sentences).strip()
    if len(joined) <= max_chars:
        return joined
    clipped = joined[: max_chars - 3].rsplit(" ", 1)[0].rstrip(" ,;:-")
    return f"{clipped}..." if clipped else joined[:max_chars]


async def _find_first_target_href(page: Any, selector: str, *, article_only: bool = False) -> str | None:
    """Return the first matching href from ``selector`` on a Wikipedia page."""
    loc = page.locator(selector)
    count = await loc.count()
    for i in range(min(count, 20)):
        href = (await loc.nth(i).get_attribute("href") or "").strip()
        if not href or href.startswith("#"):
            continue
        if article_only:
            path = urllib.parse.urlparse(href).path
            if not path.startswith("/wiki/") or ":" in path:
                continue
        return href
    return None


async def _extract_article_summary(page: Any, query: str) -> PluginResult:
    """Handle search results / disambiguation redirects and extract 2-3 lead sentences."""
    await page.wait_for_selector("#mw-content-text", timeout=_SELECTOR_TIMEOUT_MS)

    # 1. If Wikipedia landed on a search results page, follow the first result.
    search_href = await _find_first_target_href(page, ".mw-search-result-heading a")
    if search_href:
        next_url = urllib.parse.urljoin(page.url, search_href)
        await page.goto(next_url, wait_until="domcontentloaded", timeout=_NAVIGATION_TIMEOUT_MS)
        await page.wait_for_selector("#mw-content-text", timeout=_SELECTOR_TIMEOUT_MS)

    # 2. Check if this is a disambiguation page and follow the first listed target.
    is_disambig = await page.locator("#disambigbox, .dmbox").count() > 0
    if not is_disambig:
        first_p_loc = page.locator("#mw-content-text .mw-parser-output p")
        if await first_p_loc.count() > 0:
            first_text = _strip_citations(await first_p_loc.nth(0).inner_text()).lower()
            if "may refer to:" in first_text:
                is_disambig = True

    if is_disambig:
        disambig_href = await _find_first_target_href(
            page,
            "#mw-content-text .mw-parser-output ul li a",
            article_only=True,
        )
        if disambig_href:
            next_url = urllib.parse.urljoin(page.url, disambig_href)
            await page.goto(next_url, wait_until="domcontentloaded", timeout=_NAVIGATION_TIMEOUT_MS)
            await page.wait_for_selector("#mw-content-text", timeout=_SELECTOR_TIMEOUT_MS)

    # 3. Read lead paragraphs from #mw-content-text .mw-parser-output p
    paragraphs = page.locator("#mw-content-text .mw-parser-output p")
    count = await paragraphs.count()

    collected: list[str] = []
    for i in range(min(count, _MAX_SCAN_ELEMENTS)):
        raw = await paragraphs.nth(i).inner_text()
        cleaned = _strip_citations(raw)
        if len(cleaned) < 20:
            continue
        collected.append(cleaned)
        if len(collected) >= 3:
            break

    final_url = page.url
    suffix = f"\n{final_url}"
    summary_budget = max(100, _MAX_OUTPUT_CHARS - len(suffix))
    summary = _first_sentences(collected, max_chars=summary_budget)
    if not summary:
        return PluginResult(
            success=False,
            output=f"No Wikipedia article summary found for '{query}'.",
            error=f"No summary paragraphs found at {page.url}",
        )

    output = f"{summary}{suffix}"
    log.info("Wikipedia summary fetched for %r (%s)", query, final_url)
    return PluginResult(
        success=True,
        output=output,
        data={"query": query, "url": final_url},
    )


class WikipediaPlugin(PluginBase):
    """Fetches a brief summary from Wikipedia for factual, historical, or scientific queries."""

    name = "search_wikipedia"
    description = (
        "Looks up a topic, person, or concept on Wikipedia and returns a short summary."
    )
    usage_hint = "when they ask you to look something up, or mention Wikipedia"
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Topic or search term to look up on Wikipedia",
            }
        },
        "required": ["query"],
    }
    requires_approval = False

    async def execute(self, **kwargs: Any) -> PluginResult:
        """Navigate to Wikipedia via the shared Playwright browser and extract 2-3 lead sentences."""
        query = str(kwargs.get("query", "")).strip()
        if not query:
            return PluginResult(
                success=False,
                output="Please tell me what to look up on Wikipedia.",
                error="Missing required 'query' parameter.",
            )

        search_url = (
            f"https://en.wikipedia.org/w/index.php?search={urllib.parse.quote(query)}&go=Go"
        )

        async def _handler(page: Any, _response: Any) -> PluginResult:
            return await _extract_article_summary(page, query)

        result = await run_in_page(
            search_url,
            label=f"wikipedia-{query}",
            handler=_handler,
        )
        if not result.success and result.error:
            err_lower = result.error.lower()
            if "executable doesn't exist" in err_lower:
                return PluginResult(
                    success=False,
                    output="Wikipedia search needs `playwright install chromium` first.",
                    error=result.error,
                )
            if "timed out" in err_lower or "timeout" in err_lower:
                return PluginResult(
                    success=False,
                    output=f"Wikipedia lookup timed out for '{query}'. Please try again.",
                    error=result.error,
                )
        return result
