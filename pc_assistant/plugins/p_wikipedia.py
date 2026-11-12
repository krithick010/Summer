"""
plugins/p_wikipedia.py
----------------------
Read-only Wikipedia search plugin using Playwright headless Chromium.

Before using this plugin, install the Chromium browser binary once:
    playwright install chromium
"""

from __future__ import annotations

import logging
import urllib.parse
from typing import Any

from plugins.base import PluginBase, PluginResult

log = logging.getLogger(__name__)

_NAVIGATION_TIMEOUT_MS = 15_000
_SELECTOR_TIMEOUT_MS = 10_000
_MAX_PARAGRAPHS = 2
_MAX_SCAN_ELEMENTS = 5
_MAX_OUTPUT_CHARS = 1_500


class WikipediaPlugin(PluginBase):
    """Fetches a brief summary from Wikipedia for factual, historical, or scientific queries."""

    name = "search_wikipedia"
    description = (
        "Searches Wikipedia for a topic and returns a brief summary. "
        "Use this to answer factual, historical, or scientific questions."
    )
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
        """Navigate to Wikipedia via headless Chromium and extract the summary paragraphs."""
        query = str(kwargs.get("query", "")).strip()
        if not query:
            return PluginResult(
                success=False,
                output="⚠️ Please provide a search query for Wikipedia.",
                error="Missing required 'query' parameter.",
            )

        from playwright.async_api import (
            TimeoutError as PlaywrightTimeoutError,
            async_playwright,
        )

        search_url = (
            f"https://en.wikipedia.org/w/index.php?search={urllib.parse.quote(query)}"
        )

        browser = None
        try:
            async with async_playwright() as p:
                try:
                    browser = await p.chromium.launch(
                        headless=True,
                        args=["--disable-gpu"],
                    )
                    page = await browser.new_page()

                    response = await page.goto(
                        search_url,
                        wait_until="domcontentloaded",
                        timeout=_NAVIGATION_TIMEOUT_MS,
                    )
                    if response is not None and response.status >= 400:
                        return PluginResult(
                            success=False,
                            output=(
                                f"⚠️ Wikipedia returned HTTP {response.status} "
                                f"for '{query}'."
                            ),
                            error=f"HTTP {response.status} when loading {search_url}",
                        )

                    await page.wait_for_selector(
                        "#mw-content-text",
                        timeout=_SELECTOR_TIMEOUT_MS,
                    )

                    paragraphs = page.locator(
                        ".mw-parser-output > p:not(.mw-empty-elt)"
                    )
                    count = await paragraphs.count()

                    collected: list[str] = []
                    for i in range(min(count, _MAX_SCAN_ELEMENTS)):
                        text = (await paragraphs.nth(i).inner_text()).strip()
                        if text:
                            collected.append(text)
                        if len(collected) >= _MAX_PARAGRAPHS:
                            break

                    if not collected:
                        return PluginResult(
                            success=False,
                            output=f"No Wikipedia article summary found for '{query}'.",
                            error=f"No summary paragraphs found at {page.url}",
                        )

                    extracted_text = "\n\n".join(collected)
                    if len(extracted_text) > _MAX_OUTPUT_CHARS:
                        extracted_text = (
                            extracted_text[:_MAX_OUTPUT_CHARS].rsplit(" ", 1)[0] + "..."
                        )

                    final_url = page.url
                    log.info("Wikipedia summary fetched for %r (%s)", query, final_url)
                    return PluginResult(
                        success=True,
                        output=extracted_text,
                        data={"query": query, "url": final_url},
                    )
                finally:
                    if browser is not None:
                        await browser.close()

        except PlaywrightTimeoutError as exc:
            log.warning("Wikipedia lookup timed out for %r: %s", query, exc)
            return PluginResult(
                success=False,
                output=f"⚠️ Wikipedia lookup timed out for '{query}'. Please try again.",
                error=str(exc),
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("Wikipedia lookup failed for %r: %s", query, exc)
            msg = str(exc)
            if "Executable doesn't exist" in msg:
                return PluginResult(
                    success=False,
                    output=(
                        "⚠️ Chromium browser binary is not installed for Playwright. "
                        "Run `playwright install chromium` and try again."
                    ),
                    error=msg,
                )
            return PluginResult(
                success=False,
                output=f"⚠️ Failed to search Wikipedia for '{query}': {msg}",
                error=msg,
            )

