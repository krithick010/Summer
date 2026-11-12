"""
plugins/web_scraper.py
----------------------
Shared async Playwright scraper used by read-only web plugins.

Does not start with ``p_`` so ``Orchestrator._load_plugins`` ignores it.
"""

from __future__ import annotations

import logging

from plugins.base import PluginResult

log = logging.getLogger(__name__)

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
_NAVIGATION_TIMEOUT_MS = 15_000
_CAPTCHA_MARKERS = ("google.com/sorry", "captcha")


def clean_and_truncate(raw_text: str, max_chars: int) -> str:
    """Collapse newlines and whitespace into single spaces and cap character length."""
    cleaned = " ".join(raw_text.split())
    return cleaned[:max_chars]


async def _extract_headlines(page, selector: str, limit: int) -> list[str]:
    """Return up to ``limit`` cleaned non-empty texts matching ``selector``."""
    locator = page.locator(selector)
    count = await locator.count()
    items: list[str] = []
    for i in range(count):
        if len(items) >= limit:
            break
        text = " ".join((await locator.nth(i).inner_text()).split())
        if text:
            items.append(text)
    return items


async def scrape_page_body(
    url: str,
    *,
    max_chars: int,
    label: str,
    user_agent: str = DEFAULT_USER_AGENT,
    timeout_ms: int = _NAVIGATION_TIMEOUT_MS,
    headline_selector: str | None = None,
    max_headlines: int = 5,
) -> PluginResult:
    """Fetch ``url`` with headless Chromium and return cleaned text or headlines."""
    from playwright.async_api import (
        TimeoutError as PlaywrightTimeoutError,
        async_playwright,
    )

    browser = None
    context = None
    try:
        async with async_playwright() as p:
            try:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(user_agent=user_agent)
                page = await context.new_page()

                response = await page.goto(
                    url,
                    wait_until="domcontentloaded",
                    timeout=timeout_ms,
                )

                current_url = (page.url or "").lower()
                if any(marker in current_url for marker in _CAPTCHA_MARKERS):
                    log.info("%s blocked by CAPTCHA redirect (%s)", label, page.url)
                    return PluginResult(
                        success=False,
                        output="⚠️ Blocked by CAPTCHA.",
                        error="CAPTCHA redirect detected",
                    )

                if response is not None and response.status >= 400:
                    log.info("%s failed with HTTP %d (%s)", label, response.status, page.url)
                    return PluginResult(
                        success=False,
                        output=f"⚠️ {label} returned HTTP {response.status}.",
                        error=f"HTTP {response.status} when loading {url}",
                    )

                output_text = ""
                if headline_selector:
                    headlines = await _extract_headlines(page, headline_selector, max_headlines)
                    if headlines:
                        formatted = "\n".join(
                            f"{idx}. {title}" for idx, title in enumerate(headlines, 1)
                        )
                        output_text = formatted[:max_chars]

                if not output_text:
                    raw_text = await page.locator("body").inner_text(timeout=timeout_ms)
                    output_text = clean_and_truncate(raw_text, max_chars)

                if not output_text:
                    log.info("%s failed: empty page returned (%s)", label, page.url)
                    return PluginResult(
                        success=False,
                        output=f"⚠️ {label} returned an empty page.",
                        error=f"Empty <body> text at {page.url}",
                    )

                final_url = page.url
                log.info("%s succeeded (%d chars, %s)", label, len(output_text), final_url)
                return PluginResult(
                    success=True,
                    output=output_text,
                    data={"url": final_url, "length": len(output_text)},
                )
            finally:
                if context is not None:
                    await context.close()
                if browser is not None:
                    await browser.close()

    except PlaywrightTimeoutError as exc:
        log.info("%s failed (timeout): %s", label, exc)
        return PluginResult(
            success=False,
            output=f"⚠️ {label} timed out. Please try again.",
            error=str(exc),
        )
    except Exception as exc:  # noqa: BLE001
        log.info("%s failed: %s", label, exc)
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
            output=f"⚠️ {label} failed: {msg}",
            error=msg,
        )
