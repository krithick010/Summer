"""
plugins/web_scraper.py
----------------------
Shared lazy Playwright browser layer used by read-only web plugins.

Does not start with ``p_`` so ``Orchestrator._load_plugins`` ignores it.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
import logging
from pathlib import Path
import re
from typing import Any, Awaitable, Callable

import config
from plugins.base import PluginResult

log = logging.getLogger(__name__)

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
_NAVIGATION_TIMEOUT_MS = 15_000
_CAPTCHA_MARKERS = ("google.com/sorry", "captcha")
_BLOCKED_RESOURCE_TYPES = frozenset({"image", "media", "font"})
_MAX_DEBUG_SCREENSHOTS = 10

_browser_lock: asyncio.Lock | None = None
_browser_lock_loop: asyncio.AbstractEventLoop | None = None
_playwright: Any = None
_browser: Any = None
_browser_loop: asyncio.AbstractEventLoop | None = None


def _get_lock() -> asyncio.Lock:
    """Return an asyncio.Lock bound to the current running event loop."""
    global _browser_lock, _browser_lock_loop
    loop = asyncio.get_running_loop()
    if _browser_lock is None or _browser_lock_loop is not loop:
        _browser_lock = asyncio.Lock()
        _browser_lock_loop = loop
    return _browser_lock


async def _abort_heavy_resources(route: Any) -> None:
    """Abort image, media, and font requests to keep page loads fast."""
    if getattr(route.request, "resource_type", "") in _BLOCKED_RESOURCE_TYPES:
        await route.abort()
    else:
        await route.continue_()


async def _get_browser() -> Any:
    """Lazily launch and return the shared Chromium browser instance."""
    global _playwright, _browser, _browser_loop
    loop = asyncio.get_running_loop()
    async with _get_lock():
        if _browser_loop is not loop:
            _browser = None
            _playwright = None
            _browser_loop = loop

        if _browser is not None:
            is_connected = getattr(_browser, "is_connected", None)
            if is_connected is None or is_connected():
                return _browser
            _browser = None

        from playwright.async_api import async_playwright

        if _playwright is None:
            _playwright = await async_playwright().start()

        _browser = await _playwright.chromium.launch(
            headless=config.BROWSER_HEADLESS,
            args=["--disable-gpu"],
        )
        return _browser


async def close_browser() -> None:
    """Close the shared Playwright browser and stop Playwright on shutdown."""
    global _playwright, _browser, _browser_loop
    async with _get_lock():
        if _browser is not None:
            try:
                await _browser.close()
            except Exception as exc:  # noqa: BLE001
                log.debug("Error closing shared browser: %s", exc)
            _browser = None
        if _playwright is not None:
            try:
                await _playwright.stop()
            except Exception as exc:  # noqa: BLE001
                log.debug("Error stopping Playwright: %s", exc)
            _playwright = None
        _browser_loop = None


async def _save_debug_screenshot(page: Any, label: str) -> Path | None:
    """Save a failure screenshot to DATA_DIR/debug and prune older files."""
    if page is None:
        return None
    try:
        debug_dir = Path(config.DATA_DIR) / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        safe_label = re.sub(r"[^a-zA-Z0-9_-]+", "-", label).strip("-").lower() or "page"
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        screenshot_path = debug_dir / f"{safe_label}-{timestamp}.png"
        await page.screenshot(path=str(screenshot_path))

        pngs = sorted(
            debug_dir.glob("*.png"),
            key=lambda p: (p.stat().st_mtime_ns, p.name),
            reverse=True,
        )
        for old_file in pngs[_MAX_DEBUG_SCREENSHOTS:]:
            try:
                old_file.unlink()
            except OSError:
                pass
        return screenshot_path
    except Exception as exc:  # noqa: BLE001
        log.debug("Failed to capture debug screenshot for %s: %s", label, exc)
        return None


def clean_and_truncate(raw_text: str, max_chars: int) -> str:
    """Collapse newlines and whitespace into single spaces and cap character length."""
    cleaned = " ".join(raw_text.split())
    return cleaned[:max_chars]


async def _extract_headlines(page: Any, selector: str, limit: int) -> list[str]:
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


async def run_in_page(
    url: str,
    *,
    label: str,
    handler: Callable[[Any, Any], Awaitable[PluginResult]],
    user_agent: str = DEFAULT_USER_AGENT,
    timeout_ms: int = _NAVIGATION_TIMEOUT_MS,
) -> PluginResult:
    """Open ``url`` in a fresh context on the shared browser and run ``handler``."""
    from playwright.async_api import TimeoutError as PlaywrightTimeoutError

    context = None
    page = None
    try:
        browser = await _get_browser()
        context = await browser.new_context(user_agent=user_agent)
        page = await context.new_page()
        await page.route("**/*", _abort_heavy_resources)

        response = await page.goto(
            url,
            wait_until="domcontentloaded",
            timeout=timeout_ms,
        )

        final_url = page.url or url
        status = response.status if response is not None else None
        try:
            title = await page.title()
        except Exception:  # noqa: BLE001
            title = ""

        log.info(
            "%s loaded: final_url=%s status=%s title=%r",
            label,
            final_url,
            status,
            title,
        )

        if any(marker in final_url.lower() for marker in _CAPTCHA_MARKERS):
            screenshot = await _save_debug_screenshot(page, label)
            log.info(
                "%s blocked by CAPTCHA redirect (%s) screenshot=%s",
                label,
                final_url,
                screenshot,
            )
            return PluginResult(
                success=False,
                output="⚠️ Blocked by CAPTCHA.",
                error="CAPTCHA redirect detected",
            )

        if status is not None and status >= 400:
            screenshot = await _save_debug_screenshot(page, label)
            log.info(
                "%s failed with HTTP %d (%s) screenshot=%s",
                label,
                status,
                final_url,
                screenshot,
            )
            return PluginResult(
                success=False,
                output=f"⚠️ {label} returned HTTP {status}.",
                error=f"HTTP {status} when loading {url}",
            )

        result = await handler(page, response)
        if not result.success:
            screenshot = await _save_debug_screenshot(page, label)
            log.info(
                "%s failed (%s) screenshot=%s",
                label,
                result.error or result.output,
                screenshot,
            )
        return result

    except PlaywrightTimeoutError as exc:
        screenshot = await _save_debug_screenshot(page, label)
        log.info("%s failed (timeout): %s screenshot=%s", label, exc, screenshot)
        return PluginResult(
            success=False,
            output=f"⚠️ {label} timed out. Please try again.",
            error=str(exc),
        )
    except Exception as exc:  # noqa: BLE001
        screenshot = await _save_debug_screenshot(page, label)
        msg = str(exc)
        log.info("%s failed: %s screenshot=%s", label, msg, screenshot)
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
            output=f"⚠️ {label} could not be completed right now.",
            error=msg,
        )
    finally:
        if context is not None:
            try:
                await context.close()
            except Exception as exc:  # noqa: BLE001
                log.debug("Error closing browser context: %s", exc)


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
    """Fetch ``url`` on the shared browser and return cleaned text or headlines."""

    async def _extract(page: Any, _response: Any) -> PluginResult:
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
            return PluginResult(
                success=False,
                output=f"⚠️ {label} returned an empty page.",
                error=f"Empty <body> text at {page.url}",
            )

        final_url = page.url or url
        return PluginResult(
            success=True,
            output=output_text,
            data={"url": final_url, "length": len(output_text)},
        )

    return await run_in_page(
        url,
        label=label,
        handler=_extract,
        user_agent=user_agent,
        timeout_ms=timeout_ms,
    )
