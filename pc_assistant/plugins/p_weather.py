"""
plugins/p_weather.py
--------------------
Read-only weather lookup plugin using Playwright headless Chromium and wttr.in.

Before using this plugin, install the Chromium browser binary once:
    playwright install chromium
"""

from __future__ import annotations

import urllib.parse
from typing import Any

import config
from plugins.base import PluginBase, PluginResult
from plugins.web_scraper import scrape_page_body

_WEATHER_MAX_CHARS = 600
_WEATHER_FORMAT_QUERY = (
    "format=%25l:+%25C,+%25t+(feels+%25f),+humidity+%25h,+wind+%25w"
)


class WeatherPlugin(PluginBase):
    """Fetches a compact one-line weather report from wttr.in."""

    name = "get_weather"
    description = "Gets the current weather and forecast for a specific city or location."
    parameters_schema = {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": (
                    "City to get the weather for. Leave empty to use the user's home location."
                ),
            }
        },
        "required": [],
    }
    requires_approval = False

    async def execute(self, **kwargs: Any) -> PluginResult:
        """Resolve location, fetch one-line wttr.in summary, and cap at 600 chars."""
        location = str(kwargs.get("location") or "").strip()
        if not location:
            location = config.USER_LOCATION.strip()
        if not location:
            return PluginResult(
                success=False,
                output="Which city should I check?",
                error="No location provided and USER_LOCATION is unset",
            )

        encoded_location = urllib.parse.quote(location)
        weather_url = f"https://wttr.in/{encoded_location}?{_WEATHER_FORMAT_QUERY}"
        result = await scrape_page_body(
            weather_url,
            max_chars=_WEATHER_MAX_CHARS,
            label=f"Weather lookup for '{location}'",
        )
        if result.success:
            result.data["location"] = location
        return result
