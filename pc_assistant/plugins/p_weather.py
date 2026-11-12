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

from plugins.base import PluginBase, PluginResult
from plugins.web_scraper import scrape_page_body

_DEFAULT_LOCATION = "Madurai"
_WEATHER_MAX_CHARS = 1_500


class WeatherPlugin(PluginBase):
    """Fetches clean plain-text current weather and forecast from wttr.in."""

    name = "get_weather"
    description = "Gets the current weather and forecast for a specific city or location."
    parameters_schema = {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": (
                    "City or location name to get the weather for. "
                    "If the user does not specify a location, check their saved "
                    "memories or default to 'Madurai'."
                ),
            }
        },
        "required": [],
    }
    requires_approval = False

    async def execute(self, **kwargs: Any) -> PluginResult:
        """Navigate to wttr.in and return cleaned <body> weather text."""
        location = str(kwargs.get("location") or "").strip() or _DEFAULT_LOCATION

        search_url = f"https://wttr.in/{urllib.parse.quote(location)}?0T"
        result = await scrape_page_body(
            search_url,
            max_chars=_WEATHER_MAX_CHARS,
            label=f"Weather lookup for '{location}'",
        )
        if result.success:
            result.data["location"] = location
        return result
