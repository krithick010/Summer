"""
plugins/p_system.py
-------------------
Read-only OS telemetry plugin using psutil.

Returns CPU load, RAM usage, and disk space.
No approval required — this is a purely observational action.
"""

from __future__ import annotations

import asyncio
import platform

import psutil

from plugins.base import PluginBase, PluginResult


class SystemInfoPlugin(PluginBase):
    # NOTE: name must match the tool_name used in main.py's /status handler
    name = "system_info"
    description = (
        "Returns current PC health metrics: CPU load percentage, available RAM "
        "in GB, total RAM in GB, and free disk space on the primary drive. "
        "Use this when the user asks about system status, performance, or resources."
    )
    parameters_schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }
    requires_approval = False

    async def execute(self, **kwargs) -> PluginResult:  # noqa: ANN003
        # psutil.cpu_percent(interval=…) has a blocking sleep — offload it
        # to a thread so the async event loop stays responsive.
        cpu, ram, disk = await asyncio.to_thread(self._gather_metrics)

        ram_used_gb  = round((ram.total - ram.available) / (1024 ** 3), 2)
        ram_total_gb = round(ram.total / (1024 ** 3), 2)
        ram_avail_gb = round(ram.available / (1024 ** 3), 2)
        ram_pct      = ram.percent

        disk_free_gb  = round(disk.free  / (1024 ** 3), 2)
        disk_total_gb = round(disk.total / (1024 ** 3), 2)
        disk_pct      = disk.percent

        output = (
            f"💻 *System Status*\n\n"
            f"🔲 *CPU:* `{cpu}%` load\n"
            f"🧠 *RAM:* `{ram_used_gb} / {ram_total_gb} GB` used "
            f"(`{ram_pct}%`) — `{ram_avail_gb} GB` free\n"
            f"💾 *Disk:* `{disk_free_gb} GB` free of `{disk_total_gb} GB` "
            f"(`{disk_pct}%` used)"
        )

        return PluginResult(
            success=True,
            output=output,
            data={
                "cpu_percent": cpu,
                "ram_used_gb": ram_used_gb,
                "ram_total_gb": ram_total_gb,
                "ram_percent": ram_pct,
                "disk_free_gb": disk_free_gb,
                "disk_total_gb": disk_total_gb,
                "disk_percent": disk_pct,
            },
        )

    # ------------------------------------------------------------------
    # Private helpers (run inside a thread via asyncio.to_thread)
    # ------------------------------------------------------------------

    @staticmethod
    def _gather_metrics():
        cpu = psutil.cpu_percent(interval=0.5)   # blocking sleep is safe in thread
        ram = psutil.virtual_memory()

        # Disk root differs by OS: Windows uses a drive letter, Unix uses '/'
        disk_root = "C:\\" if platform.system() == "Windows" else "/"
        disk = psutil.disk_usage(disk_root)

        return cpu, ram, disk
