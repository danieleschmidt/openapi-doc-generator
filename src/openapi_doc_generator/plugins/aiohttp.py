from __future__ import annotations

import re
from pathlib import Path

from ..discovery import RouteInfo, RoutePlugin, register_plugin


class AioHTTPPlugin(RoutePlugin):
    """Discover routes defined with aiohttp."""

    def detect(self, source: str) -> bool:
        return "aiohttp" in source.lower()

    def discover(self, app_path: str) -> list[RouteInfo]:
        text = Path(app_path).read_text()
        pattern = re.compile(
            r"app\.router\.add_(get|post|put|patch|delete)\(['\"]([^'\"]+)['\"]"
        )
        routes: list[RouteInfo] = []
        for method, path in pattern.findall(text):
            name = path.strip("/").replace("/", "_") or "root"
            routes.append(RouteInfo(path=path, methods=[method.upper()], name=name))
        return routes


register_plugin(AioHTTPPlugin())
