from __future__ import annotations

from pathlib import Path
import re
from typing import List

from ..discovery import RouteInfo, RoutePlugin, register_plugin


class AioHTTPPlugin(RoutePlugin):
    """Discover routes defined with aiohttp."""

    def detect(self, source: str) -> bool:
        return "aiohttp" in source.lower()

    def discover(self, app_path: str) -> List[RouteInfo]:
        text = Path(app_path).read_text()
        pattern = re.compile(
            r"app\.router\.add_(get|post|put|patch|delete)\(['\"]([^'\"]+)['\"]"
        )
        routes: List[RouteInfo] = []
        for method, path in pattern.findall(text):
            name = path.strip("/").replace("/", "_") or "root"
            routes.append(RouteInfo(path=path, methods=[method.upper()], name=name))
        return routes


register_plugin(AioHTTPPlugin())
