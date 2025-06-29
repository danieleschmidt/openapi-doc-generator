"""Generate migration guides between API versions."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, Iterable, Tuple


@dataclass
class MigrationGuideGenerator:
    """Compare OpenAPI specs and generate a markdown migration guide."""

    old_spec: Dict[str, Any]
    new_spec: Dict[str, Any]

    def __post_init__(self) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)

    def _endpoints(self, spec: Dict[str, Any]) -> Iterable[Tuple[str, str]]:
        for path, ops in spec.get("paths", {}).items():
            for method in ops:
                yield method.upper(), path

    def generate_markdown(self) -> str:
        """Return a markdown migration guide highlighting endpoint changes."""
        old_endpoints = set(self._endpoints(self.old_spec))
        new_endpoints = set(self._endpoints(self.new_spec))

        removed = old_endpoints - new_endpoints
        added = new_endpoints - old_endpoints

        self._logger.debug("%d endpoints removed", len(removed))
        self._logger.debug("%d endpoints added", len(added))

        lines = ["# API Migration Guide", ""]
        if removed:
            lines.append("## Removed Endpoints")
            lines.extend(f"- {m} {p}" for m, p in sorted(removed))
            lines.append("")
        if added:
            lines.append("## New Endpoints")
            lines.extend(f"- {m} {p}" for m, p in sorted(added))
            lines.append("")
        if not added and not removed:
            lines.append("No changes detected.")

        return "\n".join(lines)
