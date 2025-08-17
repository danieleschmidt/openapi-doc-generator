"""Generate migration guides between API versions."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass
class MigrationGuideGenerator:
    """Compare OpenAPI specs and generate a markdown migration guide."""

    old_spec: dict[str, Any]
    new_spec: dict[str, Any]

    def __post_init__(self) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)

    def _endpoints(self, spec: dict[str, Any]) -> Iterable[tuple[str, str]]:
        for path, ops in spec.get("paths", {}).items():
            for method in ops:
                yield method.upper(), path

    def _calculate_endpoint_changes(self) -> tuple[set, set]:
        """Calculate added and removed endpoints."""
        old_endpoints = set(self._endpoints(self.old_spec))
        new_endpoints = set(self._endpoints(self.new_spec))

        removed = old_endpoints - new_endpoints
        added = new_endpoints - old_endpoints

        self._logger.debug("%d endpoints removed", len(removed))
        self._logger.debug("%d endpoints added", len(added))

        return removed, added

    def _format_endpoint_section(self, title: str, endpoints: set) -> list[str]:
        """Format a section for added or removed endpoints."""
        if not endpoints:
            return []

        lines = [f"## {title}"]
        lines.extend(f"- {m} {p}" for m, p in sorted(endpoints))
        lines.append("")
        return lines

    def generate_markdown(self) -> str:
        """Return a markdown migration guide highlighting endpoint changes."""
        removed, added = self._calculate_endpoint_changes()

        lines = ["# API Migration Guide", ""]
        lines.extend(self._format_endpoint_section("Removed Endpoints", removed))
        lines.extend(self._format_endpoint_section("New Endpoints", added))

        if not added and not removed:
            lines.append("No changes detected.")

        return "\n".join(lines)
