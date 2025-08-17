"""Test suite generation utilities."""

from __future__ import annotations

from dataclasses import dataclass

from .config import config
from .documentator import DocumentationResult


@dataclass
class TestSuiteGenerator:
    """Generate basic pytest suites for discovered routes."""

    # Prevent pytest from treating this as a test case
    __test__ = False

    result: DocumentationResult

    def generate_pytest(self) -> str:
        """Return pytest tests for the analyzed application."""
        lines: list[str] = ["import requests", ""]
        for route in self.result.routes:
            path = route.path or "/"
            for method in route.methods:
                test_name = f"test_{route.name}_{method.lower()}"
                lines.append(f"def {test_name}():")
                lines.append(
                    f"    resp = requests.{method.lower()}('{config.TEST_BASE_URL}{path}')"
                )
                lines.append(
                    f"    assert resp.status_code == "
                    f"{config.DEFAULT_SUCCESS_STATUS_INT}"
                )
                lines.append("")
        return "\n".join(lines)
