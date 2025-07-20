"""Generate HTML for an interactive API playground."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
import html
import json

from jinja2 import Template


DEFAULT_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <title>{{ title }}</title>
  <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist/swagger-ui.css"/>
</head>
<body>
  <div id="swagger-ui"></div>
  <script src="https://unpkg.com/swagger-ui-dist/swagger-ui-bundle.js"></script>
  <script>
  const spec = {{ spec | safe }};
  SwaggerUIBundle({ spec: spec, dom_id: '#swagger-ui' });
  </script>
</body>
</html>
"""


@dataclass
class PlaygroundGenerator:
    """Convert an OpenAPI specification into Swagger UI HTML."""

    template: str = DEFAULT_TEMPLATE

    def generate(self, spec: Dict[str, Any]) -> str:
        """Return the HTML for a Swagger UI playground."""
        if not isinstance(spec, dict):
            raise TypeError("spec must be a dict")
        template = Template(self.template)
        spec_json = json.dumps(spec)
        return template.render(
            title=html.escape(spec.get("info", {}).get("title", "API")),
            spec=spec_json,
        )
