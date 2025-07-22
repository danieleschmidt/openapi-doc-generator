"""GraphQL schema utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from graphql import build_schema, get_introspection_query, graphql_sync


class GraphQLSchema:
    """Load and introspect GraphQL schemas."""

    def __init__(self, schema_path: str) -> None:
        self.schema_path = Path(schema_path)
        if not self.schema_path.exists():
            raise FileNotFoundError(schema_path)

    def introspect(self) -> Dict[str, Any]:
        """Return introspection result for the schema."""
        try:
            schema_str = self.schema_path.read_text()
        except (OSError, UnicodeDecodeError) as e:
            raise ValueError(f"Failed to read schema file: {e}")

        try:
            schema = build_schema(schema_str)
        except Exception as e:
            raise ValueError(f"Invalid GraphQL schema syntax: {e}")

        query = get_introspection_query()
        result = graphql_sync(schema, query)
        if result.errors:
            raise ValueError(result.errors[0])
        if result.data is None:
            raise ValueError("Failed to introspect schema")
        return result.data
