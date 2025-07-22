"""OpenAPI specification validation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class SpecValidator:
    """Validate OpenAPI specifications and suggest improvements."""

    # Valid HTTP methods according to OpenAPI 3.0 spec
    VALID_HTTP_METHODS = {
        "get",
        "put",
        "post",
        "delete",
        "options",
        "head",
        "patch",
        "trace",
    }

    def validate(self, spec: Dict[str, Any]) -> List[str]:
        """Return a list of suggestions for improving the spec."""
        if not isinstance(spec, dict):
            raise TypeError("spec must be a dict")

        suggestions: List[str] = []

        # Validate OpenAPI version
        suggestions.extend(self._validate_openapi_version(spec))

        # Validate required top-level fields
        suggestions.extend(self._validate_required_fields(spec))

        # Validate info section
        suggestions.extend(self._validate_info_section(spec))

        # Validate paths section
        suggestions.extend(self._validate_paths_section(spec))

        # Validate components section
        suggestions.extend(self._validate_components_section(spec))

        # Validate security
        suggestions.extend(self._validate_security(spec))

        return suggestions

    def _validate_openapi_version(self, spec: Dict[str, Any]) -> List[str]:
        """Validate OpenAPI version field."""
        suggestions = []
        version = spec.get("openapi", "")

        if not version:
            suggestions.append("Missing required 'openapi' field")
        elif not isinstance(version, str):
            suggestions.append("OpenAPI version must be a string")
        elif not version.startswith("3."):
            suggestions.append("OpenAPI version should be 3.x (e.g., '3.0.0', '3.1.0')")

        return suggestions

    def _validate_required_fields(self, spec: Dict[str, Any]) -> List[str]:
        """Validate required top-level fields."""
        suggestions = []
        required_fields = ["info", "paths"]

        for field in required_fields:
            if field not in spec:
                suggestions.append(f"Missing required '{field}' field")

        return suggestions

    def _validate_info_section(self, spec: Dict[str, Any]) -> List[str]:
        """Validate info section."""
        suggestions = []
        info = spec.get("info", {})

        if not isinstance(info, dict):
            suggestions.append("'info' field must be an object")
            return suggestions

        required_info_fields = ["title", "version"]
        for field in required_info_fields:
            if field not in info:
                suggestions.append(f"Missing required 'info.{field}' field")
            elif not isinstance(info[field], str):
                suggestions.append(f"'info.{field}' must be a string")

        return suggestions

    def _validate_paths_section(self, spec: Dict[str, Any]) -> List[str]:
        """Validate paths section."""
        suggestions = []
        paths = spec.get("paths", {})

        if not isinstance(paths, dict):
            suggestions.append("'paths' field must be an object")
            return suggestions

        for path, operations in paths.items():
            if not isinstance(operations, dict):
                suggestions.append(
                    f"Path '{path}' must contain an object with operations"
                )
                continue

            if not operations:
                suggestions.append(f"Path '{path}' has no operations")
                continue

            for method, operation in operations.items():
                # Validate HTTP method
                if method.lower() not in self.VALID_HTTP_METHODS:
                    suggestions.append(
                        f"Invalid HTTP method '{method}' in path '{path}'"
                    )
                    continue

                if not isinstance(operation, dict):
                    suggestions.append(f"Operation '{method} {path}' must be an object")
                    continue

                # Check for missing summary
                if "summary" not in operation:
                    suggestions.append(
                        f"Operation '{method} {path}' is missing summary"
                    )

                # Check for missing responses
                if "responses" not in operation:
                    suggestions.append(
                        f"Operation '{method} {path}' is missing responses"
                    )
                elif not isinstance(operation["responses"], dict):
                    suggestions.append(
                        f"Operation '{method} {path}' responses must be an object"
                    )
                elif not operation["responses"]:
                    suggestions.append(
                        f"Operation '{method} {path}' has no response definitions"
                    )

        return suggestions

    def _validate_components_section(self, spec: Dict[str, Any]) -> List[str]:
        """Validate components section."""
        suggestions = []
        components = spec.get("components", {})

        if not components:
            return suggestions

        if not isinstance(components, dict):
            suggestions.append("'components' field must be an object")
            return suggestions

        # Validate schemas
        schemas = components.get("schemas", {})
        if schemas and isinstance(schemas, dict):
            for schema_name, schema_def in schemas.items():
                if not isinstance(schema_def, dict):
                    suggestions.append(f"Schema '{schema_name}' must be an object")
                    continue

                if not schema_def:
                    suggestions.append(
                        f"Schema '{schema_name}' is empty - consider adding 'type' or 'properties'"
                    )
                elif (
                    "type" not in schema_def
                    and "properties" not in schema_def
                    and "$ref" not in schema_def
                ):
                    suggestions.append(
                        f"Schema '{schema_name}' should define 'type', 'properties', or '$ref'"
                    )

        return suggestions

    def _validate_security(self, spec: Dict[str, Any]) -> List[str]:
        """Validate security configuration."""
        suggestions = []
        components = spec.get("components", {})
        security_schemes = {}
        if isinstance(components, dict):
            security_schemes = components.get("securitySchemes", {})

        # Check if operations reference security schemes that don't exist
        paths = spec.get("paths", {})
        referenced_schemes = set()

        # Only process if paths is actually a dictionary
        if isinstance(paths, dict):
            for path, operations in paths.items():
                if isinstance(operations, dict):
                    for method, operation in operations.items():
                        if isinstance(operation, dict) and "security" in operation:
                            security = operation["security"]
                            if isinstance(security, list):
                                for security_req in security:
                                    if isinstance(security_req, dict):
                                        referenced_schemes.update(security_req.keys())

        # Check for undefined security schemes
        for scheme in referenced_schemes:
            if scheme not in security_schemes:
                suggestions.append(
                    f"Security scheme '{scheme}' is referenced but not defined in components.securitySchemes"
                )

        return suggestions
