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
            suggestions.extend(self._validate_path_operations(path, operations))

        return suggestions

    def _validate_path_operations(self, path: str, operations: Any) -> List[str]:
        """Validate operations for a single path."""
        suggestions = []
        
        if not isinstance(operations, dict):
            suggestions.append(
                f"Path '{path}' must contain an object with operations"
            )
            return suggestions

        if not operations:
            suggestions.append(f"Path '{path}' has no operations")
            return suggestions

        for method, operation in operations.items():
            suggestions.extend(self._validate_single_operation(method, path, operation))

        return suggestions

    def _validate_single_operation(self, method: str, path: str, operation: Any) -> List[str]:
        """Validate a single operation."""
        suggestions = []
        
        # Validate HTTP method
        if method.lower() not in self.VALID_HTTP_METHODS:
            suggestions.append(
                f"Invalid HTTP method '{method}' in path '{path}'"
            )
            return suggestions

        if not isinstance(operation, dict):
            suggestions.append(f"Operation '{method} {path}' must be an object")
            return suggestions

        # Check for missing summary
        if "summary" not in operation:
            suggestions.append(
                f"Operation '{method} {path}' is missing summary"
            )

        # Check for missing or invalid responses
        suggestions.extend(self._validate_operation_responses(method, path, operation))

        return suggestions

    def _validate_operation_responses(self, method: str, path: str, operation: Dict[str, Any]) -> List[str]:
        """Validate responses section of an operation."""
        suggestions = []
        
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
        security_schemes = self._extract_security_schemes(spec)
        referenced_schemes = self._find_referenced_security_schemes(spec)

        # Check for undefined security schemes
        for scheme in referenced_schemes:
            if scheme not in security_schemes:
                suggestions.append(
                    f"Security scheme '{scheme}' is referenced but not defined in components.securitySchemes"
                )

        return suggestions

    def _extract_security_schemes(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Extract security schemes from spec components."""
        components = spec.get("components", {})
        if isinstance(components, dict):
            return components.get("securitySchemes", {})
        return {}

    def _find_referenced_security_schemes(self, spec: Dict[str, Any]) -> set[str]:
        """Find all security schemes referenced in operations."""
        referenced_schemes = set()
        paths = spec.get("paths", {})

        if not isinstance(paths, dict):
            return referenced_schemes

        for operations in paths.values():
            if isinstance(operations, dict):
                for operation in operations.values():
                    if isinstance(operation, dict) and "security" in operation:
                        schemes = self._extract_schemes_from_operation(operation)
                        referenced_schemes.update(schemes)

        return referenced_schemes

    def _extract_schemes_from_operation(self, operation: Dict[str, Any]) -> set[str]:
        """Extract security scheme names from a single operation."""
        schemes = set()
        security = operation.get("security", [])
        
        if isinstance(security, list):
            for security_req in security:
                if isinstance(security_req, dict):
                    schemes.update(security_req.keys())
                    
        return schemes
