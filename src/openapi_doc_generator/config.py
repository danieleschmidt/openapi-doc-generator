"""Configuration constants and defaults for OpenAPI Doc Generator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class OpenAPIConfig:
    """Configuration for OpenAPI specification generation."""
    
    # OpenAPI specification version
    OPENAPI_VERSION: str = "3.0.0"
    
    # Default API metadata
    DEFAULT_API_TITLE: str = "API"
    DEFAULT_API_VERSION: str = "1.0.0"
    
    # Default HTTP response configuration
    DEFAULT_SUCCESS_STATUS: str = "200"
    DEFAULT_SUCCESS_DESCRIPTION: str = "Success"
    DEFAULT_SUCCESS_STATUS_INT: int = 200
    
    # Response schema template
    DEFAULT_SUCCESS_RESPONSE: Dict[str, Any] = None
    
    def __post_init__(self) -> None:
        """Initialize computed fields after dataclass creation."""
        if self.DEFAULT_SUCCESS_RESPONSE is None:
            self.DEFAULT_SUCCESS_RESPONSE = {
                self.DEFAULT_SUCCESS_STATUS: {
                    "description": self.DEFAULT_SUCCESS_DESCRIPTION
                }
            }


# Global configuration instance
config = OpenAPIConfig()