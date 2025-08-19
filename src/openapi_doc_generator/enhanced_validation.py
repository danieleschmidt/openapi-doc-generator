"""
Enhanced Input Validation and Sanitization System

This module provides comprehensive input validation, sanitization, and
security measures for all user inputs and file operations.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

from .enhanced_error_handling import ErrorCategory, ErrorContext, get_error_handler


class ValidationError(ValueError):
    """Custom validation error with enhanced context."""
    pass


class InputValidator:
    """Comprehensive input validation and sanitization system."""
    
    def __init__(self):
        self.error_handler = get_error_handler()
        
        # Define allowed file extensions for security
        self.allowed_extensions = {
            '.py', '.js', '.ts', '.jsx', '.tsx', '.json', '.yaml', '.yml',
            '.graphql', '.gql', '.md', '.txt', '.toml', '.cfg', '.ini'
        }
        
        # Define dangerous patterns to reject
        self.dangerous_patterns = [
            r'\.\./+',  # Path traversal
            r'[<>:"|?*]',  # Invalid filename characters on Windows
            r'^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(\.|$)',  # Windows reserved names
        ]
        
        # Maximum file size (50MB)
        self.max_file_size = 50 * 1024 * 1024
        
        # Maximum path length
        self.max_path_length = 4096
    
    def validate_file_path(self, file_path: Union[str, Path], 
                          operation: str = "file_access") -> Path:
        """Validate and sanitize file path with comprehensive security checks."""
        context = ErrorContext(operation=operation, file_path=str(file_path))
        
        with self.error_handler.error_context(context):
            # Convert to Path object
            if isinstance(file_path, str):
                path = Path(file_path)
            else:
                path = file_path
            
            # Basic validation
            if not path:
                raise ValidationError("File path cannot be empty")
            
            path_str = str(path)
            
            # Check path length
            if len(path_str) > self.max_path_length:
                raise ValidationError(f"File path too long (max {self.max_path_length} characters)")
            
            # Check for dangerous patterns
            for pattern in self.dangerous_patterns:
                if re.search(pattern, path_str, re.IGNORECASE):
                    raise ValidationError(f"File path contains dangerous pattern: {pattern}")
            
            # Resolve path to prevent path traversal
            try:
                resolved_path = path.resolve()
            except (OSError, ValueError) as e:
                raise ValidationError(f"Invalid file path: {e}") from e
            
            # Check if file exists
            if not resolved_path.exists():
                raise ValidationError(f"File does not exist: {resolved_path}")
            
            # Check if it's a file (not directory)
            if not resolved_path.is_file():
                raise ValidationError(f"Path is not a file: {resolved_path}")
            
            # Check file extension
            if resolved_path.suffix.lower() not in self.allowed_extensions:
                raise ValidationError(
                    f"File extension '{resolved_path.suffix}' not allowed. "
                    f"Allowed extensions: {', '.join(sorted(self.allowed_extensions))}"
                )
            
            # Check file size
            try:
                file_size = resolved_path.stat().st_size
                if file_size > self.max_file_size:
                    size_mb = file_size / (1024 * 1024)
                    max_mb = self.max_file_size / (1024 * 1024)
                    raise ValidationError(f"File too large: {size_mb:.1f}MB (max {max_mb}MB)")
            except OSError as e:
                raise ValidationError(f"Cannot access file information: {e}") from e
            
            # Check read permissions
            if not os.access(resolved_path, os.R_OK):
                raise ValidationError(f"No read permission for file: {resolved_path}")
            
            return resolved_path
    
    def validate_output_path(self, output_path: Union[str, Path], 
                           operation: str = "output_generation") -> Path:
        """Validate output path for writing files."""
        context = ErrorContext(operation=operation, file_path=str(output_path))
        
        with self.error_handler.error_context(context):
            # Convert to Path object
            if isinstance(output_path, str):
                path = Path(output_path)
            else:
                path = output_path
            
            # Basic validation
            if not path:
                raise ValidationError("Output path cannot be empty")
            
            path_str = str(path)
            
            # Check path length
            if len(path_str) > self.max_path_length:
                raise ValidationError(f"Output path too long (max {self.max_path_length} characters)")
            
            # Check for dangerous patterns
            for pattern in self.dangerous_patterns:
                if re.search(pattern, path_str, re.IGNORECASE):
                    raise ValidationError(f"Output path contains dangerous pattern: {pattern}")
            
            # Resolve parent directory
            try:
                parent_dir = path.parent.resolve()
            except (OSError, ValueError) as e:
                raise ValidationError(f"Invalid output directory: {e}") from e
            
            # Check if parent directory exists and is writable
            if not parent_dir.exists():
                raise ValidationError(f"Output directory does not exist: {parent_dir}")
            
            if not parent_dir.is_dir():
                raise ValidationError(f"Output parent path is not a directory: {parent_dir}")
            
            if not os.access(parent_dir, os.W_OK):
                raise ValidationError(f"No write permission for directory: {parent_dir}")
            
            # If file exists, check if it's writable
            resolved_path = parent_dir / path.name
            if resolved_path.exists():
                if not resolved_path.is_file():
                    raise ValidationError(f"Output path exists but is not a file: {resolved_path}")
                if not os.access(resolved_path, os.W_OK):
                    raise ValidationError(f"No write permission for existing file: {resolved_path}")
            
            return resolved_path
    
    def validate_format(self, format_str: str, allowed_formats: List[str]) -> str:
        """Validate output format."""
        context = ErrorContext(operation="format_validation")
        
        with self.error_handler.error_context(context):
            if not format_str:
                raise ValidationError("Format cannot be empty")
            
            # Sanitize format string
            sanitized_format = format_str.strip().lower()
            
            # Check against allowed formats
            if sanitized_format not in allowed_formats:
                raise ValidationError(
                    f"Invalid format '{format_str}'. "
                    f"Allowed formats: {', '.join(allowed_formats)}"
                )
            
            return sanitized_format
    
    def validate_api_title(self, title: str) -> str:
        """Validate and sanitize API title."""
        context = ErrorContext(operation="title_validation")
        
        with self.error_handler.error_context(context):
            if not title:
                raise ValidationError("API title cannot be empty")
            
            # Basic sanitization
            sanitized_title = title.strip()
            
            # Check length
            if len(sanitized_title) > 200:
                raise ValidationError("API title too long (max 200 characters)")
            
            # Check for dangerous characters
            if re.search(r'[<>&"\'`]', sanitized_title):
                raise ValidationError("API title contains dangerous characters")
            
            return sanitized_title
    
    def validate_api_version(self, version: str) -> str:
        """Validate and sanitize API version."""
        context = ErrorContext(operation="version_validation")
        
        with self.error_handler.error_context(context):
            if not version:
                raise ValidationError("API version cannot be empty")
            
            # Basic sanitization
            sanitized_version = version.strip()
            
            # Check format (semantic versioning or simple version)
            version_pattern = r'^[0-9]+(\.[0-9]+)*(-[a-zA-Z0-9\-\.]+)?$'
            if not re.match(version_pattern, sanitized_version):
                raise ValidationError(
                    f"Invalid version format '{version}'. "
                    "Use semantic versioning (e.g., '1.0.0', '2.1.3-beta')"
                )
            
            return sanitized_version
    
    def validate_log_format(self, log_format: str) -> str:
        """Validate log format."""
        allowed_formats = ['standard', 'json']
        return self.validate_format(log_format, allowed_formats)
    
    def validate_url(self, url: str, operation: str = "url_validation") -> str:
        """Validate URL format and safety."""
        context = ErrorContext(operation=operation)
        
        with self.error_handler.error_context(context):
            if not url:
                raise ValidationError("URL cannot be empty")
            
            # Basic sanitization
            sanitized_url = url.strip()
            
            # Parse URL
            try:
                parsed = urlparse(sanitized_url)
            except Exception as e:
                raise ValidationError(f"Invalid URL format: {e}") from e
            
            # Check scheme
            if parsed.scheme not in ['http', 'https']:
                raise ValidationError(f"Invalid URL scheme '{parsed.scheme}'. Only http/https allowed")
            
            # Check for localhost/private IPs in production
            if parsed.hostname:
                if parsed.hostname in ['localhost', '127.0.0.1', '::1']:
                    # Allow localhost for development
                    pass
                elif parsed.hostname.startswith('10.') or parsed.hostname.startswith('192.168.'):
                    # Private IPs - log warning but allow
                    import logging
                    logging.getLogger(__name__).warning(f"Using private IP in URL: {parsed.hostname}")
            
            return sanitized_url
    
    def validate_environment_variables(self, env_vars: Dict[str, str]) -> Dict[str, str]:
        """Validate environment variables for security."""
        context = ErrorContext(operation="env_validation")
        
        with self.error_handler.error_context(context):
            validated_vars = {}
            
            # List of sensitive variable patterns to warn about
            sensitive_patterns = [
                r'.*password.*', r'.*secret.*', r'.*key.*', r'.*token.*',
                r'.*api_key.*', r'.*access.*', r'.*auth.*'
            ]
            
            for key, value in env_vars.items():
                # Validate key
                if not re.match(r'^[A-Z_][A-Z0-9_]*$', key):
                    raise ValidationError(f"Invalid environment variable name: {key}")
                
                # Check for sensitive data in non-secure contexts
                for pattern in sensitive_patterns:
                    if re.match(pattern, key.lower()):
                        # Log warning without exposing value
                        import logging
                        logging.getLogger(__name__).warning(
                            f"Potentially sensitive environment variable: {key}"
                        )
                        break
                
                # Basic value validation
                if len(value) > 10000:  # Reasonable limit
                    raise ValidationError(f"Environment variable {key} value too long")
                
                validated_vars[key] = value
            
            return validated_vars
    
    def validate_plugin_name(self, plugin_name: str) -> str:
        """Validate plugin name for security."""
        context = ErrorContext(operation="plugin_validation", component="plugin")
        
        with self.error_handler.error_context(context):
            if not plugin_name:
                raise ValidationError("Plugin name cannot be empty")
            
            # Basic sanitization
            sanitized_name = plugin_name.strip()
            
            # Check format
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', sanitized_name):
                raise ValidationError(
                    f"Invalid plugin name '{plugin_name}'. "
                    "Must be a valid Python identifier"
                )
            
            # Check length
            if len(sanitized_name) > 100:
                raise ValidationError("Plugin name too long (max 100 characters)")
            
            return sanitized_name
    
    def sanitize_user_input(self, user_input: str, max_length: int = 1000) -> str:
        """General user input sanitization."""
        if not user_input:
            return ""
        
        # Basic sanitization
        sanitized = user_input.strip()
        
        # Check length
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        # Remove or escape dangerous characters
        sanitized = re.sub(r'[<>&"\'`]', '', sanitized)
        
        return sanitized


# Global validator instance
_global_validator = InputValidator()


def get_validator() -> InputValidator:
    """Get global validator instance."""
    return _global_validator