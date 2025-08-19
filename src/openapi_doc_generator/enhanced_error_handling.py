"""
Enhanced Error Handling and Recovery System

This module provides enterprise-grade error handling, validation, and
recovery mechanisms for the OpenAPI documentation generator, ensuring
robust operation under all conditions.
"""

import logging
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Generator, List, Optional, Type


class ErrorSeverity(Enum):
    """Error severity levels for categorized handling."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for specialized handling."""
    VALIDATION = "validation"
    FILESYSTEM = "filesystem"
    NETWORK = "network"
    PARSING = "parsing"
    FRAMEWORK_DETECTION = "framework_detection"
    SCHEMA_INFERENCE = "schema_inference"
    OUTPUT_GENERATION = "output_generation"
    PLUGIN_LOADING = "plugin_loading"


@dataclass
class ErrorContext:
    """Context information for error reporting and analysis."""
    operation: str
    file_path: Optional[str] = None
    framework: Optional[str] = None
    component: Optional[str] = None
    user_input: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, float]] = None


@dataclass
class EnhancedError:
    """Enhanced error with categorization and context."""
    original_error: Exception
    category: ErrorCategory
    severity: ErrorSeverity
    context: ErrorContext
    recovery_suggestions: List[str]
    user_friendly_message: str
    technical_details: str


class EnhancedErrorHandler:
    """Enterprise-grade error handling and recovery system."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.error_count = 0
        self.recovery_attempts = 0
        self.error_history: List[EnhancedError] = []
    
    @contextmanager
    def error_context(self, context: ErrorContext) -> Generator[None, None, None]:
        """Context manager for enhanced error handling."""
        try:
            yield
        except Exception as e:
            enhanced_error = self._enhance_error(e, context)
            self._log_enhanced_error(enhanced_error)
            self._attempt_recovery(enhanced_error)
            raise enhanced_error.original_error from e
    
    def _enhance_error(self, error: Exception, context: ErrorContext) -> EnhancedError:
        """Enhance error with categorization and context."""
        category = self._categorize_error(error, context)
        severity = self._assess_severity(error, category, context)
        
        recovery_suggestions = self._generate_recovery_suggestions(error, category, context)
        user_friendly_message = self._generate_user_friendly_message(error, category, context)
        technical_details = self._generate_technical_details(error, context)
        
        enhanced_error = EnhancedError(
            original_error=error,
            category=category,
            severity=severity,
            context=context,
            recovery_suggestions=recovery_suggestions,
            user_friendly_message=user_friendly_message,
            technical_details=technical_details
        )
        
        self.error_history.append(enhanced_error)
        self.error_count += 1
        
        return enhanced_error
    
    def _categorize_error(self, error: Exception, context: ErrorContext) -> ErrorCategory:
        """Categorize error based on type and context."""
        error_type = type(error).__name__
        
        if isinstance(error, (ValueError, TypeError)) and "schema" in str(error).lower():
            return ErrorCategory.SCHEMA_INFERENCE
        elif isinstance(error, (FileNotFoundError, PermissionError, OSError)):
            return ErrorCategory.FILESYSTEM
        elif isinstance(error, (ImportError, ModuleNotFoundError)) and context.component == "plugin":
            return ErrorCategory.PLUGIN_LOADING
        elif isinstance(error, (SyntaxError, ValueError)) and context.operation == "parsing":
            return ErrorCategory.PARSING
        elif "framework" in context.operation.lower():
            return ErrorCategory.FRAMEWORK_DETECTION
        elif "validation" in context.operation.lower():
            return ErrorCategory.VALIDATION
        elif "network" in str(error).lower() or "connection" in str(error).lower():
            return ErrorCategory.NETWORK
        else:
            return ErrorCategory.OUTPUT_GENERATION
    
    def _assess_severity(self, error: Exception, category: ErrorCategory, context: ErrorContext) -> ErrorSeverity:
        """Assess error severity based on type, category, and context."""
        if isinstance(error, (SystemExit, KeyboardInterrupt)):
            return ErrorSeverity.CRITICAL
        elif isinstance(error, (MemoryError, RecursionError)):
            return ErrorSeverity.CRITICAL
        elif category == ErrorCategory.FILESYSTEM and isinstance(error, PermissionError):
            return ErrorSeverity.HIGH
        elif category == ErrorCategory.PLUGIN_LOADING:
            return ErrorSeverity.LOW  # Plugins are optional
        elif category == ErrorCategory.VALIDATION:
            return ErrorSeverity.MEDIUM
        elif isinstance(error, FileNotFoundError) and context.file_path:
            return ErrorSeverity.HIGH
        else:
            return ErrorSeverity.MEDIUM
    
    def _generate_recovery_suggestions(self, error: Exception, category: ErrorCategory, 
                                     context: ErrorContext) -> List[str]:
        """Generate context-aware recovery suggestions."""
        suggestions = []
        
        if category == ErrorCategory.FILESYSTEM:
            if isinstance(error, FileNotFoundError):
                suggestions.extend([
                    f"Verify that the file '{context.file_path}' exists",
                    "Check file path for typos or case sensitivity",
                    "Ensure you have read permissions for the file"
                ])
            elif isinstance(error, PermissionError):
                suggestions.extend([
                    "Check file permissions and ensure read access",
                    "Run with appropriate user privileges",
                    "Verify the file is not locked by another process"
                ])
        
        elif category == ErrorCategory.FRAMEWORK_DETECTION:
            suggestions.extend([
                "Ensure your application uses a supported framework (FastAPI, Flask, Django, etc.)",
                "Check that framework-specific decorators and patterns are used correctly",
                "Verify imports and dependencies are properly configured"
            ])
        
        elif category == ErrorCategory.PLUGIN_LOADING:
            suggestions.extend([
                "Check that optional plugin dependencies are installed",
                "Verify plugin configuration and entry points",
                "Consider disabling problematic plugins temporarily"
            ])
        
        elif category == ErrorCategory.SCHEMA_INFERENCE:
            suggestions.extend([
                "Ensure proper type annotations are used in your code",
                "Check that dataclasses or Pydantic models are correctly defined",
                "Verify complex types have proper schema definitions"
            ])
        
        return suggestions
    
    def _generate_user_friendly_message(self, error: Exception, category: ErrorCategory, 
                                      context: ErrorContext) -> str:
        """Generate user-friendly error message."""
        base_messages = {
            ErrorCategory.FILESYSTEM: "There was a problem accessing the file system.",
            ErrorCategory.FRAMEWORK_DETECTION: "Unable to detect or process your web framework.",
            ErrorCategory.PLUGIN_LOADING: "A plugin failed to load (this may not affect core functionality).",
            ErrorCategory.SCHEMA_INFERENCE: "Unable to automatically infer API schemas from your code.",
            ErrorCategory.PARSING: "There was a problem parsing your application code.",
            ErrorCategory.VALIDATION: "Input validation failed.",
            ErrorCategory.NETWORK: "A network-related error occurred.",
            ErrorCategory.OUTPUT_GENERATION: "Failed to generate the requested output format."
        }
        
        base_message = base_messages.get(category, "An unexpected error occurred.")
        
        if context.file_path:
            return f"{base_message} File: {context.file_path}"
        elif context.framework:
            return f"{base_message} Framework: {context.framework}"
        else:
            return base_message
    
    def _generate_technical_details(self, error: Exception, context: ErrorContext) -> str:
        """Generate technical details for debugging."""
        details = [
            f"Error Type: {type(error).__name__}",
            f"Error Message: {str(error)}",
            f"Operation: {context.operation}",
        ]
        
        if context.file_path:
            details.append(f"File Path: {context.file_path}")
        if context.framework:
            details.append(f"Framework: {context.framework}")
        if context.component:
            details.append(f"Component: {context.component}")
        
        details.append(f"Traceback: {traceback.format_exc()}")
        
        return "\n".join(details)
    
    def _log_enhanced_error(self, enhanced_error: EnhancedError) -> None:
        """Log enhanced error with appropriate level."""
        log_level = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }[enhanced_error.severity]
        
        self.logger.log(
            log_level,
            f"[{enhanced_error.category.value.upper()}] {enhanced_error.user_friendly_message}\n"
            f"Technical Details: {enhanced_error.technical_details}\n"
            f"Recovery Suggestions: {'; '.join(enhanced_error.recovery_suggestions)}"
        )
    
    def _attempt_recovery(self, enhanced_error: EnhancedError) -> None:
        """Attempt automatic recovery for certain error types."""
        self.recovery_attempts += 1
        
        # Implement specific recovery strategies
        if enhanced_error.category == ErrorCategory.PLUGIN_LOADING:
            self.logger.info("Plugin loading failed, continuing with core functionality")
            return  # Graceful degradation
        
        # Add more recovery strategies as needed
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get comprehensive error summary for monitoring."""
        if not self.error_history:
            return {"status": "healthy", "error_count": 0}
        
        categories = {}
        severities = {}
        
        for error in self.error_history:
            categories[error.category.value] = categories.get(error.category.value, 0) + 1
            severities[error.severity.value] = severities.get(error.severity.value, 0) + 1
        
        return {
            "status": "has_errors",
            "total_errors": self.error_count,
            "recovery_attempts": self.recovery_attempts,
            "categories": categories,
            "severities": severities,
            "recent_errors": [
                {
                    "category": error.category.value,
                    "severity": error.severity.value,
                    "message": error.user_friendly_message,
                    "operation": error.context.operation
                }
                for error in self.error_history[-5:]  # Last 5 errors
            ]
        }


# Global error handler instance
_global_error_handler = EnhancedErrorHandler()


def get_error_handler() -> EnhancedErrorHandler:
    """Get global error handler instance."""
    return _global_error_handler


def with_error_handling(operation: str, file_path: Optional[str] = None, 
                       framework: Optional[str] = None, component: Optional[str] = None):
    """Decorator for adding enhanced error handling to functions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            context = ErrorContext(
                operation=operation,
                file_path=file_path,
                framework=framework,
                component=component
            )
            
            with get_error_handler().error_context(context):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator