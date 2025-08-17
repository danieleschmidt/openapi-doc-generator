"""Structured logging configuration for OpenAPI Doc Generator."""

import json
import logging
import logging.config
import os
import sys
import time
import uuid
from typing import Optional


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hostname = os.uname().nodename if hasattr(os, 'uname') else 'unknown'

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Base log entry
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "hostname": self.hostname,
            "process_id": os.getpid(),
        }

        # Add correlation ID if available
        if hasattr(record, 'correlation_id'):
            log_entry["correlation_id"] = record.correlation_id

        # Add extra fields from record
        extra_fields = [
            'operation', 'duration_ms', 'memory_mb', 'file_path',
            'framework', 'route_count', 'error_code', 'user_agent'
        ]

        for field in extra_fields:
            if hasattr(record, field):
                log_entry[field] = getattr(record, field)

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info)
            }

        # Add stack info if present
        if record.stack_info:
            log_entry["stack_info"] = record.stack_info

        return json.dumps(log_entry)

    def formatTime(self, record: logging.LogRecord) -> str:
        """Format timestamp in ISO format."""
        return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(record.created))


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""

    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        # Add correlation ID to message if available
        message = record.getMessage()
        if hasattr(record, 'correlation_id'):
            message = f"[{record.correlation_id}] {message}"

        # Add performance info if available
        if hasattr(record, 'duration_ms'):
            message += f" (took {record.duration_ms:.2f}ms)"

        # Apply color
        color = self.COLORS.get(record.levelname, '')
        reset = self.RESET if color else ''

        formatted = f"{color}[{record.levelname}]{reset} {record.name}: {message}"

        # Add exception info if present
        if record.exc_info:
            formatted += f"\n{self.formatException(record.exc_info)}"

        return formatted


class CorrelationFilter(logging.Filter):
    """Filter to add correlation ID to log records."""

    def __init__(self):
        super().__init__()
        self._correlation_id: Optional[str] = None

    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation ID to record."""
        if not hasattr(record, 'correlation_id'):
            record.correlation_id = self._correlation_id or self.generate_correlation_id()
        return True

    def set_correlation_id(self, correlation_id: str) -> None:
        """Set correlation ID for current context."""
        self._correlation_id = correlation_id

    def clear_correlation_id(self) -> None:
        """Clear correlation ID."""
        self._correlation_id = None

    @staticmethod
    def generate_correlation_id() -> str:
        """Generate a new correlation ID."""
        return str(uuid.uuid4())[:8]


# Global correlation filter
correlation_filter = CorrelationFilter()


def setup_logging(
    level: str = "INFO",
    format_type: str = "text",
    enable_correlation: bool = True,
    log_file: Optional[str] = None
) -> None:
    """Setup logging configuration."""

    # Determine log level
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Choose formatter
    if format_type.lower() == "json":
        formatter = JSONFormatter()
    else:
        formatter = ColoredFormatter()

    # Setup handlers
    handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(JSONFormatter())  # Always use JSON for files
        handlers.append(file_handler)

    # Add correlation filter if enabled
    if enable_correlation:
        for handler in handlers:
            handler.addFilter(correlation_filter)

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        force=True
    )

    # Configure specific loggers
    logging.getLogger("openapi_doc_generator").setLevel(log_level)

    # Suppress noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with proper configuration."""
    logger = logging.getLogger(name)

    # Ensure logger has correlation filter
    if correlation_filter not in list(logger.filters):
        logger.addFilter(correlation_filter)

    return logger


def set_correlation_id(correlation_id: str) -> None:
    """Set correlation ID for current context."""
    correlation_filter.set_correlation_id(correlation_id)


def clear_correlation_id() -> None:
    """Clear correlation ID."""
    correlation_filter.clear_correlation_id()


def generate_correlation_id() -> str:
    """Generate a new correlation ID."""
    return correlation_filter.generate_correlation_id()


# Default logging configuration from environment
def configure_from_env() -> None:
    """Configure logging from environment variables."""
    log_level = os.getenv("LOG_LEVEL", "INFO")
    log_format = os.getenv("LOG_FORMAT", "text")
    log_file = os.getenv("LOG_FILE")

    setup_logging(
        level=log_level,
        format_type=log_format,
        log_file=log_file
    )


# Auto-configure if this module is imported
if not logging.getLogger().handlers:
    configure_from_env()
