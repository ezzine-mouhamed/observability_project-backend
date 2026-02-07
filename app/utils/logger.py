import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from app.config import Config


class StructuredTextFormatter(logging.Formatter):
    """Human-readable structured log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        # Get formatted timestamp
        timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
        
        # Get basic message
        message = record.getMessage()
        
        # Add extra fields if present - THIS IS WHAT WAS MISSING
        extra_info = ""
        if hasattr(record, "extra") and record.extra:
            # Format extra fields as key=value pairs
            extra_parts = []
            for key, value in record.extra.items():
                if key == "timestamp":
                    continue  # Skip timestamp, we have our own
                if key == "traceback" and record.exc_info:
                    continue  # Skip traceback if we have exc_info
                extra_parts.append(f"{key}={self._format_value(value)}")
            
            if extra_parts:
                extra_info = " | " + " | ".join(extra_parts)
        
        # Build the log line with extra info
        log_line = f"{timestamp} {record.levelname:8} {record.name:40} - {message}{extra_info}"
        
        # Add exception if present
        if record.exc_info:
            exc_str = self.formatException(record.exc_info)
            if exc_str:
                # Clean and format exception
                exc_lines = exc_str.strip().split('\n')
                formatted_exc = "\n    ".join(exc_lines)
                log_line = f"{log_line}\n    {formatted_exc}"
        
        return log_line
    
    def _format_value(self, value: Any) -> str:
        """Format a value for log output."""
        if isinstance(value, dict):
            items = []
            for k, v in value.items():
                items.append(f"{k}:{self._format_value(v)}")
            return f"{{{', '.join(items)}}}"
        elif isinstance(value, list):
            items = [self._format_value(v) for v in value[:3]]
            suffix = "..." if len(value) > 3 else ""
            return f"[{', '.join(items)}{suffix}]"
        elif isinstance(value, str):
            # Truncate very long strings
            if len(value) > 100:
                return f'"{value[:100]}..."'
            return f'"{value}"'
        elif isinstance(value, (int, float, bool)):
            return str(value)
        elif value is None:
            return "null"
        else:
            str_val = str(value)
            if len(str_val) > 100:
                return f"{str_val[:100]}..."
            return str_val


class ContextFilter(logging.Filter):
    """Filter to add context to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        # Ensure extra attributes exist
        if not hasattr(record, "extra"):
            record.extra = {}
        return True


def get_logger(name: str) -> logging.Logger:
    """Get a logger with structured text formatting."""
    logger = logging.getLogger(name)

    # Avoid duplicate handlers
    if not logger.handlers:
        logger.setLevel(getattr(logging, Config.LOG_LEVEL))

        # Add console handler with structured text formatter
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(StructuredTextFormatter())
        console_handler.addFilter(ContextFilter())

        logger.addHandler(console_handler)
        logger.propagate = False

    return logger


def setup_logging():
    """Setup application-wide logging configuration."""
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, Config.LOG_LEVEL))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add structured text formatter to root logger
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(StructuredTextFormatter())
    console_handler.addFilter(ContextFilter())

    root_logger.addHandler(console_handler)


def log_execution_context(
    operation: str, context: Dict[str, Any], logger: Optional[logging.Logger] = None
):
    """Log execution context in a structured way."""
    if logger is None:
        logger = get_logger(__name__)

    logger.info(f"Execution context: {operation}", extra={
        "operation": operation,
        "context": context,
    })


def log_decision(
    decision_type: str,
    reasoning: str,
    context: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
):
    """Log a decision with reasoning."""
    if logger is None:
        logger = get_logger(__name__)

    logger.info(f"Decision: {decision_type}", extra={
        "decision_type": decision_type,
        "reasoning": reasoning,
        "context": context,
    })


def log_error_with_context(
    error: Exception, 
    context: Dict[str, Any], 
    logger: Optional[logging.Logger] = None
):
    """Log an error with context."""
    if logger is None:
        logger = get_logger(__name__)

    logger.error(f"Error: {type(error).__name__}", extra={
        "error_type": type(error).__name__,
        "error_message": str(error),
        "context": context,
    }, exc_info=True)  # ADDED exc_info=True to include stack trace
