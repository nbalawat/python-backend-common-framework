"""Log formatters for different output formats."""

import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional
from pythonjsonlogger import jsonlogger


class JsonFormatter(jsonlogger.JsonFormatter):
    """JSON log formatter with enhanced fields."""
    
    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        style: str = "%",
        validate: bool = True,
        *,
        defaults: Optional[Dict[str, Any]] = None,
        static_fields: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(fmt, datefmt, style, validate, defaults=defaults)
        self.static_fields = static_fields or {}
        
    def add_fields(
        self,
        log_record: Dict[str, Any],
        record: logging.LogRecord,
        message_dict: Dict[str, Any],
    ) -> None:
        """Add custom fields to log record."""
        super().add_fields(log_record, record, message_dict)
        
        # Add static fields
        log_record.update(self.static_fields)
        
        # Add standard fields
        log_record["timestamp"] = datetime.utcnow().isoformat()
        log_record["logger"] = record.name
        log_record["level"] = record.levelname
        log_record["thread"] = record.thread
        log_record["thread_name"] = record.threadName
        log_record["process"] = record.process
        log_record["process_name"] = record.processName
        
        # Add location info
        if hasattr(record, "pathname"):
            log_record["file"] = record.pathname
            log_record["line"] = record.lineno
            log_record["function"] = record.funcName
            
        # Add exception info if present
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
            
        # Remove internal fields
        for field in ["message", "msg", "args", "created", "msecs", "relativeCreated"]:
            log_record.pop(field, None)


class StructuredFormatter(logging.Formatter):
    """Human-readable structured formatter."""
    
    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        style: str = "%",
        validate: bool = True,
        *,
        colors: bool = True,
    ) -> None:
        super().__init__(fmt, datefmt, style, validate)
        self.colors = colors
        
    def format(self, record: logging.LogRecord) -> str:
        """Format log record in structured human-readable format."""
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        # Get level with color
        if self.colors:
            level_colors = {
                "DEBUG": "\033[36m",    # Cyan
                "INFO": "\033[32m",     # Green
                "WARNING": "\033[33m",  # Yellow
                "ERROR": "\033[31m",    # Red
                "CRITICAL": "\033[35m", # Magenta
            }
            reset = "\033[0m"
            level = f"{level_colors.get(record.levelname, '')}{record.levelname:8}{reset}"
        else:
            level = f"{record.levelname:8}"
            
        # Format message
        message = record.getMessage()
        
        # Format location
        location = f"{record.name}:{record.funcName}:{record.lineno}"
        
        # Build base log line
        log_line = f"{timestamp} {level} [{location}] {message}"
        
        # Add extra fields
        extra_fields = {
            k: v for k, v in record.__dict__.items()
            if k not in ["name", "msg", "args", "created", "filename", "funcName",
                         "levelname", "levelno", "lineno", "module", "msecs", "message",
                         "pathname", "process", "processName", "relativeCreated", "thread",
                         "threadName", "exc_info", "exc_text", "stack_info"]
        }
        
        if extra_fields:
            field_str = " ".join(f"{k}={v}" for k, v in extra_fields.items())
            log_line += f" | {field_str}"
            
        # Add exception info
        if record.exc_info:
            log_line += f"\n{self.formatException(record.exc_info)}"
            
        return log_line