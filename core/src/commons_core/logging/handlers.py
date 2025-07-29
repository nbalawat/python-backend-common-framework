"""Custom log handlers for various outputs."""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from logging.handlers import RotatingFileHandler as BaseRotatingFileHandler

from .formatters import JsonFormatter, StructuredFormatter


class CloudHandler(logging.Handler):
    """Base handler for cloud logging services."""
    
    def __init__(self, service: str, **kwargs: Any) -> None:
        super().__init__()
        self.service = service
        self.kwargs = kwargs
        
    def emit(self, record: logging.LogRecord) -> None:
        """Emit log record to cloud service."""
        # This is a base class - specific implementations would go in cloud module
        pass


class RotatingFileHandler(BaseRotatingFileHandler):
    """Enhanced rotating file handler with compression support."""
    
    def __init__(
        self,
        filename: str,
        mode: str = "a",
        maxBytes: int = 10485760,  # 10MB
        backupCount: int = 5,
        encoding: Optional[str] = "utf-8",
        delay: bool = False,
        compress: bool = True,
    ) -> None:
        super().__init__(filename, mode, maxBytes, backupCount, encoding, delay)
        self.compress = compress
        
    def doRollover(self) -> None:
        """Perform rollover with optional compression."""
        super().doRollover()
        
        if self.compress and self.backupCount > 0:
            import gzip
            import shutil
            
            # Compress the rotated file
            source = f"{self.baseFilename}.1"
            if Path(source).exists():
                with open(source, "rb") as f_in:
                    with gzip.open(f"{source}.gz", "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                Path(source).unlink()


def get_handler(name: str, format: str = "json", **kwargs: Any) -> Optional[logging.Handler]:
    """Get a configured log handler by name."""
    if name == "console":
        handler = logging.StreamHandler(sys.stdout)
        if format == "json":
            handler.setFormatter(JsonFormatter())
        else:
            handler.setFormatter(StructuredFormatter(colors=sys.stdout.isatty()))
        return handler
        
    elif name == "file":
        filename = kwargs.get("filename", "app.log")
        handler = RotatingFileHandler(filename, **kwargs)
        handler.setFormatter(JsonFormatter())
        return handler
        
    elif name.startswith("cloud:"):
        service = name.split(":", 1)[1]
        return CloudHandler(service, **kwargs)
        
    return None