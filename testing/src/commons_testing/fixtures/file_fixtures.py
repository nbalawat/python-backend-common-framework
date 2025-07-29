"""File-based testing fixtures."""

import tempfile
import pytest
from pathlib import Path
from typing import Generator, Any
from contextlib import contextmanager

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture  
def temp_file() -> Generator[Path, None, None]:
    """Create a temporary file for testing."""
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        file_path = Path(tmpfile.name)
        try:
            yield file_path
        finally:
            if file_path.exists():
                file_path.unlink()

@contextmanager
def create_temp_file(content: str = "", suffix: str = "") -> Generator[Path, None, None]:
    """Context manager for creating temporary files."""
    with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as tmpfile:
        tmpfile.write(content)
        file_path = Path(tmpfile.name)
    
    try:
        yield file_path
    finally:
        if file_path.exists():
            file_path.unlink()