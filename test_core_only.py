#!/usr/bin/env python3
"""Test just the core module functionality."""

import subprocess
import sys

test_script = '''
# Test core module
import commons_core
print("✓ commons_core imported successfully")

# Test core functionality  
from commons_core import ConfigManager, get_logger, BaseModel
logger = get_logger(__name__)
logger.info("Logger working!")

config = ConfigManager()
print("✓ Core ConfigManager created")

# Test BaseModel
class TestModel(BaseModel):
    name: str
    value: int = 42

model = TestModel(name="test")
print(f"✓ BaseModel working: {model.name}, {model.value}")

print("✓ Core module fully functional!")
'''

result = subprocess.run(
    ["uv", "run", "--no-project", "python", "-c", test_script],
    capture_output=True,
    text=True
)

if result.returncode == 0:
    print(result.stdout)
    print("SUCCESS: Core module is working!")
else:
    print("FAILED:")
    print(result.stderr)