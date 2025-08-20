#!/usr/bin/env python3
"""Basic test without optional dependencies."""

import subprocess
import sys

# Test basic imports without extras
test_script = '''
import commons_core
import commons_testing
import commons_cloud
import commons_k8s
import commons_events
import commons_llm
import commons_pipelines
import commons_workflows
import commons_agents
import commons_data

print("✓ All modules imported successfully!")

# Test core functionality
from commons_core import ConfigManager, get_logger

logger = get_logger(__name__)
logger.info("Logger working!")

config = ConfigManager()
print("✓ Core functionality working!")
'''

# Run test in uv environment without extras
result = subprocess.run(
    ["uv", "run", "--no-project", "python", "-c", test_script],
    capture_output=True,
    text=True
)

if result.returncode == 0:
    print(result.stdout)
else:
    print("Error:", result.stderr)