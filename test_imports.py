#!/usr/bin/env python3
"""Simple test to verify modules can be imported."""

modules = [
    "commons_core",
    "commons_testing", 
    "commons_cloud",
    "commons_k8s",
    "commons_events",
    "commons_llm",
    "commons_pipelines",
    "commons_workflows",
    "commons_agents",
    "commons_data"
]

print("Testing module imports...")
for module in modules:
    try:
        __import__(module)
        print(f"✓ {module}")
    except Exception as e:
        print(f"✗ {module}: {e}")