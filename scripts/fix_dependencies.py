#!/usr/bin/env python3
"""Fix workspace dependencies in all modules."""

import os
import re
from pathlib import Path

# Map of modules and their internal dependencies
INTERNAL_DEPS = {
    "cloud": ["commons-core"],
    "k8s": ["commons-core"],
    "testing": ["commons-core"],
    "events": ["commons-core"],
    "pipelines": ["commons-core"],
    "workflows": ["commons-core"],
    "llm": ["commons-core"],
    "agents": ["commons-core", "commons-llm"],
    "data": ["commons-core"],
}

def fix_module_dependencies(module_path: Path, internal_deps: list):
    """Fix dependencies in a module's pyproject.toml."""
    pyproject_path = module_path / "pyproject.toml"
    if not pyproject_path.exists():
        print(f"Skipping {module_path.name}: pyproject.toml not found")
        return
        
    content = pyproject_path.read_text()
    
    # Remove version constraints from internal dependencies
    for dep in internal_deps:
        content = re.sub(rf'"{dep}>=[\d.]+",?', f'"{dep}",', content)
    
    # Add [tool.uv.sources] section if not present
    if "[tool.uv.sources]" not in content:
        sources_section = "\n[tool.uv.sources]\n"
        for dep in internal_deps:
            sources_section += f'{dep} = {{ workspace = true }}\n'
        
        # Insert before [project.optional-dependencies] or [build-system]
        if "[project.optional-dependencies]" in content:
            content = content.replace("[project.optional-dependencies]", 
                                      sources_section + "\n[project.optional-dependencies]")
        elif "[build-system]" in content:
            content = content.replace("[build-system]", 
                                      sources_section + "\n[build-system]")
        else:
            content += sources_section
    
    pyproject_path.write_text(content)
    print(f"Fixed dependencies in {module_path.name}")

def main():
    """Fix all module dependencies."""
    root = Path(__file__).parent.parent
    
    for module, deps in INTERNAL_DEPS.items():
        module_path = root / module
        if module_path.exists():
            fix_module_dependencies(module_path, deps)
    
    print("\nDependencies fixed in all modules")

if __name__ == "__main__":
    main()