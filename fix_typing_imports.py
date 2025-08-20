#!/usr/bin/env python3
"""Fix missing typing imports across all modules."""

import os
import re
from pathlib import Path

# Common typing imports that might be missing
TYPING_IMPORTS = [
    'Any', 'Dict', 'List', 'Optional', 'Union', 'Tuple', 'Type', 'Callable',
    'Iterator', 'Iterable', 'Mapping', 'Sequence', 'Set', 'FrozenSet',
    'TypeVar', 'Generic', 'Protocol', 'ClassVar', 'Final', 'Literal'
]

def find_python_files(directory):
    """Find all Python files in directory."""
    python_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files

def analyze_file(filepath):
    """Analyze a Python file for missing typing imports."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find existing typing imports
    existing_imports = set()
    typing_import_match = re.search(r'from typing import (.+)', content)
    if typing_import_match:
        imports_str = typing_import_match.group(1)
        # Handle multi-line imports and clean up
        imports_str = re.sub(r'\s+', ' ', imports_str)
        existing_imports.update(imp.strip() for imp in imports_str.split(','))
    
    # Find used typing annotations
    used_types = set()
    for typing_type in TYPING_IMPORTS:
        if re.search(rf'\b{typing_type}\b', content) and typing_type not in existing_imports:
            used_types.add(typing_type)
    
    return existing_imports, used_types

def fix_file(filepath, existing_imports, missing_imports):
    """Fix missing typing imports in a file."""
    if not missing_imports:
        return False
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find where to insert/update typing import
    typing_import_line = -1
    for i, line in enumerate(lines):
        if line.startswith('from typing import'):
            typing_import_line = i
            break
    
    all_imports = sorted(existing_imports | missing_imports)
    new_import_line = f"from typing import {', '.join(all_imports)}\n"
    
    if typing_import_line >= 0:
        # Replace existing import
        lines[typing_import_line] = new_import_line
    else:
        # Insert new import after other imports
        insert_pos = 0
        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                insert_pos = i + 1
            elif line.strip() == '':
                continue
            else:
                break
        lines.insert(insert_pos, new_import_line)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    return True

def main():
    """Fix typing imports in all Python files."""
    base_dir = '/Users/nbalawat/development/python-common-modules'
    
    # Find all Python files in src directories
    python_files = []
    for module in ['core', 'testing', 'cloud', 'k8s', 'events', 'llm', 'pipelines', 'workflows', 'agents', 'data']:
        src_dir = os.path.join(base_dir, module, 'src')
        if os.path.exists(src_dir):
            python_files.extend(find_python_files(src_dir))
    
    print(f"Found {len(python_files)} Python files")
    
    fixed_files = 0
    for filepath in python_files:
        try:
            existing_imports, missing_imports = analyze_file(filepath)
            if missing_imports:
                print(f"Fixing {filepath}: adding {', '.join(sorted(missing_imports))}")
                if fix_file(filepath, existing_imports, missing_imports):
                    fixed_files += 1
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
    
    print(f"Fixed {fixed_files} files")

if __name__ == "__main__":
    main()