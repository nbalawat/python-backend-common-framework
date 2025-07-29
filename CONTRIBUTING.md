# Contributing to Python Commons

Thank you for your interest in contributing to Python Commons! This document provides guidelines and instructions for contributing.

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md) to ensure a welcoming environment for all contributors.

## Getting Started

### Prerequisites

- Python 3.11 or higher
- uv package manager
- Git

### Development Setup

1. Fork the repository on GitHub
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/python-commons.git
   cd python-commons
   ```

3. Install uv:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

4. Set up development environment:
   ```bash
   make setup-dev
   ```

5. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

### Code Style

We use:
- **Black** for code formatting
- **Ruff** for linting
- **mypy** for type checking

Run formatting and linting:
```bash
make format
make lint
make type-check
```

### Testing

Write tests for all new functionality:

```bash
# Run all tests
make test

# Run tests for specific module
make test-module MODULE=core

# Run tests with coverage
make test-cov
```

### Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Test additions or modifications
- `chore:` Maintenance tasks

Examples:
```
feat(core): add retry mechanism to config loader
fix(cloud): handle S3 multipart upload failures
docs(k8s): update examples for operator pattern
```

### Pull Request Process

1. Ensure all tests pass
2. Update documentation as needed
3. Add entries to CHANGELOG.md
4. Submit PR against `develop` branch
5. Wait for code review

### Module Structure

When adding new functionality to a module:

```
module/
├── src/
│   └── commons_module/
│       ├── __init__.py       # Public API exports
│       ├── feature.py        # Implementation
│       └── _internal.py      # Internal utilities
├── tests/
│   ├── test_feature.py       # Unit tests
│   └── integration/          # Integration tests
├── README.md                 # Module documentation
└── pyproject.toml           # Module configuration
```

## Adding a New Module

1. Create module structure:
   ```bash
   mkdir -p new_module/{src/commons_new_module,tests}
   ```

2. Create `pyproject.toml` with proper dependencies
3. Add module to workspace in root `pyproject.toml`
4. Update Makefile to include new module
5. Add comprehensive README.md
6. Write thorough tests

## Documentation

- Use Google-style docstrings
- Include type hints for all functions
- Provide usage examples in docstrings
- Update module README with new features

Example:
```python
def process_data(
    data: List[Dict[str, Any]],
    validate: bool = True,
    timeout: Optional[float] = None,
) -> ProcessResult:
    """Process data with optional validation.
    
    Args:
        data: List of data items to process
        validate: Whether to validate data before processing
        timeout: Optional timeout in seconds
        
    Returns:
        ProcessResult containing processed data and metadata
        
    Raises:
        ValidationError: If validation fails
        TimeoutError: If processing exceeds timeout
        
    Example:
        >>> data = [{"id": 1, "value": "test"}]
        >>> result = process_data(data, validate=True)
        >>> print(result.success_count)
        1
    """
```

## Release Process

1. Update version in module's `pyproject.toml`
2. Update CHANGELOG.md
3. Create PR to `main` branch
4. After merge, tag release:
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```

## Getting Help

- Open an issue for bugs or feature requests
- Join discussions for questions
- Check existing issues before creating new ones

## License

By contributing, you agree that your contributions will be licensed under the MIT License.