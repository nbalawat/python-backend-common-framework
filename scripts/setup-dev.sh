#!/bin/bash
set -e

echo "Setting up Python Commons development environment..."

# Check Python version
python_version=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.11"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python $required_version or higher is required (found $python_version)"
    exit 1
fi

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Create virtual environment
echo "Creating virtual environment..."
uv venv

# Install dependencies
echo "Installing dependencies..."
uv sync --all-extras

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
uv run pre-commit install

# Create necessary directories
mkdir -p docs/{api,examples}
mkdir -p scripts

# Set up git hooks
echo "Setting up git hooks..."
cat > .git/hooks/pre-push << 'EOF'
#!/bin/bash
# Run tests before push
make test
EOF
chmod +x .git/hooks/pre-push

echo "Development environment setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "Common commands:"
echo "  make test          - Run all tests"
echo "  make lint          - Run linting"
echo "  make format        - Format code"
echo "  make test-module MODULE=core  - Test specific module"