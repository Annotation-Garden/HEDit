#!/bin/bash
# Test HEDit package from TestPyPI
#
# This script creates a clean virtual environment, installs hedit from TestPyPI,
# and runs basic functionality tests to verify the package works correctly.
#
# Usage:
#   ./scripts/test_testpypi_package.sh [version]
#
# Examples:
#   ./scripts/test_testpypi_package.sh           # Install latest
#   ./scripts/test_testpypi_package.sh 0.6.3-dev # Install specific version

set -e

VERSION="${1:-}"
VENV_DIR="/tmp/hedit-testpypi-test"
TEST_API_KEY="${OPENROUTER_API_KEY_FOR_TESTING:-}"

echo "========================================"
echo "HEDit TestPyPI Package Test"
echo "========================================"
echo ""

# Check for test API key
if [ -z "$TEST_API_KEY" ]; then
    echo "WARNING: OPENROUTER_API_KEY_FOR_TESTING not set"
    echo "Some tests will be skipped"
    echo ""
fi

# Clean up previous test environment
if [ -d "$VENV_DIR" ]; then
    echo "Removing previous test environment..."
    rm -rf "$VENV_DIR"
fi

# Create fresh virtual environment with uv
echo "Creating virtual environment with uv..."
uv venv "$VENV_DIR" --python 3.12

# Activate
source "$VENV_DIR/bin/activate"

echo ""
echo "========================================"
echo "1. Installing hedit from TestPyPI"
echo "========================================"

# Install from TestPyPI with PyPI fallback for dependencies
# Note: --index-strategy unsafe-best-match allows finding hedit on TestPyPI
# even though it also exists on PyPI (with different versions)
if [ -n "$VERSION" ]; then
    # Normalize version to PEP 440 format
    # 0.6.3-dev -> 0.6.3.dev0, 0.6.3-alpha -> 0.6.3a0, 0.6.3-beta -> 0.6.3b0
    NORMALIZED_VERSION=$(echo "$VERSION" | sed 's/-dev/.dev0/; s/-alpha/a0/; s/-beta/b0/; s/-rc/rc0/')
    echo "Installing hedit==$NORMALIZED_VERSION (from $VERSION)..."
    uv pip install \
        --index-url https://test.pypi.org/simple/ \
        --extra-index-url https://pypi.org/simple/ \
        --index-strategy unsafe-best-match \
        "hedit==$NORMALIZED_VERSION"
else
    echo "Installing latest hedit from TestPyPI..."
    uv pip install \
        --index-url https://test.pypi.org/simple/ \
        --extra-index-url https://pypi.org/simple/ \
        --index-strategy unsafe-best-match \
        hedit
fi

echo ""
echo "========================================"
echo "2. Basic CLI Tests"
echo "========================================"

echo ""
echo "--- hedit --version ---"
hedit --version

echo ""
echo "--- hedit --help ---"
hedit --help | head -20

echo ""
echo "--- hedit health (API mode) ---"
hedit health || echo "Health check failed (expected if API is down)"

echo ""
echo "========================================"
echo "3. Installing standalone extras"
echo "========================================"

if [ -n "$VERSION" ]; then
    echo "Installing hedit[standalone]==$NORMALIZED_VERSION..."
    uv pip install \
        --index-url https://test.pypi.org/simple/ \
        --extra-index-url https://pypi.org/simple/ \
        --index-strategy unsafe-best-match \
        "hedit[standalone]==$NORMALIZED_VERSION"
else
    echo "Installing hedit[standalone] from TestPyPI..."
    uv pip install \
        --index-url https://test.pypi.org/simple/ \
        --extra-index-url https://pypi.org/simple/ \
        --index-strategy unsafe-best-match \
        "hedit[standalone]"
fi

echo ""
echo "--- hedit health --standalone ---"
if [ -n "$TEST_API_KEY" ]; then
    hedit health --standalone --api-key "$TEST_API_KEY"
else
    echo "SKIPPED: Standalone health requires API key"
    echo "(Standalone mode needs API key for LLM initialization)"
fi

echo ""
echo "========================================"
echo "4. Validation Tests"
echo "========================================"

echo ""
echo "--- Valid HED string (API mode) ---"
hedit validate "Sensory-event, Visual-presentation" || echo "Validation failed"

echo ""
echo "--- Valid HED string (standalone) ---"
# Note: Standalone validation uses Python hedtools, doesn't need API key
hedit validate "Sensory-event, Visual-presentation" --standalone || echo "Validation failed"

echo ""
echo "--- Invalid HED string (standalone) ---"
# Should report as invalid
hedit validate "NotARealTag, InvalidTag" --standalone && echo "WARNING: Expected to fail for invalid tags" || echo "Correctly identified invalid tags"

echo ""
echo "========================================"
echo "5. Annotation Tests (requires API key)"
echo "========================================"

if [ -n "$TEST_API_KEY" ]; then
    echo ""
    echo "--- Annotate (API mode) ---"
    hedit annotate "participant pressed a button" --api-key "$TEST_API_KEY" -o json | head -20

    echo ""
    echo "--- Annotate (standalone mode) ---"
    hedit annotate "red circle appeared on screen" --api-key "$TEST_API_KEY" --standalone -o json | head -20
else
    echo "SKIPPED: No API key provided"
    echo "Set OPENROUTER_API_KEY_FOR_TESTING to run annotation tests"
fi

echo ""
echo "========================================"
echo "6. Package Info"
echo "========================================"

echo ""
echo "--- Installed packages ---"
uv pip list | grep -E "hedit|langgraph|langchain|hed"

echo ""
echo "--- Package location ---"
python -c "import src.cli; print(src.cli.__file__)"

echo ""
echo "========================================"
echo "Test Complete!"
echo "========================================"
echo ""
echo "Virtual environment: $VENV_DIR"
echo "To clean up: rm -rf $VENV_DIR"
echo ""
echo "To activate manually:"
echo "  source $VENV_DIR/bin/activate"
echo ""

# Deactivate
deactivate
