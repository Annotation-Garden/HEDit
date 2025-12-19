# Manual Testing: TestPyPI Package

This document describes how to test the HEDit package from TestPyPI before releasing to production PyPI.

## Prerequisites

- `uv` installed (fast Python package manager)
- `OPENROUTER_API_KEY_FOR_TESTING` environment variable (optional, for annotation tests)

## Quick Test

```bash
# Run the automated test script
./scripts/test_testpypi_package.sh

# Or with specific version
./scripts/test_testpypi_package.sh 0.6.3-dev
```

## Manual Test Steps

### 1. Create Clean Environment

```bash
# Create fresh venv
uv venv /tmp/hedit-test --python 3.12
source /tmp/hedit-test/bin/activate
```

### 2. Install from TestPyPI

```bash
# Install base package (API client mode)
uv pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    hedit==0.6.3-dev

# Verify installation
hedit --version
hedit --help
```

### 3. Test API Mode

```bash
# Health check
hedit health

# Validate HED string
hedit validate "Sensory-event, Visual-presentation"

# Annotate (requires API key)
hedit annotate "button press" --api-key $OPENROUTER_API_KEY_FOR_TESTING
```

### 4. Install Standalone Extras

```bash
# Install standalone dependencies (~2GB)
uv pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    "hedit[standalone]==0.6.3-dev"
```

### 5. Test Standalone Mode

```bash
# Health check (shows validator type)
hedit health --standalone

# Validate locally
hedit validate "Sensory-event, Visual-presentation" --standalone

# Annotate locally (requires API key for LLM)
hedit annotate "red circle appeared" --api-key $OPENROUTER_API_KEY_FOR_TESTING --standalone
```

### 6. Verify Validator Selection

```bash
# Check which validator is used
hedit health --standalone -o json | jq '.validator_type'
# Should show "javascript" if Node.js available, otherwise "python"
```

## Expected Results

| Test | Expected Outcome |
|------|------------------|
| `hedit --version` | Shows version (e.g., 0.6.3-dev) |
| `hedit health` | Shows API status |
| `hedit health --standalone` | Shows local dependencies status |
| `hedit validate "Event"` | Valid HED string |
| `hedit validate "InvalidTag"` | Invalid with error message |
| `hedit annotate "..."` | Returns HED annotation |

## Cleanup

```bash
deactivate
rm -rf /tmp/hedit-test
```

## Troubleshooting

### Import Error
If you get import errors, ensure you installed with `--extra-index-url https://pypi.org/simple/` to get dependencies from regular PyPI.

### Standalone Dependencies Missing
If standalone mode fails, ensure you installed with `hedit[standalone]` extras.

### Validator Type is Python
If `validator_type` shows "python" but you want JavaScript:
1. Ensure Node.js is installed
2. Set `HED_VALIDATOR_PATH` environment variable to hed-javascript location
