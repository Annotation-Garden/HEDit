# Manual Testing: TestPyPI Package

This document describes how to test the HEDit package from TestPyPI before releasing to production PyPI.

## Prerequisites

- `uv` installed (fast Python package manager)
- `HEDIT_ANTHROPIC_API_KEY` environment variable with an Anthropic API key (`sk-ant-...`; optional, for bring your own key (BYOK) annotation tests)

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

# Annotate (no key needed in API mode; add a BYOK key to bill your own account)
hedit annotate "button press" --api-key $HEDIT_ANTHROPIC_API_KEY
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

# Annotate locally (requires an Anthropic key for the LLM; also works with
# the server credentials ANTHROPIC_API_KEY/ANTHROPIC_BASE_URL/ANTHROPIC_WORKSPACE_ID)
hedit annotate "red circle appeared" --api-key $HEDIT_ANTHROPIC_API_KEY --standalone
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
