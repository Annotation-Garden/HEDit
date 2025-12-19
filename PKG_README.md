# HEDit

[![PyPI version](https://badge.fury.io/py/hedit.svg)](https://pypi.org/project/hedit/)
[![Tests](https://github.com/Annotation-Garden/hedit/actions/workflows/test.yml/badge.svg)](https://github.com/Annotation-Garden/hedit/actions/workflows/test.yml)

Convert natural language event descriptions into valid [HED](https://hedtags.org) (Hierarchical Event Descriptors) annotations.

Part of the [Annotation Garden Initiative](https://annotation.garden).

## Installation

```bash
# Default (lightweight API client, ~100MB)
pip install hedit

# Standalone mode (run locally without backend, ~2GB)
pip install hedit[standalone]
```

## Quick Start

```bash
# Configure your OpenRouter API key (https://openrouter.ai)
hedit init --api-key sk-or-v1-xxx

# Generate HED annotation from text
hedit annotate "participant pressed the left button"

# Generate HED from an image
hedit annotate-image stimulus.png

# Validate a HED string
hedit validate "Sensory-event, Visual-presentation"
```

## Commands

| Command | Description |
|---------|-------------|
| `hedit init` | Configure API key and preferences |
| `hedit annotate "text"` | Convert natural language to HED |
| `hedit annotate-image <file>` | Generate HED from image |
| `hedit validate "HED-string"` | Validate HED annotation |
| `hedit health` | Check service status |
| `hedit config show` | Display configuration |

## Options

```bash
hedit annotate "text" -o json          # JSON output for scripting
hedit annotate "text" --schema 8.3.0   # Specific HED schema version
hedit annotate "text" --standalone     # Run locally (requires hedit[standalone])
```

## How It Works

HEDit uses a multi-agent system (LangGraph) with feedback loops:

1. **Annotation Agent** - Generates initial HED tags
2. **Validation Agent** - Checks syntax and tag validity
3. **Evaluation Agent** - Assesses faithfulness to input
4. **Assessment Agent** - Identifies missing elements

Annotations are automatically refined until validation passes.

## Links

- [Documentation](https://docs.annotation.garden/hedit)
- [GitHub Repository](https://github.com/Annotation-Garden/HEDit)
- [HED Standard](https://hedtags.org)
- [OpenRouter](https://openrouter.ai) - Get an API key

## License

MIT
