# Examples and Test Scripts

This directory contains example scripts and test files for HEDit.

## Files

- **test_vision_integration.py** - End-to-end test for vision model integration
  - Tests `/annotate-from-image` endpoint
  - Converts images to base64 and validates response
  - Usage: `python examples/test_vision_integration.py <image_path>`

- **test_image.jpg** - Sample test image (market scene from SCCN dataset)

- **test_examples.py** - Basic examples and sanity checks

- **test_quality_comparison.py** - Quality comparison tests for different models

- **test_ultrafast_cerebras.py** - Performance tests with Cerebras models

## Usage

Run tests from the project root:

```bash
# Test vision integration
python examples/test_vision_integration.py examples/test_image.jpg

# Other tests
python examples/test_examples.py
```

## Note

These are informal test scripts for development and validation. For formal unit tests, see the `tests/` directory.
