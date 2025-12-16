"""Integration tests for HEDit CLI with real API calls.

These tests use OPENROUTER_API_KEY_FOR_TESTING to make real API calls.
Tests are skipped if the key is not present.

Run with: pytest tests/test_cli_integration.py -v -m integration
Skip integration tests: pytest -v -m "not integration"
"""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from typer.testing import CliRunner

# Load environment variables from .env file
load_dotenv()

# Check if OpenRouter testing key is available
OPENROUTER_TEST_KEY = os.getenv("OPENROUTER_API_KEY_FOR_TESTING")
SKIP_REASON = "OPENROUTER_API_KEY_FOR_TESTING not set"

# API endpoint - use production or local
API_URL = os.getenv("HEDIT_TEST_API_URL", "https://api.annotation.garden/hedit")

# Set NO_COLOR for clean test output
os.environ["NO_COLOR"] = "1"
os.environ["TERM"] = "dumb"
os.environ["COLUMNS"] = "200"

from src.cli.main import app  # noqa: E402

runner = CliRunner()


@pytest.fixture
def test_api_key() -> str:
    """Get OpenRouter API key for testing."""
    if not OPENROUTER_TEST_KEY:
        pytest.skip(SKIP_REASON)
    return OPENROUTER_TEST_KEY


@pytest.fixture
def temp_config_dir(tmp_path):
    """Create a temporary config directory for testing."""
    from unittest.mock import patch

    config_dir = tmp_path / "hedit"
    config_dir.mkdir()

    with (
        patch("src.cli.config.CONFIG_DIR", config_dir),
        patch("src.cli.config.CONFIG_FILE", config_dir / "config.yaml"),
        patch("src.cli.config.CREDENTIALS_FILE", config_dir / "credentials.yaml"),
    ):
        yield config_dir


@pytest.mark.integration
@pytest.mark.timeout(120)
class TestCLIAnnotateIntegration:
    """Integration tests for annotate command with real API calls."""

    def test_annotate_simple_description(self, test_api_key, temp_config_dir):
        """Test annotating a simple description."""
        result = runner.invoke(
            app,
            [
                "annotate",
                "A red circle appears on the screen",
                "--api-key",
                test_api_key,
                "--api-url",
                API_URL,
            ],
        )

        # Check command completed (exit code 0 = success, 1 = annotation failed but ran)
        assert result.exit_code in [0, 1], (
            f"Unexpected exit code: {result.exit_code}\n{result.output}"
        )

        # Check output contains annotation result
        assert "HED Annotation" in result.output or "annotation" in result.output.lower()

    def test_annotate_with_json_output(self, test_api_key, temp_config_dir):
        """Test annotating with JSON output format."""
        result = runner.invoke(
            app,
            [
                "annotate",
                "Participant pressed the left button",
                "--api-key",
                test_api_key,
                "--api-url",
                API_URL,
                "-o",
                "json",
            ],
        )

        assert result.exit_code in [0, 1], (
            f"Unexpected exit code: {result.exit_code}\n{result.output}"
        )

        # Check JSON output
        import json

        try:
            data = json.loads(result.output)
            assert "annotation" in data
            assert "is_valid" in data
            assert "status" in data
        except json.JSONDecodeError:
            pytest.fail(f"Invalid JSON output: {result.output}")

    def test_annotate_complex_description(self, test_api_key, temp_config_dir):
        """Test annotating a more complex description."""
        result = runner.invoke(
            app,
            [
                "annotate",
                "A high-pitched beep sound plays while a green arrow points to the left side of the screen",
                "--api-key",
                test_api_key,
                "--api-url",
                API_URL,
                "-o",
                "json",
            ],
        )

        assert result.exit_code in [0, 1]

        import json

        data = json.loads(result.output)
        annotation = data.get("annotation", "")
        # Check that some relevant HED tags are present
        assert annotation, "Annotation should not be empty"


@pytest.mark.integration
@pytest.mark.timeout(60)
class TestCLIValidateIntegration:
    """Integration tests for validate command with real API calls."""

    def test_validate_valid_hed_string(self, test_api_key, temp_config_dir):
        """Test validating a valid HED string."""
        result = runner.invoke(
            app,
            [
                "validate",
                "Sensory-event, Visual-presentation",
                "--api-key",
                test_api_key,
                "--api-url",
                API_URL,
            ],
        )

        # Valid HED string should pass
        assert result.exit_code == 0, f"Expected valid HED: {result.output}"
        assert "Valid" in result.output

    def test_validate_invalid_hed_string(self, test_api_key, temp_config_dir):
        """Test validating an invalid HED string."""
        result = runner.invoke(
            app,
            [
                "validate",
                "NotARealHEDTag",
                "--api-key",
                test_api_key,
                "--api-url",
                API_URL,
            ],
        )

        # Invalid HED string should fail
        assert result.exit_code == 1, f"Expected invalid HED: {result.output}"

    def test_validate_json_output(self, test_api_key, temp_config_dir):
        """Test validate with JSON output."""
        result = runner.invoke(
            app,
            [
                "validate",
                "Event",
                "--api-key",
                test_api_key,
                "--api-url",
                API_URL,
                "-o",
                "json",
            ],
        )

        import json

        data = json.loads(result.output)
        assert "is_valid" in data


@pytest.mark.integration
@pytest.mark.timeout(30)
class TestCLIHealthIntegration:
    """Integration tests for health command."""

    def test_health_check(self, temp_config_dir):
        """Test health check endpoint."""
        result = runner.invoke(
            app,
            [
                "health",
                "--api-url",
                API_URL,
            ],
        )

        # Health check should work without API key
        assert result.exit_code == 0, f"Health check failed: {result.output}"
        assert "healthy" in result.output.lower() or "status" in result.output.lower()


@pytest.mark.integration
@pytest.mark.timeout(30)
class TestCLIInitIntegration:
    """Integration tests for init command."""

    def test_init_saves_and_uses_config(self, test_api_key, temp_config_dir):
        """Test that init saves config and subsequent commands use it."""
        # First, initialize with API key
        init_result = runner.invoke(
            app,
            [
                "init",
                "--api-key",
                test_api_key,
                "--api-url",
                API_URL,
            ],
        )

        assert init_result.exit_code == 0, f"Init failed: {init_result.output}"
        assert "saved" in init_result.output.lower() or "success" in init_result.output.lower()

        # Verify config was saved
        config_file = temp_config_dir / "config.yaml"
        creds_file = temp_config_dir / "credentials.yaml"
        assert config_file.exists(), "Config file not created"
        assert creds_file.exists(), "Credentials file not created"


@pytest.mark.integration
@pytest.mark.timeout(180)
class TestCLIImageAnnotateIntegration:
    """Integration tests for annotate-image command."""

    @pytest.fixture
    def test_image(self, tmp_path) -> Path:
        """Create a simple test image (red circle on white background)."""
        try:
            from PIL import Image, ImageDraw

            # Create a simple image with a red circle
            img = Image.new("RGB", (100, 100), "white")
            draw = ImageDraw.Draw(img)
            draw.ellipse([20, 20, 80, 80], fill="red", outline="red")

            image_path = tmp_path / "test_circle.png"
            img.save(image_path)
            return image_path
        except ImportError:
            pytest.skip("PIL not available for image tests")
            return None  # Never reached, but satisfies type checker

    def test_annotate_image(self, test_api_key, temp_config_dir, test_image):
        """Test annotating an image."""
        result = runner.invoke(
            app,
            [
                "annotate-image",
                str(test_image),
                "--api-key",
                test_api_key,
                "--api-url",
                API_URL,
                "-o",
                "json",
            ],
        )

        assert result.exit_code in [0, 1], (
            f"Unexpected exit code: {result.exit_code}\n{result.output}"
        )

        import json

        data = json.loads(result.output)
        assert "annotation" in data
        assert "image_description" in data
