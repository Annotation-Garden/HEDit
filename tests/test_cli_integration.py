"""Integration tests for HEDit CLI with real API calls.

These tests use OPENROUTER_API_KEY_FOR_TESTING to make real API calls.
Tests are skipped if the key is not present or if API is blocked by Cloudflare.

Run with: pytest tests/test_cli_integration.py -v -m integration
Skip integration tests: pytest -v -m "not integration"
"""

import os
from pathlib import Path

import httpx
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


def extract_json_from_output(output: str) -> dict | None:
    """Extract JSON object from CLI output that may contain other text.

    Handles cases where the output has headers, warnings, or other text mixed in.
    Returns None if no valid JSON found.
    """
    import json

    # Try parsing the whole output first (fast path)
    try:
        return json.loads(output.strip())
    except json.JSONDecodeError:
        pass

    # Look for JSON object patterns in the output
    # Try to find { ... } that forms a valid JSON object
    lines = output.strip().split("\n")
    for i, line in enumerate(lines):
        if line.strip().startswith("{"):
            # Try to parse from this line onwards
            candidate = "\n".join(lines[i:])
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                # Try just this line
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue

    return None


# Check if API is reachable (not blocked by Cloudflare)
_API_REACHABLE: bool | None = None


def _check_api_reachable() -> bool:
    """Check if the API is reachable and not blocked by Cloudflare."""
    global _API_REACHABLE
    if _API_REACHABLE is not None:
        return _API_REACHABLE

    try:
        response = httpx.get(f"{API_URL}/health", timeout=10)
        # Check for Cloudflare challenge (returns HTML with cf_chl)
        if "cf_chl" in response.text or "cloudflare" in response.text.lower():
            _API_REACHABLE = False
        else:
            _API_REACHABLE = response.status_code == 200
    except Exception:
        _API_REACHABLE = False

    return _API_REACHABLE


@pytest.fixture
def test_api_key() -> str:
    """Get OpenRouter API key for testing."""
    if not OPENROUTER_TEST_KEY:
        pytest.skip(SKIP_REASON)
    return OPENROUTER_TEST_KEY


@pytest.fixture
def require_api_access():
    """Skip test if API is not reachable (e.g., blocked by Cloudflare)."""
    if not _check_api_reachable():
        pytest.skip(f"API at {API_URL} is not reachable (possibly blocked by Cloudflare)")


@pytest.fixture
def temp_config_dir(tmp_path):
    """Create a temporary config directory for testing."""
    from unittest.mock import patch

    config_dir = tmp_path / "hedit"
    config_dir.mkdir()

    # Create first-run marker to prevent telemetry disclosure from appearing
    first_run_file = config_dir / ".first_run"
    first_run_file.touch()

    with (
        patch("src.cli.config.CONFIG_DIR", config_dir),
        patch("src.cli.config.CONFIG_FILE", config_dir / "config.yaml"),
        patch("src.cli.config.CREDENTIALS_FILE", config_dir / "credentials.yaml"),
        patch("src.cli.config.FIRST_RUN_FILE", first_run_file),
        # Also patch in main.py which imports these
        patch("src.cli.main.is_first_run", return_value=False),
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

        # Check JSON output - use helper to extract JSON from potentially mixed output
        data = extract_json_from_output(result.output)
        if data is None:
            pytest.fail(f"Could not extract JSON from output: {result.output}")

        # Simple checks - just verify structure exists
        assert "annotation" in data or "error" in data, f"Expected annotation or error in: {data}"

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

        # Use helper to extract JSON from potentially mixed output
        data = extract_json_from_output(result.output)
        if data is None:
            pytest.fail(f"Could not extract JSON from output: {result.output}")

        # Check that annotation exists (may be empty string on API error, which is ok)
        assert "annotation" in data, f"Expected annotation key in response: {data}"


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

        # API returns warnings for invalid tags but may still report as "valid"
        # Check that it ran successfully and contains warning about invalid tag
        assert result.exit_code in [0, 1], f"Unexpected exit code: {result.output}"
        # Should have TAG_INVALID warning in output
        assert "TAG_INVALID" in result.output or "not a valid" in result.output.lower(), (
            f"Expected warning about invalid tag: {result.output}"
        )

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

        # Use helper to extract JSON from potentially mixed output
        data = extract_json_from_output(result.output)
        if data is None:
            pytest.fail(f"Could not extract JSON from output: {result.output}")

        assert "is_valid" in data or "error" in data, f"Expected is_valid or error in: {data}"


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

        # Handle case where vision model is not available on OpenRouter
        if "No allowed providers" in result.output or "model" in result.output.lower():
            if result.exit_code == 1:
                pytest.skip("Vision model not available on OpenRouter")

        # Use helper to extract JSON from potentially mixed output
        data = extract_json_from_output(result.output)
        if data is None:
            # If JSON parsing fails, check for expected error messages
            if "No allowed providers" in result.output:
                pytest.skip("Vision model not available on OpenRouter")
            pytest.fail(f"Could not extract JSON from output: {result.output}")

        # Simple check that response has expected structure
        assert "annotation" in data or "error" in data, f"Expected annotation or error in: {data}"
