"""Tests for CLI main commands."""

import os
from unittest.mock import patch

# Set NO_COLOR before importing app to disable Rich formatting
os.environ["NO_COLOR"] = "1"
os.environ["TERM"] = "dumb"
os.environ["COLUMNS"] = "200"  # Wide terminal to prevent truncation

from typer.testing import CliRunner

from src.cli.main import app

runner = CliRunner()


class TestVersion:
    """Tests for version command."""

    def test_version_flag(self):
        """Test --version flag shows version."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "hedit version" in result.output

    def test_version_short_flag(self):
        """Test -V flag shows version."""
        result = runner.invoke(app, ["-V"])
        assert result.exit_code == 0
        assert "hedit version" in result.output


class TestHelp:
    """Tests for help output."""

    def test_main_help(self):
        """Test main help shows all commands."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "annotate" in result.output
        assert "annotate-image" in result.output
        assert "validate" in result.output
        assert "config" in result.output
        assert "init" in result.output
        assert "health" in result.output

    def test_annotate_help(self):
        """Test annotate command help."""
        result = runner.invoke(app, ["annotate", "--help"])
        assert result.exit_code == 0
        assert "--api-key" in result.output
        assert "--schema" in result.output
        assert "--output" in result.output

    def test_config_help(self):
        """Test config subcommand help."""
        result = runner.invoke(app, ["config", "--help"])
        assert result.exit_code == 0
        assert "show" in result.output
        assert "set" in result.output
        assert "path" in result.output


class TestAnnotateCommand:
    """Tests for annotate command."""

    def test_annotate_no_api_key(self, tmp_path):
        """Test annotate fails without API key."""
        # Clear any env vars that might leak from other tests
        env_backup = os.environ.pop("OPENROUTER_API_KEY", None)
        try:
            with (
                patch("src.cli.config.CONFIG_DIR", tmp_path),
                patch("src.cli.config.CONFIG_FILE", tmp_path / "config.yaml"),
                patch("src.cli.config.CREDENTIALS_FILE", tmp_path / "credentials.yaml"),
            ):
                result = runner.invoke(app, ["annotate", "test description"])
                assert result.exit_code == 1
                assert "No API key" in result.output or "api key" in result.output.lower()
        finally:
            if env_backup:
                os.environ["OPENROUTER_API_KEY"] = env_backup

    def test_annotate_with_api_key(self, tmp_path):
        """Test annotate with API key makes request."""
        mock_response = {
            "annotation": "Test-annotation",
            "is_valid": True,
            "is_faithful": True,
            "is_complete": False,
            "validation_attempts": 1,
            "validation_errors": [],
            "validation_warnings": [],
            "status": "success",
        }

        with (
            patch("src.cli.config.CONFIG_DIR", tmp_path),
            patch("src.cli.config.CONFIG_FILE", tmp_path / "config.yaml"),
            patch("src.cli.config.CREDENTIALS_FILE", tmp_path / "credentials.yaml"),
            patch("src.cli.client.HEDitClient.annotate", return_value=mock_response),
        ):
            result = runner.invoke(
                app,
                ["annotate", "test description", "--api-key", "test-key"],
            )
            assert result.exit_code == 0
            assert "Test-annotation" in result.output

    def test_annotate_json_output(self, tmp_path):
        """Test annotate with JSON output."""
        mock_response = {
            "annotation": "Test-annotation",
            "is_valid": True,
            "is_faithful": True,
            "is_complete": False,
            "validation_attempts": 1,
            "validation_errors": [],
            "validation_warnings": [],
            "status": "success",
        }

        with (
            patch("src.cli.config.CONFIG_DIR", tmp_path),
            patch("src.cli.config.CONFIG_FILE", tmp_path / "config.yaml"),
            patch("src.cli.config.CREDENTIALS_FILE", tmp_path / "credentials.yaml"),
            patch("src.cli.client.HEDitClient.annotate", return_value=mock_response),
        ):
            result = runner.invoke(
                app,
                ["annotate", "test description", "--api-key", "test-key", "-o", "json"],
            )
            assert result.exit_code == 0
            assert '"annotation"' in result.output
            assert '"is_valid"' in result.output


class TestValidateCommand:
    """Tests for validate command."""

    def test_validate_with_api_key(self, tmp_path):
        """Test validate command."""
        mock_response = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "parsed_string": "Event",
        }

        with (
            patch("src.cli.config.CONFIG_DIR", tmp_path),
            patch("src.cli.config.CONFIG_FILE", tmp_path / "config.yaml"),
            patch("src.cli.config.CREDENTIALS_FILE", tmp_path / "credentials.yaml"),
            patch("src.cli.client.HEDitClient.validate", return_value=mock_response),
        ):
            result = runner.invoke(
                app,
                ["validate", "Event", "--api-key", "test-key"],
            )
            assert result.exit_code == 0
            assert "Valid" in result.output

    def test_validate_invalid_hed(self, tmp_path):
        """Test validate with invalid HED string."""
        mock_response = {
            "is_valid": False,
            "errors": ["Invalid tag: NotATag"],
            "warnings": [],
        }

        with (
            patch("src.cli.config.CONFIG_DIR", tmp_path),
            patch("src.cli.config.CONFIG_FILE", tmp_path / "config.yaml"),
            patch("src.cli.config.CREDENTIALS_FILE", tmp_path / "credentials.yaml"),
            patch("src.cli.client.HEDitClient.validate", return_value=mock_response),
        ):
            result = runner.invoke(
                app,
                ["validate", "NotATag", "--api-key", "test-key"],
            )
            assert result.exit_code == 1
            assert "Invalid" in result.output or "Error" in result.output


class TestHealthCommand:
    """Tests for health command."""

    def test_health_success(self, tmp_path):
        """Test health check success."""
        mock_response = {
            "status": "healthy",
            "version": "0.6.1-dev",
            "llm_available": True,
            "validator_available": True,
        }

        with (
            patch("src.cli.config.CONFIG_DIR", tmp_path),
            patch("src.cli.config.CONFIG_FILE", tmp_path / "config.yaml"),
            patch("src.cli.config.CREDENTIALS_FILE", tmp_path / "credentials.yaml"),
            patch("src.cli.client.HEDitClient.health", return_value=mock_response),
        ):
            result = runner.invoke(app, ["health"])
            assert result.exit_code == 0
            assert "healthy" in result.output

    def test_health_custom_url(self, tmp_path):
        """Test health with custom API URL."""
        mock_response = {
            "status": "healthy",
            "version": "0.6.1-dev",
            "llm_available": True,
            "validator_available": True,
        }

        with (
            patch("src.cli.config.CONFIG_DIR", tmp_path),
            patch("src.cli.config.CONFIG_FILE", tmp_path / "config.yaml"),
            patch("src.cli.config.CREDENTIALS_FILE", tmp_path / "credentials.yaml"),
            patch("src.cli.client.HEDitClient.health", return_value=mock_response),
        ):
            result = runner.invoke(app, ["health", "--api-url", "https://custom.api.com"])
            assert result.exit_code == 0


class TestConfigCommands:
    """Tests for config subcommands."""

    def test_config_show(self, tmp_path):
        """Test config show command."""
        with (
            patch("src.cli.config.CONFIG_DIR", tmp_path),
            patch("src.cli.config.CONFIG_FILE", tmp_path / "config.yaml"),
            patch("src.cli.config.CREDENTIALS_FILE", tmp_path / "credentials.yaml"),
        ):
            result = runner.invoke(app, ["config", "show"])
            assert result.exit_code == 0
            assert "Configuration" in result.output

    def test_config_path(self, tmp_path):
        """Test config path command."""
        with (
            patch("src.cli.config.CONFIG_DIR", tmp_path),
            patch("src.cli.config.CONFIG_FILE", tmp_path / "config.yaml"),
            patch("src.cli.config.CREDENTIALS_FILE", tmp_path / "credentials.yaml"),
        ):
            result = runner.invoke(app, ["config", "path"])
            assert result.exit_code == 0
            assert "config" in result.output.lower()

    def test_config_set(self, tmp_path):
        """Test config set command."""
        with (
            patch("src.cli.config.CONFIG_DIR", tmp_path),
            patch("src.cli.config.CONFIG_FILE", tmp_path / "config.yaml"),
            patch("src.cli.config.CREDENTIALS_FILE", tmp_path / "credentials.yaml"),
        ):
            # Ensure config dir exists
            tmp_path.mkdir(exist_ok=True)

            result = runner.invoke(app, ["config", "set", "models.default", "test-model"])
            assert result.exit_code == 0
            assert "Set" in result.output or "Success" in result.output


class TestInitCommand:
    """Tests for init command."""

    def test_init_saves_config(self, tmp_path):
        """Test init command saves configuration."""
        with (
            patch("src.cli.config.CONFIG_DIR", tmp_path),
            patch("src.cli.config.CONFIG_FILE", tmp_path / "config.yaml"),
            patch("src.cli.config.CREDENTIALS_FILE", tmp_path / "credentials.yaml"),
            patch(
                "src.cli.client.HEDitClient.health",
                return_value={"status": "healthy"},
            ),
        ):
            # Ensure directory exists
            tmp_path.mkdir(exist_ok=True)

            result = runner.invoke(app, ["init", "--api-key", "test-key-12345"])
            assert result.exit_code == 0
            assert "Success" in result.output or "saved" in result.output.lower()

    def test_init_with_custom_settings(self, tmp_path):
        """Test init with custom model and provider."""
        with (
            patch("src.cli.config.CONFIG_DIR", tmp_path),
            patch("src.cli.config.CONFIG_FILE", tmp_path / "config.yaml"),
            patch("src.cli.config.CREDENTIALS_FILE", tmp_path / "credentials.yaml"),
            patch(
                "src.cli.client.HEDitClient.health",
                return_value={"status": "healthy"},
            ),
        ):
            tmp_path.mkdir(exist_ok=True)

            result = runner.invoke(
                app,
                [
                    "init",
                    "--api-key",
                    "test-key",
                    "--model",
                    "gpt-4o",
                    "--provider",
                    "OpenAI",
                ],
            )
            assert result.exit_code == 0
