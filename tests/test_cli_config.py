"""Tests for CLI configuration management."""

import os
from unittest.mock import patch

import pytest

from src.cli.config import (
    CLIConfig,
    CredentialsConfig,
    clear_credentials,
    get_api_key,
    get_effective_config,
    get_machine_id,
    is_first_run,
    load_config,
    load_credentials,
    mark_first_run_complete,
    save_config,
    save_credentials,
    update_config,
)


@pytest.fixture
def temp_config_dir(tmp_path):
    """Create a temporary config directory."""
    config_dir = tmp_path / "hedit"
    config_dir.mkdir()

    # Patch the config paths
    with (
        patch("src.cli.config.CONFIG_DIR", config_dir),
        patch("src.cli.config.CONFIG_FILE", config_dir / "config.yaml"),
        patch("src.cli.config.CREDENTIALS_FILE", config_dir / "credentials.yaml"),
        patch("src.cli.config.MACHINE_ID_FILE", config_dir / "machine_id"),
        patch("src.cli.config.FIRST_RUN_FILE", config_dir / ".first_run"),
    ):
        yield config_dir


class TestCredentialsConfig:
    """Tests for credentials configuration."""

    def test_default_credentials(self):
        """Test default credentials are None."""
        creds = CredentialsConfig()
        assert creds.openrouter_api_key is None

    def test_credentials_with_key(self):
        """Test credentials with API key."""
        creds = CredentialsConfig(openrouter_api_key="sk-or-test-key")
        assert creds.openrouter_api_key == "sk-or-test-key"


class TestCLIConfig:
    """Tests for CLI configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CLIConfig()
        assert config.api.url == "https://api.annotation.garden/hedit"
        assert config.models.default == "mistralai/mistral-small-3.2-24b-instruct"
        assert config.models.provider == "mistral"
        assert config.models.temperature == 0.1
        assert config.settings.schema_version == "8.4.0"
        assert config.output.format == "text"

    def test_custom_config(self):
        """Test custom configuration."""
        from src.cli.config import APIConfig, ModelsConfig, OutputConfig, SettingsConfig

        config = CLIConfig(
            api=APIConfig(url="https://custom.api.com"),
            models=ModelsConfig(default="gpt-4o", provider="OpenAI", temperature=0.5),
            settings=SettingsConfig(schema_version="8.3.0"),
            output=OutputConfig(format="json"),
        )
        assert config.api.url == "https://custom.api.com"
        assert config.models.default == "gpt-4o"
        assert config.models.temperature == 0.5
        assert config.output.format == "json"


class TestConfigPersistence:
    """Tests for config file read/write."""

    def test_save_and_load_credentials(self, temp_config_dir):
        """Test saving and loading credentials."""
        creds = CredentialsConfig(openrouter_api_key="test-key-12345")
        save_credentials(creds)

        # Clear env var to test file loading (env vars take precedence)
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENROUTER_API_KEY", None)
            loaded = load_credentials()
            assert loaded.openrouter_api_key == "test-key-12345"

    def test_save_and_load_config(self, temp_config_dir):
        """Test saving and loading config."""
        config = CLIConfig()
        config.models.default = "test-model"
        config.models.temperature = 0.7
        save_config(config)

        loaded = load_config()
        assert loaded.models.default == "test-model"
        assert loaded.models.temperature == 0.7

    def test_load_missing_config(self, temp_config_dir):
        """Test loading config when file doesn't exist."""
        config = load_config()
        assert isinstance(config, CLIConfig)
        # Should return defaults
        assert config.api.url == "https://api.annotation.garden/hedit"

    def test_clear_credentials(self, temp_config_dir):
        """Test clearing credentials."""
        creds = CredentialsConfig(openrouter_api_key="test-key")
        save_credentials(creds)

        clear_credentials()

        # Should return empty credentials (clear env var for test)
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENROUTER_API_KEY", None)
            loaded = load_credentials()
            assert loaded.openrouter_api_key is None


class TestAPIKeyResolution:
    """Tests for API key resolution priority."""

    def test_override_takes_precedence(self, temp_config_dir):
        """Test that explicit override takes precedence."""
        # Save a key to file
        creds = CredentialsConfig(openrouter_api_key="stored-key")
        save_credentials(creds)

        # Override should win
        key = get_api_key(override="override-key")
        assert key == "override-key"

    def test_env_var_takes_precedence_over_file(self, temp_config_dir):
        """Test that env var takes precedence over stored file."""
        # Save a key to file
        creds = CredentialsConfig(openrouter_api_key="stored-key")
        save_credentials(creds)

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "env-key"}):
            key = get_api_key()
            assert key == "env-key"

    def test_stored_key_used_when_no_override_or_env(self, temp_config_dir):
        """Test that stored key is used as fallback."""
        creds = CredentialsConfig(openrouter_api_key="stored-key")
        save_credentials(creds)

        # Remove env var if present
        with patch.dict(os.environ, {}, clear=True):
            # Clear the specific key if it exists
            os.environ.pop("OPENROUTER_API_KEY", None)
            key = get_api_key()
            assert key == "stored-key"


class TestUpdateConfig:
    """Tests for config update functionality."""

    def test_update_model(self, temp_config_dir):
        """Test updating model setting."""
        update_config("models.default", "new-model")

        config = load_config()
        assert config.models.default == "new-model"

    def test_update_temperature(self, temp_config_dir):
        """Test updating temperature (float coercion)."""
        update_config("models.temperature", "0.5")

        config = load_config()
        assert config.models.temperature == 0.5

    def test_update_boolean(self, temp_config_dir):
        """Test updating boolean setting."""
        update_config("output.verbose", "true")

        config = load_config()
        assert config.output.verbose is True

    def test_invalid_section(self, temp_config_dir):
        """Test error on invalid section."""
        with pytest.raises(ValueError, match="Unknown section"):
            update_config("invalid.key", "value")

    def test_invalid_key(self, temp_config_dir):
        """Test error on invalid key format."""
        with pytest.raises(ValueError, match="Invalid config key"):
            update_config("toplevel", "value")


class TestEffectiveConfig:
    """Tests for effective config with overrides."""

    def test_all_overrides(self, temp_config_dir):
        """Test applying all overrides."""
        config, key = get_effective_config(
            api_key="test-key",
            api_url="https://custom.api.com",
            model="custom-model",
            temperature=0.8,
            schema_version="8.3.0",
            output_format="json",
        )

        assert key == "test-key"
        assert config.api.url == "https://custom.api.com"
        assert config.models.default == "custom-model"
        assert config.models.temperature == 0.8
        assert config.settings.schema_version == "8.3.0"
        assert config.output.format == "json"

    def test_partial_overrides(self, temp_config_dir):
        """Test applying partial overrides."""
        # Save some config first
        initial = CLIConfig()
        initial.models.default = "saved-model"
        save_config(initial)

        config, key = get_effective_config(temperature=0.5)

        # Model should be from saved config
        assert config.models.default == "saved-model"
        # Temperature should be overridden
        assert config.models.temperature == 0.5


class TestMachineID:
    """Tests for machine ID generation and caching."""

    def test_machine_id_generation(self, temp_config_dir):
        """Test that machine ID is generated on first call."""
        machine_id = get_machine_id()

        # Should be 16 hex characters
        assert len(machine_id) == 16
        assert all(c in "0123456789abcdef" for c in machine_id)

    def test_machine_id_persistence(self, temp_config_dir):
        """Test that machine ID persists across calls."""
        first_id = get_machine_id()
        second_id = get_machine_id()

        # Should return same ID
        assert first_id == second_id

    def test_machine_id_file_creation(self, temp_config_dir):
        """Test that machine ID is saved to file."""
        from src.cli.config import MACHINE_ID_FILE

        machine_id = get_machine_id()

        # File should exist
        assert MACHINE_ID_FILE.exists()

        # File should contain the ID
        saved_id = MACHINE_ID_FILE.read_text().strip()
        assert saved_id == machine_id

    def test_machine_id_loads_from_file(self, temp_config_dir):
        """Test that machine ID is loaded from existing file."""
        from src.cli.config import MACHINE_ID_FILE

        # Manually create a machine ID file
        test_id = "a1b2c3d4e5f67890"
        MACHINE_ID_FILE.write_text(test_id)

        # Should load the existing ID
        loaded_id = get_machine_id()
        assert loaded_id == test_id

    def test_machine_id_regenerates_on_corruption(self, temp_config_dir):
        """Test that corrupted machine ID file is regenerated."""
        from src.cli.config import MACHINE_ID_FILE

        # Create corrupted file (invalid format)
        MACHINE_ID_FILE.write_text("invalid-id-format")

        # Should regenerate a valid ID
        new_id = get_machine_id()
        assert len(new_id) == 16
        assert all(c in "0123456789abcdef" for c in new_id)
        assert new_id != "invalid-id-format"


class TestFirstRunDisclosure:
    """Tests for first-run disclosure tracking."""

    def test_is_first_run_when_file_missing(self, temp_config_dir):
        """Test that is_first_run returns True when marker file doesn't exist."""
        assert is_first_run() is True

    def test_is_first_run_when_file_exists(self, temp_config_dir):
        """Test that is_first_run returns False when marker file exists."""
        from src.cli.config import FIRST_RUN_FILE

        # Create the marker file
        FIRST_RUN_FILE.touch()

        assert is_first_run() is False

    def test_mark_first_run_complete(self, temp_config_dir):
        """Test that mark_first_run_complete creates the marker file."""
        from src.cli.config import FIRST_RUN_FILE

        # Initially should be first run
        assert is_first_run() is True

        # Mark as complete
        mark_first_run_complete()

        # File should exist
        assert FIRST_RUN_FILE.exists()

        # Should no longer be first run
        assert is_first_run() is False

    def test_mark_first_run_complete_idempotent(self, temp_config_dir):
        """Test that marking first run complete multiple times is safe."""
        from src.cli.config import FIRST_RUN_FILE

        # Mark as complete twice
        mark_first_run_complete()
        mark_first_run_complete()

        # File should exist
        assert FIRST_RUN_FILE.exists()

        # Should not be first run
        assert is_first_run() is False

    def test_mark_first_run_complete_handles_write_error(self, temp_config_dir):
        """Test that mark_first_run_complete handles OSError gracefully."""
        from unittest.mock import patch

        # Mock Path.touch to raise OSError
        with patch("src.cli.config.FIRST_RUN_FILE") as mock_file:
            mock_file.touch.side_effect = OSError("Permission denied")

            # Should not raise, just silently fail
            mark_first_run_complete()


class TestTelemetryConfig:
    """Tests for telemetry configuration."""

    def test_default_telemetry_config(self):
        """Test default telemetry configuration."""
        from src.cli.config import TelemetryConfig

        config = TelemetryConfig()

        assert config.enabled is True
        assert "openai/gpt-oss-120b" in config.model_blacklist

    def test_telemetry_config_disabled(self):
        """Test telemetry can be disabled."""
        from src.cli.config import TelemetryConfig

        config = TelemetryConfig(enabled=False)

        assert config.enabled is False

    def test_telemetry_config_custom_blacklist(self):
        """Test telemetry with custom model blacklist."""
        from src.cli.config import TelemetryConfig

        config = TelemetryConfig(model_blacklist=["custom/model"])

        assert config.model_blacklist == ["custom/model"]
