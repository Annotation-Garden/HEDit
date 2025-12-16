"""Configuration management for HEDit CLI.

Handles persistent storage of API keys and settings in a cross-platform config directory.
Supports environment variables as fallback/override.
"""

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

# Cross-platform config directory
# Linux: ~/.config/hedit
# macOS: ~/Library/Application Support/hedit
# Windows: C:\\Users\\<user>\\AppData\\Local\\hedit
try:
    from platformdirs import user_config_dir

    CONFIG_DIR = Path(user_config_dir("hedit", appauthor=False))
except ImportError:
    # Fallback if platformdirs not available
    CONFIG_DIR = Path.home() / ".config" / "hedit"

CONFIG_FILE = CONFIG_DIR / "config.yaml"
CREDENTIALS_FILE = CONFIG_DIR / "credentials.yaml"

# Default API endpoint
DEFAULT_API_URL = "https://api.annotation.garden/hedit"
DEFAULT_DEV_API_URL = "https://api.annotation.garden/hedit-dev"

# Default models (Cerebras-optimized)
DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_VISION_MODEL = "qwen/qwen3-vl-30b-a3b-instruct"
DEFAULT_PROVIDER = "Cerebras"


class CredentialsConfig(BaseModel):
    """Credentials stored separately with restricted permissions."""

    openrouter_api_key: str | None = Field(default=None, description="OpenRouter API key")


class ModelsConfig(BaseModel):
    """Model configuration for different agents."""

    default: str = Field(default=DEFAULT_MODEL, description="Default model for annotation")
    vision: str = Field(default=DEFAULT_VISION_MODEL, description="Vision model for images")
    provider: str | None = Field(default=DEFAULT_PROVIDER, description="Provider preference")
    temperature: float = Field(default=0.1, ge=0.0, le=1.0, description="Model temperature")


class SettingsConfig(BaseModel):
    """General settings."""

    schema_version: str = Field(default="8.4.0", description="HED schema version")
    max_validation_attempts: int = Field(default=5, ge=1, le=10, description="Max retries")
    run_assessment: bool = Field(default=False, description="Run assessment by default")


class OutputConfig(BaseModel):
    """Output formatting settings."""

    format: str = Field(default="text", description="Output format (text, json)")
    color: bool = Field(default=True, description="Enable colored output")
    verbose: bool = Field(default=False, description="Verbose output")


class APIConfig(BaseModel):
    """API endpoint configuration."""

    url: str = Field(default=DEFAULT_API_URL, description="API endpoint URL")


class CLIConfig(BaseModel):
    """Complete CLI configuration."""

    api: APIConfig = Field(default_factory=APIConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    settings: SettingsConfig = Field(default_factory=SettingsConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


def ensure_config_dir() -> None:
    """Create config directory if it doesn't exist."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def load_credentials() -> CredentialsConfig:
    """Load credentials from file or environment.

    Environment variables take precedence over stored credentials.
    """
    creds = CredentialsConfig()

    # Try loading from file first
    if CREDENTIALS_FILE.exists():
        try:
            with open(CREDENTIALS_FILE) as f:
                data = yaml.safe_load(f) or {}
                creds = CredentialsConfig(**data)
        except (yaml.YAMLError, ValueError):
            pass  # Use defaults if file is corrupted

    # Environment variables override file
    env_key = os.environ.get("OPENROUTER_API_KEY")
    if env_key:
        creds.openrouter_api_key = env_key

    return creds


def save_credentials(creds: CredentialsConfig) -> None:
    """Save credentials to file with restricted permissions."""
    ensure_config_dir()

    # Write credentials
    with open(CREDENTIALS_FILE, "w") as f:
        yaml.dump(creds.model_dump(exclude_none=True), f, default_flow_style=False)

    # Restrict permissions (Unix only)
    try:
        os.chmod(CREDENTIALS_FILE, 0o600)
    except (OSError, AttributeError):
        pass  # Windows doesn't support chmod the same way


def load_config() -> CLIConfig:
    """Load configuration from file."""
    if not CONFIG_FILE.exists():
        return CLIConfig()

    try:
        with open(CONFIG_FILE) as f:
            data = yaml.safe_load(f) or {}
            return CLIConfig(**data)
    except (yaml.YAMLError, ValueError):
        return CLIConfig()


def save_config(config: CLIConfig) -> None:
    """Save configuration to file."""
    ensure_config_dir()

    with open(CONFIG_FILE, "w") as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False)


def get_api_key(override: str | None = None) -> str | None:
    """Get API key with priority: override > env > stored.

    Args:
        override: Explicit API key from command line

    Returns:
        API key or None if not configured
    """
    if override:
        return override

    creds = load_credentials()
    return creds.openrouter_api_key


def get_effective_config(
    api_key: str | None = None,
    api_url: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    temperature: float | None = None,
    schema_version: str | None = None,
    output_format: str | None = None,
) -> tuple[CLIConfig, str | None]:
    """Get effective config with command-line overrides applied.

    Args:
        api_key: Override API key
        api_url: Override API URL
        model: Override model (if non-default, clears provider unless explicitly set)
        provider: Override provider preference (e.g., "Cerebras")
        temperature: Override temperature
        schema_version: Override schema version
        output_format: Override output format

    Returns:
        Tuple of (effective config, effective API key)

    Note:
        When a custom model is specified without an explicit provider, the provider
        is cleared. This is because the default provider (Cerebras) only supports
        specific models.
    """
    config = load_config()
    effective_key = get_api_key(api_key)

    # Apply overrides
    if api_url:
        config.api.url = api_url

    # Handle model/provider interaction:
    # If user specifies a model different from default but doesn't specify provider,
    # clear the provider (since Cerebras only supports specific models)
    if model:
        config.models.default = model
        # Clear provider if model changed and provider not explicitly set
        if provider is None and model != DEFAULT_MODEL:
            config.models.provider = None
    if provider is not None:  # Allow empty string to clear provider
        config.models.provider = provider if provider else None

    if temperature is not None:
        config.models.temperature = temperature
    if schema_version:
        config.settings.schema_version = schema_version
    if output_format:
        config.output.format = output_format

    return config, effective_key


def update_config(key: str, value: Any) -> None:
    """Update a specific config value.

    Args:
        key: Dot-notation key (e.g., "models.default", "settings.temperature")
        value: New value
    """
    config = load_config()

    # Parse dot notation
    parts = key.split(".")
    if len(parts) == 1:
        # Top-level key not supported for safety
        raise ValueError(f"Invalid config key: {key}")
    elif len(parts) == 2:
        section, field = parts
        if hasattr(config, section):
            section_obj = getattr(config, section)
            if hasattr(section_obj, field):
                # Type coercion for common types
                current = getattr(section_obj, field)
                if isinstance(current, bool):
                    value = str(value).lower() in ("true", "1", "yes")
                elif isinstance(current, int):
                    value = int(value)
                elif isinstance(current, float):
                    value = float(value)
                setattr(section_obj, field, value)
            else:
                raise ValueError(f"Unknown field: {field} in {section}")
        else:
            raise ValueError(f"Unknown section: {section}")
    else:
        raise ValueError(f"Invalid config key format: {key}")

    save_config(config)


def clear_credentials() -> None:
    """Remove stored credentials."""
    if CREDENTIALS_FILE.exists():
        CREDENTIALS_FILE.unlink()


def get_config_paths() -> dict[str, Path]:
    """Get paths to config files for debugging."""
    return {
        "config_dir": CONFIG_DIR,
        "config_file": CONFIG_FILE,
        "credentials_file": CREDENTIALS_FILE,
    }
