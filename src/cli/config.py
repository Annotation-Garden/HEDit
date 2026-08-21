"""Configuration management for HEDit CLI.

Handles persistent storage of API keys and settings in a cross-platform config directory.
Supports environment variables as fallback/override.
"""

import os
import uuid
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
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
MACHINE_ID_FILE = CONFIG_DIR / "machine_id"
FIRST_RUN_FILE = CONFIG_DIR / ".first_run"

# Default API endpoint
DEFAULT_API_URL = "https://api.annotation.garden/hedit"
DEFAULT_DEV_API_URL = "https://api.annotation.garden/hedit-dev"

# Default models (Anthropic Claude only since the 2026-08-18 migration).
# Annotation model: Claude Haiku 4.5, which runs with extended thinking and
# matched Sonnet 5's first-attempt validity for 2.3x less cost (docs/reasoning.md).
# Claude Sonnet 5 ("claude-sonnet-5") stays selectable for comparison.
DEFAULT_MODEL = "claude-haiku-4-5"

# Evaluation judge stays on Haiku regardless of the annotation model.
DEFAULT_EVAL_MODEL = "claude-haiku-4-5"

# Vision model: Claude models are natively multimodal.
DEFAULT_VISION_MODEL = "claude-haiku-4-5"


class CredentialsConfig(BaseModel):
    """Credentials stored separately with restricted permissions."""

    anthropic_api_key: str | None = Field(
        default=None, description="Anthropic API key for BYOK mode (sk-ant-...)"
    )
    # Legacy field from pre-migration configs; no longer used
    openrouter_api_key: str | None = Field(default=None, description="Deprecated (unused)")

    @property
    def api_key(self) -> str | None:
        """The active BYOK key (Anthropic only)."""
        return self.anthropic_api_key


class ModelsConfig(BaseModel):
    """Model configuration for different agents."""

    default: str = Field(default=DEFAULT_MODEL, description="Default model for annotation")
    provider: str | None = Field(default=None, description="Deprecated (unused)")
    evaluation: str | None = Field(
        default=DEFAULT_EVAL_MODEL,
        description="Model for evaluation/assessment agents",
    )
    eval_provider: str | None = Field(default=None, description="Deprecated (unused)")
    vision: str = Field(default=DEFAULT_VISION_MODEL, description="Vision model for images")
    vision_provider: str | None = Field(default=None, description="Deprecated (unused)")
    temperature: float = Field(default=0.1, ge=0.0, le=1.0, description="Model temperature")


class ExecutionMode(BaseModel):
    """Execution mode configuration."""

    mode: str = Field(
        default="api",
        description="Execution mode: 'api' (use backend) or 'standalone' (run locally)",
    )


class SettingsConfig(BaseModel):
    """General settings."""

    schema_version: str = Field(default="8.4.0", description="HED schema version")
    max_validation_attempts: int = Field(default=5, ge=1, le=10, description="Max retries")
    run_assessment: bool = Field(default=False, description="Run assessment by default")
    user_id: str | None = Field(
        default=None,
        description="Optional user ID sent as the X-User-Id header (currently unused server-side)",
    )


class OutputConfig(BaseModel):
    """Output formatting settings."""

    format: str = Field(default="text", description="Output format (text, json)")
    color: bool = Field(default=True, description="Enable colored output")
    verbose: bool = Field(default=False, description="Verbose output")
    streaming: bool = Field(default=True, description="Enable streaming progress display")


class APIConfig(BaseModel):
    """API endpoint configuration."""

    url: str = Field(default=DEFAULT_API_URL, description="API endpoint URL")


class TelemetryConfig(BaseModel):
    """Telemetry configuration."""

    enabled: bool = Field(default=True, description="Enable telemetry collection")
    model_blacklist: list[str] = Field(
        default_factory=lambda: ["openai/gpt-oss-120b"],
        description="Models to exclude from telemetry",
    )


class CLIConfig(BaseModel):
    """Complete CLI configuration."""

    api: APIConfig = Field(default_factory=APIConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    settings: SettingsConfig = Field(default_factory=SettingsConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    execution: ExecutionMode = Field(default_factory=ExecutionMode)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)


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

    # A legacy OpenRouter key on file no longer works; say so once instead
    # of letting a previously-working setup fail with a generic auth error.
    if creds.openrouter_api_key and not creds.anthropic_api_key:
        import sys

        print(
            "hedit: found a legacy OpenRouter key in the credentials file; "
            "it is no longer used. Run 'hedit init --api-key sk-ant-...' "
            "for BYOK mode, or no key for the hosted API.",
            file=sys.stderr,
        )

    # Environment variables override file. HEDIT_ANTHROPIC_API_KEY is the
    # BYOK override; the plain ANTHROPIC_API_KEY env var is deliberately NOT
    # read here because it may hold Claude Platform on AWS credentials that
    # only work with the server-mode base URL and workspace header.
    env_key = os.environ.get("HEDIT_ANTHROPIC_API_KEY")
    if env_key:
        creds.anthropic_api_key = env_key

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
    return creds.api_key


def get_effective_config(
    api_key: str | None = None,
    api_url: str | None = None,
    model: str | None = None,
    eval_model: str | None = None,
    temperature: float | None = None,
    schema_version: str | None = None,
    output_format: str | None = None,
    mode: str | None = None,
    user_id: str | None = None,
) -> tuple[CLIConfig, str | None]:
    """Get effective config with command-line overrides applied.

    Args:
        api_key: Override API key
        api_url: Override API URL
        model: Override model
        eval_model: Override evaluation model (for consistent benchmarking)
        temperature: Override temperature
        schema_version: Override schema version
        output_format: Override output format
        mode: Override execution mode ("api" or "standalone")
        user_id: Optional ID sent as X-User-Id in API mode (server ignores it)

    Returns:
        Tuple of (effective config, effective API key)
    """
    config = load_config()
    effective_key = get_api_key(api_key)

    # Apply overrides
    if api_url:
        config.api.url = api_url

    if model:
        config.models.default = model
    if eval_model:
        config.models.evaluation = eval_model

    if temperature is not None:
        config.models.temperature = temperature
    if schema_version:
        config.settings.schema_version = schema_version
    if output_format:
        config.output.format = output_format
    if mode:
        if mode not in ("api", "standalone"):
            raise ValueError(f"Invalid mode: {mode}. Must be 'api' or 'standalone'")
        config.execution.mode = mode
    if user_id:
        config.settings.user_id = user_id

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


def reset_config(preserve_credentials: bool = True) -> CLIConfig:
    """Reset configuration to defaults.

    Args:
        preserve_credentials: If True, keep BYOK API key intact (default: True)

    Returns:
        The new default configuration
    """
    # Create fresh default config
    config = CLIConfig()

    # Save the default config
    save_config(config)

    return config


def get_machine_id() -> str:
    """Get or generate a stable machine ID.

    Historically used for provider cache routing; kept as a stable local
    identifier (e.g., as a default X-User-Id). It is generated once and
    persists across pip updates.

    Returns:
        16-character hexadecimal machine ID
    """
    ensure_config_dir()

    if MACHINE_ID_FILE.exists():
        try:
            machine_id = MACHINE_ID_FILE.read_text().strip()
            # Validate format (16 hex chars)
            if len(machine_id) == 16 and all(c in "0123456789abcdef" for c in machine_id):
                return machine_id
        except (OSError, UnicodeDecodeError):
            pass  # File corrupted, regenerate

    # Generate new machine ID
    machine_id = uuid.uuid4().hex[:16]

    # Save to file
    try:
        MACHINE_ID_FILE.write_text(machine_id)
        # Readable by user only (Unix)
        try:
            os.chmod(MACHINE_ID_FILE, 0o600)
        except (OSError, AttributeError):
            pass  # Windows doesn't support chmod the same way
    except OSError:
        pass  # If we can't write, still return the ID for this session

    return machine_id


def is_first_run() -> bool:
    """Check if this is the first time HEDit is run.

    Returns:
        True if first run, False otherwise
    """
    return not FIRST_RUN_FILE.exists()


def mark_first_run_complete() -> None:
    """Mark first run as complete by creating the marker file."""
    ensure_config_dir()
    try:
        FIRST_RUN_FILE.touch()
    except OSError:
        pass  # Ignore write errors


def get_config_paths() -> dict[str, Path]:
    """Get paths to config files for debugging."""
    return {
        "config_dir": CONFIG_DIR,
        "config_file": CONFIG_FILE,
        "credentials_file": CREDENTIALS_FILE,
        "machine_id_file": MACHINE_ID_FILE,
    }
