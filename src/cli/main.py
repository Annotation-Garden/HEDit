"""HEDit CLI - Main entry point.

Command-line interface for generating HED annotations from natural language.
Supports two execution modes:
- API mode (default): Uses api.annotation.garden backend
- Standalone mode: Runs LangGraph workflow locally (requires hedit[standalone])
"""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from src.cli import output
from src.cli.client import APIError
from src.cli.config import (
    CONFIG_FILE,
    CREDENTIALS_FILE,
    CLIConfig,
    clear_credentials,
    get_config_paths,
    get_effective_config,
    is_first_run,
    load_config,
    load_credentials,
    mark_first_run_complete,
    save_config,
    save_credentials,
    update_config,
)
from src.cli.executor import ExecutionBackend, ExecutionError
from src.version import __version__

# Main app
app = typer.Typer(
    name="hedit",
    help="Generate HED annotations from natural language descriptions.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# Config subcommand group
config_app = typer.Typer(
    name="config",
    help="Manage CLI configuration.",
    no_args_is_help=True,
)
app.add_typer(config_app, name="config")

console = Console()

# Common options as type aliases
ApiKeyOption = Annotated[
    str | None,
    typer.Option(
        "--api-key",
        "-k",
        help="OpenRouter API key (or use OPENROUTER_API_KEY env var)",
        envvar="OPENROUTER_API_KEY",
    ),
]

ApiUrlOption = Annotated[
    str | None,
    typer.Option(
        "--api-url",
        help="API endpoint URL (default: api.annotation.garden/hedit)",
    ),
]

OutputFormatOption = Annotated[
    str,
    typer.Option(
        "--output",
        "-o",
        help="Output format: 'text' (human-readable) or 'json' (machine-readable)",
    ),
]

SchemaVersionOption = Annotated[
    str | None,
    typer.Option(
        "--schema",
        "-s",
        help="HED schema version (e.g., 8.3.0, 8.4.0)",
    ),
]

VerboseOption = Annotated[
    bool,
    typer.Option(
        "--verbose",
        "-v",
        help="Show detailed output",
    ),
]

ModelOption = Annotated[
    str | None,
    typer.Option(
        "--model",
        "-m",
        help="Model to use (e.g., openai/gpt-oss-120b, gpt-4o-mini)",
    ),
]

ProviderOption = Annotated[
    str | None,
    typer.Option(
        "--provider",
        help="Provider preference (e.g., Cerebras for fast inference)",
    ),
]

EvalModelOption = Annotated[
    str | None,
    typer.Option(
        "--eval-model",
        help="Model for evaluation agent (default: same as --model). Use a consistent model for benchmarking.",
    ),
]

EvalProviderOption = Annotated[
    str | None,
    typer.Option(
        "--eval-provider",
        help="Provider for evaluation model (e.g., Cerebras for qwen models).",
    ),
]

TemperatureOption = Annotated[
    float | None,
    typer.Option(
        "--temperature",
        "-t",
        help="LLM temperature (0.0-1.0, lower = more consistent)",
    ),
]

StandaloneOption = Annotated[
    bool,
    typer.Option(
        "--standalone",
        help="Run locally without backend (requires hedit[standalone])",
    ),
]

ApiModeOption = Annotated[
    bool,
    typer.Option(
        "--api",
        help="Use API backend (default)",
    ),
]

UserIdOption = Annotated[
    str | None,
    typer.Option(
        "--user-id",
        help="Custom user ID for cache optimization (default: auto-generated machine ID)",
        hidden=True,  # Not advertised, but available
    ),
]


def get_executor(
    config: CLIConfig,
    api_key: str | None,
    mode_override: str | None = None,
    user_id: str | None = None,
) -> ExecutionBackend:
    """Get the appropriate execution backend based on configuration.

    Args:
        config: CLI configuration
        api_key: OpenRouter API key
        mode_override: Override mode from --standalone/--api flags
        user_id: Custom user ID for cache optimization (default: auto-generated)

    Returns:
        Configured ExecutionBackend instance

    Raises:
        typer.Exit: If standalone mode requested but dependencies not available
    """
    mode = mode_override or config.execution.mode

    if mode == "standalone":
        from src.cli.local_executor import LocalExecutionBackend

        executor = LocalExecutionBackend(
            api_key=api_key,
            model=config.models.default,
            eval_model=config.models.evaluation,
            eval_provider=config.models.eval_provider,
            vision_model=config.models.vision,
            provider=config.models.provider,
            temperature=config.models.temperature,
            user_id=user_id,
        )

        if not executor.is_available():
            output.print_error(
                "Standalone mode requires additional dependencies",
                hint="Install with: pip install hedit[standalone]",
            )
            raise typer.Exit(1)

        return executor
    else:
        from src.cli.api_executor import APIExecutionBackend

        return APIExecutionBackend(
            api_url=config.api.url,
            api_key=api_key,
            model=config.models.default,
            eval_model=config.models.evaluation,
            eval_provider=config.models.eval_provider,
            vision_model=config.models.vision,
            provider=config.models.provider,
            temperature=config.models.temperature,
            user_id=user_id,
        )


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"hedit version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-V",
            callback=version_callback,
            is_eager=True,
            help="Show version and exit",
        ),
    ] = False,
) -> None:
    """HEDit CLI - Generate HED annotations from natural language.

    Convert event descriptions to valid HED (Hierarchical Event Descriptors)
    annotations using AI-powered multi-agent system.

    Get started:
        hedit init --api-key YOUR_OPENROUTER_KEY
        hedit annotate "A red circle appears on screen"
    """
    pass


@app.command()
def init(
    api_key: Annotated[
        str | None,
        typer.Option(
            "--api-key",
            "-k",
            help="OpenRouter API key (get one at https://openrouter.ai/keys)",
            prompt="OpenRouter API key",
            hide_input=True,
        ),
    ] = None,
    api_url: ApiUrlOption = None,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help="Default model for annotation",
        ),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option(
            "--provider",
            help="Provider preference (e.g., Cerebras for fast inference)",
        ),
    ] = None,
    temperature: Annotated[
        float | None,
        typer.Option(
            "--temperature",
            "-t",
            help="LLM temperature (0.0-1.0, lower = more consistent)",
        ),
    ] = None,
    standalone: Annotated[
        bool,
        typer.Option(
            "--standalone",
            help="Set default mode to standalone (run locally without backend)",
        ),
    ] = False,
) -> None:
    """Initialize HEDit CLI with your API key and preferences.

    This saves your configuration to ~/.config/hedit/ so you don't need
    to provide the API key for every command.

    Get an OpenRouter API key at: https://openrouter.ai/keys

    Examples:
        hedit init --api-key YOUR_KEY           # API mode (default)
        hedit init --api-key YOUR_KEY --standalone  # Standalone mode
    """
    # Show telemetry disclosure on first run
    if is_first_run():
        show_telemetry_disclosure()
        mark_first_run_complete()

    # Load existing config
    config = load_config()
    creds = load_credentials()

    # Update with provided values
    if api_key:
        creds.openrouter_api_key = api_key
    if api_url:
        config.api.url = api_url
    if model:
        config.models.default = model
    if provider:
        config.models.provider = provider
    if temperature is not None:
        config.models.temperature = temperature
    if standalone:
        config.execution.mode = "standalone"

    # Save
    save_credentials(creds)
    save_config(config)

    output.print_success("Configuration saved!")
    output.print_info(f"Config file: {CONFIG_FILE}")
    output.print_info(f"Credentials: {CREDENTIALS_FILE}")
    output.print_info(f"Execution mode: {config.execution.mode}")

    # Test connection based on mode
    if creds.openrouter_api_key:
        if config.execution.mode == "standalone":
            output.print_progress("Checking standalone mode dependencies")
            try:
                executor = get_executor(config, creds.openrouter_api_key)
                health = executor.health()
                if health.get("status") == "healthy":
                    output.print_success("Standalone mode ready!")
                    if not health.get("validator_available"):
                        output.print_info(
                            "Note: hedtools not installed; local validation unavailable"
                        )
                else:
                    output.print_info(f"Status: {health.get('status', 'unknown')}")
            except ExecutionError as e:
                output.print_error(f"Standalone mode issue: {e}", hint=e.detail)
        else:
            output.print_progress("Testing API connection")
            try:
                executor = get_executor(config, creds.openrouter_api_key)
                health = executor.health()
                if health.get("status") == "healthy":
                    output.print_success("API connection successful!")
                else:
                    output.print_info(f"API status: {health.get('status', 'unknown')}")
            except ExecutionError as e:
                output.print_error(f"Could not connect to API: {e}", hint=e.detail)
            except APIError as e:
                output.print_error(
                    f"Could not connect to API: {e}", hint="Check your API key and URL"
                )
            except Exception as e:
                output.print_error(f"Connection test failed: {e}")


@app.command()
def annotate(
    description: Annotated[
        str,
        typer.Argument(help="Natural language event description"),
    ],
    api_key: ApiKeyOption = None,
    api_url: ApiUrlOption = None,
    model: ModelOption = None,
    eval_model: EvalModelOption = None,
    eval_provider: EvalProviderOption = None,
    provider: ProviderOption = None,
    temperature: TemperatureOption = None,
    schema_version: SchemaVersionOption = None,
    output_format: OutputFormatOption = "text",
    max_attempts: Annotated[
        int,
        typer.Option(
            "--max-attempts",
            help="Maximum validation attempts",
        ),
    ] = 5,
    assessment: Annotated[
        bool,
        typer.Option(
            "--assessment/--no-assessment",
            help="Run completeness assessment",
        ),
    ] = False,
    standalone: StandaloneOption = False,
    api_mode: ApiModeOption = False,
    verbose: VerboseOption = False,
    user_id: UserIdOption = None,
) -> None:
    """Generate HED annotation from a text description.

    Examples:
        hedit annotate "A red circle appears on the left side of the screen"
        hedit annotate "Participant pressed the spacebar" --schema 8.4.0
        hedit annotate "Audio beep plays" -o json > result.json
        hedit annotate "..." --model gpt-4o-mini --temperature 0.2
        hedit annotate "..." --standalone  # Run locally
    """
    # Show telemetry disclosure on first run
    if is_first_run():
        show_telemetry_disclosure()
        mark_first_run_complete()

    # Determine mode override
    mode_override = None
    if standalone:
        mode_override = "standalone"
    elif api_mode:
        mode_override = "api"

    config, effective_key = get_effective_config(
        api_key=api_key,
        api_url=api_url,
        model=model,
        eval_model=eval_model,
        eval_provider=eval_provider,
        provider=provider,
        temperature=temperature,
        schema_version=schema_version,
        output_format=output_format,
        user_id=user_id,
    )

    if not effective_key:
        output.print_error(
            "No API key configured",
            hint="Run 'hedit init' or provide --api-key",
        )
        raise typer.Exit(1)

    # Show progress if not piped
    mode_name = mode_override or config.execution.mode
    if not output.is_piped():
        output.print_progress(f"Generating HED annotation ({mode_name} mode)")

    try:
        executor = get_executor(config, effective_key, mode_override, config.settings.user_id)
        result = executor.annotate(
            description=description,
            schema_version=schema_version or config.settings.schema_version,
            max_validation_attempts=max_attempts,
            run_assessment=assessment,
        )
        output.print_annotation_result(result, output_format, verbose)

        # Exit with error code if annotation failed
        if result.get("status") != "success" or not result.get("is_valid"):
            raise typer.Exit(1)

    except ExecutionError as e:
        output.print_error(str(e), hint=e.detail)
        raise typer.Exit(1) from None
    except APIError as e:
        output.print_error(str(e), hint=e.detail)
        raise typer.Exit(1) from None


@app.command("annotate-image")
def annotate_image(
    image: Annotated[
        Path,
        typer.Argument(help="Path to image file (PNG, JPG, etc.)"),
    ],
    prompt: Annotated[
        str | None,
        typer.Option(
            "--prompt",
            help="Custom prompt for vision model",
        ),
    ] = None,
    api_key: ApiKeyOption = None,
    api_url: ApiUrlOption = None,
    model: ModelOption = None,
    eval_model: EvalModelOption = None,
    eval_provider: EvalProviderOption = None,
    provider: ProviderOption = None,
    temperature: TemperatureOption = None,
    schema_version: SchemaVersionOption = None,
    output_format: OutputFormatOption = "text",
    max_attempts: Annotated[
        int,
        typer.Option(
            "--max-attempts",
            help="Maximum validation attempts",
        ),
    ] = 5,
    assessment: Annotated[
        bool,
        typer.Option(
            "--assessment/--no-assessment",
            help="Run completeness assessment",
        ),
    ] = False,
    standalone: StandaloneOption = False,
    api_mode: ApiModeOption = False,
    verbose: VerboseOption = False,
    user_id: UserIdOption = None,
) -> None:
    """Generate HED annotation from an image.

    First generates a description using a vision model, then annotates it.

    Examples:
        hedit annotate-image stimulus.png
        hedit annotate-image photo.jpg --prompt "Describe the experimental setup"
        hedit annotate-image screen.png -o json > result.json
        hedit annotate-image stimulus.png --standalone  # Run locally
    """
    # Show telemetry disclosure on first run
    if is_first_run():
        show_telemetry_disclosure()
        mark_first_run_complete()

    # Validate image exists
    if not image.exists():
        output.print_error(f"Image file not found: {image}")
        raise typer.Exit(1)

    # Determine mode override
    mode_override = None
    if standalone:
        mode_override = "standalone"
    elif api_mode:
        mode_override = "api"

    config, effective_key = get_effective_config(
        api_key=api_key,
        api_url=api_url,
        model=model,
        eval_model=eval_model,
        eval_provider=eval_provider,
        provider=provider,
        temperature=temperature,
        schema_version=schema_version,
        output_format=output_format,
        user_id=user_id,
    )

    if not effective_key:
        output.print_error(
            "No API key configured",
            hint="Run 'hedit init' or provide --api-key",
        )
        raise typer.Exit(1)

    mode_name = mode_override or config.execution.mode
    if not output.is_piped():
        output.print_progress(f"Analyzing image and generating HED annotation ({mode_name} mode)")

    try:
        executor = get_executor(config, effective_key, mode_override, config.settings.user_id)
        result = executor.annotate_image(
            image_path=image,
            prompt=prompt,
            schema_version=schema_version or config.settings.schema_version,
            max_validation_attempts=max_attempts,
            run_assessment=assessment,
        )
        output.print_image_annotation_result(result, output_format, verbose)

        if result.get("status") != "success" or not result.get("is_valid"):
            raise typer.Exit(1)

    except ExecutionError as e:
        output.print_error(str(e), hint=e.detail)
        raise typer.Exit(1) from None
    except APIError as e:
        output.print_error(str(e), hint=e.detail)
        raise typer.Exit(1) from None


@app.command()
def validate(
    hed_string: Annotated[
        str,
        typer.Argument(help="HED annotation string to validate"),
    ],
    api_key: ApiKeyOption = None,
    api_url: ApiUrlOption = None,
    schema_version: SchemaVersionOption = None,
    output_format: OutputFormatOption = "text",
    standalone: StandaloneOption = False,
    api_mode: ApiModeOption = False,
) -> None:
    """Validate a HED annotation string.

    Checks if the HED string is syntactically correct and semantically valid
    according to the HED schema.

    Examples:
        hedit validate "Sensory-event, Visual-presentation"
        hedit validate "(Red, Circle)" --schema 8.4.0
        hedit validate "Event" -o json
        hedit validate "Event" --standalone  # Validate locally with hedtools
    """
    # Show telemetry disclosure on first run
    if is_first_run():
        show_telemetry_disclosure()
        mark_first_run_complete()

    # Determine mode override
    mode_override = None
    if standalone:
        mode_override = "standalone"
    elif api_mode:
        mode_override = "api"

    config, effective_key = get_effective_config(
        api_key=api_key,
        api_url=api_url,
        schema_version=schema_version,
        output_format=output_format,
    )

    # For standalone validation, we don't need an API key (uses hedtools locally)
    effective_mode = mode_override or config.execution.mode
    if effective_mode != "standalone" and not effective_key:
        output.print_error(
            "No API key configured",
            hint="Run 'hedit init' or provide --api-key, or use --standalone for local validation",
        )
        raise typer.Exit(1)

    if not output.is_piped():
        output.print_progress(f"Validating HED string ({effective_mode} mode)")

    try:
        executor = get_executor(config, effective_key, mode_override)
        result = executor.validate(
            hed_string=hed_string,
            schema_version=schema_version or config.settings.schema_version,
        )
        output.print_validation_result(result, output_format)

        if not result.get("is_valid"):
            raise typer.Exit(1)

    except ExecutionError as e:
        output.print_error(str(e), hint=e.detail)
        raise typer.Exit(1) from None
    except APIError as e:
        output.print_error(str(e), hint=e.detail)
        raise typer.Exit(1) from None


# Config subcommands


@config_app.command("show")
def config_show(
    show_key: Annotated[
        bool,
        typer.Option(
            "--show-key",
            help="Show full API key (default: masked)",
        ),
    ] = False,
) -> None:
    """Show current configuration."""
    config = load_config()
    creds = load_credentials()

    # Merge for display
    config_dict = config.model_dump()
    config_dict["credentials"] = {"openrouter_api_key": creds.openrouter_api_key}

    output.print_config(config_dict, show_key)

    # Show file paths
    paths = get_config_paths()
    output.print_info(f"\nConfig directory: {paths['config_dir']}")


@config_app.command("set")
def config_set(
    key: Annotated[
        str,
        typer.Argument(help="Config key (e.g., models.default, settings.temperature)"),
    ],
    value: Annotated[
        str,
        typer.Argument(help="New value"),
    ],
) -> None:
    """Set a configuration value.

    Examples:
        hedit config set models.default gpt-4o
        hedit config set settings.temperature 0.2
        hedit config set api.url https://api.example.com/hedit
    """
    try:
        update_config(key, value)
        output.print_success(f"Set {key} = {value}")
    except ValueError as e:
        output.print_error(str(e))
        raise typer.Exit(1) from None


@config_app.command("path")
def config_path() -> None:
    """Show configuration file paths."""
    paths = get_config_paths()
    console.print(f"Config directory: {paths['config_dir']}")
    console.print(f"Config file: {paths['config_file']}")
    console.print(f"Credentials file: {paths['credentials_file']}")


@config_app.command("clear-credentials")
def config_clear_credentials(
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Skip confirmation",
        ),
    ] = False,
) -> None:
    """Remove stored API credentials."""
    if not force:
        confirm = typer.confirm("Are you sure you want to remove stored credentials?")
        if not confirm:
            raise typer.Abort()

    clear_credentials()
    output.print_success("Credentials removed")


@app.command()
def health(
    api_url: ApiUrlOption = None,
    standalone: StandaloneOption = False,
    api_mode: ApiModeOption = False,
) -> None:
    """Check health status of the execution backend.

    Examples:
        hedit health                 # Check API health
        hedit health --standalone    # Check standalone mode dependencies
    """
    # Determine mode override
    mode_override = None
    if standalone:
        mode_override = "standalone"
    elif api_mode:
        mode_override = "api"

    config, _ = get_effective_config(api_url=api_url)
    effective_mode = mode_override or config.execution.mode

    try:
        # For health check, we don't require an API key
        executor = get_executor(config, api_key=None, mode_override=mode_override)
        result = executor.health()

        status = result.get("status", "unknown")
        version = result.get("version", "unknown")
        mode = result.get("mode", effective_mode)
        llm = "[green][x][/]" if result.get("llm_available") else "[red][ ][/]"
        validator = "[green][x][/]" if result.get("validator_available") else "[red][ ][/]"

        console.print(f"Mode: [bold]{mode}[/]")
        if mode == "api":
            console.print(f"API: {config.api.url}")
        console.print(f"Status: [bold]{status}[/]")
        console.print(f"Version: {version}")
        console.print(f"LLM: {llm}")
        console.print(f"Validator: {validator}")

        # Show dependency details for standalone mode
        if mode == "standalone" and "dependencies" in result:
            deps = result["dependencies"]
            console.print("\nDependencies:")
            for dep, available in deps.items():
                status_icon = "[green][x][/]" if available else "[red][ ][/]"
                console.print(f"  {status_icon} {dep}")

    except ExecutionError as e:
        output.print_error(str(e), hint=e.detail)
        raise typer.Exit(1) from None
    except APIError as e:
        output.print_error(str(e), hint=e.detail)
        raise typer.Exit(1) from None
    except Exception as e:
        output.print_error(f"Health check failed: {e}")
        raise typer.Exit(1) from None


def show_telemetry_disclosure() -> None:
    """Display first-run telemetry disclosure notice."""
    from rich.panel import Panel

    disclosure_text = (
        "[bold]Welcome to HEDit![/]\n\n"
        "HEDit collects anonymous usage data to improve the annotation service:\n"
        "  • Input descriptions and generated annotations\n"
        "  • Model performance metrics (latency, iterations)\n"
        "  • Validation results\n\n"
        "[dim]What is NOT collected:[/]\n"
        "  • API keys or credentials\n"
        "  • Personal information\n"
        "  • File paths or system details\n\n"
        "[bold cyan]To disable:[/] hedit config set telemetry.enabled false\n"
        "[bold cyan]To view config:[/] hedit config show"
    )

    panel = Panel(
        disclosure_text,
        title="[bold]Privacy & Data Collection[/]",
        border_style="cyan",
        padding=(1, 2),
    )

    console.print()
    console.print(panel)
    console.print()


def cli() -> None:
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    cli()
