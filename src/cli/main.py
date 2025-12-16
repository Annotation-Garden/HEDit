"""HEDit CLI - Main entry point.

Command-line interface for generating HED annotations from natural language.
Uses the HEDit API (api.annotation.garden/hedit) with bring-your-own-key (BYOK) support.
"""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from src.cli import output
from src.cli.client import APIError, create_client
from src.cli.config import (
    CONFIG_FILE,
    CREDENTIALS_FILE,
    clear_credentials,
    get_config_paths,
    get_effective_config,
    load_config,
    load_credentials,
    save_config,
    save_credentials,
    update_config,
)
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
        help="Output format",
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
            "-p",
            help="Provider preference (e.g., Cerebras for fast inference)",
        ),
    ] = None,
) -> None:
    """Initialize HEDit CLI with your API key and preferences.

    This saves your configuration to ~/.config/hedit/ so you don't need
    to provide the API key for every command.

    Get an OpenRouter API key at: https://openrouter.ai/keys
    """
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

    # Save
    save_credentials(creds)
    save_config(config)

    output.print_success("Configuration saved!")
    output.print_info(f"Config file: {CONFIG_FILE}")
    output.print_info(f"Credentials: {CREDENTIALS_FILE}")

    # Test connection
    if creds.openrouter_api_key:
        output.print_progress("Testing API connection")
        try:
            client = create_client(config, creds.openrouter_api_key)
            health = client.health()
            if health.get("status") == "healthy":
                output.print_success("API connection successful!")
            else:
                output.print_info(f"API status: {health.get('status', 'unknown')}")
        except APIError as e:
            output.print_error(f"Could not connect to API: {e}", hint="Check your API key and URL")
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
    verbose: VerboseOption = False,
) -> None:
    """Generate HED annotation from a text description.

    Examples:
        hedit annotate "A red circle appears on the left side of the screen"
        hedit annotate "Participant pressed the spacebar" --schema 8.4.0
        hedit annotate "Audio beep plays" -o json > result.json
    """
    config, effective_key = get_effective_config(
        api_key=api_key,
        api_url=api_url,
        schema_version=schema_version,
        output_format=output_format,
    )

    if not effective_key:
        output.print_error(
            "No API key configured",
            hint="Run 'hedit init' or provide --api-key",
        )
        raise typer.Exit(1)

    # Show progress if not piped
    if not output.is_piped():
        output.print_progress("Generating HED annotation")

    try:
        client = create_client(config, effective_key)
        result = client.annotate(
            description=description,
            schema_version=schema_version or config.settings.schema_version,
            max_validation_attempts=max_attempts,
            run_assessment=assessment,
        )
        output.print_annotation_result(result, output_format, verbose)

        # Exit with error code if annotation failed
        if result.get("status") != "success" or not result.get("is_valid"):
            raise typer.Exit(1)

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
            "-p",
            help="Custom prompt for vision model",
        ),
    ] = None,
    api_key: ApiKeyOption = None,
    api_url: ApiUrlOption = None,
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
    verbose: VerboseOption = False,
) -> None:
    """Generate HED annotation from an image.

    First generates a description using a vision model, then annotates it.

    Examples:
        hedit annotate-image stimulus.png
        hedit annotate-image photo.jpg --prompt "Describe the experimental setup"
        hedit annotate-image screen.png -o json > result.json
    """
    # Validate image exists
    if not image.exists():
        output.print_error(f"Image file not found: {image}")
        raise typer.Exit(1)

    config, effective_key = get_effective_config(
        api_key=api_key,
        api_url=api_url,
        schema_version=schema_version,
        output_format=output_format,
    )

    if not effective_key:
        output.print_error(
            "No API key configured",
            hint="Run 'hedit init' or provide --api-key",
        )
        raise typer.Exit(1)

    if not output.is_piped():
        output.print_progress("Analyzing image and generating HED annotation")

    try:
        client = create_client(config, effective_key)
        result = client.annotate_image(
            image_path=image,
            prompt=prompt,
            schema_version=schema_version or config.settings.schema_version,
            max_validation_attempts=max_attempts,
            run_assessment=assessment,
        )
        output.print_image_annotation_result(result, output_format, verbose)

        if result.get("status") != "success" or not result.get("is_valid"):
            raise typer.Exit(1)

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
) -> None:
    """Validate a HED annotation string.

    Checks if the HED string is syntactically correct and semantically valid
    according to the HED schema.

    Examples:
        hedit validate "Sensory-event, Visual-presentation"
        hedit validate "(Red, Circle)" --schema 8.4.0
        hedit validate "Event" -o json
    """
    config, effective_key = get_effective_config(
        api_key=api_key,
        api_url=api_url,
        schema_version=schema_version,
        output_format=output_format,
    )

    if not effective_key:
        output.print_error(
            "No API key configured",
            hint="Run 'hedit init' or provide --api-key",
        )
        raise typer.Exit(1)

    if not output.is_piped():
        output.print_progress("Validating HED string")

    try:
        client = create_client(config, effective_key)
        result = client.validate(
            hed_string=hed_string,
            schema_version=schema_version or config.settings.schema_version,
        )
        output.print_validation_result(result, output_format)

        if not result.get("is_valid"):
            raise typer.Exit(1)

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
) -> None:
    """Check API health status."""
    config, _ = get_effective_config(api_url=api_url)

    try:
        client = create_client(config)
        result = client.health()

        status = result.get("status", "unknown")
        version = result.get("version", "unknown")
        llm = "[green][x][/]" if result.get("llm_available") else "[red][ ][/]"
        validator = "[green][x][/]" if result.get("validator_available") else "[red][ ][/]"

        console.print(f"API: {config.api.url}")
        console.print(f"Status: [bold]{status}[/]")
        console.print(f"Version: {version}")
        console.print(f"LLM: {llm}")
        console.print(f"Validator: {validator}")

    except APIError as e:
        output.print_error(str(e), hint=e.detail)
        raise typer.Exit(1) from None
    except Exception as e:
        output.print_error(f"Could not connect to API: {e}")
        raise typer.Exit(1) from None


def cli() -> None:
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    cli()
