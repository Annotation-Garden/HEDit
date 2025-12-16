"""Output formatting for HEDit CLI.

Uses Rich for beautiful terminal output with colors, tables, and panels.
"""

import json
import sys
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Console for stdout (results)
console = Console()
# Console for stderr (status messages, errors)
err_console = Console(stderr=True)


def print_json(data: dict[str, Any]) -> None:
    """Print data as formatted JSON."""
    print(json.dumps(data, indent=2))


def print_annotation_result(
    result: dict[str, Any],
    output_format: str = "text",
    verbose: bool = False,
) -> None:
    """Print annotation result in specified format.

    Args:
        result: API response dictionary
        output_format: "text" or "json"
        verbose: Include extra details
    """
    if output_format == "json":
        print_json(result)
        return

    # Text format with Rich
    status = result.get("status", "unknown")
    is_valid = result.get("is_valid", False)
    is_faithful = result.get("is_faithful", False)
    annotation = result.get("annotation", "")

    # Status indicator
    if status == "success" and is_valid:
        status_style = "bold green"
        status_text = "SUCCESS"
    elif is_valid:
        status_style = "bold yellow"
        status_text = "VALID (with warnings)"
    else:
        status_style = "bold red"
        status_text = "FAILED"

    # Build status line
    status_parts = []
    status_parts.append(
        f"[{'green' if is_valid else 'red'}]{'[x]' if is_valid else '[ ]'} Valid[/]"
    )
    status_parts.append(
        f"[{'green' if is_faithful else 'yellow'}]{'[x]' if is_faithful else '[ ]'} Faithful[/]"
    )
    if result.get("is_complete") is not None:
        is_complete = result.get("is_complete", False)
        status_parts.append(
            f"[{'green' if is_complete else 'yellow'}]{'[x]' if is_complete else '[ ]'} Complete[/]"
        )

    attempts = result.get("validation_attempts", 0)
    status_parts.append(f"[dim]({attempts} validation attempt{'s' if attempts != 1 else ''})[/]")

    # Main panel
    content = Text()
    content.append("Annotation:\n", style="bold")
    content.append(f"  {annotation}\n\n")
    content.append("Status: ")
    content.append(" ".join(status_parts))

    # Warnings
    warnings = result.get("validation_warnings", [])
    if warnings:
        content.append("\n\nWarnings:\n", style="bold yellow")
        for w in warnings:
            content.append(f"  - {w}\n", style="yellow")

    # Errors
    errors = result.get("validation_errors", [])
    if errors:
        content.append("\n\nErrors:\n", style="bold red")
        for e in errors:
            content.append(f"  - {e}\n", style="red")

    # Verbose output
    if verbose:
        if result.get("evaluation_feedback"):
            content.append("\n\nEvaluation:\n", style="bold dim")
            content.append(f"  {result['evaluation_feedback']}\n", style="dim")
        if result.get("assessment_feedback"):
            content.append("\n\nAssessment:\n", style="bold dim")
            content.append(f"  {result['assessment_feedback']}\n", style="dim")

    console.print(
        Panel(
            content,
            title=f"[{status_style}]HED Annotation - {status_text}[/]",
            border_style=status_style.replace("bold ", ""),
        )
    )


def print_image_annotation_result(
    result: dict[str, Any],
    output_format: str = "text",
    verbose: bool = False,
) -> None:
    """Print image annotation result.

    Args:
        result: API response dictionary
        output_format: "text" or "json"
        verbose: Include extra details
    """
    if output_format == "json":
        print_json(result)
        return

    # Text format - show image description first, then annotation
    image_desc = result.get("image_description", "")

    # Build content
    content = Text()
    content.append("Image Description:\n", style="bold cyan")
    content.append(f"  {image_desc}\n\n", style="cyan")

    # Then show annotation like normal
    annotation = result.get("annotation", "")
    is_valid = result.get("is_valid", False)
    is_faithful = result.get("is_faithful", False)

    content.append("HED Annotation:\n", style="bold")
    content.append(f"  {annotation}\n\n")

    # Status
    status_parts = []
    status_parts.append(
        f"[{'green' if is_valid else 'red'}]{'[x]' if is_valid else '[ ]'} Valid[/]"
    )
    status_parts.append(
        f"[{'green' if is_faithful else 'yellow'}]{'[x]' if is_faithful else '[ ]'} Faithful[/]"
    )

    content.append("Status: ")
    content.append(" ".join(status_parts))

    # Warnings/errors
    warnings = result.get("validation_warnings", [])
    if warnings:
        content.append("\n\nWarnings:\n", style="bold yellow")
        for w in warnings:
            content.append(f"  - {w}\n", style="yellow")

    errors = result.get("validation_errors", [])
    if errors:
        content.append("\n\nErrors:\n", style="bold red")
        for e in errors:
            content.append(f"  - {e}\n", style="red")

    status = result.get("status", "unknown")
    status_style = "green" if status == "success" and is_valid else "red"

    console.print(
        Panel(
            content,
            title=f"[bold {status_style}]Image Annotation[/]",
            border_style=status_style,
        )
    )


def print_validation_result(
    result: dict[str, Any],
    output_format: str = "text",
) -> None:
    """Print validation result.

    Args:
        result: API response dictionary
        output_format: "text" or "json"
    """
    if output_format == "json":
        print_json(result)
        return

    is_valid = result.get("is_valid", False)
    errors = result.get("errors", [])
    warnings = result.get("warnings", [])

    content = Text()

    if is_valid:
        content.append("[x] Valid HED string\n", style="bold green")
        if result.get("parsed_string"):
            content.append("\nNormalized form:\n", style="dim")
            content.append(f"  {result['parsed_string']}", style="dim")
    else:
        content.append("[ ] Invalid HED string\n", style="bold red")

    if errors:
        content.append("\n\nErrors:\n", style="bold red")
        for e in errors:
            content.append(f"  - {e}\n", style="red")

    if warnings:
        content.append("\n\nWarnings:\n", style="bold yellow")
        for w in warnings:
            content.append(f"  - {w}\n", style="yellow")

    status_style = "green" if is_valid else "red"
    console.print(
        Panel(
            content,
            title=f"[bold {status_style}]HED Validation[/]",
            border_style=status_style,
        )
    )


def print_config(config: dict[str, Any], show_key: bool = False) -> None:
    """Print current configuration.

    Args:
        config: Configuration dictionary
        show_key: Whether to show full API key (vs masked)
    """
    table = Table(title="HEDit Configuration", show_header=True, header_style="bold")
    table.add_column("Setting", style="cyan")
    table.add_column("Value")

    def add_section(section_name: str, section_data: dict) -> None:
        for key, value in section_data.items():
            full_key = f"{section_name}.{key}"
            # Mask API key unless explicitly requested
            if "key" in key.lower() and value and not show_key:
                display_value = f"{value[:8]}...{value[-4:]}" if len(str(value)) > 12 else "***"
            else:
                display_value = str(value) if value is not None else "[dim]not set[/]"
            table.add_row(full_key, display_value)

    for section_name, section_data in config.items():
        if isinstance(section_data, dict):
            add_section(section_name, section_data)

    console.print(table)


def print_error(message: str, hint: str | None = None) -> None:
    """Print an error message.

    Args:
        message: Error message
        hint: Optional hint for resolution
    """
    err_console.print(f"[bold red]Error:[/] {message}")
    if hint:
        err_console.print(f"[dim]Hint: {hint}[/]")


def print_success(message: str) -> None:
    """Print a success message."""
    err_console.print(f"[bold green]Success:[/] {message}")


def print_info(message: str) -> None:
    """Print an info message."""
    err_console.print(f"[dim]{message}[/]")


def print_progress(message: str) -> None:
    """Print a progress message to stderr (doesn't interfere with piped output)."""
    err_console.print(f"[dim]{message}...[/]")


def is_piped() -> bool:
    """Check if stdout is being piped."""
    return not sys.stdout.isatty()
