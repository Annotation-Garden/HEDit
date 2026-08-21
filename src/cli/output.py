"""Output formatting for HEDit CLI.

Uses Rich for beautiful terminal output with colors, tables, and panels.
"""

import json
import sys
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.status import Status
from rich.table import Table
from rich.text import Text

# Console for stdout (results)
console = Console()
# Console for stderr (status messages, errors)
err_console = Console(stderr=True)


def print_json(data: dict[str, Any]) -> None:
    """Print data as formatted JSON."""
    print(json.dumps(data, indent=2))


def _check(flag: bool) -> str:
    """Checkbox marker for a status flag, escaped for Rich markup."""
    return r"\[x]" if flag else r"\[ ]"


def _money(amount: float) -> str:
    """Format a dollar amount without rounding small costs away to zero."""
    return f"${amount:.6f}" if abs(amount) < 0.01 else f"${amount:.4f}"


def format_usage_lines(usage: dict[str, Any] | None) -> list[str]:
    """Render token, cost, and prompt-cache figures as display lines.

    Every HEDit annotation re-sends a large static HED vocabulary guide,
    which prompt caching serves at a tenth of the input price after the first
    call. Whoever owns the key is paying the bill, so the savings are worth
    reporting alongside the annotation.

    Args:
        usage: Usage summary from an API response or a local run

    Returns:
        Display lines, or an empty list when no LLM usage was reported
    """
    if not usage or not usage.get("calls"):
        return []

    calls = usage.get("calls", 0)
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    cached = usage.get("cache_read_tokens", 0)

    plural = "s" if calls != 1 else ""
    lines = [
        f"{input_tokens:,} input / {output_tokens:,} output tokens in {calls} LLM call{plural}"
    ]

    if cached:
        hit_rate = usage.get("cache_hit_rate") or 0.0
        lines.append(f"{cached:,} input tokens read from cache ({hit_rate:.0%} of input)")

    cost = usage.get("cost_usd")
    if cost is not None:
        savings = usage.get("savings_usd") or 0.0
        cost_line = _money(cost)
        if savings > 0:
            pct = usage.get("savings_pct") or 0.0
            cost_line += f", saved {_money(savings)} ({pct:.0%}) by prompt caching"
        elif cached:
            cost_line += ", no cache savings on this request"
        else:
            cost_line += ", nothing cached yet (the next run reuses this prompt)"
        unpriced = usage.get("unpriced_calls") or 0
        if unpriced:
            cost_line += f" (excludes {unpriced} call{'s' if unpriced != 1 else ''} with no price)"
        lines.append(cost_line)

    return lines


def _append_usage_section(content: Text, result: dict[str, Any]) -> None:
    """Append the usage and cache-savings section to a result panel."""
    lines = format_usage_lines(result.get("usage"))
    if not lines:
        return
    content.append("\n\nUsage and cache savings:\n", style="bold cyan")
    for line in lines:
        content.append(f"  {line}\n", style="cyan")


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
    status_parts.append(f"[{'green' if is_valid else 'red'}]{_check(is_valid)} Valid[/]")
    status_parts.append(
        f"[{'green' if is_faithful else 'yellow'}]{_check(is_faithful)} Faithful[/]"
    )
    if result.get("is_complete") is not None:
        is_complete = result.get("is_complete", False)
        status_parts.append(
            f"[{'green' if is_complete else 'yellow'}]{_check(is_complete)} Complete[/]"
        )

    attempts = result.get("validation_attempts", 0)
    status_parts.append(f"[dim]({attempts} validation attempt{'s' if attempts != 1 else ''})[/]")

    # Main panel
    content = Text()
    content.append("Annotation:\n", style="bold")
    content.append(f"  {annotation}\n\n")
    content.append("Status: ")
    # Status parts carry Rich markup, which Text.append would print literally.
    content.append_text(Text.from_markup(" ".join(status_parts)))

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

    _append_usage_section(content, result)

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
    status_parts.append(f"[{'green' if is_valid else 'red'}]{_check(is_valid)} Valid[/]")
    status_parts.append(
        f"[{'green' if is_faithful else 'yellow'}]{_check(is_faithful)} Faithful[/]"
    )

    content.append("Status: ")
    # Status parts carry Rich markup, which Text.append would print literally.
    content.append_text(Text.from_markup(" ".join(status_parts)))

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

    _append_usage_section(content, result)

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


@contextmanager
def streaming_status(initial_message: str = "Connecting...") -> Generator[Status, None, None]:
    """Context manager for streaming status updates.

    Yields a Status object that can be updated with new messages.
    The status spinner updates in place (single line).

    Example:
        with streaming_status("Starting...") as status:
            for event_type, data in stream:
                update_streaming_status(status, event_type, data)
    """
    with err_console.status(f"[dim]{initial_message}[/]", spinner="dots") as status:
        yield status


def update_streaming_status(status: Status, event_type: str, data: dict[str, Any]) -> None:
    """Update streaming status with event data.

    Args:
        status: Rich Status object from streaming_status()
        event_type: SSE event type (progress, validation, image_description, etc.)
        data: Event data dictionary
    """
    if event_type == "progress":
        message = data.get("message", "Processing...")
        attempt = data.get("attempt")

        if attempt:
            status.update(f"[dim]{message} (attempt {attempt})[/]")
        else:
            status.update(f"[dim]{message}[/]")

    elif event_type == "validation":
        valid = data.get("valid", False)
        attempt = data.get("attempt", 1)
        message = data.get("message", "")

        if valid:
            status.update("[green]Validation passed[/]")
        else:
            status.update(f"[yellow]Attempt {attempt}: {message}[/]")

    elif event_type == "image_description":
        # Show that image description was generated
        description = data.get("description", "")
        # Truncate long descriptions for status display
        if len(description) > 60:
            description = description[:57] + "..."
        status.update(f"[cyan]Image described: {description}[/]")


def is_piped() -> bool:
    """Check if stdout is being piped."""
    return not sys.stdout.isatty()
