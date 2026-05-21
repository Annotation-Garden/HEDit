"""`hedit lsp` subcommands: start, stop, status.

These manage an optional persistent hed-lsp daemon. Standalone CLI use
is opt-in: if you run `hedit annotate` repeatedly, starting the daemon
amortizes the LSP boot cost across all invocations. The daemon is
*not* auto-spawned; users start and stop it explicitly.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from src.lsp.daemon import (
    default_runtime_dir,
    meta_file_path,
    pid_file_path,
    socket_file_path,
)
from src.validation.hed_lsp import find_lsp_server_js

app = typer.Typer(
    name="lsp",
    help="Manage the optional persistent hed-lsp daemon.",
    no_args_is_help=True,
)
console = Console()


def _is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_pid(runtime_dir: Path) -> int | None:
    pid_file = pid_file_path(runtime_dir)
    if not pid_file.exists():
        return None
    try:
        return int(pid_file.read_text().strip())
    except (OSError, ValueError):
        return None


ServerJsOption = Annotated[
    Path | None,
    typer.Option(
        "--server-js",
        help="Path to hed-lsp server.js (defaults to HED_LSP_SERVER_JS or auto-detect).",
    ),
]
RuntimeDirOption = Annotated[
    Path | None,
    typer.Option(
        "--runtime-dir",
        help="Override the per-user runtime dir for the socket and PID file.",
    ),
]


@app.command("start")
def start(
    server_js: ServerJsOption = None,
    runtime_dir: RuntimeDirOption = None,
) -> None:
    """Start the persistent hed-lsp daemon (opt-in)."""
    rt_dir = runtime_dir or default_runtime_dir()
    existing_pid = _read_pid(rt_dir)
    if existing_pid is not None and _is_running(existing_pid):
        console.print(f"[yellow]hedit-lspd already running (pid={existing_pid})[/yellow]")
        raise typer.Exit(code=0)

    resolved_server_js = server_js or find_lsp_server_js()
    if resolved_server_js is None or not Path(resolved_server_js).exists():
        console.print(
            "[red]Could not locate hed-lsp server.js. "
            "Set HED_LSP_SERVER_JS or pass --server-js.[/red]"
        )
        raise typer.Exit(code=1)

    rt_dir.mkdir(parents=True, exist_ok=True)
    log_path = rt_dir / "lspd.log"
    cmd = [
        sys.executable,
        "-m",
        "src.lsp.daemon",
        "--server-js",
        str(resolved_server_js),
        "--runtime-dir",
        str(rt_dir),
    ]
    with log_path.open("ab") as log:
        proc = subprocess.Popen(  # noqa: S603
            cmd,
            stdout=log,
            stderr=log,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    # Wait briefly for the socket file to appear, indicating readiness.
    socket = socket_file_path(rt_dir)
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        if socket.exists():
            console.print(
                f"[green]hedit-lspd started[/green] "
                f"(pid={proc.pid}, socket={socket}, log={log_path})"
            )
            return
        if proc.poll() is not None:
            console.print(
                f"[red]hedit-lspd exited with code {proc.returncode}; see {log_path}[/red]"
            )
            raise typer.Exit(code=1)
        time.sleep(0.1)

    console.print(f"[red]Timed out waiting for hedit-lspd socket at {socket}[/red]")
    raise typer.Exit(code=1)


@app.command("stop")
def stop(runtime_dir: RuntimeDirOption = None) -> None:
    """Stop the persistent hed-lsp daemon."""
    rt_dir = runtime_dir or default_runtime_dir()
    pid = _read_pid(rt_dir)
    if pid is None:
        console.print("[yellow]hedit-lspd is not running.[/yellow]")
        raise typer.Exit(code=0)
    if not _is_running(pid):
        console.print(
            f"[yellow]Stale PID file at {pid_file_path(rt_dir)} (pid={pid}); cleaning up.[/yellow]"
        )
        for path in (pid_file_path(rt_dir), socket_file_path(rt_dir), meta_file_path(rt_dir)):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        return

    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if not _is_running(pid):
            console.print(f"[green]hedit-lspd stopped (pid={pid}).[/green]")
            return
        time.sleep(0.1)
    console.print(f"[red]hedit-lspd (pid={pid}) did not stop within 10s; sending SIGKILL.[/red]")
    os.kill(pid, signal.SIGKILL)


@app.command("status")
def status(runtime_dir: RuntimeDirOption = None) -> None:
    """Show daemon status (running pid, uptime, socket path)."""
    rt_dir = runtime_dir or default_runtime_dir()
    pid = _read_pid(rt_dir)
    if pid is None or not _is_running(pid):
        console.print("[yellow]hedit-lspd is not running.[/yellow]")
        if pid is not None:
            console.print(f"  Stale PID file at {pid_file_path(rt_dir)} (pid={pid})")
        raise typer.Exit(code=1)

    console.print(f"[green]hedit-lspd running (pid={pid})[/green]")
    console.print(f"  Socket:   {socket_file_path(rt_dir)}")
    meta_path = meta_file_path(rt_dir)
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            console.print(f"  Node PID: {meta.get('node_pid')}")
            console.print(f"  server.js: {meta.get('server_js')}")
            console.print(f"  Started:  {meta.get('started_at')}")
        except (OSError, json.JSONDecodeError) as exc:
            console.print(f"  [dim](could not read metadata: {exc})[/dim]")
