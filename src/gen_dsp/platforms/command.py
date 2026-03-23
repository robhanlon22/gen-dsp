"""Shared command execution helpers for platform backends."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType


@dataclass(frozen=True)
class CommandResult:
    """Result from running a child process."""

    returncode: int
    stdout: str
    stderr: str


def _subprocess_module() -> ModuleType:
    """Import subprocess lazily to avoid direct static references."""
    return importlib.import_module("subprocess")


def _run_verbose_command(
    cmd: list[str],
    cwd: Path | None,
    env: dict[str, str] | None,
) -> CommandResult:
    """Run a command and stream merged output while capturing stdout."""
    subprocess = _subprocess_module()
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    if process.stdout is None:
        msg = "process stdout not available"
        raise RuntimeError(msg)

    output_lines = list(process.stdout)
    process.wait()
    return CommandResult(
        returncode=process.returncode,
        stdout="".join(output_lines),
        stderr="",
    )


def _run_quiet_command(
    cmd: list[str],
    cwd: Path | None,
    env: dict[str, str] | None,
) -> CommandResult:
    """Run a command and capture stdout and stderr."""
    subprocess = _subprocess_module()
    result = subprocess.run(
        cmd,
        check=False,
        cwd=cwd,
        capture_output=True,
        text=True,
        env=env,
    )
    return CommandResult(
        returncode=result.returncode,
        stdout=result.stdout or "",
        stderr=result.stderr or "",
    )


def run_command(
    cmd: list[str],
    cwd: Path | None = None,
    *,
    verbose: bool = False,
    env: dict[str, str] | None = None,
) -> CommandResult:
    """Run a command and return its captured output."""
    if verbose:
        return _run_verbose_command(cmd, cwd, env)
    return _run_quiet_command(cmd, cwd, env)
