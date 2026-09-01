# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime logging and subprocess helpers.

All user-facing status, warnings and fatal errors funnel through this module
so the style and the process-exit behavior stay consistent across the backends.

Notebook mode (auto-detected under Jupyter, overridable via the
``ALLO_NOTEBOOK`` environment variable or :func:`set_notebook_mode`) renders to
stdout, replaces the live ``rich`` spinner -- which flickers as a widget in
notebooks -- with a static status line, and raises :class:`AlloFatalError`
instead of calling ``SystemExit`` so a stray failure does not stop the kernel.
"""

from __future__ import annotations

import os
import shlex
import subprocess

from collections.abc import Mapping, Sequence, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Callable, NoReturn, TypeVar, cast

from rich.console import Console
from rich.markup import escape

from .errors import AlloFatalError

F = TypeVar("F", bound=Callable[..., Any])
ErrorCallback = Callable[[Exception], None]
ExitCallback = Callable[[], None]

_FALSEY = frozenset({"", "0", "false", "no", "off"})

# ---------------------------------------------------------------------------
# Output destination (notebook-aware)
# ---------------------------------------------------------------------------

_notebook_override: bool | None = None


def in_notebook() -> bool:
    """Resolve whether Jupyter-friendly output is active.

    Precedence: an explicit :func:`set_notebook_mode` override, then the
    ``ALLO_NOTEBOOK`` environment variable, then ``rich``'s auto-detection.
    """
    if _notebook_override is not None:
        return _notebook_override
    env = os.getenv("ALLO_NOTEBOOK")
    if env is not None:
        return env.strip().lower() not in _FALSEY
    return Console().is_jupyter


def _make_console() -> Console:
    if in_notebook():
        # Write ANSI to stdout instead of emitting one rich display widget per
        # `print`. Jupyter coalesces consecutive stdout writes into a single
        # output block, so a cell's backend log lines (Compiling..., Running...)
        # render as one block rather than a stack that looks like separate cells.
        # `in_notebook()` still governs spinner-vs-static status independently.
        return Console(force_jupyter=False)
    # Jupyter renders stderr as a red error block, so terminals target stderr.
    return Console(stderr=True)


console = _make_console()


def set_notebook_mode(enabled: bool | None) -> None:
    """Force notebook output on/off; pass ``None`` to restore auto-detection."""
    global _notebook_override, console
    _notebook_override = enabled
    console = _make_console()


def _print(markup: str, **kwargs: Any) -> None:
    console.print(markup, **kwargs)


def _message(label: str, style: str, message: str, *, dim_body: bool = False) -> None:
    """Emit one ``Label body`` line, the single sink for all leveled logs."""
    text = message.rstrip()
    if not text:
        return
    body = f"[dim]{escape(text)}[/dim]" if dim_body else escape(text)
    _print(f"[{style}]{label}[/] {body}" if label else body)


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------


def _captured_output(stdout: str, stderr: str) -> str:
    return "\n".join(stream.rstrip() for stream in (stdout, stderr) if stream.strip())


def completed_output(result: subprocess.CompletedProcess[str]) -> str:
    return _captured_output(result.stdout or "", result.stderr or "")


def text_tail(text: str, max_lines: int) -> str:
    if max_lines <= 0:
        return ""
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])


def read_text_tail(path: str | os.PathLike[str], *, max_lines: int = 100) -> str:
    try:
        return text_tail(
            Path(path).read_text(encoding="utf-8", errors="replace"),
            max_lines,
        )
    except OSError:
        return ""


# ---------------------------------------------------------------------------
# Errors and process exit
# ---------------------------------------------------------------------------


@dataclass
class CommandError(RuntimeError):
    cmd: Sequence[str | os.PathLike[str]]
    returncode: int
    cwd: str | os.PathLike[str] | None = None
    stdout: str = ""
    stderr: str = ""

    @property
    def output(self) -> str:
        return _captured_output(self.stdout, self.stderr)

    def output_tail(self, max_lines: int) -> str:
        return text_tail(self.output, max_lines)

    def __str__(self) -> str:
        message = (
            f"Command failed with exit code {self.returncode}: "
            f"{shlex.join(os.fspath(arg) for arg in self.cmd)}"
        )
        if self.cwd is not None:
            message += f"\nWorking directory: {self.cwd}"
        return message


def _abort(label: str, message: str, *, exit_code: int = 1) -> NoReturn:
    """The single fatal-exit path shared by :func:`terminate`/:func:`log_fatal`."""
    text = message.strip() or label
    if in_notebook():
        raise AlloFatalError(text, exit_code=exit_code) from None
    _print(f"[red]{label}[/] {escape(text)}")
    raise SystemExit(exit_code) from None


def terminate(error: Exception, *, exit_code: int = 1) -> NoReturn:
    reason = str(error) or error.__class__.__name__
    _abort("Error", reason, exit_code=exit_code)


def log_fatal(message: str) -> NoReturn:
    _abort("Fatal", message)


def terminate_on_error(func: F) -> F:
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as error:
            terminate(error)

    return cast(F, wrapper)


# ---------------------------------------------------------------------------
# Leveled logging
# ---------------------------------------------------------------------------


def log_detail(message: str) -> None:
    _message("", "", message, dim_body=True)


def log_info(message: str) -> None:
    _message("Info", "dim", message, dim_body=True)


def log_warning(message: str) -> None:
    _message("Warning", "yellow", message)


def log_debug(message: str) -> None:
    if os.getenv("ALLO_DEBUG") is not None:
        _message("Debug", "dim", message, dim_body=True)


def log_tail(title: str, text: str, *, max_lines: int = 100) -> None:
    tail = text_tail(text, max_lines)
    if tail:
        log_detail(f"{title} (last {max_lines} lines):\n{tail}")


# ---------------------------------------------------------------------------
# Staged execution
# ---------------------------------------------------------------------------


@contextmanager
def _status(name: str) -> Generator[None, None, None]:
    # The live spinner is a flickering widget under Jupyter; fall back to a
    # plain status line there.
    if in_notebook():
        _message("Running", "cyan", name)
        yield
    else:
        with console.status(f"[cyan]{escape(name)}[/]", spinner="dots"):
            yield


@contextmanager
def stage(
    name: str,
    *,
    on_error: ErrorCallback | None = None,
    on_exit: ExitCallback | None = None,
) -> Generator[None, None, None]:
    try:
        with _status(name):
            yield
    except Exception as error:
        _print(f"[red]Fail[/] {escape(name)}")
        if on_error is not None:
            on_error(error)
        if on_exit is not None:
            on_exit()
        terminate(error)
    else:
        _print(f"[green]Success[/] {escape(name)}")
        if on_exit is not None:
            on_exit()


def run_command(
    cmd: Sequence[str | os.PathLike[str]],
    *,
    cwd: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
    stage_name: str | None = None,
) -> subprocess.CompletedProcess[str]:
    def invoke() -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            [os.fspath(arg) for arg in cmd],
            cwd=cwd,
            env=dict(env) if env is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise CommandError(
                cmd=cmd,
                cwd=cwd,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        return result

    if stage_name is None:
        return invoke()

    with stage(stage_name):
        return invoke()
