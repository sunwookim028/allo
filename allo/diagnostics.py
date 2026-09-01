# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared source-diagnostic rendering.

Both the compiler (:class:`allo.compiler.errors.CompilationError`) and the
scheduler (:class:`allo.schedule.errors.ScheduleError`) point at a source
location and render a ``file:line:col`` header with the offending line and a
caret underline. This module holds the one implementation they share.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from dataclasses import dataclass

from rich.console import Console
from rich.text import Text

from .errors import AlloError


@dataclass(frozen=True)
class DiagnosticLocation:
    file_name: str
    line: int
    col: int = 0
    source_line: str | None = None
    span: int = 1


def first_code_col(source_line: str | None) -> int:
    if source_line is None:
        return 0
    return len(source_line) - len(source_line.lstrip())


def diagnostic_color() -> bool:
    """Whether to emit ANSI color in rendered diagnostics.

    Color is only useful when the destination is a real terminal: Jupyter
    shows exception text as plain text, so escape codes would be garbled.
    """
    return Console(stderr=True).is_terminal


def render_diagnostic(
    message: str,
    location: DiagnosticLocation | None,
    *,
    notes: Sequence[str] = (),
    color: bool | None = None,
    width: int = 120,
) -> str:
    """Render a diagnostic to a string, with optional source context."""
    if color is None:
        color = diagnostic_color()

    console = Console(
        file=io.StringIO(),
        record=True,
        force_terminal=color,
        # Under Jupyter, rich would otherwise route print() to display(), a raised
        # diagnostic scatters across several output cells. Pin it non-Jupyter.
        force_jupyter=False,
        color_system="auto" if color else None,
        width=width,
    )

    if location is None:
        console.print(Text.assemble(("error", "bold red"), ": ", message))
    else:
        header = f"{location.file_name}:{location.line}:{location.col + 1}"
        console.print(
            Text.assemble((header, "bold"), ": ", ("error", "bold red"), ": ", message)
        )
        if location.source_line is not None:
            line_width = len(str(location.line))
            console.print(
                Text.assemble(
                    (f"{location.line:>{line_width}}", "bold cyan"),
                    " | ",
                    location.source_line,
                )
            )
            console.print(
                Text.assemble(
                    " " * line_width,
                    " | ",
                    " " * location.col,
                    ("^" * max(1, location.span), "bold green"),
                )
            )

    for note in notes:
        console.print(Text.assemble(("note", "bold cyan"), ": ", note))

    return console.export_text(styles=color).rstrip()


class DiagnosticError(AlloError):
    """Base for errors that render a ``file:line:col`` source diagnostic.

    Subclasses supply :meth:`_diagnostic`; the message/location/notes it
    returns are rendered by :func:`render_diagnostic` (wrapped at
    :attr:`render_width`). ``str(err)`` is the rendered diagnostic.
    """

    render_width: int = 120

    def _diagnostic(self) -> tuple[str, DiagnosticLocation | None, Sequence[str]]:
        raise NotImplementedError

    def render(self, *, color: bool | None = None) -> str:
        message, location, notes = self._diagnostic()
        return render_diagnostic(
            message, location, notes=notes, color=color, width=self.render_width
        )

    def __str__(self) -> str:
        return "\n" + self.render()
