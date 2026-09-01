# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exceptions raised by the DSA frontend.

The line these draw is the one ``AGENTS.md`` asks for: ``assert`` is for invariants
that hold **by design** and whose failure means the frontend itself is broken;
anything that depends on *user* input — an ISA description, a hand-written
instruction stream, a source program — raises instead. That distinction is not
stylistic. Every check below used to be an ``assert``, so ``python -O`` deleted the
whole validation layer at once: an instruction that did not fit was no longer
rejected, ``_fit`` (which selects candidates by catching the rejection) reported that
*everything* fit, and compilation continued on a program that had failed.

Three sources of error, three exception families:

- :class:`AcceleratorDescriptionError` — the ISA description is wrong or uses
  something the frontend does not support. Raised while declaring or tracing an
  instruction, never while compiling a program.
- :class:`AssemblyError` — a hand-written instruction stream (an ``@oracle`` body, an
  ``@I.expand`` body, a call into a ``CompiledProgram``) is invalid.
- :class:`CompileError` — a source program cannot be compiled onto this ISA. Its
  subclasses name the stage that refused it, which is also what ``search._fit``
  catches when it probes whether a candidate instruction fits.
"""


class DSAError(Exception):
    """Base class for every error the DSA frontend raises."""


class AcceleratorDescriptionError(DSAError):
    """The ISA description is invalid, inconsistent, or uses an unsupported feature."""


class AssemblyError(DSAError):
    """A hand-written instruction stream is invalid (bad operands, bad call site)."""


class CompileError(DSAError):
    """A source program cannot be compiled onto this ISA."""


class NoMatchError(CompileError):
    """Stage 1: no instruction computes some part of the source program."""


class ShapeError(CompileError):
    """Stage 2: an instruction's shape parameters cannot be solved for this source."""


class DTypeError(CompileError):
    """Stage 2: an instruction's datapath cannot hold the source's element type."""


class LayoutError(CompileError):
    """Stage 2: two accesses of one value disagree on how it is laid out."""


class AllocationError(CompileError):
    """Stage 3: the program cannot be placed — capacity, routing, or spilling."""
