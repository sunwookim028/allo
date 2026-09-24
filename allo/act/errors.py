# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Provenance: cherry-picked from Kai Shao's ACT work --
# https://github.com/kkkaishao/allo, branch ``act``, commit ``3c1ad38``,
# file ``allo/exp/dsa/errors.py``. See ``ATTRIBUTION.md``.

"""Exceptions raised when a source program is recognized or compiled.

The line these draw is the one ``AGENTS.md`` asks for: ``assert`` is for invariants
that hold **by design** and whose failure means the frontend itself is broken;
anything that depends on *user* input -- a machine description, a hand-written
instruction stream, a source program -- raises instead. That distinction is not
stylistic. Every check in Kai's original used to be an ``assert``, so ``python -O``
deleted the whole validation layer at once: an instruction that did not fit was no
longer rejected, the candidate filter (which selects by catching the rejection)
reported that *everything* fit, and compilation continued on a program that had
failed.

Three sources of error, three exception families:

- :class:`AcceleratorDescriptionError` -- the machine description is wrong or uses
  something the frontend does not support. Raised while declaring a machine, never
  while compiling a program.
- :class:`AssemblyError` -- a hand-written instruction stream is invalid.
- :class:`CompileError` -- a source program cannot be compiled onto this machine.
  Its subclasses name the stage that refused it.

Only :class:`CompileError` and its subclasses are raised by ``allo/act/`` today:
the recognizer (:mod:`allo.act.recognize`) and the TOSA frontend
(:mod:`allo.act.frontend`) refuse with :class:`NoMatchError`, :class:`ShapeError`
and :class:`DTypeError`. The other two families are kept so the taxonomy stays
whole as the rest of the flow arrives; ``allo.act.workload.SpecError`` predates
this file and is unrelated (it guards a hand-written :class:`~allo.act.workload.
Workload`, not a source program).
"""


class ACTError(Exception):
    """Base class for every error this frontend raises."""


class AcceleratorDescriptionError(ACTError):
    """The machine description is invalid, inconsistent, or uses an unsupported feature."""


class AssemblyError(ACTError):
    """A hand-written instruction stream is invalid (bad operands, bad call site)."""


class CompileError(ACTError):
    """A source program cannot be compiled onto this machine."""


class NoMatchError(CompileError):
    """Stage 1: no instruction computes some part of the source program."""


class ShapeError(CompileError):
    """Stage 2: an instruction's shape parameters cannot be solved for this source."""


class DTypeError(CompileError):
    """Stage 2: an instruction's datapath cannot hold the source's element type."""


class LayoutError(CompileError):
    """Stage 2: two accesses of one value disagree on how it is laid out."""


class AllocationError(CompileError):
    """Stage 3: the program cannot be placed -- capacity, routing, or spilling."""


class QuantizationError(CompileError):
    """The source's quantization is not one this machine can execute.

    Fork-local, not in Kai's taxonomy: his path has no integer corpus, so a
    quantized op was simply unrecognized. Ours is int8/int32, so the *reason* an
    op is refused is load-bearing -- a non-zero zero-point is a corpus constraint
    (see :mod:`allo.act.recognize`), and ``tosa.rescale`` is a hardware opcode
    that has been approved and not yet built.
    """
