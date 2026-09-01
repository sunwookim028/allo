# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Allo DSA frontend + search backend.

Describe an accelerator ISA — its memory hierarchy and instructions, each with a
traced access region and a traced compute region — and get two things from it:
the ``allo.buffer`` / ``allo.define`` catalog (``ISA.catalog``), and a compiler
from a TOSA source program onto that ISA (``ISA.compile_program``).
"""

from . import access, errors, primitive
from .access import collapse, contiguous, expand, layout, strided, view
from .core import ISA, BufferKind, BufferSpec, Instruction, InstructionSpec
from .epoch import Config, Dep, Epoch, Region, Schedule, Sigma
from .errors import (
    AcceleratorDescriptionError,
    AllocationError,
    AssemblyError,
    CompileError,
    DSAError,
    DTypeError,
    LayoutError,
    NoMatchError,
    ShapeError,
)
from .oracle import OracleConfig

__all__ = [
    "ISA",
    "BufferSpec",
    "Instruction",
    "InstructionSpec",
    "OracleConfig",
    "BufferKind",
    # the denotational layer (epoch.py): a compiled program as epochs, with σ
    "Config",
    "Dep",
    "Epoch",
    "Region",
    "Schedule",
    "Sigma",
    "access",
    "errors",
    "primitive",
    "strided",
    "expand",
    "collapse",
    "contiguous",
    "view",
    "layout",
    # errors: one base, three families (see errors.py)
    "DSAError",
    "AcceleratorDescriptionError",
    "AssemblyError",
    "CompileError",
    "NoMatchError",
    "ShapeError",
    "DTypeError",
    "LayoutError",
    "AllocationError",
]
