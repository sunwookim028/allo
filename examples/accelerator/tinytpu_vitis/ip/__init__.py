# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A library of parametrized TPU units, and the machinery to compose them.

``ip.units`` holds the units; ``ip.compose`` turns a set of them plus a wiring
into an Allo dataflow region; ``ip.tinytpu`` is one architecture this library
instantiates. Nothing in ``ip.units`` imports a parameter, an opcode or a
channel -- the architecture supplies them -- so the same unit composes into a
different machine, or the same machine at a different size, unedited.
"""

from .assembler import Assembler, ProgramError
from .compose import Architecture, Channel, Memory, Unit, unit
from .params import TpuParams
from .programs import GemmPrograms, MemoryMap
from .reduce import DotTree, ReduceParams
from .tinytpu import TinyTPU, architecture

__all__ = ["Architecture", "Assembler", "Channel", "DotTree", "GemmPrograms",
           "Memory", "MemoryMap", "ProgramError", "ReduceParams", "TinyTPU",
           "TpuParams", "Unit", "architecture", "unit"]
