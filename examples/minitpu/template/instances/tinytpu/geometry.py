# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""TinyTPU's parameter set as a geometry record (README D-20).

``ip/params.py``'s ``TpuParams`` already derives the memory sizes and the
packed-word widths from ``T`` and ``MAXDIM``; what it cannot say is WHICH
arithmetic the words carry: ``VW = T * 8`` and ``AW = T * 32`` are the int8
engine's widths written as literals. Here the record holds the engine
(``mac``, a ``compose.Engine``) and the widths follow from it:
``VW = T * mac.IN_BITS``, ``AW = T * mac.ACC_BITS``. A geometry bound to a
wider engine moves every packed word through this one relation.

The frozen ``TpuParams`` is still built (``params``) because the frozen
assembler, memory map and reference programs take one; its own assertions
(``T >= 4``, ``MAXDIM % T``, the 11-bit address ceiling) run at construction
and are not restated here.
"""

from __future__ import annotations

from dataclasses import dataclass

from allo.compose import Engine
from examples.minitpu.template.engines import INT8_INT32
from examples.tinytpu.ip.assembler import AR_RAW_DIST
from examples.tinytpu.ip.isa import ISA_NAMESPACE
from examples.tinytpu.ip.params import PARAMETER_NAMES, TpuParams


@dataclass(frozen=True)
class TinyTpuGeometry:
    """The declared numbers. Everything else is a property."""

    T: int = 4
    MAXDIM: int = 64
    QD: int = 16
    IMEM_SIZE: int = 56
    DMA_WORDS: int = 1
    mac: Engine = INT8_INT32
    #: The accumulator distance contract (``ip/assembler.py``): the engine's
    #: ``inter false`` claim on ``ar`` is true only for programs that keep an
    #: ``ar`` read this many ``accu`` steps after its write.
    AR_RAW_DIST: int = AR_RAW_DIST
    # `None` derives; an explicit value is a deliberately-sized build.
    spad_rows: int = None
    nvr: int = None
    nar: int = None

    def __post_init__(self):
        # The one relation the frozen assembler asserts (`Assembler.__init__`)
        # and nothing at composition time did: a T-row GEMM must satisfy the
        # accumulator distance contract, or the shipped program order cannot
        # be assembled at all.
        assert self.AR_RAW_DIST <= self.T, (
            f"AR_RAW_DIST={self.AR_RAW_DIST} > T={self.T}: a T-row GEMM "
            f"re-reads an ar row T accu steps after writing it, so no shipped "
            f"program meets the accumulator distance contract (README D-20)")
        assert self.mac.IN_BITS * self.T >= 32, (
            f"T={self.T} lanes of {self.mac.IN_BITS} bits: a packed operand "
            f"word must hold the two 16-bit counts spm sends down wcol")
        self.params  # TpuParams's own assertions, at construction

    @property
    def params(self) -> TpuParams:
        """The frozen parameter record the assembler and programs take."""
        return TpuParams(T=self.T, MAXDIM=self.MAXDIM, SPAD_ROWS=self.spad_rows,
                         NVR=self.nvr, NAR=self.nar, QD=self.QD,
                         IMEM_SIZE=self.IMEM_SIZE, DMA_WORDS=self.DMA_WORDS)

    @property
    def VW(self) -> int:
        """Packed operand word: T lanes of the engine's operand."""
        return self.T * self.mac.IN_BITS

    @property
    def AW(self) -> int:
        """Packed accumulator word: T lanes of the engine's accumulator."""
        return self.T * self.mac.ACC_BITS

    @property
    def WPR(self) -> int:
        return self.params.WPR

    @property
    def OPERAND_ROWS(self) -> int:
        return self.params.OPERAND_ROWS

    @property
    def SPAD_ROWS(self) -> int:
        return self.params.SPAD_ROWS

    @property
    def NVR(self) -> int:
        return self.params.NVR

    @property
    def NAR(self) -> int:
        return self.params.NAR

    def namespace(self) -> dict:
        """What the units bind. The ISA layout (``ip/isa.py``) rides in the
        same namespace because ``Architecture`` takes one parameter set
        (record gap G5): the opcode numbers are the instance's, not derived,
        and no property of this record names them."""
        ns = dict(ISA_NAMESPACE)
        ns.update({name: getattr(self, name) for name in PARAMETER_NAMES})
        ns["AR_RAW_DIST"] = self.AR_RAW_DIST
        return ns
