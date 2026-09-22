# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""TinyTPU: one architecture the library composes, at any parameter set.

Eight units, sixteen channels, four memories. The channels are the design's
defining choice -- every distribution is a *chain* carrying packed words, never
a T-way fan-out, because a fan-out needs stream depth proportional to the whole
run and so never back-pressures. Why, and what it measured, is on
``docs/source/designs/tinytpu_isa.rst``.
"""

from examples.accelerator.tinytpu_vitis.ip.assembler import Assembler
from examples.accelerator.tinytpu_vitis.ip.compose import Architecture, Channel, Memory
from examples.accelerator.tinytpu_vitis.ip.isa import ISA_NAMESPACE
from examples.accelerator.tinytpu_vitis.ip.params import TpuParams
from examples.accelerator.tinytpu_vitis.ip.programs import GemmPrograms, MemoryMap
from examples.accelerator.tinytpu_vitis.ip.units.accumulator import accu
from examples.accelerator.tinytpu_vitis.ip.units.dma_load import dma_ld
from examples.accelerator.tinytpu_vitis.ip.units.dma_store import dma_st
from examples.accelerator.tinytpu_vitis.ip.units.pe import pe
from examples.accelerator.tinytpu_vitis.ip.units.scratchpad import spm
from examples.accelerator.tinytpu_vitis.ip.units.sequencer import sequencer
from examples.accelerator.tinytpu_vitis.ip.units.vector_regs import vru
from examples.accelerator.tinytpu_vitis.ip.units.weight_loader import wld


def channels():
    """Control first, then the data paths, then the array's chains."""
    return (
        Channel("c_dld", "UInt(64)", "QD", carries="sequencer -> dma_ld"),
        Channel("c_spm", "UInt(64)", "QD", carries="sequencer -> spm"),
        Channel("c_vru", "UInt(64)", "QD", carries="sequencer -> vru"),
        Channel("c_acc", "UInt(64)", "QD", carries="sequencer -> accu"),
        Channel("c_dst", "UInt(64)", "QD", carries="sequencer -> dma_st"),
        Channel("dma2sp", "UInt(VW)", "QD", carries="dma_ld -> scratchpad"),
        Channel("dma2vr", "UInt(VW)", "QD", carries="dma_ld -> operand vregs"),
        Channel("sp2vr", "UInt(VW)", "QD", carries="scratchpad -> vregs (vld)"),
        Channel("ac2sp", "UInt(VW)", "QD", carries="accumulator -> dma_st"),
        Channel("wcol", "UInt(VW)", "QD", ("T",),
                "header + weight words, down column 0"),
        Channel("wrow", "UInt(VW)", "QD", ("T", "T"), "... then east along row i"),
        Channel("acol", "UInt(VW)", "QD", ("T",),
                "activation words, down column 0"),
        Channel("a_fwd", "int8", "QD", ("T", "T"), "one lane, east"),
        Channel("p_fwd", "int32", "QD", ("T", "T"), "partial sums, south"),
        Channel("cw", "UInt(AW)", "QD", ("T",),
                "bottom row packs T psums going east"),
        # Depth 4 is the shadow weight register: the next `mm`'s weight is
        # latched while this one computes (Gemmini's c1/c2 double buffer).
        Channel("wq", "UInt(32)", "4", ("T", "T"),
                "wld(i, j) -> pe(i, j): (weight lane, row count) per `mm`"),
    )


def memories():
    """A, B and C are **flat**, addressed `row * MAXDIM + col`: that is what
    DRAM is, and `wrap_io=False` refuses multi-dimensional arguments to nested
    kernels (limitations register item 14)."""
    return (Memory("imem", "UInt(64)[IMEM_SIZE]"),
            Memory("A", "int8[MAXDIM * MAXDIM]"),
            Memory("B", "int8[MAXDIM * MAXDIM]"),
            Memory("C", "int8[MAXDIM * MAXDIM]"))


def units():
    """Declaration order is dataflow order, and Vitis csim runs the processes
    in it: a consumer declared before its producer reads an empty stream."""
    return (sequencer, dma_ld, spm, vru, wld, pe, accu, dma_st)


def architecture(params=None, name="tinytpu_isa"):
    params = params or TpuParams()
    namespace = dict(ISA_NAMESPACE)
    namespace.update(params.namespace())
    return Architecture(name=name, parameters=namespace,
                        memories=memories(), channels=channels(),
                        units=units())


class TinyTPU:
    """One built TinyTPU: the region, its Vitis directives, its assembler and
    its reference programs, all at the same parameter set."""

    def __init__(self, params=None, name="tinytpu_isa"):
        self.params = params or TpuParams()
        self.memory_map = MemoryMap.of(self.params)
        self.architecture = architecture(self.params, name)
        self.assembler = Assembler(self.params)
        self.programs = GemmPrograms(self.params, self.memory_map)

    @property
    def region(self):
        return self.architecture.region()

    def schedule(self, s):
        return self.architecture.directives(s)
