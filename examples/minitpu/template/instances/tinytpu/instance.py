# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""TinyTPU-isa composed from the template: a base plus the accumulator option.

``base(geometry)`` is the machine without a result path: the sequencer, the
DMA loader, the scratchpad, the vector registers, the weight loaders and
the PE array at the geometry's engine. It is a DRAFT (``Architecture(draft=
True)``): its sequencer dispatches to the accumulator's queues, so alone it
fails the netlist rules naming ``c_acc``, and D-19 judges legality on the
composed result. ``architecture(geometry)`` is the
base with ``ACCUMULATOR`` composed in (README D-19), which is the frozen
``ip/tinytpu.py`` wiring unit for unit, channel for channel, in the same
declaration order (Vitis csim runs processes in it: limitations item 15).

``TinyTpuInstance`` is ``ip.tinytpu.TinyTPU``'s interface on the instance,
so the harness glue (``glue/microarch_isa.py``) can hand the frozen gates
either build. ``compare_isa`` is the ISA-slot comparison the track asks for:
``isa_slots(arch)`` against ``isa_spec.json``, with the module each opcode
belongs to DERIVED from the spec's own actions.
"""

from __future__ import annotations

import json
import os

from allo.compose import Architecture, Channel, Memory, isa_slots
from examples.minitpu.template.instances.tinytpu import accumulator
from examples.minitpu.template.instances.tinytpu.geometry import TinyTpuGeometry
from examples.minitpu.template.instances.tinytpu.pe import pe
from examples.tinytpu.ip.assembler import Assembler
from examples.tinytpu.ip.programs import GemmPrograms, MemoryMap
from examples.tinytpu.ip.units.dma_load import dma_ld
from examples.tinytpu.ip.units.scratchpad import spm
from examples.tinytpu.ip.units.sequencer import sequencer
from examples.tinytpu.ip.units.vector_regs import vru
from examples.tinytpu.ip.units.weight_loader import wld

SPEC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                    "..", "..", "tinytpu", "isa_spec.json")

#: The slots the base machine always has. ``dma_st`` (opcode 2) is retired --
#: the sequencer dispatches nothing for it and it behaves as ``nop`` -- and
#: stays a base slot because the spec still numbers it.
BASE_SLOTS = ("nop", "dma_ld", "dma_st", "vld", "loop", "endloop")


def base_channels():
    """``ip/tinytpu.py``'s channels less the option's three. The array's
    lanes are declared in the ENGINE's widths: ``acol``/``cw`` as T lanes of
    ``MAC_IN_BITS``/``MAC_OUT_BITS``, ``a_fwd``/``p_fwd`` as one ``MAC_IN``/
    ``MAC_ACC`` -- so a channel the PE moves an engine-typed value on is held
    to the engine by ``compose`` (D-15), where the frozen design wrote
    ``int8``/``int32``/``UInt(VW)``/``UInt(AW)``."""
    return (
        Channel("c_dld", "UInt(64)", "QD", carries="sequencer -> dma_ld"),
        Channel("c_spm", "UInt(64)", "QD", carries="sequencer -> spm"),
        Channel("c_vru", "UInt(64)", "QD", carries="sequencer -> vru"),
        Channel("dma2sp", "UInt(VW)", "QD", carries="dma_ld -> scratchpad"),
        Channel("dma2vr", "UInt(VW)", "QD", carries="dma_ld -> operand vregs"),
        Channel("sp2vr", "UInt(VW)", "QD", carries="scratchpad -> vregs (vld)"),
        Channel("wcol", "UInt(VW)", "QD", ("T",),
                "header + weight words, down column 0"),
        Channel("wrow", "UInt(VW)", "QD", ("T", "T"), "... then east along row i"),
        Channel("acol", depth="QD", shape=("T",), lanes="T", lane_bits="MAC_IN_BITS",
                carries="activation words, down column 0"),
        Channel("a_fwd", "MAC_IN", "QD", ("T", "T"), "one lane, east"),
        Channel("p_fwd", "MAC_ACC", "QD", ("T", "T"), "partial sums, south"),
        Channel("cw", depth="QD", shape=("T",), lanes="T", lane_bits="MAC_OUT_BITS",
                carries="bottom row packs T psums going east"),
        # Depth 4 is the shadow weight register: the next `mm`'s weight is
        # latched while this one computes (Gemmini's c1/c2 double buffer).
        Channel("wq", "UInt(32)", "4", ("T", "T"),
                "wld(i, j) -> pe(i, j): (weight lane, row count) per `mm`"),
    )


def base_memories():
    return (Memory("imem", "UInt(64)[IMEM_SIZE]"),
            Memory("A", "int8[MAXDIM * MAXDIM]"),
            Memory("B", "int8[MAXDIM * MAXDIM]"))


def base(geometry: TinyTpuGeometry = None, name="tinytpu_base") -> Architecture:
    geometry = geometry or TinyTpuGeometry()
    return Architecture(
        name=name, parameters=geometry,
        memories=base_memories(), channels=base_channels(),
        units=(sequencer, dma_ld, spm, vru, wld, pe),
        engines={"MAC": geometry.mac},
        # The column chain sums strictly sequentially in ascending
        # contraction index (isa_spec.json numerics.accumulate.order).
        order="sequential",
        slots=BASE_SLOTS,
        # The base is not a legal machine on its own: the sequencer
        # dispatches to `c_acc`/`c_dst`, which only the option reads. The
        # netlist rules run on the composed result (README D-19).
        draft=True)


def architecture(geometry: TinyTpuGeometry = None, name="tinytpu_isa") -> Architecture:
    geometry = geometry or TinyTpuGeometry()
    return Architecture.with_options(
        base(geometry), accumulator.option(geometry.AR_RAW_DIST), name=name)


class TinyTpuInstance:
    """One built instance: ``ip.tinytpu.TinyTPU``'s interface on the
    template composition, at one geometry."""

    def __init__(self, geometry: TinyTpuGeometry = None, name="tinytpu_isa"):
        self.geometry = geometry or TinyTpuGeometry()
        self.params = self.geometry.params
        self.memory_map = MemoryMap.of(self.params)
        self.architecture = architecture(self.geometry, name)
        self.assembler = Assembler(self.params, self.geometry.AR_RAW_DIST)
        self.programs = GemmPrograms(self.params, self.memory_map)

    @property
    def region(self):
        return self.architecture.region()

    def schedule(self, s):
        return self.architecture.directives(s)


# --- the ISA-slot comparison (README D-19; "ISA spec generalised to instances")

def spec_modules(spec) -> dict:
    """``{opcode: module}`` as the SPEC implies it: an opcode with an action
    at a unit the accumulator option brings belongs to that module; every
    other opcode is the base's. The spec has no module field of its own --
    that is gap G1 in the record -- so this is derived from ``actions``."""
    option_units = {"accu", "dma_st"}
    out = {}
    for o in spec["opcodes"]:
        units = {a["unit"] for a in o.get("actions", [])}
        out[o["name"]] = "accumulator" if units & option_units else "base"
    return out


def compare_isa(arch: Architecture, spec=None) -> dict:
    """``isa_slots(arch)`` against the spec's opcodes. Returns the slot
    tables and the differences, one list per kind, empty when they agree."""
    if spec is None:
        with open(SPEC, encoding="utf-8") as f:
            spec = json.load(f)
    ours = isa_slots(arch)
    theirs = spec_modules(spec)
    return {
        "instance": ours,
        "spec": theirs,
        "missing_in_instance": sorted(set(theirs) - set(ours)),
        "extra_in_instance": sorted(set(ours) - set(theirs)),
        "module_differs": sorted(s for s in set(ours) & set(theirs)
                                 if ours[s] != theirs[s]),
    }
