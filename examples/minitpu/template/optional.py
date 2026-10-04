# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional modules as deltas over an architecture (Q3 / H15).

Today an optional module is a Python conditional over ``Architecture.units``
and ``channels``. That is enough to build either variant and not enough to
build either variant *safely*: the SFU sits between the ALU and the
writeback, so leaving it out must also rewire the writeback's input, and
the ISA slot that dispatched to it must stop assembling. Two hazards, both
silent in the conditional form: a channel left declared with one endpoint,
and a program that names a unit the machine does not have.

An ``Option`` names everything one module brings: units, channels, the
rebinding of a neighbour's port when the module is present, and the ISA
slots that exist only with it. ``assemble(base, *options)`` composes an
``Architecture`` from a base and its options and lets ``compose``'s netlist
checks (``nothing reads``/``nothing writes``) fall where they fall; the
``Option.isa`` slots are what an assembler is allowed to emit.

The demonstration rig is one VPU lane: ``issue -> alu -> writeback``, with
``sfu`` optional between ``alu`` and ``writeback``. The units are stubs
(the ALU and SFU bodies are track A's); the mechanism is the point.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from allo.compose import Architecture, Channel, Memory, Unit, unit

from examples.minitpu.template.instantiate import instance


@dataclass(frozen=True)
class Base:
    name: str
    parameters: dict
    memories: tuple
    channels: tuple
    units: tuple
    isa: tuple = ()           # the slots the base machine always has


@dataclass(frozen=True)
class Option:
    """One optional module: what it adds and what it rebinds."""

    name: str
    units: tuple = ()
    channels: tuple = ()
    rebind: dict = field(default_factory=dict)   # {unit name: {free name: new name}}
    isa: tuple = ()           # the slots that exist only with this module
    parameters: dict = field(default_factory=dict)


def assemble(base: Base, *options: Option, name=None):
    """The architecture of ``base`` with ``options`` present, and the ISA
    slots it may assemble. ``compose`` refuses a channel with a missing
    endpoint, so an option that forgets a rebind is refused here, not
    silently built."""
    units = list(base.units)
    channels = list(base.channels)
    params = dict(base.parameters)
    isa = list(base.isa)
    for opt in options:
        params.update(opt.parameters)
        channels += list(opt.channels)
        isa += list(opt.isa)
        for i, u in enumerate(units):
            if u.name in opt.rebind:
                units[i] = instance(u, u.name, opt.rebind[u.name])
        units += list(opt.units)
    arch = Architecture(name=name or f"{base.name}_{'_'.join(o.name for o in options) or 'base'}",
                        parameters=params, memories=base.memories,
                        channels=tuple(channels), units=tuple(units))
    return arch, tuple(isa)


def assemble_program(program, isa):
    """The one rule an assembler for an optional machine needs: a slot that
    is not in this instance's ISA is refused, naming the module."""
    for op in program:
        assert op in isa, (
            f"{op!r} is not an instruction of this instance: the module that "
            f"implements it is not composed in (available: {sorted(isa)})")
    return list(program)


# --- the VPU-lane rig ------------------------------------------------------

OP_ADD, OP_GELU = 0, 6


@unit(memories=("OPS", "A", "B"), writes=("to_alu",),
      parameters=("N_OPS",), isa=())
def issue(ops: UInt(32)[N_OPS], a_mem: UInt(32)[N_OPS], b_mem: UInt(32)[N_OPS]):
    for i in range(N_OPS):
        word: UInt(64) = 0
        word[0:16] = a_mem[i]
        word[16:32] = b_mem[i]
        word[32:36] = ops[i]
        to_alu.put(word)


@unit(reads=("to_alu",), writes=("alu_out",), parameters=("N_OPS", "OP_ADD"))
def alu(): # a stub: add is integer add, every other op passes `a` through
    for i in range(N_OPS):
        word: UInt(64) = to_alu.get()
        a: UInt(16) = word[0:16]
        b: UInt(16) = word[16:32]
        op: UInt(4) = word[32:36]
        r: UInt(16) = a
        if op == OP_ADD:
            r = a + b
        out: UInt(64) = word
        out[0:16] = r
        alu_out.put(out)


@unit(reads=("alu_out",), writes=("sfu_out",), parameters=("N_OPS", "OP_GELU"))
def sfu(): # a stub: GELU is `x ^ 0x5555` here; the real SFU is track A's S1
    for i in range(N_OPS):
        word: UInt(64) = alu_out.get()
        op: UInt(4) = word[32:36]
        r: UInt(16) = word[0:16]
        if op == OP_GELU:
            r = r ^ 0x5555
        out: UInt(64) = word
        out[0:16] = r
        sfu_out.put(out)


@unit(memories=("OUT",), reads=("alu_out",), parameters=("N_OPS",))
def writeback(out_mem: UInt(32)[N_OPS]):
    for i in range(N_OPS):
        word: UInt(64) = alu_out.get()
        out_mem[i] = word[0:16]


def vpu_lane_base(n_ops: int) -> Base:
    return Base(
        name="vpu_lane",
        parameters={"N_OPS": n_ops, "QD": 4, "OP_ADD": OP_ADD, "OP_GELU": OP_GELU},
        memories=(Memory("OPS", "UInt(32)[N_OPS]"), Memory("A", "UInt(32)[N_OPS]"),
                  Memory("B", "UInt(32)[N_OPS]"), Memory("OUT", "UInt(32)[N_OPS]")),
        channels=(Channel("to_alu", "UInt(64)", "QD"), Channel("alu_out", "UInt(64)", "QD")),
        units=(issue, alu, writeback),
        isa=("vadd",))


SFU_OPTION = Option(
    name="sfu",
    units=(sfu,),
    channels=(Channel("sfu_out", "UInt(64)", "QD"),),
    rebind={"writeback": {"alu_out": "sfu_out"}},
    isa=("vgelu",))

#: The conditional form: the SFU unit and its channel, but no rebind -- what a
#: `if with_sfu:` over the unit list would produce. compose must refuse it.
SFU_OPTION_NO_REBIND = Option(
    name="sfu_norebind", units=(sfu,),
    channels=(Channel("sfu_out", "UInt(64)", "QD"),), isa=("vgelu",))


def reference(ops, a, b, with_sfu):
    r = np.where(ops == OP_ADD, (a + b) & 0xFFFF, a)
    if with_sfu:
        r = np.where(ops == OP_GELU, r ^ 0x5555, r)
    return r


import numpy as np  # noqa: E402  (after the units: the body text is read by compose)


def run_lane(arch, n_ops, with_sfu, seed=0):
    import allo.dataflow as df
    rng = np.random.default_rng(seed)
    ops = rng.choice([OP_ADD, OP_GELU, 3], n_ops).astype(np.uint32)
    a = rng.integers(0, 1 << 16, n_ops).astype(np.uint32)
    b = rng.integers(0, 1 << 16, n_ops).astype(np.uint32)
    out = np.zeros(n_ops, dtype=np.uint32)
    df.build(arch.region(), target="simulator")(ops, a, b, out)
    return out, reference(ops, a, b, with_sfu)
