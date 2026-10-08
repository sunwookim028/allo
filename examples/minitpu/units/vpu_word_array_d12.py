# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""README D-12 on ``vpu_word_array`` (VMEM): two ``rw`` ports, two owners.

One ``compose.Memory`` ``vmem`` (``words`` x ``UInt(W)``, never reset) with a
compute port (``rw``, read latency 3) and a DMA port (``rw``, read latency 2),
each write visible one edge later, and a same-word cross-port write an
obligation of the composition (MiniTPU issue #21). Two units, one per port:
``port_c`` and ``port_d``. Each makes one access per cycle: it reads the
addressed word every cycle (S6: unconditional; a read the RTL does not make is
a masked slot) and writes it under ``if en & we`` -- one ``rw`` access, one
address. ``compose`` lowers the memory to a server kernel holding the storage
and the two read pipes (the pipe written as data, as the U2 record's ``wire``
form). ``dev/records/minitpu/u2_d12_prototype_2026-10-04.rst``.
"""

from __future__ import annotations

import os

from allo.compose import Architecture, Channel, Memory, Port, Sram, unit
from allo.ir.types import UInt, int32, uint1  # noqa: F401  (names the bodies use)

# The SRAM macros the `sram` lowering has (asic_memories_2026-10-04.rst): OpenRAM
# 2RW, FreePDK45, kept with the record. `small` (32 x 64 b) is the dry run, `w512`
# (512 x 64 b) the largest built and the bank macro; `mid` (4,096 x 64 b) is eight
# `w512` banks (s.5: one 4,096-word macro does not build in hours). Any other
# instance stays on registers unless the caller passes its own Sram.
_REC = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "dev", "records", "minitpu",
    "asic_memories_2026-10-04", "openram")
SRAMS = {
    "small": lambda: Sram.from_openram(
        os.path.join(_REC, "sram_2rw_64x32_freepdk45.v"),
        os.path.join(_REC, "sram_2rw_64x32_freepdk45_TT_1p0V_25C.lib"),
        catapult_lib=os.environ.get("MINITPU_SRAM_SMALL_CATAPULT_LIB"),
    ),
    "w512": lambda: Sram.from_openram(
        os.path.join(_REC, "sram_2rw_64x512_freepdk45.v"),
        os.path.join(_REC, "sram_2rw_64x512_freepdk45_TT_1p0V_25C.lib"),
        catapult_lib=os.environ.get("MINITPU_SRAM_W512_CATAPULT_LIB"),
    ),
    "mid": lambda: SRAMS["w512"]().banked(8).mapped(c=0, d=1),
}


def _geom(inst):
    from examples.minitpu.units.vpu_word_array import GEOM  # noqa: PLC0415 (cycle)

    return GEOM[inst]


@unit(
    memories=("CE", "CW", "CA", "CD", "DE", "DW", "DA", "DD"),
    writes=("ce", "cw", "ca", "cd", "de", "dw", "da", "dd"),
    parameters=("N", "W", "AW"),
)
def src(
    xce: uint1[N],
    xcw: uint1[N],
    xca: UInt(AW)[N],
    xcd: UInt(W)[N],
    xde: uint1[N],
    xdw: uint1[N],
    xda: UInt(AW)[N],
    xdd: UInt(W)[N],
):
    for t in range(N):
        ce.put(xce[t])
        cw.put(xcw[t])
        ca.put(xca[t])
        cd.put(xcd[t])
        de.put(xde[t])
        dw.put(xdw[t])
        da.put(xda[t])
        dd.put(xdd[t])


@unit(
    memories=("vmem.c",),
    reads=("ce", "cw", "ca", "cd"),
    writes=("qc",),
    parameters=("N", "W", "AW"),
)
def port_c(mem):
    for _ in range(N):
        e: uint1 = ce.get()
        w: uint1 = cw.get()
        a_: UInt(AW) = ca.get()
        a: int32 = a_  # B4; B5
        d: UInt(W) = cd.get()
        qc.put(mem[a])
        ew: uint1 = e & w
        if ew:
            mem[a] = d


@unit(
    memories=("vmem.d",),
    reads=("de", "dw", "da", "dd"),
    writes=("qd",),
    parameters=("N", "W", "AW"),
)
def port_d(mem):
    for _ in range(N):
        e: uint1 = de.get()
        w: uint1 = dw.get()
        a_: UInt(AW) = da.get()
        a: int32 = a_  # B4; B5
        d: UInt(W) = dd.get()
        qd.put(mem[a])
        ew: uint1 = e & w
        if ew:
            mem[a] = d


# Port-kind probes (asic_memories_2026-10-04.rst s.6): a MiniTPU port whose
# accesses are narrowed to one direction, so it can sit on a macro's `r` or `w`
# port. NOT MiniTPU's VMEM (whose two ports both read and write); used only to
# ask Catapult whether a non-`rw` macro port reaches II=1. A read-only port
# reads every cycle and ignores `we`; a write-only port writes under `en & we`
# and answers 0 (its read slots are not the RTL's).
@unit(
    memories=("vmem.c",),
    reads=("ce", "cw", "ca", "cd"),
    writes=("qc",),
    parameters=("N", "W", "AW"),
)
def port_c_r(mem):
    for _ in range(N):
        e: uint1 = ce.get()
        w: uint1 = cw.get()
        a_: UInt(AW) = ca.get()
        a: int32 = a_  # B4; B5
        d: UInt(W) = cd.get()
        qc.put(mem[a])


@unit(
    memories=("vmem.c",),
    reads=("ce", "cw", "ca", "cd"),
    writes=("qc",),
    parameters=("N", "W", "AW"),
)
def port_c_w(mem):
    for _ in range(N):
        e: uint1 = ce.get()
        w: uint1 = cw.get()
        a_: UInt(AW) = ca.get()
        a: int32 = a_  # B4; B5
        d: UInt(W) = cd.get()
        z: UInt(W) = 0
        qc.put(z)
        ew: uint1 = e & w
        if ew:
            mem[a] = d


@unit(
    memories=("vmem.d",),
    reads=("de", "dw", "da", "dd"),
    writes=("qd",),
    parameters=("N", "W", "AW"),
)
def port_d_r(mem):
    for _ in range(N):
        e: uint1 = de.get()
        w: uint1 = dw.get()
        a_: UInt(AW) = da.get()
        a: int32 = a_  # B4; B5
        d: UInt(W) = dd.get()
        qd.put(mem[a])


@unit(
    memories=("vmem.d",),
    reads=("de", "dw", "da", "dd"),
    writes=("qd",),
    parameters=("N", "W", "AW"),
)
def port_d_w(mem):
    for _ in range(N):
        e: uint1 = de.get()
        w: uint1 = dw.get()
        a_: UInt(AW) = da.get()
        a: int32 = a_  # B4; B5
        d: UInt(W) = dd.get()
        z: UInt(W) = 0
        qd.put(z)
        ew: uint1 = e & w
        if ew:
            mem[a] = d


_PORT_UNITS = {("c", "rw"): port_c, ("c", "r"): port_c_r, ("c", "w"): port_c_w,
               ("d", "rw"): port_d, ("d", "r"): port_d_r, ("d", "w"): port_d_w}


@unit(memories=("QC", "QD"), reads=("qc", "qd"), parameters=("N", "W"))
def sink(yc: UInt(W)[N], yd: UInt(W)[N]):
    for t in range(N):
        yc[t] = qc.get()
        yd[t] = qd.get()


def architecture(n, inst="narrow", reset=False, impl=None, kinds=("rw", "rw")):
    """``impl``: the memory's declared lowering -- an ``Sram`` (the macro
    path), ``"registers"``, or None for the instance's macro from ``SRAMS``
    when it has one and registers otherwise. ``region(lowering=...)`` still
    overrides it (the ``d12_server`` variants do). ``kinds``: the compute and
    DMA ports' kinds; MiniTPU's are ``("rw", "rw")``, anything else is a
    port-kind probe (above)."""
    ww, words, aw, rl, drl = _geom(inst)
    if impl is None and inst in SRAMS:
        impl = SRAMS[inst]()
    ck, dk = kinds
    vmem = Memory(
        "vmem",
        "UInt(W)",
        rows=str(words),
        ports=(
            Port("c", ck, latency=None if ck == "w" else rl, visible=1),
            Port("d", dk, latency=None if dk == "w" else drl, visible=1),
        ),
        collision="obligation",
        reset=reset,
        impl=impl,
    )
    a, d = "UInt(AW)", "UInt(W)"
    return Architecture(
        name="wa_d12",
        parameters={"N": n, "W": ww, "AW": aw},
        memories=(
            Memory("CE", "uint1[N]"),
            Memory("CW", "uint1[N]"),
            Memory("CA", f"{a}[N]"),
            Memory("CD", f"{d}[N]"),
            Memory("DE", "uint1[N]"),
            Memory("DW", "uint1[N]"),
            Memory("DA", f"{a}[N]"),
            Memory("DD", f"{d}[N]"),
            Memory("QC", f"{d}[N]"),
            Memory("QD", f"{d}[N]"),
            vmem,
        ),
        channels=tuple(
            Channel(c, t, "2", kind="wire")
            for c, t in (
                ("ce", "uint1"),
                ("cw", "uint1"),
                ("ca", a),
                ("cd", d),
                ("de", "uint1"),
                ("dw", "uint1"),
                ("da", a),
                ("dd", d),
            )
        )
        + (Channel("qc", d, "2", kind="comb"), Channel("qd", d, "2", kind="comb")),
        units=(src, _PORT_UNITS["c", ck], _PORT_UNITS["d", dk], sink),
        obligations={
            "vmem": "MiniTPU issue #21: the program keeps compute and DMA "
            "off one word in one cycle (asserted in simulation only, "
            "vpu_vmem_simd.sv:121); masked in verdicts"
        },
    )


def make(target, reset=False, lowering="server"):
    """A ``VARIANTS`` maker. ``lowering="server"`` (alias of ``registers``):
    the flop form; ``"sram"``: the instance's macro (refused for an instance
    without one, naming it)."""

    def f(n, w=64, inst="narrow"):
        assert _geom(inst)[0] == w
        if lowering == "sram":
            assert inst in SRAMS, (
                f"vpu_word_array {inst}: no SRAM macro for this instance (SRAMS has "
                f"{sorted(SRAMS)}); asic_memories_2026-10-04.rst")
        return architecture(n, inst, reset).region(target, {"vmem": lowering})

    f.__name__ = f"d12_{lowering}_{target}"
    return f
