# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""README D-12 prototype: ``vpu_regfile`` as a memory with declared ports.

One ``compose.Memory`` ``vreg`` (32 rows of ``UInt(W)``, never reset) with
three read ports at latency 0 (asynchronous, D-13) and one write port whose
write is visible one edge later, and four units, each a separate kernel:
three readers (``rd_a``/``rd_b``/``rd_c``, one port each) and one writeback
unit (``wb``, the write port). ``src``/``sink`` replay MiniTPU's per-cycle
trace and record the responses, as every other regfile variant does.

The units bind PORTS (``memories=("vreg.ra",)``); ``compose`` checks one owner
per port, direction and accesses per iteration, and lowers the memory:
``Architecture.region(target, lowering={"vreg": ...})``. The record is
``dev/records/minitpu/u2_d12_prototype_2026-10-04.rst``.

Links: the unit ports are ``Wire`` (the read data ``comb``) -- MiniTPU's port
shape, for SystemC/Catapult; ``target="simulator"`` emits them as Streams.
"""

from __future__ import annotations

from allo.compose import Architecture, Channel, Memory, Port, unit
from allo.ir.types import UInt, int32, uint1  # noqa: F401  (names the bodies use)


@unit(
    memories=("RA", "RB", "RC", "WA", "WD", "WE"),
    writes=("ra", "rb", "rc", "wa", "wd", "we"),
    parameters=("N", "W"),
)
def src(
    xa: UInt(5)[N],
    xb: UInt(5)[N],
    xc: UInt(5)[N],
    xwa: UInt(5)[N],
    xwd: UInt(W)[N],
    xwe: uint1[N],
):
    for t in range(N):
        ra.put(xa[t])
        rb.put(xb[t])
        rc.put(xc[t])
        wa.put(xwa[t])
        wd.put(xwd[t])
        we.put(xwe[t])


@unit(memories=("vreg.ra",), reads=("ra",), writes=("qa",), parameters=("N",))
def rd_a(mem):
    for _ in range(N):
        a5: UInt(5) = ra.get()
        a: int32 = a5  # B4 workaround; B5: not in one step
        qa.put(mem[a])


@unit(memories=("vreg.rb",), reads=("rb",), writes=("qb",), parameters=("N",))
def rd_b(mem):
    for _ in range(N):
        b5: UInt(5) = rb.get()
        b: int32 = b5  # B4 workaround; B5: not in one step
        qb.put(mem[b])


@unit(memories=("vreg.rc",), reads=("rc",), writes=("qc",), parameters=("N",))
def rd_c(mem):
    for _ in range(N):
        c5: UInt(5) = rc.get()
        c: int32 = c5  # B4 workaround; B5: not in one step
        qc.put(mem[c])


@unit(memories=("vreg.w",), reads=("wa", "wd", "we"), parameters=("N", "W"))
def wb(mem):
    for _ in range(N):
        x5: UInt(5) = wa.get()
        x: int32 = x5  # B4 workaround; B5: not in one step
        d: UInt(W) = wd.get()
        e: uint1 = we.get()
        if e:
            mem[x] = d


@unit(memories=("QA", "QB", "QC"), reads=("qa", "qb", "qc"), parameters=("N", "W"))
def sink(ya: UInt(W)[N], yb: UInt(W)[N], yc: UInt(W)[N]):
    for t in range(N):
        ya[t] = qa.get()
        yb[t] = qb.get()
        yc[t] = qc.get()


VREG = Memory(
    "vreg",
    "UInt(W)",
    rows="32",
    ports=(
        Port("ra", "r", latency=0),
        Port("rb", "r", latency=0),
        Port("rc", "r", latency=0),
        Port("w", "w", visible=1),
    ),
    collision="refuse",
    reset=False,
    impl="registers",  # the ASIC form: one flop array, three combinational read muxes
)


def architecture(n, w=16, units=None, vreg=VREG):
    a5, dw = "UInt(5)", "UInt(W)"
    return Architecture(
        name="rf_d12",
        parameters={"N": n, "W": w},
        memories=(
            Memory("RA", f"{a5}[N]"),
            Memory("RB", f"{a5}[N]"),
            Memory("RC", f"{a5}[N]"),
            Memory("WA", f"{a5}[N]"),
            Memory("WD", f"{dw}[N]"),
            Memory("WE", "uint1[N]"),
            Memory("QA", f"{dw}[N]"),
            Memory("QB", f"{dw}[N]"),
            Memory("QC", f"{dw}[N]"),
            vreg,
        ),
        channels=(
            Channel("ra", a5, "2", kind="wire"),
            Channel("rb", a5, "2", kind="wire"),
            Channel("rc", a5, "2", kind="wire"),
            Channel("wa", a5, "2", kind="wire"),
            Channel("wd", dw, "2", kind="wire"),
            Channel("we", "uint1", "2", kind="wire"),
            Channel("qa", dw, "2", kind="comb"),
            Channel("qb", dw, "2", kind="comb"),
            Channel("qc", dw, "2", kind="comb"),
        ),
        units=units or (src, rd_a, rd_b, rd_c, wb, sink),
    )


def make(lowering, target):
    """A ``VARIANTS`` maker: ``make(n, w)`` -> the lowered region."""

    def f(n, w=16):
        # `replica` is FPGA-only (LUTRAM-style copies; asic_memories_2026-10-04.rst):
        # the harness asks for it by name, so it says so.
        tech = "fpga" if lowering == "replica" else None
        return architecture(n, w).region(target, {"vreg": lowering}, technology=tech)

    f.__name__ = f"d12_{lowering}_{target}"
    return f
