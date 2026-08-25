# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The 7-series fabric: Artix/Kintex/Zynq-7000"""

from __future__ import annotations

from collections.abc import Mapping

from ....lang.ip import OperatorIP
from ..device import (
    CombKind,
    Const,
    Device,
    Interp,
    Linear,
    Piecewise,
    Resource,
    Step,
    Table,
    Tiled,
)
from . import ip, vivado
from .spec import (
    MULTIWRITE_LUT_PER_BIT,
    MUX_LUT_COST,
    ROM_LUT_COST,
    SRL_MIN_DEPTH,
    Derived,
    FabricTiming,
    Grade,
    IPRow,
    Part,
    StorageSpec,
    StorageTiming,
    add_ip_rows,
)

NAME = "series7"

#: A slice mux combines LUT outputs and is counted apart from them: a SLICE
#: holds two MUXF7 to its four LUT6, so a die has half as many as it has LUTs.
DERIVED = {
    "carry4": Derived("lut", 4),
    "slicem_lut": Derived("lut", 2),
    "muxf": Derived("lut", 2),
}

GRADE_1 = Grade("-1", default_freq_mhz=100.0)


TIMING: Mapping[Grade, FabricTiming] = {
    GRADE_1: FabricTiming(
        comb={
            CombKind.ADD: Interp(
                {8: 2.325, 16: 2.480, 32: 2.936, 64: 3.848, 96: 4.836, 128: 5.650}
            ),
            CombKind.SUB: Interp(
                {8: 2.325, 16: 2.480, 32: 2.936, 64: 3.848, 96: 4.836, 128: 5.650}
            ),
            CombKind.MUL: Interp(
                {8: 5.083, 16: 5.618, 32: 9.297, 64: 13.319, 96: 15.762, 128: 18.207}
            ),
            CombKind.DIV: Interp({8: 17.031, 16: 38.884, 32: 96.829, 64: 232.4}),
            CombKind.REM: Interp({8: 18.471, 16: 39.683, 32: 98.933, 64: 237.4}),
            CombKind.NEG: Interp({32: 1.086, 64: 1.254}),
            CombKind.MIN: Interp(
                {8: 3.389, 16: 3.454, 32: 4.166, 64: 4.678, 96: 5.343, 128: 5.911}
            ),
            CombKind.MAX: Interp(
                {8: 3.459, 16: 3.641, 32: 4.030, 64: 5.007, 96: 5.379, 128: 5.907}
            ),
            CombKind.CMP: Interp(
                {8: 2.036, 16: 2.284, 32: 2.438, 64: 2.789, 96: 3.241, 128: 3.701}
            ),
            CombKind.AND: Interp(
                {8: 1.397, 16: 1.463, 32: 1.463, 64: 1.463, 96: 1.463, 128: 1.463}
            ),
            CombKind.OR: Interp(
                {8: 1.397, 16: 1.463, 32: 1.463, 64: 1.463, 96: 1.463, 128: 1.463}
            ),
            CombKind.XOR: Interp(
                {8: 1.397, 16: 1.463, 32: 1.463, 64: 1.463, 96: 1.463, 128: 1.463}
            ),
            CombKind.SHL: Interp(
                {8: 2.445, 16: 3.376, 32: 4.336, 64: 5.132, 96: 6.076, 128: 7.002}
            ),
            CombKind.SHR: Interp(
                {8: 2.847, 16: 3.685, 32: 4.568, 64: 6.255, 96: 6.255, 128: 6.619}
            ),
            CombKind.SELECT: Interp(
                {8: 1.736, 16: 1.982, 32: 2.088, 64: 2.088, 96: 2.224, 128: 2.224}
            ),
            CombKind.INT_CAST: Interp(
                {16: 1.216, 32: 1.216, 64: 1.510, 96: 1.510, 128: 1.510}
            ),
        },
        storage={
            "register": StorageTiming(0, 1, 1.086, 1.086),
            "lutram": StorageTiming(1, 1, 2.815, 3.449),
            "bram": StorageTiming(1, 1, 3.427, 1.390),
            # A table is never written; this row carries the read at the
            # reference shape, refined by `rom` below at the array's own.
            "rom": StorageTiming(1, 1, 5.658, 0.0),
        },
        stream=StorageTiming(0, 1, 2.815, 3.449),
        reg_ns=1.086,
        # One-hot select cones, routed, marginal over the register floor and
        # monotone over fan-in; the width factor is pinned to 1.0 at 32 bits.
        mux=Interp(
            {
                2: 1.123,
                3: 1.123,
                4: 2.610,
                8: 2.610,
                12: 3.264,
                16: 3.505,
                24: 4.575,
                40: 4.575,
            }
        ),
        mux_w=Interp({1: 0.53, 8: 0.73, 16: 0.83, 32: 1.0, 64: 1.75}),
        # A constant table's read, routed, over its depth at the same reference
        # width. It grows with the depth where an addressed row's read delay is
        # flat, which bounds how deep a table is worth building.
        rom=Interp(
            {
                64: 2.083,
                256: 4.401,
                512: 5.161,
                1024: 5.658,
                2048: 6.960,
                4096: 7.154,
                16384: 8.539,
            }
        ),
        rom_w=Interp({1: 0.57, 8: 0.92, 16: 0.97, 32: 1.0, 64: 1.25}),
    ),
}

SCATTER_STORAGE = "register"

#: No cone campaign has run on this fabric, so every row below carries these
#: three numbers rather than its own: an unmeasured entry cone, no output cone,
#: and the grade's characterization clock as the warranted period. They are
#: stated on each row rather than defaulted so a reader sees which rows are
#: measurements and which are placeholders. Replace them per row with
#: `drafts/char/measure_cones.py` run against a 7-series part.
_UNMEASURED = {
    "in_delay_ns": 0.5,
    "min_period_ns": 1000.0 / GRADE_1.default_freq_mhz,
    "out_delay_ns": 0.0,
}

#: Operator cores measured on this fabric, each inside a registered wrapper so
#: the number covers the whole path a caller sees. Several rows under one
#: archetype declare several cores for the library to choose between. ``lut`` is
#: logic sites only: the shift registers a core holds internally are split out
#: as ``slicem_lut``.
#:
#: The trailing comment on each row is that core's achieved Fmax in MHz from an
#: older campaign that recorded one worst-path number per core. It is the only
#: timing evidence this fabric has, and it is NOT what the rows declare: their
#: three delays are the `_UNMEASURED` placeholder above, which is why the two
#: disagree. Nothing reads the comment; a cone campaign against this fabric
#: replaces both.
IP: Mapping[OperatorIP, IPRow | tuple[IPRow, ...]] = {
    ip.fadd: (
        IPRow(
            7,
            {"lut": 252, "slicem_lut": 13, "ff": 238, "dsp": 2, "carry4": 19},
            **_UNMEASURED,
        ),  # 181
        IPRow(
            5,
            {"lut": 363, "slicem_lut": 13, "ff": 242, "carry4": 36},
            **_UNMEASURED,
            variant="nodsp",
        ),  # 157
    ),
    ip.fsub: (
        IPRow(
            7,
            {"lut": 252, "slicem_lut": 13, "ff": 238, "dsp": 2, "carry4": 19},
            **_UNMEASURED,
        ),  # 181
        IPRow(
            5,
            {"lut": 363, "slicem_lut": 13, "ff": 242, "carry4": 36},
            **_UNMEASURED,
            variant="nodsp",
        ),  # 157
    ),
    ip.fmul: IPRow(
        4,
        {"lut": 114, "slicem_lut": 1, "ff": 109, "dsp": 2, "carry4": 14},
        **_UNMEASURED,
    ),  # 214
    ip.fdiv: IPRow(
        12, {"lut": 760, "slicem_lut": 39, "ff": 477, "carry4": 194}, **_UNMEASURED
    ),  # 114
    ip.fcmp: IPRow(1, {"lut": 64, "ff": 2, "carry4": 12}, **_UNMEASURED),  # 194
    ip.dadd: (
        IPRow(
            14,
            {"lut": 721, "slicem_lut": 90, "ff": 872, "dsp": 3, "carry4": 51},
            **_UNMEASURED,
        ),  # 210
        IPRow(
            6,
            {"lut": 719, "slicem_lut": 16, "ff": 542, "carry4": 72},
            **_UNMEASURED,
            variant="nodsp",
        ),  # 169
    ),
    ip.dsub: (
        IPRow(
            14,
            {"lut": 721, "slicem_lut": 90, "ff": 872, "dsp": 3, "carry4": 51},
            **_UNMEASURED,
        ),  # 210
        IPRow(
            6,
            {"lut": 719, "slicem_lut": 16, "ff": 542, "carry4": 72},
            **_UNMEASURED,
            variant="nodsp",
        ),  # 169
    ),
    ip.dmul: IPRow(
        9,
        {"lut": 136, "slicem_lut": 36, "ff": 429, "dsp": 10, "carry4": 27},
        **_UNMEASURED,
    ),  # 213
    ip.ddiv: IPRow(
        32, {"lut": 3195, "slicem_lut": 72, "ff": 3027, "carry4": 794}, **_UNMEASURED
    ),  # 119
    ip.dcmp: IPRow(1, {"lut": 118, "ff": 2, "carry4": 21}, **_UNMEASURED),  # 182
    ip.bfadd: IPRow(
        4, {"lut": 175, "slicem_lut": 1, "ff": 118, "carry4": 24}, **_UNMEASURED
    ),  # 175
    ip.bfsub: IPRow(
        4, {"lut": 175, "slicem_lut": 1, "ff": 118, "carry4": 24}, **_UNMEASURED
    ),  # 175
    ip.bfmul: IPRow(
        2, {"lut": 60, "ff": 34, "dsp": 1, "carry4": 9}, **_UNMEASURED
    ),  # 185
    ip.i2f: IPRow(
        3, {"lut": 168, "slicem_lut": 1, "ff": 99, "carry4": 20}, **_UNMEASURED
    ),  # 163
    ip.f2i: IPRow(3, {"lut": 183, "ff": 127, "carry4": 11}, **_UNMEASURED),  # 186
    ip.fcvt: IPRow(2, {"lut": 50, "ff": 99, "carry4": 1}, **_UNMEASURED),  # 321
    ip.bf2f: IPRow(2, {"lut": 34, "ff": 53, "carry4": 1}, **_UNMEASURED),  # 363
    ip.imul16: (
        IPRow(4, {"ff": 16, "dsp": 1}, **_UNMEASURED),  # 514
        IPRow(1, {"dsp": 1}, **_UNMEASURED),  # 188
    ),
    ip.imul32: (
        IPRow(2, {"ff": 32, "dsp": 3}, **_UNMEASURED),  # 119
        IPRow(1, {"ff": 32, "dsp": 3}, **_UNMEASURED),  # 106
    ),
    ip.imul64: IPRow(6, {"slicem_lut": 64, "ff": 81, "dsp": 10}, **_UNMEASURED),  # 135
    ip.idiv8: IPRow(
        4, {"lut": 126, "slicem_lut": 1, "ff": 166, "carry4": 27}, **_UNMEASURED
    ),  # 102
    ip.udiv8: (
        IPRow(
            4, {"lut": 126, "slicem_lut": 1, "ff": 166, "carry4": 27}, **_UNMEASURED
        ),  # 102
        IPRow(2, {"lut": 110, "ff": 132, "carry4": 27}, **_UNMEASURED),  # 104
    ),
    ip.irem8: IPRow(
        4, {"lut": 126, "slicem_lut": 1, "ff": 166, "carry4": 27}, **_UNMEASURED
    ),  # 102
    ip.urem8: (
        IPRow(
            4, {"lut": 126, "slicem_lut": 1, "ff": 166, "carry4": 27}, **_UNMEASURED
        ),  # 102
        IPRow(2, {"lut": 110, "ff": 132, "carry4": 27}, **_UNMEASURED),  # 104
    ),
    ip.idiv16: IPRow(
        8, {"lut": 376, "slicem_lut": 2, "ff": 578, "carry4": 93}, **_UNMEASURED
    ),  # 103
    ip.udiv16: IPRow(
        8, {"lut": 376, "slicem_lut": 2, "ff": 578, "carry4": 93}, **_UNMEASURED
    ),  # 103
    ip.irem16: IPRow(
        8, {"lut": 376, "slicem_lut": 2, "ff": 578, "carry4": 93}, **_UNMEASURED
    ),  # 103
    ip.urem16: IPRow(
        8, {"lut": 376, "slicem_lut": 2, "ff": 578, "carry4": 93}, **_UNMEASURED
    ),  # 103
    ip.idiv32: IPRow(
        16, {"lut": 1264, "slicem_lut": 2, "ff": 2170, "carry4": 313}, **_UNMEASURED
    ),  # 102
    ip.udiv32: IPRow(
        16, {"lut": 1264, "slicem_lut": 2, "ff": 2170, "carry4": 313}, **_UNMEASURED
    ),  # 102
    ip.irem32: IPRow(
        16, {"lut": 1264, "slicem_lut": 2, "ff": 2170, "carry4": 313}, **_UNMEASURED
    ),  # 102
    ip.urem32: IPRow(
        16, {"lut": 1264, "slicem_lut": 2, "ff": 2170, "carry4": 313}, **_UNMEASURED
    ),  # 102
    ip.idiv64: IPRow(
        68, {"lut": 4608, "slicem_lut": 6, "ff": 12808, "carry4": 1137}, **_UNMEASURED
    ),  # 174
    ip.udiv64: (
        IPRow(
            68,
            {"lut": 4608, "slicem_lut": 6, "ff": 12808, "carry4": 1137},
            **_UNMEASURED,
        ),  # 174
        IPRow(
            66,
            {"lut": 4481, "slicem_lut": 2, "ff": 12677, "carry4": 1105},
            **_UNMEASURED,
        ),  # 186
    ),
    ip.irem64: IPRow(
        68, {"lut": 4608, "slicem_lut": 6, "ff": 12808, "carry4": 1137}, **_UNMEASURED
    ),  # 174
    ip.urem64: (
        IPRow(
            68,
            {"lut": 4608, "slicem_lut": 6, "ff": 12808, "carry4": 1137},
            **_UNMEASURED,
        ),  # 174
        IPRow(
            66,
            {"lut": 4481, "slicem_lut": 2, "ff": 12677, "carry4": 1105},
            **_UNMEASURED,
        ),  # 186
    ),
}


#: SLICEM sites one bit of a 64-deep distributed RAM occupies for one instance,
#: meaning a write port and one addressed read. Measured at the 32-bit
#: reference width as half of a two-read series (88 / 352 / 704 sites at
#: 64 / 256 / 512 x 32).
LUTRAM_SITES_PER_BIT = 1.375

_STORAGE = {
    "register": StorageSpec(
        ("lut", "ff"),
        lambda r: {
            r["lut"]: (Linear(MULTIWRITE_LUT_PER_BIT), Linear(1.0)),
            r["ff"]: (Linear(1.0), Linear(1.0)),
        },
    ),
    # Distributed RAM has one write port and one addressed read, in separate
    # structures (no pool). A second read address costs a further copy of the
    # array, charged as a further instance.
    "lutram": StorageSpec(
        ("slicem_lut",),
        lambda r: {r["slicem_lut"]: (Tiled(64), Linear(LUTRAM_SITES_PER_BIT))},
        inst_reads=1,
        inst_writes=1,
        ram_style="distributed",
    ),
    # A read-only table is logic, not storage: one LUT6 is a 64-entry one-bit
    # lookup, so it costs `width * ceil(depth/64)` of them and has no address
    # bus to contend for. Its read is the cone through those LUTs, so it is the
    # one row timed over the array's own shape.
    "rom": StorageSpec(
        ("lut",),
        lambda r: {r["lut"]: ROM_LUT_COST},
        is_table=True,
    ),
    # Two ports, each reading or writing in a cycle; two writers and a
    # concurrent reader together exceed the pool.
    "bram": StorageSpec(
        ("bram36",),
        lambda r: {r["bram36"]: Tiled(36864)},
        inst_reads=2,
        inst_writes=2,
        inst_ports=2,
        ram_style="block",
    ),
    "uram": StorageSpec(
        ("uram288",),
        lambda r: {r["uram288"]: Tiled(294912)},
        inst_reads=2,
        inst_writes=2,
        inst_ports=2,
        ram_style="ultra",
        can_init=False,
    ),
}


def _comb_uses(r: Mapping[str, Resource]) -> dict[CombKind, dict | None]:
    """What one instance of each native operator kind spends, over its operand
    width. ``None`` means free rather than unpriced: ``icast`` renames bits and
    ``neg`` flips a float sign, so neither reaches a cell the part charges for."""
    lut, dsp, carry4 = r["lut"], r["dsp"], r["carry4"]
    logic = {lut: Linear(1.0)}
    addsub = {lut: Linear(1.0), carry4: Tiled(4)}
    compare = {lut: Linear(1.0), carry4: Tiled(8)}
    minmax = {lut: Linear(2.0), carry4: Tiled(8)}
    shift = {lut: Interp({8: 15, 16: 44, 32: 107, 64: 265, 96: 427, 128: 573})}
    # The DSP count steps in whole slices; the fabric logic around them grows
    # with width. Past 64 bits the partial-product tree takes more carry chains
    # than `Tiled` charges: measured 57 against 16 at 128 bits.
    multiply = {
        lut: Interp({8: 39, 16: 0, 32: 15, 64: 41, 96: 153, 128: 316}),
        dsp: Table({8: 0, 16: 1, 32: 3, 64: 10, 96: 21, 128: 36}),
        carry4: Tiled(8),
    }
    divide = {
        lut: Interp({8: 125, 16: 397, 32: 1344, 64: 4731, 96: 10188, 128: 17575}),
        carry4: Interp({8: 21, 16: 92, 32: 312, 64: 1136, 96: 2472, 128: 4320}),
    }
    return {
        CombKind.AND: logic,
        CombKind.OR: logic,
        CombKind.XOR: logic,
        CombKind.SELECT: logic,
        CombKind.ADD: addsub,
        CombKind.SUB: addsub,
        CombKind.CMP: compare,
        CombKind.MIN: minmax,
        CombKind.MAX: minmax,
        CombKind.SHL: shift,
        CombKind.SHR: shift,
        CombKind.MUL: multiply,
        CombKind.DIV: divide,
        CombKind.REM: divide,
        CombKind.NEG: None,
        CombKind.INT_CAST: None,
    }


def _chain_uses(r: Mapping[str, Resource]) -> dict:
    """What one delay chain spends, over its depth and bit width."""
    per_stage = [
        (Linear(1.0, base=-1.0), Const(1.0)),
        (Piecewise(SRL_MIN_DEPTH, Linear(-1.0, base=1.0), Const(0.0)), Const(1.0)),
    ]
    return {
        r["ff"]: [(Step(SRL_MIN_DEPTH, 1.0, 2.0), Linear(1.0))] + per_stage,
        r["lut"]: (Step(SRL_MIN_DEPTH, 0.0, 1.0), Linear(1.0)),
        # An SRL32E holds 32 stages. `ceil(depth/32)` is a derivation, not a
        # measurement, and the UltraScale+ sweep contradicts it: the chain's
        # first and last stage stay in flip-flops, so the shift registers hold
        # `depth - 2`. Left as it stands until this fabric is swept.
        r["slicem_lut"]: (
            Piecewise(SRL_MIN_DEPTH, Const(0.0), Tiled(32)),
            Linear(1.0),
        ),
    }


def _chain_uses_norst(r: Mapping[str, Resource]) -> dict:
    """The same chain with no synchronous reset: the SRL absorbs every interior
    stage, so only the two end registers stay in flip-flops."""
    return {
        r["ff"]: (Step(SRL_MIN_DEPTH, 1.0, 2.0), Linear(1.0)),
        r["slicem_lut"]: (
            Piecewise(SRL_MIN_DEPTH, Const(0.0), Tiled(32)),
            Linear(1.0),
        ),
    }


def build(part: Part) -> Device:
    """The :class:`Device` for one 7-series die."""
    timing = TIMING.get(part.grade)
    if timing is None:
        raise ValueError(
            f"{NAME} has not been characterized at grade {part.grade.name!r}, so "
            f"{part.name!r} cannot be built; measure that grade and add it to "
            "TIMING rather than reading a neighbouring grade's delays"
        )
    d = Device(part.name, part=part.part, fabric=NAME, grade=part.grade.name)

    res = {name: d.add_resource(name, cap) for name, cap in part.capacity.items()}
    for name, derived in DERIVED.items():
        if derived.source in part.capacity:
            capacity = part.capacity[derived.source] // derived.divisor
            res[name] = d.add_resource(name, capacity)

    for name, t in timing.storage.items():
        spec = _STORAGE[name]
        if not all(n in res for n in spec.needs):
            continue
        if spec.is_table and timing.rom is None:
            continue  # unmeasured at this grade: no table can be timed here
        d.add_storage(
            name,
            read_latency=t.read_latency,
            write_latency=t.write_latency,
            read_delay_ns=t.read_ns,
            write_delay_ns=t.write_ns,
            is_scatter=name == SCATTER_STORAGE,
            is_table=spec.is_table,
            inst_reads=spec.inst_reads,
            inst_writes=spec.inst_writes,
            inst_ports=spec.inst_ports,
            ram_style=spec.ram_style,
            can_init=spec.can_init,
            uses=spec.uses(res),
            read_delay_depth=timing.rom if spec.is_table else None,
            read_delay_width=timing.rom_w if spec.is_table else None,
        )
    d.set_stream_timing(*timing.stream)

    comb = _comb_uses(res)
    for kind, delay in timing.comb.items():
        d.set_comb_delay(kind, delay, uses=comb[kind])

    add_ip_rows(d, IP, res)

    d.set_mux_uses({res["lut"]: (MUX_LUT_COST, Linear(1.0))})
    if timing.mux:
        d.set_mux_delay(timing.mux, timing.mux_w)
    d.set_chain_uses(_chain_uses(res))
    d.set_chain_uses_norst(_chain_uses_norst(res))
    d.set_register_floor(timing.reg_ns)
    d.set_default_frequency(part.grade.default_freq_mhz)
    d.realizer = vivado.realize
    return d.validate()


pynqz2 = build(
    Part(
        name="pynqz2",
        part="xc7z020clg400-1",
        grade=GRADE_1,
        capacity={
            "lut": 53_200,
            "ff": 106_400,
            "dsp": 220,
            "bram36": 140,
        },
    )
)

DEVICES = (pynqz2,)
