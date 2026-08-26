# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Versal fabric."""

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

NAME = "versal"

#: A slice mux combines LUT outputs and is counted apart from them: a CLB holds
#: four MUXF7 to its eight LUT6, so a die has half as many as it has LUTs.
DERIVED = {
    "carry8": Derived("lut", 8),
    "slicem_lut": Derived("lut", 2),
    "muxf": Derived("lut", 2),
}

GRADE_2MP = Grade("-2MP", default_freq_mhz=375.0)

TIMING: Mapping[Grade, FabricTiming] = {
    GRADE_2MP: FabricTiming(
        comb={
            CombKind.ADD: Interp(
                {8: 0.860, 16: 0.933, 32: 1.069, 64: 1.179, 96: 1.364, 128: 1.561}
            ),
            CombKind.SUB: Interp(
                {8: 0.860, 16: 0.933, 32: 1.069, 64: 1.179, 96: 1.364, 128: 1.561}
            ),
            CombKind.MUL: Interp(
                {8: 1.439, 16: 2.320, 32: 3.324, 64: 4.079, 96: 4.932, 128: 5.203}
            ),
            CombKind.DIV: Interp({8: 6.018, 16: 13.603, 32: 28.187, 64: 59.2}),
            CombKind.REM: Interp({8: 6.651, 16: 14.861, 32: 30.021, 64: 63.0}),
            CombKind.NEG: Interp({32: 0.410, 64: 0.520}),
            CombKind.MIN: Interp(
                {8: 1.269, 16: 1.291, 32: 1.465, 64: 1.625, 96: 1.699, 128: 1.736}
            ),
            CombKind.MAX: Interp(
                {8: 1.244, 16: 1.441, 32: 1.441, 64: 1.583, 96: 1.733, 128: 1.792}
            ),
            CombKind.CMP: Interp(
                {8: 0.812, 16: 0.910, 32: 0.910, 64: 0.975, 96: 0.975, 128: 1.041}
            ),
            CombKind.AND: Interp(
                {8: 0.540, 16: 0.549, 32: 0.549, 64: 0.549, 96: 0.549, 128: 0.554}
            ),
            CombKind.OR: Interp(
                {8: 0.540, 16: 0.549, 32: 0.549, 64: 0.549, 96: 0.549, 128: 0.554}
            ),
            CombKind.XOR: Interp(
                {8: 0.540, 16: 0.549, 32: 0.549, 64: 0.549, 96: 0.549, 128: 0.554}
            ),
            CombKind.SHL: Interp(
                {8: 1.014, 16: 1.133, 32: 1.441, 64: 1.628, 96: 1.960, 128: 2.100}
            ),
            CombKind.SHR: Interp(
                {8: 1.070, 16: 1.402, 32: 1.595, 64: 1.812, 96: 1.887, 128: 1.954}
            ),
            CombKind.SELECT: Interp(
                {8: 0.603, 16: 0.642, 32: 0.642, 64: 0.711, 96: 0.753, 128: 0.803}
            ),
            CombKind.INT_CAST: Interp(
                {16: 0.449, 32: 0.600, 64: 0.611, 96: 0.611, 128: 0.611}
            ),
        },
        storage={
            "register": StorageTiming(0, 1, 0.410, 0.410),
            "lutram": StorageTiming(1, 1, 1.268, 1.196),
            "bram": StorageTiming(1, 1, 1.299, 0.673),
            "uram": StorageTiming(2, 1, 1.057, 0.485),
            # A table is never written; this row carries the read at the
            # reference shape, refined by `rom` below at the array's own.
            "rom": StorageTiming(1, 1, 1.779, 0.0),
        },
        stream=StorageTiming(0, 1, 1.268, 1.196),
        reg_ns=0.410,
        # One-hot select cones, routed, marginal over the register floor and
        # monotone over fan-in; the width factor is pinned to 1.0 at 32 bits.
        mux=Interp(
            {
                2: 0.362,
                3: 0.395,
                4: 0.630,
                6: 0.636,
                8: 0.698,
                12: 0.774,
                16: 0.857,
                40: 1.096,
            }
        ),
        mux_w=Interp({1: 0.64, 8: 0.90, 16: 0.94, 32: 1.0, 64: 1.25}),
        # A constant table's read, routed, over its depth at the same reference
        # width. It grows with the depth where an addressed row's read delay is
        # flat, which bounds how deep a table is worth building.
        rom=Interp(
            {
                64: 0.912,
                256: 1.362,
                512: 1.535,
                1024: 1.779,
                2048: 1.919,
                4096: 1.963,
                16384: 2.437,
            }
        ),
        rom_w=Interp({1: 0.70, 8: 0.83, 16: 0.92, 32: 1.0, 64: 1.06}),
    ),
}

SCATTER_STORAGE = "register"

#: No cone campaign has run on this fabric, so every row below carries these
#: three numbers rather than its own: an unmeasured entry cone, no output cone,
#: and the grade's characterization clock as the warranted period. They are
#: stated on each row rather than defaulted so a reader sees which rows are
#: measurements and which are placeholders. Replace them per row with
#: `drafts/char/measure_cones.py` run against a Versal part.
_UNMEASURED = {
    "in_delay_ns": 0.5,
    "min_period_ns": 1000.0 / GRADE_2MP.default_freq_mhz,
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
    ip.fadd: IPRow(
        7,
        {"lut": 317, "slicem_lut": 13, "ff": 238, "dsp": 2, "carry8": 10},
        **_UNMEASURED,
    ),  # 436
    ip.fsub: IPRow(
        7,
        {"lut": 317, "slicem_lut": 13, "ff": 238, "dsp": 2, "carry8": 10},
        **_UNMEASURED,
    ),  # 436
    ip.fmul: IPRow(
        4,
        {"lut": 167, "slicem_lut": 1, "ff": 109, "dsp": 2, "carry8": 9},
        **_UNMEASURED,
    ),  # 502
    ip.fdiv: IPRow(
        16, {"lut": 1451, "slicem_lut": 39, "ff": 699, "carry8": 111}, **_UNMEASURED
    ),  # 452
    ip.fcmp: IPRow(1, {"lut": 105, "ff": 2, "carry8": 7}, **_UNMEASURED),  # 502
    ip.dadd: IPRow(
        14,
        {"lut": 864, "slicem_lut": 90, "ff": 861, "dsp": 3, "carry8": 26},
        **_UNMEASURED,
    ),  # 502
    ip.dsub: IPRow(
        14,
        {"lut": 864, "slicem_lut": 90, "ff": 861, "dsp": 3, "carry8": 26},
        **_UNMEASURED,
    ),  # 502
    ip.dmul: IPRow(
        9,
        {"lut": 298, "slicem_lut": 62, "ff": 397, "dsp": 7, "carry8": 15},
        **_UNMEASURED,
    ),  # 531
    ip.ddiv: IPRow(
        32, {"lut": 6216, "slicem_lut": 72, "ff": 3027, "carry8": 396}, **_UNMEASURED
    ),  # 388
    ip.dcmp: IPRow(1, {"lut": 197, "ff": 2, "carry8": 12}, **_UNMEASURED),  # 456
    ip.bfadd: IPRow(
        4, {"lut": 250, "slicem_lut": 1, "ff": 118, "carry8": 12}, **_UNMEASURED
    ),  # 440
    ip.bfsub: IPRow(
        4, {"lut": 250, "slicem_lut": 1, "ff": 118, "carry8": 12}, **_UNMEASURED
    ),  # 440
    ip.bfmul: IPRow(
        2, {"lut": 91, "ff": 34, "dsp": 1, "carry8": 6}, **_UNMEASURED
    ),  # 482
    ip.i2f: IPRow(
        3, {"lut": 243, "slicem_lut": 1, "ff": 99, "carry8": 11}, **_UNMEASURED
    ),  # 478
    ip.f2i: IPRow(3, {"lut": 222, "ff": 127, "carry8": 6}, **_UNMEASURED),  # 612
    ip.fcvt: IPRow(2, {"lut": 54, "ff": 99, "carry8": 1}, **_UNMEASURED),  # 787
    ip.bf2f: IPRow(2, {"lut": 36, "ff": 53, "carry8": 1}, **_UNMEASURED),  # 792
    ip.imul16: (
        IPRow(3, {"dsp": 1}, **_UNMEASURED),  # 990
        IPRow(1, {"dsp": 1}, **_UNMEASURED),  # 470
    ),
    ip.imul32: IPRow(2, {"ff": 32, "dsp": 3}, **_UNMEASURED),  # 407
    ip.imul64: IPRow(3, {"slicem_lut": 23, "ff": 64, "dsp": 6}, **_UNMEASURED),  # 409
    ip.idiv8: IPRow(
        12, {"lut": 210, "slicem_lut": 2, "ff": 264, "carry8": 18}, **_UNMEASURED
    ),  # 737
    ip.udiv8: (
        IPRow(
            12, {"lut": 210, "slicem_lut": 2, "ff": 264, "carry8": 18}, **_UNMEASURED
        ),  # 737
        IPRow(
            10, {"lut": 194, "slicem_lut": 1, "ff": 245, "carry8": 18}, **_UNMEASURED
        ),  # 737
    ),
    ip.irem8: IPRow(
        12, {"lut": 210, "slicem_lut": 2, "ff": 264, "carry8": 18}, **_UNMEASURED
    ),  # 737
    ip.urem8: (
        IPRow(
            12, {"lut": 210, "slicem_lut": 2, "ff": 264, "carry8": 18}, **_UNMEASURED
        ),  # 737
        IPRow(
            10, {"lut": 194, "slicem_lut": 1, "ff": 245, "carry8": 18}, **_UNMEASURED
        ),  # 737
    ),
    ip.idiv16: IPRow(
        20, {"lut": 673, "slicem_lut": 2, "ff": 904, "carry8": 55}, **_UNMEASURED
    ),  # 672
    ip.udiv16: (
        IPRow(
            20, {"lut": 673, "slicem_lut": 2, "ff": 904, "carry8": 55}, **_UNMEASURED
        ),  # 672
        IPRow(
            8, {"lut": 642, "slicem_lut": 1, "ff": 574, "carry8": 51}, **_UNMEASURED
        ),  # 378
    ),
    ip.irem16: IPRow(
        20, {"lut": 673, "slicem_lut": 2, "ff": 904, "carry8": 55}, **_UNMEASURED
    ),  # 672
    ip.urem16: (
        IPRow(
            20, {"lut": 673, "slicem_lut": 2, "ff": 904, "carry8": 55}, **_UNMEASURED
        ),  # 672
        IPRow(
            8, {"lut": 642, "slicem_lut": 1, "ff": 574, "carry8": 51}, **_UNMEASURED
        ),  # 378
    ),
    ip.idiv32: IPRow(
        36, {"lut": 2368, "slicem_lut": 4, "ff": 3336, "carry8": 173}, **_UNMEASURED
    ),  # 650
    ip.udiv32: (
        IPRow(
            36, {"lut": 2368, "slicem_lut": 4, "ff": 3336, "carry8": 173}, **_UNMEASURED
        ),  # 650
        IPRow(
            16, {"lut": 2306, "slicem_lut": 1, "ff": 2166, "carry8": 165}, **_UNMEASURED
        ),  # 380
    ),
    ip.irem32: IPRow(
        36, {"lut": 2368, "slicem_lut": 4, "ff": 3336, "carry8": 173}, **_UNMEASURED
    ),  # 650
    ip.urem32: (
        IPRow(
            36, {"lut": 2368, "slicem_lut": 4, "ff": 3336, "carry8": 173}, **_UNMEASURED
        ),  # 650
        IPRow(
            16, {"lut": 2306, "slicem_lut": 1, "ff": 2166, "carry8": 165}, **_UNMEASURED
        ),  # 380
    ),
    ip.idiv64: IPRow(
        68, {"lut": 8832, "slicem_lut": 6, "ff": 12808, "carry8": 601}, **_UNMEASURED
    ),  # 528
    ip.udiv64: (
        IPRow(
            68,
            {"lut": 8832, "slicem_lut": 6, "ff": 12808, "carry8": 601},
            **_UNMEASURED,
        ),  # 528
        IPRow(
            66,
            {"lut": 8706, "slicem_lut": 2, "ff": 12677, "carry8": 585},
            **_UNMEASURED,
        ),  # 550
    ),
    ip.irem64: IPRow(
        68, {"lut": 8832, "slicem_lut": 6, "ff": 12808, "carry8": 601}, **_UNMEASURED
    ),  # 528
    ip.urem64: (
        IPRow(
            68,
            {"lut": 8832, "slicem_lut": 6, "ff": 12808, "carry8": 601},
            **_UNMEASURED,
        ),  # 528
        IPRow(
            66,
            {"lut": 8706, "slicem_lut": 2, "ff": 12677, "carry8": 585},
            **_UNMEASURED,
        ),  # 550
    ),
}


#: SLICEM sites one bit of a 64-deep distributed RAM occupies for one instance,
#: meaning a write port and one addressed read. Measured at the 32-bit
#: reference width as half of a two-read series (120 / 480 / 960 sites at
#: 64 / 256 / 512 x 32).
LUTRAM_SITES_PER_BIT = 1.875

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
        read_first=True,
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
    lut, dsp, carry8 = r["lut"], r["dsp"], r["carry8"]
    logic = {lut: Linear(1.0)}
    addsub = {lut: Linear(1.0), carry8: Tiled(8)}
    compare = {lut: Linear(0.5), carry8: Tiled(16)}
    minmax = {lut: Linear(1.5), carry8: Tiled(16)}
    shift = {lut: Interp({8: 15, 16: 44, 32: 107, 64: 265, 96: 427, 128: 573})}
    # The DSP count steps in whole slices; the fabric logic around them grows
    # with width. Past 64 bits the partial-product tree takes more carry chains
    # than `Tiled` charges: measured 11 against 8 at 128 bits.
    multiply = {
        lut: Interp({8: 36, 16: 0, 32: 10, 64: 19, 96: 82, 128: 255}),
        dsp: Table({8: 0, 16: 1, 32: 3, 64: 6, 96: 15, 128: 21}),
        carry8: Tiled(16),
    }
    divide = {
        lut: Interp({8: 118, 16: 384, 32: 1301, 64: 4734, 96: 10189, 128: 17552}),
        carry8: Interp({8: 14, 16: 49, 32: 163, 64: 599, 96: 1283, 128: 2223}),
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
    """The :class:`Device` for one Versal die."""
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
            read_first=spec.read_first,
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


vck190 = build(
    Part(
        name="vck190",
        part="xcvc1902-vsva2197-2MP-e-S",
        grade=GRADE_2MP,
        capacity={
            "lut": 899_840,
            "ff": 1_799_680,
            "dsp": 1_968,
            "bram36": 967,
            "uram288": 463,
        },
    )
)

DEVICES = (vck190,)
