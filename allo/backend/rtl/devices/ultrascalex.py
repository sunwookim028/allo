# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The UltraScale+ fabric."""

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

NAME = "ultrascalex"

DERIVED = {
    "carry8": Derived("lut", 8),
    "slicem_lut": Derived("lut", 2),
}

GRADE_2L = Grade("-2L", default_freq_mhz=300.0)
GRADE_2LV = Grade("-2LV", default_freq_mhz=300.0)

TIMING: Mapping[Grade, FabricTiming] = {
    GRADE_2L: FabricTiming(
        comb={
            CombKind.ADD: Interp(
                {8: 0.671, 16: 0.818, 32: 0.916, 64: 1.057, 96: 1.218, 128: 1.669}
            ),
            CombKind.SUB: Interp(
                {8: 0.671, 16: 0.818, 32: 0.916, 64: 1.057, 96: 1.218, 128: 1.669}
            ),
            CombKind.MUL: Interp(
                {8: 1.653, 16: 2.353, 32: 3.241, 64: 4.970, 96: 5.759, 128: 6.564}
            ),
            CombKind.DIV: Interp({8: 5.106, 16: 10.788, 32: 24.732, 64: 59.4}),
            CombKind.REM: Interp({8: 5.409, 16: 11.212, 32: 25.144, 64: 60.3}),
            CombKind.NEG: Interp({32: 0.400, 64: 0.419}),
            CombKind.MIN: Interp(
                {8: 0.980, 16: 1.113, 32: 1.498, 64: 1.527, 96: 1.546, 128: 1.844}
            ),
            CombKind.MAX: Interp(
                {8: 0.947, 16: 1.332, 32: 1.562, 64: 1.562, 96: 1.689, 128: 1.725}
            ),
            CombKind.CMP: Interp(
                {8: 0.656, 16: 0.717, 32: 0.791, 64: 0.873, 96: 0.995, 128: 1.329}
            ),
            CombKind.AND: Interp(
                {8: 0.437, 16: 0.469, 32: 0.484, 64: 0.495, 96: 0.495, 128: 0.495}
            ),
            CombKind.OR: Interp(
                {8: 0.437, 16: 0.469, 32: 0.484, 64: 0.495, 96: 0.495, 128: 0.495}
            ),
            CombKind.XOR: Interp(
                {8: 0.446, 16: 0.446, 32: 0.495, 64: 0.495, 96: 0.495, 128: 0.495}
            ),
            CombKind.SHL: Interp(
                {8: 0.727, 16: 1.004, 32: 1.537, 64: 1.915, 96: 2.096, 128: 2.239}
            ),
            CombKind.SHR: Interp(
                {8: 1.463, 16: 1.463, 32: 1.467, 64: 1.936, 96: 2.283, 128: 2.283}
            ),
            CombKind.SELECT: Interp(
                {8: 0.540, 16: 0.540, 32: 0.540, 64: 0.955, 96: 0.955, 128: 0.955}
            ),
            CombKind.INT_CAST: Interp(
                {16: 0.413, 32: 0.433, 64: 0.744, 96: 0.744, 128: 0.744}
            ),
        },
        storage={
            "register": StorageTiming(0, 1, 0.419, 0.419),
            "lutram": StorageTiming(1, 1, 1.574, 1.718),
            "bram": StorageTiming(1, 1, 1.345, 0.510),
            "uram": StorageTiming(2, 1, 1.379, 0.444),
            # A table is never written; this row carries the read at the
            # reference shape, refined by `rom` below at the array's own.
            "rom": StorageTiming(1, 1, 1.864, 0.0),
        },
        stream=StorageTiming(0, 1, 1.574, 1.718),
        reg_ns=0.419,
        # One-hot select cones, routed, marginal over the register floor and
        # monotone over fan-in; the width factor is pinned to 1.0 at 32 bits.
        mux=Interp(
            {2: 0.200, 3: 0.287, 4: 0.644, 6: 0.777, 8: 1.214, 16: 1.214, 40: 1.238}
        ),
        mux_w=Interp({1: 0.17, 8: 0.77, 16: 0.77, 32: 1.0, 64: 1.0}),
        # A constant table's read, routed, over its depth at the same reference
        # width. It grows with the depth where an addressed row's read delay is
        # flat, which bounds how deep a table is worth building.
        rom=Interp(
            {
                64: 0.654,
                256: 1.380,
                512: 1.753,
                1024: 1.864,
                2048: 2.328,
                4096: 2.703,
                16384: 3.254,
            }
        ),
        rom_w=Interp({1: 0.66, 8: 0.92, 16: 0.92, 32: 1.0, 64: 1.10}),
    ),
    GRADE_2LV: FabricTiming(
        comb={
            CombKind.ADD: Interp(
                {8: 0.934, 16: 1.054, 32: 1.312, 64: 1.675, 96: 1.974, 128: 2.307}
            ),
            CombKind.SUB: Interp(
                {8: 0.934, 16: 1.054, 32: 1.312, 64: 1.675, 96: 1.974, 128: 2.307}
            ),
            CombKind.MUL: Interp(
                {8: 2.032, 16: 2.933, 32: 4.158, 64: 6.524, 96: 7.716, 128: 8.892}
            ),
            # The 64-bit DIV and REM entries are extrapolated, not measured.
            CombKind.DIV: Interp({8: 6.598, 16: 15.439, 32: 36.402, 64: 87.4}),
            CombKind.REM: Interp({8: 6.830, 16: 15.289, 32: 37.631, 64: 90.3}),
            CombKind.NEG: Interp({32: 0.541, 64: 0.667}),
            CombKind.MIN: Interp(
                {8: 1.425, 16: 1.425, 32: 1.570, 64: 1.851, 96: 2.108, 128: 2.395}
            ),
            CombKind.MAX: Interp(
                {8: 1.425, 16: 1.425, 32: 1.590, 64: 1.853, 96: 2.238, 128: 2.683}
            ),
            CombKind.CMP: Interp(
                {8: 0.782, 16: 1.101, 32: 1.101, 64: 1.273, 96: 1.474, 128: 1.637}
            ),
            CombKind.AND: Interp(
                {8: 0.655, 16: 0.655, 32: 0.655, 64: 0.655, 96: 0.681, 128: 0.681}
            ),
            CombKind.OR: Interp(
                {8: 0.655, 16: 0.655, 32: 0.655, 64: 0.655, 96: 0.681, 128: 0.681}
            ),
            CombKind.XOR: Interp(
                {8: 0.653, 16: 0.653, 32: 0.653, 64: 0.653, 96: 0.681, 128: 0.681}
            ),
            CombKind.SHL: Interp(
                {8: 1.622, 16: 1.622, 32: 1.772, 64: 2.038, 96: 2.368, 128: 2.857}
            ),
            CombKind.SHR: Interp(
                {8: 1.035, 16: 1.420, 32: 2.046, 64: 2.324, 96: 2.436, 128: 2.922}
            ),
            CombKind.SELECT: Interp(
                {8: 0.685, 16: 0.685, 32: 0.718, 64: 0.948, 96: 1.057, 128: 1.057}
            ),
            CombKind.INT_CAST: Interp(
                {16: 0.555, 32: 0.638, 64: 0.697, 96: 0.845, 128: 0.846}
            ),
        },
        storage={
            "register": StorageTiming(0, 1, 0.638, 0.638),
            "lutram": StorageTiming(1, 1, 1.311, 1.698),
            "bram": StorageTiming(1, 1, 1.871, 0.646),
            "uram": StorageTiming(2, 1, 2.391, 0.754),
            "rom": StorageTiming(1, 1, 2.282, 0.0),
        },
        stream=StorageTiming(0, 1, 1.311, 1.698),
        reg_ns=0.638,
        mux=Interp(
            {
                2: 0.315,
                3: 0.315,
                4: 0.596,
                6: 1.062,
                8: 1.113,
                16: 1.192,
                24: 1.540,
                40: 1.582,
            }
        ),
        mux_w=Interp({1: 0.36, 8: 0.59, 16: 0.59, 32: 1.0, 64: 1.20}),
        rom=Interp(
            {
                64: 0.844,
                256: 1.828,
                512: 2.087,
                1024: 2.282,
                2048: 2.395,
                4096: 2.713,
                16384: 3.240,
            }
        ),
        rom_w=Interp({1: 0.63, 8: 0.80, 16: 0.89, 32: 1.0, 64: 1.02}),
    ),
}

SCATTER_STORAGE = "register"


#: Operator cores measured on this fabric, each inside a registered wrapper so
#: the number covers the whole path a caller sees. The trailing comment on each
#: row is that core's achieved Fmax in MHz, a record of the characterization run
#: and not an input to the cost model. Several rows under one archetype declare
#: several cores for the library to choose between. A row is warranted at the
#: part's default clock unless ``min_period_ns`` declares its own floor, the
#: fastest clock a routed design has closed with the row inside; a floor above
#: the default period is derated from the wrapper measurement rather than
#: measured in context. The latency-1 rows are combinational up to their output
#: register and carry the measured cone in ``in_delay_ns``, which gates them
#: the same way. ``lut`` counts logic sites only: the shift registers a core
#: holds internally are split out as ``slicem_lut``.
IP: Mapping[OperatorIP, IPRow | tuple[IPRow, ...]] = {
    ip.fadd: (
        IPRow(
            7,
            {"lut": 257, "slicem_lut": 13, "ff": 238, "dsp": 2, "carry8": 10},
            min_period_ns=3.11,
        ),  # 432
        IPRow(
            5,
            {"lut": 370, "slicem_lut": 13, "ff": 242, "carry8": 17},
            min_period_ns=2.47,
        ),  # 439
        IPRow(3, {"lut": 385, "ff": 152, "carry8": 17}, min_period_ns=2.04),  # 425
        IPRow(2, {"lut": 382, "ff": 75, "carry8": 17}, min_period_ns=3.05),  # 326
        IPRow(1, {"lut": 392, "ff": 33, "carry8": 17}, in_delay_ns=4.51),  # 203
    ),
    ip.fsub: (
        IPRow(
            7, {"lut": 257, "slicem_lut": 13, "ff": 238, "dsp": 2, "carry8": 10}
        ),  # 432
        IPRow(5, {"lut": 370, "slicem_lut": 13, "ff": 242, "carry8": 17}),  # 439
        IPRow(3, {"lut": 385, "ff": 152, "carry8": 17}, min_period_ns=2.79),  # 425
        # See fadd l2: in-context 328/342 MHz, holds 300 (prior 4.09 locked out).
        IPRow(2, {"lut": 382, "ff": 75, "carry8": 17}, min_period_ns=3.05),  # 326
        IPRow(1, {"lut": 392, "ff": 33, "carry8": 17}, in_delay_ns=4.51),  # 203
    ),
    # The 2-cycle multiply measures 376 MHz standalone but routes below 300 MHz
    # inside a module, so its floor derates the wrapper number and admits the
    # row only below the default clock.
    ip.fmul: (
        IPRow(
            4,
            {"lut": 114, "slicem_lut": 1, "ff": 109, "dsp": 2, "carry8": 9},
            min_period_ns=2.65,
        ),  # 570
        IPRow(
            3, {"lut": 80, "ff": 94, "dsp": 2, "carry8": 8}, min_period_ns=2.06
        ),  # 473
        IPRow(
            2, {"lut": 81, "ff": 51, "dsp": 2, "carry8": 8}, min_period_ns=3.55
        ),  # 376
        IPRow(1, {"lut": 79, "ff": 33, "dsp": 2, "carry8": 8}, in_delay_ns=3.37),  # 264
        # Max-DSP builds (C_Mult_Usage=Max_Usage)
        IPRow(
            4,
            {"lut": 87, "slicem_lut": 1, "ff": 73, "dsp": 3, "carry8": 4},
            mnemonic="maxdsp",
            min_period_ns=1.89,
        ),  # 528
        IPRow(
            3,
            {"lut": 87, "ff": 59, "dsp": 3, "carry8": 4},
            mnemonic="maxdsp",
            min_period_ns=1.89,
        ),  # 530
    ),
    # The 10-cycle divide matches the 12-cycle row's frequency at no more area.
    # The deeper row stays declared for clocks the shorter one cannot hold.
    ip.fdiv: (
        IPRow(
            12,
            {"lut": 771, "slicem_lut": 39, "ff": 477, "carry8": 109},
            min_period_ns=2.39,
        ),  # 374
        IPRow(10, {"lut": 771, "slicem_lut": 34, "ff": 478, "carry8": 109}),  # 371
        IPRow(
            8,
            {"lut": 774, "slicem_lut": 33, "ff": 375, "carry8": 109},
            min_period_ns=3.28,
        ),  # 307
    ),
    ip.fcmp: IPRow(1, {"lut": 63, "ff": 2, "carry8": 7}, min_period_ns=1.87),  # 610
    ip.fsqrt: IPRow(
        8, {"lut": 432, "slicem_lut": 13, "ff": 291, "carry8": 67}, min_period_ns=3.25
    ),  # 321
    ip.dadd: (
        IPRow(
            14, {"lut": 721, "slicem_lut": 90, "ff": 872, "dsp": 3, "carry8": 30}
        ),  # 575
        IPRow(6, {"lut": 719, "slicem_lut": 16, "ff": 542, "carry8": 40}),  # 519
        IPRow(
            4,
            {"lut": 722, "slicem_lut": 13, "ff": 355, "carry8": 38},
            min_period_ns=2.12,
        ),  # 430
        IPRow(3, {"lut": 726, "ff": 282, "carry8": 38}, min_period_ns=3.71),  # 360
        IPRow(2, {"lut": 762, "ff": 139, "carry8": 38}, min_period_ns=3.30),  # 302
        IPRow(1, {"lut": 822, "ff": 65, "carry8": 38}, in_delay_ns=6.02),  # 155
    ),
    ip.dsub: (
        IPRow(
            14, {"lut": 721, "slicem_lut": 90, "ff": 872, "dsp": 3, "carry8": 30}
        ),  # 575
        IPRow(6, {"lut": 719, "slicem_lut": 16, "ff": 542, "carry8": 40}),  # 519
        IPRow(
            4,
            {"lut": 722, "slicem_lut": 13, "ff": 355, "carry8": 38},
            min_period_ns=2.90,
        ),  # 430
        IPRow(3, {"lut": 726, "ff": 282, "carry8": 38}, min_period_ns=3.71),  # 360
        IPRow(2, {"lut": 762, "ff": 139, "carry8": 38}, min_period_ns=3.30),  # 302
        IPRow(1, {"lut": 822, "ff": 65, "carry8": 38}, in_delay_ns=6.02),  # 155
    ),
    ip.dmul: (
        IPRow(
            9,
            {"lut": 200, "slicem_lut": 62, "ff": 397, "dsp": 7, "carry8": 15},
            min_period_ns=2.12,
        ),  # 498
        IPRow(
            5,
            {"lut": 136, "slicem_lut": 19, "ff": 184, "dsp": 7, "carry8": 15},
            min_period_ns=2.12,
        ),  # 389
        IPRow(
            3, {"lut": 135, "ff": 133, "dsp": 7, "carry8": 15}, min_period_ns=3.85
        ),  # 347
        IPRow(
            1, {"lut": 137, "ff": 65, "dsp": 7, "carry8": 15}, in_delay_ns=6.65
        ),  # 141
        # Max-DSP builds
        IPRow(
            9,
            {"lut": 165, "slicem_lut": 63, "ff": 372, "dsp": 8, "carry8": 12},
            mnemonic="maxdsp",
            min_period_ns=1.79,
        ),  # 560
        IPRow(
            5,
            {"lut": 109, "slicem_lut": 19, "ff": 214, "dsp": 8, "carry8": 12},
            mnemonic="maxdsp",
            min_period_ns=2.71,
        ),  # 369
    ),
    ip.ddiv: IPRow(
        32, {"lut": 3195, "slicem_lut": 72, "ff": 3027, "carry8": 398}
    ),  # 398
    ip.dcmp: IPRow(1, {"lut": 117, "ff": 2, "carry8": 12}, min_period_ns=2.90),  # 564
    ip.dsqrt: IPRow(
        20, {"lut": 1696, "slicem_lut": 51, "ff": 1280, "carry8": 243}
    ),  # 329
    ip.bfadd: IPRow(4, {"lut": 198, "slicem_lut": 1, "ff": 118, "carry8": 12}),  # 537
    ip.bfsub: IPRow(4, {"lut": 198, "slicem_lut": 1, "ff": 118, "carry8": 12}),  # 537
    ip.bfmul: IPRow(2, {"lut": 60, "ff": 34, "dsp": 1, "carry8": 6}),  # 521
    # IEEE fp16
    ip.hadd: IPRow(2, {"lut": 199, "ff": 43, "carry8": 12}),  # 385
    ip.hsub: IPRow(2, {"lut": 199, "ff": 43, "carry8": 12}),  # 385
    ip.hmul: IPRow(2, {"lut": 48, "ff": 31, "dsp": 1, "carry8": 6}),  # 501
    ip.hdiv: IPRow(6, {"lut": 223, "slicem_lut": 19, "ff": 150, "carry8": 29}),  # 468
    ip.hcmp: IPRow(1, {"lut": 35, "ff": 2, "carry8": 5}),  # 782
    ip.i2f: IPRow(3, {"lut": 168, "slicem_lut": 1, "ff": 99, "carry8": 11}),  # 490
    ip.f2i: IPRow(3, {"lut": 183, "ff": 127, "carry8": 6}),  # 678
    ip.fcvt: IPRow(2, {"lut": 50, "ff": 99, "carry8": 1}),  # 1032
    ip.bf2f: IPRow(2, {"lut": 34, "ff": 53, "carry8": 1}),  # 1181
    # Each multiply also carries a `mullut` row: the same core built in fabric
    # instead of DSP columns, declared at the depth of the DSP row it competes
    # with. Selection ranks depth before price, so a shorter fabric row would
    # take every multiply; at equal depth a `dsp` resource weight picks between
    # them by what each spends.
    ip.imul16: (
        IPRow(3, {"dsp": 1}),  # 1073
        IPRow(1, {"dsp": 1}),  # 544
        IPRow(
            1,
            {"lut": 192, "ff": 16, "carry8": 22},
            mnemonic="mullut",
            in_delay_ns=2.10,
        ),  # 398
    ),
    ip.imul32: (
        IPRow(2, {"ff": 32, "dsp": 3}, min_period_ns=2.94),  # 341
        IPRow(
            2,
            {"lut": 768, "ff": 112, "carry8": 76},
            mnemonic="mullut",
            min_period_ns=2.50,
        ),  # 401
        # A combinational 3-DSP cascade up to its output register: routed in
        # context the cone runs 3.0 ns (2.9 ns of DSP logic plus route), which
        # rules the row out at 300 MHz and leaves it to lower targets.
        IPRow(1, {"ff": 32, "dsp": 3}, in_delay_ns=3.0),  # 320
    ),
    ip.imul64: (
        IPRow(6, {"slicem_lut": 64, "ff": 81, "dsp": 10}),  # 333
        IPRow(
            6,
            {"lut": 3072, "slicem_lut": 16, "ff": 2168, "carry8": 280},
            mnemonic="mullut",
            min_period_ns=1.77,
        ),  # 568
    ),
    ip.imulw33: IPRow(3, {"ff": 34, "dsp": 4}),  # 431
    ip.imuladd32: IPRow(3, {"lut": 47, "ff": 113, "dsp": 3, "carry8": 6}),  # 448
    ip.idiv8: IPRow(4, {"lut": 126, "slicem_lut": 1, "ff": 166, "carry8": 18}),  # 311
    ip.udiv8: IPRow(4, {"lut": 126, "slicem_lut": 1, "ff": 166, "carry8": 18}),  # 311
    ip.irem8: IPRow(4, {"lut": 126, "slicem_lut": 1, "ff": 166, "carry8": 18}),  # 311
    ip.urem8: IPRow(4, {"lut": 126, "slicem_lut": 1, "ff": 166, "carry8": 18}),  # 311
    ip.idiv16: IPRow(8, {"lut": 376, "slicem_lut": 2, "ff": 578, "carry8": 55}),  # 319
    ip.udiv16: IPRow(8, {"lut": 376, "slicem_lut": 2, "ff": 578, "carry8": 55}),  # 319
    ip.irem16: IPRow(8, {"lut": 376, "slicem_lut": 2, "ff": 578, "carry8": 55}),  # 319
    ip.urem16: IPRow(8, {"lut": 376, "slicem_lut": 2, "ff": 578, "carry8": 55}),  # 319
    ip.idiv32: IPRow(
        16, {"lut": 1264, "slicem_lut": 2, "ff": 2170, "carry8": 173}
    ),  # 345
    ip.udiv32: IPRow(
        16, {"lut": 1264, "slicem_lut": 2, "ff": 2170, "carry8": 173}
    ),  # 345
    ip.irem32: IPRow(
        16, {"lut": 1264, "slicem_lut": 2, "ff": 2170, "carry8": 173}
    ),  # 345
    ip.urem32: IPRow(
        16, {"lut": 1264, "slicem_lut": 2, "ff": 2170, "carry8": 173}
    ),  # 345
    ip.idiv64: IPRow(
        68, {"lut": 4608, "slicem_lut": 6, "ff": 12808, "carry8": 601}
    ),  # 579
    ip.udiv64: IPRow(
        32, {"lut": 4481, "slicem_lut": 1, "ff": 8422, "carry8": 585}
    ),  # 305
    ip.irem64: IPRow(
        68, {"lut": 4608, "slicem_lut": 6, "ff": 12808, "carry8": 601}
    ),  # 579
    ip.urem64: IPRow(
        32, {"lut": 4481, "slicem_lut": 1, "ff": 8422, "carry8": 585}
    ),  # 305
}


#: Rows that replace the base entry for their archetype at one grade, with a
#: tuple on either side standing for the whole candidate set.
IP_BY_GRADE: Mapping[Grade, Mapping[OperatorIP, IPRow | tuple[IPRow, ...]]] = {
    # The 2-cycle unsigned 8-bit divider (311 MHz against 249) and the 24-cycle
    # double divider (308 against 208) close on -2L and miss the same 300 MHz
    # clock on -2LV, so each is a candidate at this grade only.
    GRADE_2L: {
        ip.udiv8: (
            IPRow(4, {"lut": 126, "slicem_lut": 1, "ff": 166, "carry8": 18}),  # 311
            IPRow(2, {"lut": 110, "ff": 132, "carry8": 18}),  # 311
        ),
        ip.urem8: (
            IPRow(4, {"lut": 126, "slicem_lut": 1, "ff": 166, "carry8": 18}),  # 311
            IPRow(2, {"lut": 110, "ff": 132, "carry8": 18}),  # 311
        ),
        ip.ddiv: (
            IPRow(
                32, {"lut": 3195, "slicem_lut": 72, "ff": 3027, "carry8": 398}
            ),  # 398
            IPRow(
                24,
                {"lut": 3198, "slicem_lut": 72, "ff": 2064, "carry8": 398},
                min_period_ns=2.90,
            ),  # 308
        ),
    },
    # The low-voltage grade closes none of the -2L integer division rows and
    # needs a deeper multiply, so most of its integer arithmetic is a row of
    # its own.
    GRADE_2LV: {
        # The low-voltage float ladder is slower throughout: each archetype
        # keeps its depth at the default clock and the shallow rows carry
        # correspondingly higher floors.
        ip.fadd: (
            IPRow(
                7, {"lut": 257, "slicem_lut": 13, "ff": 238, "dsp": 2, "carry8": 10}
            ),  # 350
            IPRow(5, {"lut": 370, "slicem_lut": 13, "ff": 242, "carry8": 17}),  # 349
            IPRow(2, {"lut": 382, "ff": 75, "carry8": 17}, min_period_ns=4.63),  # 289
            IPRow(1, {"lut": 392, "ff": 33, "carry8": 17}, in_delay_ns=6.36),  # 143
        ),
        ip.fsub: (
            IPRow(
                7, {"lut": 257, "slicem_lut": 13, "ff": 238, "dsp": 2, "carry8": 10}
            ),  # 350
            IPRow(5, {"lut": 370, "slicem_lut": 13, "ff": 242, "carry8": 17}),  # 349
            IPRow(2, {"lut": 382, "ff": 75, "carry8": 17}, min_period_ns=4.63),  # 289
            IPRow(1, {"lut": 392, "ff": 33, "carry8": 17}, in_delay_ns=6.36),  # 143
        ),
        ip.fmul: (
            IPRow(
                4, {"lut": 114, "slicem_lut": 1, "ff": 109, "dsp": 2, "carry8": 9}
            ),  # 421
            IPRow(
                2, {"lut": 81, "ff": 51, "dsp": 2, "carry8": 8}, min_period_ns=4.53
            ),  # 294
            IPRow(
                1, {"lut": 79, "ff": 33, "dsp": 2, "carry8": 8}, in_delay_ns=4.40
            ),  # 198
        ),
        ip.dadd: (
            IPRow(
                14, {"lut": 721, "slicem_lut": 90, "ff": 872, "dsp": 3, "carry8": 30}
            ),  # 398
            IPRow(6, {"lut": 719, "slicem_lut": 16, "ff": 542, "carry8": 40}),  # 361
            IPRow(3, {"lut": 726, "ff": 282, "carry8": 38}, min_period_ns=4.28),  # 312
            IPRow(1, {"lut": 822, "ff": 65, "carry8": 38}, in_delay_ns=10.07),  # 93
        ),
        ip.dsub: (
            IPRow(
                14, {"lut": 721, "slicem_lut": 90, "ff": 872, "dsp": 3, "carry8": 30}
            ),  # 398
            IPRow(6, {"lut": 719, "slicem_lut": 16, "ff": 542, "carry8": 40}),  # 361
            IPRow(3, {"lut": 726, "ff": 282, "carry8": 38}, min_period_ns=4.28),  # 312
            IPRow(1, {"lut": 822, "ff": 65, "carry8": 38}, in_delay_ns=10.07),  # 93
        ),
        ip.dmul: (
            IPRow(
                9, {"lut": 200, "slicem_lut": 62, "ff": 397, "dsp": 7, "carry8": 15}
            ),  # 443
            IPRow(
                8, {"lut": 198, "slicem_lut": 63, "ff": 360, "dsp": 7, "carry8": 15}
            ),  # 386
            IPRow(
                3, {"lut": 135, "ff": 133, "dsp": 7, "carry8": 15}, min_period_ns=5.03
            ),  # 265
            IPRow(
                1, {"lut": 137, "ff": 65, "dsp": 7, "carry8": 15}, in_delay_ns=8.75
            ),  # 107
        ),
        ip.fdiv: IPRow(
            16, {"lut": 765, "slicem_lut": 39, "ff": 699, "carry8": 111}
        ),  # 361
        # This grade holds its multiplies at other depths than -2L, so each
        # fabric row sits at the depth of the DSP row beside it here.
        ip.imul16: (
            IPRow(3, {"dsp": 1}),  # 837
            IPRow(1, {"dsp": 1}),  # 412
            IPRow(
                1,
                {"lut": 192, "ff": 16, "carry8": 22},
                mnemonic="mullut",
                in_delay_ns=2.44,
            ),  # 326
        ),
        ip.imul32: (
            IPRow(3, {"ff": 32, "dsp": 3}),  # 341
            IPRow(
                3,
                {"lut": 768, "ff": 384, "carry8": 76},
                mnemonic="mullut",
                min_period_ns=3.01,
            ),  # 333
        ),
        ip.imul64: (
            IPRow(8, {"slicem_lut": 113, "ff": 160, "dsp": 10}),  # 325
            IPRow(
                8,
                {"lut": 3072, "slicem_lut": 80, "ff": 2232, "carry8": 280},
                mnemonic="mullut",
                min_period_ns=2.07,
            ),  # 483
        ),
        ip.idiv8: IPRow(
            12, {"lut": 130, "slicem_lut": 2, "ff": 264, "carry8": 18}
        ),  # 804
        ip.irem8: IPRow(
            12, {"lut": 130, "slicem_lut": 2, "ff": 264, "carry8": 18}
        ),  # 804
        ip.udiv8: IPRow(
            4, {"lut": 113, "slicem_lut": 1, "ff": 162, "carry8": 18}
        ),  # 302
        ip.urem8: IPRow(
            4, {"lut": 113, "slicem_lut": 1, "ff": 162, "carry8": 18}
        ),  # 302
        ip.idiv16: IPRow(
            20, {"lut": 384, "slicem_lut": 2, "ff": 904, "carry8": 55}
        ),  # 745
        ip.irem16: IPRow(
            20, {"lut": 384, "slicem_lut": 2, "ff": 904, "carry8": 55}
        ),  # 745
        ip.udiv16: IPRow(
            8, {"lut": 353, "slicem_lut": 1, "ff": 574, "carry8": 51}
        ),  # 324
        ip.urem16: IPRow(
            8, {"lut": 353, "slicem_lut": 1, "ff": 574, "carry8": 51}
        ),  # 324
        ip.idiv32: IPRow(
            36, {"lut": 1280, "slicem_lut": 4, "ff": 3336, "carry8": 173}
        ),  # 572
        ip.irem32: IPRow(
            36, {"lut": 1280, "slicem_lut": 4, "ff": 3336, "carry8": 173}
        ),  # 572
        ip.udiv32: IPRow(
            34, {"lut": 1217, "slicem_lut": 1, "ff": 3269, "carry8": 165}
        ),  # 626
        ip.urem32: IPRow(
            34, {"lut": 1217, "slicem_lut": 1, "ff": 3269, "carry8": 165}
        ),  # 626
        ip.idiv64: IPRow(
            68, {"lut": 4608, "slicem_lut": 6, "ff": 12808, "carry8": 601}
        ),  # 439
        ip.irem64: IPRow(
            68, {"lut": 4608, "slicem_lut": 6, "ff": 12808, "carry8": 601}
        ),  # 439
        ip.udiv64: IPRow(
            66, {"lut": 4481, "slicem_lut": 2, "ff": 12677, "carry8": 585}
        ),  # 454
        ip.urem64: IPRow(
            66, {"lut": 4481, "slicem_lut": 2, "ff": 12677, "carry8": 585}
        ),  # 454
    },
}


#: SLICEM sites one bit of a 64-deep distributed RAM occupies for one instance,
#: meaning a write port and one addressed read. Measured 640 LUT as memory at
#: 1024x32 with one read; the two-read series (80 / 320 / 640 sites at 64 /
#: 256 / 512 x 32) lands on exactly twice this.
LUTRAM_SITES_PER_BIT = 1.25

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
    # array, charged as a further instance: measured 640 / 1280 / 1920 / 2560
    # LUT as memory at 1024x32 for one through four reads, 640 per copy. A
    # SLICEM LUT holds 64 bits, so a bit of a `d`-deep array takes
    # `ceil(d/64)` sites.
    "lutram": StorageSpec(
        ("slicem_lut",),
        lambda r: {r["slicem_lut"]: (Tiled(64), Linear(LUTRAM_SITES_PER_BIT))},
        inst_reads=1,
        inst_writes=1,
        ram_style="distributed",
    ),
    # A read-only table is logic, not storage: one LUT6 is a 64-entry one-bit
    # lookup and has no address bus to contend for. Its read is the cone
    # through those LUTs, so it is the one row timed over the array's own shape.
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
    compare = {lut: Linear(1.0), carry8: Tiled(16)}
    minmax = {lut: Linear(2.0), carry8: Tiled(16)}
    shift = {lut: Interp({8: 15, 16: 44, 32: 107, 64: 265, 96: 427, 128: 573})}
    # The DSP count steps in whole slices; the fabric logic around them grows
    # with width. Past 64 bits the partial-product tree takes more carry chains
    # than `Tiled` charges: measured 31 against 8 at 128 bits.
    multiply = {
        lut: Interp({8: 39, 16: 0, 32: 15, 64: 41, 96: 153, 128: 316}),
        dsp: Table({8: 0, 16: 1, 32: 3, 64: 10, 96: 21, 128: 34}),
        carry8: Tiled(16),
    }
    divide = {
        lut: Interp({8: 125, 16: 377, 32: 1344, 64: 4731, 96: 10188, 128: 17575}),
        carry8: Interp({8: 14, 16: 50, 32: 172, 64: 600, 96: 1284, 128: 2224}),
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
        # An SRL32E holds 32 stages, so an extracted chain takes ceil(depth/32)
        # sites a bit and a shallower one takes none.
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
    """The :class:`Device` for one UltraScale+ die."""
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

    add_ip_rows(
        d, {**IP, **IP_BY_GRADE.get(part.grade, {})}, res, part.grade.default_freq_mhz
    )

    d.set_mux_uses({res["lut"]: (MUX_LUT_COST, Linear(1.0))})
    if timing.mux:
        d.set_mux_delay(timing.mux, timing.mux_w)
    d.set_chain_uses(_chain_uses(res))
    d.set_chain_uses_norst(_chain_uses_norst(res))
    # Routed designs pack 1.22 to 1.25 LUT instances per occupied site.
    d.set_lut_packing(0.80)
    d.set_register_floor(timing.reg_ns)
    d.set_default_frequency(part.grade.default_freq_mhz)
    d.realizer = vivado.realize
    return d.validate()
