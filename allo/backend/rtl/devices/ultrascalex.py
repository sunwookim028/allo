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
    ROM_MUXF_COST,
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

#: A slice mux combines LUT outputs and is counted apart from them: a CLB holds
#: four MUXF7 to its eight LUT6, so a die has half as many as it has LUTs.
DERIVED = {
    "carry8": Derived("lut", 8),
    "slicem_lut": Derived("lut", 2),
    "muxf": Derived("lut", 2),
}

GRADE_2L = Grade("-2L", default_freq_mhz=300.0)
GRADE_2LV = Grade("-2LV", default_freq_mhz=300.0)

TIMING: Mapping[Grade, FabricTiming] = {
    GRADE_2L: FabricTiming(
        comb={
            CombKind.ADD: Interp(
                {8: 0.671, 16: 0.818, 32: 0.949, 64: 1.04, 96: 1.181, 128: 1.375}
            ),
            CombKind.SUB: Interp(
                {8: 0.671, 16: 0.818, 32: 0.949, 64: 1.04, 96: 1.181, 128: 1.375}
            ),
            CombKind.MUL: Interp(
                {8: 1.553, 16: 2.353, 32: 3.241, 64: 5.049, 96: 5.77, 128: 6.517}
            ),
            # 64 bits is extrapolated by the 16 to 32 ratio, not measured.
            CombKind.DIV: Interp({8: 5.147, 16: 10.711, 32: 24.237, 64: 54.8}),
            # 64 bits is extrapolated by the 16 to 32 ratio, not measured.
            CombKind.REM: Interp({8: 5.502, 16: 11.284, 32: 25.87, 64: 59.3}),
            CombKind.NEG: Interp({32: 0.4, 64: 0.419}),
            CombKind.MIN: Interp(
                {8: 0.979, 16: 1.105, 32: 1.406, 64: 1.442, 96: 1.487, 128: 1.706}
            ),
            CombKind.MAX: Interp(
                {8: 0.95, 16: 1.272, 32: 1.498, 64: 1.498, 96: 1.583, 128: 1.749}
            ),
            CombKind.CMP: Interp(
                {8: 0.656, 16: 0.717, 32: 0.791, 64: 0.845, 96: 1.023, 128: 1.35}
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
                {8: 0.725, 16: 0.971, 32: 1.55, 64: 1.869, 96: 2.091, 128: 2.26}
            ),
            CombKind.SHR: Interp(
                {8: 0.894, 16: 1.166, 32: 1.434, 64: 1.94, 96: 2.245, 128: 2.342}
            ),
            CombKind.SELECT: Interp(
                {8: 0.54, 16: 0.54, 32: 0.54, 64: 0.925, 96: 0.935, 128: 0.935}
            ),
            CombKind.INT_CAST: Interp(
                {16: 0.413, 32: 0.433, 64: 0.557, 96: 0.744, 128: 0.744}
            ),
        },
        storage={
            "register": StorageTiming(0, 1, 0.419, 0.419),
            "lutram": StorageTiming(1, 1, 1.497, 1.544),
            "bram": StorageTiming(1, 1, 1.345, 0.51),
            "uram": StorageTiming(2, 1, 1.388, 0.647),
            # A table is never written; this row carries the read at the
            # reference shape, refined by `rom` below at the array's own.
            "rom": StorageTiming(1, 1, 2.121, 0.0),
        },
        stream=StorageTiming(0, 1, 1.497, 1.544),
        reg_ns=0.419,
        mux=Interp(
            {
                2: 0.263,
                3: 0.281,
                4: 0.638,
                5: 1.044,
                6: 1.044,
                7: 1.18,
                8: 1.18,
                9: 1.18,
                10: 1.18,
                11: 1.18,
                12: 1.18,
                14: 1.18,
                16: 1.18,
                18: 1.18,
                20: 1.201,
                22: 1.201,
                24: 1.219,
                26: 1.219,
                28: 1.456,
                30: 1.527,
                32: 1.527,
                36: 1.527,
                40: 1.527,
                48: 1.625,
                56: 1.774,
                64: 1.774,
            }
        ),
        mux_w=Interp({1: 0.2, 8: 1.0, 16: 1.0, 32: 1.0, 64: 1.29}),
        rom=Interp(
            {
                64: 0.654,
                256: 1.38,
                512: 1.842,
                1024: 2.121,
                2048: 2.121,
                4096: 2.463,
                16384: 3.248,
            }
        ),
        rom_w=Interp({1: 0.47, 8: 0.68, 16: 0.83, 32: 1.0, 64: 1.03}),
    ),
    GRADE_2LV: FabricTiming(
        comb={
            CombKind.ADD: Interp(
                {8: 0.934, 16: 1.054, 32: 1.383, 64: 1.611, 96: 1.938, 128: 2.391}
            ),
            CombKind.SUB: Interp(
                {8: 0.934, 16: 1.054, 32: 1.383, 64: 1.611, 96: 1.938, 128: 2.391}
            ),
            CombKind.MUL: Interp(
                {8: 2.028, 16: 2.933, 32: 4.144, 64: 6.534, 96: 7.778, 128: 8.911}
            ),
            # 64 bits is extrapolated by the 16 to 32 ratio, not measured.
            CombKind.DIV: Interp({8: 6.507, 16: 14.627, 32: 35.992, 64: 88.6}),
            # 64 bits is extrapolated by the 16 to 32 ratio, not measured.
            CombKind.REM: Interp({8: 7.184, 16: 15.062, 32: 37.604, 64: 93.9}),
            CombKind.NEG: Interp({32: 0.541, 64: 0.667}),
            CombKind.MIN: Interp(
                {8: 1.425, 16: 1.425, 32: 1.581, 64: 1.802, 96: 2.184, 128: 2.401}
            ),
            CombKind.MAX: Interp(
                {8: 1.425, 16: 1.425, 32: 1.663, 64: 1.97, 96: 2.3, 128: 2.575}
            ),
            CombKind.CMP: Interp(
                {8: 0.782, 16: 1.101, 32: 1.101, 64: 1.385, 96: 1.418, 128: 1.636}
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
                {8: 1.117, 16: 1.367, 32: 1.671, 64: 2.116, 96: 2.33, 128: 2.802}
            ),
            CombKind.SHR: Interp(
                {8: 1.066, 16: 1.483, 32: 1.803, 64: 2.318, 96: 2.482, 128: 2.949}
            ),
            CombKind.SELECT: Interp(
                {8: 0.685, 16: 0.685, 32: 0.703, 64: 0.948, 96: 0.973, 128: 1.005}
            ),
            CombKind.INT_CAST: Interp(
                {16: 0.555, 32: 0.638, 64: 0.695, 96: 0.722, 128: 0.921}
            ),
        },
        storage={
            "register": StorageTiming(0, 1, 0.638, 0.638),
            "lutram": StorageTiming(1, 1, 1.234, 1.742),
            "bram": StorageTiming(1, 1, 1.871, 0.646),
            "uram": StorageTiming(2, 1, 1.734, 0.754),
            # A table is never written; this row carries the read at the
            # reference shape, refined by `rom` below at the array's own.
            "rom": StorageTiming(1, 1, 2.175, 0.0),
        },
        stream=StorageTiming(0, 1, 1.234, 1.742),
        reg_ns=0.638,
        mux=Interp(
            {
                2: 0.221,
                3: 0.221,
                4: 0.499,
                5: 0.876,
                6: 0.876,
                7: 0.876,
                8: 0.876,
                9: 0.993,
                10: 1.138,
                11: 1.138,
                12: 1.138,
                14: 1.139,
                16: 1.163,
                18: 1.239,
                20: 1.448,
                22: 1.46,
                24: 1.67,
                26: 1.67,
                28: 1.67,
                30: 1.67,
                32: 1.67,
                36: 1.67,
                40: 1.67,
                48: 1.67,
                56: 1.67,
                64: 1.904,
            }
        ),
        mux_w=Interp({1: 0.3, 8: 0.84, 16: 0.84, 32: 1.0, 64: 1.3}),
        rom=Interp(
            {
                64: 0.88,
                256: 1.828,
                512: 2.075,
                1024: 2.175,
                2048: 2.339,
                4096: 2.662,
                16384: 3.272,
            }
        ),
        rom_w=Interp({1: 0.64, 8: 0.84, 16: 0.94, 32: 1.0, 64: 1.18}),
    ),
}

SCATTER_STORAGE = "register"


#: Operator cores measured on this fabric, each inside a registered wrapper so
#: the numbers cover the whole path a caller sees. Several rows under one
#: archetype declare several cores for the library to choose between.
#:
#: Every number below, delays and area alike, comes from one run of
#: `drafts/char/measure_cones.py` per grade, written back by
#: `drafts/char/apply_cones.py`. No field takes a default, is carried over from
#: an older campaign, or is shared with another grade.
#:
#: The three arcs are timed apart because the model charges them in three
#: places: ``in_delay_ns`` from the data ports to the core's first internal
#: register (less the register floor), ``min_period_ns`` the worst path between
#: two internal registers, ``out_delay_ns`` from the last one back to the ports.
#: The period a row needs for a cycle of its own is their max, which is the
#: whole register-to-register path, so its achievable Fmax is read off the row
#: rather than recorded beside it.
#:
#: ``lut`` counts logic sites only: the shift registers a core holds internally
#: are split out as ``slicem_lut``.
IP: Mapping[OperatorIP, IPRow | tuple[IPRow, ...]] = {
    ip.fadd: (
        IPRow(
            7,
            {"lut": 251, "slicem_lut": 12, "ff": 234, "dsp": 2, "carry8": 10},
            in_delay_ns=0.81,
            min_period_ns=2.2,
            out_delay_ns=0.5,
        ),
        IPRow(
            5,
            {"lut": 364, "slicem_lut": 12, "ff": 236, "carry8": 17, "muxf": 2},
            in_delay_ns=1.9,
            min_period_ns=1.96,
            out_delay_ns=0.52,
            variant="nodsp",
        ),
        IPRow(
            3,
            {"lut": 378, "ff": 147, "carry8": 17, "muxf": 2},
            in_delay_ns=2.07,
            min_period_ns=2.33,
            out_delay_ns=0.73,
            variant="nodsp",
        ),
        IPRow(
            2,
            {"lut": 386, "ff": 72, "carry8": 17, "muxf": 4},
            in_delay_ns=2.46,
            min_period_ns=2.79,
            out_delay_ns=0.58,
            variant="nodsp",
        ),
        IPRow(
            1,
            {"lut": 367, "ff": 32, "carry8": 17, "muxf": 2},
            in_delay_ns=4.81,
            min_period_ns=0.4,
            out_delay_ns=1.26,
            variant="nodsp",
        ),
    ),
    ip.fsub: (
        IPRow(
            7,
            {"lut": 252, "slicem_lut": 12, "ff": 234, "dsp": 2, "carry8": 10},
            in_delay_ns=0.86,
            min_period_ns=2.4,
            out_delay_ns=0.56,
        ),
        IPRow(
            5,
            {"lut": 364, "slicem_lut": 12, "ff": 236, "carry8": 17, "muxf": 2},
            in_delay_ns=1.9,
            min_period_ns=1.96,
            out_delay_ns=0.52,
            variant="nodsp",
        ),
        IPRow(
            3,
            {"lut": 378, "ff": 147, "carry8": 17, "muxf": 2},
            in_delay_ns=2.07,
            min_period_ns=2.33,
            out_delay_ns=0.73,
            variant="nodsp",
        ),
        # See fadd l2: in-context 328/342 MHz, holds 300 (prior 4.09 locked out).
        IPRow(
            2,
            {"lut": 386, "ff": 72, "carry8": 17, "muxf": 4},
            in_delay_ns=2.46,
            min_period_ns=2.79,
            out_delay_ns=0.58,
            variant="nodsp",
        ),
        IPRow(
            1,
            {"lut": 367, "ff": 32, "carry8": 17, "muxf": 2},
            in_delay_ns=4.81,
            min_period_ns=0.4,
            out_delay_ns=1.26,
            variant="nodsp",
        ),
    ),
    # The 2-cycle multiply measures 376 MHz standalone but routes below 300 MHz
    # inside a module, so its floor derates the wrapper number and admits the
    # row only below the default clock.
    ip.fmul: (
        IPRow(
            4,
            {"lut": 106, "ff": 102, "dsp": 2, "carry8": 9},
            in_delay_ns=1.06,
            min_period_ns=1.57,
            out_delay_ns=0.56,
        ),
        IPRow(
            3,
            {"lut": 78, "ff": 90, "dsp": 2, "carry8": 8},
            in_delay_ns=1.6,
            min_period_ns=1.67,
            out_delay_ns=0.52,
        ),
        IPRow(
            2,
            {"lut": 79, "ff": 48, "dsp": 2, "carry8": 8},
            in_delay_ns=2.17,
            min_period_ns=1.6,
            out_delay_ns=0.53,
        ),
        IPRow(
            1,
            {"lut": 80, "ff": 32, "dsp": 2, "carry8": 8},
            in_delay_ns=3.41,
            min_period_ns=3.37,
            out_delay_ns=0.6,
        ),
        # The maximum-DSP build, declared at the depths where it is the fastest
        # core of that depth. It spends a third DSP; where it is only smaller and
        # slower it stays undeclared, the depth ladder already covering that.
        IPRow(
            3,
            {"lut": 84, "ff": 55, "dsp": 3, "carry8": 4},
            in_delay_ns=1.21,
            min_period_ns=1.78,
            out_delay_ns=0.65,
            mnemonic="mul_maxdsp",
            variant="maxdsp",
        ),
        IPRow(
            2,
            {"lut": 79, "ff": 47, "dsp": 3, "carry8": 3},
            in_delay_ns=1.52,
            min_period_ns=2.48,
            out_delay_ns=0.51,
            mnemonic="mul_maxdsp",
            variant="maxdsp",
        ),
        IPRow(
            1,
            {"lut": 75, "ff": 32, "dsp": 3, "carry8": 3},
            in_delay_ns=3.29,
            min_period_ns=3.25,
            out_delay_ns=0.74,
            mnemonic="mul_maxdsp",
            variant="maxdsp",
        ),
    ),
    # The 10-cycle divide matches the 12-cycle row's frequency at no more area.
    # The deeper row stays declared for clocks the shorter one cannot hold.
    ip.fdiv: (
        IPRow(
            12,
            {"lut": 764, "slicem_lut": 37, "ff": 467, "carry8": 109},
            in_delay_ns=2.0,
            min_period_ns=2.64,
            out_delay_ns=0.51,
        ),
        IPRow(
            10,
            {"lut": 764, "slicem_lut": 32, "ff": 468, "carry8": 109},
            in_delay_ns=2.07,
            min_period_ns=2.75,
            out_delay_ns=0.41,
        ),
        IPRow(
            8,
            {"lut": 766, "slicem_lut": 31, "ff": 365, "carry8": 109},
            in_delay_ns=2.69,
            min_period_ns=3.19,
            out_delay_ns=0.42,
        ),
    ),
    ip.fcmp: IPRow(
        1,
        {"lut": 13, "ff": 1, "carry8": 2},
        in_delay_ns=0.57,
        min_period_ns=0.0,
        out_delay_ns=0.33,
    ),
    ip.fsqrt: IPRow(
        8,
        {"lut": 431, "slicem_lut": 12, "ff": 282, "carry8": 67},
        in_delay_ns=1.74,
        min_period_ns=3.0,
        out_delay_ns=0.46,
    ),
    ip.dadd: (
        IPRow(
            14,
            {
                "lut": 700,
                "slicem_lut": 89,
                "ff": 862,
                "dsp": 3,
                "carry8": 30,
                "muxf": 11,
            },
            in_delay_ns=0.76,
            min_period_ns=1.64,
            out_delay_ns=0.46,
        ),
        IPRow(
            6,
            {"lut": 704, "slicem_lut": 15, "ff": 536, "carry8": 40, "muxf": 11},
            in_delay_ns=1.44,
            min_period_ns=1.9,
            out_delay_ns=0.67,
            variant="nodsp",
        ),
        IPRow(
            4,
            {"lut": 706, "slicem_lut": 12, "ff": 350, "carry8": 38, "muxf": 11},
            in_delay_ns=1.89,
            min_period_ns=2.14,
            out_delay_ns=0.69,
            variant="nodsp",
        ),
        IPRow(
            3,
            {"lut": 711, "ff": 277, "carry8": 38, "muxf": 11},
            in_delay_ns=2.4,
            min_period_ns=2.65,
            out_delay_ns=0.54,
            variant="nodsp",
        ),
        IPRow(
            2,
            {"lut": 752, "ff": 136, "carry8": 38, "muxf": 6},
            in_delay_ns=2.8,
            min_period_ns=3.17,
            out_delay_ns=0.58,
            variant="nodsp",
        ),
        IPRow(
            1,
            {"lut": 812, "ff": 64, "carry8": 38, "muxf": 8},
            in_delay_ns=6.16,
            min_period_ns=0.5,
            out_delay_ns=0.8,
            variant="nodsp",
        ),
    ),
    ip.dsub: (
        IPRow(
            14,
            {
                "lut": 701,
                "slicem_lut": 89,
                "ff": 862,
                "dsp": 3,
                "carry8": 30,
                "muxf": 11,
            },
            in_delay_ns=1.05,
            min_period_ns=1.75,
            out_delay_ns=0.52,
        ),
        IPRow(
            6,
            {"lut": 704, "slicem_lut": 15, "ff": 536, "carry8": 40, "muxf": 11},
            in_delay_ns=1.44,
            min_period_ns=1.9,
            out_delay_ns=0.67,
            variant="nodsp",
        ),
        IPRow(
            4,
            {"lut": 706, "slicem_lut": 12, "ff": 350, "carry8": 38, "muxf": 11},
            in_delay_ns=1.89,
            min_period_ns=2.14,
            out_delay_ns=0.69,
            variant="nodsp",
        ),
        IPRow(
            3,
            {"lut": 711, "ff": 277, "carry8": 38, "muxf": 11},
            in_delay_ns=2.4,
            min_period_ns=2.65,
            out_delay_ns=0.54,
            variant="nodsp",
        ),
        IPRow(
            2,
            {"lut": 752, "ff": 136, "carry8": 38, "muxf": 6},
            in_delay_ns=2.8,
            min_period_ns=3.17,
            out_delay_ns=0.58,
            variant="nodsp",
        ),
        IPRow(
            1,
            {"lut": 812, "ff": 64, "carry8": 38, "muxf": 8},
            in_delay_ns=6.16,
            min_period_ns=0.5,
            out_delay_ns=0.8,
            variant="nodsp",
        ),
    ),
    ip.dmul: (
        IPRow(
            9,
            {"lut": 192, "slicem_lut": 61, "ff": 390, "dsp": 7, "carry8": 15},
            in_delay_ns=1.29,
            min_period_ns=1.51,
            out_delay_ns=0.46,
        ),
        IPRow(
            5,
            {"lut": 134, "slicem_lut": 18, "ff": 181, "dsp": 7, "carry8": 15},
            in_delay_ns=2.24,
            min_period_ns=1.71,
            out_delay_ns=0.55,
        ),
        IPRow(
            3,
            {"lut": 133, "ff": 129, "dsp": 7, "carry8": 15},
            in_delay_ns=2.37,
            min_period_ns=2.89,
            out_delay_ns=0.51,
        ),
        IPRow(
            1,
            {"lut": 143, "ff": 64, "dsp": 7, "carry8": 15},
            in_delay_ns=6.49,
            min_period_ns=6.45,
            out_delay_ns=0.65,
        ),
    ),
    ip.ddiv: IPRow(
        32,
        {"lut": 3189, "slicem_lut": 70, "ff": 3017, "carry8": 398},
        in_delay_ns=1.73,
        min_period_ns=2.56,
        out_delay_ns=0.59,
    ),
    ip.dcmp: IPRow(
        1,
        {"lut": 23, "ff": 1, "carry8": 4},
        in_delay_ns=0.84,
        min_period_ns=0.0,
        out_delay_ns=0.34,
    ),
    ip.dsqrt: IPRow(
        20,
        {"lut": 1695, "slicem_lut": 50, "ff": 1203, "carry8": 243},
        in_delay_ns=1.81,
        min_period_ns=3.1,
        out_delay_ns=0.55,
    ),
    ip.bfadd: IPRow(
        4,
        {"lut": 195, "ff": 113, "carry8": 12},
        in_delay_ns=1.07,
        min_period_ns=1.63,
        out_delay_ns=0.84,
    ),
    ip.bfsub: IPRow(
        4,
        {"lut": 195, "ff": 113, "carry8": 12},
        in_delay_ns=1.07,
        min_period_ns=1.63,
        out_delay_ns=0.84,
    ),
    ip.bfmul: IPRow(
        2,
        {"lut": 58, "ff": 31, "dsp": 1, "carry8": 6},
        in_delay_ns=2.05,
        min_period_ns=1.44,
        out_delay_ns=0.71,
    ),
    # IEEE fp16
    ip.hadd: IPRow(
        2,
        {"lut": 188, "ff": 40, "carry8": 12, "muxf": 3},
        in_delay_ns=2.12,
        min_period_ns=2.49,
        out_delay_ns=1.04,
        variant="nodsp",
    ),
    ip.hsub: IPRow(
        2,
        {"lut": 188, "ff": 40, "carry8": 12, "muxf": 3},
        in_delay_ns=2.12,
        min_period_ns=2.49,
        out_delay_ns=1.04,
        variant="nodsp",
    ),
    ip.hmul: IPRow(
        2,
        {"lut": 46, "ff": 28, "dsp": 1, "carry8": 6},
        in_delay_ns=1.38,
        min_period_ns=1.71,
        out_delay_ns=0.77,
    ),
    ip.hdiv: IPRow(
        6,
        {"lut": 216, "slicem_lut": 17, "ff": 140, "carry8": 29},
        in_delay_ns=1.57,
        min_period_ns=2.14,
        out_delay_ns=0.38,
    ),
    ip.hcmp: IPRow(
        1,
        {"lut": 7, "ff": 1, "carry8": 2},
        in_delay_ns=0.84,
        min_period_ns=0.0,
        out_delay_ns=0.33,
    ),
    ip.i2f: IPRow(
        3,
        {"lut": 163, "slicem_lut": 1, "ff": 95, "carry8": 11, "muxf": 3},
        in_delay_ns=0.61,
        min_period_ns=1.81,
        out_delay_ns=0.4,
    ),
    ip.f2i: IPRow(
        3,
        {"lut": 174, "ff": 121, "carry8": 6, "muxf": 2},
        in_delay_ns=0.76,
        min_period_ns=1.43,
        out_delay_ns=0.43,
    ),
    ip.fcvt: IPRow(
        2,
        {"lut": 50, "ff": 97, "carry8": 1},
        in_delay_ns=0.4,
        min_period_ns=0.85,
        out_delay_ns=0.57,
    ),
    ip.bf2f: IPRow(
        2,
        {"lut": 34, "ff": 51, "carry8": 1},
        in_delay_ns=0.26,
        min_period_ns=0.74,
        out_delay_ns=0.43,
    ),
    # Each multiply also carries a `mullut` row: the same core built in fabric
    # instead of DSP columns, declared at the depth of the DSP row it competes
    # with. Selection ranks depth before price, so a shorter fabric row would
    # take every multiply; at equal depth a `dsp` resource weight picks between
    # them by what each spends.
    ip.imul16: (
        IPRow(3, {"dsp": 1}, in_delay_ns=0.08, min_period_ns=0.89, out_delay_ns=0.74),
        IPRow(1, {"dsp": 1}, in_delay_ns=1.41, min_period_ns=0.27, out_delay_ns=1.11),
        IPRow(
            1,
            {"lut": 192, "ff": 16, "carry8": 22},
            in_delay_ns=2.01,
            min_period_ns=0.0,
            out_delay_ns=0.46,
            mnemonic="mullut",
            variant="nodsp",
        ),
    ),
    ip.imul32: (
        IPRow(
            2,
            {"ff": 17, "dsp": 3},
            in_delay_ns=0.31,
            min_period_ns=2.9,
            out_delay_ns=0.67,
        ),
        IPRow(
            2,
            {"lut": 768, "ff": 112, "carry8": 76},
            in_delay_ns=2.06,
            min_period_ns=1.41,
            out_delay_ns=0.35,
            mnemonic="mullut",
            variant="nodsp",
        ),
        # A combinational 3-DSP cascade up to its output register: routed in
        # context the cone runs 3.0 ns (2.9 ns of DSP logic plus route), which
        # rules the row out at 300 MHz and leaves it to lower targets.
        IPRow(
            1,
            {"ff": 17, "dsp": 3},
            in_delay_ns=2.7,
            min_period_ns=2.69,
            out_delay_ns=0.74,
        ),
    ),
    ip.imul64: (
        IPRow(
            6,
            {"slicem_lut": 64, "ff": 81, "dsp": 10},
            in_delay_ns=0.52,
            min_period_ns=2.96,
            out_delay_ns=0.52,
        ),
        IPRow(
            6,
            {"lut": 3072, "slicem_lut": 16, "ff": 2168, "carry8": 280},
            in_delay_ns=1.27,
            min_period_ns=1.25,
            out_delay_ns=0.52,
            mnemonic="mullut",
            variant="nodsp",
        ),
    ),
    ip.imulw33: IPRow(
        3, {"ff": 34, "dsp": 4}, in_delay_ns=0.12, min_period_ns=2.28, out_delay_ns=0.98
    ),
    ip.imuladd32: IPRow(
        3,
        {"lut": 47, "ff": 113, "dsp": 3, "carry8": 6},
        in_delay_ns=0.17,
        min_period_ns=2.22,
        out_delay_ns=0.47,
    ),
    ip.idiv8: IPRow(
        4,
        {"lut": 128, "ff": 78, "carry8": 17},
        in_delay_ns=0.34,
        min_period_ns=3.02,
        out_delay_ns=0.38,
    ),
    ip.udiv8: IPRow(
        4,
        {"lut": 102, "ff": 88, "carry8": 17},
        in_delay_ns=0.03,
        min_period_ns=2.75,
        out_delay_ns=0.83,
    ),
    ip.irem8: IPRow(
        4,
        {"lut": 128, "ff": 78, "carry8": 17},
        in_delay_ns=0.34,
        min_period_ns=3.02,
        out_delay_ns=0.38,
    ),
    ip.urem8: IPRow(
        4,
        {"lut": 102, "ff": 88, "carry8": 17},
        in_delay_ns=0.03,
        min_period_ns=2.75,
        out_delay_ns=0.83,
    ),
    ip.idiv16: IPRow(
        8,
        {"lut": 378, "slicem_lut": 2, "ff": 340, "carry8": 58},
        in_delay_ns=0.58,
        min_period_ns=2.9,
        out_delay_ns=0.44,
    ),
    ip.udiv16: IPRow(
        8,
        {"lut": 334, "ff": 368, "carry8": 50},
        in_delay_ns=0.0,
        min_period_ns=2.46,
        out_delay_ns=0.83,
    ),
    ip.irem16: IPRow(
        8,
        {"lut": 378, "slicem_lut": 2, "ff": 340, "carry8": 58},
        in_delay_ns=0.58,
        min_period_ns=2.9,
        out_delay_ns=0.44,
    ),
    ip.urem16: IPRow(
        8,
        {"lut": 334, "ff": 368, "carry8": 50},
        in_delay_ns=0.0,
        min_period_ns=2.46,
        out_delay_ns=0.83,
    ),
    ip.idiv32: IPRow(
        16,
        {"lut": 1274, "slicem_lut": 2, "ff": 1444, "carry8": 180},
        in_delay_ns=0.46,
        min_period_ns=2.95,
        out_delay_ns=0.63,
    ),
    ip.udiv32: IPRow(
        16,
        {"lut": 1182, "ff": 1504, "carry8": 164},
        in_delay_ns=0.3,
        min_period_ns=2.94,
        out_delay_ns=1.11,
    ),
    ip.irem32: IPRow(
        16,
        {"lut": 1274, "slicem_lut": 2, "ff": 1444, "carry8": 180},
        in_delay_ns=0.46,
        min_period_ns=2.95,
        out_delay_ns=0.63,
    ),
    ip.urem32: IPRow(
        16,
        {"lut": 1182, "ff": 1504, "carry8": 164},
        in_delay_ns=0.3,
        min_period_ns=2.94,
        out_delay_ns=1.11,
    ),
    ip.idiv64: IPRow(
        68,
        {"lut": 4665, "slicem_lut": 6, "ff": 12804, "carry8": 616},
        in_delay_ns=1.2,
        min_period_ns=1.67,
        out_delay_ns=0.78,
    ),
    ip.udiv64: IPRow(
        32,
        {"lut": 4414, "ff": 6080, "carry8": 584},
        in_delay_ns=0.27,
        min_period_ns=3.19,
        out_delay_ns=1.33,
    ),
    ip.irem64: IPRow(
        68,
        {"lut": 4665, "slicem_lut": 6, "ff": 12804, "carry8": 616},
        in_delay_ns=1.2,
        min_period_ns=1.67,
        out_delay_ns=0.78,
    ),
    ip.urem64: IPRow(
        32,
        {"lut": 4414, "ff": 6080, "carry8": 584},
        in_delay_ns=0.27,
        min_period_ns=3.19,
        out_delay_ns=1.33,
    ),
}


#: Rows that replace the base entry for their archetype at one grade, with a
#: tuple on either side standing for the whole candidate set.
IP_BY_GRADE: Mapping[Grade, Mapping[OperatorIP, IPRow | tuple[IPRow, ...]]] = {
    # The 2-cycle unsigned 8-bit divider (311 MHz against 249) and the 24-cycle
    # double divider (308 against 208) close on -2L and miss the same 300 MHz
    # clock on -2LV, so each is a candidate at this grade only.
    GRADE_2L: {
        ip.udiv8: (
            IPRow(
                4,
                {"lut": 102, "ff": 88, "carry8": 17},
                in_delay_ns=0.03,
                min_period_ns=2.75,
                out_delay_ns=0.83,
            ),
            IPRow(
                2,
                {"lut": 99, "ff": 40, "carry8": 17},
                in_delay_ns=0.06,
                min_period_ns=3.08,
                out_delay_ns=2.6,
            ),
        ),
        ip.urem8: (
            IPRow(
                4,
                {"lut": 102, "ff": 88, "carry8": 17},
                in_delay_ns=0.03,
                min_period_ns=2.75,
                out_delay_ns=0.83,
            ),
            IPRow(
                2,
                {"lut": 99, "ff": 40, "carry8": 17},
                in_delay_ns=0.06,
                min_period_ns=3.08,
                out_delay_ns=2.6,
            ),
        ),
        ip.ddiv: (
            IPRow(
                32,
                {"lut": 3189, "slicem_lut": 70, "ff": 3017, "carry8": 398},
                in_delay_ns=1.73,
                min_period_ns=2.56,
                out_delay_ns=0.59,
            ),
            IPRow(
                24,
                {"lut": 3181, "slicem_lut": 70, "ff": 2057, "carry8": 398},
                in_delay_ns=2.7,
                min_period_ns=3.29,
                out_delay_ns=0.51,
            ),
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
                7,
                {"lut": 251, "slicem_lut": 12, "ff": 234, "dsp": 2, "carry8": 10},
                in_delay_ns=0.91,
                min_period_ns=2.81,
                out_delay_ns=0.65,
            ),
            IPRow(
                5,
                {"lut": 364, "slicem_lut": 12, "ff": 236, "carry8": 17, "muxf": 2},
                in_delay_ns=2.04,
                min_period_ns=2.5,
                out_delay_ns=0.5,
                variant="nodsp",
            ),
            IPRow(
                2,
                {"lut": 386, "ff": 72, "carry8": 17, "muxf": 4},
                in_delay_ns=3.03,
                min_period_ns=3.63,
                out_delay_ns=0.75,
                variant="nodsp",
            ),
            IPRow(
                1,
                {"lut": 367, "ff": 32, "carry8": 17, "muxf": 2},
                in_delay_ns=6.27,
                min_period_ns=0.64,
                out_delay_ns=0.65,
                variant="nodsp",
            ),
        ),
        ip.fsub: (
            IPRow(
                7,
                {"lut": 252, "slicem_lut": 12, "ff": 234, "dsp": 2, "carry8": 10},
                in_delay_ns=1.52,
                min_period_ns=2.95,
                out_delay_ns=0.68,
            ),
            IPRow(
                5,
                {"lut": 364, "slicem_lut": 12, "ff": 236, "carry8": 17, "muxf": 2},
                in_delay_ns=2.04,
                min_period_ns=2.5,
                out_delay_ns=0.5,
                variant="nodsp",
            ),
            IPRow(
                2,
                {"lut": 386, "ff": 72, "carry8": 17, "muxf": 4},
                in_delay_ns=3.03,
                min_period_ns=3.63,
                out_delay_ns=0.75,
                variant="nodsp",
            ),
            IPRow(
                1,
                {"lut": 367, "ff": 32, "carry8": 17, "muxf": 2},
                in_delay_ns=6.27,
                min_period_ns=0.64,
                out_delay_ns=0.65,
                variant="nodsp",
            ),
        ),
        ip.fmul: (
            IPRow(
                4,
                {"lut": 106, "ff": 102, "dsp": 2, "carry8": 9},
                in_delay_ns=1.91,
                min_period_ns=2.1,
                out_delay_ns=0.66,
            ),
            IPRow(
                2,
                {"lut": 79, "ff": 48, "dsp": 2, "carry8": 8},
                in_delay_ns=2.52,
                min_period_ns=2.15,
                out_delay_ns=0.66,
            ),
            IPRow(
                1,
                {"lut": 80, "ff": 32, "dsp": 2, "carry8": 8},
                in_delay_ns=4.52,
                min_period_ns=4.58,
                out_delay_ns=0.68,
            ),
            # The maximum-DSP build, at the depths where it is this grade's
            # fastest core of that depth. Not the same depths as -2L's.
            IPRow(
                4,
                {"lut": 84, "ff": 70, "dsp": 3, "carry8": 4},
                in_delay_ns=1.31,
                min_period_ns=2.31,
                out_delay_ns=0.52,
                mnemonic="mul_maxdsp",
                variant="maxdsp",
            ),
            IPRow(
                2,
                {"lut": 79, "ff": 47, "dsp": 3, "carry8": 3},
                in_delay_ns=2.03,
                min_period_ns=3.06,
                out_delay_ns=0.61,
                mnemonic="mul_maxdsp",
                variant="maxdsp",
            ),
            IPRow(
                1,
                {"lut": 75, "ff": 32, "dsp": 3, "carry8": 3},
                in_delay_ns=4.25,
                min_period_ns=4.3,
                out_delay_ns=0.58,
                mnemonic="mul_maxdsp",
                variant="maxdsp",
            ),
        ),
        ip.dadd: (
            IPRow(
                14,
                {
                    "lut": 700,
                    "slicem_lut": 89,
                    "ff": 862,
                    "dsp": 3,
                    "carry8": 30,
                    "muxf": 11,
                },
                in_delay_ns=1.4,
                min_period_ns=2.19,
                out_delay_ns=0.89,
            ),
            IPRow(
                6,
                {"lut": 704, "slicem_lut": 15, "ff": 536, "carry8": 40, "muxf": 11},
                in_delay_ns=2.03,
                min_period_ns=2.58,
                out_delay_ns=1.26,
                variant="nodsp",
            ),
            IPRow(
                3,
                {"lut": 711, "ff": 277, "carry8": 38, "muxf": 11},
                in_delay_ns=2.89,
                min_period_ns=3.64,
                out_delay_ns=1.18,
                variant="nodsp",
            ),
            IPRow(
                1,
                {"lut": 812, "ff": 64, "carry8": 38, "muxf": 8},
                in_delay_ns=8.81,
                min_period_ns=0.72,
                out_delay_ns=0.76,
                variant="nodsp",
            ),
        ),
        ip.dsub: (
            IPRow(
                14,
                {
                    "lut": 701,
                    "slicem_lut": 89,
                    "ff": 862,
                    "dsp": 3,
                    "carry8": 30,
                    "muxf": 11,
                },
                in_delay_ns=1.25,
                min_period_ns=1.95,
                out_delay_ns=0.7,
            ),
            IPRow(
                6,
                {"lut": 704, "slicem_lut": 15, "ff": 536, "carry8": 40, "muxf": 11},
                in_delay_ns=2.03,
                min_period_ns=2.58,
                out_delay_ns=1.26,
                variant="nodsp",
            ),
            IPRow(
                3,
                {"lut": 711, "ff": 277, "carry8": 38, "muxf": 11},
                in_delay_ns=2.89,
                min_period_ns=3.64,
                out_delay_ns=1.18,
                variant="nodsp",
            ),
            IPRow(
                1,
                {"lut": 812, "ff": 64, "carry8": 38, "muxf": 8},
                in_delay_ns=8.81,
                min_period_ns=0.72,
                out_delay_ns=0.76,
                variant="nodsp",
            ),
        ),
        ip.dmul: (
            IPRow(
                9,
                {"lut": 192, "slicem_lut": 61, "ff": 390, "dsp": 7, "carry8": 15},
                in_delay_ns=1.58,
                min_period_ns=2.06,
                out_delay_ns=0.59,
            ),
            IPRow(
                8,
                {"lut": 190, "slicem_lut": 62, "ff": 353, "dsp": 7, "carry8": 15},
                in_delay_ns=1.46,
                min_period_ns=2.12,
                out_delay_ns=0.68,
            ),
            IPRow(
                3,
                {"lut": 133, "ff": 129, "dsp": 7, "carry8": 15},
                in_delay_ns=2.59,
                min_period_ns=3.49,
                out_delay_ns=0.76,
            ),
            IPRow(
                1,
                {"lut": 143, "ff": 64, "dsp": 7, "carry8": 15},
                in_delay_ns=8.76,
                min_period_ns=8.8,
                out_delay_ns=0.66,
            ),
        ),
        ip.fdiv: IPRow(
            16,
            {"lut": 758, "slicem_lut": 37, "ff": 665, "carry8": 111},
            in_delay_ns=1.63,
            min_period_ns=2.74,
            out_delay_ns=0.91,
        ),
        # This grade holds its multiplies at other depths than -2L, so each
        # fabric row sits at the depth of the DSP row beside it here.
        ip.imul16: (
            IPRow(
                3, {"dsp": 1}, in_delay_ns=0.0, min_period_ns=1.14, out_delay_ns=0.95
            ),
            IPRow(
                1, {"dsp": 1}, in_delay_ns=1.78, min_period_ns=0.34, out_delay_ns=0.95
            ),
            IPRow(
                1,
                {"lut": 192, "ff": 16, "carry8": 22},
                in_delay_ns=2.27,
                min_period_ns=0.0,
                out_delay_ns=0.69,
                mnemonic="mullut",
                variant="nodsp",
            ),
        ),
        ip.imul32: (
            IPRow(
                3,
                {"ff": 32, "dsp": 3},
                in_delay_ns=0.08,
                min_period_ns=2.89,
                out_delay_ns=0.54,
            ),
            IPRow(
                3,
                {"lut": 768, "ff": 384, "carry8": 76},
                in_delay_ns=2.24,
                min_period_ns=2.19,
                out_delay_ns=0.52,
                mnemonic="mullut",
                variant="nodsp",
            ),
        ),
        ip.imul64: (
            IPRow(
                8,
                {"slicem_lut": 111, "ff": 158, "dsp": 10},
                in_delay_ns=0.29,
                min_period_ns=3.04,
                out_delay_ns=0.61,
            ),
            IPRow(
                8,
                {"lut": 3072, "slicem_lut": 80, "ff": 2232, "carry8": 280},
                in_delay_ns=1.5,
                min_period_ns=1.77,
                out_delay_ns=0.95,
                mnemonic="mullut",
                variant="nodsp",
            ),
        ),
        ip.idiv8: IPRow(
            12,
            {"lut": 135, "slicem_lut": 2, "ff": 260, "carry8": 17},
            in_delay_ns=0.39,
            min_period_ns=1.16,
            out_delay_ns=0.47,
        ),
        ip.irem8: IPRow(
            12,
            {"lut": 135, "slicem_lut": 2, "ff": 260, "carry8": 17},
            in_delay_ns=0.39,
            min_period_ns=1.16,
            out_delay_ns=0.47,
        ),
        ip.udiv8: IPRow(
            4,
            {"lut": 102, "ff": 88, "carry8": 17},
            in_delay_ns=0.0,
            min_period_ns=3.35,
            out_delay_ns=0.9,
        ),
        ip.urem8: IPRow(
            4,
            {"lut": 102, "ff": 88, "carry8": 17},
            in_delay_ns=0.0,
            min_period_ns=3.35,
            out_delay_ns=0.9,
        ),
        ip.idiv16: IPRow(
            20,
            {"lut": 393, "slicem_lut": 2, "ff": 900, "carry8": 58},
            in_delay_ns=0.48,
            min_period_ns=1.43,
            out_delay_ns=0.56,
        ),
        ip.irem16: IPRow(
            20,
            {"lut": 393, "slicem_lut": 2, "ff": 900, "carry8": 58},
            in_delay_ns=0.48,
            min_period_ns=1.43,
            out_delay_ns=0.56,
        ),
        ip.udiv16: IPRow(
            8,
            {"lut": 334, "ff": 368, "carry8": 50},
            in_delay_ns=0.01,
            min_period_ns=3.03,
            out_delay_ns=1.25,
        ),
        ip.urem16: IPRow(
            8,
            {"lut": 334, "ff": 368, "carry8": 50},
            in_delay_ns=0.01,
            min_period_ns=3.03,
            out_delay_ns=1.25,
        ),
        ip.idiv32: IPRow(
            36,
            {"lut": 1305, "slicem_lut": 4, "ff": 3332, "carry8": 180},
            in_delay_ns=0.75,
            min_period_ns=1.85,
            out_delay_ns=0.89,
        ),
        ip.irem32: IPRow(
            36,
            {"lut": 1305, "slicem_lut": 4, "ff": 3332, "carry8": 180},
            in_delay_ns=0.75,
            min_period_ns=1.85,
            out_delay_ns=0.89,
        ),
        ip.udiv32: IPRow(
            34,
            {"lut": 1182, "ff": 3200, "carry8": 164},
            in_delay_ns=0.57,
            min_period_ns=1.77,
            out_delay_ns=0.82,
        ),
        ip.urem32: IPRow(
            34,
            {"lut": 1182, "ff": 3200, "carry8": 164},
            in_delay_ns=0.57,
            min_period_ns=1.77,
            out_delay_ns=0.82,
        ),
        ip.idiv64: IPRow(
            68,
            {"lut": 4665, "slicem_lut": 6, "ff": 12804, "carry8": 616},
            in_delay_ns=1.35,
            min_period_ns=2.3,
            out_delay_ns=0.71,
        ),
        ip.irem64: IPRow(
            68,
            {"lut": 4665, "slicem_lut": 6, "ff": 12804, "carry8": 616},
            in_delay_ns=1.35,
            min_period_ns=2.3,
            out_delay_ns=0.71,
        ),
        ip.udiv64: IPRow(
            66,
            {"lut": 4414, "ff": 12544, "carry8": 584},
            in_delay_ns=0.74,
            min_period_ns=2.19,
            out_delay_ns=0.7,
        ),
        ip.urem64: IPRow(
            66,
            {"lut": 4414, "ff": 12544, "carry8": 584},
            in_delay_ns=0.74,
            min_period_ns=2.19,
            out_delay_ns=0.7,
        ),
        # Stated rather than inherited: the base table is the -2L die, where
        # the same cores run up to 1.6x faster.
        ip.fcmp: IPRow(
            1,
            {"lut": 13, "ff": 1, "carry8": 2},
            in_delay_ns=0.93,
            min_period_ns=0.0,
            out_delay_ns=0.46,
        ),
        ip.fsqrt: IPRow(
            8,
            {"lut": 431, "slicem_lut": 12, "ff": 283, "carry8": 67},
            in_delay_ns=2.33,
            min_period_ns=4.15,
            out_delay_ns=0.52,
        ),
        ip.ddiv: IPRow(
            32,
            {"lut": 3189, "slicem_lut": 70, "ff": 3017, "carry8": 398},
            in_delay_ns=2.57,
            min_period_ns=3.27,
            out_delay_ns=0.65,
        ),
        ip.dcmp: IPRow(
            1,
            {"lut": 23, "ff": 1, "carry8": 4},
            in_delay_ns=0.98,
            min_period_ns=0.0,
            out_delay_ns=0.46,
        ),
        ip.dsqrt: IPRow(
            20,
            {"lut": 1695, "slicem_lut": 50, "ff": 1203, "carry8": 243},
            in_delay_ns=2.16,
            min_period_ns=4.26,
            out_delay_ns=0.72,
        ),
        ip.bfadd: IPRow(
            4,
            {"lut": 195, "ff": 113, "carry8": 12},
            in_delay_ns=1.42,
            min_period_ns=2.14,
            out_delay_ns=0.59,
        ),
        ip.bfsub: IPRow(
            4,
            {"lut": 195, "ff": 113, "carry8": 12},
            in_delay_ns=1.42,
            min_period_ns=2.14,
            out_delay_ns=0.59,
        ),
        ip.bfmul: IPRow(
            2,
            {"lut": 58, "ff": 31, "dsp": 1, "carry8": 6},
            in_delay_ns=1.73,
            min_period_ns=1.8,
            out_delay_ns=0.69,
        ),
        ip.hadd: IPRow(
            2,
            {"lut": 188, "ff": 40, "carry8": 12, "muxf": 3},
            in_delay_ns=2.2,
            min_period_ns=2.9,
            out_delay_ns=0.62,
            variant="nodsp",
        ),
        ip.hsub: IPRow(
            2,
            {"lut": 188, "ff": 40, "carry8": 12, "muxf": 3},
            in_delay_ns=2.2,
            min_period_ns=2.9,
            out_delay_ns=0.62,
            variant="nodsp",
        ),
        ip.hmul: IPRow(
            2,
            {"lut": 46, "ff": 28, "dsp": 1, "carry8": 6},
            in_delay_ns=1.79,
            min_period_ns=1.81,
            out_delay_ns=0.94,
        ),
        ip.hdiv: IPRow(
            6,
            {"lut": 216, "slicem_lut": 17, "ff": 140, "carry8": 29},
            in_delay_ns=2.14,
            min_period_ns=2.93,
            out_delay_ns=0.8,
        ),
        ip.hcmp: IPRow(
            1,
            {"lut": 7, "ff": 1, "carry8": 2},
            in_delay_ns=0.52,
            min_period_ns=0.0,
            out_delay_ns=0.43,
        ),
        ip.i2f: IPRow(
            3,
            {"lut": 163, "slicem_lut": 1, "ff": 95, "carry8": 11, "muxf": 3},
            in_delay_ns=0.7,
            min_period_ns=2.44,
            out_delay_ns=0.56,
        ),
        ip.f2i: IPRow(
            3,
            {"lut": 174, "ff": 121, "carry8": 6, "muxf": 2},
            in_delay_ns=1.07,
            min_period_ns=2.24,
            out_delay_ns=0.5,
        ),
        ip.fcvt: IPRow(
            2,
            {"lut": 50, "ff": 97, "carry8": 1},
            in_delay_ns=0.48,
            min_period_ns=1.04,
            out_delay_ns=0.65,
        ),
        ip.bf2f: IPRow(
            2,
            {"lut": 34, "ff": 51, "carry8": 1},
            in_delay_ns=0.26,
            min_period_ns=0.92,
            out_delay_ns=0.52,
        ),
        ip.imulw33: IPRow(
            3,
            {"ff": 34, "dsp": 4},
            in_delay_ns=0.06,
            min_period_ns=3.04,
            out_delay_ns=0.9,
        ),
        ip.imuladd32: IPRow(
            3,
            {"lut": 47, "ff": 113, "dsp": 3, "carry8": 6},
            in_delay_ns=0.06,
            min_period_ns=2.85,
            out_delay_ns=0.76,
        ),
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
        lambda r: {r["lut"]: ROM_LUT_COST, r["muxf"]: ROM_MUXF_COST},
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
    # A carry chain is used only from 16 bits up; below that the adder is built
    # in LUTs, 3 more of them than the wide form's one a bit.
    addsub = {
        lut: Piecewise(16, Linear(1.0, base=3.0), Linear(1.0)),
        carry8: Piecewise(16, Const(0.0), Tiled(8)),
    }
    compare = {lut: Linear(1.0), carry8: Tiled(16)}
    minmax = {lut: Linear(2.0), carry8: Tiled(16)}
    # The DSP count steps in whole slices; the fabric logic around them grows
    # with width, and past 64 bits so does the partial-product tree's carry.
    multiply = {
        lut: Interp({8: 39, 16: 0, 32: 15, 64: 41, 96: 153, 128: 316}),
        dsp: Table({8: 0, 16: 1, 32: 3, 64: 10, 96: 21, 128: 34}),
        carry8: Interp({8: 3, 16: 0, 32: 2, 64: 4, 96: 14, 128: 31}),
    }
    # One core computes quotient and remainder, so they share a carry chain
    # count, and the remainder costs the extra LUTs that restore it.
    divrem_carry = Interp({8: 14, 16: 49, 32: 171, 64: 599, 96: 1283, 128: 2223})
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
        CombKind.SHL: {
            lut: Interp({8: 15, 16: 42, 32: 103, 64: 238, 96: 412, 128: 558})
        },
        CombKind.SHR: {
            lut: Interp({8: 15, 16: 44, 32: 107, 64: 265, 96: 427, 128: 573})
        },
        CombKind.MUL: multiply,
        CombKind.DIV: {
            lut: Interp({8: 104, 16: 339, 32: 1214, 64: 4478, 96: 9790, 128: 17150}),
            carry8: divrem_carry,
        },
        CombKind.REM: {
            lut: Interp({8: 125, 16: 376, 32: 1342, 64: 4729, 96: 10186, 128: 17573}),
            carry8: divrem_carry,
        },
        # Free, and measured so: a resize synthesizes to nothing, and a float
        # negate is one inverter that the LUT consuming it absorbs. The `negf`
        # DUT reports that inverter as a LUT, having no consumer standalone.
        CombKind.NEG: None,
        CombKind.INT_CAST: None,
    }


#: SLICEM sites one bit of an extracted chain takes. An SRL32E holds 32 stages
#: and the chain's first and last sit in flip-flops, so the shift registers hold
#: `depth - 2`. Measured 1 site a bit at depths 32 through 34 and 2 at 64
#: through 66.
_CHAIN_SRL = Piecewise(SRL_MIN_DEPTH, Const(0.0), Tiled(32, offset=-2.0))

#: A one-bit chain whose stages carry a reset is never extracted, however deep
#: it is: measured at 33, 64, 65, 66, 96, 97 and 128. Wider ones always are.
_RST_EXTRACTS = Piecewise(2, Const(0.0), Linear(1.0))


def _chain_uses(r: Mapping[str, Resource]) -> dict:
    """What one delay chain spends, over its depth and bit width."""
    per_stage = [
        (Linear(1.0, base=-1.0), Const(1.0)),
        (Piecewise(SRL_MIN_DEPTH, Linear(-1.0, base=1.0), Const(0.0)), Const(1.0)),
    ]
    return {
        r["ff"]: [(Step(SRL_MIN_DEPTH, 1.0, 2.0), Linear(1.0))] + per_stage,
        r["lut"]: (Step(SRL_MIN_DEPTH, 0.0, 1.0), _RST_EXTRACTS),
        r["slicem_lut"]: (_CHAIN_SRL, _RST_EXTRACTS),
    }


def _chain_uses_norst(r: Mapping[str, Resource]) -> dict:
    """The same chain with no synchronous reset: the SRL absorbs every interior
    stage, so only the two end registers stay in flip-flops."""
    return {
        r["ff"]: (Step(SRL_MIN_DEPTH, 1.0, 2.0), Linear(1.0)),
        r["slicem_lut"]: (_CHAIN_SRL, Linear(1.0)),
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

    add_ip_rows(d, {**IP, **IP_BY_GRADE.get(part.grade, {})}, res)

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
