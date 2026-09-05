# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Alveo data-centre cards."""

from __future__ import annotations

from . import ultrascalex
from .spec import Part

_2L = ultrascalex.GRADE_2L

u55c = ultrascalex.build(
    Part(
        name="u55c",
        part="xcu55c-fsvh2892-2L-e",
        grade=_2L,
        capacity={
            "lut": 1_303_680,
            "ff": 2_607_360,
            "dsp": 9_024,
            "bram36": 2_016,
            "uram288": 960,
        },
    )
)

u280 = ultrascalex.build(
    Part(
        name="u280",
        part="xcu280-fsvh2892-2L-e",
        grade=_2L,
        capacity={
            "lut": 1_303_680,
            "ff": 2_607_360,
            "dsp": 9_024,
            "bram36": 2_016,
            "uram288": 960,
        },
    )
)

u250 = ultrascalex.build(
    Part(
        name="u250",
        part="xcu250-figd2104-2L-e",
        grade=_2L,
        capacity={
            "lut": 1_728_000,
            "ff": 3_456_000,
            "dsp": 12_288,
            "bram36": 2_688,
            "uram288": 1_280,
        },
    )
)

DEVICES = (u55c, u280, u250)
