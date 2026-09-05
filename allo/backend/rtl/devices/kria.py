# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kria system-on-modules: Zynq UltraScale+ silicon on the `ultrascalex`
fabric, differing from the Alveo cards in capacity and speed grade only."""

from __future__ import annotations

from . import ultrascalex
from .spec import Part

#: Capacities read off the part itself with `get_property`, not off a data
#: sheet.
kv260 = ultrascalex.build(
    Part(
        name="kv260",
        part="xck26-sfvc784-2LV-c",
        grade=ultrascalex.GRADE_2LV,
        capacity={
            "lut": 117_120,
            "ff": 234_240,
            "dsp": 1_248,
            "bram36": 144,
            "uram288": 64,
        },
    )
)

DEVICES = (kv260,)
