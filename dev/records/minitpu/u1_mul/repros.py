# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal repros for u1_mul_2026-10-02.rst (S4, S5, N1).

    $ALLO_PYTHON dev/records/minitpu/u1_mul/repros.py [s4|s5|n1] [<scratch dir>]
"""

import os
import sys

import ml_dtypes
import numpy as np

import allo.dataflow as df
from allo.ir.types import UInt, bfloat16, float32, uint16

ROOT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/u1_mul_repros"


def s4():
    """bf16 -> f32 widening does not compile in SystemC csim."""

    @df.region()
    def top(A: bfloat16[4], C: float32[4]):
        @df.kernel(mapping=[1], args=[A, C])
        def k(a: bfloat16[4], c: float32[4]):
            for i in range(4):
                c[i] = a[i]  # arith.extf

    a = np.array([1.5, -2, 3, 0.25], dtype=ml_dtypes.bfloat16)
    c = np.zeros(4, np.float32)
    df.build(top, target="simulator")(a, c)
    print("simulator", c)
    mod = df.build(top, target="systemc", mode="csim", project=os.path.join(ROOT, "s4"))
    mod(a, c)  # g++: conversion from 'ac::bfloat16' to non-scalar type
    print("systemc", c)  # 'ac_ieee_float<binary32>' requested


def s5():
    """A UInt(24) output port: the simulator runs it, SystemC csim cannot
    read it back (KeyError: 'ui24')."""

    @df.region()
    def top(A: uint16[4], C: UInt(24)[4]):
        @df.kernel(mapping=[1], args=[A, C])
        def k(a: uint16[4], c: UInt(24)[4]):
            for i in range(4):
                c[i] = a[i]

    a = np.arange(4, dtype=np.uint16)
    c = np.zeros(4, np.uint32)
    df.build(top, target="simulator")(a, c)
    print("simulator", c)
    df.build(top, target="systemc", mode="csim", project=os.path.join(ROOT, "s5"))(a, c)
    print("systemc", c)


def n1():
    """The same bf16 multiply: NaN sign differs between the two backends."""

    @df.region()
    def top(A: bfloat16[4], B: bfloat16[4], C: bfloat16[4]):
        @df.kernel(mapping=[1], args=[A, B, C])
        def k(a: bfloat16[4], b: bfloat16[4], c: bfloat16[4]):
            for i in range(4):
                c[i] = a[i] * b[i]

    a = np.array([0x7FC0, 0xFFC0, 0x7F80, 0x0000], np.uint16)
    b = np.array([0xBF80, 0x3F80, 0x0000, 0xFF80], np.uint16)  # -1, 1, 0, -inf
    for tgt, kw in (("simulator", {}),
                    ("systemc", dict(mode="csim", project=os.path.join(ROOT, "n1")))):
        c = np.zeros(4, ml_dtypes.bfloat16)
        df.build(top, target=tgt, **kw)(a.view(ml_dtypes.bfloat16),
                                         b.view(ml_dtypes.bfloat16), c)
        print(tgt, [hex(x) for x in c.view(np.uint16)])
    # simulator  ['0x7fc0', '0xffc0', '0xffc0', '0xffc0']  (NaN operand's sign;
    #                                                      x86 -NaN for Inf x 0)
    # systemc    ['0xffc0', '0xffc0', '0x7fc0', '0xffc0']  (sign = a^b)
    # MiniTPU    ['0x7fc0', '0x7fc0', '0x7fc0', '0x7fc0']


if __name__ == "__main__":
    {"s4": s4, "s5": s5, "n1": n1}[sys.argv[1] if len(sys.argv) > 1 else "n1"]()
