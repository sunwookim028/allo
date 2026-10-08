# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""M-R0 probe (b): PR #48's MemPort bound to a region boundary int32 array, under target="simulator".

Needs a tree with PR #48 (allo/backend/rtl.py); run from that tree's root. The RTL is the PR's own
tests/ip_integration/rtl/memory.v: per call it reads X[0] and writes X[0] + 1 (a one-word RAM client).
Case "llvm" is the PR's own test_memory_binding (the only MemPort path the PR tests); case "simulator" is
the same IP called by a kernel of a @df.region whose boundary array is X.
"""
import sys
import traceback
from pathlib import Path

import numpy as np
import allo
import allo.dataflow as df
from allo import RTLModule, MemPort
from allo.ir.types import int32

RTL = Path("tests/ip_integration/rtl/memory.v").resolve()


def ip(name):
    return RTLModule("memory", RTL, name=name, done="ap_done",
                     ports=[MemPort("X", 4, "int32_t", "addr", "ce", q="q", we="we", d="d")])


def case_llvm():
    mem = ip("mem_llvm")

    def top(X: int32[4]):
        for _ in range(2):
            mem(X)

    mod = allo.customize(top).build(target="llvm")
    x = np.array([9, 2, 3, 4], dtype=np.int32)
    mod(x)
    return x


def case_simulator():
    mem = ip("mem_sim")

    @df.region()
    def top(X: int32[4]):
        @df.kernel(mapping=[1], args=[X])
        def host(ddr: int32[4]):
            for _ in range(2):
                mem(ddr)

    mod = df.build(top, target="simulator")
    x = np.array([9, 2, 3, 4], dtype=np.int32)
    mod(x)
    return x


for name, case in (("llvm", case_llvm), ("simulator", case_simulator)):
    try:
        got = case()
        ok = got.tolist() == [11, 2, 3, 4]
        print(f"MEMPORT {name}: {'PASS' if ok else 'WRONG'} {got.tolist()} (want [11, 2, 3, 4])", flush=True)
    except Exception as error:  # the verdict is the point; print the whole error
        print(f"MEMPORT {name}: ERROR {type(error).__name__}: {error}", flush=True)
        traceback.print_exc(file=sys.stdout)
