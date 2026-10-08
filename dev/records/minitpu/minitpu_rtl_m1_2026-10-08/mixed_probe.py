# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""M-R1 first test: one RTLModule with a MemPort AND stream ports, inside a @df.region, target="simulator".

Region: feed (prog -> cmd stream), core (the IP call, the boundary RAM M), sink (status stream -> st array).
"""
import os
import sys
from pathlib import Path

import numpy as np
import allo.dataflow as df
from allo import RTLModule, Port, MemPort
from allo.ir.types import int32, Stream

HERE = Path(__file__).resolve().parent
NCMD = 6

ip = RTLModule(
    "mixed_probe", HERE / "mixed_probe.sv", name="mixed_probe_sim",
    ports=[Port("cmd", "cmd_data", "cmd_valid", "cmd_ready", size=NCMD),
           Port("st", "st_data", "st_valid", "st_ready", dir="out"),
           MemPort("M", 16, "int32_t", "m_addr", "m_ce", q="m_q", we="m_we", d="m_d")],
    clock="clk", reset="rst_n", reset_active_high=False, start=None, done="done",
    verilator_args=["--build-jobs", "4"])


@df.region()
def top(P: int32[NCMD], M: int32[16], S: int32[NCMD]):
    cmd: Stream[int32, 4]
    st: Stream[int32, 4]

    @df.kernel(mapping=[1], args=[P])
    def feed(p: int32[NCMD]):
        for i in range(NCMD):
            cmd.put(p[i])

    @df.kernel(mapping=[1], args=[M])
    def core(m: int32[16]):
        ip(cmd, st, m)

    @df.kernel(mapping=[1], args=[S])
    def sink(s: int32[NCMD]):
        for i in range(NCMD):
            s[i] = st.get()


mod = df.build(top, target="simulator")
for trial in range(2):
    mem = np.arange(16, dtype=np.int32) * 10
    addrs, incs = [3, 5, 3, 0, 15, 3], [1, 2, 4, 8, 16, 32]
    prog = np.array([(inc << 16) | a for a, inc in zip(addrs, incs)], dtype=np.int32)
    prog[-1] |= 1 << 15
    want_mem, want_st = mem.copy(), []
    for a, inc in zip(addrs, incs):
        want_st.append(want_mem[a])
        want_mem[a] += inc
    st = np.zeros(NCMD, dtype=np.int32)
    mod(prog, mem, st)
    ok = st.tolist() == want_st and mem.tolist() == want_mem.tolist()
    print(f"MIXED trial {trial}: {'PASS' if ok else 'FAIL'} st={st.tolist()} want={want_st} "
          f"mem[0,3,5,15]={mem[[0, 3, 5, 15]].tolist()} want={want_mem[[0, 3, 5, 15]].tolist()}", flush=True)
