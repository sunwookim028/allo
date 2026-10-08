# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# epoch form, queue full (no pops), then two flushes on consecutive cycles
# U4 track A record, T-5: run with the epoch form of examples/minitpu/template/flush_stream.py.
# With the 1-bit tag test (before the fix) gap 0 issued 2, 3 before target 4; now 4 first.
import sys
import allo.dataflow as df
from examples.minitpu.template import flush_stream as F
from examples.minitpu.units import fetch
from examples.minitpu.harness.traces import Trace
gap = int(sys.argv[1])
t = Trace(fetch._ports("iram"))
t.idle(2, rst_n=0)
for a in range(8):
    t.cycle(instr_write_en=1, iram_addr=a, dma_iram_din=0x100 + a, pause=1)
t.cycle(pc_flush=1, restart_addr=0, pause=1)
t.idle(8)                      # fill: no pops
t.cycle(pc_flush=1, restart_addr=2)
t.idle(gap)
t.cycle(pc_flush=1, restart_addr=4)
t.idle(10, bundle_pop=1)
cmd = t.cmd()
n = len(cmd["rst_n"])
mod = df.build(F.architecture(n, "epoch").region("simulator"), target="simulator")
got = fetch.run_f1(mod, cmd, n, 128)
print("gap", gap, "completed; addr", [int(x) for x in got["bundle_addr"][-10:]], flush=True)
