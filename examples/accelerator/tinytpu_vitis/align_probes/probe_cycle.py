# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Does a two-process cycle (X -> Y -> X) run on the Allo simulator, Vitis csim and cosim?
X owns mem; ops: 0 = send mem[a] to Y (Y doubles it and sends back) ; 1 = receive into mem[a]."""
import os, sys, subprocess, numpy as np, allo
from allo.ir.types import int32, UInt, Stream
import allo.dataflow as df
N = 16
@df.region()
def top(P: int32[64], O: int32[N]):
    cx: Stream[int32, 4]
    cy: Stream[int32, 4]
    xy: Stream[int32, 2]
    yx: Stream[int32, 2]
    @df.kernel(mapping=[1], args=[P])
    def seq(lP: int32[64]):
        n: int32 = lP[0]
        cx.put(n)
        cy.put(lP[63])
        for i in range(n):
            w: int32 = lP[1 + i]
            cx.put(w)
            if (w & 1) == 0:
                cy.put(w)
    @df.kernel(mapping=[1], args=[O])
    def X(lO: int32[N]):
        mem: int32[N]
        for i in range(N):
            mem[i] = i + 1
        n: int32 = cx.get()
        for i in range(n):
            w: int32 = cx.get()
            op: int32 = w & 1
            a: int32 = w >> 1
            if op == 0:
                xy.put(mem[a])
            else:
                mem[a] = yx.get()
        for i in range(N):
            lO[i] = mem[i]
    @df.kernel(mapping=[1])
    def Y():
        n: int32 = cy.get()
        for i in range(n):
            w: int32 = cy.get()
            v: int32 = xy.get()
            yx.put(v * 2)
# program: send a (op0) then receive into b (op1): mem[b] = 2*mem[a]
prog = [(0, 3), (1, 5), (0, 5), (1, 7), (0, 0), (1, 0)]
P = np.zeros(64, np.int32); P[0] = len(prog); P[63] = sum(1 for o, _ in prog if o == 0)
for i, (op, a) in enumerate(prog): P[1 + i] = (a << 1) | op
def gold():
    m = np.arange(1, N + 1, dtype=np.int32); pend = []
    for op, a in prog:
        if op == 0: pend.append(2 * m[a])
        else: m[a] = pend.pop(0)
    return m
mode = sys.argv[1] if len(sys.argv) > 1 else "sim"
if mode == "sim":
    mod = df.build(top, target="simulator")
    O = np.zeros(N, np.int32); mod(P, O)
    print("SIM", "OK" if (O == gold()).all() else f"WRONG {O} vs {gold()}")
else:
    prj = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".scratch", "cyc.prj")
    os.makedirs(prj, exist_ok=True)
    s = df.customize(top)
    s.build(target="vitis_hls", mode="csyn", project=prj, wrap_io=False)
    import re
    src = open(f"{prj}/kernel.cpp").read()
    dep = iter([64, N])
    src = re.sub(r"(#pragma HLS interface m_axi port=\w+ offset=slave bundle=gmem\d+)", lambda m: m.group(1) + f" depth={next(dep)}", src)
    open(f"{prj}/kernel.cpp", "w").write(src)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    import threaded_csim; print("threads:", threaded_csim.patch(f"{prj}/kernel.cpp", "top"))
    g = gold()
    tb = ("#include <cstdio>\n#include <cstdint>\nextern \"C\" void top(int32_t*, int32_t*);\n"
          f"static int32_t P[64] = {{{', '.join(map(str, P))}}};\nstatic int32_t G[{N}] = {{{', '.join(map(str, g))}}};\n"
          f"static int32_t O[{N}];\nint main(){{ top(P, O); int bad=0; for(int i=0;i<{N};i++) bad += O[i]!=G[i];"
          f" printf(\"TB mismatches = %d\\n\", bad); return bad!=0; }}\n")
    open(f"{prj}/tb.cpp", "w").write(tb)
    tcl = f"""open_project out.prj -reset
open_solution -reset solution1 -flow_target vivado
set_top top
add_files kernel.cpp
add_files -tb tb.cpp -cflags "-std=gnu++0x"
set_part {{xcu280-fsvh2892-2L-e}}
create_clock -period 3.33
csim_design -ldflags "-B/usr/bin -lpthread"

csynth_design
cosim_design -trace_level none -rtl verilog -ldflags "-B/usr/bin -lpthread"
exit
"""
    open(f"{prj}/run.tcl", "w").write(tcl)
    subprocess.call(["bash", "-lc", f"source /opt/xilinx/Vitis_HLS/2023.2/settings64.sh && cd {prj} && timeout 1500 vitis_hls -f run.tcl > vitis.log 2>&1"])
    log = open(f"{prj}/vitis.log", errors="replace").read()
    for l in log.splitlines():
        if "TB mismatches" in l or "ERROR" in l or "read while empty" in l or "deadlock" in l.lower() or "Pass" in l or "Fail" in l or "200-779" in l or "feedback" in l.lower():
            print("  ", l[:200])
