"""vpu_fifo (U2, plan F2) in RTLGen's frontend (kkkaishao/allo allo-rtlgen).

    python fifo_rtlgen.py <trace.npz> <variant> <N> [width] [depth]

``trace``: the explicit ring of ``units/vpu_fifo.py``'s ``trace`` variant
transcribed (``u32`` data; ``u8`` flags and rst; ``i32`` pointers and count);
``trace_part``: the same with ``mem`` completely partitioned. One cosim of N
cycles (the first N of the joined command trace).
"""
import sys, time
import numpy as np
from allo import kernel
from allo.lang import i32, u8, u32, u64

path, variant, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
W = int(sys.argv[4]) if len(sys.argv) > 4 else 32
D = int(sys.argv[5]) if len(sys.argv) > 5 else 4
T = u32 if W <= 32 else u64
d = dict(np.load(path))
npT = np.uint32 if W <= 32 else np.uint64


@kernel
def fifo(RST: u8[N], PUSH: u8[N], PD: T[N], POP: u8[N], QD: T[N], QE: u8[N], QF: u8[N]):
    mem: T[D]
    rd: i32 = 0
    wr: i32 = 0
    cnt: i32 = 0
    for t in range(N, name="t"):
        empty: i32 = 0
        full: i32 = 0
        if cnt == 0:
            empty = 1
        if cnt == D:
            full = 1
        QD[t] = mem[rd]
        QE[t] = empty
        QF[t] = full
        r: u8 = RST[t]
        p: u8 = PUSH[t]
        x: T = PD[t]
        q: u8 = POP[t]
        if r == 0:
            rd = 0
            wr = 0
            cnt = 0
        else:
            do_push: i32 = 0
            do_pop: i32 = 0
            if p == 1:
                if full == 0:
                    do_push = 1
                elif q == 1:
                    do_push = 1
            if q == 1:
                if empty == 0:
                    do_pop = 1
                elif p == 1:
                    do_pop = 1
            if do_push == 1:
                mem[wr] = x
                if wr == D - 1:
                    wr = 0
                else:
                    wr = wr + 1
            if do_pop == 1:
                if rd == D - 1:
                    rd = 0
                else:
                    rd = rd + 1
            cnt = cnt + do_push - do_pop


t0 = time.time()
sch = fifo.schedule()
if variant.endswith("_part"):
    sch.partition("mem", dim=1, kind=sch.Complete)
m = sch.export("rtl")
print(f"EXPORTED {variant} N={N} W={W} D={D} in {time.time() - t0:.1f}s", flush=True)
open(f"fifo_{variant}_{N}.sv", "w").write(m.sv if hasattr(m, "sv") else str(getattr(m, "verilog", "")))
ins = [d["rst"][:N].astype(np.uint8), d["push"][:N].astype(np.uint8), d["pd"][:N].astype(npT), d["pop"][:N].astype(np.uint8)]
q = [np.zeros(N, npT), np.zeros(N, np.uint8), np.zeros(N, np.uint8)]
cyc = m.cosim(*ins, *q, timeout=6 * N + 1000)
np.savez(f"got_{variant}_{N}.npz", qd=q[0], qe=q[1], qf=q[2])
print(f"COSIM {variant} N={N} cycles={cyc} {time.time() - t0:.1f}s", flush=True)
