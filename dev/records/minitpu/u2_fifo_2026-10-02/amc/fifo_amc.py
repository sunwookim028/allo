"""vpu_fifo (U2, plan F2) through AMC's vendored Allo frontend.

    python fifo_amc.py <trace.npz> <N> <targets> <sched> [width] [depth]

targets: comma list of llvm, amc. sched: ``none`` | ``pipeline`` (the t loop)
| ``part`` (complete partition of ``mem`` + pipeline). The kernel is the
``trace`` variant of ``examples/minitpu/units/vpu_fifo.py`` as a plain
function: ``uint8`` flags/rst, ``uint32`` data, ``int32`` pointers.
"""
import os, sys, time, traceback
import numpy as np

path, N, targets, sched = sys.argv[1], int(sys.argv[2]), sys.argv[3].split(","), sys.argv[4]
W = int(sys.argv[5]) if len(sys.argv) > 5 else 32
D = int(sys.argv[6]) if len(sys.argv) > 6 else 4
T = "uint32" if W <= 32 else "uint64"
src = f'''
from allo.ir.types import int32, uint8, uint32, uint64
N = {N}
D = {D}
def fifo(rst: uint8[N], push: uint8[N], pd: {T}[N], pop: uint8[N], qd: {T}[N], qe: uint8[N], qf: uint8[N]):
    mem: {T}[D] = 0
    rd: int32 = 0
    wr: int32 = 0
    cnt: int32 = 0
    for t in range(N):
        empty: int32 = 0
        full: int32 = 0
        if cnt == 0:
            empty = 1
        if cnt == D:
            full = 1
        qd[t] = mem[rd]
        qe[t] = empty
        qf[t] = full
        r: uint8 = rst[t]
        p: uint8 = push[t]
        x: {T} = pd[t]
        q: uint8 = pop[t]
        if r == 0:
            rd = 0
            wr = 0
            cnt = 0
        else:
            do_push: int32 = 0
            do_pop: int32 = 0
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
'''
kp = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"fifo_amc_kernel_{N}_{W}_{D}.py")
open(kp, "w").write(src)
import importlib.util
spec = importlib.util.spec_from_file_location(f"fifo_amc_kernel_{N}", kp)
K = importlib.util.module_from_spec(spec); spec.loader.exec_module(K)
import allo

d = np.load(path)
npT = np.uint32 if W <= 32 else np.uint64
ins = [d["rst"][:N].astype(np.uint8), d["push"][:N].astype(np.uint8), d["pd"][:N].astype(npT), d["pop"][:N].astype(np.uint8)]
for tgt in targets:
    t0 = time.time()
    try:
        s = allo.customize(K.fifo)
        if sched in ("pipeline", "part"):
            loops = s.get_loops()
            print(loops, flush=True)
            s.pipeline(loops["S_t_0"]["t"])
        if sched == "part":
            s.partition(s.mem, partition_type="complete", dim=0)
        f = s.build(target=tgt)
        q = [np.zeros(N, npT), np.zeros(N, np.uint8), np.zeros(N, np.uint8)]
        f(*ins, *q)
        np.savez(f"got_{tgt}_{sched}_{N}.npz", qd=q[0], qe=q[1], qf=q[2])
        print(f"RAN {tgt} {sched} N={N} {time.time() - t0:.1f}s", flush=True)
        if hasattr(f, "rpt"):
            print("   rpt", {k: v for k, v in dict(f.rpt).items() if not isinstance(v, (str, bytes)) or len(v) < 200}, flush=True)
        if hasattr(f, "dump_allocation"):
            f.dump_allocation(f"amc_{tgt}_{sched}_{N}.alloc.txt")
        if hasattr(f, "dump_verilog"):
            f.dump_verilog(f"amc_{tgt}_{sched}_{N}_hdl")
    except Exception as e:
        print(f"FAIL {tgt} {sched} N={N}: {type(e).__name__}: {str(e)[:600]}", flush=True)
        traceback.print_exc(limit=3)
