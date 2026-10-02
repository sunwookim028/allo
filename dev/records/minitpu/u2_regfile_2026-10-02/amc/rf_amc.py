"""vpu_regfile (U2, plan R1) through AMC's vendored Allo frontend.

    python rf_amc.py <trace.npz> <N> <targets> <sched>

targets: comma list of llvm, amc. sched: ``none`` | ``pipeline`` (the t
loop) | ``part`` (complete partition of ``mem`` + pipeline). The kernel is
the ``trace`` variant of ``examples/minitpu/units/vpu_regfile.py`` as a plain
function: ``uint8`` address/we arrays, ``uint16`` data, ``int32`` index locals.
"""
import os, sys, time, traceback
import numpy as np

path, N, targets, sched = sys.argv[1], int(sys.argv[2]), sys.argv[3].split(","), sys.argv[4]
src = '''
from allo.ir.types import int32, uint8, uint16
N = %d
def rf(ra: uint8[N], rb: uint8[N], rc: uint8[N], wa: uint8[N], wd: uint16[N], we: uint8[N],
       qa: uint16[N], qb: uint16[N], qc: uint16[N]):
    mem: uint16[32] = 0
    for t in range(N):
        ia: int32 = ra[t]
        ib: int32 = rb[t]
        ic: int32 = rc[t]
        qa[t] = mem[ia]
        qb[t] = mem[ib]
        qc[t] = mem[ic]
        iw: int32 = wa[t]
        dw: uint16 = wd[t]
        if we[t] != 0:
            mem[iw] = dw
''' % N
kp = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"rf_amc_kernel_{N}.py")
open(kp, "w").write(src)
import importlib.util
spec = importlib.util.spec_from_file_location(f"rf_amc_kernel_{N}", kp)
K = importlib.util.module_from_spec(spec); spec.loader.exec_module(K)
import allo

d = np.load(path)
ins = [d[k][:N].astype(np.uint8) for k in ("ra", "rb", "rc", "wa")] + [d["wd"][:N].astype(np.uint16), d["we"][:N].astype(np.uint8)]
for tgt in targets:
    t0 = time.time()
    try:
        s = allo.customize(K.rf)
        if sched in ("pipeline", "part"):
            loops = s.get_loops()
            print(loops, flush=True)
            s.pipeline(loops["S_t_0"]["t"])
        if sched == "part":
            s.partition(s.mem, partition_type="complete", dim=0)
        f = s.build(target=tgt)
        q = [np.zeros(N, np.uint16) for _ in range(3)]
        f(*ins, *q)
        np.savez(f"got_{tgt}_{sched}_{N}.npz", qa=q[0], qb=q[1], qc=q[2])
        print(f"RAN {tgt} {sched} N={N} {time.time() - t0:.1f}s", flush=True)
        if hasattr(f, "rpt"):
            print("   rpt", {k: v for k, v in dict(f.rpt).items() if not isinstance(v, (str, bytes)) or len(v) < 200}, flush=True)
        if hasattr(f, "dump_allocation"):
            f.dump_allocation(f"amc_{tgt}_{sched}_{N}.alloc.txt")
        if hasattr(f, "dump_verilog"):
            f.dump_verilog(f"amc_{tgt}_{sched}_{N}_hdl")
        for a in ("module", "mlir", "hdl", "verilog", "sv"):
            if hasattr(f, a):
                open(f"amc_{tgt}_{sched}_{N}.{a}.txt", "w").write(str(getattr(f, a)))
    except Exception as e:
        print(f"FAIL {tgt} {sched} N={N}: {type(e).__name__}: {str(e)[:600]}", flush=True)
        traceback.print_exc(limit=3)
