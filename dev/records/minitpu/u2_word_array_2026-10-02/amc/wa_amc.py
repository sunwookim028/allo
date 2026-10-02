"""vpu_word_array (U2, narrow 8 x 64 b, latencies 3/2) through AMC's vendored
Allo frontend, the ``trace`` variant (pipe as data).

    python wa_amc.py <trace.npz> <N> <targets> <sched> [variant]

targets: comma list of llvm, amc. sched: ``none`` | ``pipeline`` (the t
loop) | ``part`` (complete partition of ``mem`` + pipeline). variant:
``trace`` (one access per port per cycle; default) | ``trace_rw`` (the
simulation model: every enabled port reads and writes). ``uint8`` en/we/
addr arrays, ``uint64`` data, ``int32`` index locals.
"""
import os, sys, time, traceback
import numpy as np

path, N, targets, sched = sys.argv[1], int(sys.argv[2]), sys.argv[3].split(","), sys.argv[4]
variant = sys.argv[5] if len(sys.argv) > 5 else "trace"
rd_c = "if ce[t] != 0:" if variant == "trace_rw" else "if ce[t] != 0 and cw[t] == 0:"
rd_d = "if de[t] != 0:" if variant == "trace_rw" else "if de[t] != 0 and dw[t] == 0:"
src = f'''
from allo.ir.types import int32, uint8, uint64
N = {N}
def wa(ce: uint8[N], cw: uint8[N], ca: uint8[N], cd: uint64[N],
       de: uint8[N], dw: uint8[N], da: uint8[N], dd: uint64[N],
       qc: uint64[N], qd: uint64[N]):
    mem: uint64[8] = 0
    pc: uint64[3] = 0
    pd: uint64[2] = 0
    for t in range(N):
        pc[2] = pc[1]
        pc[1] = pc[0]
        pd[1] = pd[0]
        ac: int32 = ca[t]
        ad: int32 = da[t]
        dc: uint64 = cd[t]
        dd_: uint64 = dd[t]
        {rd_c}
            pc[0] = mem[ac]
        {rd_d}
            pd[0] = mem[ad]
        if ce[t] != 0 and cw[t] != 0:
            mem[ac] = dc
        if de[t] != 0 and dw[t] != 0:
            mem[ad] = dd_
        qc[t] = pc[2]
        qd[t] = pd[1]
'''
kp = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"wa_amc_kernel_{variant}_{N}.py")
open(kp, "w").write(src)
import importlib.util
spec = importlib.util.spec_from_file_location(f"wa_amc_kernel_{variant}_{N}", kp)
K = importlib.util.module_from_spec(spec); spec.loader.exec_module(K)
import allo

d = np.load(path)
ins = [d[k][:N].astype(np.uint8) for k in ("ce", "cw", "ca")] + [d["cd"][:N].astype(np.uint64)]
ins += [d[k][:N].astype(np.uint8) for k in ("de", "dw", "da")] + [d["dd"][:N].astype(np.uint64)]
for tgt in targets:
    t0 = time.time()
    try:
        s = allo.customize(K.wa)
        if sched in ("pipeline", "part"):
            loops = s.get_loops()
            print(loops, flush=True)
            s.pipeline(loops["S_t_0"]["t"])
        if sched == "part":
            s.partition(s.mem, partition_type="complete", dim=0)
        f = s.build(target=tgt)
        q = [np.zeros(N, np.uint64) for _ in range(2)]
        f(*ins, *q)
        np.savez(f"got_{tgt}_{sched}_{variant}_{N}.npz", qc=q[0], qd=q[1])
        print(f"RAN {tgt} {sched} {variant} N={N} {time.time() - t0:.1f}s", flush=True)
        if hasattr(f, "rpt"):
            print("   rpt", {k: v for k, v in dict(f.rpt).items() if not isinstance(v, (str, bytes)) or len(v) < 200}, flush=True)
        if hasattr(f, "dump_allocation"):
            f.dump_allocation(f"amc_{tgt}_{sched}_{variant}_{N}.alloc.txt")
        if hasattr(f, "dump_verilog"):
            f.dump_verilog(f"amc_{tgt}_{sched}_{variant}_{N}_hdl")
    except Exception as e:
        print(f"FAIL {tgt} {sched} {variant} N={N}: {type(e).__name__}: {str(e)[:600]}", flush=True)
        traceback.print_exc(limit=3)
