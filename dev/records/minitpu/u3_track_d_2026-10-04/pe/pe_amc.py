"""mxu_pe P1 through AMC's vendored Allo frontend: generate the kernel
(pe_gen.py amc), build on ``llvm`` (frontend semantics) and ``amc``
(Verilator RTL), run the Phase 0 trace prefix in ONE call (the PE's state is
loop-carried), compare on defined slots.
    python pe_amc.py <oracles/pe.npz> <N> <targets> <sched: none|pipeline>
"""
import sys, os, time, subprocess, importlib.util, traceback
import numpy as np
path, N, targets, sched = sys.argv[1], int(sys.argv[2]), sys.argv[3].split(","), sys.argv[4]
HERE = os.path.dirname(os.path.abspath(__file__)); TMP = os.environ.get("TMPDIR", "/tmp")
kp = f"{TMP}/pe_amc_kernel_{N}.py"
subprocess.check_call([sys.executable, f"{HERE}/pe_gen.py", "amc", str(N), kp])
spec = importlib.util.spec_from_file_location(f"pe_amc_kernel_{N}", kp); K = importlib.util.module_from_spec(spec); spec.loader.exec_module(K)
import allo
d = np.load(path)
IN = [("rst_ni", np.uint8), ("weight_commit_i", np.uint8), ("weight_commit_bank_i", np.uint8), ("lhs_i", np.uint16), ("lhs_valid_i", np.uint8),
      ("weight_i", np.uint32), ("weight_valid_i", np.uint8), ("psum_i", np.uint32), ("psum_valid_i", np.uint8)]
OUT = [("weight_commit_o", np.uint8), ("lhs_o", np.uint16), ("lhs_valid_o", np.uint8), ("weight_o", np.uint32), ("psum_o", np.uint32), ("psum_valid_o", np.uint8)]
for tgt in targets:
    t0 = time.time()
    try:
        s = allo.customize(K.pe)
        if sched == "pipeline":
            loops = s.get_loops(); s.pipeline(loops["S_t_0"]["t"])
        f = s.build(target=tgt)
        print(f"BUILT {tgt} {sched} N={N} {time.time() - t0:.1f}s", flush=True)
        if tgt == "amc":
            if hasattr(f, "dump_allocation"): f.dump_allocation(f"pe_amc_{sched}_{N}.alloc.txt")
            if hasattr(f, "dump_schedule"): f.dump_schedule(f"pe_amc_{sched}_{N}.loopschedule.mlir")
            if hasattr(f, "dump_verilog"): f.dump_verilog(f"pe_amc_{sched}_{N}_hdl")
        ins = [np.ascontiguousarray(d[p][:N].astype(ty)) for p, ty in IN]
        outs = [np.zeros(N, ty) for _, ty in OUT]
        t1 = time.time(); f(*ins, *outs)
        cyc = int(f.rpt["cycles"]) if tgt == "amc" else None
        np.savez(f"pe_got_{tgt}_{sched}_{N}.npz", **{p: g for (p, _), g in zip(OUT, outs)})
        tot = bad = 0; ex = []
        for (p, _), g in zip(OUT, outs):
            df = d["def_" + p][:N]; w = d["want_" + p][:N]
            diff = df & (g.astype(np.uint64) != w)
            tot += int(df.sum()); bad += int(diff.sum())
            ex += [f"{p} cycle {i}: got {int(g[i]):x} rtl {int(w[i]):x}" for i in np.flatnonzero(diff)[:2]]
        print(f"{'MATCH' if bad == 0 else 'DIFF '} pe {tgt} {sched} N={N}: {tot - bad}/{tot} defined, cycles={cyc} ({time.time() - t1:.1f}s)", flush=True)
        for e in ex: print("    e.g.", e)
    except Exception as e:
        print(f"FAIL {tgt} {sched} N={N}: {type(e).__name__}: {str(e)[:800]}", flush=True); traceback.print_exc(limit=4)
