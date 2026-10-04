"""mxu_pe P1 through RTLGen: generate the kernel (pe_gen.py rtlgen), export,
cosim the Phase 0 trace prefix, compare on defined slots.
    python pe_rtlgen.py <oracles/pe.npz> <N> [freq_mhz]
"""
import sys, os, time, subprocess, importlib.util
import numpy as np
path, N = sys.argv[1], int(sys.argv[2])
freq = float(sys.argv[3]) if len(sys.argv) > 3 else 300.0
HERE = os.path.dirname(os.path.abspath(__file__))
kp = os.path.join(os.getcwd(), f"pe_rtlgen_kernel_{N}.py")
subprocess.check_call([sys.executable, f"{HERE}/pe_gen.py", "rtlgen", str(N), kp])
spec = importlib.util.spec_from_file_location(f"pe_rtlgen_kernel_{N}", kp); K = importlib.util.module_from_spec(spec); spec.loader.exec_module(K)
d = np.load(path)
IN = [("rst_ni", np.uint8), ("weight_commit_i", np.uint8), ("weight_commit_bank_i", np.uint8), ("lhs_i", np.uint16), ("lhs_valid_i", np.uint8),
      ("weight_i", np.uint32), ("weight_valid_i", np.uint8), ("psum_i", np.uint32), ("psum_valid_i", np.uint8)]
OUT = [("weight_commit_o", np.uint8), ("lhs_o", np.uint16), ("lhs_valid_o", np.uint8), ("weight_o", np.uint32), ("psum_o", np.uint32), ("psum_valid_o", np.uint8)]
t0 = time.time()
rtl = K.pe.schedule().export("rtl", freq_mhz=freq)
print(f"EXPORTED pe N={N} freq={freq} in {time.time() - t0:.1f}s", flush=True)
res = rtl.schedule(); f = res.func("pe")
print("SCHED latency:", f.latency, "| II:", [r.interval for r in res.cyclic()])
open(f"pe_N{N}_f{int(freq)}.sv", "w").write(rtl.verilog)
q = rtl.estimation
print("QOR: lat", q.latency, "fmax %.1f" % q.fmax, "area", q.area)
print("CRIT:", str(q.critical_paths[0])[:300] if q.critical_paths else None)
ins = [d[p][:N].astype(ty) for p, ty in IN]
outs = [np.zeros(N, ty) for _, ty in OUT]
t0 = time.time()
cyc = rtl.cosim(*ins, *outs, timeout=int(os.environ.get("COSIM_TIMEOUT", 8 * N + 5000))).cycles
print(f"COSIM pe N={N} cycles={cyc} ({time.time() - t0:.1f}s)", flush=True)
np.savez(f"pe_got_N{N}.npz", **{p: g for (p, _), g in zip(OUT, outs)})
tot = bad = 0; ex = []
for (p, _), g in zip(OUT, outs):
    df = d["def_" + p][:N]; w = d["want_" + p][:N]
    diff = df & (g.astype(np.uint64) != w)
    tot += int(df.sum()); bad += int(diff.sum())
    ex += [f"{p} cycle {i}: got {int(g[i]):x} rtl {int(w[i]):x}" for i in np.flatnonzero(diff)[:2]]
print(f"{'MATCH' if bad == 0 else 'DIFF '} pe rtlgen N={N}: {tot - bad}/{tot} defined", flush=True)
for e in ex:
    print("    e.g.", e)
