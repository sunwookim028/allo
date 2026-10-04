"""AMC finding (the tree's blocker): a shift chain of K loop-carried
registers, each a 1-element array read once at the top and written once at
the bottom of the iteration (the form the PE needed, pe_gen.py), aborts the
unscheduled lowering once the chain is deep enough:
  circt/lib/Dialect/LoopSchedule/Utils.cpp:442 getChainingSharedOperatorsProblem
  (scf.WhileOp ...): Assertion `succeeded(depInserted)' failed.
    python repro_reg_chain.py <K> [pipeline]"""
import os, sys, importlib.util, numpy as np
TMP = os.environ.get("TMPDIR", "/tmp"); K = int(sys.argv[1]); sched = sys.argv[2] if len(sys.argv) > 2 else "none"
L = ["from allo.ir.types import int32, uint16", "N = 16", "def k(a: uint16[N], o: uint16[N]):"]
L += [f"    q{i}_r: int32[1] = 0" for i in range(K)]
L += ["    for t in range(N):"] + [f"        q{i}: int32 = q{i}_r[0]" for i in range(K)] + ["        x: int32 = a[t]"]
L += [f"        q{i} = q{i - 1}" for i in range(K - 1, 0, -1)] + ["        q0 = x", f"        o[t] = q{K - 1}"]
L += [f"        q{i}_r[0] = q{i}" for i in range(K)]
kp = f"{TMP}/u3d_chain_{K}.py"; open(kp, "w").write("\n".join(L) + "\n")
spec = importlib.util.spec_from_file_location(f"u3d_chain_{K}", kp); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
import allo
a = (np.arange(16) * 7 + 3).astype(np.uint16); want = np.concatenate([np.zeros(K - 1, int), a[:16 - K + 1]])  # o[t] = x[t - (K - 1)]
for tgt in ("llvm", "amc"):
    s = allo.customize(M.k)
    if sched == "pipeline":
        s.pipeline(s.get_loops()["S_t_0"]["t"])
    f = s.build(target=tgt); o = np.zeros(16, np.uint16); f(a, o)
    print(f"chain K={K} {sched} {tgt}: {'OK' if (o == want).all() else 'WRONG ' + str(o.tolist())}", flush=True)
