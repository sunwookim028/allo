"""Hold a built reg_wf128 project's RTL output against read-first and write-first models."""
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
import importlib.util
sys.path.insert(0, "tests/dataflow")
import test_compose_registers_write_first as T
prj = sys.argv[1]
n = 64
(wd, we, wa, ra), ins, yt, yq = T._arrays(n, 128)
T._catapult_rtl_class()(prj)(*ins, yt, yq)
got = {int(t): q for t, q in zip(yt, T._ints(yq)) if int(t) != 0}
rf, coll = T.model(wd, we, wa, ra)
mem = [None] * 8; reads = []
for t in range(n):
    if we[t]: mem[int(wa[t])] = wd[t]
    reads.append(mem[int(ra[t])])  # write applied before the read
wf = {t + 1: reads[t - 1] for t in range(1, n) if reads[t - 1] is not None}
seen = [t for t in rf if t in got]
print(f"{prj}: pairs {len(seen)}; read-first agrees {sum(got[t] == rf[t] for t in seen)}; "
      f"write-first agrees {sum(got[t] == wf.get(t) for t in seen)}; collisions in the trace {coll}")
