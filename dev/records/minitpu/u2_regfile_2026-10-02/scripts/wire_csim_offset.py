import os, sys, numpy as np
sys.path.insert(0, os.environ.get("WT", "/work/shared/users/phd/sk3463/scratch/wt-u2rf"))
import allo.dataflow as df
from examples.minitpu.units import vpu_regfile as u
from examples.minitpu.harness import check, rtl
n = 3000
cmd, spans = check._trace_all(u, "w16", n)
unit = u.INSTANCES["w16"]
packed = {p: rtl.pack(cmd[p], w) for p, w in unit.inputs}
want = rtl.run_trace(unit, packed, seed=1)
_, reason, _ = u.REF("w16", packed)
mod = df.build(u.wire(n, 16), target="systemc", mode="csim", project="/work/shared/users/phd/sk3463/scratch/u2rf/prj/wire_off")
got = u._run_flat(mod, cmd, n, 16)
for p in u.RESP:
    d = reason[p] == ""
    w = want[p][:, 0].astype(np.int64); g = got[p].astype(np.int64)
    res = {}
    for k in range(-4, 5):  # got[t + k] vs want[t]
        lo, hi = max(0, -k), min(n, n - k)
        dd = d[lo:hi]
        res[k] = int((dd & (g[lo + k:hi + k] == w[lo:hi])).sum())
    print("OFFSET", p, "defined", int(d.sum()), res, "first got", g[:6].tolist(), "want", w[:6].tolist(), flush=True)
