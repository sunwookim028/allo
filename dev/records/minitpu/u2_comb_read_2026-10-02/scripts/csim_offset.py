# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""SystemC csim of an emitted (or hand-patched) ``wire`` project vs the RTL
reference, per port, at output offsets -4..4 (the U2 pilot's
``wire_csim_offset.py``, on a project whose ``sim`` is already built).

    python csim_offset.py <prj> [<prj> ...]     # prj holds kernel.cpp and sim

Writes ``input0..5.data`` (the first 64 cycles of the joined w16 trace, the
emitted TB's trip count), runs ``./sim``, reads ``output0..2.data``.
"""
import os, subprocess, sys
import numpy as np

sys.path.insert(0, os.getcwd())
from examples.minitpu.harness import check, rtl  # noqa: E402
from examples.minitpu.units import vpu_regfile as u  # noqa: E402

N = 64
cmd, _ = check._trace_all(u, "w16", N)
unit = u.INSTANCES["w16"]
packed = {p: rtl.pack(cmd[p], w) for p, w in unit.inputs}
want = rtl.run_trace(unit, packed, seed=1)
_, reason, _ = u.REF("w16", packed)
for prj in sys.argv[1:]:
    for i, p in enumerate(u.CMD):
        with open(os.path.join(prj, f"input{i}.data"), "w") as f:
            f.write("".join(f"{int(x)}\n" for x in cmd[p][:N]))
    r = subprocess.run("./sim", cwd=prj, capture_output=True, text=True, timeout=300)
    print(f"== {os.path.basename(prj.rstrip('/'))}: sim exit {r.returncode}")
    for i, p in enumerate(u.RESP):
        g = np.array([int(x) for x in open(os.path.join(prj, f"output{i}.data")).read().split()], dtype=np.int64)
        d = reason[p][:N] == ""
        w = want[p][:N, 0].astype(np.int64)
        res = {}
        for k in range(-4, 5):  # got[t + k] vs want[t]
            lo, hi = max(0, -k), min(N, N - k)
            dd = d[lo:hi]
            res[k] = int((dd & (g[lo + k:hi + k] == w[lo:hi])).sum())
        best = max(res, key=res.get)
        print(f"   OFFSET {p}: defined {int(d.sum())}, best offset {best:+d} ({res[best]} agree); by offset {res}")
