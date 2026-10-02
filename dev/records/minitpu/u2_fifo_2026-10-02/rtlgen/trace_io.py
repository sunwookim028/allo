"""Allo-env side of the RTLGen/AMC columns for ``vpu_fifo``: dump the joined
command trace with MiniTPU's RTL response and defined masks (``dump``), or
compare an open-HLS run's responses on the defined slots (``cmp``).

    python trace_io.py dump trace_w32d4.npz [inst]
    python trace_io.py cmp trace_w32d4.npz got.npz [label]
"""
import os, sys
import numpy as np
sys.path.insert(0, os.environ.get("WT", "/work/shared/users/phd/sk3463/scratch/wt-u2ff"))
from examples.minitpu.harness import check, rtl
from examples.minitpu.units import vpu_fifo as u

KEYS = ("rst", "push", "pd", "pop")
OUTS = ("qd", "qe", "qf")
if sys.argv[1] == "dump":
    inst = sys.argv[3] if len(sys.argv) > 3 else "w32d4"
    cmd, spans = check._trace_all(u, inst)
    unit = u.INSTANCES[inst]
    packed = {p: rtl.pack(cmd[p], w) for p, w in unit.inputs}
    want = rtl.run_trace(unit, packed, seed=1)
    _, reason, _ = u.REF(inst, packed)
    out = {k: np.asarray(cmd[p], dtype=np.int64) for k, p in zip(KEYS, u.CMD)}
    for k, p in zip(OUTS, u.RESP):
        out["want_" + k] = want[p][:, 0].astype(np.int64)
        out["def_" + k] = reason[p] == ""
    np.savez(sys.argv[2], **out)
    print("DUMPED", len(out["rst"]), "cycles", [s for s in spans])
else:
    d, g = np.load(sys.argv[2]), np.load(sys.argv[3])
    lab = sys.argv[4] if len(sys.argv) > 4 else sys.argv[3]
    n = len(g["qd"])
    tot = bad = msk = 0
    ex = []
    for k in OUTS:
        dd = d["def_" + k][:n]
        diff = dd & (g[k].astype(np.int64) != d["want_" + k][:n])
        tot += int(dd.sum()); bad += int(diff.sum()); msk += int((~dd).sum())
        ex += [f"{k} cycle {i}: got {int(g[k][i]):x} rtl {int(d['want_' + k][i]):x}" for i in np.flatnonzero(diff)[:2]]
    print(f"{'MATCH' if bad == 0 else 'DIFF '} {lab}: {tot - bad}/{tot} defined over the first {n} cycles ({msk} masked)")
    for e in ex:
        print("    e.g.", e)
