"""Allo-env side of the RTLGen/AMC columns for ``vpu_word_array``: dump the
joined command trace with MiniTPU's RTL response and defined masks
(``dump``), or compare an open-HLS run's responses against them on the
defined slots (``cmp <got.npz>``). Keys: ``ce cw ca cd de dw da dd`` (the
``CMD`` order), ``want_qc``/``want_qd``, ``def_qc``/``def_qd``. Data is
``uint64`` (narrow: 64 b words).

    python trace_io.py dump trace_narrow.npz [inst]
    python trace_io.py cmp trace_narrow.npz got.npz [label]
"""
import os, sys
import numpy as np
sys.path.insert(0, os.environ.get("WT", "/work/shared/users/phd/sk3463/scratch/wt-u2wa"))
from examples.minitpu.harness import check, rtl
from examples.minitpu.units import vpu_word_array as u

K = ("ce", "cw", "ca", "cd", "de", "dw", "da", "dd")
if sys.argv[1] == "dump":
    inst = sys.argv[3] if len(sys.argv) > 3 else "narrow"
    cmd, spans = check._trace_all(u, inst)
    unit = u.INSTANCES[inst]
    packed = {p: rtl.pack(cmd[p], w) for p, w in unit.inputs}
    want = rtl.run_trace(unit, packed, seed=1)
    _, reason, _ = u.REF(inst, packed)
    out = {k: np.asarray([int(x) for x in cmd[p]], dtype=np.uint64) for k, p in zip(K, u.CMD)}
    for k, p in zip(("qc", "qd"), u.RESP):
        out["want_" + k] = want[p][:, 0].astype(np.uint64)
        out["def_" + k] = reason[p] == ""
    np.savez(sys.argv[2], **out)
    print("DUMPED", len(out["ce"]), "cycles", [s for s in spans])
else:
    d, g = np.load(sys.argv[2]), np.load(sys.argv[3])
    lab = sys.argv[4] if len(sys.argv) > 4 else sys.argv[3]
    n = len(g["qc"])
    tot = bad = msk = 0
    ex = []
    for k in ("qc", "qd"):
        dd = d["def_" + k][:n]
        diff = dd & (g[k].astype(np.uint64) != d["want_" + k][:n])
        tot += int(dd.sum()); bad += int(diff.sum()); msk += int((~dd).sum())
        ex += [f"{k} cycle {i}: got {int(g[k][i]):x} rtl {int(d['want_' + k][i]):x}" for i in np.flatnonzero(diff)[:2]]
    print(f"{'MATCH' if bad == 0 else 'DIFF '} {lab}: {tot - bad}/{tot} defined over the first {n} cycles ({msk} masked)")
    for e in ex:
        print("    e.g.", e)
