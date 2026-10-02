"""``stream_raw`` has no peek, so ``pop_data`` is defined for it only on pop
cycles. Compare its flags on every defined slot and ``pop_data`` on the pop
cycles only, on the simulator and SystemC csim.

    python flags_only.py [inst] [n]
"""
import os, sys
import numpy as np
sys.path.insert(0, os.getcwd())
import allo.dataflow as df
from examples.minitpu.harness import check, rtl
from examples.minitpu.units import vpu_fifo as u

inst = sys.argv[1] if len(sys.argv) > 1 else "w32d4"
n_max = int(sys.argv[2]) if len(sys.argv) > 2 else 0
w = u.WIDTH[inst]
cmd, spans = check._trace_all(u, inst, n_max)
n = len(cmd["rst_ni"])
unit = u.INSTANCES[inst]
packed = {p: rtl.pack(cmd[p], ww) for p, ww in unit.inputs}
want = rtl.run_trace(unit, packed, seed=1)
_, reason, _ = u.REF(inst, packed)
pop = np.asarray(cmd["pop_i"]) == 1
for backend in ("simulator", "systemc"):
    prj = f"/work/shared/users/phd/sk3463/scratch/u2ff/prj/flags_{inst}_{backend}"
    top = u.stream_raw(n, w)
    mod = df.build(top, target="simulator") if backend == "simulator" else df.build(top, target="systemc", mode="csim", project=prj)
    got = u._run_flat(mod, cmd, n, w)
    out = []
    for p in u.RESP:
        d = reason[p] == ""
        if p == "pop_data_o":
            d = d & pop
        g = np.asarray([int(x) for x in got[p]], dtype=object)
        wv = np.asarray(rtl.unpack(want[p]), dtype=object)
        bad = int((d & (g != wv)).sum())
        out.append(f"{p} {int(d.sum()) - bad}/{int(d.sum())}")
        if bad:
            i = np.flatnonzero(d & (g != wv))[:3]
            lab = [next(lb for lb, a, b in spans if a <= j < b) for j in i]
            out.append(f"(e.g. {list(zip(lab, i.tolist(), [hex(g[j]) for j in i], [hex(wv[j]) for j in i]))})")
    print(f"FLAGS-ONLY vpu_fifo:{inst} stream_raw {backend}: " + "; ".join(out), flush=True)
