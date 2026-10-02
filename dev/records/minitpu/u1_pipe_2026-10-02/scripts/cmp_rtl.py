"""Catapult RTL of an Allo U1 unit vs MiniTPU's RTL unit, in Verilator.

    python cmp_rtl.py <unit> <prj> [--top top] [--ready-period P] [--n N]

Drives ``<prj>/build/Catapult/<top>.v1/concat_sim_rtl.v`` with the ``stream``
shape (``harness/rtl.py``: independent valid/ready per input, stamps), and the
MiniTPU unit with its ``valid`` shape, on the unit's stimulus. Prints values
(classified with the unit's EXPLAIN), latency as edges from input accept to
output (the same count as MiniTPU's ``valid_o``), and cycles per vector.
"""
import argparse, os, re, sys, time
import importlib
import numpy as np

sys.path.insert(0, os.getcwd())
from examples.minitpu.harness import check, rtl  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("unit"); ap.add_argument("prj")
ap.add_argument("--top", default="top")
ap.add_argument("--ready-period", type=int, default=0)
ap.add_argument("--n", type=int, default=0)
a = ap.parse_args()

u = importlib.import_module(f"examples.minitpu.units.{a.unit}")
stim = u.stimulus()
if a.n:
    stim = stim[: a.n]
n = len(stim)
v1 = os.path.join(a.prj, "build", "Catapult", f"{a.top}.v1")
src = os.path.join(os.path.abspath(v1), "concat_sim_rtl.v")
# the top's data ports, from the emitted SystemC (In before Out, declaration order)
k = open(os.path.join(a.prj, "kernel.cpp")).read()
blk = k[k.index(f"SC_MODULE({a.top})"):]
blk = blk[: blk.index("};")]
def _w(t):
    m = re.match(r"ac_int<(\d+),", t)
    return int(m.group(1)) if m else {"ac::bfloat16": 16, "ac_ieee_float<binary32>": 32}[t]
ins = [(_w(t), p) for t, p in re.findall(r"Connections::In< (.+?) > (\w+);", blk)]
outs = [(_w(t), p) for t, p in re.findall(r"Connections::Out< (.+?) > (\w+);", blk)]
cu = rtl.RtlUnit(top=a.top, sources=[src], inputs=[(p, int(w)) for w, p in ins],
                 outputs=[(p, int(w)) for w, p in outs], shape="stream", clk="clk",
                 rst_n="rst", out_ready_period=a.ready_period)
t = time.time()
want, wl = rtl.run(u.RTL, stim.astype(np.uint64))
want = want[:, 0]
got, cyc = rtl.run(cu, stim.astype(np.uint64))
got = got[:, 0]
st = dict(rtl.last_stats)
kd = int((got != want).sum())
lat = dict(zip(*[x.tolist() for x in np.unique(cyc, return_counts=True)]))
print(f"RTL-CMP {a.unit} {a.prj}: values {n - kd}/{n} vs MiniTPU RTL; "
      f"latency catapult {lat} vs minitpu {sorted(set(wl.tolist()))} (declared {u.RTL.latency}); "
      f"{st.get('cycles', 0) / n:.4f} cyc/vector, first output cycle {st.get('first_out')}; "
      f"ready period {a.ready_period}; {time.time() - t:.1f}s")
if kd:
    for name, idx in check.classify(stim, got, want, u.EXPLAIN).items():
        ex = ", ".join(f"{stim[i, 0]:x}+{stim[i, 1]:x}: catapult {got[i]:x} minitpu {want[i]:x}" for i in idx[:2])
        print(f"    {len(idx):7d}  {name}  e.g. {ex}")
