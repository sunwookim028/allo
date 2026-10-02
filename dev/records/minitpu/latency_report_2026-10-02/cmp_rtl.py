# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Catapult RTL of an Allo U1 unit vs MiniTPU's RTL unit, in Verilator.

Copied from ``../u1_catapult_units_2026-10-02/scripts/cmp_rtl.py``; the only
change is the last block: when the project has Allo's ``latency.json``, the
measured latency is checked against it (``harness/latency.py``).

    python cmp_rtl.py <unit> <prj> [--top top] [--shape stream|bare]
                      [--ready-period P] [--n N]

From the worktree root. Generalises the pipe record's ``cmp_rtl.py`` to every
U1 unit (any number of input ports) and adds the pilot's ``bare`` shape for
Wire-port RTL.

``stream``: ``<prj>/build/Catapult/<top>.v1/concat_sim_rtl.v`` driven with
independent valid/ready per input (``harness/rtl.py``), stamps per vector:
prints the latency histogram (edges from the last input's accept to the
output, MiniTPU's count) and cycles per vector.
``bare``: plain ports (a ``Wire`` emission, ``--top`` the kernel), active-low
reset pulsed, one new input vector per edge, every edge's output recorded;
the latency is *measured* as the output offset (1..7 edges) that matches
MiniTPU on the first 3,000 vectors, then the full stimulus is compared at it.

The MiniTPU unit runs in its own shape (``comb``/``bare``/``valid``) on the
same stimulus; values are compared bit for bit and differences classified
with the unit's ``EXPLAIN``.
"""
import argparse, importlib, os, re, sys, time
import numpy as np

sys.path.insert(0, os.getcwd())
from examples.minitpu.harness import check, latency, rtl  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("unit"); ap.add_argument("prj")
ap.add_argument("--top", default="top")
ap.add_argument("--shape", default="stream")
ap.add_argument("--latency", type=int, default=0)
ap.add_argument("--ready-period", type=int, default=0)
ap.add_argument("--n", type=int, default=0)
a = ap.parse_args()

u = importlib.import_module(f"examples.minitpu.units.{a.unit}")
stim = u.stimulus()
if a.n:
    stim = stim[: a.n]
n = len(stim)
name = os.path.basename(a.prj.rstrip("/")).replace(".prj", "")
v1 = os.path.join(a.prj, "build", "Catapult", f"{a.top}.v1")
src = os.path.join(os.path.abspath(v1), "concat_sim_rtl.v")
k = open(os.path.join(a.prj, "kernel.cpp")).read()
blk = k[k.index(f"SC_MODULE({a.top})"):]
blk = blk[: blk.index("\n};")]


def _w(t):
    m = re.match(r"ac_int<(\d+),", t)
    if m:
        return int(m.group(1))
    return {"ac::bfloat16": 16, "ac_ieee_float<binary32>": 32, "bool": 1}[t]


if a.shape == "stream":
    ins = [(p, _w(t)) for t, p in re.findall(r"Connections::In< (.+?) > (\w+);", blk)]
    outs = [(p, _w(t)) for t, p in re.findall(r"Connections::Out< (.+?) > (\w+);", blk)]
else:
    ins = [(p, _w(t)) for t, p in re.findall(r"\bsc_in< (.+?) > (\w+);", blk) if p != "rst"]
    outs = [(p, _w(t)) for t, p in re.findall(r"\bsc_out< (.+?) > (\w+);", blk) if p != "done"]
assert len(ins) == stim.shape[1], (ins, stim.shape)

t = time.time()
want, wl = rtl.run(u.RTL, stim.astype(np.uint64))
want = want[:, 0]
t_m = time.time() - t


def cat(shape, n_, **kw):
    cu = rtl.RtlUnit(top=a.top, sources=[src], inputs=ins, outputs=outs, shape=shape,
                     clk="clk", rst_n="rst", **kw)
    return rtl.run(cu, stim[:n_].astype(np.uint64))


t = time.time()
if a.shape == "bare":
    # One build, one run: PAD copies of vector 0 lead (the thread's reset
    # action may drop the first inputs), then the stimulus, then PAD more.
    # raw[c] is the output after edge c+1; input k is captured by edge
    # PAD+k+1, so a result at raw[PAD+k+L-1] has latency L (rtl.py's count).
    PAD = 16
    st2 = np.concatenate([np.repeat(stim[:1], PAD, 0), stim, np.repeat(stim[-1:], PAD, 0)])
    cu = rtl.RtlUnit(top=a.top, sources=[src], inputs=ins, outputs=outs, shape="bare",
                     clk="clk", rst_n="rst", reset=True, warmup=0, latency=1)
    raw, _ = rtl._execute(cu, st2.astype(np.uint64))
    raw = raw[:, 0]
    m = min(3000, n)
    hits = {L: int((raw[PAD + L - 1: PAD + L - 1 + m] == want[:m]).sum()) for L in range(1, 8)}
    L = max(hits, key=hits.get)
    print(f"    bare: matches on the first {m} vectors by latency {hits}")
    got = raw[PAD + L - 1: PAD + L - 1 + n]
    a.latency = L
    cyc = None
else:
    got, cyc = cat("stream", n, out_ready_period=a.ready_period)
    got = got[:, 0]
st = dict(rtl.last_stats or {})
kd = int((got != want).sum())
if a.shape == "stream":
    lat = dict(zip(*[x.tolist() for x in np.unique(cyc, return_counts=True)]))
    timing = (f"latency catapult {lat} vs minitpu {sorted(set(np.asarray(wl).tolist()))[:4]} "
              f"(declared {u.RTL.latency}); {st.get('cycles', 0) / n:.4f} cyc/vector, "
              f"first output cycle {st.get('first_out')}; ready period {a.ready_period}")
else:
    timing = f"bare: measured latency {a.latency} (declared {u.RTL.latency})"
print(f"RTL-CMP {a.unit} {name}: values {n - kd}/{n} vs MiniTPU RTL; {timing}; "
      f"{time.time() - t:.1f}s (MiniTPU {t_m:.1f}s)")
try:
    ieee = u.IEEE(*[stim[:, j] for j in range(stim.shape[1])])
    print(f"    vs IEEE reference ({u.IEEE.__name__}): {int((got == ieee).sum())}/{n} equal")
except Exception as e:  # noqa: BLE001
    print(f"    (IEEE reference not compared: {type(e).__name__}: {e})")
if kd:
    for nm, idx in check.classify(stim, got, want, u.EXPLAIN).items():
        ex = ", ".join(
            "/".join(f"{int(x):x}" for x in stim[i]) + f": catapult {int(got[i]):x} minitpu {int(want[i]):x}"
            for i in idx[:2])
        print(f"    {len(idx):8d}  {nm}  e.g. {ex}")

if os.path.exists(os.path.join(a.prj, "latency.json")):
    man = latency.load(a.prj)
    kerns = [a.top] if a.top in man["units"] else sorted(k for k in man["units"] if k not in ("src_0", "sink_0"))
    kern = kerns[0]
    if a.shape == "stream" and len(kerns) > 1:
        if a.ready_period == 0:
            print("    " + latency.composite(man, kerns, lat, st.get("cycles", 0) / n))
    elif a.shape == "stream":
        if a.ready_period == 0:
            print("    " + latency.verdict(man, kern, lat, st.get("cycles", 0) / n, u.RTL.latency))
    else:
        print("    " + latency.verdict(man, kern, a.latency, None, u.RTL.latency))
