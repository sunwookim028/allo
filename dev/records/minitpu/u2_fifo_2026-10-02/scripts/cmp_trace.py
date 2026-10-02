"""Catapult RTL of an Allo ``vpu_fifo`` Wire variant vs MiniTPU's RTL, per cycle.

    python cmp_trace.py <prj> [--top fifo_0] [--inst w32d4] [--n N] [--pad 8]

From the worktree root, after ``source examples/minitpu/harness/env-zhang21.sh``.
``u2_regfile_2026-10-02/scripts/cmp_trace.py`` adapted to the FIFO's ports:
MiniTPU ``rst_ni``/``push_i``/``push_data_i``/``pop_i`` -> ``pop_data_o``/
``empty_o``/``full_o``. The Allo kernel carries ``rst_ni`` as a data Wire (the
pointer reset is in its body); Catapult's own ``rst`` port is held low for
``PAD`` rows first and high afterwards.

Output latency is **measured**: the offset L (edges; 0 = the "pre" sample of
the same cycle, L >= 1 = the "post" sample of row t + L - 1) at which the
defined slots of the first 3,000 cycles agree best; every defined slot is then
compared at that L. The unit's four step probes (``units/vpu_fifo.py``
``probes``) are run on the Catapult RTL with the same port mapping, beside
MiniTPU's (all 1).
"""
import argparse, os, re, sys, time

import numpy as np

sys.path.insert(0, os.getcwd())
from examples.minitpu.harness import check, rtl  # noqa: E402
from examples.minitpu.units import vpu_fifo as u  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("prj")
ap.add_argument("--top", default="fifo_0")
ap.add_argument("--inst", default="w32d4")
ap.add_argument("--n", type=int, default=0)
ap.add_argument("--pad", type=int, default=8)
a = ap.parse_args()

bdir = os.path.join(os.path.abspath(a.prj), "build")
cat = [d for d in sorted(os.listdir(bdir)) if d.startswith("Catapult")]
v1 = os.path.join(bdir, cat[-1], f"{a.top}.v1")
src = os.path.join(v1, "concat_sim_rtl.v")
k = open(os.path.join(a.prj, "kernel.cpp")).read()
blk = k[k.index(f"SC_MODULE({a.top})"):]
blk = blk[: blk.index("\n};")]


def _w(t):
    m = re.match(r"(?:ac_int|ap_uint)<(\d+)", t)
    return int(m.group(1)) if m else {"bool": 1}[t]


ins = [(p, _w(t)) for t, p in re.findall(r"\bsc_in< (.+?) > (\w+);", blk) if p != "rst"]
outs = [(p, _w(t)) for t, p in re.findall(r"\bsc_out< (.+?) > (\w+);", blk) if p != "done"]
assert len(ins) == len(u.CMD) and len(outs) == len(u.RESP), (ins, outs)
imap = dict(zip(u.CMD, [p for p, _ in ins]))
omap = dict(zip(u.RESP, [p for p, _ in outs]))

cmd, spans = check._trace_all(u, a.inst, a.n)
n = len(next(iter(cmd.values())))
unit = u.INSTANCES[a.inst]
packed = {p: rtl.pack(cmd[p], w) for p, w in unit.inputs}
t0 = time.time()
want = rtl.run_trace(unit, packed, seed=1)
_, reason, _ = u.REF(a.inst, packed)


def cat_unit(sample):
    return rtl.RtlUnit(top=a.top, sources=[src], inputs=[("rst", 1)] + ins,
                       outputs=[(p, w, sample) for p, w in outs], shape="trace",
                       clk="clk", rst_n="rst", reset=False)


def cat_cmd(c, nn, pad):
    """MiniTPU-named command trace -> Catapult ports, PAD reset rows first, 16 idle after."""
    out = {"rst": [0] * pad + [1] * (nn + 16)}
    for mp, cp in imap.items():
        out[cp] = [0] * pad + [int(x) for x in c[mp][:nn]] + [0] * 16
    return out


res = {}
for sample in ("pre", "post"):
    res[sample] = rtl.run_trace(cat_unit(sample), cat_cmd(cmd, n, a.pad), seed=1)
P = a.pad


def got_at(L, port):
    col = res["pre" if L == 0 else "post"][omap[port]]
    off = P + (L - 1 if L else 0)
    return col[off: off + n]


m = min(3000, n)
hits = {}
for L in range(0, 8):
    h = 0
    for p in u.RESP:
        d = reason[p][:m] == ""
        h += int((d & (got_at(L, p)[:m] == want[p][:m]).all(axis=1)).sum())
    hits[L] = h
L = max(hits, key=hits.get)
tot = bad = masked = 0
first = []
per_port = {}
for p in u.RESP:
    d = reason[p] == ""
    g = got_at(L, p)
    diff = d & (g != want[p]).any(axis=1)
    tot += int(d.sum())
    masked += int((~d).sum())
    bad += int(diff.sum())
    per_port[p] = int(diff.sum())
    for i in np.flatnonzero(diff)[:3]:
        lab = next(lb for lb, s0, s1 in spans if s0 <= i < s1)
        first.append(f"{lab} cycle {i} {p}: catapult {rtl.unpack(g[i:i+1])[0]:x} minitpu {rtl.unpack(want[p][i:i+1])[0]:x}")


def probes():
    """The unit's step probes on both RTLs (MiniTPU-named traces)."""
    out = []
    for lab, declared, _ in u.probes(a.inst):
        pass
    from examples.minitpu.harness.traces import Trace
    w, d = u.GEOM[a.inst]
    cases = []
    t = u._reset(Trace(u._defaults())).idle(4); ev = len(t); t.cycle(push_i=1, push_data_i=1).idle(4)
    cases.append(("push -> empty_o", "empty_o", t.cmd(), ev))
    t = u._reset(Trace(u._defaults())); t.cycle(push_i=1, push_data_i=1).cycle(pop_i=1).idle(4); ev = len(t)
    t.cycle(push_i=1, push_data_i=(1 << w) - 2).idle(4)
    cases.append(("push -> pop_data_o (fall-through)", "pop_data_o", t.cmd(), ev))
    t = u._reset(Trace(u._defaults()))
    for _ in range(d - 1):
        t.cycle(push_i=1, push_data_i=0)
    t.idle(4); ev = len(t); t.cycle(push_i=1, push_data_i=0).idle(4)
    cases.append(("last push -> full_o", "full_o", t.cmd(), ev))
    t = u._reset(Trace(u._defaults())); t.cycle(push_i=1, push_data_i=1).cycle(push_i=1, push_data_i=2).idle(4); ev = len(t)
    t.cycle(pop_i=1).idle(4)
    cases.append(("pop -> pop_data_o", "pop_data_o", t.cmd(), ev))
    for lab, port, c, ev in cases:
        nn = len(c["rst_ni"])
        m0 = rtl.probe_trace(cat_unit("pre"), cat_cmd(c, nn, P), omap[port], ev + P)
        m1 = rtl.probe_trace(cat_unit("post"), cat_cmd(c, nn, P), omap[port], ev + P)
        mt = rtl.probe_trace(unit, {kk: rtl.pack(c[kk], ww) for kk, ww in unit.inputs}, port, ev)
        out.append((lab, mt, m0, m1))
    return out


name = os.path.basename(a.prj.rstrip("/")).replace(".prj", "")
print(f"RTL-TRACE vpu_fifo:{a.inst} {name}: defined {tot - bad}/{tot} at measured output latency L={L} "
      f"(MiniTPU 0); {masked} masked; by port {per_port}; first-3000 hits by L {hits}; {n} cycles; {time.time() - t0:.1f}s")
for f in first:
    print(f"    e.g. {f}")
for lab, mt, m0, m1 in probes():
    print(f"    PROBE {lab}: MiniTPU {mt}; Catapult pre-sampled {m0}, post-sampled {m1}")
