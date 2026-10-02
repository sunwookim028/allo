"""Catapult RTL of an Allo ``vpu_regfile`` Wire variant vs MiniTPU's RTL, per cycle.

    python cmp_trace.py <prj> [--top rf_0] [--inst w16] [--n N] [--pad 8]

From the worktree root, after ``source examples/minitpu/harness/env-zhang21.sh``.

Both RTLs run in the harness ``trace`` shape (``harness/rtl.py``): one command
row per cycle, ``clk`` low + ``eval`` + sample "pre", rising edge, sample
"post". The command trace is ``check.py``'s (every trace and tb seed of the
instance, joined). Catapult's module gets ``PAD`` reset rows (``rst = 0``)
first; its ports are read from ``kernel.cpp`` in declaration order and mapped
onto MiniTPU's (``raddr_a/b/c``, ``waddr``, ``wdata``, ``we`` -> ``rdata_a/b/c``).

Read latency is **measured**: the output offset L (edges; 0 = the "pre"
sample of the same cycle, L >= 1 = the "post" sample of row t + L - 1, the
harness's count) at which the defined slots of the first 3,000 cycles agree;
then every defined slot is compared at that L. Write visibility is measured
by the regfile's own step probes (``units/vpu_regfile.py`` ``probes``) run on
the Catapult RTL with the same port mapping: MiniTPU gives read 0 and
write -> read 1.
"""
import argparse, os, re, sys, time

import numpy as np

sys.path.insert(0, os.getcwd())
from examples.minitpu.harness import check, rtl  # noqa: E402
from examples.minitpu.units import vpu_regfile as u  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("prj")
ap.add_argument("--top", default="rf_0")
ap.add_argument("--inst", default="w16")
ap.add_argument("--n", type=int, default=0)
ap.add_argument("--pad", type=int, default=8)
a = ap.parse_args()

v1 = os.path.join(os.path.abspath(a.prj), "build", "Catapult", f"{a.top}.v1")
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
    col = res["pre" if L == 0 else "post"][omap[port]]  # uint64[rows, nwords]
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
for p in u.RESP:
    d = reason[p] == ""
    g = got_at(L, p)
    diff = d & (g != want[p]).any(axis=1)
    tot += int(d.sum())
    masked += int((~d).sum())
    bad += int(diff.sum())
    for i in np.flatnonzero(diff)[:3]:
        lab = next(lb for lb, s0, s1 in spans if s0 <= i < s1)
        first.append(f"{lab} cycle {i} {p}: catapult {rtl.unpack(g[i:i+1])[0]:x} minitpu {rtl.unpack(want[p][i:i+1])[0]:x}")


# step probes, Catapult side (u.probes builds MiniTPU-named traces)
def probes():
    out = []
    w = u.WIDTH[a.inst]
    from examples.minitpu.harness.traces import Trace
    for port in "abc":
        for lab, build_ev in (("read", 0), ("write -> read", 1)):
            t = Trace(u._defaults())
            if build_ev == 0:
                t.cycle(waddr_i=3, wdata_i=1, we_i=1).cycle(waddr_i=4, wdata_i=(1 << w) - 2, we_i=1)
                t.idle(8, **{f"raddr_{port}_i": 3})
                ev = len(t)
                t.idle(8, **{f"raddr_{port}_i": 4})
            else:
                t.cycle(waddr_i=5, wdata_i=1, we_i=1)
                t.idle(8, **{f"raddr_{port}_i": 5})
                ev = len(t)
                t.cycle(**{f"raddr_{port}_i": 5}, waddr_i=5, wdata_i=2, we_i=1)
                t.idle(8, **{f"raddr_{port}_i": 5})
            c = t.cmd()
            nn = len(t)
            # "pre" probe sees a 0-edge path; "post" counts edges as rtl.probe_trace does
            m0 = rtl.probe_trace(cat_unit("pre"), cat_cmd(c, nn, P), omap[f"rdata_{port}_o"], ev + P)
            m1 = rtl.probe_trace(cat_unit("post"), cat_cmd(c, nn, P), omap[f"rdata_{port}_o"], ev + P)
            mt = rtl.probe_trace(unit, {kk: rtl.pack(c[kk], ww) for kk, ww in unit.inputs},
                                 f"rdata_{port}_o", ev)
            out.append((f"{lab} {port}", mt, m0, m1))
    return out


name = os.path.basename(a.prj.rstrip("/")).replace(".prj", "")
print(f"RTL-TRACE vpu_regfile:{a.inst} {name}: defined {tot - bad}/{tot} at measured read latency L={L} "
      f"(MiniTPU 0); {masked} masked; first-3000 hits by L {hits}; {n} cycles; {time.time() - t0:.1f}s")
for f in first:
    print(f"    e.g. {f}")
for lab, mt, m0, m1 in probes():
    print(f"    PROBE {lab}: MiniTPU {mt}; Catapult pre-sampled {m0}, post-sampled {m1}")
