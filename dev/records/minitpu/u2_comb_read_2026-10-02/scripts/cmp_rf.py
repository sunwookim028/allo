# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A generated regfile RTL vs MiniTPU's ``vpu_regfile.sv``, per cycle, with the
read latency and the write visibility MEASURED (U2 comb-read record).

    python cmp_rf.py <concat_sim_rtl.v> --top rf_comb \
        --ports ra,rb,rc,wa,wd,we:qa,qb,qc [--rst rst] [--inst w16] [--n N]

The U2 pilot's ``cmp_trace.py`` with the port names given on the command
line (MiniTPU order: raddr_a/b/c, waddr, wdata, we -> rdata_a/b/c) instead of
read from an emitted ``kernel.cpp``. Both RTLs run in the harness ``trace``
shape; the generated module gets ``PAD`` reset rows first. L is the output
offset at which the defined slots of the first 3,000 cycles agree: 0 = the
"pre" sample of the same cycle (an asynchronous read, MiniTPU's), L >= 1 =
the "post" sample of row t + L - 1. Then every defined slot is compared at
that L, and the regfile's own step probes are run on both RTLs.
"""
import argparse, os, sys, time

import numpy as np

sys.path.insert(0, os.getcwd())
from examples.minitpu.harness import check, rtl  # noqa: E402
from examples.minitpu.harness.traces import Trace  # noqa: E402
from examples.minitpu.units import vpu_regfile as u  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("--top", default="rf_comb")
ap.add_argument("--ports", required=True, help="in1,..,in6:out1,out2,out3 in MiniTPU order")
ap.add_argument("--rst", default="rst", help="active-low reset port name, or none")
ap.add_argument("--inst", default="w16")
ap.add_argument("--n", type=int, default=0)
ap.add_argument("--pad", type=int, default=8)
ap.add_argument("--label", default="")
a = ap.parse_args()

w = u.WIDTH[a.inst]
i_names, o_names = (s.split(",") for s in a.ports.split(":"))
assert len(i_names) == len(u.CMD) and len(o_names) == len(u.RESP)
imap = dict(zip(u.CMD, i_names))
omap = dict(zip(u.RESP, o_names))
widths = dict(zip(u.CMD + u.RESP, [5, 5, 5, 5, w, 1, w, w, w]))
ins = [(imap[p], widths[p]) for p in u.CMD]
outs = [(omap[p], widths[p]) for p in u.RESP]
has_rst = a.rst != "none"

cmd, spans = check._trace_all(u, a.inst, a.n)
n = len(next(iter(cmd.values())))
unit = u.INSTANCES[a.inst]
packed = {p: rtl.pack(cmd[p], ww) for p, ww in unit.inputs}
t0 = time.time()
want = rtl.run_trace(unit, packed, seed=1)
_, reason, _ = u.REF(a.inst, packed)


def gen_unit(sample):
    return rtl.RtlUnit(top=a.top, sources=[os.path.abspath(a.src)],
                       inputs=([(a.rst, 1)] if has_rst else []) + ins,
                       outputs=[(p, ww, sample) for p, ww in outs], shape="trace",
                       clk="clk", rst_n=a.rst if has_rst else "", reset=False)


def gen_cmd(c, nn, pad):
    out = {}
    if has_rst:
        out[a.rst] = [0] * pad + [1] * (nn + 16)
    for mp, cp in imap.items():
        out[cp] = [0] * pad + [int(x) for x in c[mp][:nn]] + [0] * 16
    return out


res = {s: rtl.run_trace(gen_unit(s), gen_cmd(cmd, n, a.pad), seed=1) for s in ("pre", "post")}
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
for p in u.RESP:
    d = reason[p] == ""
    g = got_at(L, p)
    diff = d & (g != want[p]).any(axis=1)
    tot += int(d.sum())
    masked += int((~d).sum())
    bad += int(diff.sum())
    for i in np.flatnonzero(diff)[:3]:
        lab = next(lb for lb, s0, s1 in spans if s0 <= i < s1)
        first.append(f"{lab} cycle {i} {p}: generated {rtl.unpack(g[i:i+1])[0]:x} minitpu {rtl.unpack(want[p][i:i+1])[0]:x}")


def probes():
    out = []
    for port in "abc":
        for lab, ev_kind in (("read", 0), ("write -> read", 1)):
            t = Trace(u._defaults())
            if ev_kind == 0:
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
            m0 = rtl.probe_trace(gen_unit("pre"), gen_cmd(c, nn, P), omap[f"rdata_{port}_o"], ev + P)
            m1 = rtl.probe_trace(gen_unit("post"), gen_cmd(c, nn, P), omap[f"rdata_{port}_o"], ev + P)
            mt = rtl.probe_trace(unit, {kk: rtl.pack(c[kk], ww) for kk, ww in unit.inputs},
                                 f"rdata_{port}_o", ev)
            out.append((f"{lab} {port}", mt, m0, m1))
    return out


name = a.label or os.path.basename(os.path.dirname(os.path.abspath(a.src)))
print(f"RTL-TRACE vpu_regfile:{a.inst} {name}: defined {tot - bad}/{tot} at measured read latency L={L} "
      f"(MiniTPU 0); {masked} masked; first-3000 hits by L {hits}; {n} cycles; {time.time() - t0:.1f}s")
for f in first:
    print(f"    e.g. {f}")
for lab, mt, m0, m1 in probes():
    print(f"    PROBE {lab}: MiniTPU {mt}; generated pre-sampled {m0}, post-sampled {m1}")
