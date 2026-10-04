"""Catapult RTL of an Allo ``vpu_word_array`` Wire variant vs MiniTPU's RTL, per cycle.

    python cmp_trace.py <prj> [--top wa_0] [--inst narrow] [--n N] [--pad 8]

From the worktree root, after ``source examples/minitpu/harness/env-zhang21.sh``.
``u2_regfile_2026-10-02/scripts/cmp_trace.py`` for the word array: two
response ports with **different** latencies, so the output offset ``L`` is
measured per port (``got[t]`` at the "post" row ``t + L - 1``, the harness's
edge count; MiniTPU: compute 3, DMA 2), and the unit's own step probes
(``units/vpu_word_array.py`` ``probes``: read per port, write -> read for
every port pair) run on the Catapult RTL with the same port mapping. Ports
are read from ``kernel.cpp`` in declaration order and mapped onto MiniTPU's
``CMD``/``RESP`` order. Also writes/reads the D-10 manifest
(``latency.json``) and prints the ``harness/latency.py`` verdict for the
unit kernel against the measured compute-port latency.
"""
import argparse, os, re, sys, time

import numpy as np

sys.path.insert(0, os.getcwd())
from examples.minitpu.harness import check, latency, rtl  # noqa: E402
from examples.minitpu.harness.traces import Trace  # noqa: E402
from examples.minitpu.units import vpu_word_array as u  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("prj")
ap.add_argument("--top", default="wa_0")
ap.add_argument("--inst", default="narrow")
ap.add_argument("--n", type=int, default=0)
ap.add_argument("--pad", type=int, default=8)
a = ap.parse_args()

prj = os.path.abspath(a.prj)
cat = next(d for d in ("Catapult", "Catapult_1") if os.path.isdir(os.path.join(prj, "build", d, f"{a.top}.v1")))
v1 = os.path.join(prj, "build", cat, f"{a.top}.v1")
src = os.path.join(v1, "concat_sim_rtl.v")
k = open(os.path.join(prj, "kernel.cpp")).read()
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
ww, words, aw, rl, drl = u.GEOM[a.inst]
decl = {"compute_rdata_o": rl, "dma_rdata_o": drl}
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
Ls, hits = {}, {}
for p in u.RESP:
    hits[p] = {}
    for L in range(0, 10):
        d = reason[p][:m] == ""
        hits[p][L] = int((d & (got_at(L, p)[:m] == want[p][:m]).all(axis=1)).sum())
    Ls[p] = max(hits[p], key=hits[p].get)
tot = bad = masked = 0
first = []
per_port = {}
for p in u.RESP:
    d = reason[p] == ""
    g = got_at(Ls[p], p)
    diff = d & (g != want[p]).any(axis=1)
    tot += int(d.sum())
    masked += int((~d).sum())
    bad += int(diff.sum())
    per_port[p] = (int(d.sum()) - int(diff.sum()), int(d.sum()))
    for i in np.flatnonzero(diff)[:3]:
        lab = next(lb for lb, s0, s1 in spans if s0 <= i < s1)
        first.append(f"{lab} cycle {i} {p}: catapult {rtl.unpack(g[i:i+1])[0]:x} minitpu {rtl.unpack(want[p][i:i+1])[0]:x}")


def _probe(unit_, c, port, ev):
    """``rtl.probe_trace`` that reports "never moved" instead of asserting."""
    try:
        return rtl.probe_trace(unit_, c, port, ev)
    except AssertionError as e:
        return f"never moved ({e})"


def probes():
    """The unit's step probes (MiniTPU-named traces) on both RTLs."""
    out = []
    lat = {"compute": rl, "dma": drl}
    one, two = 1, (1 << ww) - 2
    resm, resc = {}, {}
    for p in u.PORTS:
        o = "dma" if p == "compute" else "compute"
        t = Trace(u._defaults())
        t.cycle(**u._acc(o, 1, one)).cycle(**u._acc(o, 2, two)).idle(2)
        t.idle(8, **u._acc(p, 1))
        ev = len(t)
        t.idle(8, **u._acc(p, 2))
        c = t.cmd()
        resm[p] = rtl.probe_trace(unit, {kk: rtl.pack(c[kk], w2) for kk, w2 in unit.inputs}, f"{p}_rdata_o", ev)
        resc[p] = _probe(cat_unit("post"), cat_cmd(c, len(t), P), omap[f"{p}_rdata_o"], ev + P)
        out.append((f"read {p}", lat[p], resm[p], resc[p]))
    for wp in u.PORTS:
        for rp in u.PORTS:
            t = Trace(u._defaults())
            t.cycle(**u._acc(wp, 3, one)).idle(2)
            t.idle(8, **u._acc(rp, 3))
            ev = len(t)
            row = u._acc(wp, 3, two)
            if wp != rp:
                row.update({f"{rp}_en_i": 0, f"{rp}_addr_i": 3})  # held address: the probe for an unconditional-read port
            t.cycle(**row)
            t.idle(8, **u._acc(rp, 3))
            c = t.cmd()
            mm = rtl.probe_trace(unit, {kk: rtl.pack(c[kk], w2) for kk, w2 in unit.inputs}, f"{rp}_rdata_o", ev) - resm[rp]
            mc = _probe(cat_unit("post"), cat_cmd(c, len(t), P), omap[f"{rp}_rdata_o"], ev + P)
            mc = mc - resc[rp] if isinstance(mc, int) and isinstance(resc[rp], int) else mc
            out.append((f"write {wp} -> read {rp} (visibility)", 1, mm, mc))
    return out


name = os.path.basename(prj.rstrip("/")).replace(".prj", "")
# MiniTPU's rdata rows are post-sampled, so an offset of 1 (the harness's edge
# count from the "pre" sample) is the SAME output row as MiniTPU's; the read
# latency itself is the step probe below.
print(f"RTL-TRACE vpu_word_array:{a.inst} {name}: defined {tot - bad}/{tot} at output-row offset "
      + ", ".join(f"{p.split('_')[0]} {Ls[p] - 1:+d} rows vs MiniTPU {per_port[p][0]}/{per_port[p][1]}" for p in u.RESP)
      + f"; {masked} masked; {n} cycles; {time.time() - t0:.1f}s")
for p in u.RESP:
    print(f"    first-3000 hits by L, {p}: {hits[p]}")
for f in first:
    print(f"    e.g. {f}")
pr = probes()
for lab, dcl, mt, mc in pr:
    print(f"    PROBE {lab}: declared {dcl}; MiniTPU {mt}; Catapult {mc}")

# D-10: the backend's manifest, held to the measured latency. The kernel's own
# port-to-port latency is the measured read latency minus the (L - 1)
# iterations the read spends in the pipe-as-data (an iteration shift, not a
# schedule property): the manifest can only know the former.
mc_read = {lab.split()[1]: mc for lab, _, _, mc in pr if lab.startswith("read")}
kern_meas = {p: (mc_read[p] - (decl[f"{p}_rdata_o"] - 1)) if isinstance(mc_read[p], int) else None for p in u.PORTS}
print(f"    kernel port-to-port latency, measured = read probe - (L - 1): {kern_meas}")
try:
    from allo.backend import catapult as cb
    man = cb.write_latency_manifest(prj, a.top)
    for kern in man["units"]:
        print("    " + latency.verdict(man, kern, kern_meas["compute"] or 0, declared=decl["compute_rdata_o"]))
        print(f"    manifest {kern}: " + ", ".join(f"{k}={v}" for k, v in man["units"][kern].items() if k != "process"))
except Exception as e:  # noqa: BLE001
    print(f"    MANIFEST n/a: {type(e).__name__}: {str(e)[:300]}")
