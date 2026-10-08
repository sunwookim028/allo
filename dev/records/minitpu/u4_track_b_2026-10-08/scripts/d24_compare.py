# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""README D-24: the cycle-locked VPU side against the self-timed one, on the
same programs.

    $ALLO_PYTHON dev/records/minitpu/u4_track_b_2026-10-08/scripts/d24_compare.py [--prj DIR] [--out DIR]

A. **Claim timing, measured in csim** on Phase 0's ``vpu_writeback`` traces
   (every class alone, the 586-case pair sweep, random streams, early/late
   pops): ``units/vpu_wb`` ``locked`` and ``selftimed``, per class the cycles
   from a command's issue to its claim at the write port (``L``; ``W = L + 2``).
   The RTL's are the reference's (Phase 0: REF-MATCH).
B. **What the programs see**: Phase 0's sequencer programs (asm-scheduled
   straight, loops, DMA, four random programs, the two shipped images, and the
   two illegal ones), their RTL issue trace (``seq_issue`` reference), replayed
   against three calendars -- the RTL's/locked one, the self-timed one measured
   in csim (A), and the self-timed one re-derived from the Catapult manifests
   plus the links (3 cycles a hop, ``u3_fifo_composed``): RAW reads of a VREG
   still in flight, WAW pairs whose writes land out of issue order, and cycles
   in which two claims meet the one write port.
C. **The delta each would publish** (``template/gen_isa_delta``): the locked
   calendar is ``v1`` (DELTA-MATCH); the self-timed one is a version.
"""

import argparse
import json
import os
import statistics
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

BIT = {"load": 0, "alu": 1, "sfu": 2, "reduce": 3, "lane_reduce": 3, "mpop": 4, "txout": 5}
CLASSES = ("load", "alu", "sfu", "reduce", "lane_reduce", "txout", "mpop")
# Catapult manifests (3.33 ns, no latency= constraint: what a self-timed unit gets)
MANIFEST = {
    "alu": (2, "u1_catapult_units_2026-10-02 csyn_summary w_alu_bits_3p33: latency 2, II 1"),
    "sfu": (1, "u3_track_c_2026-10-04 logs/catapult/sfu_bits_3p33_free/latency.json: sfu_k_0 1, scheduled"),
    "txout": (1, "u3_track_c_2026-10-04 logs/catapult/tx_wide_3p33_free/latency.json: tx_0 1, scheduled"),
}
NO_MANIFEST = {
    "load": "VMEM compute port: no Catapult manifest of the 4,096-word port at II 1 (asic_memories: rw port refuses II 1); declared D-12 latency 3 + request register kept",
    "reduce": "tree at N=64 never built; tree16_bits_landed_3p33 is 'unreliable' (rolled loops), tree16_staged_3p33_free is N=16",
    "lane_reduce": "as reduce",
    "mpop": "the Allo MXU's pop (minitpu_rtl_mxu) runs below one token a cycle at depth 2; its latency is unresolved there",
}
CAT_HOP = 3   # Catapult push->pop per link (u3_fifo_composed_2026-10-02: 3 cycles on every one-cycle pair)
HOPS = 2      # issue -> class unit -> write-port owner


def run_wb(variant, prj):
    import allo.dataflow as df
    from examples.minitpu.harness import check
    from examples.minitpu.units import vpu_wb as U

    cmd, spans = check._trace_all(U, U.DEFAULT, 0, variant)
    n = len(next(iter(cmd.values())))
    make, runner = U.VARIANTS[variant]
    mod = df.build(make(n, 16, U.DEFAULT), target="systemc", mode="csim", project=os.path.join(prj, f"d24_{variant}"))
    got = runner(mod, cmd, n, 16)
    return cmd, spans, got


def claim_latency(cmd, got, order):
    """Per class, the issue -> mux distance of every claim on this run.

    ``order="mux"`` (a run that is cycle-locked): a claim is found at its own
    mux cycle when its source bit is high there -- what UNIT-MATCH per cycle
    already says, restated as a latency. ``order="issue"`` (the self-timed run):
    the class units and the merge keep issue order per source stream, so the
    k-th claim of a source bit is its k-th event (reduce and lane share bit 3,
    both in issue order through the one V unit)."""
    from examples.minitpu.harness import ref_ctrl_wb as R

    rst = [int(x) for x in cmd["rst_ni"]]
    _, claims, _ = R.writeback_trace({"rst_ni": rst, "ctrl_i": [int(x) for x in cmd["ctrl_i"]]})
    n = len(rst)
    claims = sorted(c for c in claims if c[3] < n)
    src = [int(v) for v in got["wb_src_o"]]
    out = {}
    if order == "mux":
        for t, cls, vd, m in claims:
            if (src[m] >> BIT[cls]) & 1:
                out.setdefault(cls, []).append(m - t)
    else:
        by_bit = {}
        for t, cls, vd, m in claims:
            by_bit.setdefault(BIT[cls], []).append((t, cls))
        ev = {}
        for c, v in enumerate(src):
            for b in range(6):
                if (v >> b) & 1:
                    ev.setdefault(b, []).append(c)
        for b, cl in by_bit.items():
            e = ev.get(b, [])
            for k, (t, cls) in enumerate(cl):
                if k < len(e):
                    out.setdefault(cls, []).append(e[k] - t)
    rtl_l = {cls: [m - t for t, c2, _, m in claims if c2 == cls] for cls in CLASSES}
    return out, rtl_l


def summarise(lat):
    return {c: {"median": statistics.median(v), "min": min(v), "max": max(v), "n": len(v)} for c, v in lat.items() if v}


def program_exposure(W):
    """RAW / WAW-reorder / port meetings on Phase 0's sequencer programs under
    the calendar ``W`` (class -> claim cycle from issue)."""
    from examples.minitpu.harness import ref_ctrl_decode as D
    from examples.minitpu.harness import rtl
    from examples.minitpu.harness.ref_ctrl_issue import issue_trace
    from examples.minitpu.units import seq_issue as SI
    from examples.minitpu.units import sequencer as S

    rows = []
    for label, cmd, legal in S.traces("shipped"):
        packed = {p: rtl.pack(cmd[p], w) for p, w in S.INPUTS}
        want, reason, _ = issue_trace(packed)
        ctrl = rtl.unpack(want["vpu_ctrl_o"])
        writes, reads = [], []
        for t, v in enumerate(ctrl):
            if reason["vpu_ctrl_o"][t]:
                continue
            f = D.unpack(D.VPU_CTRL, int(v))
            if f["alu_valid"]:
                writes.append((t, "alu", f["alu_vd"]))
                reads.append((t, f["raddr_a"]))
                if f["alu_op"] != 3:
                    reads.append((t, f["raddr_b"]))
            if f["sfu_valid"]:
                writes.append((t, "sfu", f["sfu_vd"]))
                reads.append((t, f["raddr_a"]))
            if f["reduce_valid"]:
                writes.append((t, "lane_reduce" if f["reduce_lane"] else "reduce", f["reduce_vd"]))
                reads.append((t, f["raddr_a"]))
            if f["txin_valid"]:
                reads.append((t, f["raddr_a"]))
            if f["txout_valid"]:
                writes.append((t, "txout", f["txout_vd"]))
            if f["vmem.valid"]:
                if f["vmem.op"]:
                    reads.append((t, f["vmem.vreg_idx"]))
                else:
                    writes.append((t, "load", f["vmem.vreg_idx"]))
            if f["vmatload_valid"]:
                reads += [(t, (f["vmatload_base"] + k) & 31) for k in range(4)]
            if f["vmatpush_valid"]:
                reads.append((t, f["vmatpush_vs"]))
            if f["vmatpop_valid"]:
                writes.append((t, "mpop", f["vmatpop_vd"]))
        res = {}
        for name, cal in W.items():
            land = [(t, t + cal[c], r) for t, c, r in writes]
            raw = sum(1 for t2, r2 in reads for t1, w1, r1 in land if r1 == r2 and t1 < t2 <= w1)
            waw = 0
            by_r = {}
            for t1, w1, r1 in land:
                by_r.setdefault(r1, []).append((t1, w1))
            for v in by_r.values():
                v.sort()
                waw += sum(1 for i in range(len(v)) for j in range(i + 1, len(v)) if v[i][1] >= v[j][1])
            cyc = {}
            for _, w1, _ in land:
                cyc[w1] = cyc.get(w1, 0) + 1
            meet = sum(1 for c in cyc.values() if c > 1)
            res[name] = (raw, waw, meet)
        rows.append((label, legal, len(writes), len(reads), res))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prj", default="/tmp/u4b_d24")
    ap.add_argument("--out", default=os.path.join(ROOT, "dev", "records", "minitpu", "u4_track_b_2026-10-08", "logs"))
    a = ap.parse_args()
    from examples.minitpu.template import gen_isa_delta as G
    from examples.minitpu.template.calendar import Calendar, Source

    print("A. claim timing in csim on vpu_writeback's traces (issue -> write-port mux, L; W = L + 2)")
    lat = {}
    for v in ("locked", "selftimed"):
        cmd, spans, got = run_wb(v, a.prj)
        meas, rtl_l = claim_latency(cmd, got, "mux" if v == "locked" else "issue")
        lat[v] = summarise(meas)
        if v == "locked":
            lat["rtl"] = summarise(rtl_l)
    print(f"   {'class':12s} {'rtl L':>12s} {'locked L':>12s} {'selftimed L (median, min..max, n)':>40s}")
    for c in CLASSES:
        r, l, s = lat["rtl"].get(c), lat["locked"].get(c), lat["selftimed"].get(c)
        fmt = lambda x: f"{x['median']:g} ({x['min']}..{x['max']})" if x else "-"  # noqa: E731
        print(f"   {c:12s} {fmt(r):>12s} {fmt(l):>12s} {fmt(s) + (f', n={s[chr(110)]}' if s else ''):>40s}")
    locked = Calendar()
    st_csim = Calendar(sources=tuple(Source(c, "selftimed csim", int(lat["selftimed"][c]["median"]),
                                            "d24_compare A (median)") for c in CLASSES if c in lat["selftimed"]),
                       name="selftimed-csim")
    man = []
    for s in locked.sources:
        L, why = MANIFEST.get(s.cls, (s.latency, "declared (no scheduled manifest): " + NO_MANIFEST.get(s.cls, "")))
        man.append(Source(s.cls, s.unit, L, why))
    st_cat = Calendar(sources=tuple(man), hop=HOPS * CAT_HOP, name="selftimed-cat3p33")
    W = {"locked (= RTL)": locked.W, "selftimed csim": st_csim.W, "selftimed Catapult 3.33": st_cat.W}
    print("\n   calendars W (claim cycle from issue):")
    for k, v in W.items():
        print(f"   {k:26s} " + ", ".join(f"{c} {v.get(c)}" for c in CLASSES))

    print("\nB. Phase 0's sequencer programs under each calendar: RAW in flight / WAW landing out of issue order / cycles with two claims at the write port")
    rows = program_exposure(W)
    tot = {k: [0, 0, 0] for k in W}
    for label, legal, nw, nr, res in rows:
        cells = "  ".join(f"{k.split()[0][:6]}{'-' + k.split()[1][:4] if len(k.split()) > 1 and k.startswith('self') else ''}"
                          f" {r[0]}/{r[1]}/{r[2]}" for k, r in res.items())
        print(f"   {label[:28]:28s} {'legal' if legal else 'ILLEGAL':7s} w{nw:5d} r{nr:5d}  {cells}")
        if legal:
            for k, r in res.items():
                tot[k] = [x + y for x, y in zip(tot[k], r)]
    print("   legal programs, total RAW/WAW/meet: " + "; ".join(f"{k}: {v[0]}/{v[1]}/{v[2]}" for k, v in tot.items()))

    print("\nC. the deltas (template/gen_isa_delta, base: the pinned isa/latency.json, version v1)")
    base = json.load(open(os.path.join(a.out, "..", "pins", "minitpu_9622f754_isa_latency.json")))
    for cal, name, unres, backend in (
            (locked, "allo-locked", None, "cycle-locked, simulator + SystemC csim"),
            (st_cat, "allo-selftimed",
             {k: NO_MANIFEST[c] for c, k in (("load", "WB_W_VLD"), ("reduce", "WB_W_REDUCE"),
                                             ("lane_reduce", "WB_W_LANE_REDUCE"), ("mpop", "WB_W_MPOP_FIRST"),
                                             ("mpop", "WB_W_MPOP_LAST"))},
             f"self-timed, Catapult 3.33 ns manifests + {HOPS} links x {CAT_HOP} cycles")):
        block = {"versions": {"list": {name: G.calendar_deltas(base, cal, allo_commit=G._allo_commit(), backend=backend,
                                                               unresolved_keys=unres)}}}
        path = os.path.join(a.out, f"versions_{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(block, indent=1) + "\n")
        diff = [(k, d, p) for k, d, p in G.check_base(base, cal, "v1") if d != p]
        print(f"   {name}: {len(block['versions']['list'][name]['deltas'])} deltas, "
              f"{len(block['versions']['list'][name]['unresolved'])} unresolved -> {os.path.relpath(path, ROOT)}; "
              + ("DELTA-MATCH v1 (no change)" if not diff else
                 "differs from v1 on " + ", ".join(f"{k} {p}->{d}" for k, d, p in diff)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
