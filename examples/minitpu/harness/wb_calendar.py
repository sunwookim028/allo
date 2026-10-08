# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U4: the VREG write-port calendar as data, from the RTL, held to its three
written copies.

    $ALLO_PYTHON -m examples.minitpu.harness.wb_calendar

For every VREG-writing op class it measures on ``vpu.sv`` (through
``units/vpu_writeback``): the mux cycle ``L`` (``wb_source_valid``), the
write-enable cycle ``W`` (``wb_local_valid_q``) and, by function, the first
cycle VREG port A returns the new value (``W + 1``). Then it sweeps every
ordered pair of classes at every issue distance ``d`` and records where
``vpu.sv``'s ``$onehot0`` writeback assertion fires: the *physical* calendar.

That is held to

* ``asm.py``: ``_decode(word)["writes"]`` spans and ``_Timeline``'s
  ``first_free_write`` on the same two bundles (the assembler's calendar);
* ``docs/isa_latency.json``: ``rtl_params`` and each operation's ``w``;
* ``sequencer_pkg.sv``: ``WB_W_*`` = ``VPU_*_LATENCY + VPU_WB_STAGES``.

Prints one ``CALENDAR-MATCH``/``CALENDAR-DIFF`` line per class and per
source, and the pair sweep's agreement.
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from examples.minitpu.harness import minitpu_asm, rtl  # noqa: E402
from examples.minitpu.harness import ref_ctrl_wb as R  # noqa: E402
from examples.minitpu.harness.traces import rng_for  # noqa: E402
from examples.minitpu.units import vpu_writeback as U  # noqa: E402

CLASSES = U.CLASSES  # the fixed-offset sources; mpop separately
JSON_KEY = {"load": "WB_W_VLD", "alu": "WB_W_ALU", "sfu": "WB_W_SFU", "reduce": "WB_W_REDUCE",
            "lane_reduce": "WB_W_LANE_REDUCE", "txout": "WB_W_TXOUT", "mpop": "WB_W_MPOP_FIRST"}


def asm_word(a, cls):
    b = a.AsmBuilder()
    return {"load": lambda: b.vld(3, 0), "alu": lambda: b.vadd(3, 1, 2), "sfu": lambda: b.vexp(3, 1),
            "reduce": lambda: b.vredsum(3, 1), "lane_reduce": lambda: b.vlanesum(3, 1),
            "txout": lambda: b.vtxout(3, 0), "mpop": lambda: b.vmatpop(3)}[cls]()


def asm_w(a, cls):
    spans = [list(s) for _, s in a._decode(asm_word(a, cls))["writes"]]
    assert len(spans) == 1, spans
    return spans[0]


def asm_collides(a, ca, cb, d):
    """Does asm's calendar refuse ``cb`` issued ``d`` after ``ca``?"""
    tl = a._Timeline()
    da, db = a._decode(asm_word(a, ca)), a._decode(asm_word(a, cb))
    tl.commit(da, 0)
    return any(d + w in tl.write_port for w in a._wb_span(db))


def pair_sweep():
    """RTL: every (a, b, d) as its own short trace (the assertion's time stamp
    is not a cycle in this harness, finding H2); returns the cases where
    vpu.sv's writeback assertion fired, and those the W rule predicts."""
    blocks, fired, merged = [], set(), set()
    for ca in CLASSES:
        for cb in CLASSES:
            for d in range(0, 17):
                if d == 0 and (ca == cb or (ca != "load" and cb != "load")):
                    continue
                rng = rng_for("u4-calendar", ca, cb, d)
                p = U.Prog(rng)
                p.put(6, U._op(rng, ca, rng.randrange(32)), ca)
                p.put(6 + d, U._op(rng, cb, rng.randrange(32)), cb)
                res = rtl.run_trace(U.RTL, p.cmd())
                blocks.append((6, ca, cb, d))
                if any("concurrent VREG writeback" in m for _, m in rtl.last_asserts):
                    fired.add((ca, cb, d))
                # physical: two writes merged into one write-enable cycle
                if sum(1 for v in rtl.unpack(res["wb_local_valid_o"])[6:] if v) < 2:
                    merged.add((ca, cb, d))
    rule = {(ca, cb, d) for _, ca, cb, d in blocks if R.W[ca] == d + R.W[cb]}
    return blocks, fired, merged, rule


SEQ_CLASSES = list(CLASSES) + ["mpop"]


def seq_bundle(b, cls, vd):
    return {"load": lambda: {"x": b.vld(vd, 4 * vd)}, "alu": lambda: {"v": b.vadd(vd, 30, 31)},
            "sfu": lambda: {"v": b.vexp(vd, 30)}, "reduce": lambda: {"v": b.vredsum(vd, 30)},
            "lane_reduce": lambda: {"v": b.vlanesum(vd, 30)}, "txout": lambda: {"v": b.vtxout(vd, 0)},
            "mpop": lambda: {"m": b.vmatpop(vd)}}[cls]()


def seq_pair_sweep(a):
    """The sequencer's own sim-only calendar (sequencer.sv:277-432): bundle
    ``a`` with delay ``d - 1``, then bundle ``b`` (``d = 0``: one bundle where
    the slots allow), on the whole sequencer; the cases where its write-port
    checks fire."""
    from examples.minitpu.units import sequencer as S

    fired, cases = set(), []
    for ca in SEQ_CLASSES:
        for cb in SEQ_CLASSES:
            for d in range(0, 17):
                b = a.AsmBuilder()
                sa, sb = seq_bundle(b, ca, 1), seq_bundle(b, cb, 2)
                if d == 0:
                    if set(sa) & set(sb):
                        continue
                    b.bundle(**sa, **sb)
                else:
                    b.bundle(**sa)
                    b.bundles[-1] = a._set_delay(b.bundles[-1], d - 1)
                    b.bundle(**sb)
                for _ in range(20):
                    b.bundle()
                b.bundle(b.halt())
                cmd = S.Run(b.bundles, rng_for("u4-cal-seq"), cycles=80).cmd()
                rtl.run_trace(S.RTL, {p: rtl.pack(cmd[p], w) for p, w in S.INPUTS})
                cases.append((ca, cb, d))
                if any("write-port collision" in m for _, m in rtl.last_asserts):
                    fired.add((ca, cb, d))
    rule = {(ca, cb, d) for ca, cb, d in cases if R.W[ca] == d + R.W[cb]}
    return cases, fired, rule


def main():
    a = minitpu_asm.load()
    home = rtl.minitpu_home()
    with open(os.path.join(home, "docs", "isa_latency.json"), encoding="utf-8") as f:
        doc = json.load(f)
    params = doc["rtl_params"]
    measured = {}
    for label, declared, got in U.probes("shipped"):
        cls = label.split(":")[0].split(" ")[0]
        kind = "L" if "(L)" in label else "W" if "(W)" in label else "RAW"
        measured.setdefault(cls, {}).setdefault(kind, set()).add(got)
    ok = True
    print("class        L(rtl) W(rtl) RAW(rtl) | W asm  W json  W pkg(L+2) | verdict")
    for cls in list(CLASSES) + ["mpop"]:
        m = measured[cls]
        lw, ww, raw = (sorted(m["L"]), sorted(m["W"]), sorted(m["RAW"]))
        wa = asm_w(a, cls)
        wj = params[JSON_KEY[cls]]
        wp = R.L[cls] + R.VPU_WB_STAGES
        good = (len(ww) == 1 and ww[0] == wj == wp and wa == [ww[0]] and raw == [ww[0] + 1]
                and lw == [ww[0] - R.VPU_WB_STAGES])
        ok &= good
        print(f"{cls:12s} {lw[0]:6d} {ww[0]:6d} {raw[0]:8d} | {str(wa):6s} {wj:6d} {wp:9d}  | "
              f"{'CALENDAR-MATCH' if good else 'CALENDAR-DIFF'}")
    # the per-operation "w" in isa_latency.json's operation table
    ops = {"vadd": "alu", "vgelu": "sfu", "vredsum": "reduce", "vlanered": "lane_reduce",
           "vtxout": "txout", "vld": "load", "vmatpop": "mpop"}
    for row in doc["operations"]:
        for op in row["ops"]:
            if op in ops and row["w"] != params[JSON_KEY[ops[op]]]:
                ok = False
                print(f"CALENDAR-DIFF isa_latency.json operations[{op}].w={row['w']}")
    blocks, fired, merged, rule = pair_sweep()
    asm_set = {(ca, cb, d) for _, ca, cb, d in blocks if asm_collides(a, ca, cb, d)}
    same = merged == rule == asm_set
    ok &= same
    print(f"{'CALENDAR-MATCH' if same else 'CALENDAR-DIFF'} pair sweep: {len(blocks)} (a, b, d) cases; "
          f"two writes merged into one on the RTL in {len(merged)}, rule W_a = d + W_b gives {len(rule)}, "
          f"asm _Timeline refuses {len(asm_set)}")
    for x in sorted(merged ^ rule) + sorted(rule ^ asm_set):
        print("   differs:", x)
    print(f"   colliding cases: {sorted(merged)}")
    print(f"   vpu.sv $onehot0 (vpu.sv:370) fired on {len(fired)}; silent: {sorted(merged - fired)}")
    cases, sfired, srule = seq_pair_sweep(a)
    sasm = {(ca, cb, d) for ca, cb, d in cases if d and asm_collides(a, ca, cb, d)}
    sasm |= {(ca, cb, 0) for ca, cb, d in cases if d == 0 and R.W[ca] == R.W[cb]}
    print(f"{'CALENDAR-MATCH' if sfired == srule else 'CALENDAR-DIFF'} sequencer's sim-only calendar: "
          f"{len(cases)} (a, b, d) cases incl. vmatpop; its write-port checks fired on {len(sfired)}, "
          f"the rule gives {len(srule)}, asm refuses {len(sasm)}")
    print(f"   rule but not fired: {sorted(srule - sfired)}")
    print(f"   fired but not rule: {sorted(sfired - srule)}")
    print(f"   asm vs rule: {sorted(sasm ^ srule)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
