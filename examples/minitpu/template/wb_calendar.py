# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The VREG write-port calendar as a derived record (U4 plan Q2; README D-20, D-24).

    $ALLO_PYTHON -m examples.minitpu.template.wb_calendar            # check against Phase 0
    $ALLO_PYTHON -m examples.minitpu.template.wb_calendar --refusals

MiniTPU writes the calendar three times by hand (``asm.py``'s ``W_*``,
``isa_latency.json``'s ``rtl_params.WB_W_*``, ``sequencer_pkg.sv``'s
``WB_W_* = VPU_*_LATENCY + VPU_WB_STAGES``), and ``vpu_pkg.sv`` states each
unit's latency by hand because "the lab's compute modules cannot export their
latency". Here nothing is typed beside the number it must equal:

* each writeback source is **bound** to the unit whose declared latency it
  books (D-10: a booking, compared with the manifest per backend), read from
  that unit's own declaration -- ``units/alu.py``'s ``RTL.latency``,
  ``units/sfu.py``'s, ``template/legality.TreeGeometry``'s root and tap
  latencies, VMEM's D-12 compute-port read latency plus the request register,
  the transpose's and the pop engine's one register;
* the writeback unit (W1, ``units/vpu_wb.py``) declares its own latency,
  ``VPU_WB_STAGES`` = 2;
* ``W``, the first legal consumer ``W + 1``, the collision set, ``vmatpop``'s
  precondition and the sequencer's issue numbers are **properties**.

A *self-timed* composition (D-24) books every source through ``hops`` extra
Stream links; ``Calendar.selftimed(hop)`` derives that calendar from the same
binding plus the measured per-hop latency, which is what
``gen_isa_delta`` publishes as a ``versions.list.<name>`` delta.

``check_against_phase0`` holds every derived number to
``dev/records/minitpu/u4_phase0_2026-10-08/calendar.log`` (the RTL measured
three ways) and to the pair sweeps it records.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, replace

from examples.minitpu.template.legality import MxuGeometry, TreeGeometry

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
PHASE0_LOG = os.path.join(ROOT, "dev", "records", "minitpu", "u4_phase0_2026-10-08", "calendar.log")

# The VREG-writing classes, in Phase 0's order, and their seam key.
CLASSES = ("load", "alu", "sfu", "reduce", "lane_reduce", "txout", "mpop")
JSON_KEY = {"load": "WB_W_VLD", "alu": "WB_W_ALU", "sfu": "WB_W_SFU", "reduce": "WB_W_REDUCE",
            "lane_reduce": "WB_W_LANE_REDUCE", "txout": "WB_W_TXOUT", "mpop": "WB_W_MPOP_FIRST"}


@dataclass(frozen=True)
class Source:
    """One writeback source: the unit it is bound to and that unit's declared
    latency at the writeback mux (issue -> mux), with where it was read."""

    cls: str
    unit: str
    latency: int
    declared_at: str


def bound_sources(tree: TreeGeometry = TreeGeometry(), mxu: MxuGeometry = MxuGeometry()):
    """The binding of the shipped instance: every latency read from the
    declaration of the unit it books, never typed here."""
    from examples.minitpu.units import alu, sfu, vpu_word_array_d12  # noqa: F401 (declarations)

    vmem_rl = _vmem_compute_read_latency()
    return (
        # vld: vpu_vmem_simd registers the request, then the word array's compute port reads
        Source("load", "vmem.c (D-12 port) + request register", 1 + vmem_rl,
               "units/vpu_word_array_d12.py vmem port c latency + 1 (vpu_pkg VPU_VLD_LATENCY)"),
        Source("alu", "vpu_alu", alu.RTL.latency, "units/alu.py RTL.latency"),
        Source("sfu", "sfu", sfu.RTL.latency, "units/sfu.py RTL.latency"),
        Source("reduce", "xlu_reduction_tree (root)", tree.latency,
               "template/legality.py TreeGeometry.latency = LEAF + ADDER*LEVELS"),
        Source("lane_reduce", "xlu_reduction_tree (tap)", tree.tap_latency,
               "template/legality.py TreeGeometry.tap_latency = LEAF + ADDER*LANE_LEVELS"),
        Source("txout", "xlu_transpose", 1, "units/xlu_transpose.py read register (probe 1)"),
        Source("mpop", "mxu_pop_engine", 1, "mxu_pop_engine registers the command (VPU_MPOP_LATENCY)"),
    )


def _vmem_compute_read_latency():
    """VMEM's compute-port read latency as the D-12 prototype declares it."""
    from examples.minitpu.units import vpu_word_array_d12 as V

    for m in V.architecture(8).memories:
        if m.name == "vmem":
            for p in m.ports:
                if p.name == "c" and p.latency is not None:
                    return p.latency
    raise AssertionError("vpu_word_array_d12: no compute read port 'c' with a latency")


@dataclass(frozen=True)
class Calendar:
    """The write-port calendar of one composed instance."""

    sources: tuple = field(default_factory=bound_sources)
    WB_STAGES: int = 2          # the writeback unit's declared latency (W1)
    POP_BEATS: int = 1          # MXU_POP_BEATS: one FIFO entry is a whole VREG
    hop: int = 0                # extra issue->mux cycles per source (0: cycle-locked)
    mxu: MxuGeometry = MxuGeometry()
    S_LAT: int = 2              # scalar AGU pipe (track A's S1 booking; consumed here)
    name: str = "locked"

    # ---- derived (D-20): none of these is a field ---------------------------
    @property
    def L(self) -> dict:
        """Issue -> writeback mux, per class (the bound unit + any hops)."""
        return {s.cls: s.latency + self.hop for s in self.sources}

    @property
    def W(self) -> dict:
        """The claim cycle: issue -> VREG write enable."""
        return {c: lat + self.WB_STAGES for c, lat in self.L.items()}

    @property
    def first_consumer(self) -> dict:
        """The first issue cycle that reads the new value (``isa_latency``'s L = W + 1)."""
        return {c: w + 1 for c, w in self.W.items()}

    @property
    def W_MPOP_LAST(self) -> int:
        return self.W["mpop"] + self.POP_BEATS - 1

    @property
    def mpop_precondition(self) -> int:
        """A ``vmatpop`` books ``W['mpop']`` only if issued at least this many
        cycles after its ``vmatpush`` (``matrix.result_latency.vmatpush``)."""
        return self.mxu.result_latency

    def collisions(self, classes=CLASSES[:-1], dmax=16, pairs_d0=None):
        """Ordered pairs ``(a, b, d)``: ``b`` issued ``d`` after ``a`` claims the
        write port in the same cycle, ``W_a = d + W_b``."""
        W = self.W
        out = set()
        for a in classes:
            for b in classes:
                for d in range(0, dmax + 1):
                    if pairs_d0 is not None and d == 0 and not pairs_d0(a, b):
                        continue
                    if W[a] == d + W[b]:
                        out.add((a, b, d))
        return out

    def rtl_params(self) -> dict:
        """The seam's ``rtl_params.WB_W_*`` values this calendar implies."""
        W = self.W
        out = {JSON_KEY[c]: W[c] for c in CLASSES}
        out["WB_W_MPOP_LAST"] = self.W_MPOP_LAST
        return out

    def namespace(self) -> dict:
        ns = {f"L_{c}": v for c, v in self.L.items()}
        ns.update({f"W_{c}": v for c, v in self.W.items()})
        ns.update(WB_STAGES=self.WB_STAGES, POP_BEATS=self.POP_BEATS,
                  W_MPOP_LAST=self.W_MPOP_LAST, hop=self.hop)
        ns["bound"] = {s.cls: s.latency for s in self.sources}
        return ns

    def selftimed(self, hop: int, name: str = "selftimed"):
        """The same binding through ``hop`` extra cycles of Stream links per
        source (D-24): the calendar a self-timed composition publishes."""
        return replace(self, hop=hop, name=name)


def calendar_legality(p):
    """What the write port's correctness rests on, as one condition set."""
    assert p["WB_STAGES"] >= 1, (
        f"WB_STAGES={p['WB_STAGES']}: the OR-mux is registered at least once "
        f"(vpu.sv wb_stage_*); a combinational write port changes W for every class")
    for c, lat in p["bound"].items():
        assert p[f"L_{c}"] == lat + p["hop"], (
            f"L_{c}={p[f'L_{c}']} must be the bound unit's declared latency {lat} "
            f"(+ hop {p['hop']}): a calendar typed beside the unit outlives a change to it")
        assert p[f"W_{c}"] == p[f"L_{c}"] + p["WB_STAGES"], (
            f"W_{c}={p[f'W_{c}']} must be L_{c} + WB_STAGES = {p[f'L_{c}'] + p['WB_STAGES']} "
            f"(the seam: W = unit latency + VPU_WB_STAGES)")
    assert p["W_MPOP_LAST"] == p["W_mpop"] + p["POP_BEATS"] - 1, (
        f"W_MPOP_LAST={p['W_MPOP_LAST']} must be W_mpop + POP_BEATS - 1 = "
        f"{p['W_mpop'] + p['POP_BEATS'] - 1} (sequencer_pkg.sv:573)")
    assert p["POP_BEATS"] >= 1


# ---- Phase 0 ---------------------------------------------------------------------
def read_phase0(path=PHASE0_LOG):
    """Parse ``calendar.log``: per class (L, W, RAW, asm, json, pkg), the RTL's
    colliding pairs, and the sequencer sweep's rule-but-not-fired list."""
    rows, merged, seq = {}, None, None
    with open(path, encoding="utf-8") as f:
        text = f.read()
    for m in re.finditer(r"^(\w+)\s+(\d+)\s+(\d+)\s+(\d+) \| \[(\d+)\]\s+(\d+)\s+(\d+)\s+\| (CALENDAR-\w+)", text, re.M):
        c, L, W, raw, wa, wj, wp, verdict = m.groups()
        rows[c] = dict(L=int(L), W=int(W), RAW=int(raw), asm=int(wa), json=int(wj), pkg=int(wp), verdict=verdict)
    m = re.search(r"colliding cases: (\[.*?\])\n", text)
    if m:
        merged = set(eval(m.group(1)))  # noqa: S307 -- our own log, tuples of str/int
    m = re.search(r"sequencer's sim-only calendar: (\d+) .*?the rule gives (\d+), asm refuses (\d+)", text)
    if m:
        seq = dict(cases=int(m.group(1)), rule=int(m.group(2)), asm=int(m.group(3)))
    return rows, merged, seq


def check_against_phase0(cal: Calendar = None, path=PHASE0_LOG):
    """Every derived number against the Phase 0 measurement. Returns
    ``[(label, derived, measured, ok)]``."""
    cal = cal or Calendar()
    calendar_legality(cal.namespace())
    rows, merged, seq = read_phase0(path)
    out = []
    for c in CLASSES:
        r = rows[c]
        out += [(f"{c}: L (issue -> mux)", cal.L[c], r["L"]),
                (f"{c}: W (issue -> write enable)", cal.W[c], r["W"]),
                (f"{c}: first consumer W+1 (RAW by function)", cal.first_consumer[c], r["RAW"]),
                (f"{c}: W vs asm/json/pkg", cal.W[c], (r["asm"], r["json"], r["pkg"]))]
    # the pair sweep on vpu.sv: one V op per cycle, so d=0 only with a load
    d0 = lambda a, b: a != b and (a == "load" or b == "load")  # noqa: E731
    mine = cal.collisions(pairs_d0=d0)
    out.append(("pair sweep: colliding (a, b, d) on vpu.sv", len(mine), len(merged)))
    out.append(("pair sweep: same set", sorted(mine ^ merged), []))
    # the sequencer sweep (vmatpop included; d=0 where the slots allow)
    slot = {"load": "x", "mpop": "m"}
    d0s = lambda a, b: slot.get(a, "v") != slot.get(b, "v")  # noqa: E731
    seq_rule = cal.collisions(classes=CLASSES, pairs_d0=d0s)
    out.append(("sequencer sweep: rule gives (vmatpop incl.)", len(seq_rule), seq["rule"]))
    out.append(("vmatpop precondition (result latency)", cal.mpop_precondition, 85))
    res = []
    for lab, d, m in out:
        ok = d == m if not isinstance(m, tuple) else all(d == x for x in m)
        res.append((lab, d, m, ok))
    return res


def refusals():
    """One wrong declaration per relation, each refused by ``calendar_legality``."""
    base = Calendar().namespace()
    out = []
    for label, ns in (
        ("W typed beside an ALU whose latency changed to 4",
         base | {"bound": base["bound"] | {"alu": 4}}),
        ("W_sfu typed as the unit latency (the WB stages forgotten)", base | {"W_sfu": base["L_sfu"]}),
        ("a combinational write port (WB_STAGES 0)", base | {"WB_STAGES": 0}),
        ("WB_W_MPOP_LAST kept at v1-course's 6 with a one-beat pop", base | {"W_MPOP_LAST": 6}),
        ("tree tap latency copied from the root", base | {"L_lane_reduce": base["L_reduce"]}),
    ):
        try:
            calendar_legality(ns)
            out.append((label, "ACCEPTED (bug)"))
        except AssertionError as e:
            out.append((label, "refused: " + str(e).splitlines()[0][:100]))
    return out


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--refusals", action="store_true")
    a = ap.parse_args(argv)
    cal = Calendar()
    print("binding:")
    for s in cal.sources:
        print(f"   {s.cls:12s} L={s.latency:3d}  {s.unit}  [{s.declared_at}]")
    bad = 0
    for lab, d, m, ok in check_against_phase0(cal):
        bad += not ok
        print(f"{'CALENDAR-MATCH' if ok else 'CALENDAR-DIFF '} {lab}: derived {d} phase0 {m}")
    print(f"{'CALENDAR-MATCH' if not bad else 'CALENDAR-DIFF'} {cal.name}: rtl_params {cal.rtl_params()}")
    if a.refusals:
        for lab, verdict in refusals():
            print(f"   {lab}: {verdict}")
    return 1 if bad else 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
