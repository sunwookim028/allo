# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U4 track A, README D-20: the sequencer front's geometry as a record.

MiniTPU states the front's numbers in four places that must agree: the
package (``sequencer_pkg.sv``: ``LEVEL_SEL_W``, ``STACK_DEPTH``, ``LB_CAP``,
``IRAM_MEM_LATENCY``, ``S_LAT``), widths spelled ``$clog2(...)`` inside the
units (``sequencer_loop_ctrl.sv``'s ``SP_W``, the buffer's count and index),
a comment ("costs 2 empty cycles", ``sequencer_fetch_queue.sv:25``; "must
equal sequencer_pkg::LB_CAP", ``sequencer_loop_buffer.sv``) and the bookings
the assembler consumes (``asm.py`` ``S_LAT``/``LB_CAP``/``STACK_DEPTH``,
``docs/isa_latency.json`` ``scalar.latency`` and ``resources.loop_buffer``).

Here the declared numbers are fields and every number that FOLLOWS from them
is a property (the ``legality.py`` pattern), so none can be typed beside the
number it must equal. ``control_legality`` states each relation a unit rests
on, naming the parameter and the consequence; ``check_against_phase0`` holds
the derived numbers to Phase 0's measurements and to MiniTPU's own bookings
(read from ``asm.py`` and ``isa_latency.json`` through ``harness/minitpu_asm``,
never copied); ``refusals`` gives one wrong declaration per relation.

The Allo units read their widths from ``SHIPPED`` (``units/loop_ctrl.py``),
so a geometry change moves them through this record.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass


def clog2(x: int) -> int:
    return max(0, (int(x) - 1).bit_length())


@dataclass(frozen=True)
class ControlGeometry:
    """The sequencer front's declared parameters (``sequencer_pkg.sv`` at
    ``b3ba0a4d``). ``S_LAT`` is a declared *booking* (D-10): the latency the
    scalar AGU is built to and the assembler schedules by."""

    LEVEL_SEL_W: int = 3  # loop-level select (X-slot agu_level, S-slot level)
    INSTR_ADDR_W: int = 12  # IRAM address
    LB_CAP: int = 24  # loop buffer capacity, bundles
    IRAM_MEM_LATENCY: int = 1  # IRAM read register
    FQ_DEPTH: int = 4  # fetch queue entries
    S_LAT: int = 2  # scalar write -> visible, no bypass
    SREG_ADDR_W: int = 2

    # ---- derived: storage geometry ---------------------------------------
    @property
    def STACK_DEPTH(self) -> int:
        """One frame per selectable level: a deeper stack is unaddressable by
        the AGUs, a shallower one lets a level select name a missing frame."""
        return 1 << self.LEVEL_SEL_W

    @property
    def SP_W(self) -> int:
        return clog2(self.STACK_DEPTH + 1)

    @property
    def LB_COUNT_W(self) -> int:
        """Capture count, body length and write pointer: must reach ``LB_CAP``
        (the overflow compare is ``wr_ptr == LB_CAP``)."""
        return clog2(self.LB_CAP + 1)

    @property
    def LB_IDX_W(self) -> int:
        return clog2(self.LB_CAP)

    @property
    def IRAM_ROWS(self) -> int:
        return 1 << self.INSTR_ADDR_W

    @property
    def NSREG(self) -> int:
        return 1 << self.SREG_ADDR_W

    # ---- derived: timing the assembler and the loop buffer rely on --------
    @property
    def fetch_to_valid(self) -> int:
        """Fetch request -> ``bundle_valid``: the IRAM read, then the push."""
        return self.IRAM_MEM_LATENCY + 1

    @property
    def flush_to_target(self) -> int:
        """Flush in cycle e -> the target at the head in e + this: the
        dropped landing response, the IRAM read, the push."""
        return self.IRAM_MEM_LATENCY + 2

    @property
    def branch_bubbles(self) -> int:
        """Empty cycles between a taken branch's bundle and its target."""
        return self.flush_to_target - 1

    @property
    def refetch_cost(self) -> int:
        """Cycles a body longer than ``LB_CAP`` pays per iteration over a
        replayed one: its loop.end branches every pass."""
        return self.branch_bubbles

    @property
    def lbegin_r_skip_latency(self) -> int:
        """``isa_latency.json`` C row: a skipped ``loop.begin.r`` resumes this
        many cycles after it issued (the branch is combinational)."""
        return self.flush_to_target

    def namespace(self) -> dict:
        return {"LEVEL_SEL_W": self.LEVEL_SEL_W, "STACK_DEPTH": self.STACK_DEPTH, "SP_W": self.SP_W,
                "LB_CAP": self.LB_CAP, "LB_COUNT_W": self.LB_COUNT_W, "LB_IDX_W": self.LB_IDX_W,
                "BUFFER_CAP": self.LB_CAP, "IRAM_MEM_LATENCY": self.IRAM_MEM_LATENCY,
                "FQ_DEPTH": self.FQ_DEPTH, "FLUSH_TO_TARGET": self.flush_to_target,
                "REFETCH_COST": self.refetch_cost, "S_LAT": self.S_LAT, "ASM_S_LAT": self.S_LAT,
                "INSTR_ADDR_W": self.INSTR_ADDR_W}


SHIPPED = ControlGeometry()


def control_legality(p):
    """What the front's units elaborate, comment or assume, as one condition set."""
    assert p["STACK_DEPTH"] == 1 << p["LEVEL_SEL_W"], (
        f"STACK_DEPTH={p['STACK_DEPTH']} must be 2**LEVEL_SEL_W = {1 << p['LEVEL_SEL_W']}: "
        f"agu_level and s.level are LEVEL_SEL_W bits, so a frame past 2**LEVEL_SEL_W is "
        f"unaddressable and a level below it names nothing (sequencer_pkg.sv:61)")
    assert p["SP_W"] == clog2(p["STACK_DEPTH"] + 1), (
        f"SP_W={p['SP_W']} must be clog2(STACK_DEPTH+1) = {clog2(p['STACK_DEPTH'] + 1)}: the "
        f"stack pointer must hold STACK_DEPTH itself (the overflow compare sp == STACK_DEPTH)")
    assert p["LB_COUNT_W"] == clog2(p["LB_CAP"] + 1), (
        f"LB_COUNT_W={p['LB_COUNT_W']} must be clog2(LB_CAP+1) = {clog2(p['LB_CAP'] + 1)}: "
        f"wr_ptr == LB_CAP is the capture-overflow test; a narrower count wraps first "
        f"and a body over the capacity replays a partial capture silently")
    assert (1 << p["LB_IDX_W"]) >= p["LB_CAP"], (
        f"LB_IDX_W={p['LB_IDX_W']}: 2**LB_IDX_W = {1 << p['LB_IDX_W']} < LB_CAP={p['LB_CAP']}, "
        f"so the replay index cannot reach the buffer's last entries")
    assert p["BUFFER_CAP"] == p["LB_CAP"], (
        f"the loop buffer's CAP={p['BUFFER_CAP']} must equal LB_CAP={p['LB_CAP']} "
        f"(sequencer_loop_buffer.sv: 'must equal sequencer_pkg::LB_CAP, which "
        f"sequencer_loop_ctrl uses'): the controller counts to one, the buffer stores the other")
    d = p["FQ_DEPTH"]
    assert d >= 2 and d & (d - 1) == 0, (
        f"FQ_DEPTH={d}: the push index is IDX_W'(count) (sequencer_fetch_queue.sv), "
        f"a truncation that is only the count for a power of two")
    assert d - 1 > p["IRAM_MEM_LATENCY"], (
        f"FQ_DEPTH={d}: one slot is kept for the read in flight; with "
        f"IRAM_MEM_LATENCY={p['IRAM_MEM_LATENCY']} straight-line code cannot issue every cycle")
    assert p["FLUSH_TO_TARGET"] == p["IRAM_MEM_LATENCY"] + 2, (
        f"FLUSH_TO_TARGET={p['FLUSH_TO_TARGET']} must be IRAM_MEM_LATENCY + 2 = "
        f"{p['IRAM_MEM_LATENCY'] + 2}: the landing response is dropped, then the read, then the push")
    assert p["REFETCH_COST"] == p["FLUSH_TO_TARGET"] - 1, (
        f"REFETCH_COST={p['REFETCH_COST']} (isa_latency.json resources.loop_buffer) must be the "
        f"branch's bubbles, FLUSH_TO_TARGET - 1 = {p['FLUSH_TO_TARGET'] - 1}: a body over "
        f"LB_CAP branches on every loop.end")
    assert p["S_LAT"] >= 1, f"S_LAT={p['S_LAT']}: a write is visible at the earliest one edge later"
    assert p["ASM_S_LAT"] == p["S_LAT"], (
        f"the assembler schedules S reads {p['ASM_S_LAT']} bundles after the write, the scalar "
        f"AGU is built to S_LAT={p['S_LAT']}: a read between the two sees the old value, "
        f"with no bypass and no interlock to catch it")


def _minitpu_bookings():
    """MiniTPU's own copies: asm.py's constants and isa_latency.json."""
    from examples.minitpu.harness import minitpu_asm, rtl

    asm = minitpu_asm.load()
    lat = json.load(open(os.path.join(rtl.minitpu_home(), "docs", "isa_latency.json")))
    return {"asm.S_LAT": asm.S_LAT, "asm.LB_CAP": asm.LB_CAP, "asm.STACK_DEPTH": asm.STACK_DEPTH,
            "isa_latency.scalar.latency": lat["scalar"]["latency"],
            "isa_latency.resources.loop_buffer.capacity_bundles":
                lat["resources"]["loop_buffer"]["capacity_bundles"]}


def check_against_phase0(g=SHIPPED):
    """Every derived number against Phase 0's measurement (``u4_phase0``
    latency table) and MiniTPU's bookings. Raises on any difference."""
    b = _minitpu_bookings()
    control_legality(g.namespace() | {"ASM_S_LAT": b["asm.S_LAT"]})
    rows = [("fetch request -> bundle_valid", g.fetch_to_valid, 2, "Phase 0 probe"),
            ("flush -> target at head", g.flush_to_target, 3, "Phase 0 probe"),
            ("refetch cost per iteration over LB_CAP", g.refetch_cost, 2, "Phase 0 probe (25 vs 24)"),
            ("loop.begin.r skip -> next issue", g.lbegin_r_skip_latency, 3, "Phase 0 sequencer probe"),
            ("S_LAT (scalar write -> read)", g.S_LAT, 2, "Phase 0 probe"),
            ("S_LAT vs asm.py", g.S_LAT, b["asm.S_LAT"], "asm.py S_LAT"),
            ("S_LAT vs isa_latency.json", g.S_LAT, b["isa_latency.scalar.latency"], "scalar.latency"),
            ("STACK_DEPTH vs asm.py", g.STACK_DEPTH, b["asm.STACK_DEPTH"], "asm.py STACK_DEPTH"),
            ("LB_CAP vs asm.py", g.LB_CAP, b["asm.LB_CAP"], "asm.py LB_CAP"),
            ("LB_CAP vs isa_latency.json", g.LB_CAP,
             b["isa_latency.resources.loop_buffer.capacity_bundles"], "resources.loop_buffer"),
            ("SP_W (loop depth port)", g.SP_W, 4, "u4_loop_ctrl depth width"),
            ("LB_IDX_W (replay index port)", g.LB_IDX_W, 5, "u4_loop_ctrl lb_replay_idx width")]
    bad = [r for r in rows if r[1] != r[2]]
    assert not bad, f"derived != measured: {bad}"
    return rows


def refusals():
    """One wrong declaration per relation, each refused by ``control_legality``."""
    ns = SHIPPED.namespace()
    out = []
    for label, wrong in (
        ("STACK_DEPTH typed as 16 beside LEVEL_SEL_W=3", ns | {"STACK_DEPTH": 16}),
        ("loop buffer count 4 bits for LB_CAP=24", ns | {"LB_COUNT_W": 4}),
        ("replay index 4 bits for LB_CAP=24", ns | {"LB_IDX_W": 4}),
        ("buffer CAP 32 beside LB_CAP=24", ns | {"BUFFER_CAP": 32}),
        ("fetch queue depth 3", ns | {"FQ_DEPTH": 3}),
        ("refetch cost still booked 2 after IRAM latency 2",
         ControlGeometry(IRAM_MEM_LATENCY=2).namespace() | {"REFETCH_COST": 2}),
        ("S_LAT built 3, assembler still schedules 2", ns | {"S_LAT": 3}),
    ):
        try:
            control_legality(wrong)
            out.append((label, "ACCEPTED (bug)"))
        except AssertionError as e:
            out.append((label, "refused: " + str(e).splitlines()[0][:100]))
    return out


def main():
    for label, derived, measured, src in check_against_phase0():
        print(f"DERIVED-OK  {label:42s} {derived:>4} == {measured:<4} ({src})")
    for label, verdict in refusals():
        print(f"{'REFUSED ' if verdict.startswith('refused') else 'ACCEPTED'}  {label}: {verdict}")


if __name__ == "__main__":
    main()
