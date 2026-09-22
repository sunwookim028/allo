# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The objective: cycles scored, resources constrained. FROZEN.

Evidence, not taste. The design-level loop's one verified win bought 8.6% fewer
cycles at 16x16x16 for **2.3x the block RAM** (BRAM18K 42 -> 98), +20% LUT and
+40% FF. A loop scored on cycles alone makes that trade every time it can and
calls each one progress. So the objective here has two terms.

Of the two shapes the owner offered, this module implements the FIRST --
resources as a hard constraint, cycles as the score -- and also reports the
pair, because the report has to show them side by side anyway. Why the
constraint rather than a weighted sum:

1. **No exchange rate has to be invented.** A weighted score needs a price for
   a BRAM in cycles. Without place-and-route there is no defensible number, and
   a made-up one silently decides every trade-off in the search.
2. **It is a gate, and gates compose.** A resource overrun becomes one more
   verdict in the ladder -- `gate:resources` -- computed from the same
   `csynth.xml` the score comes from, at no extra cost. CAKE's structure is
   cheap gates, then an execution gate, then a report; a constraint fits that,
   a weight does not.
3. **It is legible in a paper.** "Fewer cycles at no more than +10% of any
   resource, on every design case" is a claim a reader can check. "Score
   0.83" is not.
4. **It cannot be gamed by aggregation.** A weighted sum over design cases lets
   a big win on one case pay for a regression on another. Here every case must
   independently stay inside its resource budget and must not lose cycles.

The rule, per design case:

    correctness   bit-exact, every gate passed. Not negotiable; no budget.
    resources     every csynth resource (BRAM_18K, DSP, FF, LUT, URAM) must
                  stay within max(1.10 x baseline, baseline + FLOOR) and the
                  estimated clock must still meet the target. Outside that,
                  the candidate is REJECTED at `gate:resources`, whatever it
                  did to cycles.
    cycles        the score, per case. A candidate must not lose cycles on any
                  case and must gain on at least one.

`FLOOR` exists because a relative bound is meaningless at small counts:
`blocks_stream` synthesises to a handful of BRAMs, where 4 -> 5 is +25% and
means nothing. The floor is stated in absolute units per resource and is part
of the frozen objective, not a knob the loop tunes.

A candidate that improves cycles and exceeds the resource budget is not a
failure of the loop -- it is a TRADE, and `classify()` labels it as one so it
is reported as a trade rather than a win. The design-level run's accepted diff
would be classified `trade` by this module, which is the point.
"""

from __future__ import annotations

#: Resources are read from csynth's AreaEstimates. Every one is constrained:
#: the design-level win moved BRAM most, but FF and LUT moved too.
RESOURCES = ("bram_18k", "dsp", "ff", "lut", "uram")

#: Relative budget against the baseline, per resource, per design case.
RESOURCE_BUDGET = 1.10
#: Absolute slack, so a relative bound is not meaningless at small counts.
#: In csynth units (BRAM18K blocks, DSP slices, flip-flops, LUTs).
FLOOR = {"bram_18k": 2, "dsp": 2, "ff": 250, "lut": 250, "uram": 1}


def limit(resource: str, base: int) -> int:
    return int(max(base * RESOURCE_BUDGET, base + FLOOR.get(resource, 0)))


def resource_verdict(base_area: dict, now_area: dict) -> dict:
    """Per-resource budget check for one design case."""
    rows, over = {}, []
    for r in RESOURCES:
        b, n = base_area.get(r), now_area.get(r)
        if b is None or n is None:
            rows[r] = {"baseline": b, "now": n, "limit": None, "ok": True,
                       "note": "not reported by csynth"}
            continue
        lim = limit(r, b)
        ok = n <= lim
        rows[r] = {"baseline": b, "now": n, "limit": lim, "delta": n - b,
                   "ratio": round(n / b, 3) if b else None, "ok": ok}
        if not ok:
            over.append(f"{r} {b} -> {n} (limit {lim})")
    return {"per_resource": rows, "over_budget": over, "ok": not over}


def case_verdict(base: dict, now: dict) -> dict:
    """One design case: (cycles delta, resource verdict, classification).

    `base` and `now` are each {"cycles": {shape: int} | {"latency": int},
    "area": {...}, "estimated_ns": float}. `cycles` may be per-shape (the RTL
    cosim cases) or a single csynth latency; both are summed.
    """
    def total(d):
        c = d.get("cycles")
        if isinstance(c, dict):
            return sum(v for v in c.values() if isinstance(v, int))
        return c

    b_cyc, n_cyc = total(base), total(now)
    res = resource_verdict(base.get("area") or {}, now.get("area") or {})
    out = {
        "cycles_baseline": b_cyc, "cycles_now": n_cyc,
        "cycles_delta": (None if b_cyc is None or n_cyc is None
                         else n_cyc - b_cyc),
        "cycles_per_shape": {
            s: {"baseline": (base.get("cycles") or {}).get(s),
                "now": v,
                "delta": (None if not isinstance((base.get("cycles") or {}).get(s), int)
                          else v - base["cycles"][s])}
            for s, v in (now.get("cycles") or {}).items()
        } if isinstance(now.get("cycles"), dict) else {},
        "resources": res,
        "estimated_ns": {"baseline": base.get("estimated_ns"),
                         "now": now.get("estimated_ns")},
    }
    d = out["cycles_delta"]
    out["cycles_verdict"] = ("unknown" if d is None else
                            "better" if d < 0 else "same" if d == 0 else "worse")
    return out


def classify(cases: dict) -> dict:
    """The candidate's verdict across every design case.

    win     no case lost cycles, no case left its resource budget, and at
            least one case gained cycles.
    trade   at least one case gained cycles AND at least one case left its
            resource budget or lost cycles. Reported as a trade, never a win.
    neutral nothing moved.
    worse   no case gained.
    """
    gained = [c for c, v in cases.items() if v["cycles_verdict"] == "better"]
    lost = [c for c, v in cases.items() if v["cycles_verdict"] == "worse"]
    over = [c for c, v in cases.items() if not v["resources"]["ok"]]
    if gained and not lost and not over:
        verdict = "win"
    elif gained:
        verdict = "trade"
    elif lost or over:
        verdict = "worse"
    else:
        verdict = "neutral"
    return {
        "verdict": verdict, "gained": sorted(gained), "lost": sorted(lost),
        "over_budget": sorted(over),
        #: The one-line statement of the result. Per case, never aggregated:
        #: an abstraction that helps one design and hurts another is a
        #: finding, not a number.
        "summary": "; ".join(
            f"{c}: {v['cycles_baseline']} -> {v['cycles_now']} cycles "
            f"({v['cycles_delta']:+d})" if isinstance(v["cycles_delta"], int)
            else f"{c}: cycles unknown"
            for c, v in sorted(cases.items())),
        "resource_summary": "; ".join(
            f"{c}: " + ", ".join(
                f"{r} {row['baseline']}->{row['now']}"
                for r, row in sorted(v["resources"]["per_resource"].items())
                if row.get("delta"))
            for c, v in sorted(cases.items())
            if any(row.get("delta") for row in
                   v["resources"]["per_resource"].values())) or "no resource change",
    }


#: What the loop uses to decide keep-or-rewind. `trade` is NOT kept: a trade
#: needs a person to price it.
KEEP = ("win",)
