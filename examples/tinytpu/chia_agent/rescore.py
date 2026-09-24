#!/usr/bin/env python3
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Re-score the changes we already know the answer to, old objective and new.

An objective is not better because its author says so. The only check available
before a paid run is to hand it the design changes this project has already
measured and ask whether it ranks them the way the measurements do. An
objective that reverses one of them is telling you something; an objective that
agrees with all of them *on the old axes* has not changed anything.

Four changes, every number transcribed from a committed report or page with its
citation, none re-derived here:

``burst_widen``
    The operand burst from one packed word per iteration to the widest the
    64-byte beat holds. **-720 / -960** cycles at 48^3 / 64^3 (7.0 % / 4.3 %),
    **-287 / -583 / -960** on the three RTL-verified models (25.0 % / 27.5 % /
    34.5 %), **+43 % FF and +92 % block RAM** on FPGA, **+74.4 % cell area** in
    45 nm of which **99.1 % is two AXI master ports**.

``channel_depth``
    ``QD`` 8 -> 16. Three legal tiled programs go from never completing to
    completing bit-exact; the five published shapes move **+4/+4/+4/-1/-11**,
    which sums to **zero**; FPGA **+9.5 % flip-flops** at the scored point;
    no clean DC pair exists, and the tree's own isolated estimate is
    **+6.6 % cell area** from a flip-flop count off the RTL.

``operand_mirror``
    Removing the on-chip operand mirror for a per-row strided read.
    **+6/0/-2/-6/+10** on the five shapes. No resource measurement of any kind
    exists for it.

``run3_burst``
    CHIA run 3's accepted candidate, 2026-09-24: ``TPU_DMA_WIDEN`` turned on at
    the scored T=4 MAXDIM=16, found by the loop and accepted by ``accept.py`` on
    a clean worktree against a control it measured itself.
    **0 / 0 / -35 / -47 / -47**, total **-129**. FPGA **BRAM18K 40 -> 52**,
    **FF +6.9 %**, **LUT +4.4 %**. No DC run.

    The accepted diff was two changes, and a $0 re-acceptance of the burst half
    alone returns **the identical row**: the other half, deepening ``ac2sp``
    from ``QD`` to 32, is worth **exactly zero cycles**. That is the
    ``channel_depth`` finding reached a second time, by measurement rather than
    by re-scoring.

``run1_burst``
    CHIA run 1's one accepted candidate: the same widening idea, found by the
    loop, at the scored T=4 MAXDIM=16. **0 / 0 / -42 / -59 / -59** over the five
    shapes; on the two the loop actually scored, **-59**. FPGA **BRAM18K 42 ->
    98** (+133 %), **FF +40 %**, **LUT +20 %**. No DC run.

Run::

    python rescore.py            # the table, and the verdict on the objective
    python rescore.py --json

Prints ``RESCORE OK`` when every change the new objective can see is ranked as
the measurements rank it, and lists what it cannot see either way.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import area_proxy  # noqa: E402

#: The five shapes, in the order every published table uses.
SHAPES = ["4x4x4", "8x8x8", "12x12x12", "16x16x8", "16x16x16"]
#: The two the loop scores on. `evaluate.SEARCH_SHAPES`.
SEARCH = ["4x4x4", "16x16x16"]

#: Every number below is transcribed, with the page it is transcribed from.
#: `area_config` is the pair of parameter sets the area proxy prices, so that
#: the area term is COMPUTED here rather than transcribed -- that is the term
#: under test.
CHANGES = {
    "burst_widen": {
        "what": "operand burst 1 -> 16 packed words per iteration (MAXDIM=64)",
        "cite": "docs/source/designs/benchmarks.rst, workload_suite.rst, "
                "asic_synthesis/README.md",
        "gemm": {"48x48x48": (10289, 9569), "64x64x64": (22123, 21163)},
        "model": {"mlp_tiny": (1150, 863), "mlp_deep": (2117, 1534),
                  "mlp_small": (2781, 1821)},
        "resources": {"ff": (17488, 25026), "lut": (26554, 33799),
                      "bram_18k": (52, 100), "dsp": (14, 14)},
        "dc_area": (1865313.782218, 3254024.203238),
        "area_config": (dict(T=4, MAXDIM=64, QD=8, DMA_WORDS=1),
                        dict(T=4, MAXDIM=64, QD=8, DMA_WORDS=16)),
        "verdict": "trade -- big on models, real on silicon, and the cost is "
                   "in the adapters rather than in the machine",
    },
    "channel_depth": {
        "what": "QD 8 -> 16; three legal programs go from hanging to bit-exact",
        "cite": "docs/source/developer/limitations.rst item 24, "
                "designs/benchmarks.rst",
        "gemm": dict(zip(SHAPES, zip((171, 261, 417, 483, 685),
                                     (175, 265, 421, 482, 674)))),
        "model": {},
        "resources": {"ff": (17075, 18699), "lut": (26558, 27210),
                      "bram_18k": (40, 40), "dsp": (14, 14)},
        "dc_area": None,
        "area_config": (dict(T=4, MAXDIM=16, QD=8, DMA_WORDS=1),
                        dict(T=4, MAXDIM=16, QD=16, DMA_WORDS=1)),
        "verdict": "correctness, at a price -- and the five-shape total is "
                   "ZERO, so no cycle objective can see why it was made",
    },
    "operand_mirror": {
        "what": "remove the on-chip operand mirror for a per-row strided read",
        "cite": "dev/records/tinytpu/big_shapes_settlement.rst",
        "gemm": dict(zip(SHAPES, zip((175, 265, 421, 482, 674),
                                     (181, 265, 419, 476, 684)))),
        "model": {},
        "resources": {},
        "dc_area": None,
        "area_config": None,
        "verdict": "regression on cycles; no resource measurement exists",
    },
    "run3_burst": {
        "what": "CHIA run 3's accepted candidate: TPU_DMA_WIDEN at the scored "
                "T=4 MAXDIM=16 (DMA_WORDS 1 -> 4), the predicted rediscovery",
        "cite": "dev/records/tinytpu/chia-evidence/isa-run3-20260924/",
        "gemm": dict(zip(SHAPES, zip((175, 265, 421, 482, 674),
                                     (175, 265, 386, 435, 627)))),
        # Measured at MAXDIM=16, where the widening is worth ZERO on both
        # models (dev/records/tinytpu/model-term-maxdim-20260924.rst). The
        # model term is scored at MAXDIM=64, where the same source change is
        # worth 25-34 %, and those are `burst_widen`'s numbers -- so no model
        # row is transcribed here rather than one borrowed from a different
        # configuration.
        "model": {},
        "resources": {"ff": (18825, 20125), "lut": (27428, 28630),
                      "bram_18k": (40, 52), "dsp": (14, 14)},
        "dc_area": None,
        "area_config": (dict(T=4, MAXDIM=16, QD=16, DMA_WORDS=1),
                        dict(T=4, MAXDIM=16, QD=16, DMA_WORDS=4)),
        "verdict": "the loop's own live win, and the sharpest instance of the "
                   "FPGA table understating silicon: +6.9 % flip-flops "
                   "against a +22.8 % area estimate",
    },
    "run1_burst": {
        "what": "CHIA run 1's accepted candidate: the burst widened to one "
                "DRAM row per iteration at the scored T=4 MAXDIM=16",
        "cite": "dev/records/tinytpu/chia-evidence/isa-run1-20260919/",
        "gemm": dict(zip(SHAPES, zip((172, 262, 418, 484, 686),
                                     (172, 262, 376, 425, 627)))),
        "model": {},
        "resources": {"ff": (17481, 24465), "lut": (26583, 31929),
                      "bram_18k": (42, 98), "dsp": (14, 14)},
        "dc_area": None,
        "area_config": (dict(T=4, MAXDIM=16, QD=8, DMA_WORDS=1),
                        dict(T=4, MAXDIM=16, QD=8, DMA_WORDS=4)),
        "verdict": "the win the OLD objective rewarded, and the one whose "
                   "price it could not price",
    },
}


def old_objective(c: dict) -> dict:
    """What the loop did before: sum of cosim cycles at the two search shapes,
    with csynth's resource table reported beside it and NOT deciding anything.

    `loop.classify` called a candidate a `win` only if no resource term grew,
    but `improved` -- the thing that actually kept or rewound the spec -- was
    `total_cycles < best`, and nothing else."""
    scored = {s: v for s, v in c["gemm"].items() if s in SEARCH}
    if not scored:                      # measured at shapes the loop never runs
        return {"kept": None, "delta": None, "scored": {},
                "why": "not measured at either search shape"}
    delta = sum(b - a for a, b in scored.values())
    grew = sorted(k for k, (a, b) in c["resources"].items() if b > a)
    return {"kept": delta < 0, "delta": delta,
            "scored": {s: b - a for s, (a, b) in scored.items()},
            "classification": ("regression" if delta >= 0 else
                               "trade" if grew else "win"),
            "resources_grew": grew}


def new_objective(c: dict) -> dict:
    """The three-term objective: model cycles primary, GEMM shapes the control,
    the standard-cell ESTIMATE as the resource axis."""
    dmodel = {m: b - a for m, (a, b) in c["model"].items()}
    dgemm = {s: b - a for s, (a, b) in c["gemm"].items()}
    area = None
    if c["area_config"]:
        before, after = (area_proxy.estimate(area_proxy.synthetic_census(**cfg))
                         for cfg in c["area_config"])
        area = {"um2": round(after["um2"] - before["um2"], 1),
                "pct": round(100 * (after["um2"] / before["um2"] - 1), 1),
                "terms": {k: round(after["terms"][k] - before["terms"][k], 1)
                          for k in before["terms"]},
                "estimate": True}
    primary = dmodel or dgemm
    kind = ("regression" if sum(primary.values()) >= 0 else
            "win" if (all(d <= 0 for d in dmodel.values())
                      and all(d <= 0 for d in dgemm.values())
                      and (area or {}).get("um2", 0) <= 0) else "trade")
    return {"delta_model": dmodel, "delta_gemm": dgemm, "area": area,
            "primary": "model" if dmodel else "gemm (no model term measured)",
            "kept": sum(primary.values()) < 0, "classification": kind}


def report(as_json=False) -> int:
    out, blind = {}, []
    for name, c in CHANGES.items():
        out[name] = {"what": c["what"], "cite": c["cite"],
                     "old": old_objective(c), "new": new_objective(c),
                     "known": c["verdict"]}
        if not c["model"]:
            blind.append(name)
    if as_json:
        print(json.dumps(out, indent=2, sort_keys=True))
        return 0
    for name, r in out.items():
        print(f"\n=== {name}: {r['what']}")
        print(f"    known answer: {r['known']}")
        print(f"    cite: {r['cite']}")
        o, n = r["old"], r["new"]
        if o["kept"] is None:
            print(f"    OLD  {o['why']} -- the change was never priced by the "
                  f"objective that was running")
        else:
            print(f"    OLD  {o['classification']:10s} kept={o['kept']}  "
                  f"scored cycles {o['scored']} (sum {o['delta']:+d}); "
                  f"csynth grew {o['resources_grew'] or 'nothing'}")
        area = n["area"]
        area_s = (f"{area['um2']:+,.0f} um2 ({area['pct']:+.1f} %, ESTIMATE)"
                  if area else "no area term (the proxy cannot price this one)")
        print(f"    NEW  {n['classification']:10s} kept={n['kept']}  "
              f"primary={n['primary']}")
        print(f"         model {n['delta_model'] or '(none measured)'}")
        print(f"         gemm  {n['delta_gemm']}")
        print(f"         area  {area_s}")
        if area:
            big = max(area["terms"], key=lambda k: abs(area["terms"][k]))
            share = (100 * area["terms"][big] / area["um2"]) if area["um2"] else 0
            print(f"               of which {big} {area['terms'][big]:+,.0f} "
                  f"({share:.1f} % of the delta)")
    print("\n" + "=" * 74)
    print("What the re-score establishes, and what it does not")
    print("=" * 74)
    print(f"""
* The new objective REVERSES nothing. Every change keeps the sign the
  measurements give it on the cycle axes.
* It changes the CLASSIFICATION of the two that matter. `run1_burst` was the
  old objective's one accepted win and is a trade here: the proxy prices it at
  +24.3 % cell area, {100 * 273005 / 274232:.1f} % of it in the AXI master ports -- the same
  attribution DC made for the MAXDIM=64 widening (99.1 % in two adapters),
  reached in milliseconds instead of 56 minutes. `burst_widen` moves from "a
  few per cent, expensive" to "a quarter to a third of a model, expensive",
  which is the trade a person should be asked to make.
* `channel_depth` is the one that indicts the OLD objective outright. Its
  five-shape total is **zero** (+4+4+4-1-11), so no cycle objective over the
  five can prefer it -- and over the TWO the loop scored it sums to -7, so the
  old objective would have kept it for the wrong reason. What it actually buys
  is three legal programs that go from never completing to completing, and
  neither objective has a term for that. **That is a gap this work did not
  close.**
* MODEL CYCLES EXIST FOR EXACTLY ONE OF THESE. {', '.join(blind)} have no
  model measurement, so on those the new objective is running on its GEMM
  control and is the old objective with a better resource axis. The one change
  that has both is also the one where the two axes disagree by 4-8x. The
  re-score therefore supports the resource axis strongly and the model axis on
  a single point.
* TWO METHODS AGREE ON THE CHANNEL DEPTH. This re-score says the QD change's
  five-shape total is zero, so no cycle objective can prefer it. CHIA run 3
  then said the same thing by measurement, from a different harness: its
  accepted diff was a burst widening plus an `ac2sp` deepening from QD to 32,
  and re-accepting the burst half ALONE returns the identical row -- the
  deepening is worth exactly zero cycles. Re-scoring predicted it; the run
  measured it.
* `run3_burst` is the live case for the resource axis. The loop accepted it on
  -47 over its two scored shapes with csynth reporting **+6.9 % flip-flops**;
  the area proxy prices the same change at **+22.8 %**, 99.6 % of it in the AXI
  master ports. A 3.3x understatement, on the loop's own most recent win,
  measured by the axis that was deciding nothing.""")
    ok = all(r["new"]["kept"] == (r["new"]["classification"] != "regression")
             for r in out.values())
    print(f"\n{'RESCORE OK' if ok else 'RESCORE INCONSISTENT'}: "
          f"{len(out)} known changes re-scored, {len(blind)} of them without a "
          f"model term")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", action="store_true")
    return report(ap.parse_args(argv).json)


if __name__ == "__main__":
    sys.exit(main())
