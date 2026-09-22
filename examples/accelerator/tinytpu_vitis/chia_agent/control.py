# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The control a candidate is compared against: MEASURED per run, not looked up.

`accept.py` takes the measurement (guard 10, docs/source/extensions/chia.rst);
this module is what surrounds it. `blobs` names the design a control was
measured at, `record` is the evidence an accepted result can be re-derived
from, `unusable` is what a claim may not rest on, and `crosscheck` holds the
published numbers to their new job: a cross-check on the measurement, never
the control itself.
"""

from __future__ import annotations

import subprocess
import time

from evaluate import ALL_SHAPES, EDITABLE, PKG, REPO, TARGET_NS

#: The design's published five-shape cycles (docs/source/designs/
#: tinytpu_isa.rst, evidence/accept-control-476a70d8/): the cross-check for a
#: design whose blobs are not in `RECORDED`, where a prose-only edit lands.
PUBLISHED = dict(zip(ALL_SHAPES, (172, 262, 418, 484, 686)))
#: Controls measured by earlier no-diff runs, keyed by the blob ids of
#: (microarch_isa.py, isa_dsl.py). Cross-check only; a design whose cycles
#: deliberately move gets its entry in the same commit.
RECORDED = {
    # main @ e2451b81 (the branch point before the rebase)
    ("ac5174fe43f449e9b0b1693cda1aff6c74ab71d3",
     "10de511a2ddf7a8fa8fbf8d0de588ddbb690290f"):
        dict(zip(ALL_SHAPES, (252, 383, 591, 667, 919))),
    # main @ e620576d (check_program in assemble(), docstring fixes)
    ("cb26d5683338184f02bfcb6be13bc1ace4e5e3e9",
     "e3b55230b4c6308dfa5e7d729d49e6056040d663"):
        dict(zip(ALL_SHAPES, (252, 383, 591, 667, 919))),
    # main @ 476a70d8 (e24e433b: wld double-buffer, program prefetch, accu at
    # II=1 via s.dependence). The published numbers.
    ("98b20b8b3f9ecf289604a428ffdb28997964b9dd",
     "8f2e9aa9f518ef320cab163adc95e05737c777be"): dict(PUBLISHED),
}


def blobs(ref: str) -> dict:
    """The two editable files' git blob ids at `ref` -- the design's identity."""
    return {name: git_blob(ref, name) for name in EDITABLE}


def git_blob(ref: str, name: str) -> str | None:
    p = subprocess.run(["git", "rev-parse", f"{ref}:{PKG}/{name}"], cwd=REPO,
                       capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else None


def record(*, cycles, design, ref, estimated_ns, seconds, vouched, source,
           pristine_tree) -> dict:
    """One control measurement, with everything needed to re-derive it."""
    return {"cycles": dict(cycles), "blobs": dict(design), "ref": ref,
            "vouched": bool(vouched), "estimated_ns": estimated_ns,
            "seconds": seconds, "source": source,
            "pristine_tree": bool(pristine_tree),
            "measured_unix": int(time.time()),
            "measurement": "cosim: Vitis HLS 2023.2 + xsim C/RTL cosim (RTL), "
                           "all five SHAPES, TPU_* unset but TPU_PRJ"}


def unusable(rec, design: dict) -> list[str]:
    """Why a claim may not rest on this control record; empty means it may."""
    if not isinstance(rec, dict):
        return ["not a control record"]
    problems = []
    if not rec.get("vouched"):
        problems.append("its cosim verdict is not nonce-vouched")
    if not rec.get("pristine_tree"):
        problems.append("it was not measured on a pristine checkout")
    if rec.get("blobs") != design:
        problems.append(f"measured at {rec.get('blobs')}, not at this run's "
                        f"design {design}")
    cycles = rec.get("cycles")
    if not isinstance(cycles, dict):
        problems.append("no cycles")
    else:
        bad = [s for s in ALL_SHAPES if not isinstance(cycles.get(s), int)]
        if bad:
            problems.append(f"no measured cycles at {bad}")
        extra = sorted(set(cycles) - set(ALL_SHAPES))
        if extra:
            problems.append(f"shapes outside the frozen five: {extra}")
    est = rec.get("estimated_ns")
    if not isinstance(est, (int, float)) or est > TARGET_NS:
        problems.append(f"estimated clock {est} ns does not meet {TARGET_NS} ns")
    return problems


def crosscheck(cycles: dict, design: dict) -> dict:
    """The measured control against the recorded numbers for that design.

    Shapes the caller did not measure are not compared -- the search measures
    two of the five -- but a shape with no recorded number is a disagreement,
    not something to pass over."""
    recorded = RECORDED.get(tuple(design[name] for name in EDITABLE))
    against = ("the control recorded for this design" if recorded else
               "the published five-shape control (this design's blobs are "
               "not recorded)")
    recorded = dict(recorded or PUBLISHED)
    delta = {s: cycles[s] - recorded[s] for s in cycles if s in recorded}
    agrees = (bool(delta) and not any(delta.values())
              and not set(cycles) - set(recorded))
    return {"status": "agree" if agrees else "DISAGREES", "against": against,
            "recorded": recorded, "measured": dict(cycles), "delta": delta,
            "compared": sorted(delta)}


def banner(cc: dict) -> str:
    """Loud, because a disagreement invalidates every result from the run."""
    if cc.get("status") != "DISAGREES":
        return ""
    return "\n".join([
        "!" * 78,
        "! THE MEASURED CONTROL DISAGREES WITH " + cc["against"].upper(),
        f"!   recorded {cc['recorded']}",
        f"!   measured {cc['measured']}",
        f"!   delta    {cc['delta']}",
        "! Either the toolchain moved or the design changed behaviour. No",
        "! result from this run is trustworthy until a person has said which,",
        "! and recorded the new numbers in control.RECORDED.",
        "!" * 78])
