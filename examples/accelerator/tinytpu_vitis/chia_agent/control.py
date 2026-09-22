# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The control a candidate is compared against: MEASURED per run, not looked up.

A win is "fewer cycles than the unmodified design". That comparison is only
worth something if both numbers come from the same machine, the same Vitis
install and the same build, so the control is measured in the run that uses it
-- `accept.py`, from git at its `--ref`, before the candidate's diff is applied
-- and this module holds what surrounds that measurement:

* `blobs(ref)` -- the identity of the design the control was measured at, which
  goes into the evidence so any accepted result can be re-derived against the
  same control;
* `unusable(record, blobs)` -- what a control record must satisfy before a
  claim may rest on it: the same design, five shapes, a nonce-vouched verdict,
  a met clock. A control that fails this is refused, never silently used;
* `crosscheck(cycles, blobs)` -- the measured control against the published
  numbers. The recorded numbers are a CROSS-CHECK, never the control itself: a
  disagreement means the tools moved or the design changed behaviour, and both
  have to be known before any result from that run is trusted.

The recorded numbers used to BE the control, keyed by the blob ids of the two
editable files, and a prose-only edit to `microarch_isa.py` then moved the key
and reported `no-baseline` -- a control that invalidates itself on a comment
change, and stops the search dead when it does. As a cross-check the same table
costs nothing and cannot block: an unrecorded blob falls back to `PUBLISHED`,
so the check never goes quiet.
"""

from __future__ import annotations

import subprocess
import time

from evaluate import ALL_SHAPES, EDITABLE, PKG, REPO, TARGET_NS

#: The design's published five-shape cosim cycles (docs/source/designs/
#: tinytpu_isa.rst; measured by a no-diff control run, evidence/
#: accept-control-476a70d8/). The fallback cross-check for any design whose
#: blobs are not in `RECORDED` -- a prose-only edit lands here.
PUBLISHED = dict(zip(ALL_SHAPES, (172, 262, 418, 484, 686)))
#: Controls measured by earlier no-diff runs of `accept.py`, keyed by the blob
#: ids of (microarch_isa.py, isa_dsl.py). Cross-check only. A design whose
#: cycles deliberately move gets an entry here in the same commit, so that the
#: cross-check on the next run is against the new intent rather than the old.
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


def record(*, cycles, blobs, ref, estimated_ns, seconds, vouched, source,
           pristine_tree) -> dict:
    """One control measurement, with everything needed to re-derive it."""
    return {"cycles": dict(cycles), "blobs": dict(blobs), "ref": ref,
            "vouched": bool(vouched), "estimated_ns": estimated_ns,
            "seconds": seconds, "source": source,
            "pristine_tree": bool(pristine_tree),
            "measured_unix": int(time.time()),
            "measurement": "cosim: Vitis HLS 2023.2 + xsim C/RTL cosim (RTL), "
                           "all five SHAPES, TPU_* unset but TPU_PRJ"}


def unusable(rec, expected_blobs: dict) -> list[str]:
    """Why a claim may not rest on this control record; empty means it may."""
    if not isinstance(rec, dict):
        return ["not a control record"]
    problems = []
    if not rec.get("vouched"):
        problems.append("its cosim verdict is not nonce-vouched")
    if not rec.get("pristine_tree"):
        problems.append("it was not measured on a pristine checkout")
    if rec.get("blobs") != expected_blobs:
        problems.append(f"measured at {rec.get('blobs')}, not at this run's "
                        f"design {expected_blobs}")
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


def crosscheck(cycles: dict, blobs: dict) -> dict:
    """The measured control against the recorded numbers for that design.

    Shapes the caller did not measure are not compared -- the search measures
    two of the five -- but a shape with no recorded number is a disagreement,
    not something to pass over."""
    recorded = RECORDED.get(tuple(blobs[name] for name in EDITABLE))
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
