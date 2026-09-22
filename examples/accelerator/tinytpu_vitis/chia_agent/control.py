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

import re
import subprocess
import time

from evaluate import ALL_SHAPES, EDITABLE, PKG, REPO, SCORED, TARGET_NS


def reproduced() -> dict:
    """The published five-shape row, read from reproduce.sh's EXPECTED, which
    reproduce.sh checks on every run -- restated elsewhere, it went stale."""
    text = (REPO / PKG / "reproduce.sh").read_text()
    row = re.search(r'^EXPECTED="([^"]+)"', text, re.M).group(1)
    return {s: int(c) for s, c in (kv.split("=") for kv in row.split())}


#: The published five-shape cycles per measuring driver: the cross-check for a
#: design whose blobs are not in `RECORDED`, where a prose-only edit lands.
#: Keyed by driver because a second driver runs a different PROGRAM on the same
#: hardware, so its numbers are not this one's measured differently.
PUBLISHED = {"cosim": reproduced()}
#: Controls measured by earlier no-diff runs: (driver, microarch_isa.py blob,
#: isa_dsl.py blob) -> cycles. Cross-check only; a design whose cycles
#: deliberately move gets its entry in the same commit.
RECORDED = {
    # main @ e2451b81 (the branch point before the rebase)
    ("cosim", "ac5174fe43f449e9b0b1693cda1aff6c74ab71d3",
     "10de511a2ddf7a8fa8fbf8d0de588ddbb690290f"):
        dict(zip(ALL_SHAPES, (252, 383, 591, 667, 919))),
    # main @ e620576d (check_program in assemble(), docstring fixes)
    ("cosim", "cb26d5683338184f02bfcb6be13bc1ace4e5e3e9",
     "e3b55230b4c6308dfa5e7d729d49e6056040d663"):
        dict(zip(ALL_SHAPES, (252, 383, 591, 667, 919))),
    # main @ 476a70d8 (e24e433b: wld double-buffer, program prefetch, accu at
    # II=1 via s.dependence). The published numbers until the memory sizing
    # (05169938) took one cycle off every shape.
    ("cosim", "98b20b8b3f9ecf289604a428ffdb28997964b9dd",
     "8f2e9aa9f518ef320cab163adc95e05737c777be"):
        dict(zip(ALL_SHAPES, (172, 262, 418, 484, 686))),
    # codesign-loop @ 9f375cc6, whose mapper picks the nest: measured through
    # `codesign_cosim` (cosim 272 s, every testbench bit-exact, csynth 2.431
    # ns), blobs read from git at that commit. Four shapes are the published
    # numbers because there the mapper's pick IS the canonical nest, word for
    # word; 4x4x4 is 169 because its pick emits 24 instruction words against
    # the hand-written 28, with the same four dynamic issues.
    ("codesign_cosim", "98b20b8b3f9ecf289604a428ffdb28997964b9dd",
     "a29a8fbd253d7b3e847be09257739c370d7c0c5d"):
        dict(zip(ALL_SHAPES, (169, 262, 418, 484, 686))),
}


def blobs(ref: str) -> dict:
    """The two editable files' git blob ids at `ref` -- the design's identity."""
    return {name: git_blob(ref, name) for name in EDITABLE}


def git_blob(ref: str, name: str) -> str | None:
    p = subprocess.run(["git", "rev-parse", f"{ref}:{PKG}/{name}"], cwd=REPO,
                       capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else None


def record(*, cycles, design, ref, estimated_ns, seconds, vouched, source,
           pristine_tree, driver) -> dict:
    """One control measurement, with everything needed to re-derive it."""
    return {"cycles": dict(cycles), "blobs": dict(design), "ref": ref,
            "vouched": bool(vouched), "estimated_ns": estimated_ns,
            "seconds": seconds, "source": source, "driver": driver,
            "pristine_tree": bool(pristine_tree),
            "measured_unix": int(time.time()),
            "measurement": f"{driver}: Vitis HLS 2023.2 + xsim C/RTL cosim "
                           f"(RTL), all five SHAPES, TPU_* unset but {SCORED} and TPU_PRJ"}


def unusable(rec, design: dict, driver: str) -> list[str]:
    """Why a claim may not rest on this control record; empty means it may.

    `driver` is the check that measures the candidate: a control measured by
    a different one is a comparison between two machines described by two
    programs, which is how a driver change looks like a win."""
    if not isinstance(rec, dict):
        return ["not a control record"]
    problems = []
    if rec.get("driver") != driver:
        problems.append(f"measured by {rec.get('driver')!r}, while the "
                        f"candidate is measured by {driver!r}")
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


def crosscheck(cycles: dict, design: dict, driver: str) -> dict:
    """The measured control against the numbers recorded for that design under
    that driver.

    Shapes the caller did not measure are not compared -- the search measures
    two of the five -- but a shape with no recorded number is a disagreement,
    not something to pass over."""
    recorded = RECORDED.get((driver, *(design[name] for name in EDITABLE)))
    against = (f"the control recorded for this design under {driver!r}"
               if recorded else
               f"the published {driver!r} control (this design's blobs are "
               f"not recorded)")
    recorded = recorded or PUBLISHED.get(driver)
    if not recorded:
        return {"status": "UNRECORDED", "measured": dict(cycles),
                "against": f"nothing: no control is recorded or published for "
                           f"the driver {driver!r}",
                "recorded": None, "delta": {}, "compared": [],
                "driver": driver}
    recorded = dict(recorded)
    delta = {s: cycles[s] - recorded[s] for s in cycles if s in recorded}
    agrees = (bool(delta) and not any(delta.values())
              and not set(cycles) - set(recorded))
    return {"status": "agree" if agrees else "DISAGREES", "against": against,
            "recorded": recorded, "measured": dict(cycles), "delta": delta,
            "compared": sorted(delta)}


def banner(cc: dict) -> str:
    """Loud, because an uncross-checked control invalidates the whole run."""
    if not cc or cc.get("status") == "agree":
        return ""
    if cc["status"] == "UNRECORDED":
        why = ["! THE MEASURED CONTROL IS UNRECORDED: NOTHING CROSS-CHECKS IT",
               "!   " + cc["against"],
               f"!   measured {cc['measured']}",
               "! Put it in control.RECORDED, so that the next run of this",
               "! driver is compared with a number a person has looked at."]
    else:
        why = ["! THE MEASURED CONTROL DISAGREES WITH " + cc["against"].upper(),
               f"!   recorded {cc['recorded']}",
               f"!   measured {cc['measured']}",
               f"!   delta    {cc['delta']}",
               "! Either the toolchain moved or the design changed behaviour. No",
               "! result from this run is trustworthy until a person has said",
               "! which, and recorded the new numbers in control.RECORDED."]
    return "\n".join(["!" * 78, *why, "!" * 78])
