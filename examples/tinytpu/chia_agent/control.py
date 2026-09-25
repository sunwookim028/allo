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
    """The published five-shape row, read from `reproduce.sh`'s EXPECTED -- the
    one place a gate checks it on every run.

    Restated here as a literal it went stale: it still said 172 / 262 / 418 /
    484 / 686 after the design shipped 175 / 265 / 421 / 482 / 674, so the
    cross-check would have printed a banner and exited 3 on a correct design.
    A pin nobody notices going stale is the defect, not its value.
    """
    text = (REPO / PKG / "reproduce.sh").read_text()
    row = re.search(r'^EXPECTED="([^"]+)"', text, re.M).group(1)
    return {s: int(c) for s, c in (kv.split("=") for kv in row.split())}


#: The published five-shape cycles per measuring driver: the cross-check for a
#: design whose blobs are not in `RECORDED`, where a prose-only edit lands.
#: Keyed by driver because a second driver runs a different PROGRAM on the same
#: hardware, so its numbers are not this one's measured differently.
PUBLISHED = {"cosim": reproduced()}
#: Controls measured by earlier no-diff runs: `key(driver, blobs)` -> cycles.
#: Cross-check only; a design whose cycles deliberately move gets its entry in
#: the same commit. The pre-decomposition entries are two-file designs, which
#: is what the design WAS at those commits (`design.EDITABLE` is seventeen
#: paths: fourteen since the unit library landed, and the three ISA files since
#: the instruction set became editable), so they key on the two blobs they had.


def _two(driver, micro, dsl):
    return (driver, (("isa_dsl.py", dsl), ("microarch_isa.py", micro)))


RECORDED = {
    # main @ e2451b81 (the branch point before the rebase)
    _two("cosim", "ac5174fe43f449e9b0b1693cda1aff6c74ab71d3",
         "10de511a2ddf7a8fa8fbf8d0de588ddbb690290f"):
        dict(zip(ALL_SHAPES, (252, 383, 591, 667, 919))),
    # main @ e620576d (check_program in assemble(), docstring fixes)
    _two("cosim", "cb26d5683338184f02bfcb6be13bc1ace4e5e3e9",
         "e3b55230b4c6308dfa5e7d729d49e6056040d663"):
        dict(zip(ALL_SHAPES, (252, 383, 591, 667, 919))),
    # main @ 476a70d8 (e24e433b: wld double-buffer, program prefetch, accu at
    # II=1 via s.dependence). The published numbers until the memory sizing
    # took one cycle off two shapes and added four to three others.
    _two("cosim", "98b20b8b3f9ecf289604a428ffdb28997964b9dd",
         "8f2e9aa9f518ef320cab163adc95e05737c777be"):
        dict(zip(ALL_SHAPES, (172, 262, 418, 484, 686))),
    # codesign-loop @ 9f375cc6, whose mapper picks the nest: measured through
    # `codesign_cosim` (cosim 272 s, every testbench bit-exact, csynth 2.431
    # ns), blobs read from git at that commit. Four shapes are the published
    # numbers because there the mapper's pick IS the canonical nest, word for
    # word; 4x4x4 is 169 because its pick emits 24 instruction words against
    # the hand-written 28, with the same four dynamic issues.
    _two("codesign_cosim", "98b20b8b3f9ecf289604a428ffdb28997964b9dd",
         "a29a8fbd253d7b3e847be09257739c370d7c0c5d"):
        dict(zip(ALL_SHAPES, (169, 262, 418, 484, 686))),
}


def blobs(ref: str) -> dict:
    """The editable files' git blob ids at `ref` -- the design's identity."""
    return {rel: git_blob(ref, rel) for rel in EDITABLE}


def key(driver: str, design: dict) -> tuple:
    """A `RECORDED` key: the driver and every editable file's blob, as a sorted
    pair list. The design is a package, so "the two blobs" is no longer it."""
    return (driver, tuple(sorted(design.items())))


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
    recorded = RECORDED.get(key(driver, design))
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


#: Files that quote the published five-shape row as a literal AND are meant to
#: track it. A pin here must equal `reproduced()`; `check_pins` fails loudly
#: otherwise. This is the same failure the frozen-file lists had twice: a value
#: copied into several places with nothing comparing the copies to the source.
#: `swarm.py` is absent because it no longer holds a literal at all -- it
#: derives the row from here, which is the fix this list is only the backstop
#: for.
PINNED = (f"{PKG}/reproduce.sh",
          f"{PKG}/chia_agent/swarm.py", f"{PKG}/chia_agent/test_harness.py",
          f"{PKG}/chia_agent/test_codesign.py")
#: Literals fitted to a SUPERSEDED row that a find-and-replace would FALSIFY:
#: they are the outputs of a fit or a calibration, not quotations of the row, so
#: they need re-measuring rather than editing. Named here so the gate reports
#: them instead of passing over them in silence.
#:
#: Every entry it held on 2026-09-25 was fitted to 172 / 262 / 418 / 484 / 686
#: and has been re-derived against the shipped row;
#: `dev/records/tinytpu/refit-20260925.rst` records each derivation and its new
#: residual. What was cleared then:
#:
#:   `act/cycles.py`             PUBLISHED_CYCLES is now DERIVED from
#:                               reproduce.sh, so it is no longer a pin at all;
#:                               CRITICAL_WORK_FIT re-fitted to (179.0, 1.573)
#:                               and checked against `refit()` in __main__.
#:   `act_machine.py`            CALIBRATION's cosim column re-measured (the
#:                               five gemm rows from the published row,
#:                               gemm.relu 16x16x16 by fresh cosim: 738);
#:                               RANKING_EVIDENCE re-measured (175 vs 172).
#:   `tests/act/test_tinytpu.py` the residual bound re-derived from the refitted
#:                               fit: 40 -> 35 against a worst of 34.21.
#:   `reproduce_codesign.sh`     its control re-measured: 172 / 674, and the
#:                               mapper's pick still equals the shipped nest at
#:                               four of five shapes (checked in python).
#:
#: Add an entry the moment a new constant is fitted to a measurement, with what
#: it would take to re-measure it; the `__main__` below reports whatever is
#: here, and says so when nothing is.
NEEDS_REFIT = {
    "docs/source/extensions/act_results.rst":
        "the out-of-sample table under `act-specs-gate-measured` (twelve specs: "
        "critical work, estimate, cosim, error) is a TPU_QD=8 corpus sweep of "
        "the design at e24e433b. Its cosim column is that design's and its "
        "estimate column the (173.2, 1.621) fit, so the 9.2% in-sample / 13.9% "
        "out-of-sample / 6.0% mean claims are statistics of that PAIRING. "
        "Recomputing the estimate column under the shipped (179.0, 1.573) fit "
        "while leaving the old cosim column would make the error column a "
        "comparison between two designs, which is why it was left alone rather "
        "than edited. It needs `act/calibrate.py specs` and `act/calibrate.py "
        "variants` re-run at TPU_T=4 TPU_MAXDIM=16 -- one csynth plus twelve "
        "cosims, one of which (relu_16x16) is expected not to complete "
        "(limitations item 24). The page carries the same warning.",
}
#: A five-shape row that NAMES its shapes: `4x4x4=175, 8x8x8=265, ...`.
#: Deliberately not the bare `175 / 265 / ...` form, which cannot be told
#: apart from Gemmini's row, the mapper's row, or a sentence about what the
#: numbers used to be -- all three of which are legitimately not the published
#: row, and all three of which a looser pattern flagged. A shape-keyed row is
#: always a claim about this design, so matching it has no false positives.
_ROW_EQ = re.compile(r"4x4x4=(\d+)\D+8x8x8=(\d+)\D+12x12x12=(\d+)"
                     r"\D+16x16x8=(\d+)\D+16x16x16=(\d+)")
#: The same row as a dict literal, which is how `PUBLISHED_CYCLES` went stale.
_ROW_DICT = re.compile(r'"4x4x4":\s*(\d+)\D+"8x8x8":\s*(\d+)\D+'
                       r'"12x12x12":\s*(\d+)\D+"16x16x8":\s*(\d+)\D+'
                       r'"16x16x16":\s*(\d+)')


#: A row that is deliberately NOT the published one -- the co-design mapper's
#: pick, a comparison machine, a sentence about history -- says so on its own
#: first line or the ONE line above it. Deliberately that tight: at a six-line
#: lookback an exemption leaked onto the next unrelated row down, and a probe
#: row with the stale numbers went through the gate unflagged.
EXEMPT = "not-the-published-row"


def check_pins() -> list[str]:
    """Every tracked pin of the published row, against `reproduce.sh`."""
    want = tuple(reproduced()[s] for s in ALL_SHAPES)
    problems = []
    for rel in PINNED:
        path = REPO / rel
        if not path.is_file():
            problems.append(f"{rel}: PINNED names a file that does not exist")
            continue
        text = path.read_text()
        # The dict form is often written over several lines.
        lines = text.splitlines()
        for m in _ROW_DICT.finditer(text):
            if tuple(int(g) for g in m.groups()) != want:
                n = text[: m.start()].count("\n") + 1
                if any(EXEMPT in l for l in lines[max(0, n - 2): n]):
                    continue
                problems.append(
                    f"{rel}:{n} has a dict of {' / '.join(m.groups())}, "
                    f"published is {' / '.join(str(c) for c in want)}")
        for n, line in enumerate(lines, 1):
            m = _ROW_EQ.search(line)
            if m and tuple(int(g) for g in m.groups()) != want:
                if any(EXEMPT in l for l in lines[max(0, n - 2): n]):
                    continue
                problems.append(
                    f"{rel}:{n} quotes {' / '.join(m.groups())}, "
                    f"published is {' / '.join(str(c) for c in want)}")
    return problems


if __name__ == "__main__":
    import sys
    bad = check_pins()
    print(f"published row (from {PKG}/reproduce.sh): "
          + " ".join(f"{s}={c}" for s, c in reproduced().items()))
    for rel, what in sorted(NEEDS_REFIT.items()):
        print(f"  ACKNOWLEDGED, fitted to the superseded row: {rel} -- {what}")
    if not NEEDS_REFIT:
        print("  NEEDS_REFIT is empty: no constant in the tree is known to be "
              "fitted to a superseded row (see this module's NEEDS_REFIT "
              "comment for what was cleared, and add an entry when a new "
              "fitted constant lands)")
    for p in bad:
        print(f"  STALE PIN: {p}")
    print("PINS OK: every tracked quotation of the published row matches it"
          if not bad else f"PINS STALE: {len(bad)}")
    sys.exit(1 if bad else 0)
