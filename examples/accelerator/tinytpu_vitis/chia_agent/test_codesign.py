# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM-free end-to-end test of the CO-DESIGN loop. Run it before any paid run.

`test_harness.py` covers the design-level loop: the tool surface over MCP, the
`swarm -> loop -> opencode` path against a scripted model, the frozen-file and
import-time attacks, the spend accounting. None of that changes here and none of
it is repeated. What this file tests is the part the co-design loop adds, which
is the part that can silently produce a wrong claim:

  k1  a no-op                      reproduces the baseline EXACTLY -- the nest
                                   counts, the whole refusal histogram, the
                                   chosen nest, the cycles and the resources
  k2  a hardware change that       the encodable-nest count RISES, and the
      widens the ISA (AGU_TERMS    refusal it relieved goes to zero while the
      3 -> 4, the AGU word         next constraint appears. A fixture, not a
      repacked)                    recommendation: the loop's agent is free to
                                   attack a different refusal
  k3  the encoder widened by       the mapspace gate REJECTS it: the nests it
      deleting a legality check    newly calls encodable compute the wrong
                                   thing against isa_ref
  k4  the seam broken (the         REJECTED at gate:mapspace -- the canonical
      canonical nest no longer     nest must re-emit the candidate's own
      re-emits gemm_program)       gemm_program word for word
  k5  a broken datapath           REJECTED at gate:stress, as in the design
      (PE partial sum int16)       loop; the co-design stages never run
  k6  a program that outgrows      REJECTED, and the refusal names IMEM_SIZE.
      IMEM_SIZE                    Nothing is truncated: `assemble` raises, and
                                   the mapper counts the nest unencodable
  k8  the same check, PRESENT but   REJECTED all the same: `git grep` still hits
      made vacuous by an            the check, a text-anchored guard still sees
      unreachable conjunct          it, and only the isa_ref sweep catches it
  k7  the frozen half attacked     a sabotaged mapper in the spec directory is
                                   IGNORED (the evaluator composes it from
                                   git), and accept.py refuses a diff that
                                   touches the mapper, a test or a golden

Every case costs $0 and calls no model. k1 runs a real Vitis cosim (~3 min);
everything else is gate-only (~25 s each) or pure python.

    conda activate allo
    export LLVM_BUILD_DIR=... OMP_NUM_THREADS=8
    python test_codesign.py                 # everything, ~7 min
    python test_codesign.py --phases k2,k3,k4,k6,k7    # no Vitis, ~2 min

Writes `<run-dir>/results.json` and exits non-zero if any case failed.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent
REPO = AGENT_DIR.parents[3]
PKG = "examples/accelerator/tinytpu_vitis"
sys.path.insert(0, str(AGENT_DIR))
from design import EDITABLE  # noqa: E402

ALLO_PYTHON = os.environ.setdefault(
    "TINYTPU_ALLO_PYTHON", "/home/sk3463/miniconda3/envs/allo/bin/python")
os.environ.setdefault("LLVM_BUILD_DIR",
                      "/home/sk3463/llvm-allo-6b09f739/build")
os.environ.setdefault("OMP_NUM_THREADS", "8")

#: The published design-level baseline: what `cosim.py` measures on
#: `isa_dsl.gemm_program`, read from `control.PUBLISHED` (which reads
#: reproduce.sh) rather than restated, because restated it went stale. A loud
#: cross-check, NOT the co-design control: the co-design control is measured
#: in the same run (k1 below, and `loop.py`'s iteration 0).
import control  # noqa: E402

PUBLISHED_CYCLES = dict(control.PUBLISHED["cosim"])
#: The CO-DESIGN baseline: the same hardware, running the best nest the frozen
#: mapper can encode on it, measured through `codesign_cosim` and published
#: per driver in `control.py`. It is not the design-level number at every
#: shape, and the difference is understood:
#:
#:   16x16x16  the published number. The mapper's pick IS the canonical nest,
#:             so the program under the RTL is `gemm_program` word for word.
#:   4x4x4     three cycles fewer. At this shape N/T = K/T = 1, so the
#:             enumerator offers a nest with no emitted loops at all, while
#:             the hand-written program keeps a trip-count-1 `loop`/`endloop`
#:             pair around the output body. Same 4 dynamic issues, two fewer
#:             static instructions (24 words against 28). It is a real
#:             mapping-side find, it is bit-exact, and it is the whole of what
#:             the mapping search alone buys on this hardware.
#:
#: 8x8x8, 12x12x12 and 16x16x8 are the published numbers because there the
#: mapper's pick (`N2>K2 rows=8`, `N3>K3 rows=12`, `N2>K4 rows=16`) is
#: bit-identical to the canonical nest.
CODESIGN_CONTROL_ALL = dict(control.PUBLISHED["codesign_cosim"])
BASELINE_CYCLES = {s: CODESIGN_CONTROL_ALL[s] for s in ("4x4x4", "16x16x16")}
#: The whole refusal histogram at the two scored shapes on the shipped design.
#: Pinned, not just spot-checked: the point of the loop is that these numbers
#: move for a stated reason, so an unexplained drift is a failure.
BASELINE_MAPSPACE = {
    "4x4x4": {"total": 4, "encodable": 1, "chosen": "- rows=4",
              "refused": {"accumulator-raw-distance": 2, "agu-terms": 1}},
    "16x16x16": {"total": 1226, "encodable": 3, "chosen": "N4>K4 rows=16",
                 "refused": {"acc-peel": 1150, "emitter": 54, "agu-terms": 17,
                             "accumulator-raw-distance": 2}},
}
#: k2's expectation. AGU_TERMS=4 removes the 17 `agu-terms` refusals at
#: 16x16x16 and 1 at 4x4x4; what appears instead is the NEXT constraint, which
#: is the whole point of reporting a histogram rather than a count.
AGU4_MAPSPACE = {
    "4x4x4": {"encodable": 1, "refused": {"accumulator-raw-distance": 3}},
    "16x16x16": {"encodable": 7,
                 "refused": {"acc-peel": 1150, "emitter": 54,
                             "accumulator-raw-distance": 9, "loop-depth": 6}},
}

M_ISA, I_DSL = "microarch_isa.py", "isa_dsl.py"


def head(name: str) -> str:
    return subprocess.run(["git", "show", f"HEAD:{PKG}/{name}"], cwd=REPO,
                          capture_output=True, text=True, check=True).stdout


def replace(text: str, old: str, new: str, tag: str) -> str:
    assert text.count(old) == 1, f"{tag}: anchor occurs {text.count(old)} times"
    return text.replace(old, new, 1)


# -- the mutants -------------------------------------------------------------
#: k2's edit, and the record's variants, have ONE definition: `histogram.py`,
#: which is what the record on this page was measured with. Each edit names
#: the file it applies to, because the design is a package.
from histogram import AGU4  # noqa: E402

#: k3. The encoder's `acc-peel` position check deleted, so K-outer nests are
#: called encodable although the emitted program overwrites the accumulator on
#: every k-tile instead of only the first. This is the cheap way to "unlock
#: 1,150 nests", and it must fail.
LOOSEN_ACC_PEEL = (
    I_DSL,
    """    if ks and ks[0] != len(emitted) - 1:
        raise Unencodable(
            f"acc-peel: the emitted order is "
            f"{'>'.join(l.rank for l in emitted)}, but the k=0 tile must be a "
            f"peelable prefix, which needs K innermost")
""",
    "")

#: k4. The prologue order of `gemm_from_nest` alone, so `gemm_program` still
#: matches `gemm_program_handwritten` (bench_isa passes) but the canonical nest
#: no longer re-emits `gemm_program`. Exactly the drift the seam check is for.
BREAK_SEAM = (
    I_DSL,
    """        stage_a([])                              # the shipped prologue order
        stage_b()
""",
    """        stage_b()
        stage_a([])
""")

#: k8. The insidious version of k3: the `acc-peel` position check is still
#: THERE, verbatim -- `raise Unencodable("acc-peel: ...")` and all -- but the
#: guard has grown a conjunct that is never true, so it refuses nothing. A
#: reviewer skimming the diff sees the check; `git grep acc-peel` still hits; a
#: mutation test anchored on the check's text still finds it. Only BEHAVIOUR
#: distinguishes it from k3, which is why the mapspace gate proves every
#: survivor against `isa_ref` rather than trusting that a named check exists.
#: `len(emitted) > LOOP_DEPTH * 4` reads like a depth guard and is unreachable:
#: `emitted` is at most a handful of loops at MAXDIM=16. Arranged so the
#: CANONICAL nest stays correct, which keeps the seam check passing and forces
#: the failure to come from the isa_ref sweep -- that is what separates this
#: case from k4.
VACUOUS_ACC_PEEL = (
    I_DSL,
    "    if ks and ks[0] != len(emitted) - 1:\n",
    "    if ks and ks[0] != len(emitted) - 1 and len(emitted) > LOOP_DEPTH * 4:\n")

#: k5. [-4, 4] operands never overflow 16 bits; full-range ones do. The PE is
#: its own unit since the decomposition.
NARROW16 = ("ip/units/pe.py",
            "psum: int32 = psum_north + activation16 * weight16",
            "psum: int16 = psum_north + activation16 * weight16")

#: k6. IMEM_SIZE = NHDR + IWORDS * _MAX_STATIC, so this is 8 + 24 = 32 words,
#: and the chosen 16x16x16 program needs 34.
SHRINK_IMEM = (
    M_ISA,
    "_MAX_STATIC = 24               # longest program shipped, plus headroom\n",
    "_MAX_STATIC = 12               # longest program shipped, plus headroom\n")


def spec(run: Path, name: str, edits=(), agu4=False) -> Path:
    """A spec directory: HEAD's writable files, with exact edits applied.

    The design is a package (`design.EDITABLE`), so an edit names its file and
    the directory mirrors the tree."""
    out = run / "spec" / name
    out.mkdir(parents=True, exist_ok=True)
    files = {rel: head(rel) for rel in EDITABLE}
    for f, old, new in (list(AGU4) if agu4 else []) + list(edits):
        files[f] = replace(files[f], old, new, f"{name}/{f}")
    for f, text in files.items():
        (out / f).parent.mkdir(parents=True, exist_ok=True)
        (out / f).write_text(text)
    return out


def diff_of(spec_dir: Path) -> str:
    chunks = []
    for f in EDITABLE:
        before = head(f).splitlines(keepends=True)
        after = (spec_dir / f).read_text().splitlines(keepends=True)
        chunks += difflib.unified_diff(before, after, f"a/{f}", f"b/{f}")
    return "".join(chunks)


def evaluate(spec_dir: Path, work: Path, gate_only=True, shapes=None) -> dict:
    cmd = [sys.executable, str(AGENT_DIR / "evaluate.py"),
           "--spec-dir", str(spec_dir), "--work", str(work), "--codesign"]
    if gate_only:
        cmd.append("--gate-only")
    if shapes:
        cmd += ["--shapes", shapes]
    env = {k: v for k, v in os.environ.items() if not k.startswith("TPU_")}
    p = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True,
                       timeout=5400)
    out = (p.stdout + p.stderr).strip()
    for line in reversed(out.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                v = json.loads(line)
                v.setdefault("detail", out[-6000:] if not v.get("ok") else "")
                return v
            except json.JSONDecodeError:
                continue
    return {"ok": False, "stage": "evaluator", "detail": out[-6000:]}


# -- bookkeeping -------------------------------------------------------------
RESULTS: list[dict] = []


def check(case, expected, actual, passed, **extra):
    RESULTS.append({"case": case, "expected": expected, "actual": actual,
                    "passed": bool(passed), **extra})
    print(f"  [{'PASS' if passed else 'FAIL'}] {case}\n"
          f"         expected: {expected}\n         measured: {actual}",
          flush=True)
    return bool(passed)


def mapspace_of(v: dict) -> dict:
    return v.get("mapspace", {}).get("shapes", {})


# -- the cases ---------------------------------------------------------------
def k1(run, work):
    """A no-op reproduces the baseline exactly -- counts, histogram and pair."""
    v = evaluate(spec(run, "noop"), work / "k1", gate_only=False,
                 shapes=",".join(BASELINE_CYCLES))
    ms = mapspace_of(v)
    same = v.get("ok") and v.get("cycles") == BASELINE_CYCLES and all(
        ms.get(s, {}).get("encodable") == e["encodable"]
        and ms.get(s, {}).get("total") == e["total"]
        and ms.get(s, {}).get("chosen") == e["chosen"]
        and ms.get(s, {}).get("refused") == e["refused"]
        for s, e in BASELINE_MAPSPACE.items())
    ok = check("k1 no-op reproduces the co-design baseline",
               f"cycles {BASELINE_CYCLES} (published {PUBLISHED_CYCLES['4x4x4']}"
               f" / {PUBLISHED_CYCLES['16x16x16']}; 4x4x4 differs by the "
               f"trip-count-1 loop the mapper drops) and the pinned histogram",
               {"ok": v.get("ok"), "stage": v.get("stage"),
                "cycles": v.get("cycles"),
                "mapspace": {s: {k: d.get(k) for k in
                                 ("encodable", "total", "chosen", "refused")}
                             for s, d in ms.items()}}, same,
               resources=v.get("objective", {}).get("resources"),
               scored_nest=v.get("scored_nest"))
    if v.get("ok"):
        # The pair, for the record. Not asserted: area is a property of the
        # Vitis version, and this case is about reproducibility of the loop.
        print(f"         resources: {v['objective']['resources']}")
    return ok


def k2(run, work):
    """A hardware change that widens the ISA shows the nest count rise."""
    v = evaluate(spec(run, "agu4", agu4=True), work / "k2")
    ms = mapspace_of(v)
    rose = v.get("ok") and all(
        ms.get(s, {}).get("encodable") == e["encodable"]
        and ms.get(s, {}).get("refused") == e["refused"]
        for s, e in AGU4_MAPSPACE.items())
    return check("k2 AGU_TERMS 3->4 unlocks nests",
                 f"gate passes; encodable "
                 f"{BASELINE_MAPSPACE['16x16x16']['encodable']} -> "
                 f"{AGU4_MAPSPACE['16x16x16']['encodable']} at 16x16x16, "
                 f"agu-terms refusals -> 0, loop-depth appears",
                 {"ok": v.get("ok"), "stage": v.get("stage"),
                  "mapspace": {s: {"encodable": d.get("encodable"),
                                   "chosen": d.get("chosen"),
                                   "refused": d.get("refused")}
                               for s, d in ms.items()}}, rose)


def k3(run, work):
    """Widening the encoder by deleting a legality check is caught."""
    v = evaluate(spec(run, "loosen", [LOOSEN_ACC_PEEL]), work / "k3")
    caught = (not v.get("ok") and v.get("stage") == "gate:mapspace"
              and "WRONG against isa_ref" in v.get("detail", ""))
    return check("k3 encoder widened by deleting the acc-peel check",
                 "REJECTED at gate:mapspace, nests WRONG against isa_ref",
                 {"ok": v.get("ok"), "stage": v.get("stage"),
                  "isa_ref_lines": [l for l in v.get("detail", "").splitlines()
                                    if "WRONG against isa_ref" in l][:3]},
                 caught)


def k4(run, work):
    """The seam: the canonical nest must re-emit the candidate's gemm_program."""
    v = evaluate(spec(run, "seam", [BREAK_SEAM]), work / "k4")
    caught = (not v.get("ok") and v.get("stage") == "gate:mapspace"
              and "seam " in v.get("detail", ""))
    return check("k4 seam broken (nest != gemm_program)",
                 "REJECTED at gate:mapspace, the seam assertion names the "
                 "differing instruction",
                 {"ok": v.get("ok"), "stage": v.get("stage"),
                  "seam": [l.strip() for l in v.get("detail", "").splitlines()
                           if "seam " in l][:2]}, caught)


def k5(run, work):
    """A broken datapath is rejected before the co-design stages run."""
    v = evaluate(spec(run, "narrow16", [NARROW16]), work / "k5")
    caught = (not v.get("ok") and v.get("stage") == "gate:stress"
              and "mapspace" not in v)
    return check("k5 PE partial sum narrowed to int16",
                 "REJECTED at gate:stress; no mapspace report is produced",
                 {"ok": v.get("ok"), "stage": v.get("stage"),
                  "mapspace_reported": "mapspace" in v,
                  "stress": [l.strip() for l in v.get("detail", "").splitlines()
                             if "STRESS" in l][-1:]}, caught)


def k6(run, work):
    """A program that outgrows IMEM_SIZE is rejected, never truncated."""
    v = evaluate(spec(run, "imem", [SHRINK_IMEM]), work / "k6")
    named = "IMEM_SIZE" in v.get("detail", "")
    # ... and at the mapper layer: with imem at 32 words no nest is encodable
    # at 16x16x16, and the refusal is attributed to imem rather than the
    # program being silently shortened.
    code = (
        "import json, os, sys\n"
        f"sys.path.insert(0, {str(AGENT_DIR)!r})\n"
        "import mapspace\n"
        "from examples.accelerator.tinytpu_vitis import microarch_isa as u\n"
        "from examples.accelerator.tinytpu_vitis import isa_dsl as d\n"
        "f = mapspace.search(16, 16, 16)\n"
        "try:\n"
        "    n = len(u.assemble(d.gemm_program(16, 16, 16, False)))\n"
        "    raised = None\n"
        "except AssertionError as e:\n"
        "    n, raised = None, str(e)\n"
        "print(json.dumps({'imem': u.IMEM_SIZE, 'encodable': f['encodable'],\n"
        "                  'refused': f['refused'], 'words': n,\n"
        "                  'raised': raised}))\n")
    env = dict(os.environ, PYTHONPATH=f"{REPO}", TPU_IMEM="32")
    p = subprocess.run([ALLO_PYTHON, "-c", code], cwd=REPO, env=env,
                       capture_output=True, text=True, timeout=600)
    try:
        inner = json.loads(p.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        inner = {"error": (p.stdout + p.stderr)[-800:]}
    clean = (inner.get("encodable") == 0 and "imem" in inner.get("refused", {})
             and inner.get("words") is None
             and "IMEM_SIZE" in (inner.get("raised") or ""))
    return check("k6 the program outgrows IMEM_SIZE",
                 "the candidate is REJECTED and the refusal names IMEM_SIZE; "
                 "assemble RAISES rather than truncating, and the mapper "
                 "attributes the refusal to imem with 0 encodable",
                 {"verdict": {"ok": v.get("ok"), "stage": v.get("stage"),
                              "names_imem": named},
                  "mapper_at_imem_32": inner},
                 (not v.get("ok")) and named and clean)


def k7(run, work):
    """The frozen half: a sabotaged mapper is ignored; accept refuses the diff."""
    ok = True
    # (a) A mapper that reports 1,226 encodable nests, dropped into the spec
    # directory. The evaluator composes only the two editable files from there
    # and takes `mapspace.py` from git, so it has no effect at all.
    sab = spec(run, "sabotage")
    (sab / "mapspace.py").write_text(
        "def search(M, K, N, slots=2):\n"
        "    return {'total': 1226, 'encodable': 1226, 'refused': {},\n"
        "            'ranked': []}\n")
    (sab / "codesign_gate.py").write_text("import sys\nsys.exit(0)\n")
    (sab / "cosim.py").write_text("raise SystemExit(0)\n")
    v = evaluate(sab, work / "k7")
    ms = mapspace_of(v)
    ignored = v.get("ok") and all(
        ms.get(s, {}).get("encodable") == e["encodable"]
        and ms.get(s, {}).get("refused") == e["refused"]
        for s, e in BASELINE_MAPSPACE.items())
    ok &= check("k7a a sabotaged mapper in the spec dir is ignored",
                "the verdict is the baseline's, mapspace.py coming from git",
                {"ok": v.get("ok"), "stage": v.get("stage"),
                 "mapspace": {s: d.get("encodable") for s, d in ms.items()}},
                ignored)
    # (b) accept.py refuses a diff that touches anything but the two files.
    for target, body in (
            (f"chia_agent/mapspace.py", "-DRAM, ARRAY = \"dram\", \"array\"\n"),
            ("cosim.py", "-VITIS = \"/opt/xilinx/Vitis_HLS/2023.2/settings64.sh\"\n"),
            ("stress_isa.py", "-import itertools\n")):
        name = Path(target).name
        patch = (f"--- a/{name}\n+++ b/{name}\n@@ -1,1 +1,1 @@\n"
                 f"{body}+# sabotaged\n")
        d = run / f"bad-{name}.diff"
        d.write_text(patch)
        out = run / f"accept-{name}"
        p = subprocess.run(
            [sys.executable, str(AGENT_DIR / "accept.py"), "--codesign",
             "--diff", str(d), "--out", str(out)],
            cwd=REPO, capture_output=True, text=True, timeout=1200)
        text = p.stdout + p.stderr
        refused = "refusing: diff touches" in text
        ok &= check(f"k7b accept.py refuses a diff touching {name}",
                    "refusing: diff touches [...]",
                    [l for l in text.splitlines() if "refusing" in l][:1]
                    or text[-300:], refused)
    return ok


def k8(run, work):
    """A peel check that is present, reads plausibly, and refuses nothing."""
    v = evaluate(spec(run, "vacuous", [VACUOUS_ACC_PEEL]), work / "k8")
    text = (run / "spec" / "vacuous" / I_DSL).read_text()
    # The check is still in the file, and still says what it always said.
    present = ("raise Unencodable(" in text
               and "acc-peel: the emitted order is" in text)
    caught = (not v.get("ok") and v.get("stage") == "gate:mapspace"
              and "WRONG against isa_ref" in v.get("detail", ""))
    ms = mapspace_of(v)
    return check("k8 the acc-peel check is present but vacuous",
                 "the check is STILL IN THE SOURCE and the candidate is "
                 "REJECTED at gate:mapspace -- detection is behavioural "
                 "(isa_ref), not textual",
                 {"check_still_in_source": present, "ok": v.get("ok"),
                  "stage": v.get("stage"),
                  "encodable_claimed": {s: d.get("encodable")
                                        for s, d in ms.items()},
                  "isa_ref_lines": [l for l in v.get("detail", "").splitlines()
                                    if "WRONG against isa_ref" in l][:3]},
                 present and caught)


PHASES = {"k1": k1, "k2": k2, "k3": k3, "k4": k4, "k5": k5, "k6": k6, "k7": k7,
          "k8": k8}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phases", default=",".join(PHASES))
    ap.add_argument("--run-dir", type=Path, default=REPO / ".chia_scratch"
                    / f"codesign-test-{time.strftime('%Y%m%d-%H%M%S')}")
    a = ap.parse_args()
    run = a.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    work = run / "work"
    started = time.time()
    print(f"co-design harness test, $0, no model. run dir {run}")
    passed = True
    for name in a.phases.split(","):
        name = name.strip()
        if not name:
            continue
        if name not in PHASES:
            print(f"unknown phase {name!r}; one of {sorted(PHASES)}")
            return 2
        print(f"\n=== {name} " + "=" * 60, flush=True)
        try:
            passed &= PHASES[name](run, work)
        except Exception as e:                      # noqa: BLE001
            import traceback
            traceback.print_exc()
            passed &= check(f"{name} (raised)", "no exception",
                            f"{type(e).__name__}: {e}", False)
    n_ok = sum(1 for r in RESULTS if r["passed"])
    (run / "results.json").write_text(json.dumps(
        {"cases": RESULTS, "passed": n_ok, "total": len(RESULTS),
         "minutes": round((time.time() - started) / 60, 1),
         "usd": 0.0, "head": subprocess.run(
             ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
             text=True).stdout.strip()}, indent=1))
    print(f"\n{n_ok}/{len(RESULTS)} cases passed in "
          f"{(time.time() - started) / 60:.1f} min, $0.00")
    print(f"results: {run / 'results.json'}")
    return 0 if passed and n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
