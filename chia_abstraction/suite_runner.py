# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Allo's OWN test suites, per file, against a recorded baseline. FROZEN.

Gate (b) of the ladder: a compiler change must not break Allo. The comparison
is against a MEASURED baseline, not against zero -- `main` does not pass its own
suite. Three failures are known (`test_hierachical_mesh::test_2x2`,
`test_catapult_hls::test_catapult_float`, `test_builder::test_minmax_cast`), and
more files skip or hang than one would like.

Two decisions, both forced by measurement rather than taste:

1. **One subprocess per test FILE, with a timeout.** A single
   `pytest tests/` run hung this host at 11% collected and never finished:
   these tests build MLIR, start OpenMP teams and shell out to Vitis, and any
   of those can deadlock or abort the interpreter. In one process a hang costs
   the whole gate and names no culprit. Per file, a hang is one file's verdict
   (`timeout`), the rest still run, and the candidate is told which file.
   A SIGSEGV or an MLIR `abort()` is likewise contained (`crash`).
2. **The verdict is a per-test outcome map, compared name by name.** Not a
   count. A candidate that fixes one test and breaks another leaves the count
   unchanged; and a count cannot tell a new failure from a test that stopped
   being collected. So the baseline records
   `{"tests/test_x.py::test_y": "passed"|"failed"|"skipped"|...}` plus the
   file-level verdict, and the gate reports
   `regressions` (was passing, now is not -- BLOCKING),
   `disappeared` (was collected, now is not -- BLOCKING: deleting a test is
                  how a test suite is silently weakened),
   `repairs` (was failing, now passes -- reported, never required).

Per-test outcomes come from pytest's `--junitxml`, parsed by THIS module in the
evaluator's own process afterwards -- the XML is written into the work directory
by pytest, and the file-level rc is what the gate runner vouches for. A
candidate that rewrote the XML would still have to produce a matching rc from
an unmodified pytest, and the work directory is the only writable path in the
sandbox, so the XML is treated as a detail report and the BLOCKING comparison
is recomputed from it only after the rc says the file passed. A file whose rc
is non-zero is a failure whatever its XML says.

Tiers, so that the cheap gate stays cheap (CAKE: static and cheap gates first):

    fast    the files that exercise the schedule surface, the frontend and the
            HLS emitters, and finish in seconds to a couple of minutes each.
            This is what runs inside the search loop.
    full    every collectable file, including the slow dataflow simulations.
            Acceptance only.

    python suite_runner.py --tier fast --out DIR          # run and print JSON
    python suite_runner.py --tier fast --record FILE      # write a baseline
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

#: Files whose content decides whether a compiler change is safe, and which are
#: cheap enough to run on every candidate. Chosen by what they cover:
#: the schedule primitives, the type system, the frontend, and the two HLS
#: emitters a `maintaining` candidate is most likely to touch.
FAST = [
    "tests/test_vhls.py",            # the Vivado emitter: pragmas, dependence,
                                     # names, bit slices, while/break lowering
    "tests/test_schedule_memory.py",
    "tests/test_schedule_compute.py",
    "tests/test_schedule_structure.py",
    "tests/test_schedule_stream.py",
    "tests/test_types.py",
    "tests/test_bitop.py",
    "tests/test_memory.py",
    "tests/test_builder.py",
    "tests/test_stateful_hls.py",
    "tests/test_stateful.py",
    "tests/test_template.py",
    "tests/test_unify.py",
    "tests/test_verify.py",
    "tests/test_traceback.py",
    "tests/test_backend_utils.py",
    "tests/dataflow/test_stream_ops_ir.py",
    "tests/dataflow/test_stream_ops_hls.py",
    "tests/dataflow/test_region_stateful.py",
    "tests/dataflow/test_df_unit.py",
    "tests/dataflow/test_stream_of_blocks.py",
]
#: Everything else that is collectable on this host. `tests/pytorch` and
#: `tests/dataflow/aie` need torch, `tests/autoscheduler` needs gurobipy and a
#: Vitis platform, `tests/test_pynq.py` needs a board: all excluded by
#: EXCLUDE, not by being slow.
EXCLUDE = (
    "tests/pytorch/",
    "tests/dataflow/aie/",
    "tests/autoscheduler/",
    "tests/test_pynq.py",
    "tests/test_xls.py",
    #: A second copy of tests/test_backend_utils.py; identical basename, and
    #: pytest refuses to collect both. The one at tests/ is in FAST.
    "tests/utils/",
    #: Not tests: helper scripts and limitation repros. tests/limits/ is gate
    #: (c)'s own check (limits_runner.py), not pytest's.
    "tests/limits/",
    "tests/dataflow/hls_synth_",
    "tests/dataflow/mesh_perf.py",
    "tests/dataflow/sparse/",
)

#: Per file. Measured: the slowest FAST file is tests/test_vhls.py at ~100 s
#: with Vitis on PATH; the slowest full file is a dataflow simulation.
TIMEOUT = {"fast": 600, "full": 1200}


def files(tier: str) -> list[str]:
    if tier == "fast":
        return [f for f in FAST if (REPO / f).is_file()]
    out = []
    for p in sorted((REPO / "tests").rglob("test_*.py")):
        rel = str(p.relative_to(REPO))
        if any(rel.startswith(e) or e in rel for e in EXCLUDE):
            continue
        out.append(rel)
    return out


def _junit(path: Path, rel: str) -> dict:
    """{'<file>::<test>': outcome} from pytest's junitxml.

    The key is built from `rel` -- the file this runner was asked to run -- not
    from the XML's own `file` attribute, which `--import-mode=importlib` leaves
    empty, and which is written by the process under test in any case.
    """
    if not path.exists():
        return {}
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return {}
    out = {}
    for case in root.iter("testcase"):
        cls = (case.get("classname") or "").rsplit(".", 1)
        suffix = f"{cls[-1]}::" if len(cls) > 1 and cls[-1] else ""
        name = case.get("name") or "?"
        key = f"{rel}::{suffix}{name}"
        outcome = "passed"
        for child in case:
            if child.tag in ("failure", "error"):
                outcome = "failed"
            elif child.tag == "skipped":
                outcome = "skipped"
        out[key] = outcome
    return out


def run_file(rel: str, work: Path, timeout: int, env: dict) -> dict:
    xml = work / (rel.replace("/", "__") + ".xml")
    # cwd is a WRITABLE scratch directory, and the test path is absolute.
    # Several tests write relative paths (`df.build`'s default project is
    # `top.prj` in the cwd, and `test_mlp.py` writes weight files), and under
    # the evaluation sandbox the checkout is read-only: with cwd=<slot> eleven
    # tests failed on an UNMODIFIED tree, which would have made every
    # comparison meaningless. `--rootdir` keeps conftest discovery anchored to
    # the checkout.
    cwd = work / "cwd"
    cwd.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "pytest", str(REPO / rel), "-q", "--no-header",
           "-p", "no:cacheprovider", "--import-mode=importlib",
           f"--rootdir={REPO}", f"--junitxml={xml}"]
    t = time.time()
    # errors="replace": an MLIR abort or a Vitis log can put non-UTF-8 bytes on
    # this pipe, and a decode error here would take down the whole gate.
    p = subprocess.Popen(cmd, cwd=cwd, env=env, text=True, encoding="utf-8",
                         errors="replace", stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True)
    try:
        out, _ = p.communicate(timeout=timeout)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        out, _ = p.communicate()
        out, rc = (out or "") + f"\nTIMEOUT after {timeout}s", 124
    sec = round(time.time() - t, 1)
    # pytest: 0 ok, 1 tests failed, 2 interrupted, 4 usage, 5 no tests.
    verdict = {0: "ok", 1: "failed", 5: "empty", 124: "timeout"}.get(rc)
    if verdict is None:
        verdict = "crash" if rc < 0 or rc > 5 else "error"
    return {"file": rel, "rc": rc, "verdict": verdict, "seconds": sec,
            "tests": _junit(xml, rel), "tail": (out or "")[-1500:]}


def run(argv) -> tuple[int, dict]:
    """Called by abs_gate_runner.py inside the vouched process.

    Returns (rc, report). rc is 0 when every file RAN -- not when every test
    passed: whether the outcomes are a regression is decided by `compare()`
    against the baseline, outside this process. rc is non-zero only if the
    runner itself could not do its job.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="fast", choices=("fast", "full"))
    ap.add_argument("--work", required=True)
    a = ap.parse_args(argv)
    work = Path(a.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    todo = files(a.tier)
    report = {"tier": a.tier, "files": [], "seconds": 0.0}
    t0 = time.time()
    for rel in todo:
        r = run_file(rel, work, TIMEOUT[a.tier], env)
        print(f"  {r['verdict']:>8}  {rel}  ({r['seconds']}s, "
              f"{len(r['tests'])} tests)", flush=True)
        report["files"].append(r)
    report["seconds"] = round(time.time() - t0, 1)
    report["outcomes"] = {k: v for r in report["files"]
                          for k, v in r["tests"].items()}
    report["verdicts"] = {r["file"]: r["verdict"] for r in report["files"]}
    return 0, report


#: Outcomes that count as "this test was working".
_GOOD = ("passed", "skipped")


def compare(baseline: dict, now: dict) -> dict:
    """Regressions, disappearances and repairs against a recorded baseline."""
    b_out, n_out = baseline.get("outcomes", {}), now.get("outcomes", {})
    b_ver, n_ver = baseline.get("verdicts", {}), now.get("verdicts", {})
    regressions = sorted(k for k, v in b_out.items()
                         if v in _GOOD and n_out.get(k, "missing") not in _GOOD)
    disappeared = sorted(k for k in b_out if k not in n_out)
    repairs = sorted(k for k, v in b_out.items()
                     if v not in _GOOD and n_out.get(k) in _GOOD)
    # A file that stopped running at all is the loudest regression there is,
    # and its tests show up as `disappeared` too; name the file explicitly.
    file_regressions = sorted(
        f for f, v in b_ver.items()
        if v in ("ok", "failed") and n_ver.get(f, "missing") in
        ("timeout", "crash", "error", "empty", "missing"))
    return {
        "regressions": regressions,
        "disappeared": [d for d in disappeared
                        if d.split("::")[0] not in file_regressions],
        "file_regressions": file_regressions,
        "repairs": repairs,
        "ok": not regressions and not disappeared and not file_regressions,
        "baseline_tests": len(b_out), "now_tests": len(n_out),
        "baseline_failing": sorted(k for k, v in b_out.items() if v not in _GOOD),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier", default="fast", choices=("fast", "full"))
    ap.add_argument("--work", default=None)
    ap.add_argument("--record", type=Path,
                    help="write the result as a baseline manifest")
    ap.add_argument("--against", type=Path, help="compare with this baseline")
    a = ap.parse_args()
    work = a.work or f"/tmp/suite-{a.tier}-{os.getpid()}"
    rc, report = run(["--tier", a.tier, "--work", work])
    if a.record:
        a.record.write_text(json.dumps(report, indent=1, sort_keys=True))
        print(f"recorded {len(report['outcomes'])} test outcomes over "
              f"{len(report['files'])} files -> {a.record}")
    if a.against:
        print(json.dumps(compare(json.loads(a.against.read_text()), report),
                         indent=1))
    else:
        print(json.dumps({"verdicts": report["verdicts"],
                          "failing": sorted(k for k, v in report["outcomes"].items()
                                            if v not in _GOOD),
                          "seconds": report["seconds"]}, indent=1))
    return rc


if __name__ == "__main__":
    sys.exit(main())
