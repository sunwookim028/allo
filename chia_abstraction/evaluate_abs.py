# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Score one candidate patch, in either disposition. FROZEN: unreachable to the agent.

A candidate is a git PATCH. It is evaluated in a SLOT -- a git worktree at a
fixed path, reset to `FROZEN_REF` before every candidate -- so that the whole
tree except the patch's own files comes from git, not from any working tree:

    git -C <slot> checkout -f <ref>
    git -C <slot> clean -xffd -e /mlir/build -e /.eval      # keep the warm build
    git -C <slot> apply <patch>                             # allowlisted paths only

Fixed path, not a fresh worktree per candidate, for one measured reason: the C++
bindings must be rebuilt for every `maintaining` candidate, a CMake build
directory records absolute paths, and a cold configure-and-build is ~6 minutes
against ~30 s warm. So `mlir/build` survives the reset and `ninja` rebuilds only
what the patch changed. `mlir/**/CMakeLists.txt` is frozen precisely because
that build directory is the one thing the candidate's tree can write to.

Freshness is then verified rather than assumed: after the reset and the apply,
`git status --porcelain --untracked-files=all` must name EXACTLY the patch's
files (plus the two ignored build/work directories), and every path in
`FROZEN_CHECK` is compared byte for byte with `git show <ref>:<path>`.

The ladder, cheapest first (CAKE: filter through the cheap static gates before
spending hardware time):

    0  policy      patch_policy.check(): path allowlist, added-line policy,
                   and -- for `maintaining` -- the rule that a new schedule
                   primitive must validate its arguments and carry a docstring.
                   No I/O, milliseconds.
    1  assemble    slot reset, patch applied, frozen files byte-identical.
    2  gate:build  cmake if the patch touched CMake-visible files, then
                   `ninja -C mlir/build -j48`. ~30 s warm, minutes if a header
                   moved. A `using` candidate still runs it: it must be a
                   no-op, and if it is not, something is wrong.
    3  gate:import `import allo` and its HLS backends, and the resolved
                   `allo.__file__` must be inside the slot.
    4  gate:tests  Allo's own suites (suite_runner.py) against the RECORDED
                   baseline, per test name. `main` does not pass its own suite,
                   so the comparison is against what main measures, never zero.
    5  gate:design correctness on every design case:
                     bench_isa + stress_isa   TinyTPU-isa, functional
                     design_case <name>       csim bit-exact vs numpy, which is
                                              the only correctness signal that
                                              does not pass through Allo's own
                                              simulator
                     limits_runner            tests/limits/ verdicts unchanged,
                                              and a REPRODUCES -> FIXED is
                                              reported as a repair
    6  ppa         TinyTPU-isa RTL cosim cycles at the workload's scored
                   shapes, plus csynth latency/interval/area/clock for each
                   synthesisable design case.
    7  objective   objective.py: cycles scored, resources CONSTRAINED, per
                   design case, never aggregated.

Every candidate-importing process runs under `bwrap` when available: filesystem
read-only, only the work directory and a private `/tmp` writable, own PID
namespace. The build is the exception -- it must write `mlir/build` -- and is
sandboxed to exactly that.

Every verdict is vouched for by `abs_gate_runner.py` with a per-run nonce read
before `allo` is imported, never read from a printed line. `numpy` and
`builtins` are frozen in the gate process before the candidate's `allo` loads;
`allo` itself cannot be frozen here, because it IS the candidate. That is the
one thing this loop gives up relative to the design-level loop, and the answer
to it is that the design cases' correctness comes from `csim` -- the emitted
C++, compiled by g++ and run -- and the cosim numbers come from Vitis's own
logs, neither of which executes Allo.

What is REUSED from the design-level harness (`chia_agent/`), rather than
rewritten:
    gate_runner._Frozen/_snapshot/_changed   the module-freezing primitive,
                                             imported by abs_gate_runner.py
    evaluate.parse_synth / check_memory_model / SIMTIME_SLACK
                                             the csynth parse and the
                                             memory-model checks, imported here
    bench_isa.py / stress_isa.py / cosim.py  the TinyTPU-isa design case's own
                                             frozen gates, run as they are
    param_check.py                           the parametricity gate
    preflight.py / spend.py / llm.py         billing gate, spend accounting,
                                             the opencode LLM wrapper
What is rewritten, and why: the gate runner (ordering -- `allo` is the
candidate), the policy (a diff over a compiler, not two spec files), the
evaluator (a slot worktree with a C++ rebuild, not a composed tree), and the
loop (two dispositions, per-iteration instrumentation).

    python evaluate_abs.py --patch P --disposition maintaining --slot 0 \
                           [--workload gemm_int8_16] [--tier loop|accept]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DESIGN_PKG = "examples/accelerator/tinytpu_vitis"
DESIGN_AGENT = REPO / DESIGN_PKG / "chia_agent"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(DESIGN_AGENT))

import design_cases                                          # noqa: E402
import limits_runner                                         # noqa: E402
import objective                                             # noqa: E402
import patch_policy                                          # noqa: E402
import suite_runner                                          # noqa: E402
import workloads                                             # noqa: E402
#: The design-level harness's csynth parse and memory-model checks, reused.
import evaluate as design_eval                               # noqa: E402

FROZEN_REF = os.environ.get("CHIA_FROZEN_REF", "HEAD")
#: main's commit the design case's own evaluator must be byte-identical to,
#: exactly as `chia_agent/evaluate.py` pins it. Kept in sync deliberately.
MAIN_BASE = design_eval.MAIN_BASE

ALLO_PYTHON = os.environ.get(
    "TINYTPU_ALLO_PYTHON", "/home/sk3463/miniconda3/envs/allo/bin/python")
LLVM_BUILD_DIR = os.environ.get(
    "LLVM_BUILD_DIR", "/home/sk3463/llvm-allo-6b09f739/build")
VITIS_BIN = os.environ.get("CHIA_VITIS_BIN",
                           "/opt/xilinx/Vitis_HLS/2023.2/bin")
BWRAP = shutil.which("bwrap")

#: Paths compared byte for byte with git after the patch is applied. Every
#: gate, golden, test and design file the verdict depends on.
FROZEN_CHECK = [
    f"{DESIGN_PKG}/{f}" for f in
    ("cosim.py", "bench_isa.py", "stress_isa.py", "isa_ref.py", "kpn_model.py",
     "microarch_isa.py", "isa_dsl.py")
] + [
    f"{DESIGN_PKG}/chia_agent/{f}" for f in
    ("gate_runner.py", "param_check.py", "spec_policy.py", "evaluate.py")
] + [
    "chia_abstraction/abs_gate_runner.py",
    "chia_abstraction/design_cases.py",
    "chia_abstraction/evaluate_abs.py",
    "chia_abstraction/limits_runner.py",
    "chia_abstraction/objective.py",
    "chia_abstraction/patch_policy.py",
    "chia_abstraction/suite_runner.py",
    "chia_abstraction/workloads.py",
    "mlir/CMakeLists.txt",
    "mlir/lib/CMakeLists.txt",
    "mlir/lib/Translation/CMakeLists.txt",
]
#: The design's evaluator is frozen against main, not just against the ref:
#: this loop must measure the design exactly as main does.
DESIGN_EVALUATOR = [f"{DESIGN_PKG}/{f}" for f in
                    ("cosim.py", "bench_isa.py", "stress_isa.py", "isa_ref.py",
                     "kpn_model.py")]
#: A `using` candidate patches these, so they cannot also be frozen against
#: the ref in that disposition.
_USING_EDITS = {f"{DESIGN_PKG}/microarch_isa.py", f"{DESIGN_PKG}/isa_dsl.py"}

#: Untracked paths the reset keeps and the freshness check ignores.
KEEP = ("mlir/build", ".eval")

BUILD_TIMEOUT = 2400
GATE_TIMEOUT = 600
SUITE_TIMEOUT = {"loop": 1200, "accept": 5400}
LIMITS_TIMEOUT = {"loop": 1800, "accept": 5400}
CASE_TIMEOUT = 900
COSIM_TIMEOUT = 2400
TARGET_NS = design_eval.TARGET_NS
SIMTIME_SLACK = design_eval.SIMTIME_SLACK

BASELINE_DIR = HERE / "baseline"


class _Done(Exception):
    """Not an error: `--gate-only` stops the ladder before PPA."""


class Reject(Exception):
    def __init__(self, stage, detail):
        super().__init__(detail)
        self.stage, self.detail = stage, str(detail)


# -- shell --------------------------------------------------------------------
def git(args, cwd=REPO, check=True, binary=False):
    p = subprocess.run(["git", *args], cwd=cwd, capture_output=True,
                       check=False)
    if check and p.returncode:
        raise Reject("git", f"git {' '.join(args)} failed: "
                            f"{p.stderr.decode(errors='replace')[:500]}")
    return p.stdout if binary else p.stdout.decode(errors="replace")


def run(cmd, cwd, env, timeout, writable=None, ro=None, stdin=None):
    if writable is not None:
        cmd = boxed(cmd, writable, ro)
    t = time.time()
    p = subprocess.Popen(cmd, cwd=cwd, env=env, text=True, encoding="utf-8",
                         errors="replace", stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, start_new_session=True,
                         stdin=subprocess.PIPE if stdin is not None
                         else subprocess.DEVNULL)
    try:
        out, _ = p.communicate(input=stdin, timeout=timeout)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)     # the whole group: vitis included
        out, _ = p.communicate()
        out, rc = (out or "") + f"\nTIMEOUT after {timeout}s (deadlock?)", 124
    return rc, out or "", round(time.time() - t, 1)


def boxed(cmd, writable: Path, ro: Path | None = None):
    """Read-only filesystem, `writable` writable, private /tmp, own PID ns.

    Bind ORDER matters, and it is the reverse of the design-level evaluator's.
    There the evaluation tree sits INSIDE the work directory, so `--bind work`
    then `--ro-bind tree` makes the tree read-only within a writable work dir.
    Here the work directory (`<slot>/.eval`) sits inside the slot, so the
    read-only bind of the slot has to come FIRST -- otherwise it shadows the
    writable bind and every gate dies with EROFS on its own work directory.
    """
    if not BWRAP:
        return list(cmd)
    ro_first = ["--ro-bind", str(ro), str(ro)] if ro else []
    return [BWRAP, "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc",
            "--tmpfs", "/tmp", "--tmpfs", "/dev/shm", *ro_first,
            "--bind", str(writable), str(writable),
            "--unshare-pid", "--die-with-parent", "--", *cmd]


def env_for(slot: Path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("TPU_")}
    env.update({
        "PYTHONPATH": str(slot),
        "PATH": f"{Path(ALLO_PYTHON).parent}:{VITIS_BIN}:{env['PATH']}",
        "LLVM_BUILD_DIR": LLVM_BUILD_DIR,
        "OMP_NUM_THREADS": "8",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    return env


# -- 1. assemble --------------------------------------------------------------
def slot_path(n: int) -> Path:
    return REPO / ".chia_scratch" / f"slot{n}"


def ensure_slot(n: int, ref: str) -> Path:
    slot = slot_path(n)
    if not (slot / ".git").exists():
        slot.parent.mkdir(parents=True, exist_ok=True)
        git(["worktree", "add", "--detach", str(slot), ref])
    return slot


def reset_slot(slot: Path, ref: str):
    git(["checkout", "-f", ref], cwd=slot)
    git(["clean", "-xffd", *sum((["-e", f"/{k}"] for k in KEEP), [])],
        cwd=slot)
    dirty = [l for l in git(["status", "--porcelain",
                             "--untracked-files=all"], cwd=slot).splitlines()
             if l.strip() and not any(k in l for k in KEEP)]
    if dirty:
        raise Reject("assemble", "the slot is not clean after reset:\n"
                     + "\n".join(dirty[:20]))


def apply_patch(slot: Path, patch: Path, disposition: str, expected: set):
    rc, out, _ = run(["git", "apply", "--verbose", str(patch.resolve())],
                     slot, dict(os.environ), 120)
    if rc:
        raise Reject("assemble", f"git apply failed:\n{out[-3000:]}")
    touched = set()
    for line in git(["status", "--porcelain", "--untracked-files=all"],
                    cwd=slot).splitlines():
        line = line.strip()
        if not line or any(k in line for k in KEEP):
            continue
        touched.add(line.split(maxsplit=1)[1].strip('"'))
    extra = touched - expected
    if extra:
        raise Reject("tamper", f"the patch changed files it did not declare: "
                               f"{sorted(extra)}")
    missing = expected - touched
    if missing:
        raise Reject("assemble", f"the patch declared {sorted(missing)} but "
                                 f"changed nothing there")
    return sorted(touched)


def verify_frozen(slot: Path, ref: str, disposition: str) -> dict:
    """Every frozen path byte-identical to git, and the design's own evaluator
    byte-identical to main. Returns the manifest for the tamper re-checks."""
    manifest = {}
    for rel in FROZEN_CHECK:
        if disposition == "using" and rel in _USING_EDITS:
            continue
        want = git(["show", f"{ref}:{rel}"], binary=True)
        p = slot / rel
        if not p.is_file():
            raise Reject("tamper", f"frozen file {rel} is missing")
        got = p.read_bytes()
        if got != want:
            raise Reject("tamper", f"frozen file {rel} differs from "
                                   f"{ref[:8]} by {len(got) - len(want)} bytes")
        manifest[rel] = hashlib.sha256(got).hexdigest()
    for rel in DESIGN_EVALUATOR:
        if git(["show", f"{ref}:{rel}"], binary=True) != git(
                ["show", f"{MAIN_BASE}:{rel}"], binary=True):
            raise Reject("setup", f"{rel} @ {ref[:8]} differs from main @ "
                                  f"{MAIN_BASE}; this loop must measure the "
                                  f"design exactly as main does")
    return manifest


def recheck_frozen(slot: Path, manifest: dict, after: str):
    for rel, digest in manifest.items():
        p = slot / rel
        if not p.is_file() or hashlib.sha256(
                p.read_bytes()).hexdigest() != digest:
            raise Reject("tamper", f"after {after}, the frozen file {rel} "
                                   f"changed")


# -- 2. build -----------------------------------------------------------------
def build(slot: Path, work: Path, out: Path) -> dict:
    env = env_for(slot)
    bld = slot / "mlir" / "build"
    logs = {}
    if not (bld / "build.ninja").exists():
        rc, o, sec = run(
            ["cmake", "-G", "Ninja", "-S", "mlir", "-B", "mlir/build",
             f"-DMLIR_DIR={LLVM_BUILD_DIR}/lib/cmake/mlir",
             f"-DPython3_EXECUTABLE={ALLO_PYTHON}",
             f"-DPython_EXECUTABLE={ALLO_PYTHON}",
             "-DMLIR_BINDINGS_PYTHON_NB_DOMAIN=allo"], slot, env, 900)
        (out / "cmake.log").write_text(o)
        logs["cmake_seconds"] = sec
        if rc:
            raise Reject("gate:build", f"cmake failed:\n{o[-3000:]}")
    # The build is the one stage that must write; it writes only mlir/build.
    rc, o, sec = run(["ninja", "-C", "mlir/build", "-j", "48"], slot, env,
                     BUILD_TIMEOUT, writable=bld)
    (out / "ninja.log").write_text(o)
    logs["ninja_seconds"] = sec
    if rc:
        # The compiler diagnostic is the whole feedback for a C++ candidate.
        raise Reject("gate:build", f"ninja failed:\n{o[-6000:]}")
    logs["rebuilt_targets"] = len(re.findall(r"^\[\d+/\d+\]", o, re.M))
    return logs


# -- 3..6 vouched checks ------------------------------------------------------
def vouched(check, slot: Path, cwd, env, work: Path, timeout, args=(),
            log: Path | None = None):
    """(ok, rc, out, seconds). `ok` only on `CHIA-GATE <check> OK <nonce>`."""
    nonce = secrets.token_hex(16)
    runner = str(slot / "chia_abstraction" / "abs_gate_runner.py")
    rc, out, sec = run([ALLO_PYTHON, runner, check, *args], cwd, env, timeout,
                       writable=work, ro=slot, stdin=nonce + "\n")
    ok = rc == 0 and f"CHIA-GATE {check} OK {nonce}" in out.splitlines()
    out = out.replace(nonce, "<nonce>")   # never into a verdict or a log
    if log:
        log.write_text(out)
    return ok, rc, out, sec


def _payload(out: str, tag: str):
    for line in reversed(out.splitlines()):
        if line.startswith(tag + " "):
            try:
                return json.loads(line[len(tag) + 1:])
            except json.JSONDecodeError:
                continue
    return None


def gate_import(slot, env, work, out):
    ok, rc, o, sec = vouched("build_import", slot, slot, env, work,
                             GATE_TIMEOUT, log=out / "import.log")
    if not ok:
        raise Reject("gate:import", o[-4000:])
    m = re.search(r"^ALLO (\S+)$", o, re.M)
    if not m or not m.group(1).startswith(str(slot)):
        raise Reject("gate:import", f"allo resolved to {m and m.group(1)}, "
                                    f"not inside the slot {slot}")
    return {"allo": m.group(1), "seconds": sec}


def gate_tests(slot, env, work, out, tier, record=False):
    """Allo's own suites against the recorded baseline.

    The baseline is RECORDED BY THIS FUNCTION on a `--record-baseline` run, not
    by invoking `suite_runner.py` by hand. That is not tidiness: a hand-run
    baseline is measured outside the bubblewrap sandbox, with a writable
    checkout, and it does not reproduce inside it. Measured 2026-09-22: a
    no-patch control against a hand-recorded baseline showed ELEVEN
    regressions on an unmodified tree (eight in `tests/test_verify.py`, plus
    `test_df_unit::test_uint` and two in `test_stream_of_blocks`), every one an
    artifact of the environment rather than of any candidate. A baseline that
    does not reproduce makes every delta meaningless, so the baseline now
    comes from the same vouched, sandboxed path a candidate does.
    """
    suite_tier = "full" if tier == "accept" else "fast"
    baseline_file = BASELINE_DIR / f"suite_{suite_tier}.json"
    if not record and not baseline_file.exists():
        raise Reject("setup", f"no recorded test baseline at {baseline_file}; "
                              f"run evaluate_abs.py --record-baseline first")
    ok, rc, o, sec = vouched("pytest", slot, slot, env, work,
                             SUITE_TIMEOUT[tier],
                             args=["--tier", suite_tier,
                                   "--work", str(work / "suite")],
                             log=out / "suite.log")
    report = _payload(o, "SUITE")
    if not ok or report is None:
        raise Reject("gate:tests", f"the suite runner did not complete:\n"
                                   f"{o[-4000:]}")
    (out / "suite.json").write_text(json.dumps(report, indent=1))
    if record:
        BASELINE_DIR.mkdir(parents=True, exist_ok=True)
        baseline_file.write_text(json.dumps(report, indent=1, sort_keys=True))
        return {"seconds": sec, "tier": suite_tier, "RECORDED": str(baseline_file),
                "tests": len(report["outcomes"]),
                "already_failing": sorted(
                    k for k, v in report["outcomes"].items()
                    if v not in suite_runner._GOOD)}
    cmp = suite_runner.compare(json.loads(baseline_file.read_text()), report)
    if not cmp["ok"]:
        raise Reject("gate:tests", json.dumps(
            {k: cmp[k] for k in ("regressions", "disappeared",
                                 "file_regressions")}, indent=1)
            + "\n\nthe comparison is against main's MEASURED baseline "
            f"({cmp['baseline_tests']} tests, "
            f"{len(cmp['baseline_failing'])} already failing), not against "
            "zero.")
    return {"seconds": sec, "tier": suite_tier, **{
        k: cmp[k] for k in ("repairs", "baseline_tests", "now_tests")}}


def gate_design_functional(slot, env, work, out):
    """TinyTPU-isa's own frozen functional gates, run as main runs them."""
    res = {}
    ok, rc, o, sec = vouched("bench_isa", slot, slot, env, work, GATE_TIMEOUT,
                             log=out / "bench_isa.log")
    if not ok or not re.search(r"^  ALL EXACT$", o, re.M) or "FAILURES" in o:
        raise Reject("gate:bench_isa", o[-4000:])
    res["bench_isa"] = {"seconds": sec, "line": "ALL EXACT"}
    ok, rc, o, sec = vouched("stress_isa", slot, slot, env, work, GATE_TIMEOUT,
                             log=out / "stress_isa.log")
    m = re.search(r"^  STRESS OK: (\d+)/(\d+) runs exact", o, re.M)
    if not ok or not m or m.group(1) != m.group(2):
        raise Reject("gate:stress", o[-4000:])
    res["stress_isa"] = {"seconds": sec, "runs": int(m.group(1)),
                         "line": m.group(0).strip()}
    return res


def gate_cases(slot, env, work, out, names, csyn: bool,
               allow_csyn_failure: bool = False):
    """Every named design case, vouched. `allow_csyn_failure` is for the PPA
    pass over cases `design_cases.CSYN_OK` says Vitis refuses today: a failure
    there is the status quo, not a regression, and a SUCCESS there is a
    newly-expressible architecture."""
    res = {}
    for name in names:
        args = [name, "--work", str(work / "cases")]
        if csyn:
            args.append("--csyn")
        ok, rc, o, sec = vouched("design_case", slot, slot, env, work,
                                 CASE_TIMEOUT, args=args,
                                 log=out / f"case_{name}.log")
        rep = _payload(o, "CASE")
        tolerated = (allow_csyn_failure
                     and not design_cases.CSYN_OK.get(name, True))
        if not ok or rep is None:
            if tolerated:
                res[name] = {"case": name, "csyn": {"failed_as_expected": True},
                             "seconds_total": sec}
                continue
            raise Reject(f"gate:case:{name}", o[-4000:])
        res[name] = {**rep, "seconds_total": sec}
    return res


def gate_limits(slot, env, work, out, tier, record=False):
    """tests/limits/ verdicts against the recorded baseline; same rule as
    gate_tests -- the baseline is recorded from inside the sandbox."""
    lim_tier = "full" if tier == "accept" else "fast"
    baseline_file = BASELINE_DIR / "limits.json"
    if not record and not baseline_file.exists():
        raise Reject("setup", f"no recorded limits baseline at {baseline_file}")
    # limits_runner spawns a subprocess per repro, which by design provokes
    # MLIR aborts and `sys.exit(1)`; it is not vouched, because its verdict is
    # a COMPARISON against a recorded baseline, made here, and every repro is
    # a frozen file from git. Its output is the input to that comparison.
    rc, o, sec = run([ALLO_PYTHON, str(slot / "chia_abstraction" /
                                       "limits_runner.py"),
                      "--tier", lim_tier, "--work", str(work / "limits")],
                     slot, env, LIMITS_TIMEOUT[tier], writable=work, ro=slot)
    (out / "limits.log").write_text(o)
    report = _payload(o, "LIMITS")
    if report is None:
        raise Reject("gate:limits", f"limits_runner produced no LIMITS line:\n"
                                    f"{o[-4000:]}")
    if record:
        BASELINE_DIR.mkdir(parents=True, exist_ok=True)
        baseline_file.write_text(json.dumps(report, indent=1, sort_keys=True))
        return {"seconds": sec, "tier": lim_tier,
                "RECORDED": str(baseline_file),
                "verdicts": report["verdicts"], "silent": report["silent"]}
    cmp = limits_runner.compare(json.loads(baseline_file.read_text()), report)
    if not cmp["ok"]:
        raise Reject("gate:limits", json.dumps(
            {k: cmp[k] for k in ("regressions", "silenced", "vanished")},
            indent=1))
    return {"seconds": sec, "tier": lim_tier, "fixed": cmp["fixed"],
            "items": cmp["now_items"]}


# -- 6. PPA -------------------------------------------------------------------
def cosim_ppa(slot, env, work, out, shapes):
    """TinyTPU-isa RTL cosim cycles + csynth, cross-checked outside the process.

    The per-shape checks are `chia_agent/evaluate.py`'s, and `parse_synth` and
    `check_memory_model` are imported from it: each shape reported exactly once
    with `mismatches = 0`, its own cosim log carrying that line and a PASS, and
    the cycle count agreeing with the log's simulated time. Vitis writes those
    logs, so they are the one number in the ladder that no Python the candidate
    controls produced.
    """
    prj = work / "isa_sweep.prj"
    env = dict(env, TPU_SHAPES=",".join(shapes), TPU_PRJ=str(prj))
    ok, rc, o, sec = vouched("cosim", slot, work, env, work, COSIM_TIMEOUT,
                             log=out / "cosim.log")
    if not ok:
        raise Reject("ppa:cosim", o[-4000:])
    try:
        synth = design_eval.parse_synth(prj)
        design_eval.check_memory_model(prj)
    except design_eval.Reject as r:
        raise Reject(f"ppa:{r.stage}", r.detail)
    cycles = {}
    for s in shapes:
        M, K, N = (int(x) for x in s.split("x"))
        pat = re.compile(rf"^\s*{M}x\s*{K}x\s*{N}\s+cycles=(\S+)\s+(.*)$")
        rows = [m for m in (pat.match(l) for l in o.splitlines()) if m]
        if len(rows) != 1:
            raise Reject("ppa:cosim",
                         f"shape {s} reported {len(rows)} times\n{o[-2500:]}")
        n, tb = rows[0].group(1), rows[0].group(2)
        want = f"TB {M}x{K}x{N} mismatches = 0 / {M * N}"
        if n == "None" or tb.strip() != want:
            raise Reject("ppa:cosim", f"{s}: cycles={n} tb='{tb}' "
                                      f"(want '{want}')\n{o[-2500:]}")
        log = prj / f"cosim_{M}x{K}x{N}.log"
        text = log.read_text(errors="replace") if log.exists() else ""
        if want not in text or "C/RTL co-simulation finished: PASS" not in text:
            raise Reject("ppa:cosim", f"{s}: its own cosim log lacks "
                                      f"'{want}' + PASS")
        t = [int(x) for x in re.findall(
            r'RTL Simulation : \d+ / 1 \[n/a\] @ "(\d+)"', text)]
        if len(t) == 2:
            sim = (t[1] - t[0]) / (TARGET_NS * 1000)
            if abs(sim - int(n)) > SIMTIME_SLACK:
                raise Reject("ppa:cosim", f"{s}: report says {n} cycles, "
                                          f"simulated time says {sim:.0f}")
        cycles[s] = int(n)
        shutil.copy2(log, out / log.name) if log.exists() else None
    return {"cycles": cycles, "area": synth["area"],
            "estimated_ns": synth["estimated_ns"], "seconds": sec}


# -- 7. the verdict -----------------------------------------------------------
def baseline_ppa(ref: str) -> dict:
    f = BASELINE_DIR / "ppa.json"
    if not f.exists():
        raise Reject("setup", f"no recorded PPA baseline at {f}; run "
                              f"evaluate_abs.py --record-baseline first")
    rec = json.loads(f.read_text())
    if rec.get("ref") != ref:
        # Loud, not fatal: the baseline was measured at one commit and the
        # candidate is evaluated at another. A number compared across commits
        # is not a measurement of the candidate.
        rec["WARNING"] = (f"baseline measured at {rec.get('ref')}, candidate "
                          f"at {ref}")
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patch", type=Path,
                    help="the candidate diff; omit for a no-op control")
    ap.add_argument("--disposition", default="maintaining",
                    choices=patch_policy.DISPOSITIONS)
    ap.add_argument("--workload", default=workloads.DEFAULT_WORKLOAD)
    ap.add_argument("--slot", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tier", default="loop", choices=("loop", "accept"))
    ap.add_argument("--gate-only", action="store_true",
                    help="stop after the correctness gates; no PPA, no "
                         "objective. ~5 min instead of ~10")
    ap.add_argument("--record-baseline", action="store_true",
                    help="write baseline/ppa.json from this (no-patch) run")
    ap.add_argument("--skip", default="",
                    help="comma-separated gate names to record but not block "
                         "on. TEST HOOK: only test_abs_harness.py uses it, to "
                         "show what a gate LOWER in the ladder would have said")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    skip = {s for s in a.skip.split(",") if s}
    spec = workloads.get(a.workload)
    started = time.time()
    result = {"ok": False, "disposition": a.disposition, "tier": a.tier,
              "workload": a.workload, "patch": str(a.patch) if a.patch else None,
              "sandbox": bool(BWRAP), "stages": {}, "hints": [], "skipped": sorted(skip),
              "measurement": "TinyTPU-isa: Vitis HLS 2023.2 csynth + xsim "
                             "C/RTL cosim cycles. Other design cases: csim "
                             "bit-exactness + csynth latency/interval/area. "
                             "Resources CONSTRAINED, cycles scored; see "
                             "objective.py"}

    def stage(name, fn):
        if name in skip:
            try:
                result["stages"][name] = fn()
            except Reject as r:
                result["stages"][name] = {"WOULD_HAVE_REJECTED": r.stage,
                                          "detail": r.detail[-2000:]}
            return result["stages"].get(name)
        v = fn()
        result["stages"][name] = v
        return v

    try:
        ref = git(["rev-parse", "--verify", f"{FROZEN_REF}^{{commit}}"]).strip()
        result["frozen_ref"] = ref
        diff = a.patch.read_text() if a.patch else ""

        # 0. policy -- milliseconds, no I/O.
        if diff:
            expected = patch_policy.paths_in_diff(diff)
            problems, hints = patch_policy.check(a.disposition, diff)
            result["hints"] = hints
            result["touches"] = sorted(expected)
            if problems:
                raise Reject("policy", "; ".join(problems))
        else:
            expected, result["touches"] = set(), []

        slot = ensure_slot(a.slot, ref)
        work = slot / ".eval"
        if work.exists():
            shutil.rmtree(work)          # a stale report is never a result
        work.mkdir(parents=True)

        # 1. assemble
        reset_slot(slot, ref)
        if diff:
            apply_patch(slot, a.patch, a.disposition, expected)
            # Rule 4 needs the PATCHED source, so it is re-run here.
            problems = patch_policy.primitive_violations(
                diff, (slot / "allo/customize.py").read_text()
                if a.disposition == "maintaining" else None)
            if problems:
                raise Reject("policy", "; ".join(problems))
        manifest = verify_frozen(slot, ref, a.disposition)
        result["stages"]["assemble"] = {"slot": str(slot),
                                        "frozen_checked": len(manifest)}
        env = env_for(slot)
        check = lambda after: recheck_frozen(slot, manifest, after)

        # 2..3 the cheap gates
        stage("build", lambda: build(slot, work, a.out))
        check("the build")
        stage("import", lambda: gate_import(slot, env, work, a.out))
        check("the import gate")

        # 4. Allo's own suites, against the measured baseline
        stage("tests", lambda: gate_tests(slot, env, work, a.out,
                                          a.tier, a.record_baseline))
        check("the test suites")

        # 5. correctness on every design case
        stage("design_functional",
              lambda: gate_design_functional(slot, env, work, a.out))
        check("the design's functional gates")
        names = (design_cases.ALL_CASES if a.tier == "accept"
                 else design_cases.LOOP_CASES)
        stage("cases", lambda: gate_cases(slot, env, work, a.out, names, False))
        check("the design cases")
        stage("limits", lambda: gate_limits(slot, env, work, a.out,
                                            a.tier, a.record_baseline))
        check("tests/limits")

        # 6. PPA
        if a.gate_only:
            result["ok"] = True
            result["gate_only"] = True
            raise _Done
        ppa = {}
        ppa["tinytpu_isa"] = stage(
            "ppa_tinytpu",
            lambda: cosim_ppa(slot, env, work, a.out, spec["scored_shapes"]))
        check("cosim")
        # Every case is asked for csynth, including the ones that do not
        # synthesise at HEAD: a candidate that made `systolic_1d`
        # synthesisable has made a second architecture EXPRESSIBLE, which is
        # this project's standard and is worth more than a faster current
        # design. `CSYN_OK` says which cases csynth refuses today, and a case
        # that csynths anyway is recorded in `newly_expressible`.
        syn = stage("ppa_cases", lambda: gate_cases(
            slot, env, work, a.out, design_cases.ALL_CASES, True,
            allow_csyn_failure=True))
        check("the design cases' csynth")
        newly = sorted(
            n for n, rep in (syn or {}).items()
            if not design_cases.CSYN_OK.get(n, True)
            and (rep.get("csyn") or {}).get("latency_worst") is not None)
        result["newly_expressible"] = newly
        for name, rep in (syn or {}).items():
            c = rep.get("csyn") or {}
            if "skipped" in c or "failed_as_expected" in c or not c:
                continue
            ppa[name] = {"cycles": {"latency": c.get("latency_worst"),
                                    "interval": c.get("interval_max")},
                         "area": c.get("area"),
                         "estimated_ns": c.get("estimated_ns")}
        result["ppa"] = ppa

        # 7. the objective
        if a.record_baseline:
            if a.patch:
                raise Reject("setup", "--record-baseline needs a no-patch run")
            BASELINE_DIR.mkdir(parents=True, exist_ok=True)
            (BASELINE_DIR / "ppa.json").write_text(json.dumps(
                {"ref": ref, "workload": a.workload, "cases": ppa,
                 "measured": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                           time.gmtime())},
                indent=1, sort_keys=True))
            result["recorded_baseline"] = str(BASELINE_DIR / "ppa.json")
        else:
            base = baseline_ppa(ref)
            cases = {n: objective.case_verdict(base["cases"][n], ppa[n])
                     for n in ppa if n in base.get("cases", {})}
            limits_fixed = (result["stages"].get("limits") or {}).get("fixed", [])
            result["objective"] = {
                "cases": cases,
                **objective.classify(cases, newly_expressible=newly,
                                     limits_fixed=limits_fixed)}
            if base.get("WARNING"):
                result["objective"]["WARNING"] = base["WARNING"]
            over = result["objective"]["over_budget"]
            if over and not a.record_baseline:
                # A resource overrun is a gate, not a weighting. Recorded as a
                # rejection; the objective block still shows what it bought.
                result["stages"]["resources"] = {
                    "over_budget": over,
                    "detail": {c: cases[c]["resources"]["over_budget"]
                               for c in over}}
                raise Reject("gate:resources",
                             f"over the resource budget on {over}: "
                             + json.dumps({c: cases[c]["resources"]["over_budget"]
                                           for c in over}))
        result["ok"] = True
    except _Done:
        pass
    except Reject as r:
        result.update(stage=r.stage, detail=r.detail[-6000:])
    except Exception as e:                                # noqa: BLE001
        import traceback
        result.update(stage="harness", detail=traceback.format_exc()[-6000:])
    result["seconds"] = round(time.time() - started, 1)
    detail = result.pop("detail", None)
    if detail:
        print(detail)
    (a.out / "verdict.json").write_text(json.dumps(result, indent=1,
                                                   sort_keys=True))
    print(json.dumps(result, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
