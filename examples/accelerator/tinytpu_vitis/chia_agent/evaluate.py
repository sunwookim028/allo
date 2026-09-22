# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Score one TinyTPU-isa candidate. FROZEN: the agent can neither edit nor import it.

A candidate is exactly two files, `microarch_isa.py` and `isa_dsl.py`, sitting in
a spec directory. Everything else the score depends on is taken from git at
`FROZEN_REF`, never from the working tree, so no edit anywhere on disk can move
the objective:

    frozen (from git @ FROZEN_REF)          editable (from --spec-dir)
    ------------------------------          --------------------------
    tinytpu_vitis/cosim.py                  tinytpu_vitis/microarch_isa.py
      the testbench generator,              tinytpu_vitis/isa_dsl.py
      the numpy golden reference,
      every Vitis TCL setting
    tinytpu_vitis/shapes.py       (the five benchmark shapes, one definition)
    tinytpu_vitis/bench_isa.py
    tinytpu_vitis/stress_isa.py   (main's correctness gate)
    tinytpu_vitis/isa_ref.py      (the ISA as numpy; stress_isa's reference)
    tinytpu_vitis/kpn_model.py    (stress_isa's deadlock diagnosis)
    chia_agent/gate_runner.py     (runs each check, vouches for its verdict)
    examples/__init__.py

The two tiers:

1. **gate** -- `bench_isa.py` (the published [-4, 4] setup) and main's
   `stress_isa.py` (492 runs at 476a70d8: full-range/corner/boundary operands, all 64
   shapes, prefilled C compared in full, vector and random programs, many
   invocations of one build) must both pass. Functional, on Allo's simulator,
   ~12 s. Each runs under `gate_runner.py`, and the verdict is its
   `CHIA-GATE <check> OK <nonce>` line -- a fresh nonce per run, handed over on
   stdin before the candidate is imported -- never the check's own printed
   `ALL EXACT` / `STRESS OK`, which the candidate's code could print itself.
   Then the parametricity gate: `param_check.py` rebuilds the candidate at
   TPU_MAXDIM=8, TPU_MAXDIM=12 and TPU_T=8/TPU_MAXDIM=32 (so T is varied too)
   and requires it to honour the parameters and be exact at
   every GEMM shape of that configuration (and on random programs), so a win
   that only exists at the scored T=4 / MAXDIM=16 is rejected (`gate:param`).
   The policy also refuses a literal T/MAXDIM and a net loss of more than 15
   comment/docstring lines against the frozen ref.
2. **score** -- `cosim.py` (Vitis HLS 2023.2 csynth + xsim C/RTL cosim), one
   testbench per shape, each bit-exact against numpy. The score is the SUM of
   cosim cycles over the requested shapes. It is an RTL measurement.

The memory model is not the candidate's to choose: every `TPU_*` environment
variable is scrubbed before `cosim.py` runs, so `-m_axi_latency` stays at its
default 0 and `-random_stall` stays off, and the generated `kernel.cpp` / TCL are
checked for interface-latency overrides afterwards.

Independent checks on the result, none of which trust the candidate's process:
T == 4 and MAXDIM == 16 (the matched-array premise of the Gemmini comparison,
docs/source/designs/gemmini_comparison.rst); each
requested shape reported exactly once; each shape's own cosim log carries
`mismatches = 0 / M*N` and a PASS; the cycle count agrees with that log's
simulated time; the csynth estimated clock meets the 3.33 ns target. The work
directory is wiped first, so a stale report can never be read as a result.

Every process that imports the candidate runs in a bubblewrap sandbox when
`bwrap` is available: the whole filesystem read-only, the evaluation tree
read-only too, only the work directory and a private `/tmp` writable, in its own
PID namespace so that a kill takes every descendant with it. That is the answer
to "the candidate's import-time code rewrites the gate": a test mutant narrowed
the datapath AND rewrote the tree's stress gate through `numpy.savetxt`, which
the policy did not know, and passed the gate. After every stage the tree is
also compared byte-for-byte with what was composed (no file changed, none
added) and the checkout's tracked files with how they were, so a host without
`bwrap` still catches the tamper after the fact.

Prints one JSON line (last line of stdout) and exits 0 iff the candidate passed.

    python evaluate.py --spec-dir DIR --work DIR [--shapes 4x4x4,16x16x16] [--gate-only]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent
REPO = AGENT_DIR.parents[3]
PKG = "examples/accelerator/tinytpu_vitis"

#: Frozen files are read from this commit (resolved to a hash per run), i.e.
#: from what is COMMITTED, never from the working tree. Only a person, in a
#: commit, can change them.
FROZEN_REF = os.environ.get("CHIA_FROZEN_REF", "HEAD")
#: main's commit this branch is based on (476a70d8: the gap-attribution stack,
#: 172 / 262 / 418 / 484 / 686, and AR_RAW_DIST in check_program). The design's own evaluator -- cosim.py, bench_isa.py
#: and the stress gate with its reference model -- must be byte-identical to
#: it, so this branch cannot drift from how main measures and verifies the
#: design. Moving it is a deliberate, reviewed commit.
MAIN_BASE = "476a70d8"
DESIGN_EVALUATOR = [f"{PKG}/{f}" for f in (
    "cosim.py", "bench_isa.py", "stress_isa.py", "isa_ref.py", "kpn_model.py",
    "shapes.py")]
GATE_RUNNER = f"{PKG}/chia_agent/gate_runner.py"
PARAM_CHECK = f"{PKG}/chia_agent/param_check.py"
FROZEN = [
    "examples/__init__.py",
    *DESIGN_EVALUATOR,
    GATE_RUNNER,
    PARAM_CHECK,
]
#: The parametricity gate: configurations the candidate is rebuilt at and must
#: be exact at (param_check.py), besides the scored T=4 / MAXDIM=16. 12 is
#: deliberately not a power of two, and the third case VARIES T.
#:
#: This list used to say "MAXDIM only: main's design supports T=4 alone (it
#: fails check_program at TPU_T=8)". That was FALSE. Measured on main,
#: `TPU_T=8 TPU_MAXDIM=32 param_check.py` prints `PARAM OK: 408/408 runs
#: exact` and `TPU_T=8 TPU_MAXDIM=32 bench_isa.py 32 32 32` prints
#: `ALL EXACT`. What is limited is the HARNESS, and the limit is MAXDIM/T >= 3,
#: not T == 4: three test-program generators address column block 2, which
#: exists only at that ratio, so at T=8 with MAXDIM=16 nine of 24 random seeds
#: cannot be generated and param_check refuses for want of programs rather
#: than for a wrong answer. MAXDIM=32 gives ratio 4 and every seed generates.
PARAM_CONFIGS = [{"TPU_MAXDIM": "8"}, {"TPU_MAXDIM": "12"},
                 {"TPU_T": "8", "TPU_MAXDIM": "32"}]
EDITABLE = ("microarch_isa.py", "isa_dsl.py")
#: What in the checkout itself the evaluation depends on: the `allo` package
#: (on PYTHONPATH), and this directory's evaluator, policy and design.
CHECKOUT_WATCH = ["allo", "examples/__init__.py", PKG]
#: The five benchmark shapes come from `{PKG}/shapes.py`, the one definition,
#: loaded BY PATH: this harness runs in a conda env that has no `allo`, so it
#: cannot import any design module, and `shapes.py` imports nothing so that it
#: can. `accept.py` and `test_harness.py` take `ALL_SHAPES` from here.
def _load_shapes():
    path = REPO / PKG / "shapes.py"
    spec = importlib.util.spec_from_file_location("tinytpu_shapes", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SHAPES = _load_shapes().SHAPES
ALL_SHAPES = [f"{M}x{K}x{N}" for (M, K, N) in SHAPES]
SEARCH_SHAPES = ["4x4x4", "16x16x16"]
TARGET_NS = 3.33
#: The unmodified design gates in ~5 s per script and cosims in ~125 s. A
#: candidate whose dataflow deadlocks blocks forever in the simulator, so the
#: gate fails it in minutes rather than the quarter-hour the smoke run lost.
GATE_TIMEOUT = 240
COSIM_TIMEOUT = 1800
#: xsim's transaction window runs a few cycles past HLS's latency count.
SIMTIME_SLACK = 12

#: Candidate processes run under this. Absent -> integrity checks only.
BWRAP = shutil.which("bwrap")

ALLO_PYTHON = os.environ.get(
    "TINYTPU_ALLO_PYTHON", "/home/sk3463/miniconda3/envs/allo/bin/python")
LLVM_BUILD_DIR = os.environ.get(
    "LLVM_BUILD_DIR", "/home/sk3463/llvm-allo-6b09f739/build")


class Reject(Exception):
    def __init__(self, stage, detail):
        super().__init__(detail)
        self.stage, self.detail = stage, detail


def git_show(ref, path):
    out = subprocess.run(["git", "show", f"{ref}:{path}"], cwd=REPO,
                         capture_output=True, check=False)
    if out.returncode:
        raise Reject("setup", f"cannot read frozen {path} @ {ref}: "
                              f"{out.stderr.decode()[:300]}")
    return out.stdout


def resolve_ref(ref):
    out = subprocess.run(["git", "rev-parse", "--verify", f"{ref}^{{commit}}"],
                         cwd=REPO, capture_output=True, text=True)
    if out.returncode:
        raise Reject("setup", f"cannot resolve {ref}")
    return out.stdout.strip()


def compose(spec_dir: Path, tree: Path, ref: str):
    """Evaluation tree = frozen files from git + the candidate's two files.

    Returns {relative path: sha256} for every file in the tree."""
    for rel in DESIGN_EVALUATOR:
        if git_show(ref, rel) != git_show(MAIN_BASE, rel):
            raise Reject("setup", f"{rel} @ {ref[:8]} differs from main @ {MAIN_BASE}")
    # The policy is executed from git too, not imported from the working tree.
    policy = {"__name__": "spec_policy"}
    exec(compile(git_show(ref, f"{PKG}/chia_agent/spec_policy.py"),
                 "spec_policy.py", "exec"), policy)
    policy_violations = policy["policy_violations"]
    for rel in FROZEN:
        dst = tree / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(git_show(ref, rel))
    for name in EDITABLE:
        src = spec_dir / name
        if not src.is_file():
            raise Reject("setup", f"spec dir has no {name}")
        text = src.read_text(encoding="utf-8")
        problems = policy_violations(name, text) + policy["doc_violations"](
            name, git_show(ref, f"{PKG}/{name}").decode("utf-8"), text)
        if problems:
            raise Reject("policy", "; ".join(problems))
        (tree / PKG / name).write_text(text, encoding="utf-8")
    # Anything else in the spec dir is ignored, not merged: only these two
    # files are the candidate.
    manifest = tree_manifest(tree)
    for f in tree.rglob("*"):
        if f.is_file():
            f.chmod(0o444)
    return manifest


def tree_manifest(tree: Path) -> dict:
    return {str(f.relative_to(tree)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(tree.rglob("*")) if f.is_file()}


def checkout_state() -> str:
    """Content of this checkout's tracked files (the evaluator included), as a
    hash of their diff against HEAD -- a file already dirty still counts."""
    diff = subprocess.run(["git", "diff", "HEAD", "--binary", "--", *CHECKOUT_WATCH],
                          cwd=REPO, capture_output=True).stdout
    return hashlib.sha256(diff).hexdigest()


def verify(tree: Path, manifest: dict, checkout: str, after: str):
    """Nothing the candidate ran may have changed the tree or the checkout."""
    now = tree_manifest(tree)
    if now != manifest:
        changed = sorted(k for k in set(now) | set(manifest)
                         if now.get(k) != manifest.get(k))
        raise Reject("tamper", f"after {after}, the evaluation tree changed: {changed}")
    if checkout_state() != checkout:
        raise Reject("tamper", f"after {after}, the checkout's tracked files changed")


def env_for(tree: Path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("TPU_")}
    env.update({
        # tree first (the candidate + frozen files), then this checkout for the
        # `allo` package and its in-tree mlir bindings.
        "PYTHONPATH": f"{tree}:{REPO}",
        "LLVM_BUILD_DIR": LLVM_BUILD_DIR,
        "OMP_NUM_THREADS": "8",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    return env


def sandboxed(cmd, work: Path, tree: Path):
    """`cmd` under bubblewrap: read-only everything, but `work` and a private
    /tmp; `tree` (inside `work`) read-only again; own PID namespace."""
    if not BWRAP:
        return cmd
    return [BWRAP, "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc",
            "--tmpfs", "/tmp", "--tmpfs", "/dev/shm",
            "--bind", str(work), str(work), "--ro-bind", str(tree), str(tree),
            "--unshare-pid", "--die-with-parent", "--", *cmd]


def run(cmd, cwd, env, timeout, work=None, tree=None, stdin=None):
    t = time.time()
    if work is not None:
        cmd = sandboxed(cmd, work, tree)
    p = subprocess.Popen(cmd, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, start_new_session=True,
                         stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL)
    try:
        out, _ = p.communicate(input=stdin, timeout=timeout)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        # The whole group: a deadlocked simulator (or vitis_hls under cosim)
        # must not outlive the verdict.
        os.killpg(p.pid, signal.SIGKILL)
        out, _ = p.communicate()
        out, rc = (out or "") + f"\nTIMEOUT after {timeout}s (deadlock?)", 124
    return rc, out, time.time() - t


def check_invariants(tree, env, work):
    code = ("import json; from examples.accelerator.tinytpu_vitis import "
            "microarch_isa as u; print(json.dumps({'T': u.T, 'MAXDIM': u.MAXDIM,"
            " 'IMEM_SIZE': u.IMEM_SIZE}))")
    rc, out, _ = run([ALLO_PYTHON, "-c", code], tree, env, 300, work, tree)
    if rc:
        raise Reject("import", out[-3000:])
    inv = json.loads(out.strip().splitlines()[-1])
    if inv["T"] != 4 or inv["MAXDIM"] != 16:
        raise Reject("invariant",
                     f"T={inv['T']} MAXDIM={inv['MAXDIM']}; the comparison is a "
                     f"4x4 array at MAXDIM 16 and both are frozen")
    return inv


def gate_runner_cmd(root: Path, check, args=()):
    """The command that runs a frozen check under `root`'s gate_runner.py."""
    return [ALLO_PYTHON, str(root / GATE_RUNNER), check, *args]


def vouch(check, root: Path, spawn, args=()):
    """The nonce-vouched gate call. ONE definition; `accept.py` imports this
    one rather than keeping a second copy, because it is the primitive that
    decides whether a candidate's gate really passed.

    `spawn(cmd, stdin)` runs `cmd` and returns `(rc, out, seconds)`; how it is
    sandboxed and logged is the caller's business. Everything
    security-relevant is here:

    * the nonce is minted per call and reaches the runner on **stdin** only --
      never the environment, the command line, or a file the candidate reads;
    * `vouched` is True only if the output carries
      `CHIA-GATE <check> OK <nonce>` as a WHOLE line, with this run's nonce.
      Nothing the candidate prints can produce it: the runner reads the nonce
      before the candidate is imported and prints the line only when the check
      RETURNED success;
    * the nonce is scrubbed from the output that is returned, so it never
      reaches a verdict, a log, or anything the agent can read.

    Returns `(vouched, rc, out, seconds)`."""
    nonce = secrets.token_hex(16)
    rc, out, sec = spawn(gate_runner_cmd(root, check, args), nonce + "\n")
    ok = rc == 0 and f"CHIA-GATE {check} OK {nonce}" in out.splitlines()
    return ok, rc, out.replace(nonce, "<nonce>"), sec


def vouched(check, tree, env, work, cwd, timeout, args=()):
    """Run a frozen check under gate_runner.py; (vouched, rc, out, seconds)."""
    return vouch(check, tree, lambda cmd, stdin: run(
        cmd, cwd, env, timeout, work, tree, stdin=stdin), args)


def gate(tree, env, work, verify_now):
    ok, rc, out, sec = vouched("bench_isa", tree, env, work, tree, GATE_TIMEOUT)
    verify_now("bench_isa")
    if not ok or not re.search(r"^  ALL EXACT$", out, re.M) or "FAILURES" in out:
        raise Reject("gate:bench_isa", out[-4000:])
    ok2, rc2, out2, sec2 = vouched("stress_isa", tree, env, work, tree, GATE_TIMEOUT)
    verify_now("stress_isa")
    # The runner vouches that stress_isa.main() returned 0. The count is
    # recorded, and must be internally consistent (n/n); it is not pinned,
    # because stress_isa skips the unrolled reference programs that do not fit
    # the candidate's imem, as it does on main.
    m = re.search(r"^  STRESS OK: (\d+)/(\d+) runs exact", out2, re.M)
    if not ok2 or not m or m.group(1) != m.group(2):
        raise Reject("gate:stress", out2[-4000:])
    # 3. Parametricity: the same candidate, rebuilt at other MAXDIMs, must
    # build, honour the parameter, and be exact (param_check.py).
    param, sec3 = {}, 0.0
    for cfg in PARAM_CONFIGS:
        ok3, rc3, out3, s3 = vouched("param_check", tree, dict(env, **cfg), work,
                                     tree, GATE_TIMEOUT)
        sec3 += s3
        tag = ",".join(f"{k}={v}" for k, v in cfg.items())
        verify_now(f"param_check {tag}")
        m3 = re.search(r"^  PARAM OK: (\d+)/(\d+) runs exact", out3, re.M)
        if not ok3 or not m3 or m3.group(1) != m3.group(2):
            raise Reject("gate:param", f"at {tag}:\n" + out3[-4000:])
        param[tag] = lines_with(out3, "PARAM OK")[0]
    return {"bench_isa": "ALL EXACT", "stress": lines_with(out2, "STRESS OK")[0],
            "stress_runs": int(m.group(1)), "param": param, "vouched": True,
            "seconds": round(sec + sec2 + sec3, 1)}


def lines_with(text, needle):
    return [l.strip() for l in text.splitlines() if needle in l]


def parse_synth(prj: Path):
    xml = prj / "out.prj/solution1/syn/report/csynth.xml"
    if not xml.exists():
        raise Reject("csynth", "no csynth.xml -- synthesis failed")
    root = ET.parse(xml).getroot()
    target = float(root.findtext(".//TargetClockPeriod"))
    est = float(root.findtext(".//EstimatedClockPeriod"))
    res = root.find(".//AreaEstimates/Resources")
    area = {k.lower(): int(res.findtext(k)) for k in
            ("BRAM_18K", "DSP", "FF", "LUT", "URAM")} if res is not None else {}
    if abs(target - TARGET_NS) > 1e-6:
        raise Reject("csynth", f"target clock {target} ns != frozen {TARGET_NS}")
    if est > TARGET_NS:
        raise Reject("timing", f"estimated clock {est} ns misses {TARGET_NS} ns; "
                               f"cycles at a clock the design cannot meet are "
                               f"not comparable")
    return {"target_ns": target, "estimated_ns": est, "area": area}


def check_memory_model(prj: Path):
    kernel = (prj / "kernel.cpp").read_text(errors="replace")
    pragmas = re.findall(r"#pragma HLS interface m_axi[^\n]*", kernel)
    if len(pragmas) != 4:
        raise Reject("memory-model", f"expected 4 m_axi ports, found {len(pragmas)}")
    bad = [p for p in pragmas if re.search(r"\blatency\s*=", p)]
    if bad or "config_interface" in kernel:
        raise Reject("memory-model", f"interface latency override in kernel.cpp: {bad}")
    for log in [prj / "csynth.log", *prj.glob("cosim_*.log"), prj / "run.tcl"]:
        if log.exists():
            text = log.read_text(errors="replace")
            if "m_axi_latency" in text or "random_stall" in text:
                raise Reject("memory-model", f"{log.name} changes the memory model")


def score(tree, env, work: Path, shapes, verify_now):
    prj = work / "isa_sweep.prj"
    # cosim.py (main since e620576d) puts its project next to itself by default,
    # which is the read-only tree here; TPU_PRJ is a path, not a memory-model
    # knob, and is the only TPU_* variable set.
    env = dict(env, TPU_SHAPES=",".join(shapes), TPU_PRJ=str(prj))
    ok, rc, out, sec = vouched("cosim", tree, env, work, work, COSIM_TIMEOUT)
    verify_now("cosim")
    if not ok:
        raise Reject("cosim", out[-4000:])
    synth = parse_synth(prj)
    check_memory_model(prj)
    cycles = {}
    for s in shapes:
        M, K, N = (int(x) for x in s.split("x"))
        pat = re.compile(rf"^\s*{M}x\s*{K}x\s*{N}\s+cycles=(\S+)\s+(.*)$")
        rows = [pat.match(l) for l in out.splitlines()]
        rows = [r for r in rows if r]
        if len(rows) != 1:
            raise Reject("cosim", f"shape {s} reported {len(rows)} times\n{out[-3000:]}")
        n, tb = rows[0].group(1), rows[0].group(2)
        want = f"TB {M}x{K}x{N} mismatches = 0 / {M * N}"
        if n == "None" or tb.strip() != want:
            raise Reject("cosim", f"{s}: cycles={n} tb='{tb}' (want '{want}')\n"
                                  f"{out[-3000:]}")
        log = prj / f"cosim_{M}x{K}x{N}.log"
        text = log.read_text(errors="replace") if log.exists() else ""
        if want not in text or "C/RTL co-simulation finished: PASS" not in text:
            raise Reject("cosim", f"{s}: its own cosim log lacks '{want}' + PASS")
        t = [int(x) for x in re.findall(r'RTL Simulation : \d+ / 1 \[n/a\] @ "(\d+)"', text)]
        if len(t) == 2:
            sim_cycles = (t[1] - t[0]) / (TARGET_NS * 1000)
            if abs(sim_cycles - int(n)) > SIMTIME_SLACK:
                raise Reject("cosim", f"{s}: report says {n} cycles, simulated "
                                      f"time says {sim_cycles:.0f}")
        cycles[s] = int(n)
    return cycles, synth, round(sec, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec-dir", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--shapes", default=",".join(SEARCH_SHAPES))
    ap.add_argument("--gate-only", action="store_true")
    a = ap.parse_args()
    shapes = [s for s in a.shapes.split(",") if s]
    started = time.time()
    result = {"ok": False, "frozen_ref": FROZEN_REF, "shapes": shapes,
              "measurement": "cosim: Vitis HLS 2023.2 csynth + xsim C/RTL cosim "
                             "(RTL), m_axi_latency default 0, one bit-exact TB "
                             "per shape"}
    try:
        bad = [s for s in shapes if s not in ALL_SHAPES]
        if bad:
            raise Reject("setup", f"shapes {bad} are not in the frozen SHAPES")
        work = a.work.resolve()
        spec = a.spec_dir.resolve()
        if (work == spec or work in spec.parents or spec in work.parents
                or work == REPO or work in REPO.parents
                or (work.is_relative_to(REPO)
                    and not work.is_relative_to(REPO / ".chia_scratch"))):
            # It is rmtree'd below: never the spec, the checkout, or above them.
            raise Reject("setup", f"work dir {work} overlaps the spec dir {spec} "
                                  f"or the checkout; it would be wiped")
        # Wiped every time: a stale cosim report must never be read as a result.
        if work.exists():
            shutil.rmtree(work)
        tree = work / "tree"
        tree.mkdir(parents=True)
        ref = resolve_ref(FROZEN_REF)
        result["frozen_ref"] = ref
        checkout = checkout_state()
        manifest = compose(spec, tree, ref)
        verify_now = lambda after: verify(tree, manifest, checkout, after)
        result["sandbox"] = bool(BWRAP)
        env = env_for(tree)
        result["invariants"] = check_invariants(tree, env, work)
        verify_now("the import check")
        result["gate"] = gate(tree, env, work, verify_now)
        if not a.gate_only:
            cyc, synth, sec = score(tree, env, work, shapes, verify_now)
            result.update(cycles=cyc, total_cycles=sum(cyc.values()),
                          synth=synth, cosim_seconds=sec)
        result["ok"] = True
    except Reject as r:
        result.update(stage=r.stage, detail=r.detail[-4000:])
    result["seconds"] = round(time.time() - started, 1)
    detail = result.pop("detail", None)
    if detail:
        print(detail)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
