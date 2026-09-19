"""Acceptance of a claimed winner: a CLEAN checkout, all five shapes, bit-exact.

The search scores two shapes in a composed evaluation tree. That is a search
convenience, not evidence. A claim is accepted only by this script:

1. a fresh `git worktree` of `--ref` (default HEAD) under `.chia_scratch/`,
2. the candidate diff applied with `git apply` (it may touch only
   microarch_isa.py and isa_dsl.py -- anything else is refused),
3. that checkout's own `mlir/` bindings built in-tree against $LLVM_BUILD_DIR
   (~40 s; no symlink into any other checkout),
4. `bench_isa.py`, main's `stress_isa.py` (the correctness gate) and
   `cosim.py` with no `TPU_*` variable set except `TPU_PRJ` (where the Vitis
   project goes) -- so all five SHAPES, default memory model -- each run from
   that checkout under `chia_agent/gate_runner.py`, which vouches for each
   verdict with a per-run nonce instead of trusting printed lines,
5. results and logs copied into `--out` BEFORE the worktree is removed.

`ok` means *correct*: bit-exact at all five shapes, ALL EXACT, stress_isa,
clock.
Whether it is a *win* is a separate field, `claim`, against the unmodified
design's five cosim numbers at `--ref`:

    win          ok, and the five-shape total is lower
    not-better   ok, and it is not -- correct but no improvement; nothing to claim
    rejected     not ok
    no-baseline  ok, but no baseline is known for this design (pass --baseline)

The baseline is recorded below per design (keyed by the git blob ids of the two
editable files), or read from a control run's `accept.json` via `--baseline`.

    python accept.py --diff RUN/worker/best.diff --out RUN/accept-worker
    python accept.py --out RUN/accept-baseline          # no diff: the control
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import time
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent
REPO = AGENT_DIR.parents[3]
PKG = "examples/accelerator/tinytpu_vitis"
ALLO_PYTHON = os.environ.get(
    "TINYTPU_ALLO_PYTHON", "/home/sk3463/miniconda3/envs/allo/bin/python")
LLVM_BUILD_DIR = os.environ.get(
    "LLVM_BUILD_DIR", "/home/sk3463/llvm-allo-6b09f739/build")
ENV_BIN = str(Path(ALLO_PYTHON).parent)
#: Control runs of the unmodified design: (microarch_isa.py blob, isa_dsl.py
#: blob) -> five-shape cosim cycles. Measured by this script with no --diff.
BASELINES = {
    # main @ e2451b81 (the branch point before the rebase)
    ("ac5174fe43f449e9b0b1693cda1aff6c74ab71d3",
     "10de511a2ddf7a8fa8fbf8d0de588ddbb690290f"):
        {"4x4x4": 252, "8x8x8": 383, "12x12x12": 591, "16x16x8": 667,
         "16x16x16": 919},
    # main @ e620576d (check_program in assemble(), docstring fixes)
    ("cb26d5683338184f02bfcb6be13bc1ace4e5e3e9",
     "e3b55230b4c6308dfa5e7d729d49e6056040d663"):
        {"4x4x4": 252, "8x8x8": 383, "12x12x12": 591, "16x16x8": 667,
         "16x16x16": 919},
}


BWRAP = shutil.which("bwrap")
#: bench_isa / stress on the unmodified design take ~5-10 s each; a deadlocked
#: candidate would otherwise hold the acceptance for the full cosim timeout.
GATE_TIMEOUT = 600


def sh(cmd, cwd, env=None, log=None, timeout=7200, stdin=None):
    t = time.time()
    p = subprocess.Popen(cmd, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, start_new_session=True,
                         stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL)
    try:
        out, _ = p.communicate(input=stdin, timeout=timeout)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)   # the whole group, simulator included
        out, _ = p.communicate()
        out, rc = (out or "") + f"\nTIMEOUT after {timeout}s", 124
    if log:
        Path(log).write_text(out)
    return rc, out, round(time.time() - t, 1)


def vouched(check, wt, cwd, env, log, writable, timeout=7200):
    """As evaluate.py: the check under gate_runner.py (from the checkout, i.e.
    from git at --ref), passed only on its nonce line. Returns (ok, rc, out, s)."""
    nonce = secrets.token_hex(16)
    rc, out, sec = sh(boxed([ALLO_PYTHON, str(wt / PKG / "chia_agent" / "gate_runner.py"),
                             check], writable), cwd, env, None, timeout,
                      stdin=nonce + "\n")
    ok = rc == 0 and f"CHIA-GATE {check} OK {nonce}" in out.splitlines()
    out = out.replace(nonce, "<nonce>")
    Path(log).write_text(out)
    return ok, rc, out, sec


def boxed(cmd, writable: Path):
    """As evaluate.py: candidate-importing processes see a read-only
    filesystem except `writable` and a private /tmp."""
    if not BWRAP:
        return cmd
    return [BWRAP, "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc",
            "--tmpfs", "/tmp", "--tmpfs", "/dev/shm",
            "--bind", str(writable), str(writable),
            "--unshare-pid", "--die-with-parent", "--", *cmd]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--diff", type=Path)
    ap.add_argument("--ref", default="HEAD")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--baseline", type=Path,
                    help="a control run's accept.json (default: recorded BASELINES)")
    a = ap.parse_args()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    ref = subprocess.run(["git", "rev-parse", a.ref], cwd=REPO, capture_output=True,
                         text=True, check=True).stdout.strip()
    wt = REPO / ".chia_scratch" / f"accept-{out.name}-{int(time.time())}"
    result = {"ref": ref, "diff": str(a.diff) if a.diff else None, "ok": False,
              "measurement": "cosim: Vitis HLS 2023.2 + xsim C/RTL cosim (RTL), "
                             "clean checkout, all five SHAPES, TPU_* unset"}
    subprocess.run(["git", "worktree", "add", "--detach", str(wt), ref], cwd=REPO,
                   check=True, capture_output=True)
    try:
        if a.diff:
            diff = a.diff.read_text()
            touched = set(re.findall(r"^\+\+\+ b/(\S+)", diff, re.M)) | set(
                re.findall(r"^--- a/(\S+)", diff, re.M))
            if not touched or not touched <= {"microarch_isa.py", "isa_dsl.py"}:
                raise SystemExit(f"refusing: diff touches {sorted(touched)}")
            (out / "candidate.diff").write_text(diff)
            rc, o, _ = sh(["git", "apply", f"--directory={PKG}", "-p1",
                           str(a.diff.resolve())], wt)
            if rc:
                raise SystemExit(f"git apply failed:\n{o}")
        # The spec policy, from git at the ref, on the patched files -- the same
        # check the search applied, which a hand-made diff has not been through.
        policy = {"__name__": "spec_policy"}
        exec(compile(subprocess.run(
            ["git", "show", f"{ref}:{PKG}/chia_agent/spec_policy.py"], cwd=REPO,
            capture_output=True, check=True).stdout, "spec_policy.py", "exec"), policy)
        problems = [p for f in ("microarch_isa.py", "isa_dsl.py") for p in
                    policy["policy_violations"](f, (wt / PKG / f).read_text())]
        result["policy"] = problems
        if problems:
            raise SystemExit(f"refusing: spec policy: {problems}")
        rc, o, _ = sh(["git", "status", "--porcelain"], wt)
        result["checkout_status"] = o.strip().splitlines()
        tracked = lambda: sh(["git", "status", "--porcelain",
                              "--untracked-files=no"], wt)[1]

        env = {k: v for k, v in os.environ.items() if not k.startswith("TPU_")}
        env.update(PATH=f"{ENV_BIN}:{env['PATH']}", LLVM_BUILD_DIR=LLVM_BUILD_DIR,
                   OMP_NUM_THREADS="8", PYTHONPATH=str(wt),
                   PYTHONDONTWRITEBYTECODE="1")
        rc, o, sec = sh(["cmake", "-G", "Ninja", "-S", "mlir", "-B", "mlir/build",
                         f"-DMLIR_DIR={LLVM_BUILD_DIR}/lib/cmake/mlir",
                         f"-DPython3_EXECUTABLE={ALLO_PYTHON}",
                         f"-DPython_EXECUTABLE={ALLO_PYTHON}",
                         "-DMLIR_BINDINGS_PYTHON_NB_DOMAIN=allo"], wt, env,
                        out / "cmake.log")
        if rc:
            raise SystemExit("cmake failed; see cmake.log")
        rc, o, sec = sh(["ninja", "-C", "mlir/build", "-j", "48"], wt, env,
                        out / "ninja.log")
        if rc:
            raise SystemExit("ninja failed; see ninja.log")
        rc, o, _ = sh([ALLO_PYTHON, "-c", "import allo,os;print(os.path.realpath("
                       "allo.__file__))"], "/", env)
        result["allo_resolves_to"] = o.strip()
        assert o.strip().startswith(str(wt)), o

        cos = wt / ".cosim"
        cos.mkdir()
        result["sandbox"] = bool(BWRAP)
        clean = tracked()

        def untouched(stage):
            if tracked() != clean:
                result["tamper"] = f"tracked files changed during {stage}"
                raise SystemExit(result["tamper"])

        ok1, rc, o, sec = vouched("bench_isa", wt, wt, env, out / "bench_isa.log", cos,
                                  timeout=GATE_TIMEOUT)
        untouched("bench_isa")
        result["bench_isa"] = {"rc": rc, "vouched": ok1, "seconds": sec,
                               "all_exact": bool(re.search(r"^  ALL EXACT$", o, re.M))}
        ok2, rc2, o2, sec2 = vouched("stress_isa", wt, wt, env, out / "stress_isa.log",
                                     cos, timeout=GATE_TIMEOUT)
        untouched("stress_isa")
        result["stress_isa"] = {"rc": rc2, "vouched": ok2, "seconds": sec2,
                                "line": [l.strip() for l in o2.splitlines()
                                         if "STRESS OK" in l or "STRESS FAILED" in l][-1:]}
        # cosim.py puts its project next to itself by default; keep it in the
        # writable .cosim directory instead.
        ok3, rc3, o3, sec3 = vouched("cosim", wt, cos,
                                     dict(env, TPU_PRJ=str(cos / "isa_sweep.prj")),
                                     out / "cosim.log", cos)
        untouched("cosim")
        rows = re.findall(r"^\s*(\d+)x\s*(\d+)x\s*(\d+)\s+cycles=(\S+)\s+(.*)$", o3, re.M)
        result["cosim"] = {f"{m}x{k}x{n}": {"cycles": None if c == "None" else int(c),
                                            "tb": tb.strip()}
                           for m, k, n, c, tb in rows}
        result["cosim_seconds"] = sec3
        prj = cos / "isa_sweep.prj"
        for f in prj.glob("cosim_*.log"):
            shutil.copy2(f, out / f.name)
        xml = prj / "out.prj/solution1/syn/report/csynth.xml"
        if xml.exists():
            shutil.copy2(xml, out / "csynth.xml")
            t = xml.read_text()
            result["estimated_ns"] = float(re.search(
                r"<EstimatedClockPeriod>([\d.]+)", t).group(1))
        exact = all(v["tb"] == f"TB {s} mismatches = 0 / "
                    f"{int(s.split('x')[0]) * int(s.split('x')[2])}"
                    and v["cycles"] is not None for s, v in result["cosim"].items())
        result["ok"] = (ok1 and result["bench_isa"]["all_exact"]
                        and ok2 and ok3 and len(result["cosim"]) == 5
                        and exact and result.get("estimated_ns", 99) <= 3.33)
        if a.baseline:
            base = {s: v["cycles"] for s, v in
                    json.loads(a.baseline.read_text())["cosim"].items()}
        else:
            blobs = tuple(subprocess.run(
                ["git", "rev-parse", f"{ref}:{PKG}/{f}"], cwd=REPO,
                capture_output=True, text=True).stdout.strip()
                for f in ("microarch_isa.py", "isa_dsl.py"))
            base = BASELINES.get(blobs)
        result["baseline"] = base
        if not result["ok"]:
            result["claim"] = "rejected"
        elif not base or set(base) != set(result["cosim"]):
            result["claim"] = "no-baseline"
        else:
            got = {s: v["cycles"] for s, v in result["cosim"].items()}
            result["delta"] = {s: got[s] - base[s] for s in base}
            result["delta_total"] = sum(result["delta"].values())
            result["claim"] = "win" if result["delta_total"] < 0 else "not-better"
    finally:
        (out / "accept.json").write_text(json.dumps(result, indent=1))
        print(json.dumps(result, indent=1))
        # Results are already under --out; only now may the checkout go.
        if not a.keep:
            subprocess.run(["git", "worktree", "remove", "--force", str(wt)],
                           cwd=REPO, capture_output=True)


if __name__ == "__main__":
    main()
