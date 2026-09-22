"""Acceptance of a claimed winner: a CLEAN checkout, all five shapes, bit-exact.

The search scores two shapes in a composed evaluation tree. That is a search
convenience, not evidence. A claim is accepted only by this script:

1. a fresh `git worktree` of `--ref` (default HEAD) under `.chia_scratch/`,
2. that checkout's own `mlir/` bindings built in-tree against $LLVM_BUILD_DIR
   (~40 s; no symlink into any other checkout),
3. **the control**: `cosim.py` on the committed design, all five shapes, in
   that same worktree and off that same build, while the candidate's diff is
   still nowhere on disk -- the number every candidate in the run is compared
   against (`--control` reuses an earlier control run's, so one measurement
   serves a whole run),
4. the candidate diff applied with `git apply` (it may touch only
   microarch_isa.py and isa_dsl.py -- anything else is refused),
5. `bench_isa.py`, main's `stress_isa.py` (the correctness gate),
   `cosim.py`, and `cosim.py` again with `TPU_TB=stress` (the RTL correctness
   testbench; `--no-rtl-stress` skips it), with no `TPU_*` variable set except `TPU_PRJ` (where the Vitis
   project goes) -- so all five SHAPES, default memory model -- each run from
   that checkout under `chia_agent/gate_runner.py`, which vouches for each
   verdict with a per-run nonce instead of trusting printed lines,
6. results and logs copied into `--out` BEFORE the worktree is removed.

`ok` means *correct*: bit-exact at all five shapes, ALL EXACT, stress_isa,
clock. Whether it is a *win* is a separate field, `claim`, and it is decided
against the control of step 3 -- a number produced minutes earlier on this
machine, not one published at some past commit:

    win          ok, and the five-shape total is lower than the control's
    not-better   ok, and it is not -- correct but no improvement; nothing to claim
    rejected     not ok
    no-control   the control could not be measured or reused; nothing is
                 comparable, and no result of this run means anything

The control's own five testbenches are bit-exact per shape, which is what the
comparison needs; the committed design's functional gates are main's business
(`reproduce.sh`), not re-litigated here. `control.RECORDED` and the published
numbers are a CROSS-CHECK on the measured control, never the control itself: a
disagreement prints a banner and exits 3, because it means either the tools
moved or the design changed behaviour.

    python accept.py --diff RUN/worker/best.diff --out RUN/accept-worker
    python accept.py --out RUN/control                  # no diff: the control
    python accept.py --diff D --out O --control RUN/control/accept.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent
REPO = AGENT_DIR.parents[3]
PKG = "examples/accelerator/tinytpu_vitis"
sys.path.insert(0, str(AGENT_DIR))
import control  # noqa: E402
#: The nonce-vouched gate call has ONE definition, in evaluate.py, and this
#: script imports it. It used to exist here as a second copy; a
#: security-critical primitive that can drift between two copies is the one
#: kind of duplication this harness cannot afford.
from evaluate import vouch  # noqa: E402
ALLO_PYTHON = os.environ.get(
    "TINYTPU_ALLO_PYTHON", "/home/sk3463/miniconda3/envs/allo/bin/python")
LLVM_BUILD_DIR = os.environ.get(
    "LLVM_BUILD_DIR", "/home/sk3463/llvm-allo-6b09f739/build")
ENV_BIN = str(Path(ALLO_PYTHON).parent)

BWRAP = shutil.which("bwrap")
#: bench_isa / stress on the unmodified design take ~5-10 s each; a deadlocked
#: candidate would otherwise hold the acceptance for the full cosim timeout.
GATE_TIMEOUT = 600
COSIM_ROW = r"^\s*(\d+)x\s*(\d+)x\s*(\d+)\s+cycles=(\S+)\s+(.*)$"


class NoControl(SystemExit):
    """No control measurement, so nothing this run measured is comparable."""


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
    """`evaluate.vouch` with this script's sandbox and logging: the check runs
    under the WORKTREE's gate_runner.py (i.e. from git at --ref), and is passed
    only on its nonce line. Returns (ok, rc, out, s)."""
    ok, rc, out, sec = vouch(
        check, wt,
        lambda cmd, stdin: sh(boxed(cmd, writable), cwd, env, None, timeout,
                              stdin=stdin))
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


def build_bindings(wt: Path, env, out: Path) -> str:
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
    assert o.strip().startswith(str(wt)), o
    return o.strip()


def cosim_pass(wt: Path, env, work: Path, out: Path, prefix=""):
    """One nonce-vouched five-shape cosim with its project inside `work`.

    Returns ({vouched, rows, estimated_ns, seconds}, the run's output)."""
    work.mkdir(parents=True, exist_ok=True)
    prj = work / "isa_sweep.prj"
    ok, rc, text, sec = vouched("cosim", wt, work, dict(env, TPU_PRJ=str(prj)),
                                out / f"{prefix}cosim.log", work)
    for f in prj.glob("cosim_*.log"):
        shutil.copy2(f, out / f"{prefix}{f.name}")
    est = None
    xml = prj / "out.prj/solution1/syn/report/csynth.xml"
    if xml.exists():
        shutil.copy2(xml, out / f"{prefix}csynth.xml")
        est = float(re.search(r"<EstimatedClockPeriod>([\d.]+)",
                              xml.read_text()).group(1))
    rows = {f"{m}x{k}x{n}": {"cycles": None if c == "None" else int(c),
                             "tb": tb.strip()}
            for m, k, n, c, tb in re.findall(COSIM_ROW, text, re.M)}
    return {"vouched": ok, "rows": rows, "estimated_ns": est,
            "seconds": sec}, text


def five_exact(rows: dict) -> bool:
    return len(rows) == 5 and all(
        v["cycles"] is not None
        and v["tb"] == (f"TB {s} mismatches = 0 / "
                        f"{int(s.split('x')[0]) * int(s.split('x')[2])}")
        for s, v in rows.items())


def measure_control(wt: Path, env, out: Path, ref: str, design: dict, tracked,
                    keep: bool) -> dict:
    """The committed design, measured in THIS worktree off THIS build, before
    the candidate's diff exists on disk. That ordering is what a candidate
    cannot get past: no line of it has been written, let alone run, when these
    cycles are measured."""
    if tracked().strip():
        raise NoControl(f"refusing to measure the control on a checkout that is "
                        f"not pristine:\n{tracked()}")
    work = wt / ".control"
    passed, _ = cosim_pass(wt, env, work, out, "control-")
    if tracked().strip():
        raise NoControl(f"the control measurement changed tracked files:\n{tracked()}")
    if not keep:
        shutil.rmtree(work, ignore_errors=True)   # hundreds of MB, already read
    if not (passed["vouched"] and five_exact(passed["rows"])):
        raise NoControl(f"the control did not measure: vouched="
                        f"{passed['vouched']}, rows={passed['rows']}; see "
                        f"{out / 'control-cosim.log'}")
    return control.record(
        cycles={s: v["cycles"] for s, v in passed["rows"].items()},
        blobs=design, ref=ref, estimated_ns=passed["estimated_ns"],
        seconds=passed["seconds"], vouched=True, pristine_tree=True,
        source=f"measured in this run from git at {ref[:8]}, before the "
               f"candidate diff was applied")


def reuse_control(path: Path, design: dict) -> dict:
    """An earlier control run's record, for the same design, or nothing."""
    try:
        rec = (json.loads(path.read_text()) or {}).get("control")
    except (OSError, json.JSONDecodeError) as why:
        raise NoControl(f"cannot read the control in {path}: {why}")
    problems = control.unusable(rec, design)
    if problems:
        raise NoControl(f"refusing the control in {path}: {'; '.join(problems)}")
    return dict(rec, source=f"reused from {path}: {rec['source']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--diff", type=Path)
    ap.add_argument("--ref", default="HEAD")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--no-rtl-stress", action="store_true",
                    help="skip cosim.py's TPU_TB=stress correctness testbench")
    ap.add_argument("--control", type=Path,
                    help="an earlier control run's accept.json: reuse its measured "
                         "control (same design only) instead of measuring again")
    a = ap.parse_args()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    ref = subprocess.run(["git", "rev-parse", a.ref], cwd=REPO, capture_output=True,
                         text=True, check=True).stdout.strip()
    wt = REPO / ".chia_scratch" / f"accept-{out.name}-{int(time.time())}"
    result = {"ref": ref, "diff": str(a.diff) if a.diff else None, "ok": False,
              "design": control.blobs(ref),
              "measurement": "cosim: Vitis HLS 2023.2 + xsim C/RTL cosim (RTL), "
                             "clean checkout, all five SHAPES, TPU_* unset"}
    subprocess.run(["git", "worktree", "add", "--detach", str(wt), ref], cwd=REPO,
                   check=True, capture_output=True)
    try:
        tracked = lambda: sh(["git", "status", "--porcelain",
                              "--untracked-files=no"], wt)[1]
        env = {k: v for k, v in os.environ.items() if not k.startswith("TPU_")}
        env.update(PATH=f"{ENV_BIN}:{env['PATH']}", LLVM_BUILD_DIR=LLVM_BUILD_DIR,
                   OMP_NUM_THREADS="8", PYTHONPATH=str(wt),
                   PYTHONDONTWRITEBYTECODE="1")
        result["sandbox"] = bool(BWRAP)
        # What the diff may touch, before anything is measured: a diff that is
        # refused anyway must not cost a five-shape cosim first.
        if a.diff:
            diff = a.diff.read_text()
            touched = set(re.findall(r"^\+\+\+ b/(\S+)", diff, re.M)) | set(
                re.findall(r"^--- a/(\S+)", diff, re.M))
            if not touched or not touched <= {"microarch_isa.py", "isa_dsl.py"}:
                raise SystemExit(f"refusing: diff touches {sorted(touched)}")
            (out / "candidate.diff").write_text(diff)
        result["allo_resolves_to"] = build_bindings(wt, env, out)

        # The control next, off this build, with the candidate still nowhere
        # on disk. A no-diff run IS the control: its own measurement below.
        ctl = None
        if a.control:
            ctl = reuse_control(a.control, result["design"])
        elif a.diff:
            ctl = measure_control(wt, env, out, ref, result["design"], tracked,
                                  a.keep)
        if ctl:
            result["control"] = ctl

        if a.diff:
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
                    policy["policy_violations"](f, (wt / PKG / f).read_text())
                    + policy["doc_violations"](f, subprocess.run(
                        ["git", "show", f"{ref}:{PKG}/{f}"], cwd=REPO,
                        capture_output=True, text=True, check=True).stdout,
                        (wt / PKG / f).read_text())]
        result["policy"] = problems
        if problems:
            raise SystemExit(f"refusing: spec policy: {problems}")
        rc, o, _ = sh(["git", "status", "--porcelain"], wt)
        result["checkout_status"] = o.strip().splitlines()

        cos = wt / ".cosim"
        clean = tracked()

        def untouched(stage):
            if tracked() != clean:
                result["tamper"] = f"tracked files changed during {stage}"
                raise SystemExit(result["tamper"])

        cos.mkdir()
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
        # Parametricity, as in the search's gate: rebuilt at other MAXDIMs,
        # exact (param_check.py, frozen at --ref).
        result["param"], param_ok = {}, True
        for cfg in ({"TPU_MAXDIM": "8"}, {"TPU_MAXDIM": "12"}):
            tag = ",".join(f"{k}={v}" for k, v in cfg.items())
            okp, rcp, op, secp = vouched("param_check", wt, wt, dict(env, **cfg),
                                         out / f"param_check_{tag}.log", cos,
                                         timeout=GATE_TIMEOUT)
            untouched(f"param_check {tag}")
            mp = re.search(r"^  PARAM OK: (\d+)/(\d+) runs exact", op, re.M)
            good = okp and mp is not None and mp.group(1) == mp.group(2)
            param_ok &= good
            result["param"][tag] = {"ok": good, "seconds": secp,
                                    "line": [l.strip() for l in op.splitlines()
                                             if "PARAM " in l][-1:]}
        # cosim.py puts its project next to itself by default; keep it in the
        # writable .cosim directory instead.
        candidate, o3 = cosim_pass(wt, env, cos, out)
        untouched("cosim")
        ok3, result["cosim"] = candidate["vouched"], candidate["rows"]
        result["cosim_seconds"] = candidate["seconds"]
        if candidate["estimated_ns"] is not None:
            result["estimated_ns"] = candidate["estimated_ns"]
        # The RTL correctness testbench (main's cosim.py TPU_TB=stress): several
        # calls on one RTL instance, corner/full/boundary/mid operands, C
        # prefilled and compared in full, plus a vector program. It is what
        # catches an RTL-only failure the simulator cannot, e.g. a dependence
        # pragma that is false at a short read-after-write distance.
        rtl_ok = True
        if not a.no_rtl_stress:
            cos2 = wt / ".cosim_stress"
            cos2.mkdir()
            ok4, rc4, o4, sec4 = vouched(
                "cosim", wt, cos2,
                dict(env, TPU_PRJ=str(cos2 / "isa_sweep.prj"), TPU_TB="stress"),
                out / "cosim_stress.log", cos2)
            untouched("cosim stress")
            lines4 = [l.strip() for l in o4.splitlines()
                      if re.match(r"\s*\d+x\s*\d+x\s*\d+\s+cycles=", l)]
            rtl_ok = (ok4 and len(lines4) == 5
                      and all(re.search(r"stress mismatches = 0 over \d+ calls$", l)
                              for l in lines4)
                      and "COSIM OK (testbench=stress)" in o4)
            result["cosim_rtl_stress"] = {"vouched": ok4, "ok": rtl_ok,
                                          "shapes": lines4, "seconds": sec4}
        exact = five_exact(result["cosim"])
        result["ok"] = (ok1 and result["bench_isa"]["all_exact"]
                        and ok2 and ok3 and param_ok and exact
                        and result.get("estimated_ns", 99) <= 3.33
                        and rtl_ok)
        if ctl is None:
            ctl = control.record(
                cycles={s: v["cycles"] for s, v in result["cosim"].items()},
                blobs=result["design"], ref=ref, seconds=candidate["seconds"],
                estimated_ns=candidate["estimated_ns"], vouched=ok3,
                pristine_tree=not result["checkout_status"],
                source="no diff was applied: this run's own measurement is "
                       "the control")
            ctl["problems"] = control.unusable(ctl, result["design"])
            result["control"] = ctl
        # The cross-check is on the CONTROL, so it is reported even for a
        # candidate that was rejected: the tools may have moved under both.
        if not ctl.get("problems"):
            result["crosscheck"] = control.crosscheck(ctl["cycles"],
                                                      result["design"])
        if not result["ok"]:
            result["claim"] = "rejected"
        elif ctl.get("problems"):
            result["claim"] = "no-control"
        else:
            got = {s: v["cycles"] for s, v in result["cosim"].items()}
            result["delta"] = {s: got[s] - ctl["cycles"][s] for s in ctl["cycles"]}
            result["delta_total"] = sum(result["delta"].values())
            result["claim"] = "win" if result["delta_total"] < 0 else "not-better"
    except NoControl:
        result["claim"] = "no-control"
        raise
    finally:
        (out / "accept.json").write_text(json.dumps(result, indent=1))
        print(json.dumps(result, indent=1))
        print(control.banner(result.get("crosscheck") or {}), end="")
        # Results are already under --out; only now may the checkout go.
        if not a.keep:
            subprocess.run(["git", "worktree", "remove", "--force", str(wt)],
                           cwd=REPO, capture_output=True)
    if (result.get("crosscheck") or {}).get("status") == "DISAGREES":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
