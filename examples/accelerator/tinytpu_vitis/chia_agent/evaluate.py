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
      SHAPES, the testbench generator,      tinytpu_vitis/isa_dsl.py
      the numpy golden reference,
      every Vitis TCL setting
    tinytpu_vitis/bench_isa.py
    chia_agent/stress.py
    examples/__init__.py

The two tiers:

1. **gate** -- `bench_isa.py` must exit 0 and print `ALL EXACT`; `stress.py`
   (full-range operands, sentinel-filled C, extra shapes) must pass. Functional,
   on Allo's simulator, seconds.
2. **score** -- `cosim.py` (Vitis HLS 2023.2 csynth + xsim C/RTL cosim), one
   testbench per shape, each bit-exact against numpy. The score is the SUM of
   cosim cycles over the requested shapes. It is an RTL measurement.

The memory model is not the candidate's to choose: every `TPU_*` environment
variable is scrubbed before `cosim.py` runs, so `-m_axi_latency` stays at its
default 0 and `-random_stall` stays off, and the generated `kernel.cpp` / TCL are
checked for interface-latency overrides afterwards.

Independent checks on the result, none of which trust the candidate's process:
T == 4 and MAXDIM == 16 (the matched-array premise of COMPARISON.md); each
requested shape reported exactly once; each shape's own cosim log carries
`mismatches = 0 / M*N` and a PASS; the cycle count agrees with that log's
simulated time; the csynth estimated clock meets the 3.33 ns target. The work
directory is wiped first, so a stale report can never be read as a result.

Prints one JSON line (last line of stdout) and exits 0 iff the candidate passed.

    python evaluate.py --spec-dir DIR --work DIR [--shapes 4x4x4,16x16x16] [--gate-only]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent
REPO = AGENT_DIR.parents[2]
PKG = "examples/accelerator/tinytpu_vitis"

#: Frozen files are read from this commit (resolved to a hash per run), i.e.
#: from what is COMMITTED, never from the working tree. Only a person, in a
#: commit, can change them.
FROZEN_REF = os.environ.get("CHIA_FROZEN_REF", "HEAD")
#: main's commit at branch time. The design's own evaluator -- cosim.py and
#: bench_isa.py -- must be byte-identical to it, so this branch cannot drift
#: from how main measures the design.
MAIN_BASE = "e2451b81"
DESIGN_EVALUATOR = [f"{PKG}/cosim.py", f"{PKG}/bench_isa.py"]
FROZEN = [
    "examples/__init__.py",
    *DESIGN_EVALUATOR,
    f"{PKG}/chia_agent/stress.py",
]
EDITABLE = ("microarch_isa.py", "isa_dsl.py")
ALL_SHAPES = ["4x4x4", "8x8x8", "12x12x12", "16x16x8", "16x16x16"]
SEARCH_SHAPES = ["4x4x4", "16x16x16"]
TARGET_NS = 3.33
#: xsim's transaction window runs a few cycles past HLS's latency count.
SIMTIME_SLACK = 12

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
    """Evaluation tree = frozen files from git + the candidate's two files."""
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
        problems = policy_violations(name, text)
        if problems:
            raise Reject("policy", "; ".join(problems))
        (tree / PKG / name).write_text(text, encoding="utf-8")
    # Anything else in the spec dir is ignored, not merged: only these two
    # files are the candidate.


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


def run(cmd, cwd, env, timeout):
    t = time.time()
    try:
        p = subprocess.run(cmd, cwd=cwd, env=env, text=True,
                           capture_output=True, timeout=timeout)
        out, rc = p.stdout + p.stderr, p.returncode
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") if isinstance(e.stdout, str) else ""
        out, rc = out + f"\nTIMEOUT after {timeout}s", 124
    return rc, out, time.time() - t


def check_invariants(tree, env):
    code = ("import json; from examples.accelerator.tinytpu_vitis import "
            "microarch_isa as u; print(json.dumps({'T': u.T, 'MAXDIM': u.MAXDIM,"
            " 'IMEM_SIZE': u.IMEM_SIZE}))")
    rc, out, _ = run([ALLO_PYTHON, "-c", code], tree, env, 300)
    if rc:
        raise Reject("import", out[-3000:])
    inv = json.loads(out.strip().splitlines()[-1])
    if inv["T"] != 4 or inv["MAXDIM"] != 16:
        raise Reject("invariant",
                     f"T={inv['T']} MAXDIM={inv['MAXDIM']}; the comparison is a "
                     f"4x4 array at MAXDIM 16 and both are frozen")
    return inv


def gate(tree, env):
    bench = str(tree / PKG / "bench_isa.py")
    rc, out, sec = run([ALLO_PYTHON, bench], tree, env, 900)
    lines = out.strip().splitlines()
    if rc or not lines or lines[-1].strip() != "ALL EXACT" or "FAILURES" in out:
        raise Reject("gate:bench_isa", out[-4000:])
    stress = str(tree / PKG / "chia_agent" / "stress.py")
    rc2, out2, sec2 = run([ALLO_PYTHON, stress], tree, env, 900)
    if rc2 or "STRESS OK" not in out2:
        raise Reject("gate:stress", out2[-4000:])
    return {"bench_isa": "ALL EXACT", "stress": lines_with(out2, "STRESS OK")[0],
            "seconds": round(sec + sec2, 1)}


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


def score(tree, env, work: Path, shapes):
    cosim = str(tree / PKG / "cosim.py")
    env = dict(env, TPU_SHAPES=",".join(shapes))
    rc, out, sec = run([ALLO_PYTHON, cosim], work, env, 3600)
    prj = work / "isa_sweep.prj"
    if rc:
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
        # Wiped every time: a stale cosim report must never be read as a result.
        if work.exists():
            shutil.rmtree(work)
        tree = work / "tree"
        tree.mkdir(parents=True)
        ref = resolve_ref(FROZEN_REF)
        result["frozen_ref"] = ref
        compose(a.spec_dir.resolve(), tree, ref)
        env = env_for(tree)
        result["invariants"] = check_invariants(tree, env)
        result["gate"] = gate(tree, env)
        if not a.gate_only:
            cyc, synth, sec = score(tree, env, work, shapes)
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
