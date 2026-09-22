# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`tests/limits/` verdicts, against a recorded baseline. FROZEN.

`tests/limits/` is the register of measured Allo limitations as executable
repros. Each file prints one verdict line per item -- `[item N] REPRODUCES`,
`[item N] FIXED`, `[item N] SKIPPED` -- and, deliberately, does NOT assert: it
is a repro, not a test. So pytest cannot run them and the gate has to read the
verdict lines.

That makes this corpus the most interesting gate in the ladder for a
`maintaining` candidate, because it is bidirectional:

  * a verdict going REPRODUCES -> FIXED is exactly what the loop is for, and is
    reported as `fixed` -- the strongest evidence a candidate can produce that
    it removed a real limitation rather than tuned a number;
  * a verdict going FIXED -> REPRODUCES is a REGRESSION and blocks the
    candidate: a limitation that a fork commit closed has been reopened;
  * a file that stops producing a verdict at all (crash, timeout, import
    error) blocks too. That is how a repro is silently disabled.

The items are run one subprocess per file with a timeout, for the same reason
`suite_runner.py` does: these repros deliberately provoke MLIR aborts,
frontend `sys.exit(1)` (issue #30) and simulator deadlocks.

`item15_csim_declaration_order.py` is the only one that needs `vitis_hls`, and
it can hang twice for 240 s by design (it is the silent-hang item). It is in
`SLOW` and runs at acceptance only.

    python limits_runner.py --record chia_abstraction/baseline/limits.json
    python limits_runner.py --against chia_abstraction/baseline/limits.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
LIMITS = REPO / "tests" / "limits"

#: Needs Vitis and can hang twice by design. Acceptance tier only.
SLOW = ("item15_csim_declaration_order.py",)
TIMEOUT = 900
VERDICT = re.compile(r"^\s*\[(?:item|limitation)\s*([A-Za-z0-9._-]+)\]\s+"
                     r"(REPRODUCES|FIXED|SKIPPED|CANNOT-REPRODUCE|"
                     r"NOT-A-LIMITATION)", re.M)


def files(tier: str) -> list[str]:
    out = []
    for p in sorted(LIMITS.glob("*.py")):
        if p.name.startswith("_"):
            continue
        if tier == "fast" and p.name in SLOW:
            continue
        out.append(p.name)
    return out


def run_one(name: str, env: dict, cwd: Path) -> dict:
    t = time.time()
    # A writable scratch cwd, as in suite_runner: these repros build HLS
    # projects with relative default paths, and the evaluation checkout is
    # read-only.
    p = subprocess.Popen([sys.executable, str(LIMITS / name)], cwd=cwd,
                         env=env, text=True, encoding="utf-8",
                         errors="replace", stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True)
    try:
        out, _ = p.communicate(timeout=TIMEOUT)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        out, _ = p.communicate()
        out, rc = (out or "") + f"\nTIMEOUT after {TIMEOUT}s", 124
    verdicts = {item: v for item, v in VERDICT.findall(out or "")}
    return {"file": name, "rc": rc, "seconds": round(time.time() - t, 1),
            "verdicts": verdicts,
            "produced_a_verdict": bool(verdicts),
            "tail": (out or "")[-1200:]}


def run(argv) -> tuple[int, dict]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="fast", choices=("fast", "full"))
    ap.add_argument("--work", default=None)
    a = ap.parse_args(argv)
    cwd = Path(a.work or f"/tmp/limits-{os.getpid()}").resolve() / "cwd"
    cwd.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    report = {"tier": a.tier, "items": [], "seconds": 0.0}
    t0 = time.time()
    for name in files(a.tier):
        r = run_one(name, env, cwd)
        print(f"  {'v' if r['produced_a_verdict'] else 'X'} {name}: "
              f"{r['verdicts'] or 'NO VERDICT'} ({r['seconds']}s)", flush=True)
        report["items"].append(r)
    report["seconds"] = round(time.time() - t0, 1)
    report["verdicts"] = {f"{r['file']}:{k}": v for r in report["items"]
                          for k, v in r["verdicts"].items()}
    report["silent"] = sorted(r["file"] for r in report["items"]
                              if not r["produced_a_verdict"])
    return 0, report


def compare(baseline: dict, now: dict) -> dict:
    b, n = baseline.get("verdicts", {}), now.get("verdicts", {})
    regressions = sorted(f"{k}: {b[k]} -> {n.get(k, 'NO VERDICT')}"
                         for k in b
                         if b[k] == "FIXED" and n.get(k) != "FIXED")
    fixed = sorted(f"{k}: {b[k]} -> {n[k]}" for k in b
                   if b[k] == "REPRODUCES" and n.get(k) == "FIXED")
    silenced = sorted(set(now.get("silent", ()))
                      - set(baseline.get("silent", ())))
    vanished = sorted(k for k in b if k not in n)
    return {"regressions": regressions, "fixed": fixed,
            "silenced": silenced, "vanished": vanished,
            "ok": not regressions and not silenced and not vanished,
            "baseline_items": len(b), "now_items": len(n)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier", default="fast", choices=("fast", "full"))
    ap.add_argument("--work", default=None)
    ap.add_argument("--record", type=Path)
    ap.add_argument("--against", type=Path)
    a = ap.parse_args()
    rc, report = run(["--tier", a.tier]
                     + (["--work", a.work] if a.work else []))
    if a.record:
        a.record.write_text(json.dumps(report, indent=1, sort_keys=True))
        print(f"recorded {len(report['verdicts'])} verdicts -> {a.record}")
    if a.against:
        print(json.dumps(compare(json.loads(a.against.read_text()), report),
                         indent=1))
    else:
        print(json.dumps({"verdicts": report["verdicts"],
                          "silent": report["silent"],
                          "seconds": report["seconds"]}, indent=1))
    return rc


if __name__ == "__main__":
    sys.exit(main())
