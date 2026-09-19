"""What the model has cost, read from opencode's own session database.

opencode records every assistant message's `cost` (USD) in
`~/.local/share/opencode/opencode.db`, and each session's total in
`session.cost`. These are opencode's figures: token counts times ITS model
price table, not Google's invoice. The invoice (and the remaining credit on
CHIA2026) is read in the Cloud Console, Billing -> CHIA2026; gcloud and the
Billing API do not expose the credit balance.

Two questions, two functions:

* `spent_since(t0)` -- this run. Sums messages created since the run started:
  the global figure a per-run cap needs, counting every worker at once and a
  turn still in flight as soon as opencode writes it. Over-counts if something
  else on this account uses opencode at the same time -- the safe direction.
* `chia2026_spend()` -- cumulative, for the cap across runs
  (`CHIA_TOTAL_CAP_USD`). **Attribution is by cutover time**, recorded in
  the tracked `billing.json`: every `google-vertex` session created at or after
  the moment chia.env switched to chia2026-tinytpu counts against CHIA2026;
  everything before it billed test-adrs (the general account) and is reported
  separately, never against the cap. Why time and not a per-session project
  tag: opencode stores no GCP project anywhere in a session or message
  (`session.model` is only `{providerID, id}`), and the project is chosen per
  process, so it cannot be read back. From the cutover on, the pre-flight gate
  refuses any CHIA run whose project does not bill CHIA2026, so every CHIA
  session after it did. What the rule can get wrong is only in the safe
  direction: another opencode user of google-vertex on this Unix account after
  the cutover would be counted against CHIA's cap, not missed.

The live DB is the only source. `chia_runs/opencode_sessions.db` in the
chia-codesign worktree is a 2026-09-05 snapshot whose sessions are all in the
live DB already; it is not read, so nothing is counted twice.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path

DB = Path(os.environ.get(
    "OPENCODE_DB", Path.home() / ".local/share/opencode/opencode.db"))
BILLING = Path(__file__).resolve().parent / "billing.json"
#: Only this provider bills a GCP project.
VERTEX = "google-vertex"


def _connect():
    return sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=30)


def spent_since(t0_ms: int, t1_ms: int | None = None) -> dict:
    """USD and tokens for all opencode assistant messages in [t0, t1]."""
    t1_ms = t1_ms or int(time.time() * 1000) + 10**9
    out = {"usd": 0.0, "messages": 0, "input": 0, "output": 0, "cache_read": 0}
    if not DB.exists():
        return out
    con = _connect()
    try:
        rows = con.execute(
            "select data from message where time_created between ? and ?",
            (t0_ms, t1_ms)).fetchall()
    finally:
        con.close()
    for (data,) in rows:
        d = json.loads(data)
        if d.get("role") != "assistant":
            continue
        tok = d.get("tokens") or {}
        out["usd"] += d.get("cost", 0) or 0
        out["messages"] += 1
        out["input"] += tok.get("input", 0) or 0
        out["output"] += tok.get("output", 0) or 0
        out["cache_read"] += (tok.get("cache") or {}).get("read", 0) or 0
    out["usd"] = round(out["usd"], 4)
    return out


def billing() -> dict:
    return json.loads(BILLING.read_text())


def sessions() -> list[dict]:
    """Every opencode session: id, time_created (ms), provider, model, cost."""
    if not DB.exists():
        return []
    con = _connect()
    try:
        rows = con.execute("select id, time_created, model, cost, title "
                           "from session order by time_created").fetchall()
    finally:
        con.close()
    out = []
    for sid, tc, model, cost, title in rows:
        try:
            m = json.loads(model or "{}")
        except json.JSONDecodeError:
            m = {}
        out.append({"id": sid, "time_created": tc, "provider": m.get("providerID"),
                    "model": m.get("id"), "cost": cost or 0.0, "title": title})
    return out


def chia2026_spend(before_ms: int | None = None) -> dict:
    """Cumulative google-vertex spend attributed to CHIA2026 (by cutover),
    counting sessions created before `before_ms` only, if given -- a run's
    own sessions are its per-run cap's business, not "spent so far"."""
    cur = billing()["current"]
    end = before_ms if before_ms is not None else float("inf")
    rows = [s for s in sessions()
            if s["provider"] == VERTEX and cur["cutover_ms"] <= s["time_created"] < end]
    return {"usd": round(sum(s["cost"] for s in rows), 4), "sessions": len(rows),
            "project": cur["project"], "billing_account": cur["billing_account"],
            "cutover_ms": cur["cutover_ms"], "rows": rows}


def _runs(repo: Path) -> list[dict]:
    """CHIA runs recorded under chia_runs/, as [t0, t1) windows."""
    runs = []
    for rj in sorted((repo / "chia_runs").glob("*/run.json")):
        try:
            r = json.loads(rj.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not r.get("t0_ms"):
            continue
        wall = None
        sj = rj.parent / "summary.json"
        if sj.exists():
            try:
                wall = json.loads(sj.read_text()).get("wall_seconds")
            except json.JSONDecodeError:
                pass
        runs.append({"run": rj.parent.name, "t0": r["t0_ms"],
                     "t1": r["t0_ms"] + int((wall or 0) * 1000) + 60_000 if wall else None,
                     "budget_usd": r.get("budget_usd")})
    runs.sort(key=lambda r: r["t0"])
    for a, b in zip(runs, runs[1:]):
        if a["t1"] is None or a["t1"] > b["t0"]:
            a["t1"] = b["t0"]
    return runs


def report(repo: Path, cap_usd: float) -> str:
    """The spend section of gcp_setup.sh's report."""
    b = billing()
    cut = b["current"]["cutover_ms"]
    allrows = sessions()
    runs = _runs(repo)

    def run_of(s):
        for r in runs:
            if s["time_created"] >= r["t0"] and (r["t1"] is None
                                                 or s["time_created"] < r["t1"]):
                return r["run"]
        return "(no recorded run: smoke.py, manual, or other opencode use)"

    def day(ms):
        return time.strftime("%Y-%m-%d", time.gmtime(ms / 1000))

    lines = []
    cur = [s for s in allrows if s["provider"] == VERTEX and s["time_created"] >= cut]
    old = [s for s in allrows if s["provider"] == VERTEX and s["time_created"] < cut]
    other = [s for s in allrows if s["provider"] != VERTEX]
    spent = sum(s["cost"] for s in cur)
    c = b["current"]
    lines.append(f"CHIA spend on {c['billing_account_name']} ({c['billing_account']}, "
                 f"project {c['project']}), since the cutover {c['cutover_utc']}:")
    lines.append(f"  total ${spent:.4f} over {len(cur)} session(s); cap "
                 f"CHIA_TOTAL_CAP_USD=${cap_usd:.2f}; remaining ${cap_usd - spent:.4f}")
    by = {}
    for s in cur:
        by.setdefault(run_of(s), []).append(s)
    for name, ss in by.items():
        lines.append(f"    {name}: ${sum(s['cost'] for s in ss):.4f}, {len(ss)} session(s), "
                     f"{day(ss[0]['time_created'])}..{day(ss[-1]['time_created'])}")
    for p in b["previous"]:
        lines.append(f"Historical, NOT on CHIA2026 -- billed project {p['project']} "
                     f"({p['billing_account']}, {p['billing_account_name']}), "
                     f"before the cutover:")
        lines.append(f"  total ${sum(s['cost'] for s in old):.4f} over {len(old)} "
                     f"google-vertex session(s)")
        days = {}
        for s in old:
            days.setdefault(day(s["time_created"]), []).append(s["cost"])
        for d, cs in sorted(days.items()):
            lines.append(f"    {d}: ${sum(cs):.4f} ({len(cs)} sessions)")
    if other:
        lines.append(f"Not billed to any GCP project: {len(other)} session(s) on "
                     f"{sorted({s['provider'] for s in other})}, "
                     f"${sum(s['cost'] for s in other):.4f}")
    lines.append("These are opencode's figures (token counts x opencode's price "
                 "table), not the invoice. The account's own charges and the "
                 "remaining credit are read in the Cloud Console: Billing -> "
                 "CHIA2026 -> Reports / Credits.")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    if sys.argv[1:2] == ["report"]:
        print(report(Path(sys.argv[2]) if len(sys.argv) > 2
                     else Path(__file__).resolve().parents[4],
                     float(os.environ.get("CHIA_TOTAL_CAP_USD", "100"))))
    elif sys.argv[1:2] == ["chia2026"]:
        r = chia2026_spend()
        r.pop("rows")
        print(json.dumps(r))
    else:
        print(json.dumps(spent_since(int(sys.argv[1]),
                                     int(sys.argv[2]) if len(sys.argv) > 2 else None)))
