"""What the model has cost, read from opencode's own session database.

opencode records every assistant message's `cost` (USD) in
`~/.local/share/opencode/opencode.db`, and each session's total in
`session.cost`. These are opencode's figures: token counts times ITS model
price table, not Google's invoice. The invoice (and the remaining credit on
CHIA2026) is read in the Cloud Console, Billing -> CHIA2026; gcloud and the
Billing API do not expose the credit balance.

Three questions, three functions:

* `run_spend(prefix, t0)` -- this run, for the per-run cap. Sums the messages
  of the sessions titled with the run's tag (`loop.py` passes `--title` to
  opencode, so a call that timed out -- which returns no session id and
  reports `usage` of $0 while being billed -- is still found) plus any session
  id the run saw returned. A turn in flight counts as soon as opencode writes
  it, and every worker of the run shares the prefix.
* `spent_since(t0)` -- every message on the account since t0. Context only,
  never a cap: with two tracks running it counts the other track's spend and
  stops the wrong run (measured: a counter read $15.82 against a $15 cap, of
  which $4.46 was the run's own; run 2 was stopped after 2 of 5 iterations
  this way).
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

DEFAULT_TOTAL_CAP_USD = 500.0   # as preflight.DEFAULT_TOTAL_CAP_USD
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


def run_spend(prefix: str, t0_ms: int, session_ids=()) -> dict:
    """USD per session for this run: sessions created since t0 whose title
    starts with `prefix`, and the named `session_ids`."""
    out = {"usd": 0.0, "messages": 0, "sessions": {}}
    if not DB.exists():
        return out
    ids = sorted({str(s) for s in session_ids if s})
    con = _connect()
    try:
        titled = [r[0] for r in con.execute(
            "select id from session where time_created >= ? "
            "and substr(title, 1, ?) = ?", (t0_ms, len(prefix), prefix))]
        wanted = sorted(set(titled) | set(ids))
        rows = con.execute(
            f"select session_id, data from message where session_id in "
            f"({','.join('?' * len(wanted))})", wanted).fetchall() if wanted else []
    finally:
        con.close()
    for sid in wanted:
        out["sessions"][sid] = 0.0
    for sid, data in rows:
        d = json.loads(data)
        if d.get("role") != "assistant":
            continue
        out["sessions"][sid] += d.get("cost", 0) or 0
        out["messages"] += 1
    out["sessions"] = {k: round(v, 4) for k, v in out["sessions"].items()}
    out["usd"] = round(sum(out["sessions"].values()), 4)
    return out


def of_run(run_dir: Path) -> dict:
    """What a run cost, from `<run>/run.json` alone: no memory of the session
    that started it, and never the `usage` field (which reports $0.00 for a
    call that timed out and was billed in full -- every large call of run 3 did
    exactly that, so the run read $0.00 on `usage` and $20.72 here)."""
    run_dir = Path(run_dir)
    run = json.loads((run_dir / "run.json").read_text())
    # Runs before the tag was recorded (run 1, the smoke run) have no
    # `run_tag`; it is derivable, and if their sessions were never titled the
    # derived tag simply finds nothing, which is reported rather than raised.
    tag = run.get("run_tag") or f"chia-run {run_dir.name}@{run['t0_ms']}"
    spend = run_spend(tag + " ", run["t0_ms"])
    cumulative = chia2026_spend()
    cumulative.pop("rows", None)
    cap = float(os.environ.get("CHIA_TOTAL_CAP_USD", DEFAULT_TOTAL_CAP_USD))
    out = {"run_dir": str(run_dir.resolve()),
            "run_tag": tag,
            "run_tag_source": "run.json" if run.get("run_tag") else
                              "derived: this run.json predates the field",
            "t0_ms": run["t0_ms"],
            "source": "opencode's own database (spend.run_spend), "
                      "NEVER the `usage` field",
            "run_usd": spend["usd"],
            "run_sessions": len(spend["sessions"]),
            "per_session_usd": spend["sessions"],
            "model_messages": spend["messages"],
            "per_run_cap_usd": run.get("budget_usd"),
            "cumulative_chia2026_usd_after": cumulative["usd"],
            "cumulative_sessions_after": cumulative["sessions"],
            "total_cap_usd": cap}
    if not spend["usd"]:
        summary = run_dir / "summary.json"
        recorded = (json.loads(summary.read_text()).get("spend", {}).get("usd")
                    if summary.exists() else None)
        out["note"] = ("no session carries this tag -- no model call was "
                       "made, the run predates the titling, or its sessions "
                       "are gone. "
                       + (f"summary.json recorded ${recorded}. "
                          if recorded is not None else "")
                       + "`spend.py report <repo>` attributes by time window "
                         "instead.")
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
    """CHIA runs recorded under chia_runs/ (live output) and the tracked
    dev/records/tinytpu/chia-evidence/ directory, as [t0, t1) windows."""
    runs = []
    found = [*(repo / "chia_runs").glob("*/run.json"),
             *(repo / "dev/records/tinytpu/chia-evidence").glob("*/run.json")]
    seen = set()
    for rj in sorted(found, key=lambda p: p.parent.name):
        if rj.parent.name in seen:
            continue
        seen.add(rj.parent.name)
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
                     else Path(__file__).resolve().parents[3],
                     float(os.environ.get("CHIA_TOTAL_CAP_USD",
                                          DEFAULT_TOTAL_CAP_USD))))
    elif sys.argv[1:2] == ["run"]:
        print(json.dumps(of_run(Path(sys.argv[2])), indent=1))
    elif sys.argv[1:2] == ["chia2026"]:
        r = chia2026_spend()
        r.pop("rows")
        print(json.dumps(r))
    else:
        print(json.dumps(spent_since(int(sys.argv[1]),
                                     int(sys.argv[2]) if len(sys.argv) > 2 else None)))
