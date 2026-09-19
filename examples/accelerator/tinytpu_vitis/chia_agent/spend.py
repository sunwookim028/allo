"""What the model has cost so far, read from opencode's own session database.

opencode records every assistant message's `cost` (USD, from its model price
table) in `~/.local/share/opencode/opencode.db`. Summing the messages created
since a run started is the global figure a spend cap needs: it counts every
worker at once, and it includes a turn still in flight as soon as opencode
writes it. It over-counts if something else on this account uses opencode at
the same time -- the safe direction for a cap.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path

DB = Path(os.environ.get(
    "OPENCODE_DB", Path.home() / ".local/share/opencode/opencode.db"))


def spent_since(t0_ms: int, t1_ms: int | None = None) -> dict:
    """USD and tokens for all opencode assistant messages in [t0, t1]."""
    t1_ms = t1_ms or int(time.time() * 1000) + 10**9
    out = {"usd": 0.0, "messages": 0, "input": 0, "output": 0, "cache_read": 0}
    if not DB.exists():
        return out
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=30)
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


if __name__ == "__main__":
    import sys
    print(json.dumps(spent_since(int(sys.argv[1]),
                                 int(sys.argv[2]) if len(sys.argv) > 2 else None)))
