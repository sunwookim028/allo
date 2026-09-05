# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The subprocess protocol the bed's three drivers share.

Each driver runs one case as `python -m <module> --one key::variant::axis` and
reads the row back off the one marked JSON line the child prints. A run that
times out, or that dies before printing that line, becomes a row of its own, so
a solver that never terminates and an assert that fires are both results.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def run_child(
    module: str, mark: str, argv: list[str], timeout: int, base: dict
) -> tuple[dict, str]:
    """Run one case in its own process, returning its row and the child's
    combined output. ``base`` is the row identity a timeout or a crash carries."""
    env = dict(os.environ)
    env["XILINX_VITIS"] = "/nonexistent"
    env["PYTHONPATH"] = str(REPO)
    env.setdefault("ALLO_LOG_LEVEL", "warn")
    try:
        p = subprocess.run(
            [sys.executable, "-m", module, *argv],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=str(REPO),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {**base, "status": "timeout"}, ""
    text = p.stdout + p.stderr
    for line in p.stdout.splitlines():
        if line.startswith(mark):
            return json.loads(line[len(mark) :]), text
    return {**base, "status": "crash", "error": text[-3000:]}, text
