# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniTPU ``minitpu_core`` (M-R0's standalone build) through RTLModule's own path.

Usage: core_build.py <exported MiniTPU tree at b3ba0a4d> [build-jobs]
(from the root of a tree with the pr48-merge follow-up; VERILATOR and CXX set,
gcc-toolset-13 first on PATH).

1. ``validate_rtl()`` -- the ``--json-only`` port -- on the whole core. The one
   placeholder Port binds three real pins (and ``done``); the core has many more inputs, so the
   expected outcome is ``ValueError: Unbound RTL input pins`` *after* Verilator
   elaborated the core and the pins were read from the JSON tree.
2. The ``--cc --build`` model build with ``verilator_args=["--build-jobs", N]``
   (validation bypassed, since 1 refuses by design), timed; every Verilator
   command line is printed, so the pass-through is visible.
"""
import sys
import time
from pathlib import Path

import allo.backend.rtl as rtl
from allo import RTLModule, Port

M = Path(sys.argv[1]).resolve()
JOBS = sys.argv[2] if len(sys.argv) > 2 else "16"
files = [
    M / line.strip()
    for line in (M / "src/core/core.f").read_text().splitlines()
    if line.strip() and not line.startswith("+incdir")
]
_run = rtl._run


def logged(command, env=None):
    if "verilator" in Path(command[0]).name:
        flags = [c for c in command[1:] if not c.endswith((".sv", ".v"))]
        print("CMD verilator", " ".join(flags), flush=True)
    return _run(command, env)


rtl._run = logged
ip = RTLModule(
    "minitpu_core",
    files,
    include_paths=[M / "src/pkg"],
    # Placeholder binding of real pins (only the build is the point here).
    ports=[
        Port("cmd", "program_id_csr", "start", "dm_rsp_ready", size=1, ctype="uint32_t")
    ],
    done="done",
    clock="clk",
    reset="rst_n",
    reset_active_high=False,
    start=None,
    persistent=True,
    verilator_args=["--build-jobs", JOBS],
)
t0 = time.time()
try:
    pins = ip.validate_rtl()
    print("VALIDATE unexpectedly passed", len(pins))
except ValueError as err:
    msg = str(err)
    print(f"VALIDATE ValueError after {time.time() - t0:.1f}s: {msg}")
    print("VALIDATE unbound inputs:", msg.count("'") // 2)
ip.validate_rtl = lambda: {}
t0 = time.time()
ip._prepare_simulation()
wall = time.time() - t0
lib = Path(ip._link_inputs[0])
print(f"BUILD ok wall={wall:.0f}s {lib.name} {lib.stat().st_size} B")
