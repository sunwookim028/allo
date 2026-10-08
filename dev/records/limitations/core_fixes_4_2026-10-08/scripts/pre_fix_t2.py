# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""T-2 before the fix: the regression test's owner/sink on the OLD server form
(``_server`` forced to the declared-link body on Stream links: put after the
shift, pipe uninitialised). The read port is declared first, because the old
server took every port's address before any put, so an owner reaching its
read port before its write port deadlocked the exchange when ``w`` was
declared first (the order the regression test uses). Run from the worktree
root with PYTHONPATH=$PWD."""
import sys

sys.path.insert(0, "tests/dataflow")
import test_compose_port_latency as t  # noqa: E402
from allo.compose import Architecture, Channel, Memory, Port  # noqa: E402

orig = Architecture._server
Architecture._server = lambda self, m, it, sa, sram=False, links="declared": orig(self, m, it, sa, sram, "declared")
for L in (1, 2):
    n = 48
    mem = Memory("m", "UInt(16)", rows="8", ports=(Port("r", "r", latency=L), Port("w", "w", visible=1)),
                 collision="undefined", reset=False)
    a = Architecture(name=f"pre{L}", parameters={"N": n},
                     memories=(Memory("OI", "int32[N]"), Memory("OQ", "UInt(16)[N]"), mem),
                     channels=(Channel("o_i", "int32", "2", kind="wire"),
                               Channel("o_q", "UInt(16)", "2", kind="wire")),
                     units=(t.owner, t.sink))
    pairs = t._run(a.build("simulator", {"m": "registers"}), n)
    try:
        t._check(pairs, t.model(n, L), n, L, "PRE-FIX simulator (old server form)")
        print("L", L, "pre-fix: agrees")
    except AssertionError as e:
        print("L", L, str(e)[:200])
