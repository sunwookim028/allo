# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests of ``harness/rtl.py``'s ``trace`` shape on ``trace_tiny.sv``.

    $ALLO_PYTHON -m examples.minitpu.harness.tests.test_trace

Needs Verilator only.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 4)))
from examples.minitpu.harness import rtl  # noqa: E402

SV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trace_tiny.sv")
U = rtl.RtlUnit(
    top="trace_tiny",
    sources=[SV],
    inputs=[("rst_ni", 1), ("we_i", 1), ("waddr_i", 2), ("wdata_i", 100), ("raddr_i", 2)],
    outputs=[("async_o", 100), ("reg_o", 100, "post"), ("count_o", 8), ("junk_o", 32)],
    shape="trace",
    assertions=True,
)


def main():
    big = [(1 << 99) | (0xABC << 40) | k for k in range(8)]  # > 64 bits, 2 words
    n = 12
    cmd = {
        "rst_ni": [0] + [1] * (n - 1),
        "we_i": [0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0],
        "waddr_i": [0, 1, 2, 0, 1, 0, 3, 0, 0, 0, 0, 0],
        "wdata_i": [0, big[0], big[1], 0, big[2], 0, big[3], big[4], big[5], 0, 0, 0],
        "raddr_i": [1, 1, 1, 2, 1, 1, 3, 3, 0, 0, 3, 0],
    }
    r = rtl.run_trace(U, cmd, seed=1)
    a, g = rtl.unpack(r["async_o"]), rtl.unpack(r["reg_o"])
    # pre-edge: the cycle-1 write of addr 1 is not seen by cycle 1's read
    assert a[2] == big[0] and a[4] == big[0] and a[5] == big[2], [hex(x) for x in a]
    assert a[3] == big[1] and a[7] == big[3]
    # wide values round-trip, bit 99 included
    assert a[10] == big[3] and (a[10] >> 99) == 1
    # registered read of latency 2 ("post"): cycle-t read in row t + 1
    assert g[3] == big[0] and g[4] == big[1] and g[6] == big[2] and g[8] == big[3], [hex(x) for x in g]
    assert g[7] != big[3]  # read and write of one address in a cycle: the old word
    # reset counter, "pre": row t shows the count before edge t + 1
    c = rtl.unpack(r["count_o"])
    assert c[1] == 0 and c[2] == 1 and c[11] == 6, c
    # the assertion fired on the cycle the count was 5 (counts 5 in row 7)
    fired = [cy for cy, _ in rtl.last_asserts]
    assert fired == [c.index(5)], (fired, c, rtl.last_asserts)
    # x-initial unique: never-written state differs by seed, written does not
    r2 = rtl.run_trace(U, cmd, seed=2)
    assert (r2["junk_o"] != r["junk_o"]).any(), "junk_o did not change with the seed"
    a2 = rtl.unpack(r2["async_o"])
    assert a2[6] != a[6], "row 6 reads address 3 before its first write lands"
    assert [x for i, x in enumerate(a2) if i >= 2 and i != 6] == [x for i, x in enumerate(a) if i >= 2 and i != 6]
    assert rtl.unpack(r2["async_o"])[0] != a[0], "unwritten RAM entry did not change with the seed"
    # probes: async read 0, write visibility 1, registered read 2
    step = dict(cmd, raddr_i=[0] * 10 + [1] * 2 + [2] * 10)
    step = {k: (v + [0] * 10 if k != "raddr_i" else v) for k, v in step.items()}
    step["rst_ni"] = [1] * 22
    assert rtl.probe_trace(U, step, "async_o", 12) == 0
    assert rtl.probe_trace(U, step, "reg_o", 12) == 2
    vis = dict(cmd, raddr_i=[3] * n, we_i=[0] * 6 + [1] + [0] * 5,
               waddr_i=[3] * n, wdata_i=[0] * 6 + [big[7]] + [0] * 5)
    vis["rst_ni"] = [1] * n
    pre = rtl.unpack(rtl.run_trace(U, vis)["async_o"])
    assert pre[6] != big[7] and pre[7] == big[7]
    assert rtl.probe_trace(U, vis, "async_o", 6) == 1
    print("PASS test_trace")


if __name__ == "__main__":
    main()
