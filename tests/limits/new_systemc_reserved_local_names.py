# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U4 track C finding C1 (dev/records/minitpu/u4_track_c_2026-10-08.rst): the
SystemC emitter gives every kernel process the ports ``clk``, ``rst`` and
``done`` (``SC_MODULE(<kernel>_0)``) and emits a kernel-local of the same
name into the process body unrenamed, so a user local called ``done`` (the
natural name for MiniTPU's ``dma_channel_done`` register) shadows the port
and ``done.write(true)`` fails to compile in g++. The simulator builds and
runs the same region. Loud (a compile error), not silent; the class of
item 20 (an emitted name colliding with another), on the SystemC fork.
Workaround: rename the local. Fix: rename user locals that collide with the
process's generated members (or prefix the generated ones)."""
import os
import tempfile

import _worktree
from _worktree import verdict

ITEM = "new-systemc-reserved-local"
import numpy as np  # noqa: E402

import allo.dataflow as df  # noqa: E402
from allo.ir.types import int32  # noqa: E402


@df.region()
def top_done(A: int32[4], B: int32[4]):
    @df.kernel(mapping=[1], args=[A, B])
    def k(a: int32[4], b: int32[4]):
        done: int32 = 0
        for i in range(4):
            done = done + a[i]
            b[i] = done


@df.region()
def top_acc(A: int32[4], B: int32[4]):
    @df.kernel(mapping=[1], args=[A, B])
    def k(a: int32[4], b: int32[4]):
        acc: int32 = 0
        for i in range(4):
            acc = acc + a[i]
            b[i] = acc


def _run(top, prj):
    a = np.arange(1, 5, dtype=np.int32)
    b = np.zeros(4, dtype=np.int32)
    mod = df.build(top, target="systemc", mode="csim", project=prj)
    mod(a, b)
    return b.tolist()


def main():
    want = [1, 3, 6, 10]
    sim = np.zeros(4, dtype=np.int32)
    df.build(top_done, target="simulator")(np.arange(1, 5, dtype=np.int32), sim)
    res = {"simulator, local `done`": sim.tolist()}
    with tempfile.TemporaryDirectory() as d:
        for name, top in (("acc", top_acc), ("done", top_done)):
            try:
                res[f"systemc, local `{name}`"] = _run(top, os.path.join(d, name))
            except Exception as e:  # noqa: BLE001
                res[f"systemc, local `{name}`"] = f"{type(e).__name__}: {str(e)[:80]}"
    print(res)
    ok_ctrl = res["systemc, local `acc`"] == want and res["simulator, local `done`"] == want
    verdict(ITEM, ok_ctrl and res["systemc, local `done`"] != want, str(res))


if __name__ == "__main__":
    _worktree.run_guarded(ITEM, main)
