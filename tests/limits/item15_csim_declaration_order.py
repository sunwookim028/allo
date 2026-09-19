# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Item 15: Vitis csim executes dataflow processes in declaration order, so a
consumer declared before its producer reads an empty stream.

Runs `df.build(target="vitis_hls", mode="csim")` -- which is Allo's own g++
build of kernel.cpp against the Vitis HLS headers (IPModule, link_hls=True),
not `csim_design` -- twice: producer-first and consumer-first. Needs
/opt/xilinx/Vitis_HLS/2023.2/settings64.sh sourced (vitis_hls on PATH)."""
import os
import shutil
import traceback
import numpy as np
import _worktree
from _worktree import verdict

ITEM = 15
import allo.dataflow as df
from allo.ir.types import Stream, int32

HERE = os.path.dirname(os.path.abspath(__file__))


@df.region()
def prod_first(A: int32[8], B: int32[8]):
    s: Stream[int32, 2]

    @df.kernel(mapping=[1], args=[A])
    def producer(a: int32[8]):
        for i in range(8):
            s.put(a[i] + 1)

    @df.kernel(mapping=[1], args=[B])
    def consumer(b: int32[8]):
        for i in range(8):
            b[i] = s.get()


@df.region()
def cons_first(A: int32[8], B: int32[8]):
    s: Stream[int32, 2]

    @df.kernel(mapping=[1], args=[B])
    def consumer(b: int32[8]):
        for i in range(8):
            b[i] = s.get()

    @df.kernel(mapping=[1], args=[A])
    def producer(a: int32[8]):
        for i in range(8):
            s.put(a[i] + 1)


def run(top, tag):
    prj = os.path.join(HERE, f"_item15_{tag}.prj")
    try:
        mod = df.build(top, target="vitis_hls", mode="csim", project=prj)
        # the two process calls, in the order the top function makes them
        calls = [l.strip() for l in mod.hls_code.splitlines()
                 if l.strip().startswith(("producer", "consumer"))]
        a = np.arange(8, dtype=np.int32)
        b = np.zeros(8, dtype=np.int32)
        mod(a, b)
        return ("ok" if (b == a + 1).all() else f"WRONG {b.tolist()}"), calls
    except (Exception, SystemExit) as e:  # noqa: BLE001
        return f"{type(e).__name__}: {str(e)[:120]}", []
    finally:
        shutil.rmtree(prj, ignore_errors=True)


def main():
    import subprocess
    import sys

    if len(sys.argv) > 1:  # child: one variant
        top = {"pf": prod_first, "cf": cons_first}[sys.argv[1]]
        r, calls = run(top, sys.argv[1])
        print(f"CHILD {r} calls={calls}")
        return
    if shutil.which("vitis_hls") is None:
        print(f"[item {ITEM}] SKIPPED: vitis_hls not on PATH")
        return
    res = {}
    for tag in ("pf", "cf"):
        try:
            p = subprocess.run([sys.executable, os.path.abspath(__file__), tag],
                               capture_output=True, text=True, check=False,
                               timeout=int(os.environ.get("ITEM15_TIMEOUT", 240)))
            line = next((l for l in p.stdout.splitlines() if l.startswith("CHILD")),
                        f"CHILD died rc={p.returncode}: {(p.stdout + p.stderr)[-200:]}")
            res[tag] = line[6:]
        except subprocess.TimeoutExpired:
            res[tag] = "HUNG (csim blocked on an empty hls::stream; no error printed)"
        finally:
            shutil.rmtree(os.path.join(HERE, f"_item15_{tag}.prj"), ignore_errors=True)
    print(f"  producer declared first: {res['pf']}")
    print(f"  consumer declared first: {res['cf']}")
    verdict(ITEM, res["pf"].startswith("ok") and not res["cf"].startswith("ok"),
            f"producer-first={res['pf'][:60]}; consumer-first={res['cf'][:70]}")


if __name__ == "__main__":
    main()
