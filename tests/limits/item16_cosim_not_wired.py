# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Item 16: `cosim` is not wired into `df.build`.

Checks, without running Vitis: (a) mode="cosim" is accepted; (b) what the
generated project would give `cosim_design` -- a C++ testbench (host.cpp with a
`main` that calls the top function), and `depth=` on each `m_axi` pragma, which
cosim needs to size its memory models."""
import os
import re
import shutil
import _worktree
from _worktree import verdict

ITEM = 16
import allo.dataflow as df
from allo.ir.types import int32

HERE = os.path.dirname(os.path.abspath(__file__))
PRJ = os.path.join(HERE, "_item16.prj")


@df.region()
def top(A: int32[16], B: int32[16]):
    @df.kernel(mapping=[1], args=[A, B])
    def k(a: int32[16], b: int32[16]):
        for i in range(16):
            b[i] = a[i] + 1


def main():
    res = {}
    try:
        df.build(top, target="vitis_hls", mode="cosim", project=PRJ, wrap_io=False)
        res["mode=cosim"] = "accepted"
    except (Exception, SystemExit) as e:  # noqa: BLE001
        res["mode=cosim"] = f"{type(e).__name__}: {str(e)[:60]}"
    shutil.rmtree(PRJ, ignore_errors=True)
    # hw_emu is the only mode whose run.tcl says cosim_design; look at what it ships.
    try:
        mod = df.build(top, target="vitis_hls", mode="hw_emu", project=PRJ, wrap_io=False)
        tcl = open(os.path.join(PRJ, "run.tcl")).read()
        host = open(os.path.join(PRJ, "host.cpp")).read()
        res["hw_emu run.tcl has cosim_design"] = "cosim_design" in tcl
        res["host.cpp is an XRT/OpenCL host"] = ("xcl2" in host or "cl::" in host)
        axi = [l.strip() for l in mod.hls_code.splitlines() if "m_axi" in l]
        res["m_axi pragmas carry depth="] = bool(axi) and all("depth" in l for l in axi)
    except (Exception, SystemExit) as e:  # noqa: BLE001
        res["hw_emu"] = f"{type(e).__name__}: {str(e)[:80]}"
    finally:
        shutil.rmtree(PRJ, ignore_errors=True)
    print(res)
    wired = res["mode=cosim"] == "accepted" or (
        res.get("hw_emu run.tcl has cosim_design") and not res.get("host.cpp is an XRT/OpenCL host")
        and res.get("m_axi pragmas carry depth="))
    verdict(ITEM, not wired, str(res))


if __name__ == "__main__":
    _worktree.run_guarded(ITEM, main)
