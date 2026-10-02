"""Emit bf16_add `native` (the harness's unit, unchanged) for Catapult csyn, then run it.

    python emit_csyn.py <n> <prj> [--pipeline]     (from the worktree root)

`mod()` runs `catapult -shell -f ../run.tcl` from <prj>/build. NO_RUN=1 emits only.
--pipeline adds s.pipeline("add_0:i") (II=1) on the kernel's loop.
"""
import os, sys, time
sys.path.insert(0, os.getcwd())
import allo.dataflow as df
from examples.minitpu.units import bf16_add
n, prj = int(sys.argv[1]), sys.argv[2]
s = df.customize(bf16_add.native(n))
if "--pipeline" in sys.argv:
    s.pipeline("add_0:i")
t = time.time()
mod = s.build(target="systemc", mode="csyn", project=prj)
print(f"BUILD_DONE {time.time()-t:.1f}s", flush=True)
if os.environ.get("NO_RUN"):
    sys.exit(0)
t = time.time()
try:
    mod(); print(f"CSYN_RETURNED_OK {time.time()-t:.1f}s", flush=True)
except Exception as e:
    print(f"CSYN_RAISED {type(e).__name__}: {e} after {time.time()-t:.1f}s", flush=True)
