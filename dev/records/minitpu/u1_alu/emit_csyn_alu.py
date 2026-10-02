"""Emit an alu.py variant for Catapult csyn (II=1 on the kernel loop) and run it.
    python emit_csyn_alu.py <variant> <n> <prj> [--no-pipeline]    (from the worktree root)
"""
import os, sys, time
sys.path.insert(0, os.getcwd())
import allo.dataflow as df
from examples.minitpu.units import alu
v, n, prj = sys.argv[1], int(sys.argv[2]), sys.argv[3]
s = df.customize(alu.VARIANTS[v][0](n))
if "--no-pipeline" not in sys.argv:
    s.pipeline("alu_0:i")
if "--unroll-lz" in sys.argv:
    # the reused adder's inner loop: its schedule does not travel with it
    s.unroll("leading_zeros17:offset")
t = time.time()
mod = s.build(target="systemc", mode="csyn", project=prj)
print(f"BUILD_DONE {time.time()-t:.1f}s", flush=True)
t = time.time()
try:
    mod(); print(f"CSYN_RETURNED_OK {time.time()-t:.1f}s", flush=True)
except Exception as e:
    print(f"CSYN_RAISED {type(e).__name__}: {str(e)[:500]} after {time.time()-t:.1f}s", flush=True)
