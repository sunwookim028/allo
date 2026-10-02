"""Emit one closer-shaped bf16_add variant for Catapult (synth_top=add_0).

    python emit_var.py <stream|wire|channel> <n> <prj> [--pipeline] [--clock=NS]
"""
import os, sys, time
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import allo.dataflow as df
import variants
name, n, prj = sys.argv[1], int(sys.argv[2]), sys.argv[3]
s = df.customize(getattr(variants, name)(n))
if "--pipeline" in sys.argv:
    s.pipeline("add_0:_")  # the add kernel's steady-state loop, II=1
    print("pipelined add_0:_")
cfg = {"synth_top": "add_0"}
for a in sys.argv:
    if a.startswith("--clock="):
        cfg["clock_period"] = float(a.split("=")[1])
t = time.time()
s.build(target="systemc", mode="csyn", project=prj, configs=cfg)
print(f"BUILD_DONE {time.time()-t:.1f}s", flush=True)
