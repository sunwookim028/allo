import sys, time
sys.path.insert(0, "/work/shared/users/phd/sk3463/allo")
from allo.dataflow import customize
from examples.tinytpu.microarch_isa import tinytpu_isa, schedule
s = customize(tinytpu_isa); schedule(s)
t = time.time()
mod = s.build(target="systemc", mode="csyn", project="/tmp/claude-1772902/-work-shared-users-phd-sk3463-allo/3b1c24a0-f6d1-4c37-88e0-31db609cddf4/scratchpad/cmp/tt_csyn.prj")
print(f"BUILD_DONE {time.time()-t:.1f}s", flush=True)
t = time.time()
try:
    mod()
    print(f"CSYN_RETURNED_OK {time.time()-t:.1f}s", flush=True)
except Exception as e:
    print(f"CSYN_RAISED {type(e).__name__}: {e} after {time.time()-t:.1f}s", flush=True)
