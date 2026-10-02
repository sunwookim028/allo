# Dump AMC's intermediate IR for a kernel: <out>/core.mlir (allo -> core
# dialects) and <out>/loopschedule.mlir, via AMCModule(skip_fsm=True).
# Usage: python dump_stages.py <module:function | file.mlir:top> <out_dir>
import sys, os, importlib
from allo.backend.amc import AMCModule
src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
mod_name, fn = src.rsplit(":", 1)
if mod_name.endswith(".mlir"):
    txt = open(mod_name).read()
else:
    import allo
    txt = str(allo.customize(getattr(importlib.import_module(mod_name), fn)).module)
open(f"{out}/allo.mlir", "w").write(txt)
m = AMCModule(txt, top_func_name=fn, allocate_amc=True, skip_fsm=True)
open(f"{out}/core.mlir", "w").write(str(m.module))
open(f"{out}/amc_alloc.mlir", "w").write(str(m.amcAllocation))
open(f"{out}/loopschedule.mlir", "w").write(str(m.loopSchedule))
print("dumped", out)
