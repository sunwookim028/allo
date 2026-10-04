# emit the mixed design for Catapult csyn: python emit_mixed.py <prj> [--top rf_0|none] [--n N]
import os, shutil, sys
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import allo.dataflow as df, mixed
prj = sys.argv[1]
top = sys.argv[sys.argv.index("--top") + 1] if "--top" in sys.argv else "rf_0"
n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 64
s = df.customize(mixed.mixed(n))
s.pipeline("rf_0:_")
cfg = {"clock_period": 3.33}
if top != "none":
    cfg["synth_top"] = top
if os.path.isdir(prj):
    shutil.rmtree(prj)
s.build(target="systemc", mode="csyn", project=prj, configs=cfg)
t = open(os.path.join(prj, "run.tcl")).read()
open(os.path.join(prj, "run.tcl.emitted"), "w").write(t)
print("EMITTED", prj)
