import os, sys, numpy as np
sys.path.insert(0, os.environ.get("WT", "/work/shared/users/phd/sk3463/scratch/wt-u2rf"))
import allo.dataflow as df
from examples.minitpu.units import vpu_regfile as u
n = 8
be = sys.argv[1]
try:
    top = u.trace(n, 256)
    mod = df.build(top, target="simulator") if be == "sim" else df.build(top, target="systemc", mode="csim", project="/work/shared/users/phd/sk3463/scratch/u2rf/prj/w256_" + be)
    print("BUILD-OK", be, flush=True)
    ra = np.zeros(n, np.uint8); wa = np.arange(n, dtype=np.uint8); we = np.ones(n, np.uint8)
    for wd in (np.array([(1 << 200) + i for i in range(n)], dtype=object), np.arange(n, dtype=np.uint64)):
        q = [np.zeros(n, dtype=wd.dtype) for _ in range(3)]
        try:
            mod(ra, ra, ra, wa, wd, we, *q)
            print("RUN", be, wd.dtype, [hex(int(x)) for x in q[0][:3]], flush=True)
        except Exception as e:
            print("RUN-FAIL", be, wd.dtype, type(e).__name__, str(e)[:300], flush=True)
except Exception as e:
    print("BUILD-FAIL", be, type(e).__name__, str(e)[:400], flush=True)
