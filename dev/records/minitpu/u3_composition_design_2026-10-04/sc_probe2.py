import sys, time
import numpy as np
import allo.dataflow as df
from examples.minitpu.template import matrix_engine as me
from examples.minitpu.template.engines import BF16_ACC24, INT8_INT32
prj = sys.argv[1]
for mat, mac, dim in ((me.SYSTOLIC, BF16_ACC24, 4), (me.SYSTOLIC, INT8_INT32, 16)):
    t = time.time()
    try:
        arch = me.mxu_rig(mat, mac, dim, 16)
        mod = df.build(arch.region(), target="systemc", mode="csim", project=f"{prj}/{mat.name}_{mac.name}_d{dim}")
        A, W = me.stimulus(mac, dim, 16, 5)
        out = np.zeros(16 * dim, np.uint32)
        mod(A.reshape(-1).astype(np.uint32), W.reshape(-1).astype(np.uint32), out)
        want = me.matrix_rows(me._signed_in(mac, A), me._signed_in(mac, W), mac, mat.order) & ((1 << mac.OUT_BITS) - 1)
        got = out.reshape(16, dim).astype(np.int64)
        print(f"SC mxu {mat.name} {mac.name} d{dim}: {np.sum(got==want)}/{got.size} ({time.time()-t:.0f}s)")
    except Exception as e:
        print(f"SC mxu {mat.name} {mac.name} d{dim}: ERROR", type(e).__name__, " ".join(str(e).split())[:200])
