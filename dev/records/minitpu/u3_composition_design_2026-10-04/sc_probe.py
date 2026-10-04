import sys, time, traceback
import numpy as np
import allo.dataflow as df
from examples.minitpu.template import mac_pe, matrix_engine as me
from examples.minitpu.template.engines import BF16_ACC24, INT8_INT32
prj = sys.argv[1]
# (1) the two-engine one-region PE rig, SystemC csim stand-in
t = time.time()
try:
    arch = mac_pe.pe_rig_two(BF16_ACC24, INT8_INT32, 256)
    mod = df.build(arch.region(), target="systemc", mode="csim", project=f"{prj}/pe_two")
    aa, wa, pa = mac_pe.stimulus(BF16_ACC24, 256, 0); ab, wb, pb = mac_pe.stimulus(INT8_INT32, 256, 1)
    oa = np.zeros(256, np.uint32); ob = np.zeros(256, np.uint32)
    mod(aa, wa, pa, oa, ab, wb, pb, ob)
    ra, rb = mac_pe.reference(BF16_ACC24, aa, wa, pa), mac_pe.reference(INT8_INT32, ab, wb, pb)
    print(f"SC pe_two: bf16 {np.sum(oa==ra)}/256 int8 {np.sum(ob==rb)}/256 ({time.time()-t:.0f}s)")
except Exception as e:
    print("SC pe_two: ERROR", type(e).__name__, " ".join(str(e).split())[:300]); traceback.print_exc(limit=2)
# (2) the tree matrix engine at bf16 DIM=4
t = time.time()
try:
    arch = me.mxu_rig(me.TREE, BF16_ACC24, 4, 32)
    mod = df.build(arch.region(), target="systemc", mode="csim", project=f"{prj}/mxu_tree_d4")
    A, W = me.stimulus(BF16_ACC24, 4, 32, 5)
    out = np.zeros(128, np.uint32)
    mod(A.reshape(-1).astype(np.uint32), W.reshape(-1).astype(np.uint32), out)
    want = me.matrix_rows(A, W, BF16_ACC24, "tree") & 0xFFFF
    got = out.reshape(32, 4).astype(np.int64)
    print(f"SC mxu tree bf16 d4: {np.sum(got==want)}/{got.size} ({time.time()-t:.0f}s)")
except Exception as e:
    print("SC mxu tree bf16 d4: ERROR", type(e).__name__, " ".join(str(e).split())[:300]); traceback.print_exc(limit=2)
