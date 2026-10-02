import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["BF_N"] = "16"
import numpy as np
from bits26_kernel import bf16_add_bits26
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../examples/minitpu/harness"))
import stimulus, ref
rtl = bf16_add_bits26.schedule().export("rtl")
q = rtl.estimation
print(f"PROBE w26: latency={q.latency} fmax={q.fmax:.0f} lut={q.area.lut} ff={q.area.ff}")
st = stimulus.binary_bf16()[:16]
o = np.zeros(16, np.uint16); rtl.cosim(st[:,0].copy(), st[:,1].copy(), o)
print("PROBE w26 cosim match:", (o == ref.vpu_bf16_add(st[:,0], st[:,1])).all())
