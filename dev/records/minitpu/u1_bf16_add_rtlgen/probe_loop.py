import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["BF_N"] = "64"
import numpy as np
from bitsloop_kernel import bf16_add_bits_loop
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../examples/minitpu/harness"))
import stimulus, ref
rtl = bf16_add_bits_loop.schedule().export("rtl")
q = rtl.estimation
print(f"PROBE loop: latency={q.latency} II={q.interval} fmax={q.fmax:.0f} lut={q.area.lut} ff={q.area.ff}")
st = stimulus.binary_bf16()[1936:1936+64]
o = np.zeros(64, np.uint16); r = rtl.cosim(st[:,0].copy(), st[:,1].copy(), o)
print("PROBE loop cosim cycles", r.cycles, "match:", (o == ref.vpu_bf16_add(st[:,0], st[:,1])).sum(), "/64")
