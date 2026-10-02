import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../examples/minitpu/harness"))
import numpy as np, stimulus, ref
from bits_scalar_kernel import bf16_add_scalar
st = stimulus.binary_bf16()
for mhz in (300, 50):
    rtl = bf16_add_scalar.schedule().export("rtl", freq_mhz=mhz)
    ok = 0; cyc = set()
    for i in range(0, 1936, 97):
        r = rtl.cosim(int(st[i,0]), int(st[i,1]))
        cyc.add(r.cycles); ok += int(r.result) == int(ref.vpu_bf16_add(st[i:i+1,0], st[i:i+1,1])[0])
    print(f"PROBE scalar {mhz}MHz cosim: {ok}/20 match, cycles {cyc}")
# exposure to a flush-to-zero IP
A, B = st[:,0], st[:,1]
sub = lambda x: ((x & 0x7F80) == 0) & ((x & 0x7F) != 0)
W = ref.vpu_bf16_add(A, B)
print("PROBE vectors with a subnormal input or result:", (sub(A) | sub(B) | sub(W)).sum(), "of", len(A))
