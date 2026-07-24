# probe_b.py
import allo
from allo.ir.types import int32
import allo.dataflow as df
from allo.customize import customize as _customize
from allo.ir.utils import get_global_vars
import numpy as np

vadd = allo.IPModule(
    top="vadd",
    impl="/home/vsy5/allo/tests/ip_integration/vadd.cpp",
    link_hls=False,          # no vitis_hls on your PATH; skips the include-path probe
)

@df.region()
def top(A: int32[32], B: int32[32], C: int32[32]):
    @df.kernel(mapping=[1], args=[A, B, C])
    def compute(a: int32[32], b: int32[32], c: int32[32]):
        vadd(a, b, c)

print("=== PRE-HOIST ===");  print(_customize(top, global_vars=get_global_vars(top)).module)
print("=== POST-HOIST ==="); print(df.customize(top).module)

mod = df.build(top, target="simulator")
np_A = np.random.randint(0, 100, (32,)).astype(np.int32)
np_B = np.random.randint(0, 100, (32,)).astype(np.int32)
allo_C = np.zeros(32).astype(np.int32)
mod(np_A, np_B, allo_C)
np.testing.assert_allclose(allo_C, np.add(np_A, np_B), atol=1e-5)
print(f"A: ", np_A, "\n" "B: ", np_B, "\n", "C: ", allo_C) 
print("simulation passed")