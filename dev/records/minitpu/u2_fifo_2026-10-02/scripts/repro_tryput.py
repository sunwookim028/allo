import numpy as np
import allo.dataflow as df
from allo.ir.types import Stream, int32, uint1

def make(use_result):
    @df.region()
    def top(X: int32[4], E: uint1[4]):
        s: Stream[int32, 4]
        @df.kernel(mapping=[1], args=[X, E])
        def k(x: int32[4], e: uint1[4]):
            for i in range(4):
                if use_result == 1:
                    ok: uint1 = s.try_put(x[i])
                    e[i] = s.empty() + ok - ok
                else:
                    junk: uint1 = s.try_put(x[i])
                    e[i] = s.empty()
    return top

for u in (1, 0):
    mod = df.build(make(u), target="simulator")
    e = np.zeros(4, np.uint8)
    mod(np.arange(4, dtype=np.int32), e)
    print("try_put result", "used" if u else "unused", "-> empty() after each put:", e.tolist(), "(expected [0,0,0,0])")
