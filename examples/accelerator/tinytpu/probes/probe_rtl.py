"""Smoke-test chia's RTL backend: does an async dataflow pair schedule as
`concurrent` and cosim to a real cycle count?

Mirrors tests/rtl/test_determinacy.py's async container, then actually runs it
under cocotb/verilator, which is the capability the HLS path lacks.
"""
import numpy as np
from allo import kernel
from allo.lang import i32, Stream

N = 16


@kernel
async def dp(s: Stream[i32]):
    for i in range(N):
        s.put(i * 2)


@kernel
async def dc(s: Stream[i32], out: i32[N]):
    for i in range(N):
        out[i] = s.get() + 1


@kernel
async def dtop(out: i32[N]):
    fifo: Stream[i32]
    await dp(fifo)
    await dc(fifo, out)


if __name__ == "__main__":
    rtl = dtop.schedule().export("rtl")
    sched = rtl.schedule()
    classes = {f.name: f.determinacy for f in sched.funcs}
    print("determinacy:", classes)
    out = np.zeros(N, dtype=np.int32)
    res = rtl.cosim(out)
    print("cosim cycles:", res.cycles)
    expect = np.arange(N, dtype=np.int32) * 2 + 1
    print("out     :", out[:8])
    print("expect  :", expect[:8])
    np.testing.assert_array_equal(out, expect)
    print("RTL COSIM PROBE: PASS")
