# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Second unit: MiniTPU acc24 adder `bits` through RTLGen; reported vs measured latency.
MiniTPU's RTL is a 3-stage pipe (MXU_ACC_ADD_LATENCY = 3).
Usage (after sourcing env.sh, from this dir): python exp_acc24.py <N> <mhz> [<mhz> ...]"""
import sys; sys.dont_write_bytecode = True  # keep the read-only U1/harness dirs free of __pycache__
import os, sys, warnings
HARN = "/work/shared/users/phd/sk3463/scratch/wt-lat/examples/minitpu/harness"
N, mhzs = int(sys.argv[1]), [float(x) for x in sys.argv[2:]]
os.environ["ACC_N"] = str(N)
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), HARN]
import numpy as np, stimulus, ref
from acc24_bits_kernel import acc24_add_bits

st = stimulus.binary_acc24()
idx = np.random.default_rng(0).choice(len(st), N, replace=False)
A = st[idx, 0].astype(np.uint32); B = st[idx, 1].astype(np.uint32)
gold = np.array([int(ref.mxu_acc24_add(int(a), int(b))) for a, b in zip(A, B)], dtype=np.uint32)
for mhz in mhzs:
    rtl = acc24_add_bits.schedule().export("rtl", freq_mhz=mhz)
    s = rtl.schedule(); fn = s.func(rtl.top)
    regs = [(r.kind.value, r.interval, r.trip_count, r.latency, r.iteration_latency, r.cost.drain) for r in fn.regions]
    q = rtl.estimation
    print(f"ACC24 N={N} {mhz:g}MHz REPORT latency={fn.latency} exact={fn.latency_is_exact} cycle_ns={s.cycle_ns:.3f} "
          f"regions(kind,II,trip,lat,iter_lat,drain)={regs} fmax={q.fmax:.0f} lut={q.area.lut} ff={q.area.ff}")
    C = np.zeros(N, np.uint32)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = rtl.cosim(A.copy(), B.copy(), C)
        for ww in w:
            print(f"  WARNING {ww.category.__name__}: {ww.message}")
    print(f"ACC24 N={N} {mhz:g}MHz MEASURED cycles={r.cycles} match={(C == gold).sum()}/{N}")
