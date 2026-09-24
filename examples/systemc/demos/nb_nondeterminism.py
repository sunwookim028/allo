# Empirically demonstrate that non-blocking stream ops are non-deterministic
# in Allo's current OpenMP simulator (no clock -> outcome depends on OS scheduler).
#
# Design (a "Type C" pattern in OmniSim's taxonomy: the NB success flag is
# OBSERVED and changes the program's output):
#   - producer: a FIXED number of try_put attempts (non-blocking, NO spin).
#   - consumer: a FIXED number of try_get attempts (non-blocking, NO spin).
# Neither kernel blocks, so there is no deadlock; the number of *successful*
# puts/gets depends entirely on how the two OMP threads happen to interleave.
#
# On real hardware these counts are a single deterministic number (fixed by
# II/latency). If the simulator were faithful, every run would print the same
# (put_ok, got). If instead we see the values VARY across runs, that is the
# non-determinism / timing-unfaithfulness we claimed.
import sys
sys.path.insert(0, "/home/zsm9/allo_sup")  # force the working checkout (two-checkouts trap)

import numpy as np
import allo
from allo.ir.types import int32, int1, Stream
import allo.dataflow as df

print("allo from:", allo.__file__)

N_ATTEMPTS = 200
DEPTH = 8


@df.region()
def top(out: int32[2]):
    S: Stream[int32, 8][1]

    @df.kernel(mapping=[1], args=[out])
    def producer(out_buf: int32[2]):
        put_ok: int32 = 0
        for i in range(200):
            ok: int1 = S[0].try_put(i)   # non-blocking, no spin
            if ok:
                put_ok += 1
        out_buf[0] = put_ok

    @df.kernel(mapping=[1], args=[out])
    def consumer(out_buf: int32[2]):
        got: int32 = 0
        for k in range(200):
            val, ok = S[0].try_get()      # non-blocking, no spin
            if ok:
                got += 1
        out_buf[1] = got


def main():
    sim = df.build(top, target="simulator")
    print(f"Built simulator. Running {30} times "
          f"(depth={DEPTH}, {N_ATTEMPTS} NB attempts each side)\n")

    results = []
    for run in range(30):
        out = np.zeros(2, dtype=np.int32)
        sim(out)
        put_ok, got = int(out[0]), int(out[1])
        results.append((put_ok, got))
        print(f"  run {run:2d}: producer put_ok = {put_ok:3d}   consumer got = {got:3d}")

    distinct = sorted(set(results))
    print("\n--- summary ---")
    print(f"distinct (put_ok, got) outcomes across 30 runs: {len(distinct)}")
    for r in distinct:
        print("   ", r)
    puts = [p for p, _ in results]
    gots = [g for _, g in results]
    print(f"put_ok range: {min(puts)}..{max(puts)}   got range: {min(gots)}..{max(gots)}")
    if len(distinct) == 1:
        print("\nRESULT: deterministic in THIS environment (single outcome).")
    else:
        print("\nRESULT: NON-DETERMINISTIC — identical program + input produced "
              f"{len(distinct)} different outputs across runs.")
        print("On hardware these counts are a fixed number; the simulator's answer "
              "is decided by the OS scheduler because there is no clock.")


if __name__ == "__main__":
    main()
