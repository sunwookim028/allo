# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hand-written MINISA assembly, functionally simulated via ``@fx.oracle``.

This is the stream a MINISA encoder would write down, typed out by hand: the
three ``Set*VNLayout`` setters once, then the Load / ExecuteMapping / Store tile
loop that runs the layer — the same shape as the RTL traces (one layout group,
then a run of ``ExecuteMapping``s), with the K dimension reduced by consecutive
accumulating passes into each output block.

An ``@oracle`` body addresses the buffers directly (hand-picked integers), which
is the hand-assembly contract: the author is the allocator. Compare with
``program.py``, where the same stream is *compiled* out of one ``tosa.matmul``
and every address below is the planner's answer instead.

Run:  ``python -m examples.accelerator.featherx_dsa.oracle``
"""

import numpy as np

from .isa import (
    TILE,
    dram,
    fx,
    load_i,
    load_w,
    mac,
    mm,
    set_ivn,
    set_ovn,
    set_wvn,
    store_o,
)


def run_gemm(M: int, K: int, N: int, seed: int = 0) -> None:
    """Emit + simulate ``C[M,N] = A[M,K] @ B[K,N]``, diffed against NumPy.

    The arena is laid out by hand: A's block at row 0, B's below it, C's below
    that — all row-major, all gathered and scattered block by block by the
    relayout movers. On-chip staging reuses one tile slot per buffer (address 0),
    exactly the reuse the compiled path's allocator discovers by liveness."""
    assert all(d % TILE == 0 for d in (M, K, N))
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((M, K)).astype(np.float32)
    b = rng.standard_normal((K, N)).astype(np.float32)
    arena = np.zeros(dram.extents, np.float32)
    arena[0:M, 0:K] = a
    arena[M : M + K, 0:N] = b
    c_row = M + K  # C's block starts below A and B

    @fx.oracle(init={dram: arena})
    def prog():
        set_ivn(r=0, c=0, d=0)
        set_wvn(r=M, c=0, d=TILE)
        set_ovn(r=0, c=0, d=2 * TILE)
        for m in range(0, M, TILE):
            for n in range(0, N, TILE):
                for k in range(0, K, TILE):
                    load_i(r=m, c=k, d=0)
                    load_w(r=M + k, c=n, d=0)
                    if k == 0:
                        mm(a=0, b=0, c=0)
                    else:
                        mac(a=0, b=0, c=0, d=0)
                store_o(s=0, r=c_row + m, c=n)
        fx.inspect(dram, label="dram")

    got = prog()["dram"][c_row : c_row + M, 0:N]
    np.testing.assert_allclose(got, a @ b, rtol=1e-3, atol=1e-3)
    n_em = (M // TILE) * (K // TILE) * (N // TILE)
    print(f"    [ok] {M}x{K}x{N}: 3 setters + {n_em} ExecuteMappings, matches NumPy")


def main() -> None:
    for shape in ((4, 4, 4), (8, 8, 8), (8, 16, 32)):
        run_gemm(*shape)
    print("All hand-written MINISA programs simulated and matched NumPy.")


if __name__ == "__main__":
    main()
