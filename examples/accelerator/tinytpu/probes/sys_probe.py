"""A real DIM x DIM weight-stationary systolic array in Allo, verified in RTL.

Structure copied from ~/core/npu/src/core/mxu/mxu_systolic_array.sv: weights are
stationary in the PEs, the left operand flows west->east along a row, and the
partial sum flows north->south down a column, so the contraction index is the
row index and each column's bottom edge emits one finished output element.

The only reason to write it time-stepped with an explicit next-state commit is
that this is the RTL semantics: every PE reads *its neighbour's register* (last
cycle's value) and writes *its own*, so there is no combinational chain across
the array and no accumulator recurrence inside a PE. Fully unrolling the two
spatial loops then has to materialise DIM*DIM multipliers and DIM*DIM adders --
which is the claim, and it is checkable in the emitted Verilog.
"""
import re
import sys
import numpy as np
from allo.lang.core import f32, i32, range as arange
from allo import kernel

DIM = 4
NSTEP = 3 * DIM - 2   # skew: t = m + (DIM-1) + j, m,j in [0,DIM)
BR = 64


@kernel
def mxu_sys(bram: f32[BR], w_addr: i32, x_addr: i32, z_addr: i32):
    """Z = X @ W^T over DIM x DIM tiles in bram (same ISA contract as mxu)."""
    Xt: f32[DIM, DIM]
    Wst: f32[DIM, DIM]
    a_reg: f32[DIM, DIM]
    ps_reg: f32[DIM, DIM]
    a_nxt: f32[DIM, DIM]
    ps_nxt: f32[DIM, DIM]
    Zt: f32[DIM, DIM]

    # Stage the operands. Wst[k][n] = W[n][k]: the weight is consumed
    # transposed, so the stationary value in PE[i][j] is W[j][i].
    for li in arange(DIM, name="li"):
        for lk in arange(DIM, name="lk"):
            Xt[li, lk] = bram[x_addr + li * DIM + lk]
            Wst[lk, li] = bram[w_addr + li * DIM + lk]

    for zi in arange(DIM, name="zi"):
        for zj in arange(DIM, name="zj"):
            a_reg[zi, zj] = 0.0
            ps_reg[zi, zj] = 0.0
            Zt[zi, zj] = 0.0

    for t in arange(NSTEP, name="t"):
        # --- one clock of the array: DIM*DIM PEs, each one MAC ---
        for i in arange(DIM, name="pi"):
            for j in arange(DIM, name="pj"):
                av: f32 = 0.0
                if j == 0:
                    m: i32 = t - i          # west edge feed, skewed by row
                    if m >= 0:
                        if m < DIM:
                            av = Xt[m, i]
                else:
                    av = a_reg[i, j - 1]    # from the PE to the west
                pv: f32 = 0.0
                if i > 0:
                    pv = ps_reg[i - 1, j]   # from the PE to the north
                a_nxt[i, j] = av
                ps_nxt[i, j] = pv + av * Wst[i, j]

        # --- bottom edge: a column emits C[m][j] once the psum has fallen
        #     through all DIM rows ---
        for oj in arange(DIM, name="oj"):
            mo: i32 = t - (DIM - 1) - oj
            if mo >= 0:
                if mo < DIM:
                    Zt[mo, oj] = ps_nxt[DIM - 1, oj]

        for ci in arange(DIM, name="ci"):
            for cj in arange(DIM, name="cj"):
                a_reg[ci, cj] = a_nxt[ci, cj]
                ps_reg[ci, cj] = ps_nxt[ci, cj]

    for di in arange(DIM, name="di"):
        for dj in arange(DIM, name="dj"):
            bram[z_addr + di * DIM + dj] = Zt[di, dj]


def build():
    s = mxu_sys.schedule()
    for b in ("Xt", "Wst", "a_reg", "ps_reg", "a_nxt", "ps_nxt", "Zt"):
        s.partition(s.buffer(b), kind=s.Complete)
    s.pipeline("lk")
    for ln in ("pi", "pj", "oj", "ci", "cj", "zj"):
        s.unroll(ln)
    s.pipeline("t")
    s.pipeline("dj")
    return s


def golden(X, W):
    return (X @ W.T).astype(np.float32)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    s = build()
    rng = np.random.default_rng(0)
    X = rng.standard_normal((DIM, DIM)).astype(np.float32)
    W = rng.standard_normal((DIM, DIM)).astype(np.float32)
    W_ADDR, X_ADDR, Z_ADDR = 0, 16, 32
    bram = np.zeros(BR, np.float32)
    bram[W_ADDR:W_ADDR + 16] = W.ravel()
    bram[X_ADDR:X_ADDR + 16] = X.ravel()

    if what == "cpu":
        mod = s.export("cpu")
        mod(bram, W_ADDR, X_ADDR, Z_ADDR)
        got = bram[Z_ADDR:Z_ADDR + 16].reshape(DIM, DIM)
        gold = golden(X, W)
        print("max err", np.abs(got - gold).max())
        print("correct", np.allclose(got, gold, rtol=1e-4, atol=1e-4))
    elif what == "rtl":
        r = s.export("rtl")
        v = r.verilog
        open("sys_mxu.sv", "w").write(v)
        nmul = len(re.findall(r"^\s*mul_f32\S*\s+\S+\s*\(", v, re.M))
        nadd = len(re.findall(r"^\s*add_f32\S*\s+\S+\s*\(", v, re.M))
        rep = r.report
        print(f"mul instances = {nmul}   add instances = {nadd}   (DIM*DIM = {DIM*DIM})")
        print("II / latency:", rep)
    elif what == "cosim":
        r = s.export("rtl")
        res = r.cosim(bram, W_ADDR, X_ADDR, Z_ADDR)
        got = bram[Z_ADDR:Z_ADDR + 16].reshape(DIM, DIM)
        gold = golden(X, W)
        print(f"cycles={res.cycles}  correct={np.allclose(got, gold, rtol=1e-4, atol=1e-4)}"
              f"  maxerr={np.abs(got - gold).max():.2e}")
