"""Cycle-exact python model of mmu's loop, to settle the skew before compiling."""
import numpy as np
DIM = 4

def mmu(X, W, rows, PHOP, acc_in=None):
    """X[rows,DIM] activations, W[n,k] weights. Returns C[rows,DIM]."""
    SKEW = (DIM - 1) * (1 + PHOP)
    NDL = PHOP        # slots 0..PHOP-1; slot PHOP-1 is PHOP iterations old
    wst = np.zeros((DIM, DIM))
    for n in range(DIM):
        for k in range(DIM):
            wst[k, n] = W[n, k]
    a_reg = np.zeros((DIM, DIM))
    ps_dl = np.zeros((DIM, DIM, NDL))
    odl = np.zeros((DIM, DIM))
    acc = np.zeros((rows, DIM)) if acc_in is None else acc_in.copy()
    zero_first = acc_in is None

    def Xr(m, i):
        return X[m, i] if 0 <= m < rows else 0.0
    def Ar(m, j):
        return acc[m, j] if 0 <= m < rows else 0.0

    for t in range(rows + SKEW):
        a_nxt = np.zeros((DIM, DIM)); ps_new = np.zeros((DIM, DIM))
        for i in range(DIM):
            for j in range(DIM):
                ma = t - i * PHOP
                av = Xr(ma, i) if j == 0 else a_reg[i, j - 1]
                pv = ps_dl[i - 1, j, PHOP - 1] if i > 0 else 0.0
                a_nxt[i, j] = av
                ps_new[i, j] = pv + av * wst[i, j]
        nodl = odl.copy()
        for sd in range(DIM - 1):
            nodl[:, DIM - 1 - sd] = odl[:, DIM - 2 - sd]
        nodl[:, 0] = ps_new[DIM - 1, :]
        odl = nodl
        for oj in range(DIM):
            mo = t - (DIM - 1) - (DIM - 1) * PHOP
            if 0 <= mo < rows:            # the pad rows are off-array here
                prev = 0.0 if zero_first else acc[mo, oj]
                acc[mo, oj] = prev + odl[oj, DIM - 1 - oj]
        a_reg = a_nxt
        nd = ps_dl.copy()
        for d in range(PHOP - 1, 0, -1):
            nd[:, :, d] = ps_dl[:, :, d - 1]
        nd[:, :, 0] = ps_new
        ps_dl = nd
    return acc

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    rows = 6
    X = rng.standard_normal((rows, DIM))
    W = rng.standard_normal((DIM, DIM))
    gold = X @ W.T
    for ph in (1, 2, 3, 4):
        got = mmu(X, W, rows, ph)
        print(f"PHOP={ph}  fresh  maxerr={np.abs(got-gold).max():.3e}")
    # accumulate path
    prev = rng.standard_normal((rows, DIM))
    for ph in (1, 2, 3, 4):
        got = mmu(X, W, rows, ph, acc_in=prev)
        print(f"PHOP={ph}  accum  maxerr={np.abs(got-(prev+gold)).max():.3e}")
