from __future__ import annotations
from allo.compose import unit

@unit(memories=("WT", "NT"), reads=("lhsx", "wx", "ctl"), parameters=("N", "D"))
def front_sink(wt_o: UInt(32)[N * D], nt_o: UInt(64)[N * D]):
    for t in range(N):
        c_t: UInt(32) = ctl.get()
        with allo.meta_for(D) as row:
            wt_o[t * D + row] = lhsx[row, 0].get()
        with allo.meta_for(D) as col:
            nt_o[t * D + col] = wx[0, col].get()
