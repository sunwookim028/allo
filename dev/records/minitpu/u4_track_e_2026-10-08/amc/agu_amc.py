"""U4 ``agu_resolve`` (track A C1 ``resolve``) through AMC's vendored Allo @ fe60c121.

    python agu_amc.py <oracles/agu_resolve.npz> <form> <tgt> <sched> <N> [rows]

forms: ``written`` (``resolve`` as track A wrote it: a function over the
row's ``UInt(32)[8]`` ivs returning ``UInt(12)``, called per row), ``amc``
(the body inlined into the row loop, A4; ``if agu_valid`` as ``== 1``).
"""
import sys
import numpy as np
from common import load, build, run, verdict, guarded

path, form, tgt, sched, N = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
rows = int(sys.argv[6]) if len(sys.argv) > 6 else None
HDR = f"from allo.ir.types import UInt, uint1, uint32, int32\nN = {N}\n"
SIG = "(IV: uint32[N, 8], LIT: uint32[N], AV: uint32[N], LVL: uint32[N], SH: uint32[N], OUT: uint32[N])"
WRITTEN = HDR + '''
def resolve(iv: UInt(32)[8], literal: UInt(12), agu_valid: uint1, level: UInt(3), shift: UInt(4)) -> UInt(12):
    lv: int32 = level
    low: UInt(12) = iv[lv]
    sh: UInt(12) = low << shift
    off: UInt(12) = 0
    if agu_valid:
        off = sh
    addr: UInt(12) = literal + off
    return addr

def aguk''' + SIG + ''':
    for t in range(N):
        row: UInt(32)[8] = 0
        for k in range(8):
            row[k] = IV[t, k]
        lit: UInt(12) = LIT[t]
        av: uint1 = AV[t]
        lvl: UInt(3) = LVL[t]
        sh: UInt(4) = SH[t]
        OUT[t] = resolve(row, lit, av, lvl, sh)
'''
AMC = HDR + '''
def aguk''' + SIG + ''':
    for t in range(N):
        literal: UInt(12) = LIT[t]
        agu_valid: UInt(1) = AV[t]
        shift: UInt(4) = SH[t]
        lv: int32 = LVL[t]
        low: UInt(12) = IV[t, lv]
        sh: UInt(12) = low << shift
        off: UInt(12) = 0
        if agu_valid == 1:
            off = sh
        addr: UInt(12) = literal + off
        OUT[t] = addr
'''
K = load(WRITTEN if form == "written" else AMC, f"agu_{form}_{N}")
d = np.load(path)
f = guarded(build, K.aguk, tgt, sched, ("S_t_0", "t"), tag=f"agu_{form}")
if f is not None:
    ins = [d["iv_flat_i"]] + [d[p][:, 0] for p in ("x_literal_i", "x_agu_valid_i", "x_agu_level_i", "x_agu_shift_i")]
    r = guarded(run, f, tgt, ins, [((), np.uint32)], N, rows, f"agu_{form}")
    if r is not None:
        (o,), cyc = r
        verdict(f"agu_resolve {form} amc-{tgt} {sched}", d, {"x_resolved_addr_o": o.reshape(-1, 1)},
                ["x_resolved_addr_o"], len(o))
