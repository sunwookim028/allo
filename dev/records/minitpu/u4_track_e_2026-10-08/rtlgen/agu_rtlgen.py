"""U4 ``agu_resolve`` (track A C1 ``resolve``) through RTLGen @ 13b55a63.

    python agu_rtlgen.py <oracles/agu_resolve.npz> <form> <N> [rows]

forms: ``inline`` (``resolve``'s body in the row loop, the iv lane array read
once at ``IV[t, level]``), ``call`` (``resolve`` a nested ``@kernel`` taking
the row's eight ivs as ``u32[8]``, as track A's region copies them).
Types spelled as RTLGen spells them (``UInt(12)`` -> ``u12``); nothing else
changed unless the script says so.
"""
import sys
import numpy as np
from allo import kernel
from allo.lang import u1, u3, u4, u12, u32, i32
from common import report_schedule, run, verdict

path, form, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = int(sys.argv[4]) if len(sys.argv) > 4 else None
d = np.load(path)


@kernel
def resolve(iv: u32[8], literal: u12, agu_valid: u1, level: u3, shift: u4) -> u12:
    lv: i32 = level
    low: u12 = iv[lv]
    sh: u12 = low << shift
    off: u12 = 0
    if agu_valid:
        off = sh
    addr: u12 = literal + off
    return addr


@kernel
def call(IV: u32[N, 8], LIT: u32[N], AV: u32[N], LVL: u32[N], SH: u32[N], OUT: u32[N]):
    for t in range(N, name="t"):
        row: u32[8]
        for k in range(8, name="k"):
            row[k] = IV[t, k]
        lit: u12 = LIT[t]
        av: u1 = AV[t]
        lvl: u3 = LVL[t]
        sh: u4 = SH[t]
        OUT[t] = resolve(row, lit, av, lvl, sh)


@kernel
def inline(IV: u32[N, 8], LIT: u32[N], AV: u32[N], LVL: u32[N], SH: u32[N], OUT: u32[N]):
    for t in range(N, name="t"):
        literal: u12 = LIT[t]
        agu_valid: u1 = AV[t]
        level: u3 = LVL[t]
        shift: u4 = SH[t]
        lv: i32 = level
        low: u12 = IV[t, lv]
        sh: u12 = low << shift
        off: u12 = 0
        if agu_valid:
            off = sh
        addr: u12 = literal + off
        OUT[t] = addr


k = {"inline": inline, "call": call}[form]
rtl = k.schedule().export("rtl")
report_schedule(rtl, form)
ins = [d["iv_flat_i"]] + [d[p][:, 0] for p in ("x_literal_i", "x_agu_valid_i", "x_agu_level_i", "x_agu_shift_i")]
(o,), cyc = run(rtl, ins, [((), np.uint32)], N, max_rows=rows)
verdict(f"agu_resolve {form} rtlgen", d, {"x_resolved_addr_o": o.reshape(-1, 1)}, ["x_resolved_addr_o"], len(o))
