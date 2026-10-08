"""U4 ``dma_addr_gen`` through RTLGen (kkkaishao/allo allo-rtlgen @ 13b55a63).

    python addr_rtlgen.py <oracles/dma_addr_gen.npz> <form> <N> [rows]

forms (each the Allo function written in RTLGen's frontend with no change
but the spelling of the types -- ``UInt(46)`` -> ``apint(46, signed=False)``):
  bits_inline  track C's ``addr`` (46-bit product, sum truncated), in the loop body
  bits_call    the same as a nested ``@kernel addr(...) -> u32`` called per row (C1's shape)
  c1_inline    track A's ``addr_gen`` (row widened to 32 bits, 32-bit product)
  c1_call      the same as a nested ``@kernel``
"""
import sys
import numpy as np
from allo import kernel
from allo.lang import u14, u32, apint
from common import report_schedule, run

path, form, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = int(sys.argv[4]) if len(sys.argv) > 4 else None
d = np.load(path)
U46 = apint(46, signed=False)


@kernel
def addr(base: u32, row: u14, stride: u32) -> u32:
    p: U46 = row * stride
    w: u32 = base + p
    return w


@kernel
def addr_gen(base_addr: u32, row: u14, stride: u32) -> u32:
    r: u32 = row
    prod: u32 = r * stride
    word_addr: u32 = base_addr + prod
    return word_addr


@kernel
def bits_call(B: u32[N], R: u32[N], S: u32[N], W: u32[N]):
    for i in range(N, name="i"):
        bb: u32 = B[i]
        rr: u14 = R[i]
        ss: u32 = S[i]
        W[i] = addr(bb, rr, ss)


@kernel
def bits_inline(B: u32[N], R: u32[N], S: u32[N], W: u32[N]):
    for i in range(N, name="i"):
        bb: u32 = B[i]
        rr: u14 = R[i]
        ss: u32 = S[i]
        p: U46 = rr * ss
        w: u32 = bb + p
        W[i] = w


@kernel
def c1_call(B: u32[N], R: u32[N], S: u32[N], W: u32[N]):
    for i in range(N, name="i"):
        rr: u14 = R[i]
        W[i] = addr_gen(B[i], rr, S[i])


@kernel
def c1_inline(B: u32[N], R: u32[N], S: u32[N], W: u32[N]):
    for i in range(N, name="i"):
        rr: u14 = R[i]
        r: u32 = rr
        prod: u32 = r * S[i]
        word_addr: u32 = B[i] + prod
        W[i] = word_addr


k = {"bits_inline": bits_inline, "bits_call": bits_call, "c1_inline": c1_inline, "c1_call": c1_call}[form]
rtl = k.schedule().export("rtl")
report_schedule(rtl, form)
(w,), cyc = run(rtl, [d["base"], d["row"], d["stride"]], [((), np.uint32)], N, max_rows=rows)
n = len(w)
bad = np.flatnonzero(w != d["want"][:n])
print(f"{'UNIT-MATCH' if len(bad) == 0 else 'UNIT-DIFF '} dma_addr_gen {form} rtlgen: {n - len(bad)}/{n} "
      f"(chunk N={N}: {sorted(set(cyc))} cycles)")
for i in bad[:5]:
    print(f"    base {d['base'][i]:#x} row {d['row'][i]:#x} stride {d['stride'][i]:#x}: tool {w[i]:#x} rtl {d['want'][i]:#x}")
