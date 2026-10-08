"""U4 ``dma_addr_gen`` through AMC's vendored Allo (amc-dialect @ fe60c121).

    python addr_amc.py <oracles/dma_addr_gen.npz> <form> <tgt: llvm|amc> <sched: none|pipeline> <N> [rows]

forms: the Allo functions as written (types from AMC's ``allo.ir.types``):
  bits_inline / bits_call   track C's ``addr`` (UInt(46) product)
  c1_inline / c1_call       track A's ``addr_gen`` (UInt(32) product)
The kernel is ``addrk``, not ``top``: a kernel named ``top`` fails FSM emission with
``redefinition of symbol named 'top'`` (finding E-A1).
``*_call`` keeps the function (a scalar-returning call: U1 A4 predicts an
abort on ``amc``); ``*_inline`` pastes its body into the row loop.
"""
import sys
import numpy as np
from common import load, build, run, guarded

path, form, tgt, sched, N = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
rows = int(sys.argv[6]) if len(sys.argv) > 6 else None
FN = {
"bits": '''
def addr(base: UInt(32), row: UInt(14), stride: UInt(32)) -> UInt(32):
    p: UInt(46) = row * stride
    w: UInt(32) = base + p
    return w
''',
"c1": '''
def addr(base_addr: UInt(32), row: UInt(14), stride: UInt(32)) -> UInt(32):
    r: UInt(32) = row
    prod: UInt(32) = r * stride
    word_addr: UInt(32) = base_addr + prod
    return word_addr
'''}
BODY_INLINE = {
"bits": '''        p: UInt(46) = rr * ss
        w: UInt(32) = bb + p
        W[i] = w''',
"c1": '''        r: UInt(32) = rr
        prod: UInt(32) = r * ss
        word_addr: UInt(32) = bb + prod
        W[i] = word_addr'''}
kind, shape = form.split("_")
src = f"from allo.ir.types import UInt, uint32\nN = {N}\n"
if shape == "call":
    src += FN[kind]
src += f'''
def addrk(B: uint32[N], R: uint32[N], S: uint32[N], W: uint32[N]):
    for i in range(N):
        bb: UInt(32) = B[i]
        rr: UInt(14) = R[i]
        ss: UInt(32) = S[i]
'''
src += ("        W[i] = addr(bb, rr, ss)\n" if shape == "call" else BODY_INLINE[kind] + "\n")
K = load(src, f"addr_{form}_{N}")
d = np.load(path)
f = guarded(build, K.addrk, tgt, sched, ("S_i_0", "i"), tag=f"addr_{form}")
if f is not None:
    r = guarded(run, f, tgt, [d["base"], d["row"], d["stride"]], [((), np.uint32)], N, rows, f"addr_{form}")
    if r is not None:
        (w,), cyc = r
        n = len(w)
        bad = np.flatnonzero(w != d["want"][:n])
        print(f"{'UNIT-MATCH' if len(bad) == 0 else 'UNIT-DIFF '} dma_addr_gen {form} amc-{tgt} {sched}: "
              f"{n - len(bad)}/{n} (cycles/chunk {sorted(set(cyc))})", flush=True)
        for i in bad[:5]:
            print(f"    base {d['base'][i]:#x} row {d['row'][i]:#x} stride {d['stride'][i]:#x}: "
                  f"tool {w[i]:#x} rtl {d['want'][i]:#x}")
