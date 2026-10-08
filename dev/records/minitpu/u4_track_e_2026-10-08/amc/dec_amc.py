"""U4 ``seq_decoder`` (track A C1) through AMC's vendored Allo @ fe60c121.

    python dec_amc.py <oracles/seq_decoder.npz> <form> <tgt> <sched> <N> [rows]

forms:
  written  the seven ``decode_*`` functions as track A wrote them (slice
           stores ``f[a:b] = v``, single-bit ``x[k]``, scalar-returning calls),
           types from AMC's ``allo.ir.types`` (the RTLGen script's bodies,
           ``apint(k)`` -> ``UInt(k)``): what AMC does with C1 as written
  amc      the same functions with U1's AMC workarounds applied: inlined (A4),
           every field insert a shift-and-or (A8), single bits as ``x[k:k+1]``
           (A2), no three-operand booleans (A5); 1-D lane ports
"""
import os, sys, re
import numpy as np
from common import load, build, run, verdict, guarded

path, form, tgt, sched, N = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
rows = int(sys.argv[6]) if len(sys.argv) > 6 else None
HERE = os.path.dirname(os.path.abspath(__file__))
HDR = f"from allo.ir.types import UInt, uint32, int32\nN = {N}\n"
SIG = ("(B0: uint32[N], B1: uint32[N], B2: uint32[N], B3: uint32[N], V0: uint32[N], V1: uint32[N], "
       "D0: uint32[N], D1: uint32[N], L0: uint32[N], L1: uint32[N], S0: uint32[N], S1: uint32[N], "
       "M: uint32[N], X: uint32[N], C: uint32[N], DL: uint32[N])")
HEAD = '''
def deck''' + SIG + ''':
    for t in range(N):
        w: UInt(128) = B3[t]
        w = (w << 32) | B2[t]
        w = (w << 32) | B1[t]
        w = (w << 32) | B0[t]
'''
STORE = '''        V0[t] = fv[0:32]
        V1[t] = fv[32:46]
        D0[t] = fd[0:32]
        D1[t] = fd[32:57]
        L0[t] = fl[0:32]
        L1[t] = fl[32:45]
        S0[t] = fs[0:32]
        S1[t] = fs[32:36]
        M[t] = fm
        X[t] = fx
        C[t] = fc
        DL[t] = w[15:22]
'''


def written():
    sys.path.insert(0, os.path.join(HERE, "..", "rtlgen"))
    txt = open(os.path.join(HERE, "..", "rtlgen", "dec_rtlgen.py")).read()
    ns = {}
    exec(txt[txt.index("BODIES = {"):txt.index("HDR = [")], ns)
    L = [HDR]
    W = {"v": 46, "m": 8, "x": 27, "d": 57, "c": 7, "l": 45, "s": 36}
    for s, (wd, body) in ns["BODIES"].items():
        body = re.sub(r"apint\((\d+), signed=False\)", r"UInt(\1)", body)
        body = re.sub(r"\bu(\d+)\b", r"UInt(\1)", body)
        L += [f"def decode_{s}(w: UInt(128)) -> UInt({W[s]}):",
              "\n".join(ln for ln in body.strip("\n").splitlines()), f"    return {s}_f", ""]
    L.append(HEAD)
    for s in W:
        L.append(f"        f{s}: UInt({W[s]}) = decode_{s}(w)")
    return "\n".join(L) + "\n" + STORE


def ins(var, w, expr, lo, width, ind=8):
    """``var[lo:lo+width] = expr`` as a shift-and-or (A8)."""
    sp = " " * ind
    return [f"{sp}{var} = {var} | ((({expr}) & {(1 << width) - 1}) << {lo})"] if width < w else [f"{sp}{var} = {expr}"]


def amc():
    L = [HDR, HEAD]
    sp = " " * 8
    # an untyped literal is int32: ``1 << 35`` folds to garbage (AMC frontend; E-A4) -- a typed one
    L += [f"{sp}ONE: UInt(64) = 1"]
    # V slot
    L += [f"{sp}v_op: UInt(46) = w[123:128]", f"{sp}v_rd: UInt(46) = w[118:123]", f"{sp}v_rb: UInt(46) = w[108:113]",
          f"{sp}v_ra: UInt(46) = w[113:118]", f"{sp}fv: UInt(46) = (v_ra << 41) | (v_rb << 36)"]
    L += [f"{sp}if v_op >= 1:", f"{sp}    if v_op <= 4:",
          f"{sp}        fv = fv | (ONE << 35) | (((v_op - 1) & 7) << 21) | (v_rd << 16)",
          f"{sp}if v_op == 11 or v_op == 12:", f"{sp}    fv = fv | (ONE << 35) | (((v_op - 7) & 7) << 21) | (v_rd << 16)",
          f"{sp}if v_op >= 5:", f"{sp}    if v_op <= 8:", f"{sp}        fv = fv | (ONE << 15) | (((v_op - 5) & 3) << 13) | (v_rd << 8)",
          f"{sp}if v_op == 9 or v_op == 10:", f"{sp}    fv = fv | (ONE << 7) | (((v_op - 9) & 1) << 6) | v_rd",
          f"{sp}if v_op == 15:", f"{sp}    fv = fv | (ONE << 7) | ((v_rb & 1) << 6) | (ONE << 5) | v_rd",
          f"{sp}if v_op == 13:", f"{sp}    fv = fv | (ONE << 34) | ((v_rb & 3) << 32)",
          f"{sp}if v_op == 14:", f"{sp}    fv = fv | (ONE << 31) | ((v_rb & 3) << 29) | (v_rd << 24)"]
    # M
    L += [f"{sp}fm: UInt(8) = w[100:108]"]
    # X
    L += [f"{sp}kind: UInt(2) = w[98:100]", f"{sp}p: UInt(64) = w[66:98]", f"{sp}fx: UInt(27) = 0",
          f"{sp}if kind == 1:",
          f"{sp}    fx = (ONE << 26) | (((p >> 31) & 1) << 25) | (((p >> 26) & 31) << 20) | (((p >> 14) & 4095) << 8) "
          f"| (((p >> 13) & 1) << 7) | ((p & 1) << 6) | (((p >> 11) & 3) << 4) | (((p >> 7) & 1) << 3) | ((p >> 8) & 7)"]
    # D
    L += [f"{sp}fd: UInt(57) = 0", f"{sp}if kind == 2:",
          f"{sp}    pd: UInt(57) = p",
          f"{sp}    fd = (ONE << 56) | (((pd >> 31) & 1) << 55) | (((pd >> 29) & 3) << 53) | (((pd >> 17) & 4095) << 41) "
          f"| (((pd >> 5) & 4095) << 29) | (((pd >> 4) & 1) << 28) | (((pd >> 2) & 3) << 2) | (pd & 3)",
          f"{sp}    if ((p >> 4) & 1) == 1:", f"{sp}        imm_d: UInt(57) = w[22:46]", f"{sp}        fd = fd | (imm_d << 4)"]
    # C
    L += [f"{sp}cop: UInt(3) = w[50:53]", f"{sp}fc: UInt(7) = 0", f"{sp}if cop == 2:", f"{sp}    fc = 16",
          f"{sp}if cop == 3:", f"{sp}    fcl: UInt(7) = w[46:50]", f"{sp}    fc = 32 | fcl", f"{sp}if cop == 4:", f"{sp}    fc = 48"]
    # L
    L += [f"{sp}fl: UInt(45) = 0", f"{sp}lstep: UInt(45) = w[38:42]", f"{sp}llo: UInt(45) = w[42:46]",
          f"{sp}limm: UInt(45) = w[22:38]",
          f"{sp}if cop == 1:", f"{sp}    fl = (ONE << 44) | (llo << 36) | (lstep << 32) | (limm << 16)",
          f"{sp}if cop == 5:", f"{sp}    lfa: UInt(45) = w[48:49]", f"{sp}    lidx: UInt(45) = w[46:48]",
          f"{sp}    fl = (ONE << 44) | (ONE << 43) | (lfa << 42) | (lidx << 40) | (llo << 36) | (lstep << 32) | limm"]
    # S
    L += [f"{sp}sv: UInt(36) = w[65:66]", f"{sp}sop: UInt(36) = w[62:65]", f"{sp}srd: UInt(36) = w[60:62]",
          f"{sp}srs: UInt(36) = w[58:60]", f"{sp}siv: UInt(36) = w[57:58]", f"{sp}slh: UInt(36) = w[53:54]",
          f"{sp}sll: UInt(36) = w[55:57]",
          f"{sp}fs: UInt(36) = (sv << 35) | (sop << 32) | (srd << 30) | (srs << 28) | (siv << 27) | (slh << 26) | (sll << 24)",
          f"{sp}if sv == 1:", f"{sp}    if sop != 1:", f"{sp}        simm: UInt(36) = w[22:46]", f"{sp}        fs = fs | simm"]
    return "\n".join(L) + "\n" + STORE


src = written() if form == "written" else amc()
K = load(src, f"dec_{form}_{N}")
d = np.load(path)
f = guarded(build, K.deck, tgt, sched, ("S_t_0", "t"), tag=f"dec_{form}")
if f is not None:
    r = guarded(run, f, tgt, [d["bundle_i"][:, k] for k in range(4)], [((), np.uint32)] * 12, N, rows, f"dec_{form}")
    if r is not None:
        o, cyc = r
        j = lambda a, b: a.astype(object) | (b.astype(object) << 32)  # noqa: E731
        fz = np.zeros(len(o[0]), dtype=object)
        for val, wd in ((j(o[0], o[1]), 46), (o[8].astype(object), 8), (o[9].astype(object), 27), (j(o[2], o[3]), 57),
                        (o[10].astype(object), 7), (j(o[4], o[5]), 45), (j(o[6], o[7]), 36), (o[11].astype(object), 7)):
            fz = (fz << wd) | val
        got = np.array([[(int(z) >> (32 * k)) & 0xFFFFFFFF for k in range(8)] for z in fz], dtype=np.uint32)
        verdict(f"seq_decoder {form} amc-{tgt} {sched}", d, {"fields_o": got}, ["fields_o"], len(o[0]))
