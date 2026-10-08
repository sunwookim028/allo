"""U4 ``seq_decoder`` (track A C1: ``decode_v/m/x/d/c/l/s`` + the delay field)
through RTLGen @ 13b55a63.

    python dec_rtlgen.py <oracles/seq_decoder.npz> <form> <N> [rows]

The seven bodies below are ``units/seq_decoder.py``'s, with the types spelled
as RTLGen spells them (``UInt(46)`` -> ``apint(46)``) and every local prefixed
by its slot (``f`` -> ``v_f``), so that one text serves both forms:
  call    each body a nested ``@kernel decode_<s>(w: u128) -> apint(k)``,
          called once per row (C1's shape: the issue unit calls them)
  inline  the seven bodies pasted into the row loop (generated text)
  inline_split  ``inline`` with every lane array port split into 1-D ports
          (``B0..B3``, ``V0, V1`` ...): the lane-array port costs II=2 (D3)
The bundle arrives as four 32-bit lanes (RTLGen refuses > 64 bits at the
host boundary, as numpy has no wider dtype) and is assembled into a ``u128``
by slice stores, as track A's region does.
"""
import os, sys, importlib.util
import numpy as np

path, form, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = int(sys.argv[4]) if len(sys.argv) > 4 else None
TMP = os.environ.get("TMPDIR", "/tmp")

BODIES = {  # slot: (result width, body using w, result named <s>_f)
"v": (46, '''
    v_op: u5 = w[123:128]
    v_rd: u5 = w[118:123]
    v_rb: u5 = w[108:113]
    v_f: apint(46, signed=False) = 0
    v_f[41:46] = w[113:118]
    v_f[36:41] = v_rb
    if v_op >= 1 and v_op <= 4:
        v_f[35] = 1
        v_f[21:24] = v_op - 1
        v_f[16:21] = v_rd
    elif v_op == 11 or v_op == 12:
        v_f[35] = 1
        v_f[21:24] = v_op - 7
        v_f[16:21] = v_rd
    elif v_op >= 5 and v_op <= 8:
        v_f[15] = 1
        v_f[13:15] = v_op - 5
        v_f[8:13] = v_rd
    elif v_op == 9 or v_op == 10:
        v_f[7] = 1
        v_f[6] = v_op - 9
        v_f[0:5] = v_rd
    elif v_op == 15:
        v_f[7] = 1
        v_f[6] = v_rb[0]
        v_f[5] = 1
        v_f[0:5] = v_rd
    elif v_op == 13:
        v_f[34] = 1
        v_f[32:34] = v_rb[0:2]
    elif v_op == 14:
        v_f[31] = 1
        v_f[29:31] = v_rb[0:2]
        v_f[24:29] = v_rd
'''),
"m": (8, '''
    m_f: u8 = w[100:108]
'''),
"x": (27, '''
    x_f: apint(27, signed=False) = 0
    if w[98:100] == 1:
        x_p: u32 = w[66:98]
        x_f[26] = 1
        x_f[25] = x_p[31]
        x_f[20:25] = x_p[26:31]
        x_f[8:20] = x_p[14:26]
        x_f[7] = x_p[13]
        x_f[6] = x_p[0]
        x_f[4:6] = x_p[11:13]
        x_f[3] = x_p[7]
        x_f[0:3] = x_p[8:11]
'''),
"d": (57, '''
    d_f: apint(57, signed=False) = 0
    if w[98:100] == 2:
        d_p: u32 = w[66:98]
        d_f[56] = 1
        d_f[55] = d_p[31]
        d_f[53:55] = d_p[29:31]
        d_f[41:53] = d_p[17:29]
        d_f[29:41] = d_p[5:17]
        d_f[28] = d_p[4]
        d_f[2:4] = d_p[2:4]
        d_f[0:2] = d_p[0:2]
        if d_p[4]:
            d_f[4:28] = w[22:46]
'''),
"c": (7, '''
    c_cop: u3 = w[50:53]
    c_f: u7 = 0
    if c_cop == 2:
        c_f[4:7] = 1
    elif c_cop == 3:
        c_f[4:7] = 2
        c_f[0:4] = w[46:50]
    elif c_cop == 4:
        c_f[4:7] = 3
'''),
"l": (45, '''
    l_cop: u3 = w[50:53]
    l_f: apint(45, signed=False) = 0
    if l_cop == 1 or l_cop == 5:
        l_f[44] = 1
        if l_cop == 5:
            l_f[43] = 1
            l_f[42] = w[48]
            l_f[40:42] = w[46:48]
            l_f[0:16] = w[22:38]
        else:
            l_f[16:32] = w[22:38]
        l_f[32:36] = w[38:42]
        l_f[36:40] = w[42:46]
'''),
"s": (36, '''
    s_f: apint(36, signed=False) = 0
    s_f[35] = w[65]
    s_f[32:35] = w[62:65]
    s_f[30:32] = w[60:62]
    s_f[28:30] = w[58:60]
    s_f[27] = w[57]
    s_f[26] = w[53]
    s_f[24:26] = w[55:57]
    if w[65] == 1 and w[62:65] != 1:
        s_f[0:24] = w[22:46]
'''),
}
HDR = ["import numpy as np", "from allo import kernel", "from allo.lang import u1, u3, u5, u7, u8, u32, u128, apint",
       f"N = {N}", ""]
LOOP_HEAD = '''
@kernel
def {name}(B: u32[N, 4], V: u32[N, 2], D: u32[N, 2], L: u32[N, 2], S: u32[N, 2], M: u32[N], X: u32[N], C: u32[N], DL: u32[N]):
    for t in range(N, name="t"):
        w: u128 = 0
        w[0:32] = B[t, 0]
        w[32:64] = B[t, 1]
        w[64:96] = B[t, 2]
        w[96:128] = B[t, 3]
'''
STORE = '''
        V[t, 0] = fv[0:32]
        V[t, 1] = fv[32:46]
        D[t, 0] = fd[0:32]
        D[t, 1] = fd[32:57]
        L[t, 0] = fl[0:32]
        L[t, 1] = fl[32:45]
        S[t, 0] = fs[0:32]
        S[t, 1] = fs[32:36]
        M[t] = fm
        X[t] = fx
        C[t] = fc
        DL[t] = w[15:22]
'''
WT = {s: (f"apint({wd}, signed=False)" if wd not in (7, 8) else f"u{wd}") for s, (wd, _) in BODIES.items()}


def indent(body, k):
    return "\n".join((" " * k + ln[4:]) if ln.strip() else "" for ln in body.strip("\n").splitlines())


SPLIT_HEAD = '''
@kernel
def {name}(B0: u32[N], B1: u32[N], B2: u32[N], B3: u32[N], V0: u32[N], V1: u32[N], D0: u32[N], D1: u32[N],
           L0: u32[N], L1: u32[N], S0: u32[N], S1: u32[N], M: u32[N], X: u32[N], C: u32[N], DL: u32[N]):
    for t in range(N, name="t"):
        w: u128 = 0
        w[0:32] = B0[t]
        w[32:64] = B1[t]
        w[64:96] = B2[t]
        w[96:128] = B3[t]
'''


def gen(form):
    if form == "inline_split":
        txt = gen("inline").replace(LOOP_HEAD.format(name="inline"), SPLIT_HEAD.format(name=form))
        for a in "VDLS":
            txt = txt.replace(f"{a}[t, 0]", f"{a}0[t]").replace(f"{a}[t, 1]", f"{a}1[t]")
        return txt
    L = list(HDR)
    if form == "call":
        for s, (wd, body) in BODIES.items():
            L += ["@kernel", f"def decode_{s}(w: u128) -> {WT[s]}:", indent(body, 4), f"    return {s}_f", ""]
        L.append(LOOP_HEAD.format(name=form))
        for s in BODIES:
            L.append(f"        f{s}: {WT[s]} = decode_{s}(w)")
    else:
        L.append(LOOP_HEAD.format(name=form))
        for s, (wd, body) in BODIES.items():
            L.append(indent(body, 8))
            L.append(f"        f{s}: {WT[s]} = {s}_f")
    L.append(STORE)
    return "\n".join(L) + "\n"


src = gen(form)
kp = f"{TMP}/u4e_dec_{form}_{N}.py"
open(kp, "w").write(src)
open(f"dec_{form}_kernel.py", "w").write(src)
spec = importlib.util.spec_from_file_location(f"u4e_dec_{form}_{N}", kp)
K = importlib.util.module_from_spec(spec); spec.loader.exec_module(K)
from common import report_schedule, run, verdict  # noqa: E402

d = np.load(path)
rtl = getattr(K, form).schedule().export("rtl")
report_schedule(rtl, form)
spec_out = [((2,), np.uint32)] * 4 + [((), np.uint32)] * 4
if form == "inline_split":
    o, cyc = run(rtl, [d["bundle_i"][:, k] for k in range(4)], [((), np.uint32)] * 12, N, max_rows=rows)
    st = lambda a, b: np.stack([a, b], axis=1)  # noqa: E731
    v, dd, l, s = st(o[0], o[1]), st(o[2], o[3]), st(o[4], o[5]), st(o[6], o[7])
    m, x, c, dl = o[8:12]
else:
    (v, dd, l, s, m, x, c, dl), cyc = run(rtl, [d["bundle_i"]], spec_out, N, max_rows=rows)
j2 = lambda a: a[:, 0].astype(object) | (a[:, 1].astype(object) << 32)  # noqa: E731
f = np.zeros(len(m), dtype=object)
for val, wd in ((j2(v), 46), (m.astype(object), 8), (x.astype(object), 27), (j2(dd), 57), (c.astype(object), 7),
                (j2(l), 45), (j2(s), 36), (dl.astype(object), 7)):
    f = (f << wd) | val
got = np.array([[(int(z) >> (32 * k)) & 0xFFFFFFFF for k in range(8)] for z in f], dtype=np.uint32)
verdict(f"seq_decoder {form} rtlgen", d, {"fields_o": got}, ["fields_o"], len(m))
