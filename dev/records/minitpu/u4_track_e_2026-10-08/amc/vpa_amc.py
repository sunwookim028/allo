"""U4 ``vpu_adapter`` (track A D-23 ``slots``) through AMC's vendored Allo @ fe60c121.

    python vpa_amc.py <oracles/vpu_adapter.npz> <form> <tgt> <sched> <N> [rows]

forms: ``written`` (``adapt_v/x/m`` as track A wrote them -- the RTLGen
script's bodies, types respelled -- called per row), ``amc`` (inlined, A4;
every slice store ``c[a:b] = v[x:y]`` mechanically rewritten as
``c = c | (((v >> x) & mask) << a)`` (A8), each ``& issue`` kept; typed
constants for shifts above bit 31, E-F1).
"""
import os, re, sys
import numpy as np
from common import load, build, run, verdict, guarded

path, form, tgt, sched, N = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
rows = int(sys.argv[6]) if len(sys.argv) > 6 else None
HERE = os.path.dirname(os.path.abspath(__file__))
HDR = f"from allo.ir.types import UInt, uint1, uint32, int32\nN = {N}\n"
SIG = ("(VI0: uint32[N], VI1: uint32[N], MI: uint32[N], XI: uint32[N], ISS: uint32[N], ROW: uint32[N], "
       "VC0: uint32[N], VC1: uint32[N], XC: uint32[N], MC: uint32[N])")
txt = open(os.path.join(HERE, "..", "rtlgen", "vpa_rtlgen.py")).read()
ns = {"A46": "apint(46, signed=False)", "A47": "apint(47, signed=False)", "A27": "apint(27, signed=False)",
      "A20": "apint(20, signed=False)", "A18": "apint(18, signed=False)"}
exec(txt[txt.index("ADAPT = {"):txt.index("GATE = {")], ns)
ADAPT = ns["ADAPT"]


def respell(s):
    s = re.sub(r"apint\((\d+), signed=False\)", r"UInt(\1)", s)
    return re.sub(r"\bu(\d+)\b", r"UInt(\1)", s)


HEAD = '''
def vpak''' + SIG + ''':
    for t in range(N):
        ONE: UInt(64) = 1
        v: UInt(64) = VI1[t]
        v = (v << 32) | VI0[t]
        x: UInt(64) = XI[t]
        m: UInt(64) = MI[t]
        issue: UInt(64) = ISS[t]
        row: UInt(64) = ROW[t]
'''
STORE = '''        VC0[t] = cv[0:32]
        VC1[t] = cv[32:47]
        XC[t] = cx
        MC[t] = cm
'''


def written():
    L = [HDR]
    for s, (args, rt, body) in ADAPT.items():
        L += [respell(f"def adapt_{s}({args}) -> {rt}:"), respell(body.strip("\n")), f"    return {s}_c", ""]
    L.append(HEAD.replace("v: UInt(64)", "v: UInt(46)").replace("x: UInt(64)", "x: UInt(27)")
             .replace("m: UInt(64)", "m: UInt(8)").replace("issue: UInt(64)", "issue: uint1")
             .replace("row: UInt(64)", "row: UInt(12)").replace("        v = (v << 32) | VI0[t]\n",
                                                                "        v = (v << 32) | VI0[t]\n"))
    L += ["        cv: UInt(47) = adapt_v(v, issue)", "        cx: UInt(20) = adapt_x(x, issue, row)",
          "        cm: UInt(18) = adapt_m(m, issue)"]
    return "\n".join(L) + "\n" + STORE


STORE_RE = re.compile(r"^\s*(\w)_c\[(\d+)(?::(\d+))?\] = (.+)$")
SRC_RE = re.compile(r"^(\w+)\[(\d+)(?::(\d+))?\]$")


def shift_or(s, line):
    """``<s>_c[a:b] = <src>`` -> ``c<s> = c<s> | ((<expr> & mask) << a)``."""
    mt = STORE_RE.match(line)
    a = int(mt.group(2)); b = int(mt.group(3)) if mt.group(3) else a + 1
    rhs = mt.group(4).strip()
    terms = [t.strip() for t in rhs.split("&")]
    exprs = []
    for t in terms:
        ms = SRC_RE.match(t.strip("()"))
        if ms:
            lo = int(ms.group(2))
            exprs.append(f"({ms.group(1)} >> {lo})" if lo else ms.group(1))
        elif "==" in t:
            var, k = [z.strip() for z in t.strip("()").split("==")]
            var = {"m_sub": "((m >> 5) & 7)"}.get(var, var)
            exprs.append(f"(1 if {var} == {k} else 0)")
        else:
            exprs.append({"m_reg": "(m & 31)"}.get(t, t))
    e = " & ".join(exprs) if len(exprs) > 1 else exprs[0]
    return f"        c{s} = c{s} | ((({e}) & {(1 << (b - a)) - 1}) << {a})"


def amc():
    L = [HDR, HEAD]
    for s, (args, rt, body) in ADAPT.items():
        L.append(f"        c{s}: UInt(64) = 0")
        for ln in body.strip("\n").splitlines():
            if STORE_RE.match(ln):
                L.append(shift_or(s, ln))
    return "\n".join(L) + "\n" + STORE


src = written() if form == "written" else amc()
K = load(src, f"vpa_{form}_{N}")
d = np.load(path)
f = guarded(build, K.vpak, tgt, sched, ("S_t_0", "t"), tag=f"vpa_{form}")
if f is not None:
    c1 = lambda p: d[p][:, 0]  # noqa: E731
    ins = [d["v_i"][:, 0], d["v_i"][:, 1], c1("m_i"), c1("x_i"), c1("issue_i"), c1("x_resolved_row_i")]
    r = guarded(run, f, tgt, ins, [((), np.uint32)] * 4, N, rows, f"vpa_{form}")
    if r is not None:
        (v0, v1, xc, mc), cyc = r
        n = len(xc)
        full = [((int(v0[t]) | (int(v1[t]) << 32)) << 38) | (int(xc[t]) << 18) | int(mc[t]) for t in range(n)]
        got = np.array([[(z >> (32 * k)) & 0xFFFFFFFF for k in range(3)] for z in full], dtype=np.uint32)
        verdict(f"vpu_adapter {form} amc-{tgt} {sched}", d, {"vpu_ctrl_o": got}, ["vpu_ctrl_o"], n)
