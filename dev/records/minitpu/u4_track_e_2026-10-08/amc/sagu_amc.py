"""U4 ``scalar_agu`` (track A S1, ``S_LAT`` = 2) through AMC's vendored Allo
@ fe60c121, per cycle against the Phase 0 trace (one ``f`` call over the first
``rows`` cycles: the state is loop-carried).

    python sagu_amc.py <oracles/scalar_agu.npz> <form> <tgt> <sched> <N>

forms: ``written`` (the RTLGen script's ``written`` text -- track A's ``s1``
-- respelled for AMC's types), ``amc`` (the RTLGen ``regs`` machine with
U3's AMC register discipline, A-D3: every register a 1-element array read
once at the top of the iteration and written once at the bottom; single bits
as ``x[k:k+1]`` (A2), no slice stores (A8), typed wide constants (E-F1),
``scalar_result`` inlined (A4)).
"""
import os, re, sys
import numpy as np
from common import load, build, run, verdict, guarded

path, form, tgt, sched, N = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
HERE = os.path.dirname(os.path.abspath(__file__))
txt = open(os.path.join(HERE, "..", "rtlgen", "sagu_rtlgen.py")).read()
ns = {"N": N, "__name__": "x", "os": os}
exec(txt[txt.index("SL = 2"):txt.index("src = WRITTEN")].replace("SL = 2", f"SL = 2\nTMP='/tmp'\nN={N}"), ns)


def respell(s):
    s = s.replace("from allo import kernel\n", "").replace("@kernel\n", "")
    s = s.replace("from allo.lang import u1, u2, u3, u4, u5, u32, i32, apint",
                  "from allo.ir.types import UInt, uint32, int32")
    s = s.replace("u24 = apint(24, signed=False)  # no u24 alias in allo.lang\n", "")
    s = re.sub(r"\bu32\[N", "uint32[N", s)
    s = re.sub(r"\bi32\b", "int32", s)
    s = re.sub(r"\bu(\d+)\b", r"UInt(\1)", s)
    s = s.replace(', name="t"', "").replace(', name="k0"', "").replace(', name="k1"', "").replace(', name="k2"', "")
    return s.replace(', name="j"', "")


REGS = [f"sreg{i}" for i in range(4)] + ["written"] + [f"p{k}_{f}" for k in range(2)
                                                      for f in ("v", "rd", "data", "prod", "mac")]
TY = {"written": "UInt(4)", "v": "UInt(1)", "rd": "UInt(2)", "mac": "UInt(1)"}


def amc():
    s = respell(ns["regs_src"]())
    # register discipline (A-D3)
    for r in REGS:
        ty = TY.get(r.split("_")[-1], TY.get(r, "UInt(32)"))
        s = s.replace(f"    {r}: {ty} = 0\n", f"    {r}_r: {ty}[1] = 0\n", 1)
    top = "".join(f"        {r}: {TY.get(r.split('_')[-1], TY.get(r, 'UInt(32)'))} = {r}_r[0]\n" for r in REGS)
    s = s.replace("    for t in range(N):\n", "    for t in range(N):\n" + top + "        ONE: UInt(32) = 1\n", 1)
    s = s.rstrip("\n") + "\n" + "".join(f"        {r}_r[0] = {r}\n" for r in REGS)
    # A2 / A8 / E-F1 / truthiness
    s = s.replace("if imm[23]:", "if imm[23:24] == 1:")
    s = s.replace("e[24:32] = 0xFF", "e = e | ((ONE << 24) * 255)")
    s = s.replace("if LA[t]:", "if LA[t] == 1:").replace("if SIV[t]:", "if SIV[t] == 1:")
    s = s.replace("if live:", "if live == 1:").replace("if p1_v:", "if p1_v == 1:").replace("if p1_mac:", "if p1_mac == 1:")
    s = s.replace("p0_mac = sop == 3", "p0_mac = 1 if sop == 3 else 0")
    s = s.replace("def regs(", "def saguk(")
    return s


src = respell(ns["WRITTEN"]).replace("def written(", "def saguk(") if form == "written" else amc()
K = load(src, f"sagu_{form}_{N}")
d = np.load(path)
f = guarded(build, K.saguk, tgt, sched, ("S_t_0", "t"), tag=f"sagu_{form}")
if f is not None:
    c = lambda p: d[p][:, 0]  # noqa: E731
    ins = [c("rst_n"), c("s_valid_i"), c("s_op_i"), c("s_rd_i"), c("s_rs_i"), c("s_use_iv_i"), c("s_level_i"),
           c("s_imm_i"), d["kernel_arg_csr_i"], d["iv_flat_i"], c("rd_sel_d_base_i"), c("rd_sel_d_stride_i"),
           c("rd_loop_bound_from_arg_i"), c("rd_sel_loop_bound_i")]
    r = guarded(run, f, tgt, [a[:N] for a in ins], [((4,), np.uint32)] + [((), np.uint32)] * 4, N, N, f"sagu_{form}")
    if r is not None:
        (sreg, ob, os_, ol, ow), cyc = r
        verdict(f"scalar_agu:lat2 {form} amc-{tgt} {sched}", d,
                {"sreg_o": sreg, "rd_data_d_base_o": ob.reshape(-1, 1), "rd_data_d_stride_o": os_.reshape(-1, 1),
                 "rd_data_loop_bound_o": ol.reshape(-1, 1), "sreg_written_o": ow.reshape(-1, 1)},
                ["sreg_o", "rd_data_d_base_o", "rd_data_d_stride_o", "rd_data_loop_bound_o", "sreg_written_o"], N)
