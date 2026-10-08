"""U4 ``dma_desc_adapter`` (track C A1 ``bits``) through AMC's vendored Allo
@ fe60c121, per cycle against the Phase 0 trace (one ``f`` call over the
first ``N`` cycles).

    python desc_amc.py <oracles/dma_desc_adapter.npz> <form> <tgt> <sched> <N>

forms: ``written`` (the RTLGen script's ``split`` text -- track C's ``bits``
with 1-D ports -- respelled), ``amc`` (the same with the register discipline
A-D3, single bits as ``d[k:k+1]`` (A2), the sign extension as a typed
shift-or (A8, E-F1)).
"""
import os, re, sys
import numpy as np
from common import load, build, run, verdict, guarded

path, form, tgt, sched, N = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
HERE = os.path.dirname(os.path.abspath(__file__))
txt = open(os.path.join(HERE, "..", "rtlgen", "desc_rtlgen.py")).read()
SRC = txt[txt.index("SRC = '''") + 9:txt.index("'''\nif form")]
sig = ", ".join([f"I{k}: uint64[N]" for k in range(6)] + [f"O{k}: uint64[N]" for k in range(9)])
io = {f"i{k}": f"I{k}[t]" for k in range(6)} | {f"o{k}": f"O{k}[t]" for k in range(9)}
s = SRC.format(N=N, name="desck", sig=sig, **io)
s = s.replace("from allo import kernel\n", "").replace("@kernel\n", "").replace(', name="t"', "")
s = s.replace("from allo.lang import u1, u12, u14, u32, u64, i32, apint", "from allo.ir.types import UInt, uint64, int32")
s = re.sub(r"\bi32\b", "int32", s)
s = re.sub(r"\bu(\d+)\b", r"UInt(\1)", s)
REGS = [("issue", "int32"), ("st_q", "UInt(1)"), ("ch_q", "UInt(1)"), ("va_q", "UInt(12)"), ("rows_q", "UInt(12)"),
        ("base_q", "UInt(32)"), ("stride_q", "UInt(32)")]
if form == "amc":
    for r, ty in REGS:
        s = s.replace(f"    {r}: {ty} = 0\n", f"    {r}_r: {ty}[1] = 0\n", 1)
    top = "".join(f"        {r}: {ty} = {r}_r[0]\n" for r, ty in REGS)
    s = s.replace("    for t in range(N):\n", "    for t in range(N):\n" + top + "        ONE: UInt(32) = 1\n", 1)
    s = s.rstrip("\n") + "\n" + "".join(f"        {r}_r[0] = {r}\n" for r, _ in REGS)
    for k in (55, 53, 27, 28):
        s = s.replace(f"d[{k}]", f"d[{k}:{k + 1}]")
    s = s.replace("disp[24:32] = 0xFF", "disp = disp | ((ONE << 24) * 255)")
K = load(s, f"desc_{form}_{N}")
d = np.load(path)
f = guarded(build, K.desck, tgt, sched, ("S_t_0", "t"), tag=f"desc_{form}")
if f is not None:
    INS = ["rst_n", "start_i", "d_i", "sreg_rd_base_i", "sreg_rd_stride_i", "desc_accept_i"]
    OUTS = ["desc_valid_o", "desc_is_store_o", "desc_channel_o", "desc_vmem_row_o", "desc_rows_o",
            "desc_cols_o", "desc_base_o", "desc_stride_o", "done_o"]
    cols = []
    for p in INS:
        a = d[p].astype(np.uint64)
        cols.append((a[:, 0] | (a[:, 1] << np.uint64(32)) if a.shape[1] > 1 else a[:, 0])[:N])
    r = guarded(run, f, tgt, cols, [((), np.uint64)] * 9, N, N, f"desc_{form}")
    if r is not None:
        outs, cyc = r
        verdict(f"dma_desc_adapter {form} amc-{tgt} {sched}", d,
                {p: (o & 0xFFFFFFFF).astype(np.uint32).reshape(-1, 1) for p, o in zip(OUTS, outs)}, OUTS, N)
