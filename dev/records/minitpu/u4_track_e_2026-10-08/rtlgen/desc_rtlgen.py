"""U4 ``dma_desc_adapter`` (track C A1 ``bits``, the two-state FSM) through
RTLGen @ 13b55a63, per cycle against the Phase 0 trace (one iteration = one
cycle; state loop-carried, so one cosim over ``rows`` cycles, N >= rows).

    python desc_rtlgen.py <oracles/dma_desc_adapter.npz> <form> <N> [rows]

forms: ``written`` (``units/dma_desc_adapter.py::bits`` respelled: the
region's lane arrays ``IN: u64[N, 6]``, ``OUT: u64[N, 9]``; geometry
constants SL=2, VA=12, RW=12, BW=14 as ``DmaGeometry()`` gives them),
``split`` (each column its own 1-D port).
"""
import os, sys, importlib.util
import numpy as np

path, form, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = int(sys.argv[4]) if len(sys.argv) > 4 else N
TMP = os.environ.get("TMPDIR", "/tmp")
SRC = '''import numpy as np
from allo import kernel
from allo.lang import u1, u12, u14, u32, u64, i32, apint
N = {N}

@kernel
def {name}({sig}):
    issue: i32 = 0
    st_q: u1 = 0
    ch_q: u1 = 0
    va_q: u12 = 0
    rows_q: u12 = 0
    base_q: u32 = 0
    stride_q: u32 = 0
    for t in range(N, name="t"):
        rst: u1 = {i0}
        start: u1 = {i1}
        d: u64 = {i2}
        sb: u32 = {i3}
        ss: u32 = {i4}
        acc: u1 = {i5}
        if rst == 0:
            issue = 0
            st_q = 0
            ch_q = 0
            va_q = 0
            rows_q = 0
            base_q = 0
            stride_q = 0
        vrow: u14 = va_q
        vrow = vrow << 2
        brow: u14 = rows_q
        brow = (brow << 2) | 3
        done: u1 = 0
        if issue == 1:
            done = acc
        {o0} = issue
        {o1} = st_q
        {o2} = ch_q
        {o3} = vrow
        {o4} = brow
        {o5} = 0
        {o6} = base_q
        {o7} = stride_q
        {o8} = done
        if rst == 1:
            if issue == 0:
                if start == 1:
                    st_q = d[55]
                    ch_q = d[53]
                    va_q = d[41:53]
                    rows_q = d[29:41]
                    disp: u32 = d[4:28]
                    if d[27] == 1:
                        disp[24:32] = 0xFF
                    if d[28] == 1:
                        base_q = sb + disp
                    else:
                        base_q = sb
                    stride_q = ss
                    issue = 1
            elif acc == 1:
                issue = 0
'''
if form == "written":
    sig = "CIN: u64[N, 6], COUT: u64[N, 9]"
    io = {f"i{k}": f"CIN[t, {k}]" for k in range(6)} | {f"o{k}": f"COUT[t, {k}]" for k in range(9)}
else:
    sig = ", ".join([f"I{k}: u64[N]" for k in range(6)] + [f"O{k}: u64[N]" for k in range(9)])
    io = {f"i{k}": f"I{k}[t]" for k in range(6)} | {f"o{k}": f"O{k}[t]" for k in range(9)}
src = SRC.format(N=N, name=form, sig=sig, **io)
kp = f"{TMP}/u4e_desc_{form}_{N}.py"
open(kp, "w").write(src); open(f"desc_{form}_kernel.py", "w").write(src)
spec = importlib.util.spec_from_file_location(f"u4e_desc_{form}_{N}", kp)
K = importlib.util.module_from_spec(spec); spec.loader.exec_module(K)
from common import report_schedule, run, verdict  # noqa: E402

d = np.load(path)
rtl = getattr(K, form).schedule().export("rtl")
report_schedule(rtl, form)
INS = ["rst_n", "start_i", "d_i", "sreg_rd_base_i", "sreg_rd_stride_i", "desc_accept_i"]
OUTS = ["desc_valid_o", "desc_is_store_o", "desc_channel_o", "desc_vmem_row_o", "desc_rows_o",
        "desc_cols_o", "desc_base_o", "desc_stride_o", "done_o"]
cols = []
for p in INS:
    a = d[p].astype(np.uint64)
    v = a[:, 0] | (a[:, 1] << np.uint64(32)) if a.shape[1] > 1 else a[:, 0]
    cols.append(v[:rows])
if form == "written":
    (o,), cyc = run(rtl, [np.stack(cols, axis=1)], [((9,), np.uint64)], N, timeout_per_row=200)
    outs = [o[:, k] for k in range(9)]
else:
    outs, cyc = run(rtl, cols, [((), np.uint64)] * 9, N, timeout_per_row=200)
n = len(outs[0])
assert all(p in d for p in OUTS), [p for p in d.files]
verdict(f"dma_desc_adapter {form} rtlgen", d, {p: (o & 0xFFFFFFFF).astype(np.uint32).reshape(-1, 1)
                                               for p, o in zip(OUTS, outs)}, OUTS, n)
