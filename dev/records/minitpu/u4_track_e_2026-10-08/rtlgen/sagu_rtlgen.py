"""U4 ``scalar_agu`` (track A S1, ``S_LAT`` = 2 = the shipped ``lat2``) through
RTLGen @ 13b55a63, per cycle against the Phase 0 trace (one iteration = one
cycle, ``pre`` outputs first, then the edge; the pipe is data).

    python sagu_rtlgen.py <oracles/scalar_agu.npz> <form> <N> [rows]

The state is loop-carried, so a run is ONE cosim over the first ``rows``
cycles (N >= rows); chunking would reset the state.
forms:
  written  ``units/scalar_agu.py::s1`` respelled for RTLGen: local arrays
           ``sreg[4]``, the pipe as ``[S_LAT]`` arrays, inner ``for k`` loops,
           ``scalar_result`` a nested ``@kernel``
  regs     the same machine with every register a scalar (``sreg0..3``,
           ``p0_v .. p1_mac``), inner loops expanded, ``scalar_result`` and
           dynamic register selects (``sreg[i]``) as if-chains: the U3
           "RTLGen cell" spelling (generated text)
  written_e1  ``written`` with E-R1 avoided: ``scalar_result`` takes ``op`` as
           ``u32`` and narrows it to a local ``u3`` (a narrow unsigned
           PARAMETER compared with 4 is never equal; ``repro_param_cmp.py``)
  regs_split  ``regs`` with the four-lane ``SREG[N, 4]`` output split into
           four 1-D ports (the lane array's 4 writes per iteration cost II=4)
"""
import os, sys, importlib.util
import numpy as np

path, form, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = int(sys.argv[4]) if len(sys.argv) > 4 else N
SL = 2
TMP = os.environ.get("TMPDIR", "/tmp")
HDR = f'''import numpy as np
from allo import kernel
from allo.lang import u1, u2, u3, u4, u5, u32, i32, apint
u24 = apint(24, signed=False)  # no u24 alias in allo.lang
N = {N}
SL = {SL}
'''
SIG = '''(RST: u32[N], SV: u32[N], SOP: u32[N], SRD: u32[N], SRS: u32[N], SIV: u32[N], SLV: u32[N],
        SIMM: u32[N], KA: u32[N, 4], IV: u32[N, 8], SB: u32[N], SS: u32[N], LA: u32[N], SLB: u32[N],
        SREG: u32[N, 4], OB: u32[N], OS: u32[N], OL: u32[N], OW: u32[N])'''

WRITTEN = HDR + '''
@kernel
def scalar_result(op: u3, base: u32, imm: u24, arg: u32) -> u32:
    r: u32 = 0
    if op == 0:
        r = imm
    elif op == 1:
        r = arg
    elif op == 2:
        e: u32 = imm
        if imm[23]:
            e[24:32] = 0xFF
        r = base + e
    elif op == 3:
        r = base
    elif op == 4:
        sh: u5 = imm[0:5]
        r = base << sh
    return r


@kernel
def written''' + SIG + ''':
    sreg: u32[4] = 0
    written: u4 = 0
    pv: u1[SL] = 0
    prd: u2[SL] = 0
    pdata: u32[SL] = 0
    pprod: u32[SL] = 0
    pmac: u1[SL] = 0
    for t in range(N, name="t"):
        live: u1 = RST[t]
        if live == 0:
            written = 0
            for k in range(4, name="k0"):
                sreg[k] = 0
            for k in range(SL, name="k1"):
                pv[k] = 0
                prd[k] = 0
                pdata[k] = 0
                pprod[k] = 0
                pmac[k] = 0
        for k in range(4, name="k2"):
            SREG[t, k] = sreg[k]
        ib: i32 = SB[t]
        OB[t] = sreg[ib]
        i_s: i32 = SS[t]
        OS[t] = sreg[i_s]
        il: i32 = SLB[t]
        raw: u32 = sreg[il]
        if LA[t]:
            raw = KA[t, il]
        bound: u32 = raw[0:16]
        if raw[16:32] != 0:
            bound = 0xFFFF
        OL[t] = bound
        OW[t] = written
        if live:
            irs: i32 = SRS[t]
            ilv: i32 = SLV[t]
            imm: u24 = SIMM[t]
            operand: u32 = sreg[irs]
            if SIV[t]:
                operand = IV[t, ilv]
            prod: u32 = operand * imm
            sop: u3 = SOP[t]
            pre: u32 = scalar_result(sop, sreg[irs], imm, KA[t, irs])
            for j in range(SL - 1, name="j"):
                k: i32 = SL - 1 - j
                pv[k] = pv[k - 1]
                prd[k] = prd[k - 1]
                pdata[k] = pdata[k - 1]
                pprod[k] = pprod[k - 1]
                pmac[k] = pmac[k - 1]
            pv[0] = SV[t]
            prd[0] = SRD[t]
            pdata[0] = pre
            pprod[0] = prod
            pmac[0] = sop == 3
            written = 0
            if pv[SL - 1]:
                fin: u32 = pdata[SL - 1]
                if pmac[SL - 1]:
                    fin = fin + pprod[SL - 1]
                iw: i32 = prd[SL - 1]
                sreg[iw] = fin
                written[iw] = 1
'''


def sel(name, idx, arr, k=4, indent=8):
    """``name = arr[idx]`` over scalars ``arr0..arr{k-1}`` as an if-chain."""
    sp = " " * indent
    L = [f"{sp}{name}: u32 = {arr}0"]
    for i in range(1, k):
        L.append(f"{sp}if {idx} == {i}:")
        L.append(f"{sp}    {name} = {arr}{i}")
    return L


def regs_src():
    L = [HDR, "", "@kernel", "def regs" + SIG + ":"]
    for i in range(4):
        L.append(f"    sreg{i}: u32 = 0")
    L.append("    written: u4 = 0")
    for k in range(SL):
        L += [f"    p{k}_v: u1 = 0", f"    p{k}_rd: u2 = 0", f"    p{k}_data: u32 = 0", f"    p{k}_prod: u32 = 0",
              f"    p{k}_mac: u1 = 0"]
    L += ['    for t in range(N, name="t"):', "        live: u1 = RST[t]", "        if live == 0:",
          "            written = 0"]
    L += [f"            sreg{i} = 0" for i in range(4)]
    for k in range(SL):
        L += [f"            p{k}_{f} = 0" for f in ("v", "rd", "data", "prod", "mac")]
    L += [f"        SREG[t, {i}] = sreg{i}" for i in range(4)]
    L += ["        ib: u2 = SB[t]"] + sel("rb", "ib", "sreg") + ["        OB[t] = rb"]
    L += ["        i_s: u2 = SS[t]"] + sel("rs_", "i_s", "sreg") + ["        OS[t] = rs_"]
    L += ["        il: u2 = SLB[t]"] + sel("raw", "il", "sreg")
    L += ["        if LA[t]:", "            ka_l: i32 = il", "            raw = KA[t, ka_l]",
          "        bound: u32 = raw[0:16]", "        if raw[16:32] != 0:", "            bound = 0xFFFF",
          "        OL[t] = bound", "        OW[t] = written", "        if live:",
          "            irs: u2 = SRS[t]", "            ilv: i32 = SLV[t]", "            imm: u24 = SIMM[t]"]
    L += sel("operand", "irs", "sreg", indent=12)
    L += ["            base: u32 = operand", "            if SIV[t]:", "                operand = IV[t, ilv]",
          "            prod: u32 = operand * imm", "            sop: u3 = SOP[t]",
          "            ka_r: i32 = irs", "            arg: u32 = KA[t, ka_r]",
          # scalar_result, inlined
          "            pre: u32 = 0", "            if sop == 0:", "                pre = imm",
          "            elif sop == 1:", "                pre = arg", "            elif sop == 2:",
          "                e: u32 = imm", "                if imm[23]:", "                    e[24:32] = 0xFF",
          "                pre = base + e", "            elif sop == 3:", "                pre = base",
          "            elif sop == 4:", "                shv: u5 = imm[0:5]", "                pre = base << shv"]
    for k in range(SL - 1, 0, -1):
        L += [f"            p{k}_{f} = p{k - 1}_{f}" for f in ("v", "rd", "data", "prod", "mac")]
    L += ["            p0_v = SV[t]", "            p0_rd = SRD[t]", "            p0_data = pre",
          "            p0_prod = prod", "            p0_mac = sop == 3", "            written = 0",
          f"            if p{SL - 1}_v:", f"                fin: u32 = p{SL - 1}_data",
          f"                if p{SL - 1}_mac:", f"                    fin = fin + p{SL - 1}_prod"]
    for i in range(4):
        L += [f"                if p{SL - 1}_rd == {i}:", f"                    sreg{i} = fin",
              f"                    written = {1 << i}"]
    return "\n".join(L) + "\n"


src = WRITTEN if form in ("written", "written_e1") else regs_src()
if form == "written_e1":
    src = src.replace("def written(", "def written_e1(").replace(
        "def scalar_result(op: u3, base: u32, imm: u24, arg: u32) -> u32:\n    r: u32 = 0\n",
        "def scalar_result(op32: u32, base: u32, imm: u24, arg: u32) -> u32:\n    op: u3 = op32\n    r: u32 = 0\n")
    assert "op32" in src
if form == "regs_split":
    src = src.replace("def regs(", "def regs_split(").replace("SREG: u32[N, 4]", "SREG0: u32[N], SREG1: u32[N], SREG2: u32[N], SREG3: u32[N]")
    for i in range(4):
        src = src.replace(f"SREG[t, {i}]", f"SREG{i}[t]")
kp = f"{TMP}/u4e_sagu_{form}_{N}.py"
open(kp, "w").write(src); open(f"sagu_{form}_kernel.py", "w").write(src)
spec = importlib.util.spec_from_file_location(f"u4e_sagu_{form}_{N}", kp)
K = importlib.util.module_from_spec(spec); spec.loader.exec_module(K)
from common import report_schedule, run, verdict  # noqa: E402

d = np.load(path)
rtl = getattr(K, form).schedule().export("rtl")
report_schedule(rtl, form)
c = lambda p: d[p][:, 0]  # noqa: E731
ins = [c("rst_n"), c("s_valid_i"), c("s_op_i"), c("s_rd_i"), c("s_rs_i"), c("s_use_iv_i"), c("s_level_i"),
       c("s_imm_i"), d["kernel_arg_csr_i"], d["iv_flat_i"], c("rd_sel_d_base_i"), c("rd_sel_d_stride_i"),
       c("rd_loop_bound_from_arg_i"), c("rd_sel_loop_bound_i")]
ins = [a[:rows] for a in ins]
if form == "regs_split":
    (s0, s1, s2, s3, ob, os_, ol, ow), cyc = run(rtl, ins, [((), np.uint32)] * 8, N, timeout_per_row=200)
    sreg = np.stack([s0, s1, s2, s3], axis=1)
else:
    (sreg, ob, os_, ol, ow), cyc = run(rtl, ins, [((4,), np.uint32)] + [((), np.uint32)] * 4, N, timeout_per_row=200)
n = len(ob)
verdict(f"scalar_agu:lat2 {form} rtlgen", d,
        {"sreg_o": sreg, "rd_data_d_base_o": ob.reshape(-1, 1), "rd_data_d_stride_o": os_.reshape(-1, 1),
         "rd_data_loop_bound_o": ol.reshape(-1, 1), "sreg_written_o": ow.reshape(-1, 1)},
        ["sreg_o", "rd_data_d_base_o", "rd_data_d_stride_o", "rd_data_loop_bound_o", "sreg_written_o"], n)
