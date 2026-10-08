"""U4 ``vpu_adapter`` (track A D-23 ``slots``: ``adapt_v/x/m``; optionally
``slots_gated``: ``gate_v/x/m`` on top) through RTLGen @ 13b55a63.

    python vpa_rtlgen.py <oracles/vpu_adapter.npz> <form> <N> [rows]

forms: ``call`` (each ``adapt_*`` a nested ``@kernel``), ``inline`` (bodies
pasted, region's lane-array ports ``VI[N, 2]``/``VC[N, 2]``),
``inline_split`` (every lane its own 1-D port), ``gated_split`` (the
``slots_gated`` deviation, judged on the D-23 contract: every valid bit, and
each payload field only where its op is valid -- ``vpu_adapter.SAMPLED_UNDER``).
Bodies are ``units/vpu_adapter.py``'s with types respelled and locals
prefixed by slot (``c`` -> ``v_c``), so one text serves every form.
"""
import os, sys, importlib.util
import numpy as np

path, form, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = int(sys.argv[4]) if len(sys.argv) > 4 else None
TMP = os.environ.get("TMPDIR", "/tmp")
A47, A46, A27, A20, A18 = (f"apint({k}, signed=False)" for k in (47, 46, 27, 20, 18))
ADAPT = {
"v": (f"v: {A46}, issue: u1", A47, '''
    v_c: apint(47, signed=False) = 0
    v_c[42:47] = v[41:46]
    v_c[37:42] = v[36:41]
    v_c[36] = v[35] & issue
    v_c[35] = v[34] & issue
    v_c[33:35] = v[32:34]
    v_c[32] = v[31] & issue
    v_c[30:32] = v[29:31]
    v_c[25:30] = v[24:29]
    v_c[21:24] = v[21:24]
    v_c[16:21] = v[16:21]
    v_c[15] = v[15] & issue
    v_c[13:15] = v[13:15]
    v_c[8:13] = v[8:13]
    v_c[7] = v[7] & issue
    v_c[6] = v[6]
    v_c[5] = v[5]
    v_c[0:5] = v[0:5]
'''),
"x": (f"x: {A27}, issue: u1, row: u12", A20, '''
    x_c: apint(20, signed=False) = 0
    x_c[19] = x[26] & issue
    x_c[18] = x[25]
    x_c[13:18] = x[20:25]
    x_c[1:13] = row
    x_c[0] = x[26] & x[25]
'''),
"m": ("m: u8, issue: u1", A18, '''
    m_sub: u3 = m[5:8]
    m_reg: u5 = m[0:5]
    m_c: apint(18, signed=False) = 0
    m_c[17] = issue & (m_sub == 1)
    m_c[12:17] = m_reg
    m_c[11] = issue & (m_sub == 2)
    m_c[6:11] = m_reg
    m_c[5] = issue & (m_sub == 3)
    m_c[0:5] = m_reg
'''),
}
GATE = {  # gate_<s>: input <s>_c, result <s>_g
"v": '''
    v_g: apint(47, signed=False) = 0
    v_alu: u1 = v_c[36]
    v_txin: u1 = v_c[35]
    v_txout: u1 = v_c[32]
    v_sfu: u1 = v_c[15]
    v_red: u1 = v_c[7]
    if v_alu | v_sfu | v_red | v_txin:
        v_g[42:47] = v_c[42:47]
    if v_alu:
        v_g[37:42] = v_c[37:42]
        v_g[21:25] = v_c[21:25]
        v_g[16:21] = v_c[16:21]
    v_g[36] = v_alu
    v_g[35] = v_txin
    if v_txin:
        v_g[33:35] = v_c[33:35]
    v_g[32] = v_txout
    if v_txout:
        v_g[30:32] = v_c[30:32]
        v_g[25:30] = v_c[25:30]
    v_g[15] = v_sfu
    if v_sfu:
        v_g[13:15] = v_c[13:15]
        v_g[8:13] = v_c[8:13]
    v_g[7] = v_red
    if v_red:
        v_g[0:7] = v_c[0:7]
''',
"x": '''
    x_g: apint(20, signed=False) = 0
    if x_c[19]:
        x_g = x_c
''',
"m": '''
    m_g: apint(18, signed=False) = 0
    if m_c[17]:
        m_g[12:18] = m_c[12:18]
    if m_c[11]:
        m_g[6:12] = m_c[6:12]
    if m_c[5]:
        m_g[0:6] = m_c[0:6]
''',
}
HDR = ["import numpy as np", "from allo import kernel", "from allo.lang import u1, u3, u5, u8, u12, u32, apint",
       f"N = {N}", ""]
HEAD_LANES = '''
@kernel
def {name}(VI: u32[N, 2], MI: u32[N], XI: u32[N], ISS: u32[N], ROW: u32[N], VC: u32[N, 2], XC: u32[N], MC: u32[N]):
    for t in range(N, name="t"):
        v: {A46} = 0
        v[0:32] = VI[t, 0]
        v[32:46] = VI[t, 1]
'''
HEAD_SPLIT = '''
@kernel
def {name}(VI0: u32[N], VI1: u32[N], MI: u32[N], XI: u32[N], ISS: u32[N], ROW: u32[N], VC0: u32[N], VC1: u32[N], XC: u32[N], MC: u32[N]):
    for t in range(N, name="t"):
        v: {A46} = 0
        v[0:32] = VI0[t]
        v[32:46] = VI1[t]
'''
COMMON = f'''        x: {A27} = XI[t]
        m: u8 = MI[t]
        issue: u1 = ISS[t]
        row: u12 = ROW[t]
'''


def indent(body, k):
    return "\n".join((" " * k + ln[4:]) if ln.strip() else "" for ln in body.strip("\n").splitlines())


def gen(form):
    L = list(HDR)
    split = form.endswith("split")
    gated = form.startswith("gated")
    if form == "call":
        for s, (args, rt, body) in ADAPT.items():
            L += ["@kernel", f"def adapt_{s}({args}) -> {rt}:", indent(body, 4), f"    return {s}_c", ""]
    L.append((HEAD_SPLIT if split else HEAD_LANES).format(name=form, A46=A46))
    L.append(COMMON)
    res = {}
    for s, (args, rt, body) in ADAPT.items():
        if form == "call":
            call_args = {"v": "v, issue", "x": "x, issue, row", "m": "m, issue"}[s]
            L.append(f"        c{s}: {rt} = adapt_{s}({call_args})")
        else:
            L.append(indent(body, 8))
            if gated:
                L.append(indent(GATE[s], 8))
            L.append(f"        c{s}: {rt} = {s}_{'g' if gated else 'c'}")
    if split:
        L += ["        VC0[t] = cv[0:32]", "        VC1[t] = cv[32:47]"]
    else:
        L += ["        VC[t, 0] = cv[0:32]", "        VC[t, 1] = cv[32:47]"]
    L += ["        XC[t] = cx", "        MC[t] = cm"]
    return "\n".join(L) + "\n"


src = gen(form)
kp = f"{TMP}/u4e_vpa_{form}_{N}.py"
open(kp, "w").write(src); open(f"vpa_{form}_kernel.py", "w").write(src)
spec = importlib.util.spec_from_file_location(f"u4e_vpa_{form}_{N}", kp)
K = importlib.util.module_from_spec(spec); spec.loader.exec_module(K)
from common import report_schedule, run, verdict  # noqa: E402

d = np.load(path)
rtl = getattr(K, form).schedule().export("rtl")
report_schedule(rtl, form)
c1 = lambda p: d[p][:, 0]  # noqa: E731
if form.endswith("split"):
    ins = [d["v_i"][:, 0], d["v_i"][:, 1], c1("m_i"), c1("x_i"), c1("issue_i"), c1("x_resolved_row_i")]
    (v0, v1, xc, mc), cyc = run(rtl, ins, [((), np.uint32)] * 4, N, max_rows=rows)
else:
    ins = [d["v_i"], c1("m_i"), c1("x_i"), c1("issue_i"), c1("x_resolved_row_i")]
    (vc, xc, mc), cyc = run(rtl, ins, [((2,), np.uint32), ((), np.uint32), ((), np.uint32)], N, max_rows=rows)
    v0, v1 = vc[:, 0], vc[:, 1]
n = len(xc)
full = [((int(v0[t]) | (int(v1[t]) << 32)) << 38) | (int(xc[t]) << 18) | int(mc[t]) for t in range(n)]
got = np.array([[(z >> (32 * k)) & 0xFFFFFFFF for k in range(3)] for z in full], dtype=np.uint32)
if not form.startswith("gated"):
    verdict(f"vpu_adapter {form} rtlgen", d, {"vpu_ctrl_o": got}, ["vpu_ctrl_o"], n)
else:
    # D-23 contract (vpu_adapter.gated_contract): layout VPU_CTRL MSB first
    LAY = [("raddr_a", 5), ("raddr_b", 5), ("alu_valid", 1), ("txin_valid", 1), ("txin_index", 2),
           ("txout_valid", 1), ("txout_index", 2), ("txout_vd", 5), ("alu_op", 4), ("alu_vd", 5),
           ("sfu_valid", 1), ("sfu_op", 2), ("sfu_vd", 5), ("reduce_valid", 1), ("reduce_op", 1),
           ("reduce_lane", 1), ("reduce_vd", 5), ("vmem.valid", 1), ("vmem.op", 1), ("vmem.vreg_idx", 5),
           ("vmem.vmem_address", 12), ("vmem_store_read_hint", 1), ("vmatload_valid", 1),
           ("vmatload_base", 5), ("vmatpush_valid", 1), ("vmatpush_vs", 5), ("vmatpop_valid", 1),
           ("vmatpop_vd", 5)]
    assert sum(w for _, w in LAY) == 85
    SAMPLED_UNDER = {
        "raddr_a": ("alu_valid", "sfu_valid", "reduce_valid", "txin_valid"), "raddr_b": ("alu_valid",),
        "txin_index": ("txin_valid",), "txout_index": ("txout_valid",), "txout_vd": ("txout_valid",),
        "alu_op": ("alu_valid",), "alu_vd": ("alu_valid",), "sfu_op": ("sfu_valid",), "sfu_vd": ("sfu_valid",),
        "reduce_op": ("reduce_valid",), "reduce_lane": ("reduce_valid",), "reduce_vd": ("reduce_valid",),
        "vmem.op": ("vmem.valid",), "vmem.vreg_idx": ("vmem.valid",), "vmem.vmem_address": ("vmem.valid",),
        "vmem_store_read_hint": ("vmem.valid",), "vmatload_base": ("vmatload_valid",),
        "vmatpush_vs": ("vmatpush_valid",), "vmatpop_vd": ("vmatpop_valid",)}

    def unpack(z):
        out, pos = {}, 85
        for nm, w in LAY:
            pos -= w
            out[nm] = (z >> pos) & ((1 << w) - 1)
        return out
    want = [sum(int(d["vpu_ctrl_o"][t, k]) << (32 * k) for k in range(3)) for t in range(n)]
    tot = bad = zeroed = 0
    for t in range(n):
        r, g = unpack(want[t]), unpack(full[t])
        for f, _ in LAY:
            if f in SAMPLED_UNDER and not any(r[v] for v in SAMPLED_UNDER[f]):
                zeroed += int(g[f] != r[f])
                continue
            tot += 1
            bad += int(g[f] != r[f])
    print(f"{'CONTRACT-MATCH' if bad == 0 else 'CONTRACT-DIFF '} vpu_adapter {form} rtlgen: {tot - bad}/{tot} "
          f"sampled fields equal ({zeroed} unsampled payload fields zeroed)", flush=True)
