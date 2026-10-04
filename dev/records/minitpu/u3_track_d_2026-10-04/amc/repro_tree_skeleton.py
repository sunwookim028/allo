"""AMC reduction of the tree abort (Utils.cpp:442 `succeeded(depInserted)` in
getChainingSharedOperatorsProblem on the unscheduled scf.while). The tree
skeleton at 4 leaves with a one-line node, minus one element per variant:
  full      lane-array in (uint16[N,4]), 2-D out (uint16[N,2]), pipe registers
            as 1-element arrays, valid pipes cleared under `if r == 0`
  noreset   no reset block
  no2dout   the 2-D output replaced by two 1-D outputs
  nopipe    no carried registers (outputs straight from the tree)
  onereg    one carried register only, no reset, 1-D ports
"""
import os, sys, importlib.util, numpy as np
TMP = os.environ.get("TMPDIR", "/tmp")
def gen(v):
    L = ["from allo.ir.types import int32, uint8, uint16", "N = 32"]
    two_d_out = v not in ("no2dout", "onereg"); pipe = v != "nopipe"; reset = v not in ("noreset", "onereg")
    lane_in = v != "onereg"
    sig = "def k(rst: uint8[N], v: uint8[N], d: uint16[N, 4], ro: uint16[N], vo: uint8[N], " + ("lro: uint16[N, 2]):" if two_d_out else "l0: uint16[N], l1: uint16[N]):")
    if not lane_in:
        sig = "def k(rst: uint8[N], v: uint8[N], d0: uint16[N], d1: uint16[N], ro: uint16[N], vo: uint8[N], l0: uint16[N], l1: uint16[N]):"
    L.append(sig)
    regs = (["vq0", "vq1", "rq0", "rq1", "lq0", "lq1"] if v != "onereg" else ["rq0"]) if pipe else []
    L += [f"    {r}_r: int32[1] = 0" for r in regs]
    L.append("    for t in range(N):")
    B = [f"{r}: int32 = {r}_r[0]" for r in regs]
    B += ["r: int32 = rst[t]", "vv: int32 = v[t]"]
    if lane_in:
        B += [f"n{i}: int32 = d[t, {i}]" for i in range(4)]
    else:
        B += ["n0: int32 = d0[t]", "n1: int32 = d1[t]", "n2: int32 = n0", "n3: int32 = n1"]
    B += ["n4: int32 = (n0 + n1) & 0xFFFF", "n5: int32 = (n2 + n3) & 0xFFFF", "n6: int32 = (n4 + n5) & 0xFFFF"]
    if pipe and v != "onereg":
        B += ["vq1 = vq0", "rq1 = rq0", "vq0 = vv", "rq0 = n6", "lq1 = lq0", "lq0 = n4"]
        if reset:
            B += ["if r == 0:", "    vq0 = 0", "    vq1 = 0"]
        B += (["lro[t, 0] = lq1", "lro[t, 1] = n5"] if two_d_out else ["l0[t] = lq1", "l1[t] = n5"]) + ["vo[t] = vq1", "ro[t] = rq1"]
    elif v == "onereg":
        B += ["rq0 = n6", "l0[t] = n4", "l1[t] = n5", "vo[t] = vv", "ro[t] = rq0"]
    else:
        B += (["lro[t, 0] = n4", "lro[t, 1] = n5"] if two_d_out else ["l0[t] = n4", "l1[t] = n5"]) + ["vo[t] = vv", "ro[t] = n6"]
    B += [f"{r}_r[0] = {r}" for r in regs]
    L += ["        " + x for x in B]
    return "\n".join(L) + "\n"
import allo
for v in sys.argv[1:]:
    kp = f"{TMP}/u3d_skel_{v}.py"; open(kp, "w").write(gen(v))
    spec = importlib.util.spec_from_file_location(f"u3d_skel_{v}", kp); K = importlib.util.module_from_spec(spec); spec.loader.exec_module(K)
    try:
        allo.customize(K.k).build(target="amc"); print(f"{v}: BUILT", flush=True)
    except Exception as e:
        print(f"{v}: FAIL {type(e).__name__}: {str(e)[:160]}", flush=True)
