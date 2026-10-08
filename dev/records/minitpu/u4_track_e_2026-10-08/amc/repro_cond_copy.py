"""Probe beside E-A6 (``repro_sagu_await.py``): a 2-deep register pipe in
the A-D3 discipline, the output read BEFORE the shift (as ``scalar_agu``'s
``pre`` outputs are). Measured: ``cond`` (shift under ``if``) and
``cond_from_input`` build and are right on ``amc``; ``uncond`` aborts in
``LoopSchedule/Utils.cpp:442 ... Assertion 'succeeded(depInserted)'`` --
U3's A-D2 already at depth 2 when the read precedes the shift (U3's repro
reads after it and needs depth 3).
    python repro_cond_copy.py <variant: cond|uncond|cond_from_input>   (AMC env, scl gcc-toolset-13)
"""
import os, sys, importlib.util
import numpy as np
v = sys.argv[1]
TMP = os.environ.get("TMPDIR", "/tmp")
body = {
    "cond": "        if C[t] == 1:\n            q1 = q0\n            q0 = A[t]\n",
    "uncond": "        q1 = q0\n        q0 = A[t]\n",
    "cond_from_input": "        if C[t] == 1:\n            q1 = A[t]\n            q0 = A[t]\n",
}[v]
src = f'''from allo.ir.types import UInt, uint32
N = 16
def k(C: uint32[N], A: uint32[N], O: uint32[N]):
    q0_r: UInt(32)[1] = 0
    q1_r: UInt(32)[1] = 0
    for t in range(N):
        q0: UInt(32) = q0_r[0]
        q1: UInt(32) = q1_r[0]
        O[t] = q1
{body}        q0_r[0] = q0
        q1_r[0] = q1
'''
kp = f"{TMP}/repro_cond_copy_{v}.py"; open(kp, "w").write(src)
spec = importlib.util.spec_from_file_location(f"rcc_{v}", kp); K = importlib.util.module_from_spec(spec); spec.loader.exec_module(K)
import allo
C = np.array([1, 0] * 8, np.uint32); A = np.arange(16, dtype=np.uint32) + 100
want = np.zeros(16, np.uint32); q0 = q1 = 0
for t in range(16):
    want[t] = q1
    if v == "uncond" or C[t]:
        q1, q0 = (q0 if v != "cond_from_input" else A[t]), A[t]
for tgt in ("llvm", "amc"):
    f = allo.customize(K.k).build(target=tgt)
    o = np.zeros(16, np.uint32); f(C, A, o)
    print(f"RESULT {v} {tgt}: {'OK' if (o == want).all() else 'WRONG ' + str(o.tolist())}", flush=True)
