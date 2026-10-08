"""AMC finding E-A1 (bug, loud): a kernel named ``top`` builds on ``llvm``
and fails on ``amc`` at FSM emission:
    loc("-":2:3): error: redefinition of symbol named 'top'
    Failed to run FSM-to-Verilog pipeline  -> RuntimeError: FSM verilog emission failed
(U1 R4 met the same message on a textual hand-off). Renamed, it builds.
    python repro_top_name.py    (AMC env, scl gcc-toolset-13)"""
import numpy as np, allo
from allo.ir.types import uint32
N = 4
def top(A: uint32[N], B: uint32[N]):
    for i in range(N):
        B[i] = A[i] + 1
def topk(A: uint32[N], B: uint32[N]):
    for i in range(N):
        B[i] = A[i] + 1
for fn in (top, topk):
    for tgt in ("llvm", "amc"):
        try:
            allo.customize(fn).build(target=tgt); print(f"RESULT {fn.__name__} {tgt}: built", flush=True)
        except Exception as e:
            print(f"RESULT {fn.__name__} {tgt}: {type(e).__name__}: {str(e)[:80]}", flush=True)
