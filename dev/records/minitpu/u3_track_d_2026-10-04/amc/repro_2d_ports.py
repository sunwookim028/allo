"""AMC reduction of the tree aborts: which construct? (a) a 2-D input read per
lane; (b) a 2-D output store per lane; (c) 1-D only with a 1-element carried
register (the PE's shape, known good). Each built alone on target amc."""
import os, sys, importlib.util, traceback, numpy as np
TMP = os.environ.get("TMPDIR", "/tmp")
srcs = {
"in2d": '''
from allo.ir.types import int32, uint16
N = 16
def k(d: uint16[N, 4], o: uint16[N]):
    for t in range(N):
        a: int32 = d[t, 0]
        b: int32 = d[t, 1]
        c: int32 = d[t, 2]
        e: int32 = d[t, 3]
        o[t] = (a + b + c + e) & 0xFFFF
''',
"out2d": '''
from allo.ir.types import int32, uint16
N = 16
def k(a: uint16[N], o: uint16[N, 4]):
    for t in range(N):
        x: int32 = a[t]
        o[t, 0] = x
        o[t, 1] = (x + 1) & 0xFFFF
        o[t, 2] = (x + 2) & 0xFFFF
        o[t, 3] = (x + 3) & 0xFFFF
''',
"in2d_reg": '''
from allo.ir.types import int32, uint16
N = 16
def k(d: uint16[N, 4], o: uint16[N]):
    q_r: int32[1] = 0
    for t in range(N):
        q: int32 = q_r[0]
        a: int32 = d[t, 0]
        b: int32 = d[t, 1]
        o[t] = q
        q = (a + b) & 0xFFFF
        q_r[0] = q
''',
"in1d_reg": '''
from allo.ir.types import int32, uint16
N = 16
def k(a: uint16[N], b: uint16[N], o: uint16[N]):
    q_r: int32[1] = 0
    for t in range(N):
        q: int32 = q_r[0]
        x: int32 = a[t]
        y: int32 = b[t]
        o[t] = q
        q = (x + y) & 0xFFFF
        q_r[0] = q
''',
}
import allo
which = sys.argv[1:] or list(srcs)
for name in which:
    kp = f"{TMP}/u3d_r2d_{name}.py"; open(kp, "w").write(srcs[name])
    spec = importlib.util.spec_from_file_location(f"u3d_r2d_{name}", kp); K = importlib.util.module_from_spec(spec); spec.loader.exec_module(K)
    try:
        f = allo.customize(K.k).build(target="amc")
        print(f"{name}: BUILT", flush=True)
    except Exception as e:
        print(f"{name}: FAIL {type(e).__name__}: {str(e)[:200]}", flush=True)
