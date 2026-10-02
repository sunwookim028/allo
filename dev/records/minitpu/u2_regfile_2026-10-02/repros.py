"""Minimal repros for the U2 vpu_regfile findings (``../u2_regfile_2026-10-02.rst``).

    python dev/records/minitpu/u2_regfile_2026-10-02/repros.py [name ...]

From the worktree root after ``source examples/minitpu/harness/env-zhang21.sh``.
Each repro prints one ``REPRO <name>: ...`` line with what it observed. They
run in subprocesses: B6 corrupts the heap and hangs, so it gets a timeout.
"""
import os, subprocess, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
PRJ = os.environ.get("REPRO_PRJ", "/tmp/u2_regfile_repros")

R = {}

# B4: an unsigned array index is sign-extended (arith.index_cast, not
# index_castui) on the simulator / LLVM path. uint8 200 and UInt(5) 20 index
# out of bounds, silently. The HLS C++ emitters are unaffected (ap_uint index).
R["B4"] = r'''
import allo, numpy as np
from allo.ir.types import UInt, uint8, uint16
def k8(ra: uint8[4], q: uint16[4]):
    mem: uint16[256] = 0
    for i in range(256):
        mem[i] = i
    for t in range(4):
        q[t] = mem[ra[t]]
def k5(ra: UInt(5)[4], q: uint16[4]):
    mem: uint16[32] = 0
    for i in range(32):
        mem[i] = i
    for t in range(4):
        q[t] = mem[ra[t]]
s = allo.customize(k5)
cast = [l.strip() for l in str(s.module).splitlines() if "index_cast" in l and "i5" in l]
q8 = np.zeros(4, np.uint16); allo.customize(k8).build()(np.array([1, 200, 255, 128], np.uint8), q8)
q5 = np.zeros(4, np.uint16); s.build()(np.array([1, 20, 31, 16], np.uint8), q5)
print(f"REPRO B4: uint8 idx [1,200,255,128] -> {q8.tolist()} (want same); "
      f"UInt(5) idx [1,20,31,16] -> {q5.tolist()}; IR: {cast[:1]}")
'''

# B5: `x: int32 = s.get()` on a Stream[UInt(5)] stores the i5 into an i32
# memref with no cast: the IR verifier rejects it (loud, not silent).
R["B5"] = r'''
import numpy as np, allo.dataflow as df
from allo.ir.types import UInt, int32, Stream
n = 4
@df.region()
def top(A: UInt(5)[n], B: int32[n]):
    s: Stream[UInt(5), 2]
    @df.kernel(mapping=[1], args=[A])
    def p(a: UInt(5)[n]):
        for t in range(n):
            s.put(a[t])
    @df.kernel(mapping=[1], args=[B])
    def c(b: int32[n]):
        for t in range(n):
            x: int32 = s.get()
            b[t] = x
try:
    df.build(top, target="simulator")
    print("REPRO B5: built (fixed?)")
except Exception as e:
    print(f"REPRO B5: {type(e).__name__}: " + [l for l in str(e).splitlines() if "affine.store" in l][0][:160])
'''

# S6: an argument array read under a condition becomes a conditional Pop()
# on the Connections stream the SystemC emitter turns the array into, so the
# stream falls out of step with the loop. Silent: wrong values, no diagnostic.
R["S6"] = r'''
import numpy as np, allo.dataflow as df
from allo.ir.types import int32
n = 6
@df.region()
def top(C: int32[n], D: int32[n], Q: int32[n]):
    @df.kernel(mapping=[1], args=[C, D, Q])
    def k(c: int32[n], d: int32[n], q: int32[n]):
        acc: int32 = 0
        for t in range(n):
            if c[t] != 0:
                acc = d[t]
            q[t] = acc
C = np.array([1, 0, 0, 1, 0, 1], np.int32); D = np.arange(10, 10 + n, dtype=np.int32)
want = np.zeros(n, np.int32); df.build(top, target="simulator")(C, D, want)
got = np.zeros(n, np.int32)
m = df.build(top, target="systemc", mode="csim", project="''' + PRJ + r'''/s6")
m(C, D, got)
k = open("''' + PRJ + r'''/s6/kernel.cpp").read()
cond = [l.strip() for l in k.splitlines() if ".Pop()" in l]
print(f"REPRO S6: simulator {want.tolist()} systemc {got.tolist()}; pops: {cond}")
'''

# B6: a numpy array narrower than the kernel's UInt(256) element is accepted
# by the simulator with a warning only, then read past its end.
R["B6"] = r'''
import numpy as np, allo.dataflow as df
from allo.ir.types import UInt
n = 4
@df.region()
def top(A: UInt(256)[n], B: UInt(256)[n]):
    @df.kernel(mapping=[1], args=[A, B])
    def k(a: UInt(256)[n], b: UInt(256)[n]):
        for t in range(n):
            b[t] = a[t]
m = df.build(top, target="simulator")
a = np.arange(n, dtype=np.uint64); b = np.zeros(n, np.uint64)
m(a, b)
print(f"REPRO B6: returned {b.tolist()} (no refusal)")
'''

# H3: Memory(latency=, depth=) never reaches the IR; SystemC drops the whole
# annotation (byte-identical emission); Vitis keeps resource/storage only.
R["H3"] = r'''
import allo, allo.dataflow as df
from allo.ir.types import uint16
from allo.memory import Memory
def k(a: uint16[4], q: uint16[4]):
    mem: uint16[32] @ Memory(resource="LUTRAM", storage_type="RAM_1WNR", latency=0, depth=32)
    for t in range(4):
        mem[t] = a[t]
        q[t] = mem[t]
s = allo.customize(k)
ir = str(s.module)
v = [l.strip() for l in str(s.build(target="vhls")).splitlines() if "bind_storage" in l]
print(f"REPRO H3: IR memref space {'52' if ', 52 : i32>' in ir else '?'}; 'latency' in IR: {'latency' in ir}; vhls: {v}")
'''

# H2: SystemC refuses a region-scope Stateful touched by >1 kernel (D-1 honoured).
R["H2"] = r'''
import allo.dataflow as df
from allo.ir.types import int32, Stateful
n = 4
@df.region()
def top(A: int32[n], B: int32[n]):
    mem: int32[8] @ Stateful = 0
    @df.kernel(mapping=[1], args=[A])
    def w(a: int32[n]):
        for t in range(n):
            mem[t] = a[t]
    @df.kernel(mapping=[1], args=[B])
    def r(b: int32[n]):
        for t in range(n):
            b[t] = mem[t]
try:
    df.build(top, target="systemc", mode="csyn", project="''' + PRJ + r'''/h2")
    print("REPRO H2: emitted (silent replica?)")
except Exception as e:
    print(f"REPRO H2: refused: {type(e).__name__}")
'''

if __name__ == "__main__":
    os.makedirs(PRJ, exist_ok=True)
    for name in sys.argv[1:] or list(R):
        try:
            src = os.path.join(PRJ, f"repro_{name}.py")  # Allo reads kernel source: a file, not -c
            with open(src, "w") as f:
                f.write(f"import sys; sys.path.insert(0, {ROOT!r})\n" + R[name])
            r = subprocess.run([sys.executable, src], capture_output=True, text=True,
                               timeout=120 if name == "B6" else 900)
            out = [l for l in (r.stdout + r.stderr).splitlines() if l.startswith("REPRO") or "error:" in l]
            print("\n".join(out) or f"REPRO {name}: exit {r.returncode}, no line; tail: {(r.stdout + r.stderr)[-300:]}",
                  flush=True)
        except subprocess.TimeoutExpired as e:
            tail = ((e.stderr or b"").decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or ""))
            print(f"REPRO {name}: TIMEOUT (hung); stderr tail: {tail[-200:]!r}", flush=True)
