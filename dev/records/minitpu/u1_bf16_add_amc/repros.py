# Minimal repros for the AMC findings (A1-A8) through AMC's own frontend.
# Run in the AMC env (source env.sh) under `scl enable gcc-toolset-13`:
#   python repros.py [name ...]      (each in a subprocess: some abort)
# Each prints FAIL (with the first error line), MISMATCH (got vs want) or OK.
import sys, subprocess, numpy as np

N = 8
A = np.array([5, 14076, 58714, 54111, 1, 200, 0, 256], np.uint16)

SRC = {}
WANT = {}

# A1: `not` has no typing rule (KeyError: <class 'ast.Not'>).
SRC["A1_not"] = """
def k(a: uint16[N], c: uint16[N]):
    for i in range(N):
        f: uint1 = a[i] == 0
        c[i] = not f
"""
WANT["A1_not"] = lambda a: (a == 0).astype(int)

# A2: single-bit index x[k] is dead code on Python >= 3.9 (it tests
# ast.Index): "Unsupported bit operation". Slices x[k:k+1] work.
SRC["A2_bit_index"] = """
def k(a: uint16[N], c: uint16[N]):
    for i in range(N):
        x: uint16 = a[i]
        c[i] = x[15]
"""
WANT["A2_bit_index"] = lambda a: a >> 15

# A3: a call result cannot be assigned to an already-declared name.
SRC["A3_call_assign"] = """
def k(a: uint16[N], c: uint16[N]):
    def f(v: uint16) -> uint16:
        return v + 1
    for i in range(N):
        r: uint16 = 0
        r = f(a[i])
        c[i] = r
"""
WANT["A3_call_assign"] = lambda a: a + 1

# A4: a scalar-returning call: FixedPointToInteger::updateCallOp
# "result type is not memref", then an IRMapping assertion abort.
SRC["A4_scalar_call"] = """
def k(a: uint16[N], c: uint16[N]):
    def f(v: uint16) -> uint16:
        return v + 1
    for i in range(N):
        r: uint16 = f(a[i])
        c[i] = r
"""
WANT["A4_scalar_call"] = lambda a: a + 1

# A5: `x or y or z` builds only `x or y` (build_BoolOp uses stmts[0:2]).
# Silent: wrong on the LLVM target too.
SRC["A5_boolop3"] = """
def k(a: uint16[N], c: uint16[N]):
    for i in range(N):
        x: uint16 = a[i]
        c[i] = x == 1 or x == 2 or x == 200
"""
WANT["A5_boolop3"] = lambda a: ((a == 1) | (a == 2) | (a == 200)).astype(int)

# A6 (= our B1): UInt < UInt picks a signed predicate from the signless type.
SRC["A6_ucmp_B1"] = """
def k(a: uint16[N], c: uint16[N]):
    for i in range(N):
        x: uint16 = a[i]
        y: uint16 = 100
        c[i] = x > y
"""
WANT["A6_ucmp_B1"] = lambda a: (a > 100).astype(int)

# A7: two loop-carried scalars, only the first read after the loop: the
# loop's exit value is wired from the *second* (miscompile). With `found`
# as uint1 instead it fails: 'comb.mux' op ... same type.
SRC["A7_two_carried"] = """
def k(a: uint16[N], c: uint16[N]):
    for i in range(N):
        x: uint16 = a[i]
        lz: uint16 = 99
        found: uint16 = 0
        for j in range(16):
            if found == 0 and ((x >> j) & 1) == 1:
                lz = j
                found = 1
        c[i] = lz
"""
WANT["A7_two_carried"] = lambda a: np.array(
    [min([j for j in range(16) if (int(x) >> j) & 1] or [99]) for x in a])
SRC["A7_two_carried_u1"] = SRC["A7_two_carried"].replace("found: uint16", "found: uint1")
WANT["A7_two_carried_u1"] = WANT["A7_two_carried"]

# A8: slice assignment lowers (lower_bit_ops) to a bit-serial scf.for; a
# value built that way and fed to another such loop and a select aborts
# LoopScheduleToFSM (IRMapping lookup assertion). Reduced by ddmin from the
# bf16 kernel.
SRC["A8_slice_set"] = """
def k(av: uint16[N], c: uint16[N]):
    for i in range(N):
        a_i: uint16 = av[i]
        b_i: uint16 = av[N - 1 - i]
        mant_a: UInt(17) = 0
        mant_a[9:16] = a_i[0:7]
        mant_b: UInt(17) = 0
        mant_b[9:16] = b_i[0:7]
        key_a: UInt(26) = 0
        key_a[0:17] = mant_a
        key_b: UInt(26) = 0
        key_b[0:17] = mant_b
        mant_large: UInt(17) = 0
        if key_a >= key_b:
            mant_large = mant_a
        c[i] = mant_large[0:16]
"""
def _a8(a):
    ma = (a.astype(np.int64) & 127) << 9
    mb = ma[::-1]
    return np.where(ma >= mb, ma, 0) & 0xFFFF
WANT["A8_slice_set"] = _a8


def one(name, target="amc"):
    import allo
    from allo.ir.types import UInt, uint1, uint16
    g = {"UInt": UInt, "uint1": uint1, "uint16": uint16, "N": N}
    import tempfile
    path = f"{tempfile.gettempdir()}/_amc_repro_{name}.py"
    open(path, "w").write("from allo.ir.types import UInt, uint1, uint16\nN = %d\n" % N + SRC[name])
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    c = np.zeros(N, np.uint16)
    try:
        f = allo.customize(m.k).build(target=target)
        f(A.copy(), c)
    except Exception as e:
        print(f"{name} [{target}]: FAIL {type(e).__name__}: {str(e).splitlines()[0][:200]}")
        return
    want = WANT[name](A)
    ok = np.array_equal(c.astype(np.int64), want.astype(np.int64))
    print(f"{name} [{target}]: {'OK' if ok else 'MISMATCH'} got {c.tolist()} want {want.tolist()}")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--one":
        one(sys.argv[2], sys.argv[3])
        sys.exit(0)
    for name in sys.argv[1:] or list(SRC):
        for target in ("llvm", "amc"):
            r = subprocess.run([sys.executable, "-u", __file__, "--one", name, target],
                               capture_output=True, text=True)
            lines = [l for l in (r.stdout + r.stderr).splitlines()
                     if l.startswith(name) or "error:" in l or "Assertion" in l]
            print("\n".join(lines[-2:]) if lines else f"{name} [{target}]: rc={r.returncode}", flush=True)
