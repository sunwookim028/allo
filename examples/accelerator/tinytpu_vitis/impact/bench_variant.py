"""Functional check of one variant on the Allo dataflow simulator, one build.

    python pyrun.py bench_variant.py <variant module>

gemm and gemm.relu at all five shapes, plus the vector-unit program. Prints
ALL EXACT or FAILURES, same criterion as ../bench_isa.py (which additionally
checks program-encoding equivalences that do not apply to a changed ISA)."""
import importlib, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import allo.dataflow as df  # noqa: E402

V = importlib.import_module(sys.argv[1])
from examples.accelerator.tinytpu_vitis.shapes import SHAPES  # noqa: E402
MAXDIM, T = V.MAXDIM, V.T
rng = np.random.default_rng(0)
A = rng.integers(-4, 5, (MAXDIM, MAXDIM)).astype(np.int8)
B = rng.integers(-4, 5, (MAXDIM, MAXDIM)).astype(np.int8)


def imem_of(prog):
    w = V.assemble(prog)
    m = np.zeros(V.IMEM_SIZE, np.uint64)
    m[: len(w)] = np.array(w, np.uint64)
    return m


mod = df.build(V.tinytpu_isa, target="simulator")
ok = True
for (M, K, N) in SHAPES:
    for relu in (False, True):
        C = np.zeros(MAXDIM * MAXDIM, np.int8)
        mod(imem_of(V.gemm_program(M, K, N, relu)), A.reshape(-1), B.reshape(-1), C)
        C = C.reshape(MAXDIM, MAXDIM)
        g = A[:M, :K].astype(np.int64) @ B[:K, :N].astype(np.int64)
        if relu:
            g = np.maximum(g, 0)
        g = np.clip(g, -128, 127).astype(np.int8)
        bad = int((C[:M, :N] != g).sum())
        ok &= bad == 0
        print(f"  {'gemm.relu' if relu else 'gemm':9s} {M:2d}x{K:2d}x{N:2d} wrong={bad}/{M*N}", flush=True)
M = 16
C = np.zeros(MAXDIM * MAXDIM, np.int8)
mod(imem_of(V.vadd_program(M, M, M)), A.reshape(-1), B.reshape(-1), C)
C = C.reshape(MAXDIM, MAXDIM)
g = 2 * (A[:M, :T].astype(np.int64) @ B[:T, :T].astype(np.int64))
g = np.clip(np.maximum(g, 0), -128, 127).astype(np.int8)
bad = int((C[:M, :T] != g).sum())
ok &= bad == 0
print(f"  vadd+relu wrong={bad}/{M*T}")
print("  ALL EXACT" if ok else "  FAILURES")
sys.exit(0 if ok else 1)
