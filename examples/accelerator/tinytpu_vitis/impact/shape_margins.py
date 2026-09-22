# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""What refuses a shape, found by raising each bound until it fires.

Every limit on the size of a runnable GEMM is in the ENCODING or in a header
slice, never in the datapath, so each one can be found by asking the shipped
generator for a shape and reading which assertion answers. That is what this
does -- one subprocess per configuration, because the constants are read at
import -- and it is how the table in
`docs/source/designs/tinytpu_isa.rst` ("Shapes that do not fit") was measured.

    python impact/shape_margins.py                 # the margins at T = 4, 8, 16
    python impact/shape_margins.py 128x768x768 ... # and these shapes

A shape is tried three ways: as the SQUARE program (`gemm_program`, which must
fit MAXDIM), as the square program on a machine whose `nr` is the 8 bits it
used to be, and as the TILED program (`gemm_tiled`, which does not have to
fit). The point of the third column is that it is the only one that says OK
for a shape a real workload uses.
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))

PROBE = r"""
import sys, traceback
sys.path.insert(0, %r)
from examples.accelerator.tinytpu_vitis import microarch_isa as U
from examples.accelerator.tinytpu_vitis import isa_dsl
how, M, K, N = sys.argv[1], *[int(x) for x in sys.argv[2:5]]
try:
    if how == "tiled":
        prog, dram = isa_dsl.gemm_tiled(M, K, N, relu=True)
    else:
        prog, dram = isa_dsl.gemm_program(M, K, N, True), None
    words = U.assemble(prog, dram)
    print(f"OK static={len(prog)} words={len(words)} "
          f"dynamic={len(U.expand(prog))}")
except Exception as e:
    tb = traceback.extract_tb(sys.exc_info()[2])[-1]
    where = f"{os.path.basename(tb.filename)}:{tb.lineno}"
    print(f"{type(e).__name__}: {e or tb.line} [{where}]")
""" % ROOT


def probe(how, shape, **env):
    e = dict(os.environ, TPU_IMEM="4096", **{k: str(v) for k, v in env.items()})
    r = subprocess.run([sys.executable, "-c", "import os\n" + PROBE, how]
                       + [str(x) for x in shape],
                       capture_output=True, text=True, env=e, cwd=ROOT)
    out = (r.stdout.strip() or r.stderr.strip().splitlines()[-1])
    return out.splitlines()[-1]


def largest_maxdim(T):
    """The largest MAXDIM that imports, by walking multiples of T."""
    d = T
    while probe("square", (T, T, T), TPU_T=T, TPU_MAXDIM=d + T).startswith("OK"):
        d += T
    return d, probe("square", (T, T, T), TPU_T=T, TPU_MAXDIM=d + T)


SHAPES = [(32, 128, 128), (64, 128, 128), (128, 128, 128), (32, 512, 128)]

if __name__ == "__main__":
    args = sys.argv[1:]
    shapes = ([tuple(int(x) for x in a.split("x")) for a in args] if args
              else SHAPES)
    print("largest MAXDIM that builds, per array width:")
    for T in (4, 8, 16):
        d, why = largest_maxdim(T)
        print(f"  T={T:2d}  MAXDIM={d:4d}   the next one: {why[:96]}")

    print("\nshapes, three ways (T=8, MAXDIM=128 -- the largest legal there):")
    for shape in shapes:
        for how, env in (("square, nr 8 bits", {"TPU_NR_BITS": 8}),
                         ("square", {}),
                         ("tiled", {})):
            cfg = dict(TPU_T=8, TPU_MAXDIM=128, TPU_DRAM=1 << 20, **env)
            got = probe("tiled" if how == "tiled" else "square", shape, **cfg)
            print(f"  {shape[0]}x{shape[1]}x{shape[2]:<4d} {how:18s} "
                  f"{got[:104]}")
