# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Emit a Catapult `mode="ppa"` handoff for TinyTPU-isa: power, not just area.

WHY THIS FILE EXISTS. Area and latency come out of `csynth_sweep.py` and
`cosim.py`; power does not, and a synthesis power number that is not
activity-annotated is a default-toggle guess. Catapult's PowerPro produces a
real, simulated-activity number through Allo's own emitted tcl -- but only for
a design "SCVerify can drive", which means a C++ testbench with `CCS_MAIN` and
`CCS_DESIGN(<top>)`. TinyTPU had none. This writes one, together with the
project Catapult consumes, so the whole thing can be handed to the licence
host with nothing to run here.

WHAT THE TESTBENCH IS, AND WHAT IT IS NOT. `go switching` simulates THIS
testbench against the pre-power RTL and the resulting SAIF is the entire basis
of the power figure, so the stimulus IS the measurement's workload. The
choices, each deliberate:

  * **16x16x16 only**, the largest published shape. TinyTPU's cycle count is a
    fixed startup/drain term plus work; at the smaller shapes the fixed term
    dominates, so their activity is mostly program load and pipeline fill.
    16x16x16 is the shape whose steady state is longest relative to that term,
    i.e. the one that most nearly measures the array doing arithmetic.
  * **Full-range int8 operands**, NOT the [-4, 4] that `bench_isa.py` and
    `cosim.py` use. That distribution exists to match Gemmini's `allo_cmp.c`
    for a like-for-like CYCLE comparison, and cycles do not depend on operand
    values. Power does: [-4, 4] holds the top five bits of every operand
    constant and would bias the number low. `--dist small` regenerates the
    Gemmini-distribution variant if the comparison is ever wanted.
  * **Four calls on one instance** -- three GEMMs (one with ReLU) at different
    seeds, then `isa_dsl.vector_program(8)`. Three seeds so one unlucky draw
    does not become the number; `vector_program` because a GEMM never exercises
    `vld`, a second accumulator region, `vadd` across three regions, or a
    `mvout` to a nonzero DRAM row, and a unit that never switches contributes
    only leakage.

  It therefore measures: one shape, one operand distribution, at T=4 /
  MAXDIM=16, nangate45 behavioural, pre-layout. It does NOT represent the
  shipped MAXDIM=64 build, a mix of shapes, a duty cycle with idle time
  between kernels, or anything post-place-and-route.

SELF-CHECKING. Every call prefills `C` with random bytes and compares the whole
of `C` afterwards, splitting the count: `errors=` counts cells the program was
expected to change (computed here, per case, as `gold != C0`), `clobbered=`
counts the rest. `errors=0` is the criterion; `clobbered` is reported but not
gating, because whole-array preservation is a property of SCVerify's memory
wrapper rather than of the arithmetic, and no one has exercised that wrapper
on a design with four array ports.

Before writing anything this script runs every case on Allo's own simulator and
requires it to reproduce `gold`, so the expectations the C++ carries are the
ones the design actually meets -- evidence the `mac16` handoff could not give,
having no reference implementation to run.

USAGE

    TPU_MAXDIM=16 python ppa_catapult.py -o <dir>     # the committed handoff
    TPU_MAXDIM=16 python ppa_catapult.py --tb-only    # regenerate just the tb
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..")))

import allo.dataflow as df                                       # noqa: E402
from allo.dataflow import customize                              # noqa: E402
from examples.tinytpu import isa_ref                             # noqa: E402
from examples.tinytpu.isa_dsl import (                           # noqa: E402
    gemm_program, vector_program)
from examples.tinytpu.microarch_isa import (                     # noqa: E402
    tinytpu_isa, assemble, schedule, MAXDIM, T, IMEM_SIZE)
from examples.tinytpu.stress_isa import (                        # noqa: E402
    gemm_gold, operands)

# The shape the testbench runs. The largest published one; see the module
# docstring for why not the whole sweep.
SHAPE = (16, 16, 16)
# The clock and library the mac16 power run used, so the two figures differ
# only in the design.
CLOCK_PERIOD = 5.0
LIBRARY = "nangate-45nm_beh"
# zhang-21's Xcelium. Emitted into run.tcl from here, not compiled into the
# backend; change it with --ncsim-root for another host.
NCSIM_ROOT = "/opt/cadence/XCELIUM2403"

TOP = "tinytpu_isa"


def written_cells(prog, A, B, C0, gold, rng):
    """Which cells of `C` this program writes -- the mask the testbench splits
    `wrong` from `clobbered` by.

    NOT simply `gold != C0`: a written cell whose new value happens to equal
    the old one would be classed as untouched (1 cell in 256 at random fill),
    and a real error there would then be reported as a clobber. Running the
    reference a second time over an independent `C0` and taking the union
    makes that coincidence need to happen twice, i.e. 1 in 65536.
    """
    alt = rng.integers(-128, 128, MAXDIM * MAXDIM).astype(np.int8)
    gold_alt = isa_ref.run(prog, A, B, alt)
    return ((gold != C0) | (gold_alt != alt)).astype(np.int8)


def cases(dist):
    """(name, prog, A, B, C0, gold, mask) for each call the testbench makes."""
    M, K, N = SHAPE
    crng = np.random.default_rng(20260924)
    out = []
    for i, relu in enumerate((False, True, False)):
        A, B = operands(dist, 900 + i)
        C0 = crng.integers(-128, 128, MAXDIM * MAXDIM).astype(np.int8)
        prog = gemm_program(M, K, N, relu)
        gold = gemm_gold(A, B, M, K, N, relu, C0)
        # Two independent references must agree before either is shipped.
        assert (isa_ref.run(prog, A, B, C0) == gold).all()
        mask = written_cells(prog, A, B, C0, gold, crng)
        # A GEMM writes exactly its M x N region; anything else means the
        # reference, not the design, is wrong.
        want = np.zeros((MAXDIM, MAXDIM), np.int8)
        want[:M, :N] = 1
        assert (mask.reshape(MAXDIM, MAXDIM) == want).all()
        out.append((f"gemm{'.relu' if relu else ''} {M}x{K}x{N} #{i}",
                    prog, A, B, C0, gold, mask))
    A, B = operands(dist, 950)
    C0 = crng.integers(-128, 128, MAXDIM * MAXDIM).astype(np.int8)
    prog = vector_program(8)
    gold = isa_ref.run(prog, A, B, C0)
    out.append(("vector_program(8)", prog, A, B, C0, gold,
                written_cells(prog, A, B, C0, gold, crng)))
    return out


def imem_of(prog):
    words = assemble(prog)
    m = np.zeros(IMEM_SIZE, np.uint64)
    m[: len(words)] = np.array(words, np.uint64)
    return m


def check_on_simulator(case_list):
    """Run every case on Allo's simulator; the gold must come back exactly.

    This is what makes the emitted expectations the DESIGN's, not just numpy's.
    """
    mod = df.build(tinytpu_isa, target="simulator")
    for name, prog, A, B, C0, gold, _mask in case_list:
        C = C0.copy()
        mod(imem_of(prog), A.reshape(-1), B.reshape(-1), C)
        bad = int((C != gold).sum())
        if bad:
            raise SystemExit(f"simulator disagrees with gold on {name}: "
                             f"{bad} cells -- not writing a testbench that "
                             f"asks the RTL for an answer the design does not "
                             f"give")
        print(f"   simulator OK: {name}")


def carr(name, vals, ctype):
    # NO `alignas(64)` here, unlike `cosim.py`'s `carr`. That one keeps the
    # `align_value(64)` promise the VITIS build makes on its `m_axi` pointers;
    # the Catapult path emits no such promise, so the attribute buys nothing.
    # It would also not compile: `static alignas(64) int8_t x[N]` puts a
    # standard attribute in the middle of the decl-specifiers, which g++
    # REJECTS under `-std=c++11` -- the standard Catapult's emitted tcl sets.
    # (cosim.py gets away with it only because Vitis compiles its testbench
    # with `-std=gnu++0x`.) Found by compiling this testbench; see RUNME.md.
    return (f"static {ctype} {name}[{len(vals)}] = {{"
            + ", ".join(str(int(v)) for v in vals) + "};\n")


def testbench(case_list, dist):
    M, K, N = SHAPE
    n = MAXDIM * MAXDIM
    src = [f"""// SCVerify testbench for the Allo-emitted `{TOP}` region (kernel.cpp).
//
// GENERATED by examples/tinytpu/ppa_catapult.py at T={T}, MAXDIM={MAXDIM},
// IMEM_SIZE={IMEM_SIZE}, operand distribution "{dist}". Do not edit; regenerate.
//
// `go switching` runs THIS against the pre-power RTL, so this stimulus is the
// workload the power number describes: {len(case_list)} calls on one instance,
// {M}x{K}x{N} GEMM plus one vector program, full-range int8. What that does and
// does not represent is in ppa_catapult.py's docstring and in RUNME.md.
//
// The design is a `@df.region()`; Allo emits it as a plain C++ function with
// array parameters and `#pragma hls_design top`, which is the same shape
// SCVerify wrapped for `mac16`. NOTE the declaration below has C++ linkage: it
// matches kernel.cpp, which the Catapult emitter writes WITHOUT `extern "C"`
// (unlike the Vitis path, whose testbenches declare the top `extern "C"`).
#include <ac_int.h>
#include <mc_scverify.h>
#include <cstdint>
#include <cstdio>

void {TOP}(uint64_t imem[{IMEM_SIZE}], int8_t A[{n}], int8_t B[{n}],
           int8_t C[{n}]);

static int8_t C[{n}];
"""]
    calls = []
    for i, (name, prog, A, B, C0, gold, mask) in enumerate(case_list):
        src += [carr(f"imem{i}", imem_of(prog), "uint64_t"),
                carr(f"A{i}", A.reshape(-1), "int8_t"),
                carr(f"B{i}", B.reshape(-1), "int8_t"),
                carr(f"C0_{i}", C0, "int8_t"),
                carr(f"gold{i}", gold, "int8_t"),
                carr(f"mask{i}", mask, "int8_t")]
        calls.append(f"""
  for (int k = 0; k < {n}; ++k) C[k] = C0_{i}[k];
  CCS_DESIGN({TOP})(imem{i}, A{i}, B{i}, C);
  wrong = clob = 0;
  for (int k = 0; k < {n}; ++k)
    if (C[k] != gold{i}[k]) {{ if (mask{i}[k]) ++wrong; else ++clob; }}
  printf("TINYTPU case {i} %-22s wrong=%d clobbered=%d\\n",
         "{name}", wrong, clob);
  errs += wrong; clobbered += clob;
""")
    src.append("CCS_MAIN(int argc, char **argv) {\n"
               "  int errs = 0, clobbered = 0, wrong, clob;\n"
               + "".join(calls)
               + f"""  printf("TINYTPU TB errors=%d clobbered=%d over {len(case_list)} calls\\n",
         errs, clobbered);
  CCS_RETURN(errs != 0);
}}
""")
    return "".join(src)


# ---------------------------------------------------------------------------
# The compile check. ace-01 has no Catapult, so the testbench would otherwise
# reach the licence host never having been through a compiler, and the first
# thing anyone learns would be a syntax error after a licence checkout. These
# shims are the two Catapult headers reduced to what the testbench uses, and
# the stand-in design replays a reference dump. Running it three times -- once
# correct, once with one result cell wrong, once with one preserved cell
# clobbered -- also shows the self-check can actually go red, and that a
# clobber is reported without failing (see the module docstring).
_SHIM_SCVERIFY = """#pragma once
#define CCS_MAIN(a, b) int main(a, b)
#define CCS_DESIGN(x) x
#define CCS_RETURN(x) return (x)
"""
_STUB = """#include <cstdint>
#include <cstdio>
#include <cstdlib>
static FILE *f = 0;
void %s(uint64_t *, int8_t *, int8_t *, int8_t *C) {
  if (!f) { f = fopen("ref.bin", "rb"); if (!f) { perror("ref.bin"); exit(2); } }
  if (fread(C, 1, %d, f) != %d) { fprintf(stderr, "short ref.bin\\n"); exit(2); }
}
"""


def compile_check(tb_path, case_list):
    """Compile the testbench with `-std=c++11` (what run.tcl sets) and run it
    against a correct and two doctored reference dumps."""
    import subprocess
    import tempfile

    n = MAXDIM * MAXDIM
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "shim"))
        open(os.path.join(d, "shim", "ac_int.h"), "w").write("#pragma once\n")
        open(os.path.join(d, "shim", "mc_scverify.h"), "w").write(_SHIM_SCVERIFY)
        open(os.path.join(d, "stub.cpp"), "w").write(_STUB % (TOP, n, n))
        cp = subprocess.run(["g++", "-std=c++11", "-I", "shim", "-o", "tb",
                             tb_path, "stub.cpp"], cwd=d,
                            capture_output=True, text=True)
        if cp.returncode:
            raise SystemExit("testbench does not compile under -std=c++11:\n"
                             + cp.stdout + cp.stderr)
        print("   compiles under -std=c++11")

        def dump(doctor):
            with open(os.path.join(d, "ref.bin"), "wb") as f:
                for i, (*_, gold, mask) in enumerate(case_list):
                    g = gold.copy()
                    doctor(i, g, mask)
                    f.write(g.astype(np.int8).tobytes())

        def run():
            r = subprocess.run(["./tb"], cwd=d, capture_output=True, text=True)
            last = r.stdout.strip().splitlines()[-1]
            return r.returncode, last

        dump(lambda i, g, m: None)
        rc, last = run()
        assert rc == 0 and "errors=0 clobbered=0" in last, last
        print(f"   correct design  -> exit 0, {last!r}")

        # one cell the program is expected to write, flipped
        def wrong(i, g, m):
            if i == 0:
                j = int(np.flatnonzero(m)[0])
                g[j] = np.int8(g[j] ^ 1)

        dump(wrong)
        rc, last = run()
        assert rc != 0 and "errors=1" in last, last
        print(f"   one result cell wrong -> exit {rc}, {last!r}")

        # one cell it must leave alone, flipped: reported, NOT a failure
        idx = [i for i, (*_, m) in enumerate(case_list) if (m == 0).any()]
        if not idx:
            print("   (no preserved cells at this shape; clobber arm skipped)")
            return
        k = idx[0]

        def clob(i, g, m):
            if i == k:
                j = int(np.flatnonzero(m == 0)[0])
                g[j] = np.int8(g[j] ^ 1)

        dump(clob)
        rc, last = run()
        assert rc == 0 and "errors=0 clobbered=1" in last, last
        print(f"   one preserved cell clobbered -> exit 0, {last!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-o", "--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "dev",
        "records", "catapult_handoff", "ppa_tinytpu"))
    ap.add_argument("--dist", default="full",
                    choices=("full", "mid", "small", "corner"),
                    help="operand distribution; 'small' is [-4,4], the "
                         "Gemmini/cycle-comparison one (biases power low)")
    ap.add_argument("--ncsim-root", default=NCSIM_ROOT)
    ap.add_argument("--tb-only", action="store_true",
                    help="write the testbench and stop (no MLIR build)")
    ap.add_argument("--no-simulator", action="store_true",
                    help="skip the simulator cross-check (not for a handoff)")
    ap.add_argument("--no-compile-check", action="store_true",
                    help="skip compiling the testbench against header shims")
    args = ap.parse_args()

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    print(f"TinyTPU-isa ppa handoff: T={T}, MAXDIM={MAXDIM}, "
          f"IMEM_SIZE={IMEM_SIZE}, shape={SHAPE}, dist={args.dist}")

    case_list = cases(args.dist)
    if not args.no_simulator:
        check_on_simulator(case_list)

    # How sharp the in-simulation self-check is. At 16x16x16 an int8 result
    # cannot avoid the clip: K=16 terms of full-range int8 saturate unless the
    # operands are within about +-3, which is exactly why the CYCLE testbench
    # uses [-4, 4]. A saturated cell still carries the SIGN of the whole
    # 16-term dot product, so the check is far from vacuous, but it is honest
    # to print how much of it is sign and how much is value.
    checked = sum(int(m.sum()) for *_, m in case_list)
    clipped = sum(int(((g == 127) | (g == -128))[m.astype(bool)].sum())
                  for *_, g, m in case_list)
    print(f"   self-check: {checked} result cells over {len(case_list)} calls, "
          f"{clipped} of them at a clip boundary "
          f"({100.0 * clipped / checked:.1f} %)")

    tb_path = os.path.join(out, "tinytpu_tb.cpp")
    with open(tb_path, "w", encoding="utf-8") as f:
        f.write(testbench(case_list, args.dist))
    print(f"   wrote {tb_path}")
    if not args.no_compile_check:
        compile_check(tb_path, case_list)
    if args.tb_only:
        return

    s = customize(tinytpu_isa)
    schedule(s)
    # wrap_io=False is the build every other TinyTPU flow uses: the units
    # address the arrays themselves.
    s.build(target="catapult", mode="ppa", project=out, wrap_io=False,
            configs={"testbench": tb_path,
                     "ncsim_root": args.ncsim_root,
                     "clock_period": CLOCK_PERIOD,
                     "library": LIBRARY})
    print(f"   wrote {out}/kernel.cpp, run.tcl, kernel.h, Makefile")


if __name__ == "__main__":
    main()
