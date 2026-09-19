# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cycle-count TinyTPU-isa on ONE fixed hardware build, sweeping the workload.

`tinytpu_isa` is synthesized once. Each shape is a different instruction stream
run on that same RTL, which is what makes the numbers comparable to Gemmini's:
`allo_cmp.c` runs every shape on one elaboration too, with `A[MAXDIM][MAXDIM]`
and MAXDIM as the stride.

Vitis `cosim_design` runs one testbench per invocation, so the sweep works by
emitting one `tb_<shape>.cpp` per shape into the *same* project and re-running
`cosim_design` -- `csynth_design` runs once, so the hardware really is built
once and the RTL under test is identical for every shape.

Why cosim rather than the synthesis report: the loop bounds are now runtime
data (that is what makes the design workload-independent), so csynth can only
report a worst-case bound. See docs/source/designs/tinytpu_isa.rst.

TWO TESTBENCH MODES -- which one you ran decides what a PASS means:

  * `TPU_TB` unset (the DEFAULT, and the only mode the published cycle counts
    252 / 383 / 591 / 667 / 919 come from): one GEMM call per shape, operands
    in [-4, 4] from seed 0 -- the distribution Gemmini's `allo_cmp.c` fills,
    kept so the comparison is like for like -- `C` zeroed, and only the
    `M x N` region compared. It is a PERFORMANCE testbench. It cannot see a
    narrowed accumulator (a T-deep partial sum of [-4, 4] fits in 9 bits), a
    wrong clip boundary, a unit ignoring a field GEMM never varies, a design
    that relies on `C` arriving zeroed, or state left by an earlier call.
  * `TPU_TB=stress`: a CORRECTNESS testbench on the same RTL. Per shape it
    calls the kernel several times in one simulation -- corner operands
    ({-128, -127, -1, 0, 1, 126, 127}), uniform full-range int8, a directed
    case whose results sit exactly on the clip and ReLU boundaries, a mid
    range, and `isa_dsl.vector_program` -- each with `C` prefilled with random
    bytes and the WHOLE of `C` compared against `isa_ref`/numpy, so the
    region must be exact and everything outside it untouched. The calls share
    one RTL instance, so each sees the `spad`/`vr`/`ar` the previous left.
    Its cycle column is the minimum over those calls; it is not the headline.
    Each per-case line appears twice: cosim runs the testbench once in C, then
    again against the RTL; the summary line is the RTL's.

The functional equivalent of `TPU_TB=stress` is `stress_isa.py` (seconds, on
Allo's simulator); run that first. This mode is for what only RTL can show.

Three toolchain fixes are needed and are applied here:
  * a real C++ testbench -- `df.build(mode=...)` handles only csim/csyn and the
    emitted `host.cpp` is an OpenCL/XRT host;
  * `-B/usr/bin` -- Vitis 2023.2's binutils 2.37 cannot read this system's
    glibc (`unknown type [0x13] section '.relr.dyn'`), so both links fail;
  * explicit `m_axi` depths, which cosim requires and Allo does not emit.
"""

import os
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..", "..")))
from allo.dataflow import customize  # noqa: E402
from examples.accelerator.tinytpu_vitis.microarch_isa import (  # noqa: E402
    tinytpu_isa, assemble, schedule, MAXDIM, T, IMEM_SIZE,
)
from examples.accelerator.tinytpu_vitis.isa_dsl import gemm_program  # noqa: E402

VITIS = "/opt/xilinx/Vitis_HLS/2023.2/settings64.sh"
LDFLAGS = "-B/usr/bin"
_ALL = [(4, 4, 4), (8, 8, 8), (12, 12, 12), (16, 16, 8), (16, 16, 16)]
# Only shapes the built array can express: every dimension must be a multiple
# of T, since one vmatpush-equivalent is a whole packed word of T lanes.
SHAPES = [s for s in _ALL if all(d % T == 0 for d in s)]
if os.environ.get("TPU_SHAPES"):        # e.g. TPU_SHAPES=4x4x4,16x16x16
    # Any shape the build can express, not only the five scored ones.
    SHAPES = [tuple(int(x) for x in t.split("x"))
              for t in os.environ["TPU_SHAPES"].split(",")]
    for _s in SHAPES:
        assert all(d % T == 0 and d <= MAXDIM for d in _s), (
            f"TPU_SHAPES: {_s} is not a multiple of T={T} within MAXDIM={MAXDIM}")

TB_MODE = os.environ.get("TPU_TB", "default")   # "default" or "stress"; see top
assert TB_MODE in ("default", "stress"), f"TPU_TB={TB_MODE!r}"
# Where the Vitis project goes. Default: next to this file, whatever the cwd.
PRJ = os.path.abspath(os.environ.get(
    "TPU_PRJ", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "isa_sweep.prj")))

# Memory-model knobs, for asking how much of the result is an ideal AXI slave.
# `-m_axi_latency` is the read latency HLS schedules against (DEFAULT 0, i.e. a
# memory that answers immediately); `-random_stall` makes cosim stall the
# top-level interfaces at random instead of never.
AXI_LATENCY = os.environ.get("TPU_AXI_LATENCY", "")
RANDOM_STALL = os.environ.get("TPU_RANDOM_STALL", "") == "1"


def vectors(M, K, N, relu=False, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.integers(-4, 5, (MAXDIM, MAXDIM)).astype(np.int8)
    B = rng.integers(-4, 5, (MAXDIM, MAXDIM)).astype(np.int8)
    words = assemble(gemm_program(M, K, N, relu))
    imem = np.zeros(IMEM_SIZE, np.uint64)
    imem[: len(words)] = np.array(words, np.uint64)
    gold = A[:M, :K].astype(np.int64) @ B[:K, :N].astype(np.int64)
    if relu:
        gold = np.maximum(gold, 0)
    return imem, A, B, np.clip(gold, -128, 127).astype(np.int8), len(words)


def carr(name, vals, ctype):
    # `alignas(64)` is not cosmetic: the kernel is emitted with
    # `align_value(64)` on its pointers, which is a PROMISE to Vitis. On real
    # hardware XRT buffers are 4 KB aligned so it holds for free; in cosim the
    # testbench is the host, so it has to keep the promise itself.
    return f"static alignas(64) {ctype} {name}[{len(vals)}] = {{" + \
           ", ".join(str(int(v)) for v in vals) + "};\n"


def testbench(M, K, N, relu=False):
    imem, A, B, gold, _ = vectors(M, K, N, relu)
    body = f"""
int main() {{
  for (int i = 0; i < {MAXDIM * MAXDIM}; i++) C[i] = 0;
  tinytpu_isa(imem, A, B, C);
  int bad = 0;
  for (int i = 0; i < {M}; i++)
    for (int j = 0; j < {N}; j++)
      if (C[i * {MAXDIM} + j] != gold[i * {N} + j]) bad++;
  printf("TB {M}x{K}x{N} mismatches = %d / {M * N}\\n", bad);
  return bad == 0 ? 0 : 1;
}}
"""
    src = ["#include <cstdio>\n#include <cstdint>\n",
           'extern "C" void tinytpu_isa(uint64_t *, int8_t *, int8_t *, int8_t *);\n',
           carr("imem", imem, "uint64_t"),
           carr("A", A.reshape(-1), "int8_t"),
           carr("B", B.reshape(-1), "int8_t"),
           carr("gold", gold.reshape(-1), "int8_t"),
           f"static alignas(64) int8_t C[{MAXDIM * MAXDIM}];\n",
           body]
    return "".join(src)


def stress_testbench(M, K, N):
    """`TPU_TB=stress`: several calls on one RTL instance, whole `C` checked.
    The cases are the ones `stress_isa.py` runs for the scored shapes."""
    from examples.accelerator.tinytpu_vitis import isa_ref
    from examples.accelerator.tinytpu_vitis.isa_dsl import vector_program
    from examples.accelerator.tinytpu_vitis.stress_isa import (
        operands, boundary_operands, gemm_gold)
    crng = np.random.default_rng(4321 + M * 10000 + K * 100 + N)
    cases = []
    for i, (dist, relu) in enumerate((("corner", False), ("full", True),
                                      ("boundary", False), ("mid", True))):
        seed = 900 + i
        A, B = (boundary_operands(M, K, N, seed) if dist == "boundary"
                else operands(dist, seed))
        C0 = crng.integers(-128, 128, MAXDIM * MAXDIM).astype(np.int8)
        prog = gemm_program(M, K, N, relu)
        gold = gemm_gold(A, B, M, K, N, relu, C0)
        assert (isa_ref.run(prog, A, B, C0) == gold).all()
        cases.append((f"gemm{'.relu' if relu else ''} {dist}", prog, A, B, C0, gold))
    A, B = operands("full", 950)
    C0 = crng.integers(-128, 128, MAXDIM * MAXDIM).astype(np.int8)
    prog = vector_program(8)
    cases.append(("vector_program(8) full", prog, A, B, C0,
                  isa_ref.run(prog, A, B, C0)))

    src = ["#include <cstdio>\n#include <cstdint>\n",
           'extern "C" void tinytpu_isa(uint64_t *, int8_t *, int8_t *, int8_t *);\n',
           f"static alignas(64) int8_t C[{MAXDIM * MAXDIM}];\n"]
    calls = []
    for i, (name, prog, A, B, C0, gold) in enumerate(cases):
        words = assemble(prog)
        imem = np.zeros(IMEM_SIZE, np.uint64)
        imem[: len(words)] = np.array(words, np.uint64)
        src += [carr(f"imem{i}", imem, "uint64_t"),
                carr(f"A{i}", A.reshape(-1), "int8_t"),
                carr(f"B{i}", B.reshape(-1), "int8_t"),
                carr(f"C0_{i}", C0, "int8_t"),
                carr(f"gold{i}", gold, "int8_t")]
        calls.append(f"""
  for (int i = 0; i < {MAXDIM * MAXDIM}; i++) C[i] = C0_{i}[i];
  tinytpu_isa(imem{i}, A{i}, B{i}, C);
  n_in = n_out = 0;
  for (int i = 0; i < {MAXDIM}; i++)
    for (int j = 0; j < {MAXDIM}; j++)
      if (C[i * {MAXDIM} + j] != gold{i}[i * {MAXDIM} + j]) {{
        if ({'i < %d && j < %d' % (M, N) if name.startswith('gemm') else '1'}) n_in++; else n_out++;
      }}
  printf("TB {M}x{K}x{N} case {i} {name:24s} wrong = %d, clobbered outside = %d\\n", n_in, n_out);
  bad += n_in + n_out;""")
    src.append("int main() {\n  int bad = 0, n_in, n_out;" + "".join(calls) + f"""
  printf("TB {M}x{K}x{N} stress mismatches = %d over {len(cases)} calls\\n", bad);
  return bad == 0 ? 0 : 1;
}}
""")
    return "".join(src)


def patch_axi_depths(prj):
    """Cosim needs to know how much memory sits behind each m_axi port."""
    depths = [IMEM_SIZE, MAXDIM * MAXDIM, MAXDIM * MAXDIM, MAXDIM * MAXDIM]
    path = os.path.join(prj, "kernel.cpp")
    src = open(path).read()
    def sub(m):
        i = int(m.group(2))
        return f"{m.group(1)} depth={depths[i]}" if i < len(depths) else m.group(0)
    out, n = re.subn(
        r"(#pragma HLS interface m_axi port=\w+ offset=slave bundle=gmem(\d+))",
        sub, src)
    assert n == len(depths), f"patched {n} m_axi pragmas, expected {len(depths)}"
    open(path, "w").write(out)


_LAT = f"config_interface -m_axi_latency {AXI_LATENCY}" if AXI_LATENCY else ""
TCL_SYN = """open_project out.prj -reset
open_solution -reset solution1 -flow_target vivado
set_top tinytpu_isa
add_files kernel.cpp
add_files -tb tb.cpp -cflags "-std=gnu++0x"
set_part {xcu280-fsvh2892-2L-e}
create_clock -period 3.33
config_interface -m_axi_max_widen_bitwidth 512
%s
csynth_design
exit
""" % _LAT

TCL_COSIM = """open_project out.prj
open_solution solution1
set_top tinytpu_isa
cosim_design -trace_level none -rtl verilog -ldflags "%s"%s
exit
""" % (LDFLAGS, " -random_stall" if RANDOM_STALL else "")


def vitis(prj, tcl, log):
    open(os.path.join(prj, "run.tcl"), "w").write(tcl)
    with open(os.path.join(prj, log), "w") as f:
        subprocess.call(["bash", "-lc",
                         f"source {VITIS} && cd {prj} && vitis_hls -f run.tcl"],
                        stdout=f, stderr=subprocess.STDOUT)
    return open(os.path.join(prj, log), errors="replace").read()


def cycles(prj):
    rpt = os.path.join(prj, "out.prj/solution1/sim/report/tinytpu_isa_cosim.rpt")
    if not os.path.exists(rpt):
        return None
    for line in open(rpt):
        if "Verilog" in line and "Pass" in line:
            n = re.findall(r"\d+", line)
            if n:
                return int(n[0])
    return None


def main():
    prj = PRJ
    tb = stress_testbench if TB_MODE == "stress" else testbench
    print(f"TinyTPU-isa: ONE build -- {T}x{T} array, MAXDIM={MAXDIM}; "
          f"sweeping {len(SHAPES)} shapes as data; testbench={TB_MODE}"
          + ("" if TB_MODE == "default" else
             " (correctness mode: cycles are NOT the published numbers)"))

    s = customize(tinytpu_isa)
    schedule(s)
    # `wrap_io=False` is the build, and it is now the right one rather than
    # half of a trade:
    #   True  -- Allo hoists every `m_axi` argument into a local buffer before
    #            the region starts, with `wrap_data_movement`'s extent taken
    #            from the STATIC type. At MAXDIM=16 that is imem 56 + A 256 +
    #            B 256 + C 256 = 824 words copied whether the program touches
    #            them or not, and it is 907 of the 1586 cycles at 16x16x16 and
    #            90% of them at 4x4x4.
    #   False -- the units address `m_axi` themselves, so each one bursts
    #            exactly what its program names. Measured fixed cost 557,
    #            marginal 20.1 cyc/instr, and faster at all five shapes.
    # An earlier measurement of `wrap_io=False` (fixed 481, marginal 39.8) is
    # what made this look like a trade. It was measured with the STRIDED access
    # pattern, which Vitis turns into a four-beat AXI transaction per row, and
    # with instruction fetch going to `m_axi` at a data-dependent address,
    # which it can only burst two words at a time. Both are properties of the
    # access pattern, not of the configuration; `microarch_isa` now avoids
    # both. `TPU_WRAP=1` still builds the hoisted variant for comparison.
    s.build(target="vitis_hls", mode="csyn", project=prj,
            wrap_io=(os.environ.get("TPU_WRAP", "0") == "1"),
            configs={"align_value": 64})
    patch_axi_depths(prj)
    open(os.path.join(prj, "tb.cpp"), "w").write(tb(*SHAPES[0]))
    print("  synthesizing once ...", flush=True)
    vitis(prj, TCL_SYN, "csynth.log")

    results, ok = {}, True
    for (M, K, N) in SHAPES:
        open(os.path.join(prj, "tb.cpp"), "w").write(tb(M, K, N))
        # A stale report from the previous shape must not be read as this one's.
        rpt = os.path.join(prj, "out.prj/solution1/sim/report/tinytpu_isa_cosim.rpt")
        if os.path.exists(rpt):
            os.remove(rpt)
        text = vitis(prj, TCL_COSIM, f"cosim_{M}x{K}x{N}.log")
        mm = [l.strip() for l in text.splitlines() if "mismatches" in l]
        n = cycles(prj)
        results[(M, K, N)] = n
        good = (n is not None and bool(mm)
                and re.search(r"mismatches = 0\b", mm[-1]) is not None)
        ok &= good
        if TB_MODE == "stress":
            for l in text.splitlines():
                if l.startswith("TB ") and " case " in l:
                    print("    " + l.strip())
        tag = mm[-1] if mm else "no TB line"
        print(f"  {M:2d}x{K:2d}x{N:2d}  cycles={n}   {tag}"
              + ("" if good else "   <-- FAIL"), flush=True)
    print("\n  shape      cycles" + ("   (min over the stress calls)"
                                    if TB_MODE == "stress" else ""))
    for k, v in results.items():
        print(f"  {k[0]:2d}x{k[1]:2d}x{k[2]:2d}   {v}")
    print("  COSIM " + ("OK" if ok else "FAILED")
          + f" (testbench={TB_MODE})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
