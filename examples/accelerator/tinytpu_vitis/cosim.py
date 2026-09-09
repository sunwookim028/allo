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
report a worst-case bound. See `RESULTS_ISA.md`.

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
    tinytpu_isa, gemm_program, assemble, schedule, MAXDIM, T, IMEM_SIZE,
)

VITIS = "/opt/xilinx/Vitis_HLS/2023.2/settings64.sh"
LDFLAGS = "-B/usr/bin"
SHAPES = [(4, 4, 4), (8, 8, 8), (12, 12, 12), (16, 16, 8), (16, 16, 16)]


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
    return f"static {ctype} {name}[{len(vals)}] = {{" + \
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
           f"static int8_t C[{MAXDIM * MAXDIM}];\n",
           body]
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


TCL_SYN = """open_project out.prj -reset
open_solution -reset solution1 -flow_target vivado
set_top tinytpu_isa
add_files kernel.cpp
add_files -tb tb.cpp -cflags "-std=gnu++0x"
set_part {xcu280-fsvh2892-2L-e}
create_clock -period 3.33
csynth_design
exit
"""

TCL_COSIM = """open_project out.prj
open_solution solution1
set_top tinytpu_isa
cosim_design -trace_level none -rtl verilog -ldflags "%s"
exit
""" % LDFLAGS


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
    prj = os.path.abspath("isa_sweep.prj")
    print(f"TinyTPU-isa: ONE build -- {T}x{T} array, MAXDIM={MAXDIM}; "
          f"sweeping {len(SHAPES)} shapes as data")

    s = customize(tinytpu_isa)
    schedule(s)
    # `wrap_io` is a measured architectural trade, not a default to accept:
    #   True  -- Allo copies each argument into a local buffer first, so the
    #            units read BRAM. Fixed cost = the declared sizes (imem + A +
    #            B + C), marginal cost 18.1 cyc/instr.
    #   False -- units read m_axi directly. Fixed cost 481 cycles, matching
    #            Gemmini's 483, but marginal cost 39.8 cyc/instr because every
    #            access pays bus latency instead of hitting a buffer.
    # Crossover is ~29 instructions, so True wins across this benchmark set
    # once `IMEM_SIZE` is trimmed to the longest admissible program.
    # Neither is what Gemmini has, which is a bursted DMA the program controls:
    # low fixed cost AND low marginal cost. That is the gap.
    s.build(target="vitis_hls", mode="csyn", project=prj,
            wrap_io=(os.environ.get("TPU_WRAP", "1") == "1"))
    patch_axi_depths(prj)
    open(os.path.join(prj, "tb.cpp"), "w").write(testbench(*SHAPES[0]))
    print("  synthesizing once ...", flush=True)
    vitis(prj, TCL_SYN, "csynth.log")

    results = {}
    for (M, K, N) in SHAPES:
        open(os.path.join(prj, "tb.cpp"), "w").write(testbench(M, K, N))
        text = vitis(prj, TCL_COSIM, f"cosim_{M}x{K}x{N}.log")
        mm = [l.strip() for l in text.splitlines() if "mismatches" in l]
        n = cycles(prj)
        results[(M, K, N)] = n
        tag = mm[0] if mm else "no TB line"
        print(f"  {M:2d}x{K:2d}x{N:2d}  cycles={n}   {tag}", flush=True)
    print("\n  shape      cycles")
    for k, v in results.items():
        print(f"  {k[0]:2d}x{k[1]:2d}x{k[2]:2d}   {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
