# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run Vitis RTL co-simulation on TinyTPU-isa and report a real cycle count.

Why this file exists. `df.build(..., target="vitis_hls", mode=...)` handles
`csim` and `csyn`; every other mode is routed to the `XDEVICE` Makefile flow
(`allo/backend/hls.py`), and the `host.cpp` Allo emits is an OpenCL/XRT host,
which is not what `cosim_design` wants -- cosim needs a plain C++ `main` that
calls the top function directly. So this generates that testbench and drives
`vitis_hls` itself.

Why cosim at all, rather than reading the synthesis report. For a
fixed-function design the csynth interval *is* the cycle count
(`microarch_ws.py` hit its roofline and could be checked statically). For an
instruction-programmable design the loop trip counts are *data*, so synthesis
can only report a worst-case bound derived from the instruction encoding --
measured at 22x the real figure before the row-count field was narrowed
(`RESULTS_ISA.md`). A programmable accelerator has to be *run* to be timed, and
that is what makes cosim the only number comparable to Gemmini's `rdcycle`.

The test vectors come from the same `gemm_program()` and the same numpy
reference the simulator uses, so the RTL is checked against the identical
program and data, and the testbench cannot drift from the design.
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
    tinytpu_isa, gemm_program, schedule, M, K, N, T, NPROG, Kt, Nt, QD,
)

VITIS_SETTINGS = "/opt/xilinx/Vitis_HLS/2023.2/settings64.sh"
from examples.accelerator.tinytpu_vitis.microarch_isa import IMEM_SIZE as IMEM_WORDS  # noqa: E402


def vectors(relu=False, seed=0):
    """Exactly what `bench_isa.py` feeds the KPN simulator."""
    rng = np.random.default_rng(seed)
    A = rng.integers(-4, 5, (M, K)).astype(np.int8)
    B = rng.integers(-4, 5, (K, N)).astype(np.int8)
    prog = gemm_program(relu)
    imem = np.zeros(IMEM_WORDS, np.uint64)
    imem[: len(prog)] = np.array(prog, np.uint64)
    gold = A.astype(np.int64) @ B.astype(np.int64)
    if relu:
        gold = np.maximum(gold, 0)
    return imem, A, B, np.clip(gold, -128, 127).astype(np.int8)


def carr(name, vals, ctype):
    body = ", ".join(str(int(v)) for v in vals)
    return f"static {ctype} {name}[{len(vals)}] = {{{body}}};\n"


def testbench(relu=False):
    imem, A, B, gold = vectors(relu)
    src = ['#include <cstdio>\n#include <cstdint>\n',
           'extern "C" void tinytpu_isa(uint64_t *, int8_t *, int8_t *, int8_t *);\n']
    src.append(carr("imem", imem, "uint64_t"))
    src.append(carr("A", A.reshape(-1), "int8_t"))
    src.append(carr("B", B.reshape(-1), "int8_t"))
    src.append(carr("gold", gold.reshape(-1), "int8_t"))
    src.append(f"static int8_t C[{M * N}];\n")
    src.append(f"""
int main() {{
  for (int i = 0; i < {M * N}; i++) C[i] = 0;
  tinytpu_isa(imem, A, B, C);
  int bad = 0;
  for (int i = 0; i < {M * N}; i++)
    if (C[i] != gold[i]) {{
      if (bad < 8) printf("C[%d] = %d, want %d\\n", i, (int)C[i], (int)gold[i]);
      bad++;
    }}
  printf("TB: {M}x{K}x{N} {'gemm.relu' if relu else 'gemm'} mismatches = %d / {M * N}\\n", bad);
  return bad == 0 ? 0 : 1;
}}
""")
    return "".join(src)


# Vitis 2023.2 ships binutils 2.37, which cannot read this system's glibc
# (`unknown type [0x13] section '.relr.dyn'`, then `cannot find libm.so.6`), so
# both the csim and cosim links fail. `-B/usr/bin` points the compiler driver at
# the system linker (2.42) and both succeed.
LDFLAGS = "-B/usr/bin"

def patch_axi_depths(prj):
    """Give every m_axi port an explicit depth.

    Cosim refuses to run without one --
        ERROR: A depth specification is required for MAXI interface port
               'gmem0' for cosimulation.
    -- because it has to know how much memory to model behind each port. Allo
    emits the pragmas without a depth, so they are patched here, in the emitted
    kernel, in the port order of the top function: imem, A, B, C."""
    depths = [IMEM_WORDS, M * K, K * N, M * N]
    path = os.path.join(prj, "kernel.cpp")
    src = open(path).read()
    def sub(m):
        i = int(m.group(2))
        return f"{m.group(1)} depth={depths[i]}" if i < len(depths) else m.group(0)
    out, n = re.subn(r"(#pragma HLS interface m_axi port=\w+ offset=slave bundle=gmem(\d+))",
                     sub, src)
    assert n == len(depths), f"patched {n} m_axi pragmas, expected {len(depths)}"
    open(path, "w").write(out)
    print(f"  patched {n} m_axi depths: {depths}")


TCL = """set hls_prj out.prj
open_project ${{hls_prj}} -reset
open_solution -reset solution1 -flow_target vivado
set_top tinytpu_isa
add_files kernel.cpp
add_files -tb tb.cpp -cflags "-std=gnu++0x"
open_solution "solution1"
set_part {{xcu280-fsvh2892-2L-e}}
create_clock -period 3.33
csim_design -ldflags "{ldflags}"
csynth_design
cosim_design -trace_level none -rtl verilog -ldflags "{ldflags}"
exit
"""


def main():
    relu = "--relu" in sys.argv
    prj = os.path.abspath(f"isa_cosim_{M}x{K}x{N}{'_relu' if relu else ''}.prj")
    print(f"TinyTPU-isa cosim: {M}x{K}x{N}, {NPROG} instructions, "
          f"Kt={Kt} Nt={Nt}, QD={QD}")

    # Emit the project through the normal path, then replace the XRT host with
    # a real cosim testbench and the tcl with one that runs csim + cosim.
    s = customize(tinytpu_isa)
    schedule(s)
    s.build(target="vitis_hls", mode="csyn", project=prj)
    patch_axi_depths(prj)
    with open(os.path.join(prj, "tb.cpp"), "w") as f:
        f.write(testbench(relu))
    with open(os.path.join(prj, "run.tcl"), "w") as f:
        f.write(TCL.format(ldflags=LDFLAGS))
    print(f"  project -> {prj}")

    cmd = f"source {VITIS_SETTINGS} && cd {prj} && vitis_hls -f run.tcl"
    log = os.path.join(prj, "cosim.log")
    print(f"  running csim + csynth + cosim (log: {log})", flush=True)
    with open(log, "w") as f:
        rc = subprocess.call(["bash", "-lc", cmd], stdout=f, stderr=subprocess.STDOUT)
    text = open(log, errors="replace").read()

    for line in text.splitlines():
        if "mismatches" in line or "Simulation failed" in line:
            print(f"  {line.strip()}")
    # The cycle count lives in the cosim report, not the log.
    rpt = os.path.join(prj, "out.prj/solution1/sim/report/tinytpu_isa_cosim.rpt")
    if os.path.exists(rpt):
        for line in open(rpt):
            if "Verilog" in line and "Pass" in line:
                n = re.findall(r"\d+", line)
                if n:
                    print(f"\n  COSIM CYCLES ({M}x{K}x{N}): {n[0]}")
                    return 0
    print("\n  no cycle count; tail of log:")
    print("\n".join("   " + l for l in text.splitlines()[-25:]))
    return rc or 1


if __name__ == "__main__":
    sys.exit(main())
