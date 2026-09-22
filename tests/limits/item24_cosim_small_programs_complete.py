# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Item 24, the small-program half: four programs that all complete.

Item 24 is a legal TinyTPU-isa program whose Vitis cosim does not finish; the
minimal case and its bisection are `item24_cosim_hang.py`. This file is the
other direction -- the attempt to reproduce it from the small end, by building
the structural feature that was suspected at the time: a data transfer inside
the emitted loop nest rather than hoisted into a prologue.

**It does not reproduce, and the hypothesis is dead.** Measured 2026-09-22, all
four cases complete: 169 / 189 / 171 / 186 cycles. `item24_cosim_hang.py` killed
the same hypothesis from the failing end -- both non-completing programs stage
every transfer in a prologue, and both mappings that do stage inside the nest
complete. In-nest staging is neither necessary nor sufficient.

It is kept as the small-program half of the evidence, and for its harness:
bounded cosim of an arbitrary program with each child in its own process group,
killed as a group on timeout, because killing `vitis_hls` leaves `xsimk` holding
a core -- one survived its parent by 32 minutes here.

    python tests/limits/item24_cosim_small_programs_complete.py [--bound 240]

Needs Vitis (see docs/source/developer/toolchains.rst); one synthesis, then one
cosim per case. Extend the ladder by adding a function to LADDER.
"""

import argparse
import os
import signal
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..")))
from examples.accelerator.tinytpu_vitis import cosim, isa_ref  # noqa: E402
from examples.accelerator.tinytpu_vitis.isa_dsl import (  # noqa: E402
    Program, Ref,
)
from examples.accelerator.tinytpu_vitis.microarch_isa import (  # noqa: E402
    IMEM_SIZE, MAXDIM, T, assemble,
)

ROWS = 4


def prologue_only():
    """The shipped shape: every transfer before every compute, no loop."""
    k = Program("prologue-only")
    k.dma_ld(src=0, dram_row=0, col_block=0, vr=0, rows=ROWS)
    k.dma_ld(src=1, dram_row=0, col_block=0, spad=0, rows=T)
    k.mm(0, 0, 0, rows=ROWS)
    k.mvout(0, dram_row=0, col_block=0, rows=ROWS)
    return k.emit()


def dma_after_compute():
    """A `dma_ld` issued after an `mm`. Straight line, no loop."""
    k = Program("dma-after-compute")
    k.dma_ld(src=0, dram_row=0, col_block=0, vr=0, rows=ROWS)
    k.dma_ld(src=1, dram_row=0, col_block=0, spad=0, rows=T)
    k.mm(0, 0, 0, rows=ROWS)
    k.dma_ld(src=0, dram_row=0, col_block=1, vr=8, rows=ROWS)
    k.mm(8, 16, 0, rows=ROWS)
    k.mvout(0, dram_row=0, col_block=0, rows=ROWS)
    k.mvout(16, dram_row=0, col_block=1, rows=ROWS)
    return k.emit()


def dma_in_loop_before_compute():
    """A `dma_ld` inside a loop, with every compute after the loop."""
    k = Program("dma-in-loop-before-compute")
    k.dma_ld(src=1, dram_row=0, col_block=0, spad=0, rows=T)
    with k.loop(2, "a") as block:
        k.dma_ld(src=0, dram_row=0, col_block=Ref().at(block, 1),
                 vr=Ref(0).at(block, 8), rows=ROWS)
    k.mm(0, 0, 0, rows=ROWS)
    k.mvout(0, dram_row=0, col_block=0, rows=ROWS)
    return k.emit()


def dma_in_loop_with_compute():
    """A `dma_ld` inside a loop that also holds the compute -- in-nest staging,
    reduced to two trips."""
    k = Program("dma-in-loop-with-compute")
    k.dma_ld(src=1, dram_row=0, col_block=0, spad=0, rows=T)
    with k.loop(2, "n") as block:
        k.dma_ld(src=0, dram_row=0, col_block=Ref().at(block, 1),
                 vr=Ref(0).at(block, 8), rows=ROWS)
        k.mm(Ref(0).at(block, 8), Ref(0).at(block, 8), 0, rows=ROWS)
        k.mvout(Ref(0).at(block, 8), dram_row=0,
                col_block=Ref().at(block, 1), rows=ROWS)
    return k.emit()


LADDER = (prologue_only, dma_after_compute, dma_in_loop_before_compute,
          dma_in_loop_with_compute)


def operands(seed=0):
    rng = np.random.default_rng(seed)
    return (rng.integers(-4, 5, (MAXDIM, MAXDIM)).astype(np.int8),
            rng.integers(-4, 5, (MAXDIM, MAXDIM)).astype(np.int8))


def testbench(program, A, B):
    """One call, whole `C` against `isa_ref`. Completion is what is measured;
    the comparison is here so a case that finishes is known to be right."""
    C0 = np.zeros(MAXDIM * MAXDIM, np.int8)
    gold = isa_ref.run(program, A.reshape(-1), B.reshape(-1), C0)
    words = assemble(program)
    imem = np.zeros(IMEM_SIZE, np.uint64)
    imem[: len(words)] = np.array(words, np.uint64)
    return "".join([
        "#include <cstdio>\n#include <cstdint>\n",
        'extern "C" void tinytpu_isa(uint64_t *, int8_t *, int8_t *, int8_t *);\n',
        cosim.carr("imem", imem, "uint64_t"),
        cosim.carr("A", A.reshape(-1), "int8_t"),
        cosim.carr("B", B.reshape(-1), "int8_t"),
        cosim.carr("gold", gold, "int8_t"),
        f"static alignas(64) int8_t C[{MAXDIM * MAXDIM}];\n",
        f"""int main() {{
  for (int i = 0; i < {MAXDIM * MAXDIM}; i++) C[i] = 0;
  tinytpu_isa(imem, A, B, C);
  int bad = 0;
  for (int i = 0; i < {MAXDIM * MAXDIM}; i++) if (C[i] != gold[i]) bad++;
  printf("TB mismatches = %d\\n", bad);
  return bad == 0 ? 0 : 1;
}}
"""])


def run_vitis(prj, tcl, log, bound):
    """Vitis in its own process group, killed as a group on timeout.

    Killing only the parent leaves `xsimk` running at 100% of a core for as
    long as the host is up; this is the whole reason the helper exists.
    """
    open(os.path.join(prj, "run.tcl"), "w").write(tcl)
    started = time.time()
    with open(os.path.join(prj, log), "w") as sink:
        child = subprocess.Popen(
            ["bash", "-lc",
             f"source {cosim.VITIS} && cd {prj} && vitis_hls -f run.tcl"],
            stdout=sink, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            child.wait(timeout=bound)
            timed_out = False
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(child.pid), signal.SIGKILL)
            child.wait()
            timed_out = True
    return timed_out, time.time() - started


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bound", type=int, default=240,
                    help="seconds one cosim may take before it is killed")
    ap.add_argument("--project", default="/tmp/item24.prj")
    args = ap.parse_args(argv)

    from allo.dataflow import customize
    from examples.accelerator.tinytpu_vitis.microarch_isa import (
        schedule, tinytpu_isa)

    A, B = operands()
    for build in LADDER:                       # every case is legal and right
        program = build()
        assemble(program)
        isa_ref.run(program, A.reshape(-1), B.reshape(-1),
                    np.zeros(MAXDIM * MAXDIM, np.int8))

    prj = args.project
    s = customize(tinytpu_isa)
    schedule(s)
    s.build(target="vitis_hls", mode="csyn", project=prj, wrap_io=False,
            configs={"align_value": 64})
    cosim.patch_axi_depths(prj)
    open(os.path.join(prj, "tb.cpp"), "w").write(
        testbench(LADDER[0](), A, B))
    print(f"synthesizing once into {prj} ...", flush=True)
    run_vitis(prj, cosim.TCL_SYN, "csynth.log", 1800)

    report = os.path.join(prj, "out.prj/solution1/sim/report",
                          "tinytpu_isa_cosim.rpt")
    stalled = []
    for build in LADDER:
        program = build()
        name = build.__name__
        open(os.path.join(prj, "tb.cpp"), "w").write(testbench(program, A, B))
        if os.path.exists(report):
            os.remove(report)
        timed_out, seconds = run_vitis(prj, cosim.TCL_COSIM,
                                       f"cosim_{name}.log", args.bound)
        text = open(os.path.join(prj, f"cosim_{name}.log"),
                    errors="replace").read()
        csim = "mismatches = 0" in text
        got = cosim.cycles(prj)
        verdict = (f"cycles={got}" if got is not None
                   else f"NO REPORT after {seconds:.0f}s"
                        f"{' (killed at the bound)' if timed_out else ''}")
        if got is None:
            stalled.append(name)
        print(f"  {name:28s} {len(program):2d} instr  "
              f"csim {'ok' if csim else 'MISMATCH'}  {verdict}", flush=True)
    print(f"\nITEM24 SMALL PROGRAMS: {len(LADDER) - len(stalled)} of "
          f"{len(LADDER)} completed"
          + (f"; no cosim report for {', '.join(stalled)}" if stalled else
             " -- in-nest staging does not reproduce item 24 at this scale"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
