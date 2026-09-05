/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_CONVERSION_PASSES_TD
#define ALLO_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

def ConvertAlloToFuncPass : Pass<"convert-allo-to-func"> {
  let summary = "Convert allo dialect to func dialect";
  let description = [{Convert allo dialect to func dialect}];
}

def LowerInstructionsPass
    : Pass<"lower-instructions", "ModuleOp"> {
  let summary = "Inline Allo instruction semantics into executable IR";
  let description = [{
    Lower ACT buffers to mutable globals and inline every emitted instruction's
    access and compute regions.  The resulting ordinary linalg/tensor/arith IR
    is used by the functional oracle and CPU execution path.
  }];
}

def LowerHLMemPass : Pass<"allo-lower-hlmem", "circt::hw::HWModuleOp"> {
  let summary = "Lower seq.hlmem and its ports onto an sv.reg array";
  let description = [{
    Replaces each `seq.hlmem` with an `sv.reg` array: a read port becomes a
    combinational read registered to its latency, with the address delayed for
    all but the last cycle; a write port becomes a guarded assignment in an
    `always_ff`. The memory's power-on contents (`allo.mem.init`) are written
    into the register through an `initial` block, with BLOCKING assignments,
    which is what synthesis reads back as a block-RAM INIT.

    The write ports share ONE always block unless the memory carries
    `allo.mem.independent_writes`, the emitter's promise that no two of them
    write one word in the same cycle. Sharing a block is what makes a
    same-address collision last-writer-wins rather than a race; a block per port
    is the only shape a Xilinx true dual port infers from, measured at 512x32 as
    one BRAM36 against 43,361 LUTs.

    Replaces CIRCT's `lower-seq-hlmem`, which puts every write in one block and
    so cannot describe the second shape.
  }];
  let dependentDialects = [
    "circt::hw::HWDialect", "circt::sv::SVDialect", "circt::seq::SeqDialect"
  ];
}

def LowerDataflowPass : Pass<"lower-dataflow", "ModuleOp"> {
  let summary = "Lower local scalar streams to the host dataflow simulator ABI";
  let description = [{
    Lower Allo local scalar stream operations to calls into the host dataflow
    simulator runtime and tag stream-connected kernels as dataflow PEs, which
    the late `allo-dataflow-spawn` pass launches as concurrent fibers on the
    marl runtime.
  }];
}

#endif // ALLO_CONVERSION_PASSES_TD
