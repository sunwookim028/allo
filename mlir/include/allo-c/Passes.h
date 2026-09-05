/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_C_PASSES_H
#define ALLO_C_PASSES_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Emits Vivado HLS C++ for `module`, streaming the result through `callback`.
/// `top` names the top function, which is emitted with `extern "C"` linkage and
/// carries the global array_partition pragmas; pass an empty string for none.
MLIR_CAPI_EXPORTED MlirLogicalResult
alloEmitVivadoHLS(MlirModule module, bool enableApFloat, unsigned indexWidth,
                  bool withLocation, MlirStringRef top,
                  MlirStringCallback callback, void *userData);

MLIR_CAPI_EXPORTED MlirLogicalResult
alloDumpRegionDependenceAnalysis(MlirModule module, MlirStringRef funcName,
                                 MlirStringCallback callback, void *userData);

MLIR_CAPI_EXPORTED MlirLogicalResult
alloEmitVerilog(MlirModule module, MlirStringCallback callback, void *userData);

/// Lowers every scheduled function in `module` to structural `hw.module`s in
/// place, and streams back through `callback` the cosim manifest: one JSON
/// object mapping each emitted module's name to its port-interface JSON.
/// `binding` names the resource-binding policy, `cycleTime` the resolved target
/// period in ns. Returns failure (callback not invoked) if emission fails.
MLIR_CAPI_EXPORTED MlirLogicalResult alloEmitDatapathToHW(
    MlirModule module, MlirStringRef binding, MlirStringRef top,
    double cycleTime, MlirStringCallback callback, void *userData);

MLIR_CAPI_EXPORTED MlirLogicalResult
alloEmitSplitVerilog(MlirModule module, MlirStringRef directory);

/// Schedules `top` and reifies the schedule into `module` in place as
/// `allo.dcp.*` ops, streaming back through `callback` the schedule report as
/// JSON: per-func regions with their per-op start times, plus the per-region
/// and whole-kernel latency. `scheduler` is "heuristic" or "exact" (CP-SAT).
/// `budget` is what one exact solve may spend, in deterministic time units;
/// zero or less takes the default. `allocate` lets an exact solve decide how
/// many copies of each operator a region builds. `workers` (zero or less takes
/// the default) and `seed` tune the CP-SAT search; changing either makes a
/// compile non-reproducible. `deterministic` off lets the workers race instead
/// of interleaving, each under `budget / workers` wall seconds, so no exact
/// solve is then reproducible. `areaSlack` is the span the area minimization
/// may pay beyond the minimal span, as a fraction of it, trading cycles for a
/// smaller design. `escalate` lets the heuristic scheduler hand a region to the
/// exact solver when its schedule is provably off its floor. Returns failure
/// (callback not invoked) on any failed phase.
MLIR_CAPI_EXPORTED MlirLogicalResult alloRunSDCSchedulingPipeline(
    MlirModule module, MlirStringRef top, float cycleTime,
    MlirStringRef scheduler, double budget, bool allocate, int workers,
    int seed, bool deterministic, double areaSlack, bool escalate,
    MlirStringCallback callback, void *userData);

#ifdef __cplusplus
}
#endif

#endif // ALLO_C_PASSES_H
