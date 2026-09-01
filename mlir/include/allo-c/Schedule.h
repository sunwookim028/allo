/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_C_SCHEDULE_H
#define ALLO_C_SCHEDULE_H

#include "allo/IR/AlloOps.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Assigns a unique `allo.schedule.id` attribute to every scheduled operation,
/// regenerating any non-unique ids.
MLIR_CAPI_EXPORTED void alloAnnotateScheduleIds(MlirModule module);

/// Removes all `allo.schedule.id` attributes from `module`.
MLIR_CAPI_EXPORTED void alloCleanupScheduleIds(MlirModule module);

/// Collects an immutable snapshot of the scheduled IR (operation hierarchy,
/// buffer-like values, traits, locations) and streams it as a JSON document
/// through `callback`. `module` must have been annotated first.
MLIR_CAPI_EXPORTED void
alloCollectScheduleSnapshotJSON(MlirModule module, MlirStringCallback callback,
                                void *userData);

MLIR_CAPI_EXPORTED constexpr const char *kScheduleIdAttr = "allo.schedule.id";
MLIR_CAPI_EXPORTED constexpr const char *kScheduleNameAttr =
    "allo.schedule.name";
MLIR_CAPI_EXPORTED constexpr const char *kPipelineIIAttr = "allo.pipeline.ii";
MLIR_CAPI_EXPORTED constexpr const char *kPipelineRewindAttr =
    "allo.pipeline.rewind";
MLIR_CAPI_EXPORTED constexpr const char *kDataflowAttr = "allo.dataflow";
MLIR_CAPI_EXPORTED constexpr const char *kUnrollFactorAttr = "allo.unroll.f";
MLIR_CAPI_EXPORTED constexpr const char *kPartitionAttr = "allo.part";
MLIR_CAPI_EXPORTED constexpr const char *kBindStorageAttr = "allo.bind.storage";
MLIR_CAPI_EXPORTED constexpr const char *kAlloSignedAttr =
    mlir::allo::kAlloSignedAttr.data();
MLIR_CAPI_EXPORTED constexpr const char *kAlloLazyAttr =
    mlir::allo::kAlloLazyAttr.data();

#ifdef __cplusplus
}
#endif

#endif // ALLO_C_SCHEDULE_H
