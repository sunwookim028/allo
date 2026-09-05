/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Block and module surgery the upstream MLIR Python bindings do not expose.
 */

#ifndef ALLO_C_IRUTILS_H
#define ALLO_C_IRUTILS_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Erases `block` from its parent region.
MLIR_CAPI_EXPORTED void alloBlockErase(MlirBlock block);

/// Splices all operations of `src` into `dst` (before `dst`'s terminator) and
/// erases `src`. `src` must have no predecessors and no arguments.
MLIR_CAPI_EXPORTED void alloBlockMergeBefore(MlirBlock src, MlirBlock dst);

/// Clones the given module and returns the clone.
MLIR_CAPI_EXPORTED MlirModule alloCloneModuleOp(MlirModule module);

#ifdef __cplusplus
}
#endif

#endif // ALLO_C_IRUTILS_H
