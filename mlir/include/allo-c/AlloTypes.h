/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * C API for the Allo dialect's custom types, used by the Python bindings.
 */

#ifndef ALLO_C_ALLOTYPES_H
#define ALLO_C_ALLOTYPES_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// StreamType  (!allo.stream<baseType, depth, [shape...]>)
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool alloTypeIsAStream(MlirType type);

MLIR_CAPI_EXPORTED MlirType alloStreamTypeGet(MlirContext ctx,
                                              MlirType baseType, uint64_t depth,
                                              intptr_t rank,
                                              const int64_t *shape);

MLIR_CAPI_EXPORTED MlirType alloStreamTypeGetBaseType(MlirType type);
MLIR_CAPI_EXPORTED uint64_t alloStreamTypeGetDepth(MlirType type);
MLIR_CAPI_EXPORTED intptr_t alloStreamTypeGetRank(MlirType type);
MLIR_CAPI_EXPORTED int64_t alloStreamTypeGetDimSize(MlirType type,
                                                    intptr_t pos);
MLIR_CAPI_EXPORTED MlirTypeID alloStreamTypeGetTypeID(void);

#ifdef __cplusplus
}
#endif

#endif // ALLO_C_ALLOTYPES_H
