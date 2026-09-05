/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_TRANSFORM_OPS_UTILS_H
#define ALLO_TRANSFORM_OPS_UTILS_H

#include "allo/IR/AlloOps.h"
#include "allo/Support/AffineValueMapBuilder.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"

#include <optional>

namespace mlir::allo {
bool affineExprUsesValue(AffineExpr expr, ValueRange mapOperands,
                         unsigned numDims, Value needle);
int findMemRefAxisFromIVs(affine::AffineStoreOp storeOp, Value iv);
Value resolveMemRefValueRoot(Value value);
// Resolve `value` to the operation (and, for a function argument, its index)
// that should carry a per-buffer HLS attribute (array_partition /
// bind_storage). Follows view-like aliases to the root buffer, then dispatches
// to its defining function argument, memref.alloc/alloca, or memref.global.
// Returns failure if `value` is not a memref or its root is none of these.
LogicalResult resolveBufferAttrCarrier(Value value, Operation *&owner,
                                       std::optional<unsigned> &argNumber);
bool isMemRefCastOrViewLike(Operation *op);
} // namespace mlir::allo

#endif // ALLO_TRANSFORM_OPS_UTILS_H
