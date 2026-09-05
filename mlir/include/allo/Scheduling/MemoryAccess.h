/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_SCHEDULING_MEMORYACCESS_H
#define ALLO_SCHEDULING_MEMORYACCESS_H

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace mlir::allo {

/// Whether an access targets an array (memref load/store) or a stream FIFO
/// (`allo.stream` get/put).
enum class AccessKind { Array, Stream };

/// A recognized memory access. `root` is the underlying buffer/stream SSA value
/// (`resolveRoot`), so distinct roots are distinct storage. `map` is the
/// element-space subscript map, one result per memref dimension; an array
/// access always carries one, a stream none. `indices` are the subscript
/// operands (array) or FIFO-select operands (stream). Whether an access is
/// affine is a question about the op, not about the map.
struct MemAccess {
  Operation *op = nullptr;
  Value root;
  AccessKind kind = AccessKind::Array;
  bool isWrite = false;
  AffineMap map;
  llvm::SmallVector<Value, 4> indices;
};

/// Recognize \p op as a memory access (affine/memref load-store, or stream
/// get/put); nullopt if it is not one.
std::optional<MemAccess> asMemAccess(Operation *op);

} // namespace mlir::allo

#endif // ALLO_SCHEDULING_MEMORYACCESS_H
