/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Scheduling/MemoryAccess.h"

#include "allo/IR/AlloOps.h"            // StreamGetOp / StreamPutOp
#include "allo/Support/AliasAnalysis.h" // resolveRoot (storage identity)

#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/STLExtras.h"

#include <cassert>

using namespace mlir;
using namespace mlir::allo;

// The base stream SSA value a stream get/put operates on.
static Value streamBaseOf(Operation *op) {
  if (auto get = dyn_cast<StreamGetOp>(op))
    return get.getStream();
  return cast<StreamPutOp>(op).getStream();
}

// Normalizes affine and non-affine accesses to one shape: a root value, an
// affine map, and index operands. A non-affine op gets the multi-dim identity
// map, so downstream pricing and bank derivation see one encoding.
std::optional<MemAccess> mlir::allo::asMemAccess(Operation *op) {
  MemAccess a;
  a.op = op;
  if (auto read = dyn_cast<affine::AffineReadOpInterface>(op)) {
    a.root = resolveRoot(read.getMemRef());
    a.map = read.getAffineMap();
    llvm::append_range(a.indices, read.getMapOperands());
    return a;
  }
  if (auto write = dyn_cast<affine::AffineWriteOpInterface>(op)) {
    a.isWrite = true;
    a.root = resolveRoot(write.getMemRef());
    a.map = write.getAffineMap();
    llvm::append_range(a.indices, write.getMapOperands());
    return a;
  }
  if (auto load = dyn_cast<memref::LoadOp>(op)) {
    a.root = resolveRoot(load.getMemRef());
    llvm::append_range(a.indices, load.getIndices());
    a.map =
        AffineMap::getMultiDimIdentityMap(a.indices.size(), op->getContext());
    return a;
  }
  if (auto store = dyn_cast<memref::StoreOp>(op)) {
    a.isWrite = true;
    a.root = resolveRoot(store.getMemRef());
    llvm::append_range(a.indices, store.getIndices());
    a.map =
        AffineMap::getMultiDimIdentityMap(a.indices.size(), op->getContext());
    return a;
  }
  if (isa<StreamGetOp, StreamPutOp>(op)) {
    a.kind = AccessKind::Stream;
    a.isWrite = isa<StreamPutOp>(op);
    a.root = resolveRoot(streamBaseOf(op));
    if (auto get = dyn_cast<StreamGetOp>(op))
      llvm::append_range(a.indices, get.getIndices());
    else
      llvm::append_range(a.indices, cast<StreamPutOp>(op).getIndices());
    return a;
  }
  return std::nullopt;
}
