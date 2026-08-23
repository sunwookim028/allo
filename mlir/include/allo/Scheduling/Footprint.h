/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_SCHEDULING_FOOTPRINT_H
#define ALLO_SCHEDULING_FOOTPRINT_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::func {
class CallOp;
} // namespace mlir::func

namespace mlir::allo {

/// Per-root access summary over a subtree.
struct Access {
  bool reads = false;
  bool writes = false;
  bool nonAffine =
      false; // >= 1 non-affine access (defeats sub-range refinement)
  llvm::SmallVector<Operation *> affine; // affine load/store ops
};

/// Memory + stream footprint of a subtree.
struct Summary {
  llvm::DenseMap<Value, Access> mem; // memref root -> access
  llvm::DenseSet<Value> streams;     // stream roots touched (get or put)
};

/// Fold one op's memory / stream effect into \p s.
void summarizeOp(Operation *op, Summary &s);

/// The ordering-hazard kind between an earlier access `a` and a later access
/// `b` on a shared memref root (program order a -> b).
enum class Conflict { None, RAW, WAR, WAW };

/// Fold a synchronous sub-kernel call's footprint into \p s, keyed by the
/// caller's operand roots: per parameter, the access direction plus the
/// callee's own affine access ops, recursing through nested calls.
///
/// Returns false when a construct defeats the summary (an unresolvable or
/// external callee, a call cycle, a view operand offset from its root's index
/// space). \p s may then hold a partial record, subsumed by the conservative
/// marks `summarizeOp` applies instead.
bool summarizeCall(func::CallOp call, Summary &s);

/// The ordering hazard between accesses recorded by `summarizeCall`. Their
/// affine ops live in the callees, each naming its own parameter rather than
/// one common memref Value, so disjointness compares polyhedral regions over
/// the index space the parameters share with the array.
Conflict callFootprintConflict(const Access &a, const Access &b);

} // namespace mlir::allo

#endif // ALLO_SCHEDULING_FOOTPRINT_H
