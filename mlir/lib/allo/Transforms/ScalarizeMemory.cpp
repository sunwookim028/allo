/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo-c/Schedule.h"            // kPartitionAttr, kBindStorageAttr
#include "allo/Support/AliasAnalysis.h" // alloAliasAnalysis
#include "allo/Support/Logging.h"
#include "allo/Transforms/Passes.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"

namespace mlir::allo {
#define GEN_PASS_DEF_SCALARIZEMEMORYPASS
#include "allo/Transforms/Passes.h.inc"
} // namespace mlir::allo

using namespace mlir;
using namespace mlir::allo;
using namespace mlir::allo::logging;

namespace {

struct ScalarizeMemoryPass
    : public allo::impl::ScalarizeMemoryPassBase<ScalarizeMemoryPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    AliasAnalysis aa = alloAliasAnalysis(func);
    affine::affineScalarReplace(func, getAnalysis<DominanceInfo>(),
                                getAnalysis<PostDominanceInfo>(), aa);

    // Forwarding above always runs; max-elements 0 only disables
    // auto-partition.
    if (maxElements == 0)
      return;

    // The arrays that SURVIVED the forwarding above. One whose every read
    // forwarded is gone entirely, and needs no storage decision at all.
    SmallVector<Operation *> survived;
    func.walk([&](Operation *op) {
      if (isa<memref::AllocOp, memref::AllocaOp>(op))
        survived.push_back(op);
    });
    MLIRContext *ctx = &getContext();
    auto complete = PartitionAttr::get(
        ctx, {PartitionAxisAttr::get(ctx, PartitionKindEnum::CompletePartition,
                                     /*factor=*/0,
                                     /*dim=*/0)});

    // Whether every subscript of \p user is a literal, which makes its access
    // wiring rather than a mux.
    auto constantSubscripts = [](Operation *user) {
      AffineMap map;
      if (auto load = dyn_cast<affine::AffineLoadOp>(user))
        map = load.getAffineMap();
      else if (auto store = dyn_cast<affine::AffineStoreOp>(user))
        map = store.getAffineMap();
      else {
        auto indices = isa<memref::LoadOp>(user)
                           ? cast<memref::LoadOp>(user).getIndices()
                           : cast<memref::StoreOp>(user).getIndices();
        return llvm::all_of(indices, [](Value index) {
          return matchPattern(index, m_Constant());
        });
      }
      return llvm::all_of(map.getResults(), llvm::IsaPred<AffineConstantExpr>);
    };

    for (Operation *op : survived) {
      // An explicit storage choice always wins over an automatic one.
      if (op->hasAttr(kPartitionAttr) || op->hasAttr(kBindStorageAttr))
        continue;
      auto type = cast<MemRefType>(op->getResult(0).getType());
      if (!type.hasStaticShape())
        continue;
      // Every use a direct access, so the array's whole traffic is in view.
      // This is what excludes handing it to a sub-kernel, which masters ports
      // on storage the caller owns and has never been tried against a Complete
      // partition, and threading it out of a region as a result.
      if (llvm::any_of(op->getUsers(), [](Operation *user) {
            return !isa<affine::AffineLoadOp, affine::AffineStoreOp,
                        memref::LoadOp, memref::StoreOp>(user);
          }))
        continue;
      bool allConst = true;
      DenseMap<Block *, unsigned> perBlock;
      unsigned most = 0;
      for (Operation *user : op->getUsers()) {
        allConst &= constantSubscripts(user);
        most = std::max(most, ++perBlock[user->getBlock()]);
      }
      // Every subscript constant means the array fully dissolves into wired
      // registers with no mux, so scalarize it at any size. A variable
      // subscript pays a read mux and a write demux, worth it only while the
      // array is small enough to bound the mux and one block issues more
      // accesses than a dual-ported row serves in a cycle.
      if (!allConst && (type.getNumElements() > maxElements || most <= 2))
        continue;

      op->setAttr(kPartitionAttr, complete);
    }
  }
};

} // namespace
