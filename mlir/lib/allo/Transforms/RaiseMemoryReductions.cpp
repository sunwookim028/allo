/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Support/Logging.h"
#include "allo/Transforms/Passes.h"
#include "allo/Transforms/ReductionUtils.h"

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::allo {
#define GEN_PASS_DEF_RAISEMEMORYREDUCTIONSPASS
#include "allo/Transforms/Passes.h.inc"
} // namespace mlir::allo

using namespace mlir;
using namespace mlir::allo;
using namespace mlir::allo::logging;

namespace {

// One promotable memory reduction: `load` and `store` touch the same
// loop-invariant location, combined by the associative `step`.
struct Candidate {
  affine::AffineLoadOp load;
  affine::AffineStoreOp store;
};

// Whether every writer of `memref` is a user of this SSA value, so the
// dependence walk over the loop sees them all: a kernel argument or a local
// allocation under Allo's aliasing contract, or a constant global.
bool writersAreVisible(Value memref) {
  if (auto arg = dyn_cast<BlockArgument>(memref))
    return isa<func::FuncOp>(arg.getOwner()->getParentOp());
  Operation *def = memref.getDefiningOp();
  if (auto get = dyn_cast<memref::GetGlobalOp>(def)) {
    auto global = SymbolTable::lookupNearestSymbolFrom<memref::GlobalOp>(
        get, get.getNameAttr());
    return global && global.getConstant();
  }
  return isa<memref::AllocOp, memref::AllocaOp>(def);
}

// The store's value is `combine(load M[inv], leaf)`, with the load reading the
// store's own location, invariant in `loop`, and both the load result and the
// stored value used only here. Returns the load, or null.
affine::AffineLoadOp matchMemoryReduction(affine::AffineStoreOp store,
                                          affine::AffineForOp loop) {
  ReductionStep step = matchReductionStep(store.getValueToStore());
  if (!step)
    return {};
  auto [lhs, rhs] = reductionOperands(step);
  auto lLoad = lhs.getDefiningOp<affine::AffineLoadOp>();
  auto rLoad = rhs.getDefiningOp<affine::AffineLoadOp>();
  affine::MemRefAccess write(store);
  bool lIsAcc = lLoad && affine::MemRefAccess(lLoad) == write;
  bool rIsAcc = rLoad && affine::MemRefAccess(rLoad) == write;
  if (lIsAcc == rIsAcc) // needs exactly one accumulator operand
    return {};
  affine::AffineLoadOp load = lIsAcc ? lLoad : rLoad;
  if (!load.getResult().hasOneUse() || !store.getValueToStore().hasOneUse())
    return {};
  if (!affine::isInvariantAccess(load, loop) ||
      !affine::isInvariantAccess(store, loop))
    return {};
  // Unconditional in the body, load reaching the store.
  if (load->getBlock() != loop.getBody() ||
      store->getBlock() != loop.getBody() || !load->isBeforeInBlock(store))
    return {};
  return load;
}

// No access in `loop` other than `cand`'s own load/store touches its location:
// an opaque user of the array, or a polyhedral dependence with the store or the
// load, rules the location out.
bool locationIsPrivate(Candidate cand, affine::AffineForOp loop) {
  Value memref = cand.store.getMemRef();
  if (!writersAreVisible(memref))
    return false;
  affine::MemRefAccess self[] = {affine::MemRefAccess(cand.store),
                                 affine::MemRefAccess(cand.load)};
  bool clash = false;
  loop.walk([&](Operation *op) {
    if (clash || op == cand.load.getOperation() ||
        op == cand.store.getOperation())
      return;
    bool affineAccess = isa<affine::AffineLoadOp, affine::AffineStoreOp>(op) &&
                        affine::MemRefAccess(op).memref == memref;
    if (affineAccess) {
      affine::MemRefAccess other(op);
      unsigned common =
          affine::getInnermostCommonLoopDepth({op, cand.store.getOperation()});
      for (unsigned d = 1; d <= common + 1 && !clash; ++d)
        for (affine::MemRefAccess &acc : self)
          if (affine::checkMemrefAccessDependence(other, acc, d).value !=
              affine::DependenceResult::NoDependence)
            clash = true;
    } else if (llvm::is_contained(op->getOperands(), memref)) {
      clash = true; // an opaque user (call / view / non-affine access)
    }
  });
  return !clash;
}

// Rebuild `loop` carrying every candidate as an iter_arg: a preload of each
// accumulator before the loop, the combine reading the iter_arg, and one
// write-back after it.
void raise(affine::AffineForOp loop, SmallVectorImpl<Candidate> &cands) {
  OpBuilder b(loop);
  Location loc = loop.getLoc();

  SmallVector<Value> inits;
  for (Candidate c : cands)
    inits.push_back(b.clone(*c.load.getOperation())->getResult(0));

  DenseSet<Operation *> skip;
  for (Candidate c : cands) {
    skip.insert(c.load.getOperation());
    skip.insert(c.store.getOperation());
  }

  auto newLoop = affine::AffineForOp::create(
      b, loc, loop.getLowerBoundOperands(), loop.getLowerBoundMap(),
      loop.getUpperBoundOperands(), loop.getUpperBoundMap(),
      loop.getStepAsInt(), inits,
      [&](OpBuilder &nb, Location nloc, Value niv, ValueRange accs) {
        IRMapping map;
        map.map(loop.getInductionVar(), niv);
        for (unsigned i = 0; i < cands.size(); ++i)
          map.map(cands[i].load.getResult(), accs[i]);
        SmallVector<Value> yields(cands.size());
        for (Operation &o : loop.getBody()->without_terminator()) {
          if (!skip.contains(&o)) {
            nb.clone(o, map);
            continue;
          }
          for (unsigned i = 0; i < cands.size(); ++i)
            if (cands[i].store.getOperation() == &o)
              yields[i] = map.lookup(cands[i].store.getValueToStore());
        }
        affine::AffineYieldOp::create(nb, nloc, yields);
      });

  b.setInsertionPointAfter(newLoop);
  for (unsigned i = 0; i < cands.size(); ++i) {
    IRMapping wb;
    Candidate c = cands[i];
    wb.map(c.store.getValueToStore(), newLoop.getResult(i));
    b.clone(*c.store.getOperation(), wb);
  }
  info(Stage::Prep, newLoop)
      << "Raising " << cands.size() << " memory reduction(s) to iter_args";
  loop.erase();
}

SmallVector<Candidate> promotable(affine::AffineForOp loop) {
  SmallVector<Candidate> cands;
  for (Operation &op : loop.getBody()->without_terminator())
    if (auto store = dyn_cast<affine::AffineStoreOp>(&op))
      if (affine::AffineLoadOp load = matchMemoryReduction(store, loop))
        cands.push_back({load, store});
  llvm::erase_if(cands,
                 [&](Candidate c) { return !locationIsPrivate(c, loop); });
  return cands;
}

struct RaiseMemoryReductionsPass
    : public allo::impl::RaiseMemoryReductionsPassBase<
          RaiseMemoryReductionsPass> {
  using RaiseMemoryReductionsPassBase::RaiseMemoryReductionsPassBase;

  void runOnOperation() override {
    // One loop per sweep: raising rebuilds the loop, so re-walk the fresh IR.
    // A raised inner loop is left with no memory reduction to find again.
    bool changed = true;
    while (changed) {
      changed = false;
      getOperation().walk([&](affine::AffineForOp loop) {
        SmallVector<Candidate> cands = promotable(loop);
        if (cands.empty())
          return WalkResult::advance();
        raise(loop, cands);
        changed = true;
        return WalkResult::interrupt();
      });
    }
  }
};

} // namespace
