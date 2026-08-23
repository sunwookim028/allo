/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo-c/Schedule.h" // kPartitionAttr
#include "allo/IR/AlloAttrs.h"
#include "allo/Transforms/Passes.h"

#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

namespace mlir::allo {
#define GEN_PASS_DEF_HOISTINVARIANTREADSPASS
#include "allo/Transforms/Passes.h.inc"
} // namespace mlir::allo

using namespace mlir;
using namespace mlir::allo;

namespace {

Value accessedMemref(Operation *op) {
  if (auto load = dyn_cast<affine::AffineLoadOp>(op))
    return load.getMemRef();
  if (auto store = dyn_cast<affine::AffineStoreOp>(op))
    return store.getMemRef();
  if (auto load = dyn_cast<memref::LoadOp>(op))
    return load.getMemRef();
  if (auto store = dyn_cast<memref::StoreOp>(op))
    return store.getMemRef();
  return {};
}

// Whether every write to \p memref is visible as a user of this SSA value:
// a kernel argument or a local allocation under Allo's aliasing contract, or a
// constant global. A view or a rewritable global can hide a writer.
bool writersAreVisible(Value memref) {
  if (auto arg = dyn_cast<BlockArgument>(memref))
    return isa<func::FuncOp>(arg.getOwner()->getParentOp());
  Operation *def = memref.getDefiningOp();
  if (auto get = dyn_cast<memref::GetGlobalOp>(def)) {
    auto global = SymbolTable::lookupNearestSymbolFrom<memref::GlobalOp>(
        get, get.getNameAttr());
    assert(global && "get_global references an undefined memref.global");
    return global.getConstant();
  }
  return isa<memref::AllocOp, memref::AllocaOp>(def);
}

PartitionAttr partitionOf(Value memref) {
  if (auto arg = dyn_cast<BlockArgument>(memref)) {
    auto func = cast<func::FuncOp>(arg.getOwner()->getParentOp());
    return func.getArgAttrOfType<PartitionAttr>(arg.getArgNumber(),
                                                kPartitionAttr);
  }
  if (auto get = memref.getDefiningOp<memref::GetGlobalOp>()) {
    auto global = SymbolTable::lookupNearestSymbolFrom<memref::GlobalOp>(
        get, get.getNameAttr());
    return global->getAttrOfType<PartitionAttr>(kPartitionAttr);
  }
  return memref.getDefiningOp()->getAttrOfType<PartitionAttr>(kPartitionAttr);
}

// A complete-partitioned array reads as wires, so hoisting its reads would
// only add registers carrying the values across the region boundary.
bool scattersToRegisters(Value memref) {
  PartitionAttr part = partitionOf(memref);
  return part && llvm::any_of(part.getPartitions(), [](PartitionAxisAttr a) {
           return a.getKind() == PartitionKindEnum::CompletePartition;
         });
}

// Banks the array's partition splits it into. Ports are per bank, so access
// pressure scales inversely with this count.
unsigned bankCount(Value memref) {
  PartitionAttr part = partitionOf(memref);
  if (!part)
    return 1;
  unsigned banks = 1;
  int64_t rank = cast<MemRefType>(memref.getType()).getRank();
  for (PartitionAxisAttr axis : part.getPartitions()) {
    assert(axis.getKind() != PartitionKindEnum::CompletePartition &&
           "scattered arrays carry no bank pressure");
    // `dim == 0` banks every dimension by this factor.
    for (int64_t d = axis.getDim() == 0 ? rank : 1; d > 0; --d)
      banks *= static_cast<unsigned>(axis.getFactor());
  }
  return banks;
}

// Whether removing \p hoistable lowers the body's port-pressure floor, the
// widest array's accesses per bank per iteration. The modulo scheduler cannot
// beat that floor, so a preload leaving it in place buys no II and only spends
// registers holding the values across the region boundary.
bool lowersPortFloor(affine::AffineForOp loop,
                     ArrayRef<Operation *> hoistable) {
  DenseSet<Operation *> moved(hoistable.begin(), hoistable.end());
  DenseMap<Value, std::pair<unsigned, unsigned>> traffic; // total, hoistable
  loop.getBody()->walk([&](Operation *op) {
    Value memref = accessedMemref(op);
    if (!memref)
      return;
    auto &counts = traffic[memref];
    ++counts.first;
    if (moved.contains(op))
      ++counts.second;
  });
  double before = 0, after = 0;
  for (auto &[memref, counts] : traffic) {
    if (scattersToRegisters(memref))
      continue;
    double banks = bankCount(memref);
    before = std::max(before, counts.first / banks);
    after = std::max(after, (counts.first - counts.second) / banks);
  }
  return after < before;
}

// The reads directly in \p loop's body whose operands are all loop-invariant
// and whose array is safe to preload. Only reads at this level move, one loop
// level at a time, which keeps their order against writes between nested loops.
SmallVector<Operation *> invariantReads(affine::AffineForOp loop) {
  // One verdict per array: an unrolled body reads the same small array many
  // times, and the array's own properties do not change between those reads.
  DenseMap<Value, bool> preloadable;
  auto arrayAllows = [&](Value memref) {
    auto [it, fresh] = preloadable.try_emplace(memref, false);
    if (fresh)
      // Any write inside the loop, or any user this pass cannot read (a call
      // mastering the array's ports), pins every read of the array.
      it->second = writersAreVisible(memref) && !scattersToRegisters(memref) &&
                   !llvm::any_of(memref.getUsers(), [&](Operation *user) {
                     return loop->isProperAncestor(user) &&
                            !isa<affine::AffineLoadOp, memref::LoadOp>(user);
                   });
    return it->second;
  };
  SmallVector<Operation *> reads;
  for (Operation &op : *loop.getBody()) {
    if (!isa<affine::AffineLoadOp, memref::LoadOp>(op))
      continue;
    if (!llvm::all_of(op.getOperands(),
                      [&](Value v) { return loop.isDefinedOutsideOfLoop(v); }))
      continue;
    if (!arrayAllows(accessedMemref(&op)))
      continue;
    reads.push_back(&op);
  }
  return reads;
}

void hoistFrom(affine::AffineForOp loop) {
  // A hoisted read runs unconditionally, so the loop needs a known non-zero
  // trip count.
  std::optional<uint64_t> trip = affine::getConstantTripCount(loop);
  if (!trip || *trip == 0)
    return;
  SmallVector<Operation *> reads = invariantReads(loop);
  if (reads.empty())
    return;
  // A leaf body is what the modulo scheduler paces, so a preload there must
  // pay for its held registers with II. A body with loops inside is paced by
  // its children, where hoisting only stops the re-read on each sweep.
  bool leaf = loop.getBody()->getOps<LoopLikeOpInterface>().empty();
  if (leaf && !lowersPortFloor(loop, reads))
    return;
  SmallVector<Operation *> hoisted;
  for (Operation *read : reads) {
    auto same = [&](Operation *h) {
      return OperationEquivalence::isEquivalentTo(
          read, h, OperationEquivalence::IgnoreLocations);
    };
    if (auto *it = llvm::find_if(hoisted, same); it != hoisted.end()) {
      read->replaceAllUsesWith(*it);
      read->erase();
    } else {
      read->moveBefore(loop);
      hoisted.push_back(read);
    }
  }
}

struct HoistInvariantReadsPass
    : public allo::impl::HoistInvariantReadsPassBase<HoistInvariantReadsPass> {
  using Base::Base;

  void runOnOperation() override {
    // Innermost first: a read hoisted into an outer body is reconsidered when
    // that loop's own turn comes.
    getOperation().walk([](affine::AffineForOp loop) { hoistFrom(loop); });
  }
};

} // namespace
