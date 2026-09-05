/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo-c/Schedule.h" // kPartitionAttr
#include "allo/IR/AlloOps.h"
#include "allo/Scheduling/MemoryModel.h" // assignedBankOf, bankSplitOf
#include "allo/Support/Logging.h"
#include "allo/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::allo {
#define GEN_PASS_DEF_RESOLVEBANKINGPASS
#include "allo/Transforms/Passes.h.inc"
} // namespace mlir::allo

using namespace mlir;
using namespace mlir::allo;
using namespace mlir::allo::logging;

namespace {

// Address map of a dcp memory access.
AffineMap accessMap(Operation *op) {
  if (auto l = dyn_cast<dcp::DCPathLoadOp>(op))
    return l.getMap();
  return cast<dcp::DCPathStoreOp>(op).getMap();
}

// Route \p op to its own bank's memref at the in-bank address. The recorded
// bank is dropped with the same edit: the access now names an unbanked memref,
// and leaving it would have the emitter route bank 3 of a one-bank memory.
void rewriteAccess(Operation *op, Value bank, AffineMap localMap) {
  if (auto l = dyn_cast<dcp::DCPathLoadOp>(op)) {
    l.getMemrefMutable().assign(bank);
    l.setMapAttr(AffineMapAttr::get(localMap));
    l.removeBankAttr();
  } else {
    auto s = cast<dcp::DCPathStoreOp>(op);
    s.getMemrefMutable().assign(bank);
    s.setMapAttr(AffineMapAttr::get(localMap));
    s.removeBankAttr();
  }
}

// Split one internal partitioned alloc into per-bank allocs if every access was
// assigned a bank; return true if it was split.
bool splitAlloc(Operation *alloc) {
  Value memref = alloc->getResult(0);
  BankLayout layout = bankLayoutOf(memref);
  // A complete partition scattered the array into registers, and an unbanked
  // array has nothing to split.
  if (layout.numBanks == 1)
    return false;
  // A skewed access records a SLOT, not a bank: its physical bank is that slot
  // rotated at run time, so routing it to one alloc would reach the wrong
  // storage at every rotation but zero. The emitter selects among the banks.
  if (layout.skew())
    return false;

  auto mt = cast<MemRefType>(memref.getType());
  ArrayRef<int64_t> shape = mt.getShape();
  struct Routed {
    Operation *op;
    unsigned bank;
    AffineMap localMap;
  };
  SmallVector<Routed> routed;
  for (Operation *user : memref.getUsers()) {
    // A non-load/store use (e.g. the memref escaping) cannot be split safely.
    if (!isa<dcp::DCPathLoadOp, dcp::DCPathStoreOp>(user))
      return false;
    std::optional<unsigned> bank = assignedBankOf(user);
    if (!bank) {
      warn(Stage::Dcp, alloc) << "Partitioned array has a data-dependent bank; "
                                 "left for the emitter "
                                 "crossbar";
      return false;
    }
    // The in-bank coordinates of the very split the decision came from, so the
    // element this access lands on inside its bank is the one the crossbar
    // would have reached.
    AffineMap map = accessMap(user);
    routed.push_back({user, *bank,
                      AffineMap::get(map.getNumDims(), map.getNumSymbols(),
                                     bankSplitOf(layout, map, shape).coords,
                                     map.getContext())});
  }

  auto bankType = MemRefType::get(layout.bankShape, mt.getElementType());

  OpBuilder b(alloc);
  SmallVector<Value> bankMemrefs;
  for (unsigned k = 0; k < layout.numBanks; ++k) {
    Operation *bankAlloc =
        isa<memref::AllocaOp>(alloc)
            ? memref::AllocaOp::create(b, alloc->getLoc(), bankType)
                  .getOperation()
            : memref::AllocOp::create(b, alloc->getLoc(), bankType)
                  .getOperation();
    // Carry every attribute except the partition (a bank *is* one physical
    // memory); keeps bind.storage / the buffer NameLoc for emit naming.
    for (NamedAttribute attr : alloc->getAttrs())
      if (attr.getName() != kPartitionAttr)
        bankAlloc->setAttr(attr.getName(), attr.getValue());
    bankMemrefs.push_back(bankAlloc->getResult(0));
  }

  for (const Routed &r : routed)
    rewriteAccess(r.op, bankMemrefs[r.bank], r.localMap);
  alloc->erase();
  return true;
}

struct ResolveBankingPass
    : public allo::impl::ResolveBankingPassBase<ResolveBankingPass> {
  void runOnOperation() override {
    SmallVector<Operation *> allocs;
    getOperation()->walk([&](Operation *op) {
      if (isa<memref::AllocOp, memref::AllocaOp>(op) &&
          op->hasAttr(kPartitionAttr))
        allocs.push_back(op);
    });
    for (Operation *alloc : allocs)
      splitAlloc(alloc);
  }
};

} // namespace
