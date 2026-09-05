/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Scheduling/MemoryAccess.h" // asMemAccess
#include "allo/Scheduling/MemoryModel.h"  // bankLayoutOf, staticBankOf
#include "allo/Support/Logging.h"
#include "allo/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h" // getConstantIntValue
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/MapVector.h"

namespace mlir::allo {
#define GEN_PASS_DEF_ASSIGNBANKSPASS
#include "allo/Transforms/Passes.h.inc"
} // namespace mlir::allo

using namespace mlir;
using namespace mlir::allo;
using namespace mlir::allo::logging;

namespace {

// The operand numbering one memref's accesses share, with what is known about
// the values each slot takes. Two accesses that name the same value get the
// same slot, which is what makes their bank forms comparable.
struct OperandSpace {
  llvm::MapVector<Value, unsigned> slot;
  llvm::SmallVector<DimRange> range; // parallel to `slot`, indexed by it

  AffineExpr dim(Value v, DimRange r) {
    auto [it, isNew] = slot.try_emplace(v, slot.size());
    if (isNew)
      range.push_back(r);
    return getAffineDimExpr(it->second, v.getContext());
  }
};

// The counted loop \p iv is the induction variable of, as its lower bound and
// step plus an upper bound when that is constant too. The step is what makes
// the value space `lb + step*d` rather than `d`.
struct Counter {
  int64_t lb, step;
  std::optional<int64_t> ub;
};

std::optional<Counter> counterOf(BlockArgument iv) {
  Operation *parent = iv.getOwner()->getParentOp();
  if (auto loop = dyn_cast<affine::AffineForOp>(parent)) {
    if (iv != loop.getInductionVar() || !loop.hasConstantLowerBound())
      return std::nullopt;
    std::optional<int64_t> ub;
    if (loop.hasConstantUpperBound())
      ub = loop.getConstantUpperBound();
    return Counter{loop.getConstantLowerBound(), loop.getStepAsInt(), ub};
  }
  if (auto loop = dyn_cast<scf::ForOp>(parent)) {
    std::optional<int64_t> lb = getConstantIntValue(loop.getLowerBound()),
                           step = getConstantIntValue(loop.getStep());
    if (iv == loop.getInductionVar() && lb && step)
      return Counter{*lb, *step, getConstantIntValue(loop.getUpperBound())};
  }
  return std::nullopt;
}

// The values \p v takes, as an expression over the operand numbering \p space:
// a constant, a counted loop's induction variable read as `lb + step*d`, or an
// index built from those by constant adds and multiplies. Anything else takes
// a slot of its own.
AffineExpr indexExpr(Value v, OperandSpace &space) {
  if (std::optional<int64_t> c = getConstantIntValue(v))
    return getAffineConstantExpr(*c, v.getContext());
  if (auto iv = dyn_cast<BlockArgument>(v)) {
    std::optional<Counter> ctr = counterOf(iv);
    if (!ctr)
      return space.dim(v, {});
    // The dim counts ITERATIONS, so a constant trip count is what bounds it,
    // and a block digit needs that bound.
    DimRange r;
    if (ctr->ub && ctr->step > 0 && *ctr->ub > ctr->lb)
      r = {0, llvm::divideCeilSigned(*ctr->ub - ctr->lb, ctr->step) - 1, true};
    return space.dim(v, r) * ctr->step + ctr->lb;
  }
  // Not a block argument, so it is an OpResult and has a defining op.
  Operation *def = v.getDefiningOp();
  if (auto add = dyn_cast<arith::AddIOp>(def))
    return indexExpr(add.getLhs(), space) + indexExpr(add.getRhs(), space);
  if (auto sub = dyn_cast<arith::SubIOp>(def))
    return indexExpr(sub.getLhs(), space) - indexExpr(sub.getRhs(), space);
  // A non-constant coefficient is not an affine multiply, so it stays opaque.
  if (auto mul = dyn_cast<arith::MulIOp>(def)) {
    if (std::optional<int64_t> c = getConstantIntValue(mul.getRhs()))
      return indexExpr(mul.getLhs(), space) * *c;
    if (std::optional<int64_t> c = getConstantIntValue(mul.getLhs()))
      return indexExpr(mul.getRhs(), space) * *c;
  }
  return space.dim(v, {});
}

// \p a's map over the values its subscripts actually TAKE, rather than over the
// operands the map names: an induction variable of constant lower bound `lb`
// and step `s` runs over `lb + s*d`, not over `d`. The step is the fact the map
// alone cannot carry, and `s.unroll` leaves one behind: a cyclic bank digit
// then reads `d mod 4`, which says nothing, where the value space says
// `(4*d) mod 4`, which is 0.
AffineMap inIterationSpace(const MemAccess &a, OperandSpace &space) {
  SmallVector<AffineExpr> dims;
  for (unsigned p = 0, e = a.map.getNumDims(); p < e; ++p)
    dims.push_back(indexExpr(a.indices[p], space));
  // Symbols keep their slots: an affine symbol is loop-invariant by
  // construction, so no induction variable is ever one.
  return a.map.replaceDimsAndSymbols(dims, /*symReplacements=*/{},
                                     std::max<unsigned>(space.slot.size(), 1),
                                     a.map.getNumSymbols());
}

// One array's accesses, gathered before any is assigned. A skew's slots are
// only billable if every access agrees on the class, so the pass collects
// first and decides second for both kinds.
struct Info {
  BankLayout layout;
  Operation *anchor = nullptr;
  OperandSpace space; // operand numbering shared by all, with their ranges
  /// The accesses that can CONTEND, grouped. Two accesses in different blocks
  /// are in different scheduling regions, so they never take a port in the same
  /// cycle and the port model never bills them against each other. A skew's
  /// class agreement therefore has to hold within a block, not across the whole
  /// array.
  llvm::MapVector<Block *, llvm::SmallVector<std::pair<Operation *, AffineMap>>>
      byBlock;
  unsigned accesses = 0;
  unsigned assigned = 0;
};

struct AssignBanksPass
    : public allo::impl::AssignBanksPassBase<AssignBanksPass> {
  void runOnOperation() override {
    llvm::MapVector<Value, Info> byMemref;

    getOperation().walk([&](Operation *op) {
      std::optional<MemAccess> a = asMemAccess(op);
      if (!a || a->kind != AccessKind::Array)
        return;
      BankLayout layout = bankLayoutOf(a->root);
      // One bank, or a complete partition scattered into registers: there is
      // nothing to choose, and every consumer already reads bank 0.
      if (layout.numBanks == 1)
        return;
      Info &in = byMemref[a->root];
      in.layout = layout;
      // A function argument has no defining op; its `allo.part` lives on the
      // function, so that is the anchor.
      if (!in.anchor)
        in.anchor = a->root.getDefiningOp() ? a->root.getDefiningOp()
                                            : getOperation().getOperation();
      ++in.accesses;
      // Every array access carries a map, the identity one when its subscript
      // is not affine, so every access is asked.
      in.byBlock[op->getBlock()].emplace_back(op,
                                              inIterationSpace(*a, in.space));
    });

    for (auto &[memref, in] : byMemref) {
      auto shape = cast<MemRefType>(memref.getType()).getShape();
      if (in.layout.skew())
        // An ARGUMENT gets no slot: its ports are boundary interfaces, one set
        // per access, so there is none to share.
        assignSlots(in, shape, !isa<BlockArgument>(memref));
      else
        for (auto &[block, accs] : in.byBlock)
          for (auto &[op, map] : accs)
            if (std::optional<int64_t> bank =
                    staticBankOf(in.layout, map, shape, in.space.range))
              record(op, *bank, in);
      report(in);
    }
  }

  void record(Operation *op, int64_t index, Info &in) {
    op->setAttr(kBankAttr,
                IntegerAttr::get(IntegerType::get(&getContext(), 64), index));
    ++in.assigned;
  }

  // A skewed array's accesses are billable by SLOT, but only ALL OF THEM: the
  // emitter shares one port per bank between the accesses of a lane, so one
  // access the analysis cannot place returns the whole array to the
  // conservative billing. The port model reads this before the solve.
  void assignSlots(Info &in, ArrayRef<int64_t> shape, bool internal) {
    if (!internal || !in.accesses)
      return;
    llvm::SmallVector<std::pair<Operation *, unsigned>> placed;
    for (auto &[block, accs] : in.byBlock) {
      AffineExpr cls;
      for (auto &[op, map] : accs) {
        std::optional<SkewSlot> s = skewSlotOf(in.layout, map, shape);
        // Within a block the accesses CAN contend, so their banks are only
        // known apart when one class covers them all.
        if (!s || (cls && s->cls != cls))
          return;
        cls = s->cls;
        placed.emplace_back(op, s->slot);
      }
    }
    for (auto &[op, slot] : placed)
      record(op, slot, in);
  }

  // What would make a bank resolve, one sentence per kind: a cyclic digit is a
  // residue and wants the subscript constant modulo the factor, a block digit
  // is a quotient and wants the subscript's whole range inside one chunk.
  static std::string resolutionAdvice(const BankLayout &layout) {
    bool cyclic = false, block = false;
    for (const BankLayout::Axis &a : layout.axes)
      (a.kind == BankLayout::Kind::Block ? block : cyclic) = true;
    std::string s;
    if (cyclic)
      s += "A cyclic bank resolves when the partitioned subscript is constant "
           "modulo the factor, as A[2*i] is under cyclic-2. ";
    if (block)
      s += "A block bank resolves when the subscript stays inside one chunk "
           "for the whole loop, as A[i] over range(8) does under block-2 of a "
           "16-element array. ";
    return s + "An access whose subscript is neither reaches every bank.";
  }

  // An unresolved access takes a port on every bank and the emitter builds a
  // crossbar, so a partition that resolves nothing costs N memories at the
  // bandwidth of one. The one outcome a user has to act on, hence a warning;
  // the fact itself is `Memory.partition_resolved` in the microarch report.
  void report(const Info &in) {
    if (in.assigned)
      return;
    unsigned banks = in.layout.numBanks;
    if (in.layout.skew()) {
      warn(Stage::Prep, in.anchor)
          << "Skew partition into " << banks
          << " banks resolves no slot, so each of the " << in.accesses
          << " accesses takes a port on every bank. A skew pays only when "
             "every "
             "access to the array shares one bank expression up to a constant, "
             "as A[i,j] and A[j,i] do and A[i,j] and A[i,2*j] do not";
      return;
    }
    warn(Stage::Prep, in.anchor)
        << "Partition into " << banks << " banks resolves none of the "
        << in.accesses
        << " accesses to one bank, so each takes a port on every bank and "
           "the emitter builds a crossbar: the partition costs area without "
           "adding bandwidth. "
        << resolutionAdvice(in.layout);
  }
};

} // namespace
