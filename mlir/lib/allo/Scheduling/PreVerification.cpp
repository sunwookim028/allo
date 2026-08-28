/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/IR/AlloOps.h"
#include "allo/Scheduling/AddressModel.h"       // addressExprsOf, addressCostOf
#include "allo/Scheduling/DependenceAnalysis.h" // isUnmodeledMemoryAccess
#include "allo/Scheduling/MemoryAccess.h"       // asMemAccess
#include "allo/Scheduling/MemoryModel.h"
#include "allo/Scheduling/OperatorLibrary.h"
#include "allo/Scheduling/ProblemBuilder.h" // whileFlushingPipelines
#include "allo/Scheduling/RegionGraph.h"
#include "allo/Support/Logging.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h" // getConstantIntValue
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::allo;
using namespace mlir::allo::logging;

// An op the reifier turns into a `dcp.compute`: a single-result non-constant op
// with no region that `convertOp` (PostConversion.cpp) does not carry on a path
// of its own. An exclusion list rather than a dialect allow-list, matching the
// reifier's own split, so an op it cannot realize is refused here on the
// offending line instead of asserting there.
static bool isComputeOp(Operation *op) {
  return op->getNumResults() == 1 && op->getNumRegions() == 0 &&
         !op->hasTrait<OpTrait::ConstantLike>() &&
         !isa<affine::AffineLoadOp, memref::LoadOp, StreamGetOp, StreamPutOp,
              StreamCreateOp, memref::AllocOp, memref::AllocaOp,
              memref::GetGlobalOp, func::CallOp>(op);
}

// Whether the operator library can be asked about \p op: it prices integer and
// float arithmetic and affine address expressions. Anything else is a backend
// gap rather than a device one, since no `@ip` row would make the datapath
// model it.
static bool isPricedOp(Operation *op) {
  return isa<arith::ArithDialect, math::MathDialect>(op->getDialect()) ||
         isa<affine::AffineApplyOp, MulAddOp>(op);
}

namespace {
// One end of a channel: which call holds it, and which way tokens move.
struct CallEnd {
  Operation *call;
  bool isInput; // the child gets from the channel
};

// A channel as this function sees it: the ends it issues itself, the ends its
// children hold, and the seed that breaks a feedback cycle.
struct Channel {
  Value root;
  bool internal = false; // declared here (`stream.create`) vs a boundary arg
  ArrayAttr init;
  SmallVector<Operation *> accesses; // this function's own get / put ops
  bool anyPut = false, anyGet = false;
  unsigned producers = 0; // children writing it; a local put is not one
  SmallVector<CallEnd> callEnds;
};
} // namespace

static LogicalResult checkChannelEnds(func::FuncOp func, const Channel &ch);
static LogicalResult checkChannelCycles(func::FuncOp func,
                                        ArrayRef<Channel> channels);
static LogicalResult checkChannels(
    func::FuncOp func,
    llvm::DenseMap<std::pair<Operation *, unsigned>, bool> &streamArgIsInput);
static LogicalResult checkOperations(func::FuncOp func,
                                     const OperatorLibrary &lib);
static LogicalResult checkSignature(func::FuncOp func);
static LogicalResult checkIndexWidth(func::FuncOp func);
static LogicalResult checkStallContract(Operation *op, StringRef symbol);
static LogicalResult checkMemories(func::FuncOp func, const MemoryLibrary &lib,
                                   DenseSet<Value> &boundaryArrays, bool isTop);
static LogicalResult checkComposition(func::FuncOp func,
                                      const DeviceModel &dev);
static LogicalResult checkArgumentAgreement(func::FuncOp func,
                                            const MemoryLibrary &lib);
static LogicalResult verifyFunc(
    func::FuncOp func, ModuleOp module, const DeviceModel &dev,
    llvm::DenseMap<std::pair<Operation *, unsigned>, bool> &streamArgIsInput,
    DenseSet<Value> &boundaryArrays, bool isTop);

// Element count above which a completely-partitioned argument's
// port-per-element boundary is worth a word. An interface-width warning on the
// argument path only, independent of `scalarize-memory`'s local-array gate.
static constexpr int64_t kScatterWarnElements = 16;

// The direction \p call imposes on \p stream, from the callee parameter it is
// passed to. Empty when the callee never resolves one, which the unused
// boundary-argument check below reports against the callee itself.
static std::optional<bool> calleeStreamDirection(
    func::CallOp call, Value stream,
    llvm::DenseMap<std::pair<Operation *, unsigned>, bool> &streamArgIsInput) {
  SymbolTableCollection syms;
  auto callee =
      syms.lookupNearestSymbolFrom<func::FuncOp>(call, call.getCalleeAttr());
  if (!callee)
    return std::nullopt;
  for (auto [k, actual] : llvm::enumerate(call.getArgOperands())) {
    if (actual != stream)
      continue;
    auto it = streamArgIsInput.find(
        {callee.getOperation(), static_cast<unsigned>(k)});
    if (it != streamArgIsInput.end())
      return it->second;
  }
  return std::nullopt;
}

static void recordStreamArgDirections(
    func::FuncOp func,
    llvm::DenseMap<std::pair<Operation *, unsigned>, bool> &streamArgIsInput) {
  for (BlockArgument arg : func.getArguments()) {
    if (!isa<StreamType>(arg.getType()))
      continue;
    for (Operation *user : arg.getUsers()) {
      std::optional<bool> dir;
      if (isa<StreamGetOp>(user))
        dir = true;
      else if (isa<StreamPutOp>(user))
        dir = false;
      else if (auto call = dyn_cast<func::CallOp>(user))
        dir = calleeStreamDirection(call, arg, streamArgIsInput);
      if (dir) {
        streamArgIsInput[{func.getOperation(), arg.getArgNumber()}] = *dir;
        break;
      }
    }
  }
}

LogicalResult verifyFunc(
    func::FuncOp func, ModuleOp module, const DeviceModel &dev,
    llvm::DenseMap<std::pair<Operation *, unsigned>, bool> &streamArgIsInput,
    DenseSet<Value> &boundaryArrays, bool isTop) {
  if (failed(checkSignature(func)) || failed(checkIndexWidth(func)) ||
      failed(checkOperations(func, dev.operators)) ||
      failed(checkMemories(func, dev.memory, boundaryArrays, isTop)) ||
      failed(checkComposition(func, dev)))
    return failure();
  return checkChannels(func, streamArgIsInput);
}

//===--------------------------------------------------------------------===//
// Signature and operations.
//===--------------------------------------------------------------------===//

LogicalResult checkSignature(func::FuncOp func) {
  for (Type t : func.getResultTypes())
    if (isa<MemRefType>(t)) {
      unsupported(Stage::Prep, Code::MemrefResult, func)
          << "Returning a memref is not lowered yet; write the result "
             "through an output argument (out-parameter) instead";
      return failure();
    }
  return success();
}

// Whether \p v is provably non-negative: a counted-loop induction variable
// with a constant non-negative lower bound, or a non-negative constant.
static bool nonNegativeOperand(Value v) {
  if (auto loop = affine::getForInductionVarOwner(v))
    return loop.hasConstantLowerBound() && loop.getConstantLowerBound() >= 0;
  if (auto loop = scf::getForInductionVarOwner(v)) {
    std::optional<int64_t> lb = getConstantIntValue(loop.getLowerBound());
    return lb && *lb >= 0;
  }
  IntegerAttr::ValueType cst;
  return matchPattern(v, m_ConstantInt(&cst)) && cst.isNonNegative();
}

// Whether \p e is provably non-negative over \p operands (dims then symbols):
// sums and products of non-negatives, or a division with a constant positive
// divisor over one.
static bool provablyNonNegative(AffineExpr e, ValueRange operands,
                                unsigned numDims) {
  if (auto c = dyn_cast<AffineConstantExpr>(e))
    return c.getValue() >= 0;
  if (auto d = dyn_cast<AffineDimExpr>(e))
    return nonNegativeOperand(operands[d.getPosition()]);
  if (auto s = dyn_cast<AffineSymbolExpr>(e))
    return nonNegativeOperand(operands[numDims + s.getPosition()]);
  auto bin = cast<AffineBinaryOpExpr>(e);
  if (!provablyNonNegative(bin.getLHS(), operands, numDims))
    return false;
  switch (e.getKind()) {
  case AffineExprKind::Add:
  case AffineExprKind::Mul:
    return provablyNonNegative(bin.getRHS(), operands, numDims);
  case AffineExprKind::FloorDiv:
  case AffineExprKind::CeilDiv:
  case AffineExprKind::Mod: {
    auto k = dyn_cast<AffineConstantExpr>(bin.getRHS());
    return k && k.getValue() > 0;
  }
  default:
    return false;
  }
}

// A standalone apply is emitted by `evalAffine`, whose floordiv/mod lowering
// is unsigned: exact only with a constant positive divisor over a non-negative
// argument. An access map's subscript is in-bounds and so non-negative by
// construction; a standalone apply has no such guarantee, so a division this
// cannot prove is refused.
static LogicalResult checkApplyDivision(affine::AffineApplyOp apply) {
  AffineMap map = apply.getAffineMap();
  bool ok = true;
  // The form the emitter builds, not the op's own: `applyExprOf` may fold a
  // division away or refold one in.
  applyExprOf(map).walk([&](AffineExpr e) {
    auto kind = e.getKind();
    if (kind != AffineExprKind::FloorDiv && kind != AffineExprKind::CeilDiv &&
        kind != AffineExprKind::Mod)
      return;
    auto bin = cast<AffineBinaryOpExpr>(e);
    auto k = dyn_cast<AffineConstantExpr>(bin.getRHS());
    if (k && k.getValue() > 0 &&
        provablyNonNegative(bin.getLHS(), apply.getOperands(),
                            map.getNumDims()))
      return;
    ok = false;
    unsupported(Stage::Prep, Code::AffineDivisionUnsupported, apply)
        << "The division in the index expression " << e
        << " is built as unsigned hardware, which needs a constant positive "
           "divisor over a provably non-negative argument, and this one is "
           "not proven either. Compute the expression in arithmetic ops "
           "instead, which lower with signed semantics";
  });
  return success(ok);
}

LogicalResult checkOperations(func::FuncOp func, const OperatorLibrary &lib) {
  WalkResult r = func.walk([&](Operation *op) {
    if (isUnmodeledMemoryAccess(op)) {
      unsupported(Stage::Prep, Code::OperationNotModelled, op)
          << "Operation '" << op->getName()
          << "' carries a memory effect the dependence analysis does not "
             "model, so scheduling would reorder it against the accesses it "
             "aliases. A whole-array assignment (`buf = A`) lowers to "
             "`memref.copy`: write the array element by element in a loop "
             "instead";
      return WalkResult::interrupt();
    }
    if (!isComputeOp(op))
      return WalkResult::advance();
    if (!isPricedOp(op)) {
      unsupported(Stage::Prep, Code::OperationNotModelled, op)
          << "Operation '" << op->getName()
          << "' produces a value no stage of the datapath models: the operator "
             "library prices arithmetic, and the scheduler would place a cell "
             "nothing can build. Express it in arithmetic instead";
      return WalkResult::interrupt();
    }
    if (auto apply = dyn_cast<affine::AffineApplyOp>(op))
      if (failed(checkApplyDivision(apply)))
        return WalkResult::interrupt();
    // The identity names the IP row's symbol or the native comb lowering,
    // and is empty when the device offers neither.
    OperatorIdentity id = operatorIdentity(op, lib);
    if (!id.realized()) {
      error(Stage::Prep, Code::OperatorNotRealized, op)
          << "Operator '" << op->getName()
          << "' is not realized by the device: it has neither an IP module "
             "nor a native lowering. Declare an @ip for it, or add native "
             "support";
      return WalkResult::interrupt();
    }
    // Every candidate IP, not the selected row: which one wins is settled
    // only once the period is, and any of them must be realizable.
    for (StringRef symbol : lib.candidateIPs(op))
      if (failed(checkStallContract(op, symbol)))
        return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return failure(r.wasInterrupted());
}

// `ce` is the only IP port ABI the emitter realizes. Nothing downstream honors
// `elastic`: consumers are scheduled at the operator's fixed latency and the
// instance gets the free-running port shape.
LogicalResult checkStallContract(Operation *op, StringRef symbol) {
  auto opr = SymbolTable::lookupNearestSymbolFrom<dcp::DCPathOperatorOp>(
      op, StringAttr::get(op->getContext(), symbol));
  assert(opr && "a matched operator row names a live dcp.operator");
  if (opr.getStall() != StallContractEnum::Elastic)
    return success();
  error(Stage::Prep, Code::StallContractUnusable, op)
      << "Operator IP '" << symbol
      << "' declares the elastic (valid/ready, variable-latency) stall "
         "contract, which is not realized. Declare style='ce'";
  return failure();
}

//===--------------------------------------------------------------------===//
// Index width: every compile-time value the datapath carries as an `index`
// (a counted loop's bounds, an index literal, an array's linear extent) must
// fit the `kIndexWidth` carrier. The width derivations downstream clamp to
// that width, so a value past it would wrap silently instead of failing.
//===--------------------------------------------------------------------===//

// Whether \p v fits the signed `kIndexWidth` carrier.
static bool fitsIndex(int64_t v) {
  return APInt(64, static_cast<uint64_t>(v), /*isSigned=*/true)
             .getSignificantBits() <= kIndexWidth;
}

static void refuseIndexWidth(Operation *op, StringRef what, int64_t v) {
  unsupported(Stage::Prep, Code::IndexWidthExceeded, op)
      << "The " << what << " " << v << " does not fit the " << kIndexWidth
      << "-bit index carrier the datapath builds, so its hardware would wrap "
         "silently";
}

// The bounds of \p op's counted loop, each present when compile-time.
struct LoopBounds {
  std::optional<int64_t> lb, ub, step;
};

static std::optional<LoopBounds> loopBoundsOf(Operation *op) {
  if (auto loop = dyn_cast<affine::AffineForOp>(op)) {
    LoopBounds b;
    if (loop.hasConstantLowerBound())
      b.lb = loop.getConstantLowerBound();
    if (loop.hasConstantUpperBound())
      b.ub = loop.getConstantUpperBound();
    b.step = loop.getStepAsInt();
    return b;
  }
  if (auto loop = dyn_cast<scf::ForOp>(op))
    return LoopBounds{getConstantIntValue(loop.getLowerBound()),
                      getConstantIntValue(loop.getUpperBound()),
                      getConstantIntValue(loop.getStep())};
  return std::nullopt;
}

// The first offending value of \p b, with its name, or nullopt. Individual
// bounds are vetted before the derived one-past value, so its arithmetic
// cannot overflow an int64.
static std::optional<std::pair<StringRef, int64_t>>
oversizedBound(const LoopBounds &b) {
  if (b.lb && !fitsIndex(*b.lb))
    return {{"loop lower bound", *b.lb}};
  if (b.ub && !fitsIndex(*b.ub))
    return {{"loop upper bound", *b.ub}};
  if (b.step && !fitsIndex(*b.step))
    return {{"loop step", *b.step}};
  // The one-past value the counter's terminator compares against
  // (`counterWidth`'s `last`) can exceed the bound by up to a step.
  if (b.lb && b.ub && b.step && *b.step > 0 && *b.ub > *b.lb) {
    int64_t trip = llvm::divideCeilSigned(*b.ub - *b.lb, *b.step);
    int64_t last = *b.lb + trip * *b.step;
    if (!fitsIndex(last))
      return {{"loop bound one-past value", last}};
  }
  return std::nullopt;
}

LogicalResult checkIndexWidth(func::FuncOp func) {
  // The linear extent decides the address width; each memref is checked at
  // its root (an argument or an allocation), not per access.
  auto oversizedExtent = [](Type t) {
    auto mt = dyn_cast<MemRefType>(t);
    if (!mt || !mt.hasStaticShape())
      return false;
    int64_t elements = 1;
    for (int64_t d : mt.getShape())
      if (llvm::MulOverflow(elements, d, elements) ||
          elements > (int64_t(1) << kIndexWidth))
        return true;
    return false;
  };
  for (BlockArgument arg : func.getArguments())
    if (oversizedExtent(arg.getType())) {
      refuseIndexWidth(func, "array extent",
                       cast<MemRefType>(arg.getType()).getNumElements());
      return failure();
    }
  WalkResult r = func.walk([&](Operation *op) {
    if (std::optional<LoopBounds> b = loopBoundsOf(op)) {
      if (auto bad = oversizedBound(*b)) {
        refuseIndexWidth(op, bad->first, bad->second);
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    }
    if (auto cst = dyn_cast<arith::ConstantOp>(op);
        cst && isa<IndexType>(cst.getType())) {
      int64_t v = cast<IntegerAttr>(cst.getValue()).getInt();
      if (!fitsIndex(v)) {
        refuseIndexWidth(op, "index literal", v);
        return WalkResult::interrupt();
      }
    }
    for (Type t : op->getResultTypes())
      if (oversizedExtent(t)) {
        refuseIndexWidth(op, "array extent",
                         cast<MemRefType>(t).getNumElements());
        return WalkResult::interrupt();
      }
    return WalkResult::advance();
  });
  return failure(r.wasInterrupted());
}

//===--------------------------------------------------------------------===//
// Storage.
//===--------------------------------------------------------------------===//

// A local buffer read before anything writes it takes whatever its storage
// holds: zeros in csim, but the previous iteration's values in hardware, since
// the buffer is reused across iterations. Legal MLIR, so this only warns.
static void warnUninitializedReads(func::FuncOp func) {
  llvm::MapVector<Value, Operation *> firstTouch;
  DenseSet<Value> opaque; // a buffer whose whole traffic is not in view
  // Pre-order is program order, and structured control flow has no other
  // execution order, so the first access seen is the first one that runs.
  func.walk<WalkOrder::PreOrder>([&](Operation *op) {
    bool isAccess = isa<affine::AffineLoadOp, affine::AffineStoreOp,
                        memref::LoadOp, memref::StoreOp>(op);
    for (Value operand : op->getOperands())
      if (isa<MemRefType>(operand.getType())) {
        if (isAccess)
          firstTouch.insert({operand, op});
        else
          opaque.insert(operand); // a sub-kernel writes through its own port
      }
  });

  for (auto [array, first] : firstTouch) {
    Operation *alloc = array.getDefiningOp();
    if (opaque.count(array) ||
        !isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(alloc) ||
        isa<affine::AffineStoreOp, memref::StoreOp>(first))
      continue;
    // A read under a condition the declaration is not under may not run, so it
    // is no evidence that nothing wrote first.
    bool conditional = false;
    for (Operation *p = first->getParentOp(); p && p != alloc->getParentOp();
         p = p->getParentOp())
      conditional |= isa<scf::IfOp, affine::AffineIfOp>(p);
    if (conditional)
      continue;
    warn(Stage::Prep, first)
        << "The local array " << array.getType()
        << " is read before anything writes it. Simulation reads zeros and "
           "the hardware reads whatever the buffer held on the previous "
           "iteration, so csim and cosim will disagree; write every element "
           "the kernel reads";
  }
}

// The array arguments that are real boundary ports: the top function's own, and
// every callee argument one is passed down to. Any other sub-kernel argument
// names storage its caller owns, which can realize any device timing including
// a combinational read; only a boundary port answers to an external driver.
static DenseSet<Value> boundaryArraysOf(func::FuncOp top) {
  DenseSet<Value> boundary;
  for (BlockArgument arg : top.getArguments())
    if (isa<MemRefType>(arg.getType()))
      boundary.insert(arg);
  SymbolTableCollection syms;
  SmallVector<func::FuncOp> work{top};
  while (!work.empty()) {
    func::FuncOp f = work.pop_back_val();
    f.walk([&](func::CallOp call) {
      auto callee = syms.lookupNearestSymbolFrom<func::FuncOp>(
          call, call.getCalleeAttr());
      if (!callee || callee.isExternal())
        return;
      // Re-walk a callee a later caller hands a boundary array it had not seen.
      // The set only grows and is bounded by the argument count, so this ends.
      bool grew = false;
      for (auto [k, actual] : llvm::enumerate(call.getArgOperands()))
        if (boundary.contains(actual))
          grew |= boundary.insert(callee.getArgument(k)).second;
      if (grew)
        work.push_back(callee);
    });
  }
  return boundary;
}

// The boundary contract of a completely-partitioned argument, reached when such
// an argument resolved to a 0-cycle read: not an addressed port but one port
// per element (`MemUnit::scattered`), since a complete partition commits the
// scheduler to unlimited combinational ports.
static LogicalResult checkScatteredArgument(func::FuncOp func, Value array,
                                            MemoryChar &mc,
                                            const MemoryLibrary &memLib,
                                            bool isTop) {
  auto elements = cast<MemRefType>(array.getType()).getNumElements();
  if (!memLib.isScatter(mc.storage)) {
    error(Stage::Prep, Code::StorageTimingUnrealizable, func)
        << "Argument array with a 0-cycle read cannot be realized; a boundary "
           "port is edge-triggered, so its datum arrives no earlier than the "
           "cycle after its address. Bind this argument to a storage impl with "
           "a >= 1 cycle read, or copy it into a local buffer";
    return failure();
  }
  // Below the top, the array is a port the child masters on caller-owned
  // storage. A scattered TOP argument reaching a child has no such owner: the
  // top would have to crossbar its input wires.
  if (!isTop) {
    unsupported(Stage::Prep, Code::ScatteredArgumentToCallee, func)
        << "Passing the completely-partitioned argument array "
        << array.getType()
        << " to a sub-kernel is not lowered yet: it crosses the top boundary "
           "as "
           "one port per element, which a sub-kernel's addressed port cannot "
           "be "
           "wired to. Copy it into a local array and pass that one";
    return failure();
  }
  // One port per element is what the argument asked for, but it is also real
  // area at the top level.
  if (elements > kScatterWarnElements)
    warn(Stage::Prep, func)
        << "The completely-partitioned argument array " << array.getType()
        << " becomes " << elements
        << " module ports, one per element; that is a wide top-level interface";
  return success();
}

LogicalResult checkMemories(func::FuncOp func, const MemoryLibrary &memLib,
                            DenseSet<Value> &boundaryArrays, bool isTop) {
  warnUninitializedReads(func);
  SmallVector<Value> arrays;
  for (BlockArgument arg : func.getArguments())
    if (isa<MemRefType>(arg.getType()))
      arrays.push_back(arg);
  func.walk([&](Operation *op) {
    if (isa<memref::AllocOp, memref::AllocaOp, memref::GetGlobalOp>(op))
      arrays.push_back(op->getResult(0));
  });

  for (Value array : arrays) {
    MemoryChar mc = characterize(array, memLib);
    StringRef storage = mc.storage;
    Operation *anchor =
        array.getDefiningOp() ? array.getDefiningOp() : func.getOperation();
    // A complete partition scatters the array regardless of its bound storage;
    // nothing else resolves an explicitly bound array to the scatter row.
    StringRef bound = boundStorageOf(array);
    if (memLib.isScatter(storage) && !bound.empty() &&
        !memLib.isScatter(bound)) {
      error(Stage::Prep, Code::ArrayLayoutConflict, anchor)
          << "Array " << array.getType() << " is bound to storage '" << bound
          << "' and also completely partitioned, which scatters it into "
             "registers; the two cannot both hold. Drop one of them";
      return failure();
    }
    // A `table` row is a lookup built out of logic with no port to take a
    // store, so only an initialized array nothing writes may bind to one.
    // Reached only through an explicit binding.
    if (memLib.isTable(storage) && !isa<BlockArgument>(array) &&
        !isConstantTable(array)) {
      error(Stage::Prep, Code::ArrayLayoutConflict, anchor)
          << "Array " << array.getType() << " is bound to storage '" << storage
          << "', which is a constant table: it holds compile-time contents and "
             "has no write port. Declare the array with its contents and write "
             "it nowhere, or bind it to a storage that can be written";
      return failure();
    }
    // The emitter realizes compile-time contents as one bank: a constant table
    // is a single `hw.aggregate_constant` and a written one a single `initial`
    // block. A complete partition is not banking and stays legal.
    if (globalInitOf(array) && mc.layout.numBanks > 1) {
      unsupported(Stage::Prep, Code::PartitionedInitializedArray, anchor)
          << "Array " << array.getType()
          << " is declared with compile-time contents and partitioned into "
          << mc.layout.numBanks
          << " banks, which the backend realizes as one bank. Drop the "
             "partition, or fill the array from the kernel instead";
      return failure();
    }
    // Nowhere to hold the array; an empty name would otherwise fall through as
    // a stream's, timed by an unrelated row.
    if (storage.empty()) {
      auto diag = error(Stage::Prep, Code::StorageNotDeclared, anchor);
      diag << "Array " << array.getType();
      if (mc.layout.registers)
        diag << " is completely partitioned, but the device marks no "
                "`dcp.storage` `scatter` for one cell per element to be built "
                "out of";
      else
        diag << " has no `bind_storage impl`, and the device marks no "
                "`default` `dcp.storage` and declares none the compiler can "
                "choose: a row it may choose has to carry a `style` to pin the "
                "array with and `uses` to price it by";
      return failure();
    }
    // A realization the device never declared would fall to the zero-timing
    // default and schedule combinationally, reading before valid.
    const StorageRealization *row = memLib.row(storage);
    if (!row) {
      error(Stage::Prep, Code::StorageNotDeclared, anchor)
          << "No memory characterization for storage '" << storage
          << "'; declare it as a `dcp.storage` on the device";
      return failure();
    }
    // The two axes of one directive: `type=` asks for a port topology and
    // `impl=` picks the structure that has to provide it. `characterize` keeps
    // the tighter of the two, so a directive asking for more than its structure
    // has would otherwise run at the structure's ports.
    if (auto want = requestedPortsOf(array); want && !row->ports.holds(*want)) {
      error(Stage::Prep, Code::ArrayLayoutConflict, anchor)
          << "Array " << array.getType() << " asks for a port topology of "
          << want->describe() << ", but storage '" << storage << "' has "
          << row->ports.describe()
          << ". Drop the `type` or bind it to a storage that has them";
      return failure();
    }
    // A structure that powers up undefined cannot hold an array declared with
    // contents. A read-only table escapes this: it is realized as logic and
    // never reaches this storage.
    if (!row->canInit && globalInitOf(array) && !mc.constantTable) {
      error(Stage::Prep, Code::StorageTimingUnrealizable, anchor)
          << "Array " << array.getType()
          << " is declared with compile-time contents and also written, so it "
             "is a memory that must come up holding them, but storage '"
          << storage
          << "' powers up undefined. Bind it to a storage that can be "
             "initialized, or fill it from the kernel instead";
      return failure();
    }
    RWLatency lat = row->timing.latency;
    // A boundary port's latency is a contract with the driver, not enforced by
    // the RTL: any latency >= 1 works, but 0 does not, since the port is
    // edge-triggered.
    if (boundaryArrays.contains(array) && lat.read < 1 &&
        failed(checkScatteredArgument(func, array, mc, memLib, isTop)))
      return failure();
    // An `seq.hlmem` write is edge-triggered too: a store commits at
    // `writeLatency - 1`, which a 0-cycle write wraps. A 0-cycle read is fine
    // internally.
    if (lat.write < 1) {
      error(Stage::Prep, Code::StorageTimingUnrealizable, anchor)
          << "Storage '" << storage
          << "' declares a 0-cycle write, which no array can be realized at: "
             "a write needs a clock edge to commit on. Give that "
             "`dcp.storage` a write latency of at least 1";
      return failure();
    }
  }
  return checkArgumentAgreement(func, memLib);
}

// Whether two memrefs are banked identically, axis for axis. Bank count alone
// is not the contract: the bank an element lands in is the mixed-radix fold of
// the axes in the order the attribute spells them, so two layouts of equal
// width can still send the same element to different banks.
static bool sameBanking(BankLayout &a, BankLayout &b) {
  if (a.registers != b.registers || a.numBanks != b.numBanks ||
      a.axes.size() != b.axes.size())
    return false;
  for (auto [x, y] : llvm::zip(a.axes, b.axes))
    if (x.dim != y.dim || x.factor != y.factor || x.kind != y.kind ||
        x.extent != y.extent)
      return false;
  return true;
}

// What caller and callee must agree on for one array argument. The array is
// built once and the sub-kernel masters a port on it, so both sides describe
// one structure.
//
// Banking: a sub-kernel masters one port group per bank and indexes each in
// that bank's own element space, so at a different layout the child addresses
// the wrong elements.
//
// Storage: the row carries the latency every side was scheduled at and the
// latency the emitter builds ports at, so two rows time one memory two ways and
// the shorter one samples before the data is valid.
LogicalResult checkArgumentAgreement(func::FuncOp func,
                                     const MemoryLibrary &lib) {
  SymbolTableCollection syms;
  WalkResult r = func.walk([&](func::CallOp call) {
    auto callee =
        syms.lookupNearestSymbolFrom<func::FuncOp>(call, call.getCalleeAttr());
    if (!callee || callee.isExternal())
      return WalkResult::advance();
    for (auto [k, actual] : llvm::enumerate(call.getArgOperands())) {
      if (!isa<MemRefType>(actual.getType()))
        continue;
      BlockArgument param = callee.getArgument(k);
      BankLayout here = bankLayoutOf(actual);
      BankLayout there = bankLayoutOf(param);
      if (!isa<BlockArgument>(actual) && !sameBanking(here, there)) {
        error(Stage::Prep, Code::ArrayLayoutConflict, call)
            << "Array argument " << k << " of sub-kernel '" << call.getCallee()
            << "' is partitioned into " << there.numBanks
            << " bank(s) there but into " << here.numBanks
            << " in the caller; a sub-kernel addresses each bank in that "
               "bank's own space, so the two partitions must match. Give the "
               "array the same partition factor in both kernels";
        return WalkResult::interrupt();
      }
      // Unlike banking, this holds for a boundary array too: its cells are the
      // caller's, but the latency both sides read them at is one number.
      std::string mine = resolvedStorageOf(actual);
      std::string theirs = resolvedStorageOf(param);
      if (mine == theirs)
        continue;
      // `row` rather than `timing`: `timing` asserts on a name the device does
      // not declare, which is the callee's own check to report.
      auto cycles = [&](StringRef name) {
        const StorageRealization *s = lib.row(name);
        std::string out;
        llvm::raw_string_ostream os(out);
        if (s)
          os << "read " << s->timing.latency.read << ", write "
             << s->timing.latency.write << " cycle(s)";
        else
          os << "undeclared";
        return out;
      };
      error(Stage::Prep, Code::StorageConflict, call)
          << "Array argument " << k << " of sub-kernel '" << call.getCallee()
          << "' is held in storage '" << mine << "' (" << cycles(mine)
          << ") in the caller but in '" << theirs << "' (" << cycles(theirs)
          << ") there; the array is built once and the sub-kernel masters a "
             "port on it, so the two would time one memory differently. Bind "
             "it to the same storage in both kernels";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return failure(r.wasInterrupted());
}

//===--------------------------------------------------------------------===//
// Composition and control shape.
//===--------------------------------------------------------------------===//

LogicalResult checkComposition(func::FuncOp func, const DeviceModel &dev) {
  if (composesOnStructuralTop(func)) {
    // A loop around a spawn reads as loose control flow to the check below, so
    // name it first: the loop is what the user has to move.
    WalkResult r = func.walk([&](func::CallOp call) {
      if (!call->getParentOfType<LoopLikeOpInterface>())
        return WalkResult::advance();
      error(Stage::Prep, Code::SpawnInLoop, call)
          << "A dataflow process is spawned inside a loop; a process is "
             "instantiated once and runs concurrently, so spawn it once and "
             "let it iterate internally (move the loop into the process)";
      return WalkResult::interrupt();
    });
    if (r.wasInterrupted())
      return failure();
    // `outline-loose-processes` lifts a loose span into a process of its own,
    // but skips a span whose live-in is not an array, a stream or a scalar.
    for (Operation &op : func.front())
      if (!isContainerStructure(op)) {
        unsupported(Stage::Prep, Code::ContainerWithDatapath, &op)
            << "A dataflow container with its own datapath (loose "
               "load/store/compute beside the process network) is not "
               "lowered yet; it composes child instances and channels only. "
               "The outliner leaves a span in place when a value crossing it "
               "is neither an array, a stream nor a scalar";
        return failure();
      }
  }

  WalkResult r = func.walk([&](scf::WhileOp w) {
    if (!whileFlushingPipelines(w, dev) || whileHasIdentityForwarding(w))
      return WalkResult::advance();
    error(Stage::Prep, Code::WhileForwardingNotIdentity, w)
        << "While loop not scheduled: its loop-carried values are not "
           "forwarded 1:1 from the before-region through `scf.condition` "
           "into the after-region (they are reordered, dropped, or "
           "recombined), which the flushing-pipeline schedule requires; "
           "carry each value through unchanged";
    return WalkResult::interrupt();
  });
  return failure(r.wasInterrupted());
}

//===--------------------------------------------------------------------===//
// Channels.
//===--------------------------------------------------------------------===//

// Check every channel's ends and the cycles between them. Both properties are
// settled by the process network's shape, which nothing between here and
// emission changes.
static LogicalResult checkChannels(
    func::FuncOp func,
    llvm::DenseMap<std::pair<Operation *, unsigned>, bool> &streamArgIsInput) {
  SmallVector<Channel> channels;
  llvm::DenseMap<Value, unsigned> index;
  auto channelFor = [&](Value stream) -> Channel & {
    auto [it, fresh] = index.try_emplace(stream, channels.size());
    if (fresh) {
      Channel ch;
      ch.root = stream;
      if (auto cr = stream.getDefiningOp<StreamCreateOp>()) {
        ch.internal = true;
        ch.init = cr.getInitAttr();
      }
      channels.push_back(std::move(ch));
    }
    return channels[it->second];
  };
  for (BlockArgument arg : func.getArguments())
    if (isa<StreamType>(arg.getType()))
      channelFor(arg);
  func.walk([&](StreamCreateOp cr) { channelFor(cr.getStream()); });

  func.walk([&](Operation *op) {
    if (isa<StreamGetOp, StreamPutOp>(op)) {
      Channel &ch = channelFor(op->getOperand(0));
      ch.accesses.push_back(op);
      (isa<StreamPutOp>(op) ? ch.anyPut : ch.anyGet) = true;
    } else if (auto call = dyn_cast<func::CallOp>(op)) {
      for (Value actual : call.getArgOperands()) {
        if (!isa<StreamType>(actual.getType()))
          continue;
        std::optional<bool> reads =
            calleeStreamDirection(call, actual, streamArgIsInput);
        if (!reads)
          continue;
        Channel &ch = channelFor(actual);
        (*reads ? ch.anyGet : ch.anyPut) = true;
        ch.producers += !*reads;
        ch.callEnds.push_back({call, *reads});
      }
    }
  });

  for (const Channel &ch : channels)
    if (failed(checkChannelEnds(func, ch)))
      return failure();
  return checkChannelCycles(func, channels);
}

LogicalResult checkChannelEnds(func::FuncOp func, const Channel &ch) {
  // An access this module issues, else the child instance holding one of the
  // channel's ends. A boundary channel with neither has only the function.
  Operation *anchor = func.getOperation();
  if (!ch.accesses.empty())
    anchor = ch.accesses.front();
  else if (!ch.callEnds.empty())
    anchor = ch.callEnds.front().call;

  // Several readers are a fan-out the emitter inserts, one FIFO each; several
  // writers are a merge, whose token interleaving is not deterministic.
  if (ch.producers > 1) {
    unsupported(Stage::Prep, Code::ChannelMultiProducer, anchor)
        << "A stream channel is written by more than one process; a channel "
           "is single-producer and a deterministic merge is not lowered yet";
    return failure();
  }
  // A port is an input or an output, so a boundary channel both read and
  // written has nothing to lower to.
  if (ch.anyPut && ch.anyGet && !ch.internal) {
    unsupported(Stage::Prep, Code::StreamArgumentBidirectional, anchor)
        << "A stream ARGUMENT both read and written inside one kernel is not "
           "lowered yet (a boundary channel lowers to one directional port); "
           "route the feedback through a second channel, or declare the "
           "channel inside the kernel";
    return failure();
  }
  // A local channel with one end only stalls by construction: the puts fill it
  // and block, or the first get waits on a token nothing produces.
  if (ch.internal && !(ch.anyPut && ch.anyGet)) {
    error(Stage::Prep, Code::ChannelEndMissing, anchor)
        << "The kernel-local stream is "
        << (ch.anyPut ? "never read" : "never written")
        << "; a channel needs both ends inside the kernel that owns it";
    return failure();
  }
  // A boundary argument nothing touches would leave a port undriven.
  if (!ch.internal && !ch.anyPut && !ch.anyGet) {
    error(Stage::Prep, Code::ChannelEndMissing, anchor)
        << "The stream argument is neither read nor written";
    return failure();
  }
  return success();
}

// A directed cycle of channels with no initial tokens deadlocks, so it
// suffices that the graph of UNSEEDED channels is acyclic. Insufficient
// seeding (fewer tokens than the recurrence distance) surfaces as a hang.
LogicalResult checkChannelCycles(func::FuncOp func,
                                 ArrayRef<Channel> channels) {
  llvm::DenseMap<Operation *, SmallVector<Operation *>> adj;
  SetVector<Operation *> nodes;
  for (const Channel &ch : channels) {
    for (const CallEnd &e : ch.callEnds)
      nodes.insert(e.call);
    if (ch.init && !ch.init.empty())
      continue;
    Operation *prod = nullptr;
    for (const CallEnd &e : ch.callEnds)
      if (!e.isInput)
        prod = e.call;
    if (!prod)
      continue; // fed from a boundary port: not part of a cycle
    for (const CallEnd &e : ch.callEnds)
      if (e.isInput)
        adj[prod].push_back(e.call);
  }

  llvm::DenseMap<Operation *, int> color; // 0 white / 1 gray / 2 black
  llvm::DenseMap<Operation *, Operation *> parent;
  SmallVector<Operation *> cycle;
  // Self-parameter recursive lambda: a local DFS with no std::function.
  auto visit = [&](auto &self, Operation *u) -> bool {
    color[u] = 1;
    for (Operation *v : adj[u]) {
      if (color[v] == 1) { // back edge -> the cycle v .. u -> v
        for (Operation *x = u; x != v; x = parent[x])
          cycle.push_back(x);
        cycle.push_back(v);
        return true;
      }
      if (color[v] == 0) {
        parent[v] = u;
        if (self(self, v))
          return true;
      }
    }
    color[u] = 2;
    return false;
  };
  for (Operation *n : nodes)
    if (cycle.empty() && color[n] == 0)
      visit(visit, n);
  if (cycle.empty())
    return success();

  std::reverse(cycle.begin(), cycle.end()); // producer order
  std::string path;
  llvm::raw_string_ostream os(path);
  for (Operation *x : cycle)
    os << cast<func::CallOp>(x).getCallee() << " -> ";
  os << cast<func::CallOp>(cycle.front()).getCallee(); // close the loop
  error(Stage::Prep, Code::DataflowCycleUnseeded, func)
      << "Dataflow feedback cycle [" << path
      << "] has no initial tokens and will deadlock; seed a channel on the "
         "cycle with an initializer, e.g. `s: Stream[T, depth] = [<init>]`";
  return failure();
}

//===----------------------------------------------------------------------===//
// Address cost: what each access's address arithmetic prices at.
//
// An address is folded into the access's affine map rather than standing as
// its own op, so chaining cannot split it; its cone reaches the solve through
// the access's own incoming delay, and a cone past the clock period raises
// the scheduled period rather than failing here (`runSDCScheduler`).
// Per-access lines are `debug` level.
//===----------------------------------------------------------------------===//
static void reportAddressCost(func::FuncOp funcOp, const DeviceModel &dev) {
  unsigned total = 0, nonTrivial = 0, banked = 0, withDivider = 0;
  double worstNow = 0.0;

  funcOp.walk([&](Operation *op) {
    std::optional<MemAccess> a = asMemAccess(op);
    if (!a || a->kind != AccessKind::Array)
      return;
    ++total;
    auto shape = cast<MemRefType>(a->root.getType()).getShape();
    AddressExprs e = addressExprsOf(bankLayoutOf(a->root), a->map, shape,
                                    assignedBankOf(op));
    if (e.bank)
      ++banked;
    AddressCost cost = addressCostOf(op, dev.operators);
    if (cost.trivial())
      return;
    ++nonTrivial;
    worstNow = std::max(worstNow, cost.delay);
    std::string addr;
    llvm::raw_string_ostream os(addr);
    os << e.offset;
    if (e.bank)
      os << " (bank " << e.bank << ")";
    if (cost.dividers || cost.reciprocals) {
      ++withDivider;
      warn(Stage::Prep, op)
          << "The address " << addr << " costs "
          << llvm::format("%.2f", cost.delay)
          << " ns because it divides at "
             "runtime. A divisor that is a power of two is a mask, and a "
             "dividend that advances with a loop counter is a register the "
             "controller maintains; a subscript that is neither pays a "
             "reciprocal multiply in the cone. Choose a power-of-two "
             "partition factor, or index the array by a loop counter";
    }
    debug(Stage::Prep, op) << "Address " << addr << ": "
                           << llvm::format("%.2f", cost.delay) << " ns at "
                           << e.width << "b [" << cost.adders << " add, "
                           << cost.dividers << " div, " << cost.reciprocals
                           << " recip, " << cost.multipliers << " mul]";
  });

  if (!total)
    return;
  debug(Stage::Prep) << "Address arithmetic in " << funcOp.getSymName().str()
                     << ": " << nonTrivial << "/" << total
                     << " array accesses cost logic, worst "
                     << llvm::format("%.2f", worstNow) << " ns (" << withDivider
                     << " carrying a divider, " << banked
                     << " decoding a bank at runtime)";
}

LogicalResult allo::runPreScheduleVerification(ModuleOp module, StringRef top) {
  module->getContext()->getOrLoadDialect<func::FuncDialect>();

  auto topFunc = module.lookupSymbol<func::FuncOp>(top);
  if (!topFunc) {
    error(Stage::Prep, Code::TopFunctionMissing, module)
        << "Top function '" << top << "' not found";
    return failure();
  }
  // The closure the emit driver visits, callees before callers so a call can
  // read facts already computed for its callee.
  auto closureOr = callGraphPostOrder(topFunc);
  if (failed(closureOr))
    return failure();

  DeviceModel dev = DeviceModel::fromModule(module);
  llvm::DenseMap<std::pair<Operation *, unsigned>, bool> streamArgIsInput;
  DenseSet<Value> boundaryArrays = boundaryArraysOf(topFunc);
  for (func::FuncOp fn : *closureOr)
    recordStreamArgDirections(fn, streamArgIsInput);
  for (func::FuncOp fn : *closureOr)
    if (failed(verifyFunc(fn, module, dev, streamArgIsInput, boundaryArrays,
                          fn == topFunc)))
      return failure();
  // Last, because it prices the addresses the banking above has just been held
  // legal: a rejected partition would make the cost meaningless.
  for (func::FuncOp fn : *closureOr)
    reportAddressCost(fn, dev);
  return success();
}
