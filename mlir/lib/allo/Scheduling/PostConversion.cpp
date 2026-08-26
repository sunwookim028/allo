/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Scheduling/LatencyModel.h" // composeSpan, composeDag
#include "allo/Scheduling/MemoryAccess.h" // asMemAccess
#include "allo/Scheduling/MemoryModel.h"  // kBankAttr
#include "allo/Scheduling/OperatorLibrary.h"
#include "allo/Scheduling/ProblemBuilder.h"
#include "allo/Scheduling/RegionGraph.h"
#include "allo/Scheduling/ScheduleModel.h"

#include "allo/IR/AlloOps.h"
#include "allo/Support/Logging.h"

#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::allo;
using namespace mlir::allo::dcp;
using namespace mlir::allo::logging;

// An `i64` attribute for an optional value, or a null attribute (elided) when
// absent.
static IntegerAttr optI64Attr(Builder &b, std::optional<int64_t> v) {
  return v ? b.getI64IntegerAttr(*v) : IntegerAttr();
}

// Erase \p op, dropping the schedule of everything under it first: an erased
// op's address is handed back out by the next `create`, so a stale entry
// would answer for whatever lands there next.
static void eraseScheduled(ScheduleModel &model, Operation *op) {
  op->walk([&](Operation *inner) { model.forget(inner); });
  op->erase();
}

// A `#allo.determinacy<...>` attribute.
static DeterminacyEnumAttr determinacyAttr(Builder &b, DeterminacyEnum d) {
  return DeterminacyEnumAttr::get(b.getContext(), d);
}

//===----------------------------------------------------------------------===//
// Per-op conversion. The `dcp.operator` symbols are already injected from the
// device model, so the reifier only references them and never materializes one.
//===----------------------------------------------------------------------===//

// Defined with the call machinery below.
static DCPathInstanceOp makeInvoke(OpBuilder &b, Location loc,
                                   TypeRange resultTypes, ValueRange operands,
                                   FlatSymbolRefAttr calleeAttr, Operation *at,
                                   int64_t start);

// Convert \p op (an op of the scheduled loop body) into its `dcp` equivalent in
// the pipeline block \p b is inserting into, mapping its results in \p map. Ops
// that are not compute/memory (constants, address arithmetic) are cloned as-is.
// Each reified memory access is recorded in \p accessMap under its source op,
// for `stampForwards` to pair once the whole block has converted.
static void convertOp(Operation &op, OpBuilder &b, IRMapping &map,
                      ScheduleModel &model, const DeviceModel &dev,
                      DenseMap<Operation *, Operation *> &accessMap) {
  Location loc = op.getLoc();
  const OpSchedule *at = model.scheduleOf(&op);
  int64_t start = at ? at->start : 0;
  auto rm = [&](Value v) { return map.lookupOrDefault(v); };
  auto remap = [&](auto values) {
    SmallVector<Value> out;
    for (Value v : values)
      out.push_back(rm(v));
    return out;
  };
  // The sub-cycle start time, from the chaining solve.
  auto setZ = [&](Operation *dst) {
    if (at && at->startInCycle)
      dst->setAttr("z", b.getF32FloatAttr(*at->startInCycle));
  };
  // An access also carries the setup delay the solve priced it against
  // (`accessCharacterization`: the port's own delay, the address cone and the
  // port select) and the two shares of it that are not the address cone, all
  // read off the original op.
  auto setAccessTiming = [&](Operation *dst) {
    setZ(dst);
    dst->setAttr(
        "in_delay",
        b.getF32FloatAttr(
            accessCharacterization(&op, dev.operators, dev.memory).inDelay));
    dst->setAttr("port_delay", b.getF32FloatAttr(dev.memory.timing(&op).delay));
    if (double sel = portSelectDelay(&op, dev.operators))
      dst->setAttr("select_delay", b.getF32FloatAttr(sel));
  };
  // Keep an op verbatim inside the region, preserving its scheduled start so
  // the schedule export can still report it.
  auto cloneKept = [&]() {
    Operation *c = b.clone(op, map);
    if (at) {
      c->setAttr("start", b.getI64IntegerAttr(start));
      setZ(c);
    }
    return c;
  };

  // A memory access's latency is the accessed memref's read/write latency,
  // asked of the memory model directly: an access is timed by its storage and
  // has no operator row.
  auto memLatency = [&]() -> uint64_t {
    return dev.memory.timing(&op).latency;
  };
  // The bank `assign-banks` decided, moved onto the dcp op's own attribute so
  // no later rewrite can drop it. Absent means the access reaches every bank.
  IntegerAttr bank = op.getAttrOfType<IntegerAttr>(kBankAttr);
  // The address map of an array access: an affine op's own map, and for a
  // non-affine one the identity map over its indices.
  auto addrMap = [&]() { return asMemAccess(&op)->map; };
  if (auto l = dyn_cast<AffineLoadOp>(&op)) {
    auto nw = DCPathLoadOp::create(
        b, loc, l.getType(), rm(l.getMemRef()), remap(l.getMapOperands()),
        addrMap(), (uint64_t)start, memLatency(), bank, IntegerAttr(),
        DenseI64ArrayAttr(), DenseI64ArrayAttr());
    setAccessTiming(nw);
    accessMap[&op] = nw;
    map.map(l.getResult(), nw.getResult());
    return;
  }
  if (auto l = dyn_cast<memref::LoadOp>(&op)) {
    auto nw = DCPathLoadOp::create(
        b, loc, l.getType(), rm(l.getMemRef()), remap(l.getIndices()),
        addrMap(), (uint64_t)start, memLatency(), bank, IntegerAttr(),
        DenseI64ArrayAttr(), DenseI64ArrayAttr());
    setAccessTiming(nw);
    accessMap[&op] = nw;
    map.map(l.getResult(), nw.getResult());
    return;
  }
  if (auto s = dyn_cast<AffineStoreOp>(&op)) {
    auto nw = DCPathStoreOp::create(
        b, loc, rm(s.getValueToStore()), rm(s.getMemRef()),
        remap(s.getMapOperands()), addrMap(), (uint64_t)start, memLatency(),
        bank, IntegerAttr(), IntegerAttr());
    setAccessTiming(nw);
    accessMap[&op] = nw;
    return;
  }
  if (auto s = dyn_cast<memref::StoreOp>(&op)) {
    auto nw = DCPathStoreOp::create(b, loc, rm(s.getValueToStore()),
                                    rm(s.getMemRef()), remap(s.getIndices()),
                                    addrMap(), (uint64_t)start, memLatency(),
                                    bank, IntegerAttr(), IntegerAttr());
    setAccessTiming(nw);
    accessMap[&op] = nw;
    return;
  }
  // Streams stay as FIFO ops, not compute; keep them verbatim with their start.
  if (isa<StreamGetOp, StreamPutOp>(&op)) {
    setAccessTiming(cloneKept());
    return;
  }
  // Declarations stay verbatim so loads that read them still resolve their
  // memref. A `memref.get_global` names a ROM built from the global's
  // initializer.
  if (isa<memref::AllocOp, memref::AllocaOp, memref::GetGlobalOp,
          StreamCreateOp>(&op)) {
    cloneKept();
    return;
  }
  // Every sub-kernel call reifies to a `dcp.instance`, including a
  // scalar-returning one. An `await` spawn differs only in start policy and
  // rides `allo.async`.
  if (auto call = dyn_cast<func::CallOp>(&op)) {
    auto inv =
        makeInvoke(b, loc, call.getResultTypes(), remap(call.getOperands()),
                   call.getCalleeAttr(), &op, start);
    if (call->hasAttr(kAlloAsyncAttr))
      inv->setAttr(kAlloAsyncAttr, b.getUnitAttr());
    for (auto [old, nw] : llvm::zip(call.getResults(), inv.getResults()))
      map.map(old, nw);
    return;
  }
  // A scheduled single-result op (not a constant) is a compute op, realized on
  // one of two exclusive paths: a combinational op carries a `comb_kind`, an IP
  // op references its injected `dcp.operator` via `op_type`.
  if (op.getNumResults() == 1 && at && !isa<arith::ConstantOp>(op)) {
    // The row an exact solve selected outranks the library's own pick: the
    // reify must realize what the schedule priced.
    OperatorChar oc = at->selectedImpl.empty()
                          ? dev.operators.lookup(&op)
                          : dev.operators.lookup(&op, at->selectedImpl);
    OperatorIdentity id = oc.identity;
    assert(id.realized() && "a scheduled compute op with no realization");
    CombOpKindEnumAttr combKind;
    FlatSymbolRefAttr opType;
    if (id.comb)
      combKind = CombOpKindEnumAttr::get(b.getContext(), *id.comb);
    else
      opType = FlatSymbolRefAttr::get(b.getContext(), id.ipSymbol);
    // The instance the allocation put it on, when a solve decided one.
    FlatSymbolRefAttr unit;
    if (at->unit)
      unit = FlatSymbolRefAttr::get(b.getContext(),
                                    model.allocatedUnits()[*at->unit].name);
    auto nw = DCPathComputeOp::create(b, loc, op.getResult(0).getType(),
                                      remap(op.getOperands()), combKind, opType,
                                      b.getI64IntegerAttr(start), unit);
    for (NamedAttribute attr : op.getAttrs())
      nw->setAttr(attr.getName(), attr.getValue());
    setZ(nw);
    // The setup delay the solve priced this op against, so the emitter's model
    // holds the schedule's own number instead of re-deriving one. The rename
    // decision travels the same way, since the operands it is judged on do not
    // survive into `dcp`.
    nw->setAttr("in_delay", b.getF32FloatAttr(oc.timing.inDelay));
    if (isZeroDelay(&op))
      nw->setAttr("rename", b.getUnitAttr());
    map.map(op.getResult(0), nw.getResult());
    return;
  }
  // Constants / address arithmetic: keep verbatim inside the region.
  cloneKept();
}

// Stamp the store->load forwarding pairs the schedule recorded onto the dcp
// accesses just reified: each store gets a func-unique `fwd_id`, each load the
// list of ids its shadow serves it from. Both ends of a pair sit in one block,
// so \p accessMap holds them together and a pair is stamped exactly once. Loads
// are ordered by their reified position, so the ids are a function of the
// program rather than of pointer hashing.
static void stampForwards(ScheduleModel &model,
                          DenseMap<Operation *, Operation *> &accessMap,
                          int64_t &nextFwdId) {
  SmallVector<
      std::pair<Operation *, ArrayRef<std::pair<Operation *, int64_t>>>>
      pairs;
  for (auto &[load, stores] : model.allForwards())
    if (accessMap.contains(load))
      pairs.push_back({load, stores});
  if (pairs.empty())
    return;
  llvm::sort(pairs, [&](const auto &a, const auto &b) {
    return accessMap.lookup(a.first)->isBeforeInBlock(
        accessMap.lookup(b.first));
  });
  Builder ab(pairs.front().first->getContext());
  for (auto [load, stores] : pairs) {
    SmallVector<int64_t> ids, offs;
    for (auto [store, offset] : stores) {
      Operation *reifiedStore = accessMap.lookup(store);
      assert(reifiedStore && "a forwarded pair reifies within one block, so "
                             "its store is in the same access map as its load");
      auto nw = cast<DCPathStoreOp>(reifiedStore);
      if (!nw.getFwdId())
        nw.setFwdIdAttr(ab.getI64IntegerAttr(nextFwdId++));
      ids.push_back(*nw.getFwdId());
      offs.push_back(offset);
    }
    auto ld = cast<DCPathLoadOp>(accessMap.lookup(load));
    ld.setFwdAttr(ab.getDenseI64ArrayAttr(ids));
    ld.setFwdOffAttr(ab.getDenseI64ArrayAttr(offs));
  }
}

//===----------------------------------------------------------------------===//
// The timing attributes a region op is built with.
//===----------------------------------------------------------------------===//

namespace {
// What a `dcp.pipeline` or `dcp.sequential` is constructed with, derived from
// the `RegionSolution` for it. A null solution leaves every field empty: this
// means either an all-constant span the solver skipped, or a residual
// wrapper that owns no solve of its own.
struct RegionAttrs {
  std::optional<int64_t> ii;
  std::optional<int64_t> length; // schedule depth, a report
  std::optional<int64_t> drain;  // terminal cycle, what a span composes from
  std::optional<int64_t> latency;
  bool latencyBound = false;

  RegionAttrs() = default;
  explicit RegionAttrs(const RegionSolution *r) {
    if (!r)
      return;
    ii = r->ii;
    length = r->length;
    drain = r->drain;
    // The region as the latency model sees it; a container's span is later
    // recomposed from its children and overwrites this. What only this
    // supplies is an assume-bounded trip.
    SpanNode n;
    n.drain = r->drain;
    n.ii = r->ii;
    n.acyclic = !r->ii;
    n.trip = r->trip;
    if (!n.trip && n.acyclic)
      n.trip = 1; // a straight-line span runs once
    latency = composeSpan(n);
    latencyBound = r->tripIsBound;
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Region materialization.
//===----------------------------------------------------------------------===//

// Whether \p loop's body holds a nested loop, i.e. it is not truly innermost.
// Must be asked BEFORE the loop's body is materialized, so it sees the raw
// affine/scf children, not the dcp children they later become.
static bool hasNestedLoop(LoopLikeOpInterface loop) {
  bool found = false;
  for (Region *r : loop.getLoopRegions()) {
    r->walk([&](Operation *op) {
      if (isa<AffineForOp, scf::ForOp, scf::WhileOp>(op)) {
        found = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (found)
      break;
  }
  return found;
}

// Compile-time-constant trip count of a counted loop, else nullopt.
static std::optional<int64_t> constantTripOf(LoopLikeOpInterface loop) {
  if (auto affineLoop = dyn_cast<AffineForOp>(loop.getOperation())) {
    if (std::optional<uint64_t> t = getConstantTripCount(affineLoop))
      return static_cast<int64_t>(*t);
    return std::nullopt;
  }
  auto scfLoop = cast<scf::ForOp>(loop.getOperation());
  std::optional<int64_t> lb = getConstantIntValue(scfLoop.getLowerBound());
  std::optional<int64_t> ub = getConstantIntValue(scfLoop.getUpperBound());
  std::optional<int64_t> step = getConstantIntValue(scfLoop.getStep());
  if (lb && ub && step && *step > 0)
    return std::max<int64_t>(0, llvm::divideCeilSigned(*ub - *lb, *step));
  return std::nullopt;
}

// The constant value of an index SSA value, seeing through a `dcp.sequential`
// result. A loop's loop-invariant constant lb/step lands in a preceding
// prologue region before the loop is reified, so the bound operand is that
// region's result rather than a foldable constant.
static std::optional<int64_t> constantIndexThroughRegions(Value v) {
  if (std::optional<int64_t> c = getConstantIntValue(v))
    return c;
  if (auto res = dyn_cast<OpResult>(v))
    if (auto seq = dyn_cast<DCPathSequentialOp>(res.getOwner()))
      return constantIndexThroughRegions(
          seq.getBody().front().getTerminator()->getOperand(
              res.getResultNumber()));
  return std::nullopt;
}

// The lower bound and step of a counted loop, carried onto the dcp.pipeline so
// the induction register holds the real IV (`lb`, `lb+step`, ...). An
// affine.for with a symbolic lb comes back as 0 here; `materializeAffineBound`
// supplies the real one.
namespace {
struct LoopBounds {
  int64_t lb = 0, step = 1; // used iff the matching Value is null (constant)
  Value lbVal, stepVal;     // runtime bound (an scf.for SSA lb/step)
};
} // namespace
static LoopBounds lbStepOf(LoopLikeOpInterface loop) {
  if (auto af = dyn_cast<AffineForOp>(loop.getOperation()))
    return {af.hasConstantLowerBound() ? af.getConstantLowerBound() : 0,
            af.getStepAsInt(), Value(), Value()};
  auto sf = cast<scf::ForOp>(loop.getOperation());
  LoopBounds r;
  if (std::optional<int64_t> c =
          constantIndexThroughRegions(sf.getLowerBound()))
    r.lb = *c;
  else
    r.lbVal = sf.getLowerBound();
  if (std::optional<int64_t> c = constantIndexThroughRegions(sf.getStep()))
    r.step = *c;
  else
    r.stepVal = sf.getStep();
  return r;
}

// The runtime upper bound of an scf.for whose trip is not a compile-time
// constant, wired as the pipeline's ub: the counter runs [lb, ub) and
// terminates on `iv+step >= ub`. An affine.for's symbolic bound goes through
// `materializeAffineBound` instead.
static Value dynamicTripBound(LoopLikeOpInterface loop) {
  auto scfLoop = dyn_cast<scf::ForOp>(loop.getOperation());
  return scfLoop ? scfLoop.getUpperBound() : Value();
}

// The value `expand-region-bounds` reified \p af's bound map into, itself
// scheduled and cut against the clock like any other operation.
static Value scheduledBound(AffineForOp af, bool isLower) {
  EntryCone cone = entryConeOf(af.getOperation());
  Value bound = isLower ? cone.lower : cone.upper;
  assert(bound && "a loop with a runtime bound carries a marker holding it");
  return bound;
}

// Create a `dcp.pipeline`'s single block: an index counter (arg 0) followed by
// one arg per iter-arg init. Counted and while pipelines share this shape.
static Block *createCounterBlock(OpBuilder &b, DCPathPipelineOp pipe,
                                 ValueRange inits, Location loc) {
  SmallVector<Type> argTypes{b.getIndexType()};
  SmallVector<Location> argLocs{loc};
  for (Value in : inits) {
    argTypes.push_back(in.getType());
    argLocs.push_back(loc);
  }
  return b.createBlock(&pipe.getBody(), {}, argTypes, argLocs);
}

// Rewrite an `scf.while` into a while `dcp.pipeline`: `trip` unset, terminated
// by `dcp.condition` carrying the condition value plus the loop-carried
// next-values. Requires identity forwarding: both the before-arg and after-arg
// of a slot map to the same iter-arg.
static void materializeWhilePipeline(const RegionAttrs &r, scf::WhileOp w,
                                     ScheduleModel &model,
                                     const DeviceModel &dev) {
  OpBuilder b(w);
  Location loc = w.getLoc();

  ValueRange inits = w.getInits();
  auto pipe = DCPathPipelineOp::create(
      b, loc, w.getResultTypes(), /*lbBound=*/Value(), /*dynamicBound=*/Value(),
      /*stepBound=*/Value(), inits, /*trip=*/IntegerAttr(),
      /*trip_bound=*/IntegerAttr(), /*lb=*/IntegerAttr(),
      /*step=*/IntegerAttr(), optI64Attr(b, r.ii), optI64Attr(b, r.length),
      optI64Attr(b, r.drain), optI64Attr(b, r.latency),
      r.latencyBound ? b.getUnitAttr() : UnitAttr(), DeterminacyEnumAttr());
  Block *blk = createCounterBlock(b, pipe, inits, loc);

  Block &before = w.getBefore().front();
  Block &after = w.getAfter().front();
  IRMapping map;
  for (unsigned j = 0, n = before.getNumArguments(); j < n; ++j) {
    map.map(before.getArgument(j), blk->getArgument(j + 1));
    map.map(after.getArgument(j), blk->getArgument(j + 1));
  }

  b.setInsertionPointToEnd(blk);
  // A while owns no forwarding (only counted-loop solves record pairs), so its
  // access map is write-only.
  DenseMap<Operation *, Operation *> accessMap;
  for (Operation &op : before.without_terminator())
    convertOp(op, b, map, model, dev, accessMap);
  for (Operation &op : after.without_terminator())
    convertOp(op, b, map, model, dev, accessMap);

  Value cond = map.lookupOrDefault(w.getConditionOp().getCondition());
  SmallVector<Value> carried;
  for (Value v : w.getYieldOp().getOperands())
    carried.push_back(map.lookupOrDefault(v));
  DCPathConditionOp::create(b, loc, cond, carried);

  for (auto [old, nw] : llvm::zip(w.getResults(), pipe.getResults()))
    old.replaceAllUsesWith(nw);
  eraseScheduled(model, w);
}

// Rewrite one counted loop (affine.for or scf.for) into a dcp.pipeline by
// converting its body ops. An already-materialized child region is cloned
// verbatim. The trip count is recorded only when it is a compile-time constant.
static DCPathPipelineOp materializeLoopToPipeline(const RegionAttrs &r,
                                                  LoopLikeOpInterface loop,
                                                  ScheduleModel &model,
                                                  const DeviceModel &dev,
                                                  int64_t &nextFwdId) {
  Operation *loopOp = loop.getOperation();
  OpBuilder b(loopOp);
  Location loc = loop.getLoc();

  ValueRange inits = loop.getInits();
  // A trip that is not a compile-time constant wires the loop's upper bound as
  // the `dynamicBound` operand. Only an scf.for has a runtime ub; affine bounds
  // are constant or affine-symbol.
  Value dynamicBound;
  if (!constantTripOf(loop)) {
    if (auto af = dyn_cast<AffineForOp>(loopOp))
      dynamicBound = scheduledBound(af, /*isLower=*/false);
    else
      dynamicBound = dynamicTripBound(loop);
  }
  // Carry the source loop's lb/step so the induction register runs the real IV.
  // Each rides an attribute when compile-time (elided at the default 0/1), else
  // an operand.
  LoopBounds bounds = lbStepOf(loop);
  // An affine.for with a symbolic lower bound materializes as a runtime lb
  // operand, which lbStepOf defaulted to 0.
  if (auto af = dyn_cast<AffineForOp>(loopOp))
    if (!af.hasConstantLowerBound())
      bounds.lbVal = scheduledBound(af, /*isLower=*/true);
  std::optional<int64_t> lbAttr, stepAttr;
  if (!bounds.lbVal && bounds.lb != 0)
    lbAttr = bounds.lb;
  if (!bounds.stepVal && bounds.step != 1)
    stepAttr = bounds.step;
  // The worst-case count of a loop with no static one
  std::optional<int64_t> trip = constantTripOf(loop);
  std::optional<int64_t> tripBound;
  if (!trip)
    tripBound = model.tripBoundOf(loopOp);
  auto pipe = DCPathPipelineOp::create(
      b, loc, loopOp->getResultTypes(), bounds.lbVal, dynamicBound,
      bounds.stepVal, inits, optI64Attr(b, trip), optI64Attr(b, tripBound),
      optI64Attr(b, lbAttr), optI64Attr(b, stepAttr), optI64Attr(b, r.ii),
      optI64Attr(b, r.length), optI64Attr(b, r.drain), optI64Attr(b, r.latency),
      r.latencyBound ? b.getUnitAttr() : UnitAttr(), DeterminacyEnumAttr());
  Block *blk = createCounterBlock(b, pipe, inits, loc);

  // The induction var is body block argument 0, iter-args follow.
  Block *body = &loop.getLoopRegions().front()->front();
  IRMapping map;
  map.map(body->getArgument(0), blk->getArgument(0));
  // Carry the source IV's NameLoc onto the counter block arg so the datapath
  // emitter can name the iteration-counter wire after the loop variable.
  blk->getArgument(0).setLoc(body->getArgument(0).getLoc());
  for (auto [i, arg] : llvm::enumerate(loop.getRegionIterArgs()))
    map.map(arg, blk->getArgument(i + 1));

  b.setInsertionPointToEnd(blk);
  DenseMap<Operation *, Operation *> accessMap;
  for (Operation &op : body->without_terminator())
    convertOp(op, b, map, model, dev, accessMap);
  stampForwards(model, accessMap, nextFwdId);

  Operation *term = body->getTerminator();
  SmallVector<Value> yields;
  for (Value v : term->getOperands())
    yields.push_back(map.lookupOrDefault(v));
  DCPathUnconditionOp::create(b, term->getLoc(), yields);

  for (auto [old, nw] : llvm::zip(loopOp->getResults(), pipe.getResults()))
    old.replaceAllUsesWith(nw);
  eraseScheduled(model, loopOp);
  return pipe;
}

// Rewrite a straight-line (acyclic) region into a dcp.sequential, with values
// used after the region yielded as its results. A region of only declarations
// is left in place, sourced directly by identity.
static void materializeSequential(const RegionAttrs &r,
                                  ArrayRef<Operation *> ops,
                                  ScheduleModel &model, const DeviceModel &dev,
                                  bool container) {
  SmallVector<Operation *> body;
  for (Operation *op : ops)
    if (!op->hasTrait<OpTrait::IsTerminator>())
      body.push_back(op);

  // In a container, a static `memref.alloc` that a later region reads must stay
  // at func level: a memref is not a datapath value to latch, and the CallUnit
  // path needs the shared buffer identity-sourced.
  llvm::SmallPtrSet<Operation *, 8> inBody(body.begin(), body.end());
  auto escapesBody = [&](Operation *op) {
    return llvm::any_of(op->getResults(), [&](Value res) {
      return llvm::any_of(res.getUsers(),
                          [&](Operation *u) { return !inBody.contains(u); });
    });
  };
  SmallVector<Operation *> work, hoisted;
  for (Operation *op : body) {
    // A literal is not a value to hand across a region boundary: yielded, its
    // consumers read the region's result and the emitter latches it into a
    // survivor register, so a shift by it becomes a barrel shifter where the
    // schedule priced wiring. Left at func level it stays a literal to every
    // consumer, inside the region and out.
    if (isa<arith::ConstantOp>(op) && escapesBody(op))
      hoisted.push_back(op);
    else if (container && isa<memref::AllocOp, memref::AllocaOp>(op) &&
             op->getNumOperands() == 0 && escapesBody(op))
      hoisted.push_back(op); // leave at func level, do not wrap or erase
    else
      work.push_back(op);
  }

  // The predicate the scheduler skips such a span by, so a region is
  // materialized exactly where a solution was solved.
  if (!spanFormsRegion(work))
    return;

  // Move the hoisted ops above the region so they dominate the wrapped uses.
  for (Operation *op : hoisted)
    op->moveBefore(work.front());

  // `spanEscapingValues` selects the same set, so the region's completion waits
  // for exactly what the solve charged.
  llvm::SmallPtrSet<Operation *, 8> inRegion(work.begin(), work.end());
  SmallVector<Value> escaping;
  for (Operation *op : work)
    for (Value res : op->getResults())
      if (llvm::any_of(res.getUsers(),
                       [&](Operation *u) { return !inRegion.contains(u); }))
        escaping.push_back(res);

  OpBuilder b(work.front());
  Location loc = work.front()->getLoc();

  SmallVector<Type> resultTypes(
      llvm::map_range(escaping, [](Value v) { return v.getType(); }));
  auto seq = DCPathSequentialOp::create(
      b, loc, resultTypes, optI64Attr(b, r.length), optI64Attr(b, r.drain),
      optI64Attr(b, r.latency), r.latencyBound ? b.getUnitAttr() : UnitAttr(),
      DeterminacyEnumAttr());
  Block *blk = b.createBlock(&seq.getBody());

  IRMapping map;
  b.setInsertionPointToEnd(blk);
  // A straight-line span owns no forwarding, so its access map is write-only.
  DenseMap<Operation *, Operation *> accessMap;
  for (Operation *op : work)
    convertOp(*op, b, map, model, dev, accessMap);

  SmallVector<Value> yields(llvm::map_range(
      escaping, [&](Value v) { return map.lookupOrDefault(v); }));
  DCPathUnconditionOp::create(b, loc, yields);

  // A boundary value's `volatile` marker is one of the uses this rewires, so
  // its anchor reads the region's result rather than the wrapped computation.
  for (auto [orig, res] : llvm::zip(escaping, seq.getResults()))
    orig.replaceAllUsesWith(res);
  for (Operation *op : llvm::reverse(work))
    eraseScheduled(model, op);
}

// Whether anything in the module targets \p mod. Asked while \p mod is being
// closed, which by post-order is before any of its callers is reified, so a
// caller still spells the edge as a `func.call`.
static bool isCalled(DCPathModuleOp mod) {
  bool called = false;
  mod->getParentOfType<ModuleOp>().walk([&](func::CallOp c) {
    if (c.getCallee() != mod.getSymName())
      return WalkResult::advance();
    called = true;
    return WalkResult::interrupt();
  });
  return called;
}

// Hold the scheduler's `allo.sched.latency` to the span the reify just built.
// The invariant is one-sided: the scheduler's number may be a loose upper
// bound, but an UNDERCOUNT (scheduler < reify) is a miscompile, since a
// consumer placed against it samples before the callee writes.
static void checkLatencyBound(DCPathModuleOp mod, std::optional<int64_t> dcpLat,
                              bool concurrent) {
  auto sched = mod->getAttrOfType<IntegerAttr>(kLatencyAttr);
  if (!sched || !dcpLat)
    return; // either side unknown: the call composes on `done`, not on a time
  // A concurrent container's span is a completion floor over processes paced by
  // back-pressure, not a schedule, so neither number times the other.
  if (concurrent || sched.getInt() == *dcpLat || !isCalled(mod))
    return;
  if (sched.getInt() > *dcpLat) {
    debug(Stage::Dcp, mod) << "Latency bound is loose for callee '"
                           << mod.getSymName() << "': scheduler "
                           << sched.getInt() << ", reify " << *dcpLat << " ("
                           << sched.getInt() - *dcpLat
                           << " cycle(s) a caller waits through)";
    return;
  }
  assert(false && "the scheduler's callee latency undercuts what the reify "
                  "builds; a consumer placed against it samples early");
  error(Stage::Dcp, Code::CompilerInconsistency, mod)
      << "Latency bound is UNSOUND for callee '" << mod.getSymName()
      << "': scheduler " << sched.getInt() << " undercuts reify " << *dcpLat
      << " by " << *dcpLat - sched.getInt()
      << " cycle(s); a consumer time-triggered off this callee would sample "
         "before it writes";
}

// Stamp `latency` and `determinacy` on every reified region, then the
// whole-kernel contract on the `dcp.module` itself. The kernel's span composes
// the top-level regions over their dependence DAG: independent siblings overlap
// (both start at the kernel's own `start`), so the span is the longest path
// through them, not the sum.
static void setDcpLatencies(DCPathModuleOp mod) {
  mod.walk([&](DCPathRegionOpInterface region) {
    RegionTiming t = dcpRegionTiming(region);
    // Total, so a region with no static span is not left carrying whatever
    // `RegionAttrs` guessed before the region existed. An assume-bounded one is
    // the exception: `dcpSpanNode` reads that back as its `assumedSpan`.
    if (t.staticLatency) {
      region.setLatency(static_cast<uint64_t>(*t.staticLatency));
    } else if (t.boundedLatency) {
      region.setLatency(static_cast<uint64_t>(*t.boundedLatency));
      region.setLatencyBound(true);
    } else if (!region.getLatencyBound()) {
      region.setLatency(std::nullopt);
    }
    region.setDeterminacy(t.determinacy);
  });

  // The top-level regions, and the ops each owns for the sibling DAG. Index
  // aligned with `dcpSpanNodes` below, which must select the same ops.
  SmallVector<SmallVector<Operation *>> topOps;
  bool bounded = false;
  for (Operation &op : mod.getBody().front()) {
    assert(!isa<DCPathInstanceOp>(op) &&
           "a call is reified inside a region, so nothing composes a bare "
           "instance at kernel scope");
    if (auto region = dyn_cast<DCPathRegionOpInterface>(&op)) {
      topOps.push_back({&op});
      bounded |= region.getLatencyBound();
    }
  }
  std::optional<int64_t> total =
      composeDag(dcpSpanNodes(mod.getBody().front(), /*topLevel=*/true),
                 siblingPredecessors(topOps));
  bool known = total.has_value();

  // What the children say about themselves, a different question from what the
  // regions holding them compose to.
  {
    bool container = false, allKnown = true, structural = false;
    mod.walk([&](DCPathInstanceOp inv) {
      container = true;
      structural |= spawnsConcurrently(inv);
      if (!inv.getLatency())
        allKnown = false;
    });
    if (container) {
      // A span composes when every child carries a contract and every region
      // has a placement. `bounded` marks the total a ceiling (a guard, an
      // assumed trip, a bounded callee), which a caller may wait out. A
      // concurrent composition's figure is a completion floor, never a bound.
      bool composable = known && allKnown;
      if (composable) {
        mod.setLatency(*total);
        mod.setLatencyBound(!structural && bounded);
        checkLatencyBound(mod, *total, structural);
      }
      // A container holding an `await` spawn or a stream-wired child is
      // `concurrent`; a purely scheduled composition is `counted_static` or
      // `indeterminate`.
      mod.setDeterminacy(structural ? DeterminacyEnum::Concurrent
                         : composable && !bounded
                             ? DeterminacyEnum::CountedStatic
                             : DeterminacyEnum::Indeterminate);
      // Only the kernel's class is stamped: it crosses a module boundary a
      // caller cannot see across. A region's own the emitter derives itself.
      return;
    }
  }

  if (known) {
    mod.setLatency(*total);
    mod.setLatencyBound(bounded);
  }
  checkLatencyBound(mod, total, /*concurrent=*/false);
  // Whole-kernel determinacy, the (latency && !latency_bound) test the op's
  // verifier holds it to.
  mod.setDeterminacy(known && !bounded ? DeterminacyEnum::CountedStatic
                                       : DeterminacyEnum::Indeterminate);
}

// Retire the scheduler's provisional whole-kernel latency once
// `setDcpLatencies` has published the exact one. It is the last thing the
// schedule left on the IR; everything else travels in the `ScheduleModel`.
static void stripScheduleCarrier(DCPathModuleOp mod) {
  mod->removeAttr(kLatencyAttr);
}

namespace {
// Post-order lowering of one function's loop/region tree, mirroring the
// scheduler's own descent so the two stay in lockstep. A counted for-loop
// always becomes a `dcp.pipeline`, whether leaf, co-scheduled level or
// sequential wrapper, and a straight-line span a `dcp.sequential`.
struct Reifier {
  func::FuncOp func;
  ScheduleModel &model;
  const DeviceModel &dev;
  // Set in run(): this func calls sub-kernels, so a shared `memref.alloc` an
  // acyclic span holds is hoisted to func level rather than yielded.
  bool container = false;
  // The next `fwd_id` a forwarded store takes, unique within this func.
  int64_t nextFwdId = 0;

  void materializeBlock(Block &block) {
    for (const SchedRegion &region : enumerateRegions(block))
      materializeRegion(region);
  }

  // One region of a block, by anchor kind. A `while` cannot flushing-pipeline
  // when it nests a loop (per-iteration length is then data-dependent), when
  // it calls a sub-kernel (one instance fired and awaited per iteration), or
  // when its continue-condition is not settled in-cycle.
  // `conditionIsCombinational` is the same predicate the scheduler routes
  // on, kept in lockstep here.
  void materializeRegion(const SchedRegion &region) {
    if (region.kind == allo::RegionKind::StraightLine) {
      materializeSequential(RegionAttrs(model.regionOf(region.ops.front())),
                            region.ops, model, dev, container);
      return;
    }
    Operation *anchor = region.anchor();
    // The boundary marker lives until the materialization below has read it
    // off, and no longer than that.
    auto marker = dyn_cast_or_null<VolatileOp>(anchor->getPrevNode());
    if (isa<AffineForOp, scf::ForOp>(anchor)) {
      materializeCountedLoop(cast<LoopLikeOpInterface>(anchor));
    } else if (auto w = dyn_cast<scf::WhileOp>(anchor)) {
      if (hasNestedLoop(w) || !conditionIsCombinational(w, dev) ||
          blockHasSyncCall(w.getAfter().front())) {
        // A while that cannot flush-pipeline takes the sequential CHECK/RUN
        // controller (`ii` unset). A non-identity-forwarding one is left raw.
        materializeBlock(w.getAfter().front());
        if (whileHasIdentityForwarding(w))
          materializeWhilePipeline(RegionAttrs(), w, model, dev);
      } else {
        // A straight-line while with a combinational condition
        // flushing-pipelines, `ii` from its own solve.
        materializeWhilePipeline(RegionAttrs(model.regionOf(anchor)), w, model,
                                 dev);
      }
    } else if (isa<scf::IfOp, AffineIfOp>(anchor)) {
      // An opaque guard left by if-conversion: materialize each branch, then
      // close the `if` into a dcp.select.
      for (Region &branch : anchor->getRegions())
        if (!branch.empty())
          materializeBlock(branch.front());
      OpBuilder b(anchor);
      // An `scf.if` carries its condition as an operand; an `affine.if` has an
      // integer set, reified beside it and cut against the clock like any other
      // computation.
      Value cond;
      if (auto sif = dyn_cast<scf::IfOp>(anchor)) {
        cond = sif.getCondition();
      } else {
        cond = entryConeOf(anchor).predicate;
        assert(cond && "a guard carries a marker holding its predicate");
      }
      closeIntoDcpSelect(b, anchor, cond);
    }
    if (marker)
      marker->erase();
  }

  // Close a scheduled if, branches already materialized, into a dcp.select with
  // condition \p cond. Latency is left unset, since a data-dependent guard has
  // no static count.
  void closeIntoDcpSelect(OpBuilder &b, Operation *ifOp, Value cond) {
    auto sel = DCPathSelectOp::create(
        b, ifOp->getLoc(), ifOp->getResultTypes(), cond,
        /*latency=*/IntegerAttr(),
        /*latency_bound=*/UnitAttr(), DeterminacyEnumAttr());
    sel.getThenRegion().takeBody(ifOp->getRegion(0));
    if (!ifOp->getRegion(1).empty())
      sel.getElseRegion().takeBody(ifOp->getRegion(1));
    for (Region *r : {&sel.getThenRegion(), &sel.getElseRegion()}) {
      if (r->empty())
        continue;
      Operation *term = r->front().getTerminator();
      OpBuilder yb(term);
      DCPathUnconditionOp::create(yb, term->getLoc(), term->getOperands());
      eraseScheduled(model, term);
    }
    for (auto [oldR, newR] : llvm::zip(ifOp->getResults(), sel.getResults()))
      oldR.replaceAllUsesWith(newR);
    eraseScheduled(model, ifOp);
  }

  // Rewrite a counted for-loop into a dcp.pipeline. The two cases must be
  // distinguished BEFORE the body is materialized, while nested loops are still
  // raw affine/scf ops: a sequential wrapper materializes every sub-region and
  // then wraps with ii = Σ child invocation latency, a leaf innermost wraps
  // directly with the ii of the solve keyed by it.
  void materializeCountedLoop(LoopLikeOpInterface loop) {
    Operation *op = loop.getOperation();
    Block &body = loop.getLoopRegions().front()->front();
    // The scheduler composed this loop's span off the same classification.
    RegionShape shape = countedLoopShape(loop);
    // A counted loop owns a solve exactly when it is a LEAF. A residual wrapper
    // owns none and synthesizes its own from the children it sequences.
    RegionSolution *sol = model.regionOf(op);
    [[maybe_unused]] DCPathPipelineOp pipe;
    if (shape == RegionShape::Container) {
      materializeBlock(body);
      pipe = materializeLoopToPipeline(sequentialWrapperAttrs(loop), loop,
                                       model, dev, nextFwdId);
    } else {
      assert(sol && "a leaf counted loop owns the solve keyed by it");
      pipe = materializeLoopToPipeline(RegionAttrs(sol), loop, model, dev,
                                       nextFwdId);
    }
    // A container whose child spans are all declarations-only builds no child
    // region and comes out a leaf; nothing else may move.
    assert((dcpRegionShape(pipe) == shape ||
            (shape == RegionShape::Container &&
             dcpRegionShape(pipe) == RegionShape::Leaf)) &&
           "the region built disagrees with the shape both composers read");
  }

  // The synthesized timing of a residual sequential wrapper, which owns no
  // solve of its own. Iterations do not overlap, so its II is one body pass.
  // A dynamic INNER trip leaves the pass itself data-dependent, so `ii` and
  // `length` are unset and the wrapper becomes a done-based sequential
  // controller; a dynamic OUTER trip keeps a concrete pass but no static
  // total, so only `latency` is unset.
  RegionAttrs sequentialWrapperAttrs(LoopLikeOpInterface loop) {
    Block &body = loop.getLoopRegions().front()->front();
    RegionAttrs r;
    // The wrapper described before it exists: the same node `dcpSpanNode`
    // reports of it afterwards.
    SpanNode n;
    n.shape = RegionShape::Container;
    n.children = dcpSpanNodes(body, /*topLevel=*/false);
    std::optional<int64_t> pass = composeSequence(n.children);
    if (!pass)
      return r;
    r.ii = *pass;
    r.length = *pass;
    n.trip = constantTripOf(loop);
    if (n.trip) {
      r.latency = composeSpan(n);
      r.latencyBound = r.latency.has_value() && spanHoldsBound(n);
    }
    return r;
  }

  void run() {
    func.walk([&](func::CallOp) {
      container = true;
      return WalkResult::interrupt();
    });
    materializeBlock(func.getBody().front());
  }
};
} // namespace

// The reify's post-condition: every kernel is a `dcp.module`, every loop and
// conditional a `dcp.*` region, and every call a `dcp.instance`, so nothing
// from func/affine/scf may survive. Module-level rather than per-kernel, so a
// kernel that produced NO dcp region at all is still seen. Non-fatal.
static void verifyDcpClosed(ModuleOp module) {
  module.walk([&](Operation *op) {
    if (isa<AffineForOp, scf::ForOp, scf::WhileOp, AffineIfOp, scf::IfOp,
            func::CallOp, func::FuncOp, VolatileOp>(op))
      warn(Stage::Dcp, op)
          << "Op '" << op->getName().getStringRef()
          << "' survived reification; the post-schedule IR should hold only "
             "dcp.module kernels of dcp.* regions and instances, with every "
             "loop, conditional and call closed";
  });
}

//===----------------------------------------------------------------------===//
// Call machinery: rewrite a `func.call` into a `dcp.instance`, the call node
// the leaf datapath models as a CallUnit.
//===----------------------------------------------------------------------===//

// A dcp.instance referencing \p calleeAttr, copying the callee's timing
// contract verbatim (\p at anchors the symbol lookup). Reification is
// post-order over the call graph, so that contract is the exact one the callee
// publishes, never the scheduler's provisional upper bound.
static DCPathInstanceOp makeInvoke(OpBuilder &b, Location loc,
                                   TypeRange resultTypes, ValueRange operands,
                                   FlatSymbolRefAttr calleeAttr, Operation *at,
                                   int64_t start) {
  auto callee = dyn_cast_or_null<DCPathModuleOp>(
      SymbolTable::lookupNearestSymbolFrom(at, calleeAttr));
  assert(callee && "a callee is reified before the caller that composes "
                   "against it, so it is already a dcp.module");
  return DCPathInstanceOp::create(b, loc, resultTypes, operands, calleeAttr,
                                  b.getI64IntegerAttr(start),
                                  optI64Attr(b, callee.getLatency()),
                                  determinacyAttr(b, callee.getDeterminacy()));
}

// Close one reified kernel over the dcp dialect: `func.func` becomes
// `dcp.module` and `func.return` becomes `dcp.output`, so the timing contract
// rides op arguments a verifier reaches. Runs after this kernel's own body is
// reified, so it holds no `func.call` left to invalidate.
static DCPathModuleOp toDcpModule(func::FuncOp func) {
  OpBuilder b(func);
  auto mod = DCPathModuleOp::create(b, func.getLoc(), func.getName(),
                                    func.getFunctionType(),
                                    DeterminacyEnum::Indeterminate);
  // Frontend provenance plus the scheduler's provisional latency, which
  // `setDcpLatencies` holds itself to before `stripScheduleCarrier` drops it.
  for (NamedAttribute a : func->getDiscardableAttrs())
    mod->setAttr(a.getName(), a.getValue());
  mod.setSymVisibilityAttr(func.getSymVisibilityAttr());
  mod.setArgAttrsAttr(func.getArgAttrsAttr());
  mod.setResAttrsAttr(func.getResAttrsAttr());

  mod.getBody().takeBody(func.getBody());
  Operation *ret = mod.getBody().front().getTerminator();
  b.setInsertionPoint(ret);
  DCPathOutputOp::create(b, ret->getLoc(), ret->getOperands());
  ret->erase();
  func.erase();
  return mod;
}

static void materializeFunc(func::FuncOp func, ScheduleModel &model,
                            const DeviceModel &dev) {
  Reifier{func, model, dev}.run();

  // A fully-deferred function still closes into a `dcp.module` so the module is
  // uniform, but publishes no contract: it never went through scheduling.
  bool hasDCP = false;
  func.walk([&](Operation *op) {
    if (isa<DCPathPipelineOp, DCPathSequentialOp>(op)) {
      hasDCP = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  DCPathModuleOp mod = toDcpModule(func);
  if (hasDCP)
    setDcpLatencies(mod);
  stripScheduleCarrier(mod);
}

static void loadDependentDialects(MLIRContext &ctx) {
  ctx.getOrLoadDialect<allo::AlloDialect>();
  ctx.getOrLoadDialect<scf::SCFDialect>();
  ctx.getOrLoadDialect<AffineDialect>();
  ctx.getOrLoadDialect<arith::ArithDialect>();
  ctx.getOrLoadDialect<func::FuncDialect>();
  ctx.getOrLoadDialect<memref::MemRefDialect>();
}

// Post-order over the call graph, so every callee is reified before the
// caller that composes against it (see `makeInvoke`). `done` keys on the
// func's address and never dereferences a closed one; the funcs it looks up
// were all live at once, so no two share an address.
static void reifyCalleesFirst(func::FuncOp func, ScheduleModel &model,
                              const DeviceModel &dev,
                              llvm::DenseSet<Operation *> &done) {
  if (!done.insert(func.getOperation()).second)
    return;
  // An already-reified callee is a `dcp.module`, which this cast skips.
  SmallVector<func::FuncOp> callees;
  func.walk([&](func::CallOp call) {
    if (auto c = dyn_cast_or_null<func::FuncOp>(
            SymbolTable::lookupNearestSymbolFrom(call, call.getCalleeAttr())))
      callees.push_back(c);
  });
  for (func::FuncOp c : callees)
    reifyCalleesFirst(c, model, dev, done);
  materializeFunc(func, model, dev);
}

void mlir::allo::runPostScheduleConversion(ModuleOp module,
                                           ScheduleModel &model) {
  loadDependentDialects(*module->getContext());
  auto dev = DeviceModel::fromModule(module);
  // Reify selects rows the way the schedule did: at the resolved period.
  dev.operators.setSelectionPeriod(model.cycleTimeNs);
  // One `dcp.unit` per allocated instance, declared at the top of the module so
  // the symbols resolve whatever order the funcs below are reified in.
  OpBuilder b(module.getBody(), module.getBody()->begin());
  for (const ScheduleModel::AllocatedUnit &u : model.allocatedUnits())
    DCPathUnitOp::create(b, module.getLoc(), b.getStringAttr(u.name),
                         FlatSymbolRefAttr::get(b.getContext(), u.opType));
  SmallVector<func::FuncOp> funcs(module.getOps<func::FuncOp>());
  llvm::DenseSet<Operation *> reified;
  for (func::FuncOp func : funcs)
    reifyCalleesFirst(func, model, dev, reified);
  verifyDcpClosed(module);
  model.record(module);
}
