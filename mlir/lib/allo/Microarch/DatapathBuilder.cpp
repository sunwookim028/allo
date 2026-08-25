/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Microarch/DatapathBuilder.h"

#include "allo/Microarch/Interface.h" // iface::ModuleInterface (CallUnit ports)
#include "allo/Microarch/Reservation.h" // verifyBinding (MRT legality)

#include "allo/IR/AlloOps.h"
#include "allo/Scheduling/AddressModel.h" // splitAddress (strength reduction)
#include "allo/Scheduling/LatencyModel.h" // siblingPredecessors (sibling order)
#include "allo/Scheduling/MemoryModel.h"  // assignedBankOf (bank decisions)
#include "allo/Scheduling/OperatorLibrary.h" // operatorIdentity
#include "allo/Support/AliasAnalysis.h"      // resolveRoot (storage identity)
#include "allo/Support/Logging.h"            // unmodelled-op diagnostic
#include "allo/Translation/EmitterState.h" // nameFromLoc, sanitizeCppIdentifier

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h" // memref::GetGlobalOp/GlobalOp (ROM)
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h" // m_ConstantInt (counter hull literals)
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Format.h"

#include <algorithm>
#include <deque>
#include <limits>
#include <numeric>

using namespace mlir;
using namespace mlir::allo;
using namespace mlir::allo::logging;

namespace mlir::allo::uarch {

//===----------------------------------------------------------------------===//
// Pure DCP structural readers.
//===----------------------------------------------------------------------===//

Value dcpMemref(Operation *op) {
  if (auto l = dyn_cast<dcp::DCPathLoadOp>(op))
    return l.getMemref();
  if (auto s = dyn_cast<dcp::DCPathStoreOp>(op))
    return s.getMemref();
  return nullptr;
}

// The addressing of a dcp memory access: its affine map plus index operands.
static void dcpAddressing(Operation *op, AffineMap &map,
                          SmallVector<Value> &operands) {
  if (auto l = dyn_cast<dcp::DCPathLoadOp>(op)) {
    map = l.getMap();
    operands.assign(l.getIndices().begin(), l.getIndices().end());
  } else if (auto s = dyn_cast<dcp::DCPathStoreOp>(op)) {
    map = s.getMap();
    operands.assign(s.getIndices().begin(), s.getIndices().end());
  }
}

// The body block of a dcp region op. A guard (dcp.select) reports its `then`
// branch; its else branch is walked separately by every caller that needs it.
static Block *regionBody(Operation *regionOp) {
  if (auto pipe = dyn_cast<dcp::DCPathPipelineOp>(regionOp))
    return &pipe.getBody().front();
  if (auto sel = dyn_cast<dcp::DCPathSelectOp>(regionOp))
    return &sel.getThenRegion().front();
  return &cast<dcp::DCPathSequentialOp>(regionOp).getBody().front();
}

void forEachBodyOp(Operation *regionOp,
                   llvm::function_ref<void(Operation *)> fn) {
  for (Operation &op : regionBody(regionOp)->without_terminator())
    fn(&op);
  // A dual guard's else branch, which `regionBody` does not report.
  if (auto sel = dyn_cast<dcp::DCPathSelectOp>(regionOp))
    if (!sel.getElseRegion().empty())
      for (Operation &op : sel.getElseRegion().front().without_terminator())
        fn(&op);
}

// Trace a pipeline iter-arg (0-based) back to the value assigned to it each
// iteration, appending to \p chain each iter-arg the walk shifts through,
// itself first. The walk stops at the first value that is not such a shift,
// which is where the recurrence reads its datum from; `chain.size()` is then
// the recurrence distance the scheduler solved against. Returns null when the
// shifts form a cycle, so there is no such datum.
static Value traceIterArgSource(dcp::DCPathPipelineOp pipe, unsigned iterArg,
                                SmallVectorImpl<unsigned> &chain) {
  Block &body = pipe.getBody().front();
  auto carried = pipe.getCarriedValues();
  Value v = carried[iterArg];
  chain.push_back(iterArg);
  llvm::SmallDenseSet<unsigned> seen;
  while (auto arg = dyn_cast<BlockArgument>(v)) {
    // Arg 0 is the counter and an arg of another block is an enclosing value;
    // neither shifts, so both end the walk.
    if (arg.getOwner() != &body || arg.getArgNumber() == 0)
      break;
    if (!seen.insert(arg.getArgNumber()).second)
      return {};
    chain.push_back(arg.getArgNumber() - 1);
    v = carried[chain.back()]; // block arg (k+1) -> carried[k]
  }
  return v;
}

// A source valid for the whole of a region's run, so a consumer ties straight
// in and needs no register: a literal, a boundary port, or a survivor an
// earlier region latched before this one started.
static bool isHeld(Source s) {
  return s.kind == Source::Kind::Const || s.kind == Source::Kind::IO ||
         s.kind == Source::Kind::Survivor;
}

// Is \p v a transient FIFO-din value, one that changes while the region is
// back-pressured (`valid & ~ready`), so it must be captured into a
// chain-enable-frozen register before it drives a FIFO write? True iff it is,
// or is a combinational function of, one of the two sources that move under
// back-pressure: a memory load (re-addressed as the counter advances/resets) or
// the loop counter (pipeline block arg 0, reset to `lb` in the drain).
// Everything else is frozen with the datapath while stalled.
static bool isTransientDin(Value v) {
  if (auto barg = dyn_cast<BlockArgument>(v))
    return isa_and_nonnull<dcp::DCPathPipelineOp>(
               barg.getOwner()->getParentOp()) &&
           barg.getArgNumber() == 0;
  auto *def = v.getDefiningOp();
  if (!def)
    return false;
  if (isa<dcp::DCPathLoadOp>(def))
    return true;
  // Stable producers: a FIFO head, a region survivor, a call result, a literal.
  if (isa<StreamGetOp, dcp::DCPathRegionOpInterface, dcp::DCPathInstanceOp,
          arith::ConstantOp>(def))
    return false;
  if (dcpLatency(def) == 0)
    return llvm::any_of(def->getOperands(),
                        [](Value o) { return isTransientDin(o); });
  return false; // a registered (latency>=1) unit's output is frozen under stall
}

//===----------------------------------------------------------------------===//
// Allocation & binding.
//===----------------------------------------------------------------------===//

void DatapathBuilder::collectConstants() {
  func.walk([&](arith::ConstantOp cst) {
    ConstCell c;
    c.id = dp.consts.size();
    c.value = static_cast<Attribute>(cst.getValue());
    c.type = cst.getType();
    producerOf[cst.getResult()] = Source{Source::Kind::Const, c.id, 0};
    dp.consts.push_back(c);
  });
}

Source DatapathBuilder::constant(int64_t v, Type t) {
  ConstCell c;
  c.id = dp.consts.size();
  // Keep the bit pattern at the type's width: an unsigned counter's one-past
  // bound (e.g. 32 in i6) has its top bit set, which the signed-fit check of
  // the plain int64 `IntegerAttr` builder would reject.
  c.value = IntegerAttr::get(
      t, APInt(cast<IntegerType>(t).getWidth(), static_cast<uint64_t>(v),
               /*isSigned=*/true, /*implicitTrunc=*/true));
  c.type = t;
  dp.consts.push_back(c);
  return Source{Source::Kind::Const, c.id, 0};
}

StreamId DatapathBuilder::getOrCreateStream(Value stream, bool isInput) {
  // Key on the storage root for the same reason a memref does: a channel
  // threaded out of a region names different Values at its two ends.
  stream = resolveRoot(stream);
  if (auto it = streamOf.find(stream); it != streamOf.end())
    return it->second;
  StreamId id = dp.streams.size();
  StreamChannel ch;
  ch.id = id;
  ch.stream = stream;
  auto st = cast<StreamType>(stream.getType());
  ch.payload = st.getBaseType();
  ch.depth = static_cast<unsigned>(st.getDepth());
  ch.isInput = isInput;
  // A channel the kernel creates itself needs no port: both its ends are here.
  ch.internal = !isa<BlockArgument>(stream);
  // Initial tokens, when the declaration carries them: what breaks a feedback
  // cycle's start dependence (see `StreamChannel::init`).
  if (auto cr = stream.getDefiningOp<StreamCreateOp>())
    ch.init = cr.getInitAttr();
  dp.streams.push_back(std::move(ch));
  streamOf[stream] = id;
  return id;
}

RegionBlock DatapathBuilder::addRegion(Operation *regionOp, RegionId ridx) {
  regionIdxOf[regionOp] = ridx;

  RegionBlock rb;
  rb.id = ridx;
  rb.op = regionOp;
  // The nearest enclosing region op is the parent, already processed by this
  // pre-order walk; nesting a region makes that parent a container.
  Operation *p = regionOp->getParentOp();
  while (p && !isa<dcp::DCPathRegionOpInterface>(p))
    p = p->getParentOp();
  if (p) {
    unsigned pidx = regionIdxOf.lookup(p);
    rb.parent = pidx;
    dp.regions[pidx].container = true;
    // A guard (dcp.select) splits its children by branch: one nested in the
    // else body is an else-child, otherwise a then-child.
    bool isElse = false;
    if (auto sel = dyn_cast<dcp::DCPathSelectOp>(p)) {
      Operation *o = regionOp;
      while (o->getParentOp() != p)
        o = o->getParentOp();
      isElse = o->getParentRegion() == &sel.getElseRegion();
    }
    (isElse ? dp.regions[pidx].elseChildren : dp.regions[pidx].children)
        .push_back(ridx);
  }

  if (isa<dcp::DCPathSelectOp>(regionOp)) {
    // A predicated container: no counter or trip of its own, it runs its
    // children once iff the predicate holds, so it stays Acyclic.
    rb.guard = true;
  } else if (auto pipe = dyn_cast<dcp::DCPathPipelineOp>(regionOp)) {
    rb.kind = RegionBlock::Kind::Cyclic;
    rb.conditional = pipe.isWhileLoop(); // dcp.condition terminator: flushing
    // The counter block arg keeps the source IV's NameLoc (preserved by the
    // reifier); carry its name so the emitter labels the iteration counter (i).
    if (auto n = nameFromLoc(pipe.getBody().front().getArgument(0).getLoc()))
      rb.counterName = sanitizeCppIdentifier(*n);
    // `ii` is absent for a data-dependent sequential wrapper: such a region has
    // children, so `emitRegion` routes it to a container path that never reads
    // `ii`, and reg-depth paths default to 1.
    if (std::optional<int64_t> ii = pipe.getIi())
      rb.ii = static_cast<unsigned>(*ii);
    if (auto t = pipe.getTripAttr())
      rb.tripCount = t.getInt();
    if (auto t = pipe.getTripBoundAttr())
      rb.tripBound = t.getInt();
    assert(!(rb.tripCount && rb.tripBound) &&
           "an exact trip and a worst-case bound on the same loop");
    // The induction bounds are resolved by `recordRegionBounds`, once the
    // counter width is known and `resolveValue` sees the whole region model.
  } else {
    assert(isa<dcp::DCPathSequentialOp>(regionOp) &&
           "a RegionBlock is a dcp pipeline / sequential / select");
    rb.kind = RegionBlock::Kind::Acyclic;
  }

  // Composition class, DERIVED from the region rather than read back off the
  // attribute the reifier stamps.
  rb.determinacy = dcpRegionTiming(regionOp).determinacy;
  // The one number that IS read back, on purpose: the model's claim about this
  // region's terminal cycle, which `emitRegion` holds the datapath to.
  if (std::optional<uint64_t> d =
          cast<dcp::DCPathRegionOpInterface>(regionOp).getDrain())
    rb.modelledDrain = static_cast<int64_t>(*d);
  return rb;
}

// `dcpRegionShape` is the one answer the emitter's dispatch, the validator's
// legality rules and the latency composer all read. The BUILT model reaches it
// down a different path (linked parent/child edges and bound CallUnits), so the
// assert catches a region op and its built model describing different hardware.
void DatapathBuilder::deriveShapes() {
  for (RegionBlock &rb : dp.regions) {
    rb.shape = dcpRegionShape(rb.op);
    [[maybe_unused]] RegionBlock::Shape modelled =
        rb.guard               ? RegionBlock::Shape::Guard
        : !rb.children.empty() ? RegionBlock::Shape::Container
        : (rb.kind == RegionBlock::Kind::Cyclic && !rb.callUnits.empty())
            ? RegionBlock::Shape::CallNode
            : RegionBlock::Shape::Leaf;
    assert(rb.shape == modelled &&
           "the region op's shape disagrees with the built model's");

    assert((rb.shape != RegionBlock::Shape::Guard || !rb.children.empty()) &&
           "a guard region has no then-branch children to predicate");
    assert((rb.shape != RegionBlock::Shape::CallNode || rb.children.empty()) &&
           "a call-node region sequences an instance, not child regions");
    // The two axes must agree in the direction the composer relies on: a
    // flushing while is always DECLARED conditional. Not a biconditional, since
    // the reifier stamps a `dcp.select` `Conditional` with `conditional` false.
    assert(
        (!rb.conditional || rb.determinacy == DeterminacyEnum::Conditional) &&
        "a while region must be declared conditional");
#ifndef NDEBUG
    // The acyclic boundary family is decided twice: the latency composer from
    // the op's nesting (`dcpSpanNode`) and the controller from the model's
    // parent edge (`!rb.parent`). Hold the two predicates together.
    bool nested = false;
    for (Operation *p = rb.op->getParentOp(); p && !nested;
         p = p->getParentOp())
      nested = isa<dcp::DCPathRegionOpInterface>(p);
    assert(rb.parent.has_value() == nested &&
           "the model's parent edge disagrees with the op's nesting, so the "
           "controller and the latency composer would pick different "
           "boundary families");
#endif
  }
}

void DatapathBuilder::bindCall(dcp::DCPathInstanceOp inv, RegionBlock &rb) {
  auto it = callees.ifaces.find(inv.getCallee());
  assert(it != callees.ifaces.end() &&
         "the callee interface must be registered (emitted bottom-up first)");
  const auto &mi = it->second;

  CallUnit cu;
  cu.id = dp.calls.size();
  cu.invoke = inv;
  cu.region = rb.id;
  cu.callee = inv.getCallee().str();
  cu.latency = inv.getLatency();
  cu.start = static_cast<unsigned>(dcpStart(inv));
  cu.async = inv->hasAttr(kAlloAsyncAttr);
  cu.determinate =
      !cu.async && inv.getDeterminacy() == DeterminacyEnum::CountedStatic;

  // Operands are in callee-argument order, so operand k is callee arg k. Each
  // memref operand contributes one MemArg per child port.
  for (auto [k, operand] : llvm::enumerate(inv.getInputs())) {
    if (isa<StreamType>(operand.getType())) {
      // A channel END: the child handshakes on three ports of its own, recorded
      // against the call/arg pair so the realization (one FIFO per consumer)
      // can wire them without scanning the calls.
      const iface::FIFO *f = mi.streamForArg(static_cast<int>(k));
      assert(f && "a stream operand with no matching callee stream port");
      StreamId sid = getOrCreateStream(operand, f->isInput);
      dp.streams[sid].callEnds.push_back(
          {cu.id, static_cast<unsigned>(cu.streamArgs.size())});
      // Buffering is a throughput hint and deeper is always KPN-safe, so a
      // channel takes the deepest request among its ends.
      dp.streams[sid].depth =
          std::max<unsigned>(dp.streams[sid].depth, f->depth);
      cu.streamArgs.push_back({static_cast<unsigned>(k), sid, f->isInput,
                               static_cast<unsigned>(f->depth), f->base,
                               f->data, f->valid, f->ready});
      continue;
    }
    if (!isa<MemRefType>(operand.getType())) {
      // A scalar operand feeds the child's scalar-input port; its driver is
      // resolved by recordCallScalars, once every region exists.
      const auto *sc = mi.scalarForArg(static_cast<int>(k));
      assert(sc && "a scalar operand with no matching callee scalar port");
      cu.scalarIns.push_back({Source{}, sc->name, sc->width});
      continue;
    }
    auto mem = memIdOf(operand);
    bool isBoundary = isa<BlockArgument>(operand);
    for (const iface::Memory *m : mi.portsForArg(static_cast<int>(k))) {
      CallUnit::MemArg ma;
      ma.calleeArg = static_cast<unsigned>(k);
      ma.mem = mem;
      ma.isBoundary = isBoundary;
      ma.isWrite = m->write;
      // The bank this child port serves: a cyclically partitioned arg exposes
      // one port group per bank, carrying (bank, factor) at the boundary.
      ma.bank = static_cast<unsigned>(m->bank);
      ma.factor = static_cast<unsigned>(m->factor);
      ma.independent = m->independent;
      ma.addr = m->addr;
      ma.data = m->data;
      ma.we = m->we;
      // `ma.topBase` (the boundary port group) is assigned by
      // enumerateBoundaryPorts, once every access is bound.
      cu.memArgs.push_back(std::move(ma));
    }
  }
  // Each scalar result is a Source::Call this region yields: one producerOf
  // entry per result, so recordRegionResults latches each as its own survivor,
  // plus the child's result-output ports for emitCalls.
  for (const iface::Result &r : mi.results)
    cu.resultPorts.push_back(r.name);
  assert(inv.getNumResults() == cu.resultPorts.size() &&
         "an invoke's result count must match the callee's result ports");
  for (auto [k, res] : llvm::enumerate(inv->getResults()))
    producerOf[res] = Source{Source::Kind::Call, cu.id, unsigned(k)};

  rb.callUnits.push_back(cu.id);
  dp.calls.push_back(std::move(cu));
}

void DatapathBuilder::bindStream(Operation *op, RegionBlock &rb) {
  auto get = dyn_cast<StreamGetOp>(op);
  auto sid = getOrCreateStream(get ? get.getStream()
                                   : cast<StreamPutOp>(op).getStream(),
                               /*isInput=*/get != nullptr);
  unsigned aidx = dp.streams[sid].accesses.size();
  StreamChannel::Access acc;
  acc.op = op;
  acc.isPut = !get;
  acc.region = rb.id;
  acc.stage = static_cast<unsigned>(dcpStart(op));
  if (auto attr = op->getAttrOfType<FloatAttr>("in_delay"))
    acc.inDelay = attr.getValueAsDouble();
  dp.streams[sid].accesses.push_back(acc);
  rb.streamAccesses.push_back({sid, aidx});
  // A get produces a token; a put consumes one, and its data driver is
  // resolved in resolveStreamOperands like a store's.
  if (get)
    producerOf[get.getResult()] = Source{Source::Kind::Stream, sid, aidx};
}

void DatapathBuilder::bindMemory(Operation *op, Value memref, RegionBlock &rb) {
  bool isWrite = isa<dcp::DCPathStoreOp>(op);
  auto mid = memIdOf(memref);
  MemUnit &m = dp.mems[mid];
  // A mismatch would time a port against a cycle the consumer's register
  // depth was not solved for; both read the same device table.
  assert(dcpLatency(op) == (isWrite ? m.writeLatency : m.readLatency) &&
         "scheduled access latency disagrees with the device memory model");
  unsigned aidx = m.accesses.size();
  MemUnit::Access acc;
  acc.op = op;
  acc.isWrite = isWrite;
  acc.region = rb.id;
  acc.stage = dcpStart(op);
  if (auto attr = op->getAttrOfType<FloatAttr>("in_delay"))
    acc.inDelay = attr.getValueAsDouble();
  if (auto attr = op->getAttrOfType<FloatAttr>("port_delay"))
    acc.portDelay = attr.getValueAsDouble();
  if (auto attr = op->getAttrOfType<FloatAttr>("select_delay"))
    acc.selectDelay = attr.getValueAsDouble();
  SmallVector<Value> operands;
  dcpAddressing(op, acc.addrMap, operands);
  // One empty slot per index operand, positional and never resized later: the
  // map names them by position.
  acc.addr.assign(operands.size(), Source{});
  // What `assign-banks` decided, split by what it means: a skew resolves a slot
  // that rotates onto a bank at run time, everything else resolves the bank
  // itself.
  std::optional<unsigned> assigned =
      m.numBanks == 1 ? std::optional<unsigned>(0) : assignedBankOf(op);
  (m.layout.skew() ? acc.slot : acc.staticBank) = assigned;
#ifndef NDEBUG
  // Skipped under a skew, where the map names a bank only at run time and there
  // is nothing to compare. Otherwise `bankAddress` builds the offset within
  // this bank out of `addrMap`, so where the map still resolves a bank on its
  // own it has to be the decided one. It often cannot: the decision read the
  // loop steps too.
  if (m.numBanks > 1 && !m.layout.skew()) {
    std::optional<int64_t> derived = staticBankOf(
        m.layout, acc.addrMap, cast<MemRefType>(m.memref.getType()).getShape());
    assert((!acc.staticBank || !derived ||
            *derived == static_cast<int64_t>(*acc.staticBank)) &&
           "the assigned bank is not the one this access's address map "
           "reaches");
  }
#endif
  m.accesses.push_back(std::move(acc));
  rb.memAccesses.push_back({mid, aidx});
  if (!isWrite)
    producerOf[op->getResult(0)] = Source{Source::Kind::Mem, mid, aidx};
  // Forwarding facts, resolved into `MemUnit::forwards` by `recordForwards`
  // once every access of the array is bound.
  if (auto s = dyn_cast<dcp::DCPathStoreOp>(op)) {
    if (std::optional<int64_t> id = s.getFwdId())
      fwdStoreOf[*id] = {mid, aidx};
  } else if (auto l = dyn_cast<dcp::DCPathLoadOp>(op)) {
    if (std::optional<ArrayRef<int64_t>> ids = l.getFwd())
      fwdLoads.push_back(
          {mid, aidx, llvm::SmallVector<int64_t, 1>(ids->begin(), ids->end())});
  }
}

void DatapathBuilder::recordForwards() {
  for (auto &[mid, load, ids] : fwdLoads) {
    MemUnit &m = dp.mems[mid];
    // The scheduler's eligibility gate, restated as build invariants.
    assert(!m.scattered && !m.isRom && !m.skewed &&
           "forwarding was decided for an array whose realization has no RAM "
           "port to shadow");
    assert(m.readLatency >= 1 && m.writeLatency == 1 &&
           "forwarding shadows exactly the one-cycle write window of a "
           "registered read");
    for (int64_t id : ids) {
      auto it = fwdStoreOf.find(id);
      assert(it != fwdStoreOf.end() && it->second.first == mid &&
             "a load's fwd list names a store of its own array");
      assert(m.accesses[it->second.second].region == m.accesses[load].region &&
             "a forwarded pair shares one region");
      m.forwards.push_back({load, it->second.second});
    }
  }
}

void DatapathBuilder::bindCompute(dcp::DCPathComputeOp comp, RegionBlock &rb) {
  FuncUnit u;
  u.id = dp.units.size();
  u.identity = operatorIdentity(comp);
  if (u.identity.comb) {
    // Combinational: emitted inline as a `comb` primitive (latency 0).
    u.latency = 0;
    u.pipelined = true;
  } else {
    // IP: the `dcp.operator` the identity names is the one copy of its stall
    // contract.
    auto opr = SymbolTable::lookupNearestSymbolFrom<dcp::DCPathOperatorOp>(
        comp, comp.getOpTypeAttr());
    assert(opr && "a dcp.compute op_type must reference a live dcp.operator");
    u.latency = static_cast<unsigned>(opr.getLatency());
    u.pipelined = opr.getPipelined();
    u.stall = opr.getStall();
  }
  // The stamped setup delay, the same number the solve was cut against; a
  // delay re-derived from the device here could disagree with it.
  auto inDelay = comp->getAttrOfType<FloatAttr>("in_delay");
  assert(inDelay && "a dcp.compute carries the in_delay the schedule priced");
  u.inDelay = inDelay.getValueAsDouble();
  // The unit's reservation slot: its issue cycle, taken modulo II in a cyclic
  // region since successive iterations overlap there.
  unsigned stage = dcpStart(comp);
  unsigned ii = rb.ii.value_or(1);
  unsigned residue = rb.kind == RegionBlock::Kind::Cyclic ? stage % ii : stage;
  std::optional<double> z;
  if (auto attr = comp->getAttrOfType<FloatAttr>("z"))
    z = attr.getValueAsDouble();
  u.boundOps.push_back({comp, stage, residue, z});
  producerOf[comp.getResult()] = Source{Source::Kind::Unit, u.id, 0};
  dp.opToUnit[comp] = u.id;
  rb.units.push_back(u.id);
  dp.units.push_back(std::move(u));
}

void DatapathBuilder::bindResource(Operation *op, RegionBlock &rb) {
  if (auto inv = dyn_cast<dcp::DCPathInstanceOp>(op))
    return bindCall(inv, rb); // a sub-kernel call -> a CallUnit
  if (isa<StreamGetOp, StreamPutOp>(op))
    return bindStream(op, rb); // a handshaked FIFO access
  if (auto mr = dcpMemref(op))
    return bindMemory(op, mr, rb); // a MemUnit port
  if (auto comp = dyn_cast<dcp::DCPathComputeOp>(op))
    return bindCompute(comp, rb); // a FuncUnit

  // A nested region op is a child region, walked in its own iteration.
  if (isa<dcp::DCPathRegionOpInterface>(op))
    return;
  // Literals are pre-registered as ConstCells (see collectConstants).
  if (isa<arith::ConstantOp>(op))
    return;
  // A declaration binds no resource: the memref / stream it defines is
  // materialized on first access.
  if (isa<memref::AllocOp, memref::AllocaOp, memref::GetGlobalOp,
          StreamCreateOp>(op))
    return;

  unsupported(Stage::Emit, Code::OperationNotModelled, op)
      << "Operation '" << op->getName()
      << "' is not modelled by the datapath, so it would be dropped from the "
         "emitted hardware";
  dp.infeasible = true;
}

void DatapathBuilder::recordRegionResults() {
  for (RegionBlock &rb : dp.regions) {
    Operation *regionOp = rb.op;

    // A guard yields from its two ARMS, not from one body terminator, and its
    // predicate is an explicit operand rather than a body value.
    if (auto sel = dyn_cast<dcp::DCPathSelectOp>(regionOp)) {
      rb.condition = resolveValue(sel.getCondition());
      auto arm = [&](Region &br) {
        SmallVector<Source> rs;
        if (!br.empty())
          for (Value v : br.front().getTerminator()->getOperands())
            rs.push_back(resolveValue(v));
        return rs;
      };
      SmallVector<Source> thenR = arm(sel.getThenRegion());
      SmallVector<Source> elseR = arm(sel.getElseRegion());
      assert((thenR.empty() || thenR.size() == elseR.size()) &&
             "a result-yielding dcp.select needs an else arm of equal arity");
      for (auto [k, then] : llvm::enumerate(thenR))
        rb.results.push_back({then, Source{}, elseR[k]});
      continue;
    }

    // A pipeline's results ARE its loop-carried recurrence: result k is the
    // final value of iter-arg k, and the verifier pairs each init with its
    // carried next 1:1. An unresolvable half stays None to keep the numbering.
    if (auto pipe = dyn_cast<dcp::DCPathPipelineOp>(regionOp)) {
      // A while's continue condition is a scheduled compute producer; a counted
      // loop has none.
      if (rb.conditional)
        rb.condition = resolveValue(pipe.getConditionValue());
      for (auto [init, next] :
           llvm::zip(pipe.getInits(), pipe.getCarriedValues()))
        rb.results.push_back(
            {resolveValue(next), resolveValue(init), Source{}});
      continue;
    }

    // A sequential region: each terminator operand lands exactly once, so there
    // is no recurrence to preload.
    for (Value res : regionBody(regionOp)->getTerminator()->getOperands())
      rb.results.push_back({resolveValue(res), Source{}, Source{}});
  }
}

void DatapathBuilder::recordCallScalars() {
  for (CallUnit &cu : dp.calls) {
    unsigned k = 0;
    for (Value operand : cast<dcp::DCPathInstanceOp>(cu.invoke).getInputs())
      if (!isa<MemRefType, StreamType>(operand.getType()))
        cu.scalarIns[k++].src = resolveValue(operand);
    assert(
        k == cu.scalarIns.size() &&
        "one scalar operand per scalar-input port (bindResource pairs them)");
  }
}

void DatapathBuilder::recordCallDeps() {
  // The MemIds a call touches, by role. Two calls share an array iff MemId
  // identity says so (a MemUnit keys on the storage root).
  auto memsOf = [&](const CallUnit &cu, std::optional<bool> write) {
    SmallVector<MemId, 4> ms;
    for (const CallUnit::MemArg &ma : cu.memArgs)
      if (!write || ma.isWrite == *write)
        ms.push_back(ma.mem);
    return ms;
  };
  auto shares = [](ArrayRef<MemId> a, ArrayRef<MemId> b) {
    return llvm::any_of(a, [&](MemId m) { return llvm::is_contained(b, m); });
  };
  // Two children joined by a CHANNEL, which back-pressure alone can order: they
  // are co-resident and the downstream one drains the queue the upstream one
  // fills, so waiting for the producer deadlocks on a queue shorter than run.
  auto channelled = [](const CallUnit &a, const CallUnit &b) {
    return llvm::any_of(a.streamArgs, [&](const CallUnit::StreamArg &x) {
      return llvm::any_of(b.streamArgs, [&](const CallUnit::StreamArg &y) {
        return x.chan == y.chan;
      });
    });
  };
  for (const RegionBlock &rb : dp.regions) {
    bool concurrent = rb.determinacy == DeterminacyEnum::Concurrent;
    for (auto [i, cid] : llvm::enumerate(rb.callUnits)) {
      CallUnit &cu = dp.calls[cid];
      auto add = [&](CallId p, bool viaResult) {
        for (CallUnit::Pred &e : cu.predecessors)
          if (e.call == p) {
            e.viaResult |= viaResult;
            return;
          }
        cu.predecessors.push_back({p, viaResult});
      };
      SmallVector<MemId, 4> cuMems = memsOf(cu, std::nullopt);
      SmallVector<MemId, 4> cuWrites = memsOf(cu, true);
      for (unsigned j = 0; j < i; ++j) {
        const CallUnit &p = dp.calls[rb.callUnits[j]];
        // Hazard direction (RAW / WAW / WAR) is the ordering in both
        // composition classes: a read-read pair commutes and overlaps, taking
        // a port each (`portGraph`). A concurrent container orders every such
        // pair the channels do not; a scheduled one only where the placement
        // or a missing contract leaves the earlier child's completion ahead.
        bool directed = shares(memsOf(p, true), cuMems) ||
                        shares(cuWrites, memsOf(p, false));
        bool hazard = concurrent
                          ? directed && !channelled(p, cu)
                          : directed && (p.start < cu.start || !p.latency);
        if (hazard)
          add(p.id, /*viaResult=*/false);
      }
      // A child consuming an earlier child's scalar RESULT is ordered after it:
      // the result port only holds from the producer's `done`.
      for (const CallUnit::ScalarArg &sa : cu.scalarIns)
        if (sa.src.kind == Source::Kind::Call)
          add(sa.src.id, /*viaResult=*/true);
    }
  }

  // The release policy, in a second pass so every predecessor edge exists.
  for (const RegionBlock &rb : dp.regions) {
    bool concurrent = rb.determinacy == DeterminacyEnum::Concurrent;
    for (CallId cid : rb.callUnits) {
      CallUnit &cu = dp.calls[cid];
      bool gated = !cu.predecessors.empty();
      // A scheduled composition placed every child at a cycle, so an ungated
      // one is released at its offset and a gated one waits on the join.
      if (!concurrent) {
        cu.startPolicy = gated ? CallUnit::StartPolicy::Handshake
                               : CallUnit::StartPolicy::TimeTriggered;
        continue;
      }
      // A concurrent container has no schedule to release against, so an edge
      // is time-triggerable only where the producer's completion cycle is
      // known: a spawn has no offset at all, a result hand-off holds only from
      // `done`, and an indeterminate producer has no cycle to name.
      bool mustJoin =
          gated && (cu.async ||
                    llvm::any_of(cu.predecessors, [&](const CallUnit::Pred &p) {
                      return p.viaResult || !dp.calls[p.call].determinate;
                    }));
      cu.startPolicy = mustJoin   ? CallUnit::StartPolicy::Handshake
                       : cu.async ? CallUnit::StartPolicy::Broadcast
                                  : CallUnit::StartPolicy::TimeTriggered;
    }
  }
}

Source DatapathBuilder::resolveValue(Value v) {
  // A scheduled producer bound during the region walk: a compute unit, a
  // memory / stream read port, a call result, or a hoisted literal.
  if (auto it = producerOf.find(v); it != producerOf.end())
    return it->second;
  if (auto *def = v.getDefiningOp()) {
    // A nested region's result: the survivor register the producing region
    // latched it into, the ONLY channel a value leaves a region by.
    if (isa<dcp::DCPathRegionOpInterface>(def))
      return Source{Source::Kind::Survivor, regionIdxOf.lookup(def),
                    cast<OpResult>(v).getResultNumber()};
    return {}; // an unmodelled producer
  }
  if (auto it = ioOf.find(v); it != ioOf.end())
    return it->second; // a scalar function argument
  // A `dcp.pipeline` block argument. Arg 0 is the induction counter: its
  // region's counter register, held stable for the whole of a nested run.
  auto barg = cast<BlockArgument>(v);
  auto pipe = dyn_cast<dcp::DCPathPipelineOp>(barg.getOwner()->getParentOp());
  if (!pipe)
    return {};
  assert(regionIdxOf.count(pipe) &&
         "every dcp region op is registered by the region walk");
  RegionId rid = regionIdxOf.lookup(pipe);
  unsigned arg = barg.getArgNumber();
  if (arg == 0)
    return Source{Source::Kind::Counter, rid, 0};
  // The rest are the loop-carried values, readable only where the region
  // latches them into a survivor (`RegionBlock::container`). A childless loop
  // fuses its accumulator in, so only `resolveOperand`'s recurrence edge reads
  // it, and a reader that reaches here gets None and is reported.
  if (!dp.regions[rid].container)
    return {};
  return Source{Source::Kind::Survivor, rid, arg - 1};
}

// Interval bounds of the values a counted counter touches, walked over the
// dcp IR: literal bounds, an enclosing counter, and the reified bound cone.
// Exact in __int128; anything the walk cannot bound keeps `kIndexWidth`.
namespace {
struct ValueHull {
  __int128 lo = 0, hi = 0;
};
struct CounterBounds {
  ValueHull lb, step;
  std::optional<ValueHull> last; // the one-past value `lb + trip*step`
  std::optional<ValueHull> ub;   // a runtime bound's own values
};
} // namespace

static ValueHull hullUnion(ValueHull a, ValueHull b) {
  return {std::min(a.lo, b.lo), std::max(a.hi, b.hi)};
}

/// Floor division at hull width; an arithmetic right shift is one by 2^k.
static __int128 divideFloor(__int128 a, __int128 b) {
  __int128 q = a / b;
  return (a % b != 0 && (a < 0) != (b < 0)) ? q - 1 : q;
}

static bool fitsSigned(const ValueHull &h, unsigned width) {
  __int128 lim = (__int128)1 << (std::min(width, 64u) - 1);
  return h.lo >= -lim && h.hi < lim;
}

static unsigned hullBits(const ValueHull &h) {
  auto bits = [](__int128 v) {
    return static_cast<unsigned>(
        APInt(64, static_cast<uint64_t>((int64_t)v), /*isSigned=*/true)
            .getSignificantBits());
  };
  return std::max(bits(h.lo), bits(h.hi));
}

// The unsigned width of a non-negative hull: the active bits of its top, one
// fewer than `hullBits` gives it, since a value that never goes negative needs
// no sign bit. At least one bit, for a hull pinned to zero.
static unsigned unsignedHullBits(const ValueHull &h) {
  assert(h.lo >= 0 && "an unsigned width for a hull that can go negative");
  return std::max(
      1u, static_cast<unsigned>(
              APInt(64, static_cast<uint64_t>((int64_t)h.hi)).getActiveBits()));
}

static std::optional<CounterBounds> counterBoundsOf(dcp::DCPathPipelineOp pipe,
                                                    unsigned fuel);

static std::optional<ValueHull> hullOfValue(Value v, unsigned fuel) {
  if (!fuel)
    return std::nullopt;
  APInt cst;
  if (matchPattern(v, m_ConstantInt(&cst)))
    return ValueHull{cst.getSExtValue(), cst.getSExtValue()};
  if (auto barg = dyn_cast<BlockArgument>(v)) {
    auto pipe =
        dyn_cast_or_null<dcp::DCPathPipelineOp>(barg.getOwner()->getParentOp());
    if (!pipe || barg.getArgNumber() != 0)
      return std::nullopt;
    auto cb = counterBoundsOf(pipe, fuel - 1);
    if (!cb)
      return std::nullopt;
    // What the body reads: lb through the last issued value, and lb itself
    // even when the loop is empty.
    __int128 hi = cb->last ? cb->last->hi - cb->step.lo : cb->ub->hi - 1;
    return ValueHull{cb->lb.lo, std::max(cb->lb.hi, hi)};
  }
  auto res = dyn_cast<OpResult>(v);
  if (!res)
    return std::nullopt;
  if (auto seq = dyn_cast<dcp::DCPathSequentialOp>(res.getOwner()))
    return hullOfValue(seq.getBody().front().getTerminator()->getOperand(
                           res.getResultNumber()),
                       fuel - 1);
  auto comp = dyn_cast<dcp::DCPathComputeOp>(res.getOwner());
  if (!comp)
    return std::nullopt;
  std::optional<CombOpKindEnum> kind = comp.getCombKind();
  if (!kind)
    return std::nullopt;
  auto of = [&](unsigned i) {
    return hullOfValue(comp.getInputs()[i], fuel - 1);
  };
  // An operand pinned to one value, which the shift and mask rules below need.
  // Read off its hull rather than the op, since a constant reaches an operand
  // through whatever region hoisted it.
  auto literal = [&](unsigned i) -> std::optional<__int128> {
    std::optional<ValueHull> h = of(i);
    if (!h || h->lo != h->hi)
      return std::nullopt;
    return h->lo;
  };
  // A widening cast bounds its result by the source type even where the source
  // value has no hull of its own.
  auto typeHull = [](Value v, bool isSigned) -> std::optional<ValueHull> {
    auto ty = dyn_cast<IntegerType>(v.getType());
    if (!ty || ty.getWidth() > 62)
      return std::nullopt;
    __int128 span = (__int128)1 << (ty.getWidth() - (isSigned ? 1 : 0));
    return isSigned ? ValueHull{-span, span - 1} : ValueHull{0, span - 1};
  };
  std::optional<ValueHull> h;
  switch (*kind) {
  case CombOpKindEnum::IndexCast:
  case CombOpKindEnum::Trunci:
    h = of(0);
    break;
  case CombOpKindEnum::Extsi:
    if (!(h = of(0)))
      h = typeHull(comp.getInputs()[0], /*isSigned=*/true);
    break;
  case CombOpKindEnum::IndexCastUi:
  case CombOpKindEnum::Extui:
    // Zero-extension preserves the value only where it is non-negative.
    if ((h = of(0)) && h->lo < 0)
      h = std::nullopt;
    if (!h)
      h = typeHull(comp.getInputs()[0], /*isSigned=*/false);
    break;
  case CombOpKindEnum::Addi:
  case CombOpKindEnum::Subi: {
    auto a = of(0), b = of(1);
    if (!a || !b)
      return std::nullopt;
    h = *kind == CombOpKindEnum::Addi ? ValueHull{a->lo + b->lo, a->hi + b->hi}
                                      : ValueHull{a->lo - b->hi, a->hi - b->lo};
    break;
  }
  case CombOpKindEnum::Muli: {
    auto a = of(0), b = of(1);
    if (!a || !b)
      return std::nullopt;
    __int128 c[] = {a->lo * b->lo, a->lo * b->hi, a->hi * b->lo, a->hi * b->hi};
    h = ValueHull{*std::min_element(c, c + 4), *std::max_element(c, c + 4)};
    break;
  }
  case CombOpKindEnum::Minsi:
  case CombOpKindEnum::Maxsi:
  case CombOpKindEnum::Minui:
  case CombOpKindEnum::Maxui: {
    auto a = of(0), b = of(1);
    if (!a || !b)
      return std::nullopt;
    // The unsigned orders agree with the signed one only over non-negatives.
    if ((*kind == CombOpKindEnum::Minui || *kind == CombOpKindEnum::Maxui) &&
        (a->lo < 0 || b->lo < 0))
      return std::nullopt;
    bool isMin =
        *kind == CombOpKindEnum::Minsi || *kind == CombOpKindEnum::Minui;
    h = isMin ? ValueHull{std::min(a->lo, b->lo), std::min(a->hi, b->hi)}
              : ValueHull{std::max(a->lo, b->lo), std::max(a->hi, b->hi)};
    break;
  }
  case CombOpKindEnum::Select: {
    // One arm or the other, so the union of the two.
    auto a = of(1), b = of(2);
    if (!a || !b)
      return std::nullopt;
    h = hullUnion(*a, *b);
    break;
  }
  case CombOpKindEnum::Shli:
  case CombOpKindEnum::Shrsi:
  case CombOpKindEnum::Shrui: {
    // A literal shift is a monotone multiply or floor-divide by 2^k; a
    // computed one is not a rule this walk carries.
    std::optional<__int128> k = literal(1);
    auto a = of(0);
    if (!k || *k < 0 || *k > 62 || !a)
      return std::nullopt;
    if (*kind == CombOpKindEnum::Shrui && a->lo < 0)
      return std::nullopt;
    __int128 p = (__int128)1 << *k;
    h = *kind == CombOpKindEnum::Shli
            ? ValueHull{a->lo * p, a->hi * p}
            : ValueHull{divideFloor(a->lo, p), divideFloor(a->hi, p)};
    break;
  }
  case CombOpKindEnum::Andi: {
    // Masking with a non-negative literal lands in [0, mask] whatever the
    // other side holds.
    std::optional<__int128> m = literal(1);
    if (!m || *m < 0)
      return std::nullopt;
    h = ValueHull{0, *m};
    break;
  }
  default:
    return std::nullopt;
  }
  // A result its own type cannot represent wraps in hardware; an index result
  // is materialized at the datapath index width.
  if (!h)
    return std::nullopt;
  unsigned width = isa<IndexType>(res.getType())
                       ? kIndexWidth
                       : cast<IntegerType>(res.getType()).getWidth();
  if (!fitsSigned(*h, width))
    return std::nullopt;
  return h;
}

static std::optional<CounterBounds> counterBoundsOf(dcp::DCPathPipelineOp pipe,
                                                    unsigned fuel) {
  if (!fuel || pipe.isWhileLoop())
    return std::nullopt;
  auto bound = [&](std::optional<int64_t> attr, Value v,
                   int64_t dflt) -> std::optional<ValueHull> {
    if (attr)
      return ValueHull{*attr, *attr};
    if (v)
      return hullOfValue(v, fuel);
    return ValueHull{dflt, dflt};
  };
  auto lb = bound(pipe.getLb(), pipe.getLbBound(), 0);
  auto step = bound(pipe.getStep(), pipe.getStepBound(), 1);
  if (!lb || !step || step->lo < 1)
    return std::nullopt;
  CounterBounds cb{*lb, *step, {}, {}};
  std::optional<int64_t> trip =
      pipe.getTrip() ? pipe.getTrip() : pipe.getTripBound();
  if (trip && *trip >= 0)
    cb.last = ValueHull{cb.lb.lo + (__int128)*trip * cb.step.lo,
                        cb.lb.hi + (__int128)*trip * cb.step.hi};
  // A materialized runtime bound needs its own hull only when no trip covers
  // it: `trip` iterations put `ub` at most `lb + trip*step`.
  if (Value ub = pipe.getDynamicBound()) {
    cb.ub = hullOfValue(ub, fuel);
    if (!cb.ub && !cb.last)
      return std::nullopt;
  }
  if (!cb.last && !cb.ub)
    return std::nullopt;
  return cb;
}

// Every value the counter register and its terminator touch: `lb` and `step`
// (each its own cell), the iv up to the one-past value, the `iv + step`
// compare operand, and a materialized `ub`.
static ValueHull storageHullOf(const CounterBounds &cb) {
  ValueHull h = hullUnion(cb.lb, cb.step);
  h = hullUnion(h, {cb.lb.lo + cb.step.lo, cb.lb.hi + cb.step.hi});
  if (cb.last)
    h = hullUnion(h, *cb.last);
  if (cb.ub)
    h = hullUnion(h, {cb.ub->lo, cb.ub->hi + cb.step.hi - 1});
  return h;
}

void DatapathBuilder::deriveCounterTypes() {
  constexpr unsigned kHullFuel = 16;
  for (RegionBlock &rb : dp.regions) {
    if (rb.kind != RegionBlock::Kind::Cyclic)
      continue;
    unsigned width = kIndexWidth;
    auto pipe = cast<dcp::DCPathPipelineOp>(rb.op);
    if (!rb.conditional) {
      if (auto cb = counterBoundsOf(pipe, kHullFuel)) {
        ValueHull h = storageHullOf(*cb);
        // A non-negative counter drops its sign bit: an unsigned register and
        // unsigned predicates hold the same range in one fewer bit.
        rb.counterUnsigned = h.lo >= 0;
        unsigned bits = rb.counterUnsigned ? unsignedHullBits(h) : hullBits(h);
        if (fitsSigned(h, 64) && bits < kIndexWidth) {
          width = bits;
          // A runtime bound narrowed below `kIndexWidth` publishes its hull,
          // so the recurrence gates can size `lb + n*step` against it. A
          // literal-bound counter keeps in-range gates by construction.
          if (pipe.getLbBound() || pipe.getStepBound() ||
              pipe.getDynamicBound()) {
            rb.counterHull = {{(int64_t)h.lo, (int64_t)h.hi}};
            rb.counterStepHi = (int64_t)cb->step.hi;
          }
        } else {
          rb.counterUnsigned = false; // the 32-bit fallback stays signed
        }
      }
    }
    rb.counterType = IntegerType::get(func.getContext(), width);
  }
}

void DatapathBuilder::recordRegionBounds() {
  // A runtime induction bound (ub / lb / step) crosses the same F->G channel a
  // data survivor does; an unresolvable one is reported, not silently run.
  auto recordBound = [&](Operation *pipe, Value b, Source &into) {
    if (!b)
      return;
    into = resolveValue(b);
    if (!into) {
      unsupported(Stage::Emit, Code::CrossRegionHandOff, pipe)
          << "Loop bound is produced by a value this region cannot read; such "
             "a cross-region value hand-off is not lowered yet";
      dp.infeasible = true;
    }
  };
  for (RegionBlock &rb : dp.regions)
    if (auto pipe = dyn_cast<dcp::DCPathPipelineOp>(rb.op)) {
      recordBound(rb.op, pipe.getDynamicBound(), rb.ubSource);
      recordBound(rb.op, pipe.getLbBound(), rb.lbSource);
      recordBound(rb.op, pipe.getStepBound(), rb.stepSource);
      // A compile-time bound ties in as a literal cell. The ub is derivable
      // only when lb and step are literal too; otherwise `tripCount` carries
      // `lb + trip*step` to `terminatorOf`, since no cell can hold arithmetic.
      int64_t lb = pipe.getLb().value_or(0), step = pipe.getStep().value_or(1);
      if (!rb.lbSource)
        rb.lbSource = constant(lb, rb.counterType);
      if (!rb.stepSource)
        rb.stepSource = constant(step, rb.counterType);
      if (!rb.ubSource && rb.tripCount && !pipe.getLbBound() &&
          !pipe.getStepBound())
        rb.ubSource = constant(lb + *rb.tripCount * step, rb.counterType);
    }
}

void DatapathBuilder::bindIOArgs() {
  for (BlockArgument arg : func.getArguments()) {
    if (isa<MemRefType>(arg.getType()))
      continue;
    // A stream arg is a FIFO channel, created lazily on its first get/put.
    if (isa<StreamType>(arg.getType()))
      continue;
    IOPort io;
    io.id = dp.ios.size();
    io.value = arg;
    io.type = arg.getType();
    ioOf[arg] = Source{Source::Kind::IO, io.id, 0};
    dp.ios.push_back(io);
  }
}

void DatapathBuilder::recordResults() {
  auto ret = cast<dcp::DCPathOutputOp>(func.getBody().front().getTerminator());
  for (auto [i, v] : llvm::enumerate(ret.getOperands())) {
    assert(!isa<MemRefType>(v.getType()) &&
           "a memref result should be an out-param by emit "
           "(buffer-results-to-out-params)");
    Result r;
    // An unresolvable result Source is swept by `validateDatapath`, so the
    // build finishes and the diagnostic is raised once, in one place.
    r.source = resolveValue(v); // survivor / passthrough IO / constant
    r.type = v.getType();
    r.name = resultPortName(i, ret.getNumOperands());
    dp.results.push_back(std::move(r));
  }
}

//===----------------------------------------------------------------------===//
// Interconnect derivation.
//===----------------------------------------------------------------------===//

Resolved DatapathBuilder::resolveOperand(Value v, Operation *consumer,
                                         unsigned ii, bool addressSlot) {
  int64_t tY = dcpStart(consumer);
  Operation *regionOp = consumer->getParentOp();

  // Register depth for an edge whose producer's result is ready at `ready`
  // (cycles after its issuing pulse): distance-many II turns plus the
  // consumer's cycle, minus the ready cycle.
  auto edge = [&](Source base, Value key, unsigned ready,
                  unsigned distance) -> Resolved {
    int64_t depth =
        static_cast<int64_t>(distance) * ii + tY - static_cast<int64_t>(ready);
    // The scheduler must never place a consumer before its operand is ready.
    // Asserting alone is not enough: the `unsigned` cast below would wrap, so a
    // release build reports, clamps to 0, and fails in `validateDatapath`.
    if (depth < 0) {
      assert(false && "the scheduler placed a consumer before its operand is "
                      "ready; the register depth would wrap");
      error(Stage::Emit, Code::CompilerInconsistency, consumer)
          << "Infeasible schedule; the operand is not ready until cycle "
          << (static_cast<int64_t>(ready) - static_cast<int64_t>(distance) * ii)
          << " but its consumer is scheduled at cycle " << tY
          << " (producer ready " << ready << ", dependence distance "
          << distance << ", II " << ii << ")";
      dp.infeasible = true;
      depth = 0;
    }
    return {base, key, static_cast<unsigned>(depth), ready};
  };

  // The one operand that does not read `v` at all: an unlatched iter_arg of the
  // consumer's own region is the loop recurrence, so the edge runs back to
  // whatever the loop assigns it, `distance` iterations away.
  unsigned ridx = regionIdxOf.lookup(regionOp);
  auto barg = dyn_cast<BlockArgument>(v);
  auto pipe =
      barg ? dyn_cast<dcp::DCPathPipelineOp>(barg.getOwner()->getParentOp())
           : dcp::DCPathPipelineOp();
  if (barg && pipe == regionOp && barg.getArgNumber() >= 1 &&
      !dp.regions[ridx].container) {
    unsigned iterArg = barg.getArgNumber() - 1;
    SmallVector<unsigned, 2> chain;
    Value next = traceIterArgSource(pipe, iterArg, chain);
    unsigned distance = chain.size();
    Source base = next ? resolveValue(next) : Source{};
    Operation *def = next ? next.getDefiningOp() : nullptr;
    Resolved r;
    // A held `next` never moves while this loop runs, so from iteration
    // `distance` on the recurrence reads it unchanged, off a wire. Anything
    // else must be produced by an op this region schedules, since only then
    // is there an iteration of it to reach back to.
    if (isHeld(base))
      r = {base, Value(), 0, 0};
    else if (base && def && def->getParentOp() == regionOp) {
      r = edge(base, next, readyCycleOf(def), distance);
      // The one edge whose delay counts cycles between two iterations, so
      // this region may not defer an issue on its own.
      dp.regions[ridx].cycleIndexedState = true;
    } else {
      // Anchored on the loop, where the faulty carried assignment is, rather
      // than on the consumer `validateDatapath` would anchor on.
      unsupported(Stage::Emit, Code::CrossRegionHandOff, pipe)
          << "Value " << iterArg
          << " carried by this loop is assigned from a value the loop's own "
             "datapath cannot read; such a cross-region value hand-off is "
             "not lowered yet";
      dp.infeasible = true;
      return {};
    }
    // The emitter re-injects the identities on this consumer input, since the
    // recurrence register may sit elsewhere in the cycle. One per iteration
    // below `distance`: a chained carry shifts one iter_arg into the next, so
    // iteration n reads the init of the stage n steps down the chain, which
    // is what `chain` enumerates.
    for (unsigned stage : chain)
      r.inits.push_back(resolveValue(pipe.getInits()[stage]));
    // An unresolvable init leaves the accumulator to free-run from reset.
    // Only this site knows an init was expected; None is normal elsewhere.
    if (!llvm::all_of(r.inits, [](Source s) { return bool(s); })) {
      unsupported(Stage::Emit, Code::CrossRegionHandOff, pipe)
          << "Loop-carried accumulator has an initial value this region "
             "cannot read; such a cross-region value hand-off is not "
             "lowered yet";
      dp.infeasible = true;
      r.inits.clear();
    }
    return r;
  }

  Source base = resolveValue(v);
  if (!base)
    return {};
  if (isHeld(base))
    return {base, Value(), 0, 0};
  // The region's own counter presents its index at cycle 0 of the iteration,
  // so a consumer scheduled at tY delays it that far. An enclosing region's
  // counter advances only between passes, so a nested consumer reads it live;
  // an address slot keeps the edge (see the declaration).
  if (base.kind == Source::Kind::Counter) {
    if (!addressSlot && base.id != ridx)
      return {base, Value(), 0, 0};
    return edge(base, v, /*ready=*/0, /*distance=*/0);
  }
  // A scheduled producer: readable only from the region it issues in, and only
  // after it lands.
  Operation *def = v.getDefiningOp();
  assert(def && "a scheduled Source is produced by an op");
  if (def->getParentOp() != regionOp)
    return {}; // cross-region hand-off unsupported
  return edge(base, v, readyCycleOf(def), /*distance=*/0);
}

void DatapathBuilder::resolveEdges() {
  // One empty slot per operand port, sized before anything fills them: a
  // recorded edge holds a pointer into these vectors, so they may not grow
  // once resolution starts. A memory access is sized where it is bound.
  for (FuncUnit &u : dp.units) {
    unsigned n = u.repOp()->getNumOperands();
    u.inputs.assign(n, Source{});
    u.inputInits.resize(n); // parallel; filled for recurrence inputs below
  }
  resolveUnitInputs();
  resolveMemoryOperands();
  resolveStreamOperands();
  // `edges` keys are pointers into these containers; a pass that grows one
  // after this point dangles the whole edge table.
  unitsBase = dp.units.data();
  memsBase = dp.mems.data();
  streamsBase = dp.streams.data();
}

void DatapathBuilder::recordEdge(const Resolved &r, Source &slot,
                                 unsigned regionIdx) {
  if (!r.base)
    return;
  if (r.depth == 0) {
    slot = r.base;
    return;
  }
  Edge e{r.base, RegKey{r.key, regionIdx}, r.depth, r.ready};
  bool isNew = edges.insert({&slot, e}).second;
  assert(isNew && "an input slot reads one driver, so it takes one edge");
  (void)isNew;
}

void DatapathBuilder::recordCarriedEdge(const Resolved &r, Value operand,
                                        Operation *consumer, Source &slot,
                                        unsigned regionIdx) {
  if (r.inits.empty()) {
    recordEdge(r, slot, regionIdx);
    return;
  }
  // Arms of one issue pulse, split by iteration: identity n at iteration n and
  // the edge itself from `inits.size()` on, the same shape a shared unit port
  // carries. Every arm is sized before any is filled, since `recordEdge` takes
  // a pointer into `sources`.
  unsigned arms = r.inits.size() + 1;
  muxBuilds.push_back({&slot,
                       regionIdx,
                       operand.getType(),
                       SmallVector<Operation *, 2>(arms, consumer),
                       {},
                       {}});
  MuxBuild &mb = muxBuilds.back();
  for (unsigned n = 0; n + 1 < arms; ++n)
    mb.phases.push_back({Mux::Phase::At, n});
  mb.phases.push_back({Mux::Phase::From, arms - 1});
  mb.sources.assign(r.inits.begin(), r.inits.end()); // held: literal/port/...
  mb.sources.resize(arms);
  recordEdge(r, mb.sources.back(), regionIdx);
}

void DatapathBuilder::resolveUnitInputs() {
  for (FuncUnit &u : dp.units) {
    Operation *op0 = u.repOp();
    unsigned ridx = regionIdxOf.lookup(op0->getParentOp());
    unsigned ii = dp.regions[ridx].ii.value_or(1);
    unsigned nPorts = op0->getNumOperands();
    if (u.boundOps.size() == 1) {
      for (unsigned k = 0; k < nPorts; ++k) {
        auto r = resolveOperand(op0->getOperand(k), op0, ii);
        recordEdge(r, u.inputs[k], ridx);
        u.inputInits[k] =
            r.inits; // empty unless k reads a loop-carried iter_arg
      }
      continue;
    }
    // Shared unit: resolve every bound op's port k independently (each may need
    // its own register depth), then a mux picks per op's issue cycle. All
    // resolved up front because a recurrence operand takes an arm per identity
    // and `recordEdge` takes a pointer into `sources`, so the list must be
    // sized before any edge lands in it.
    for (unsigned k = 0; k < nPorts; ++k) {
      SmallVector<Resolved, 2> edges;
      unsigned arms = 0;
      for (const FuncUnit::BoundOp &bo : u.boundOps) {
        edges.push_back(resolveOperand(bo.op->getOperand(k), bo.op, ii));
        arms += 1 + edges.back().inits.size();
      }
      muxBuilds.push_back(
          {&u.inputs[k], ridx, op0->getOperand(k).getType(), {}, {}, {}});
      MuxBuild &mb = muxBuilds.back();
      mb.sources.resize(arms);
      unsigned arm = 0;
      for (auto [j, r] : llvm::enumerate(edges)) {
        Operation *opj = u.boundOps[j].op;
        // Each identity rides an arm of its own rather than a mux in front of
        // the port: a shared port carries a different op's operand each cycle,
        // leaving no cycle to time such a mux against. An unresolvable init is
        // reported by `resolveOperand` and takes no arm.
        for (auto [n, init] : llvm::enumerate(r.inits)) {
          mb.ops.push_back(opj);
          mb.phases.push_back({Mux::Phase::At, static_cast<unsigned>(n)});
          mb.sources[arm++] = init; // held: a literal, a port, a survivor
        }
        mb.ops.push_back(opj);
        mb.phases.push_back(r.inits.empty() ? Mux::Phase{}
                                            : Mux::Phase{Mux::Phase::From,
                                                         static_cast<unsigned>(
                                                             r.inits.size())});
        recordEdge(r, mb.sources[arm++], ridx);
      }
    }
  }
}

// Floor-based residue, which is what an affine `mod` means: non-negative for a
// positive divisor whatever the sign of \p a, so a digit register starts in
// range and its unsigned wrap compare is exact.
static int64_t mod(int64_t a, int64_t b) {
  return a - llvm::divideFloorSigned(a, b) * b;
}

// The width a stride register is built at: enough bits for every value it
// holds, and for the raw pre-wrap sum its update compares before fixing.
// A WRAPPING register lives in `[0, wrap)`; `raw = cur + step + bump` reaches
// `2*wrap - 1` going up (`step + bump <= wrap` by construction) or borrows
// from just below zero going down, same headroom either way under the
// unsigned compare. A PLAIN accumulator runs from `init` over the loop's
// advances, one past the last iteration since a counted controller still
// computes the step it does not take.
static unsigned strideWidth(const RegionBlock::AddrStride &s,
                            std::optional<int64_t> trip) {
  auto bits = [](uint64_t v) {
    return std::min(kIndexWidth, std::max(1u, APInt(64, v).getActiveBits()));
  };
  if (s.wrap) {
    assert(s.wrap > 0 && "a wrap is a modulus, and the update compares against "
                         "it unsigned");
    return bits(2 * static_cast<uint64_t>(s.wrap) - 1);
  }
  int64_t span, last;
  if (!trip || llvm::MulOverflow(s.step + s.bump, *trip, span) ||
      llvm::AddOverflow(s.init, span, last) || s.init < 0 || last < 0)
    return kIndexWidth;
  return bits(std::max(s.init, last));
}

// The slot in \p rb holding \p want, appended if no identical stride is there.
// The width is DERIVED from the rest, so it takes no part in the comparison.
static unsigned slotFor(RegionBlock &rb, RegionBlock::AddrStride want) {
  want.width = strideWidth(want, rb.tripCount ? rb.tripCount : rb.tripBound);
  auto *it =
      llvm::find_if(rb.addrStrides, [&](const RegionBlock::AddrStride &a) {
        return a.init == want.init && a.step == want.step &&
               a.bump == want.bump && a.wrap == want.wrap &&
               a.down == want.down && a.hasCarry == want.hasCarry &&
               (!a.hasCarry || a.carry == want.carry);
      });
  if (it == rb.addrStrides.end()) {
    rb.addrStrides.push_back(want);
    it = std::prev(rb.addrStrides.end());
  }
  return static_cast<unsigned>(it - rb.addrStrides.begin());
}

// The register holding `t.coeff * digit` over region \p rid's counter, plus the
// companion residue register a quotient digit carries off.
//
// \p base is absorbed by the first NON-WRAPPING register and zeroed, which
// avoids an extra adder on the port's setup path. A wrapping register cannot
// take it: it holds a residue whose wrap assumes it stays in range.
static MemUnit::Access::ScaledTerm strideFor(Datapath &dp, unsigned rid,
                                             const SplitAddress::Term &t,
                                             int64_t &base) {
  RegionBlock &rb = dp.regions[rid];
  // The digit's argument, `scale * counter + offset`: where it starts and what
  // it advances by, which is all the register needs. Running backwards, every
  // wrap becomes a borrow and every carry a decrement.
  int64_t start = t.scale * *dp.constantOf(rb.lbSource) + t.offset;
  int64_t advance = t.scale * *dp.constantOf(rb.stepSource);
  bool down = advance < 0;
  RegionBlock::AddrStride want;
  if (!t.isDigit()) {
    want = {t.coeff * start + base, t.coeff * advance};
    base = 0;
  } else if (t.divisor == 1) {
    // A pure residue accumulates and wraps on itself.
    want = {t.coeff * mod(start, t.modulus),
            t.coeff * advance,
            0,
            t.coeff * t.modulus,
            0,
            false,
            down};
  } else {
    // A quotient advances by one wherever its argument crosses a multiple of
    // the divisor, which is what the companion residue register says. Unscaled,
    // unreferenced by any access, and shared by every digit over that argument.
    unsigned carry = slotFor(
        rb, {mod(start, t.divisor), advance, 0, t.divisor, 0, false, down});
    int64_t q = llvm::divideFloorSigned(start, t.divisor);
    want = {t.coeff * (t.modulus ? mod(q, t.modulus) : q),
            0,
            down ? -t.coeff : t.coeff,
            t.modulus ? t.coeff * t.modulus : 0,
            carry,
            true,
            down};
  }
  return {rid, slotFor(rb, want)};
}

// Merge the terms landing on the same DIGIT of the same region, \p region
// giving each operand position the region whose counter it follows. A region
// has one counter, so those terms add their coefficients rather than taking a
// register and an adder each.
static SmallVector<SplitAddress::Term>
mergeTermsByDigit(ArrayRef<SplitAddress::Term> terms,
                  ArrayRef<std::optional<unsigned>> region) {
  using Digit = std::tuple<unsigned, int64_t, int64_t, int64_t, int64_t>;
  llvm::MapVector<Digit, unsigned> group;
  SmallVector<SplitAddress::Term> merged;
  for (const SplitAddress::Term &t : terms) {
    Digit d{*region[t.operand], t.scale, t.offset, t.divisor, t.modulus};
    auto [it, isNew] = group.try_emplace(d, merged.size());
    if (isNew)
      merged.push_back(t);
    else
      merged[it->second].coeff += t.coeff;
  }
  llvm::erase_if(merged, [](const SplitAddress::Term &t) { return !t.coeff; });
  return merged;
}

// Reduce ONE cone of an address. The in-bank offset and the bank digit are the
// same kind of expression over the same operands: a bank under a cyclic
// partition is `counter mod F`, a wrap register like a delinearized subscript.
static MemUnit::Access::Reduced
reduceCone(Datapath &dp, AffineExpr e, AffineMap addrMap,
           ArrayRef<std::optional<unsigned>> region) {
  MemUnit::Access::Reduced out;
  if (!e)
    return out;
  SplitAddress sp =
      splitAddress(e, addrMap.getNumDims(), addrMap.getNumSymbols(),
                   [&](unsigned p) -> std::optional<int64_t> {
                     if (!region[p])
                       return std::nullopt;
                     return dp.constantOf(dp.regions[*region[p]].stepSource);
                   });
  int64_t base = sp.base;
  for (const SplitAddress::Term &t : mergeTermsByDigit(sp.terms, region))
    out.terms.push_back(strideFor(dp, *region[t.operand], t, base));
  // The digits the residual reads, IN ORDER and undeduplicated: it names them
  // by position, and the scheduler priced the same list from the same
  // `splitAddress`, so the two cannot disagree about which is which.
  for (const SplitAddress::Term &t : sp.reads)
    out.reads.push_back(strideFor(dp, *region[t.operand], t, base));
  out.base = base; // 0 unless no register took it
  out.residual = sp.residual;
  return out;
}

// What `buildAddr` builds after the address delay register: the residual cone
// plus the adder joining it to the register the term sum landed in. Priced by
// the same function as the whole cone, that register standing in as the one
// input the sum became.
static double postRegisterDelay(const MemUnit::Access::Reduced &r,
                                const AddressDelays &delays, unsigned width) {
  if (!r.residual)
    return 0.0;
  SplitAddress sp;
  if (r.base || !r.terms.empty())
    sp.terms.push_back({});
  sp.residual = r.residual;
  return splitAddressCost(sp, delays, width).delay;
}

// The whole cone as built, before any delay register: each reduced term
// arrives from its stride register and the residual runs beside the summing
// chain.
static double builtConeDelay(const MemUnit::Access::Reduced &r,
                             const AddressDelays &delays, unsigned width) {
  SplitAddress sp;
  sp.terms.resize(r.terms.size());
  sp.residual = r.residual;
  return splitAddressCost(sp, delays, width).delay;
}

// Address strength reduction: decide which TERMS of each access's address can
// come from registers that advance with the loop counters, and record the
// scaled counters those registers need. A term that does not qualify stays in
// the residual, so this only ever removes arithmetic.
//
// Runs after `resolveEdges` (a term has to resolve to a counter) and after
// `recordRegionBounds` (a stride is a constant only if the counter's bounds
// are), and before any chain is built, so a term it reduces costs no register.
// `splitAddress` is the same decomposition the scheduler priced the access
// with.
//
// An operand still owed a delay is read off its edge: the scaled counter is
// delayed once for the whole sum rather than per operand, so counters wanted at
// different cycles cannot share that one delay; the first one's cycle decides
// and the rest stay in the residual.
//
// The split runs on the IN-BANK OFFSET (the flat index for an unbanked
// memref), which is what lets a banked access reduce at all: `buf[i, 4*j]`
// under a cyclic-4 last axis offsets by `i*extent + j`, as linear as any.
//
// Both cones are derived symbolically (`addressExprsOf`) here and only later
// evaluated: composing the row-major strides on a coalesced nest's
// `iv -> (iv floordiv N, iv mod N)` cancels back to `iv`, which the same
// expression built out of `comb` ops cannot.
void DatapathBuilder::planAddressGenerators() {
  assert(dp.units.data() == unitsBase && dp.mems.data() == memsBase &&
         dp.streams.data() == streamsBase &&
         "a cell container grew after resolveEdges; the edge table's slot "
         "pointers are dangling");
  AddressDelays delays = addressDelaysOf(dev.operators);
  for (MemUnit &m : dp.mems) {
    auto shape = cast<MemRefType>(m.memref.getType()).getShape();
    for (MemUnit::Access &acc : m.accesses) {
      // Which operands a register can follow, decided up front so the predicate
      // handed to `splitAddress` is a pure one.
      SmallVector<std::optional<unsigned>> region(acc.addr.size());
      std::optional<unsigned> delay;
      for (unsigned p = 0, n = acc.addr.size(); p < n; ++p) {
        Source s = acc.addr[p];
        unsigned d = 0;
        if (auto *it = edges.find(&acc.addr[p]); it != edges.end()) {
          s = it->second.base;
          d = it->second.depth;
        }
        if (s.kind != Source::Kind::Counter || (delay && *delay != d))
          continue;
        RegionBlock &rb = dp.regions[s.id];
        if (!dp.constantOf(rb.lbSource) || !dp.constantOf(rb.stepSource))
          continue;
        delay = d;
        region[p] = s.id;
      }
      AddressExprs e =
          addressExprsOf(m.layout, acc.addrMap, shape, acc.staticBank);
      // The width the cone is priced at against the width the emitter builds
      // the bank's address port at: the first comes off the layout's bank
      // shape, the second off `depthWords`.
      assert(e.width == addressWidthOf(static_cast<int64_t>(m.depthWords)) &&
             "the width the address was priced at is not the one it is built "
             "at");
      acc.hasBankCone = static_cast<bool>(e.bank);
      acc.offset = reduceCone(dp, e.offset, acc.addrMap, region);
      acc.bank = reduceCone(dp, e.bank, acc.addrMap, region);
      // Both cones read the same operands, so one delay covers them, and a
      // digit the residual reads is a register like any other.
      bool anyRegister = !acc.offset.terms.empty() || !acc.bank.terms.empty() ||
                         !acc.offset.reads.empty() || !acc.bank.reads.empty();
      acc.addrDelay = anyRegister ? delay.value_or(0) : 0;
      // With the term sum landing in the delay register, only the residual
      // built after that register sits on the setup path. Both cones run
      // beside each other, so the delay is the max.
      assert(acc.inDelay + kStampedDelayEpsilon >=
                 acc.portDelay + acc.selectDelay &&
             "an access's priced setup is its port's delay plus its select and "
             "its address");
      double priced =
          std::max(0.0, acc.inDelay - acc.portDelay - acc.selectDelay);
      acc.addrSetup =
          acc.addrDelay
              ? std::max(postRegisterDelay(acc.offset, delays, e.width),
                         postRegisterDelay(acc.bank, delays,
                                           AddressDelays::refWidth))
              : priced;
      // The scheduler's split needs only the IR loops; this one also needs
      // constant bounds and one shared delay, so the built cone can run longer
      // than the priced one.
      if (logging::detail::enabled(Level::Debug)) {
        double built =
            std::max(builtConeDelay(acc.offset, delays, e.width),
                     builtConeDelay(acc.bank, delays, AddressDelays::refWidth));
        if (built > priced + 0.011)
          debug(Stage::Emit, acc.op)
              << "the address cone built here is "
              << llvm::format("%.2f", built) << " ns against the "
              << llvm::format("%.2f", priced)
              << " ns the schedule priced; a term the pricing reduced stayed "
                 "in the residual";
      }
      // An operand no residual is left reading has no consumer: its slot stays
      // empty and the delay it was owed is withdrawn before a chain carries it.
      llvm::BitVector read = residualReads(acc);
      for (unsigned p = 0, n = acc.addr.size(); p < n; ++p) {
        if (read[p])
          continue;
        if (auto *it = edges.find(&acc.addr[p]); it != edges.end())
          it->second.reduced = true;
      }
    }
  }
  // A scaled counter whose init and step equal the region's counter holds the
  // same value each cycle, so the emitter reads `rc.counter` for it rather than
  // building a second register and adder. Both the emitter and the report read
  // this fold flag.
  for (RegionBlock &rb : dp.regions) {
    std::optional<int64_t> lb = dp.constantOf(rb.lbSource);
    std::optional<int64_t> step = dp.constantOf(rb.stepSource);
    if (!lb || !step)
      continue;
    for (RegionBlock::AddrStride &s : rb.addrStrides)
      s.isCounter =
          s.init == *lb && s.step == *step && !s.bump && !s.wrap && !s.hasCarry;
  }
}

void DatapathBuilder::resolveMemoryOperands() {
  for (MemUnit &m : dp.mems)
    for (MemUnit::Access &acc : m.accesses) {
      unsigned ridx = regionIdxOf.lookup(acc.op->getParentOp());
      unsigned ii = dp.regions[ridx].ii.value_or(1);
      SmallVector<Value> operands;
      AffineMap ignored;
      dcpAddressing(acc.op, ignored, operands);
      for (unsigned k = 0, e = operands.size(); k < e; ++k)
        recordCarriedEdge(
            resolveOperand(operands[k], acc.op, ii, /*addressSlot=*/true),
            operands[k], acc.op, acc.addr[k], ridx);
      if (acc.isWrite) {
        Value datum = cast<dcp::DCPathStoreOp>(acc.op).getValue();
        recordCarriedEdge(resolveOperand(datum, acc.op, ii), datum, acc.op,
                          acc.data, ridx);
      }
    }
}

void DatapathBuilder::resolveStreamOperands() {
  // Re-stamp an access's schedule cycle. `start` is the single source both the
  // datapath and `dcpStart` read, so the attribute and the cached stage move
  // together.
  auto restamp = [](StreamChannel::Access &acc, int64_t cycle) {
    acc.op->setAttr(
        "start",
        IntegerAttr::get(cast<IntegerAttr>(acc.op->getAttr("start")).getType(),
                         cycle));
    acc.stage = static_cast<unsigned>(cycle);
  };

  // A stream put's data driver resolves through the same reg-depth path as a
  // store's; a predicated get/put's i1 predicate is likewise delayed to the
  // access stage, so it gates the handshake in deriveStallShell.
  for (StreamChannel &s : dp.streams) {
    // Cycles the bump below has inserted into each region. The scheduler put a
    // channel's accesses on DISTINCT cycles, so a bump shifts every LATER
    // access too: moving one alone would land it on the next.
    llvm::DenseMap<unsigned, unsigned> inserted;
    for (StreamChannel::Access &acc : s.accesses) {
      unsigned ridx = regionIdxOf.lookup(acc.op->getParentOp());
      unsigned ii = dp.regions[ridx].ii.value_or(1);
      unsigned &shift = inserted[ridx];
      if (shift)
        restamp(acc, dcpStart(acc.op) + shift);
      if (acc.isPut) {
        auto token = cast<StreamPutOp>(acc.op).getValue();
        auto r = resolveOperand(token, acc.op, ii);
        // AXI-S data stability: a stage>=1 put's valid pulse persists into the
        // drain under back-pressure, so a transient din could commit stale
        // data. Bump its stage by one to route it through a frozen register.
        // A held enclosing counter resolves at depth 0 but is frozen with its
        // container while this region is back-pressured, so it is no
        // transient din.
        if (r.base && r.depth == 0 && dcpStart(acc.op) >= 1 &&
            r.base.kind != Source::Kind::Counter && isTransientDin(token)) {
          restamp(acc, dcpStart(acc.op) + 1);
          ++shift;
          // What the region's drain may exceed the composed span by: each
          // access carries its own channel's accumulated shift, so the bound is
          // the largest of them and not their sum.
          RegionBlock &rb = dp.regions[ridx];
          rb.streamShift = std::max(rb.streamShift, shift);
          r = resolveOperand(token, acc.op, ii);
        }
        recordCarriedEdge(r, token, acc.op, acc.data, ridx);
      }
      auto pred = acc.isPut ? cast<StreamPutOp>(acc.op).getPred()
                            : cast<StreamGetOp>(acc.op).getPred();
      if (pred) {
        // Unlike `acc.data` (a None Source trips an assert), a None `acc.when`
        // reads as "unconditional", so an unresolved predicate would silently
        // turn a masked get/put into an every-cycle one.
        auto pr = resolveOperand(pred, acc.op, ii);
        if (!pr.base) {
          unsupported(Stage::Emit, Code::CrossRegionHandOff, acc.op)
              << "Predicate of this stream access is produced by a value the "
                 "region cannot read; such a cross-region value hand-off is "
                 "not lowered yet, and the access would otherwise fire "
                 "unconditionally";
          dp.infeasible = true;
        }
        recordCarriedEdge(pr, pred, acc.op, acc.when, ridx);
      }
    }
  }
}

// The pulse-delay depth from which a counter is cheaper than a chain on this
// device's rows: the smallest n where the counter `delayPulseCounted` builds
// (clog2(n)+1 registers, an increment, a compare and their selects) prices
// below the 1-bit n-stage chain it replaces. A device that prices neither side
// keeps the default; one whose chains never cost more never counts.
static unsigned countedDelayThreshold(const OperatorLibrary &lib) {
  auto counterWins = [&](uint64_t n) {
    int64_t b = std::max<int64_t>(1, llvm::Log2_64_Ceil(n));
    OperatorIdentity add, cmp;
    add.comb = CombOpKindEnum::Addi;
    cmp.comb = CombOpKindEnum::Cmpi;
    int64_t counter = (b + 1) * lib.pulsePrice() + lib.instancePrice(add, b) +
                      lib.instancePrice(cmp, b) + 2 * lib.muxPrice(2, b) +
                      2 * lib.muxPrice(2, 1);
    return counter < lib.chainPrice(n, 1);
  };
  constexpr uint64_t kProbeLimit = 1 << 20;
  if (lib.chainPrice(kProbeLimit, 1) == 0)
    return 64; // no chain row: cost cannot decide, keep the bounded shape
  uint64_t hi = 4;
  while (hi <= kProbeLimit && !counterWins(hi))
    hi *= 2;
  if (hi > kProbeLimit)
    return std::numeric_limits<unsigned>::max();
  // clog2(n) is constant over (hi/2, hi] and the chain row is nondecreasing,
  // so the first winning depth inside the octave is a binary search.
  uint64_t lo = hi / 2;
  while (lo + 1 < hi) {
    uint64_t mid = lo + (hi - lo) / 2;
    if (counterWins(mid))
      hi = mid;
    else
      lo = mid;
  }
  return hi;
}

// When each region has finished, relative to the issue pulse of the iteration
// that reaches it. `emitDone` counts a leaf's `done` off this.
//
// Three things a region can still owe past its issue, all in the same units:
// a store presented at its stage commits `writeLatency` cycles later, and the
// done latch rides the last of those, so it drains at the commit minus one; a
// put presents at its stage; a survivor latches on the cycle its result lands.
// A call result is not one of them, being self-timed by the child's `done`
// rather than statically captured.
//
// Safe to read a result's Source before `insertRegisters`: a result slot is
// tied by `resolveValue` and never handed to `edges`, so nothing patches it
// later.
void DatapathBuilder::recordDrainStages() {
  for (RegionBlock &rb : dp.regions) {
    unsigned drain = 0;
    for (AccRef r : rb.memAccesses) {
      const MemUnit &m = dp.mems[r.id];
      const MemUnit::Access &acc = m.accesses[r.idx];
      if (acc.isWrite)
        drain = std::max(drain, storeDrainCycle(m, acc));
    }
    for (AccRef r : rb.streamAccesses) {
      const StreamChannel::Access &acc = dp.streams[r.id].accesses[r.idx];
      if (acc.isPut)
        drain = std::max(drain, acc.stage);
    }
    for (const RegionResult &r : rb.results)
      if (r.value && r.value.kind != Source::Kind::Call)
        drain = std::max(drain, dp.readyCycle(r.value));
    rb.drainStage = drain;
  }
}

void DatapathBuilder::insertRegisters() {
  assert(dp.units.data() == unitsBase && dp.mems.data() == memsBase &&
         dp.streams.data() == streamsBase &&
         "a cell container grew after resolveEdges; the edge table's slot "
         "pointers are dangling");
  // One chain per (value, region) key, as long as its deepest surviving tap;
  // the shallower consumers read their own tap off it (Source::Reg's
  // `outPort`). A tap the address reduction withdrew buys no chain.
  llvm::DenseMap<RegKey, RegId> keyToReg;
  for (auto &[slot, e] : edges) {
    if (e.reduced)
      continue;
    auto [it, isNew] = keyToReg.try_emplace(e.key, dp.regs.size());
    if (!isNew) {
      Register &reg = dp.regs[it->second];
      assert(reg.ready == e.ready &&
             "one value's edges disagree on when it lands");
      reg.depth = std::max(reg.depth, e.depth);
      if (e.depth)
        reg.taps.push_back(e.depth);
      continue;
    }
    Register reg;
    reg.id = dp.regs.size();
    reg.value = e.key.first;
    // A counter chain carries the region's own counter width, not the 32-bit
    // index the value is typed at; the emitter narrows at the head and
    // sign-extends back at each tap.
    reg.type = e.base.kind == Source::Kind::Counter
                   ? dp.regions[e.base.id].counterType
                   : reg.value.getType();
    reg.depth = e.depth;
    reg.input = e.base;
    reg.ready = e.ready;
    if (e.depth)
      reg.taps.push_back(e.depth);
    dp.regions[e.key.second].regs.push_back(reg.id);
    dp.regs.push_back(reg);
  }
  for (auto &[key, rid] : keyToReg) {
    auto &taps = dp.regs[rid].taps;
    llvm::sort(taps);
    taps.erase(std::unique(taps.begin(), taps.end()), taps.end());
  }

  for (auto &[slot, e] : edges)
    if (!e.reduced)
      *slot = Source{Source::Kind::Reg, keyToReg[e.key], e.depth};

  materializeMuxes();
}

void DatapathBuilder::materializeMuxes() {
  auto sameSource = [](const Source &a, const Source &b) {
    return a.kind == b.kind && a.id == b.id && a.outPort == b.outPort;
  };
  auto unphased = [](const Mux::Phase &p) {
    return p.kind == Mux::Phase::Always;
  };
  for (MuxBuild &mb : muxBuilds) {
    Source &slot = *mb.slot;
    // One driver across every arm is a wire. A phased arm never collapses: its
    // pair carries the same port in different iterations, not the same value.
    if (llvm::all_of(mb.phases, unphased) &&
        llvm::all_of(mb.sources, [&](const Source &s) {
          return sameSource(s, mb.sources[0]);
        })) {
      slot = mb.sources[0];
      continue;
    }
    Mux mx;
    mx.id = dp.muxes.size();
    mx.region = mb.region;
    mx.type = mb.type;
    mx.sources.assign(mb.sources.begin(), mb.sources.end());
    mx.selectOps.assign(mb.ops.begin(), mb.ops.end());
    mx.phases.assign(mb.phases.begin(), mb.phases.end());
    // The stage each arm's pulse is delayed to, frozen here because this is the
    // last pass that touches a schedule cycle.
    for (Operation *op : mb.ops)
      mx.selectStages.push_back(dcpStart(op));
    dp.regions[mb.region].muxes.push_back(mx.id);
    slot = Source{Source::Kind::Mux, mx.id, 0};
    dp.muxes.push_back(std::move(mx));
  }
}

void DatapathBuilder::allocateUnits(ArrayRef<SmallVector<UnitId, 2>> groups) {
  if (groups.empty())
    return; // the trivial allocation, which the walk already built

  // Where each unit folds: itself, unless the policy named it in a group.
  SmallVector<UnitId> leader(dp.units.size());
  std::iota(leader.begin(), leader.end(), 0);
  for (const SmallVector<UnitId, 2> &group : groups)
    for (UnitId uid : group) {
      assert(leader[uid] == uid &&
             "a policy named one unit in two groups; the second fold would "
             "silently win and its ops would issue on a unit nothing checked "
             "them against");
      leader[uid] = group.front();
    }

  // Rebuild rather than empty the folded-away entries: a `FuncUnit` with no
  // bound op has no `repOp()`, so a dense table keeps that an invariant instead
  // of a hazard every consumer has to remember to skip.
  SmallVector<UnitId> remap(dp.units.size(), 0);
  std::vector<FuncUnit> allocated;
  for (UnitId old = 0, e = dp.units.size(); old < e; ++old) {
    if (leader[old] != old)
      continue;
    remap[old] = allocated.size();
    allocated.push_back(std::move(dp.units[old]));
    allocated.back().id = remap[old];
  }
  // The leader keeps `boundOps.front()`, so `repOp()` and every name derived
  // from it are the ones the trivial allocation would have produced.
  for (UnitId old = 0, e = dp.units.size(); old < e; ++old)
    if (leader[old] != old)
      for (const FuncUnit::BoundOp &bo : dp.units[old].boundOps)
        allocated[remap[leader[old]]].boundOps.push_back(bo);
  dp.units = std::move(allocated);

  // Region membership: the folded-away ids are gone, the survivors renumbered.
  for (RegionBlock &rb : dp.regions) {
    SmallVector<UnitId, 4> kept;
    for (UnitId uid : rb.units)
      if (leader[uid] == uid)
        kept.push_back(remap[uid]);
    rb.units = std::move(kept);
  }
  // The two provenance maps, rewritten FROM the table rather than alongside it,
  // so a Source's bound-op index cannot drift from the slot it names. They are
  // the whole of what holds a UnitId at this phase: no `record*` pass has run.
  for (const FuncUnit &u : dp.units)
    for (auto [slot, bo] : llvm::enumerate(u.boundOps)) {
      dp.opToUnit[bo.op] = u.id;
      producerOf[bo.op->getResult(0)] =
          Source{Source::Kind::Unit, u.id, static_cast<unsigned>(slot)};
    }
}

// Report each edge `siblingPredecessors` has beyond the built relation: a pair
// the composed span serializes and the hardware overlaps.
static void diffSiblingDeps(const Datapath &dp) {
  SmallVector<RegionId> topIds;
  SmallVector<SmallVector<Operation *>> nodeOps;
  for (const RegionBlock &rb : dp.regions) {
    if (rb.parent)
      continue;
    topIds.push_back(rb.id);
    nodeOps.push_back({rb.op});
  }
  auto modelled = siblingPredecessors(nodeOps);
  for (auto [i, rid] : llvm::enumerate(topIds))
    for (unsigned p : modelled[i])
      if (!llvm::is_contained(dp.regions[rid].predecessors, topIds[p]))
        debug(Stage::Emit, dp.regions[rid].op)
            << "Latency model orders region " << topIds[p] << " before region "
            << rid
            << ", the built model does not: the composed span pays for "
               "a hand-off the hardware overlaps";
}

// Composition predecessors of each top-level region (`rb.predecessors`): the
// earlier top-level siblings it must start after, all attributed to the
// top-level ancestor. Per-region binding keeps functional units from
// conflicting across regions, so the signals are the (1) to (3) below: a shared
// memref (hazard pairs only), a shared channel, a cross-region SSA edge. The
// emitter starts a predecessor-free region concurrently with the kernel
// `start` and gates the rest on their producers' joined `done`.
//
// `siblingPredecessors` answers the same question off the IR, and the two must
// agree: the composed span is published as an exact contract, so an edge only
// one side has is either a span the hardware beats or hardware the span never
// paid for (`diffSiblingDeps`).
void DatapathBuilder::recordSiblingDeps() {
  // Every op inside a top-level region maps to that region's id, a nested child
  // folding into it. A value defined outside any region has no entry.
  DenseMap<Operation *, RegionId> opTop;
  for (const RegionBlock &rb : dp.regions) {
    if (rb.parent)
      continue; // walk only from a top-level root
    RegionId rid = rb.id;
    opTop[rb.op] = rid;
    rb.op->walk([&](Operation *o) { opTop[o] = rid; });
  }

  auto addPred = [&](RegionId producer, RegionId consumer) {
    assert(producer < consumer && "a predecessor must precede its consumer");
    auto &preds = dp.regions[consumer].predecessors;
    if (!llvm::is_contained(preds, producer))
      preds.push_back(producer);
  };

  // (1) A shared memref, ordered by its hazard pairs (`hazardEdges`, at bank
  // granularity): two regions that only read the array, or that touch
  // different banks, overlap, and `Datapath::portGraph` reads any unordered
  // pair as simultaneous, so they take separate ports. A CallUnit masters
  // memref operands without a MemUnit::Access, so it counts as a sharer too,
  // at its ports' directions but bank-less. A skewed layout shares one port
  // per lane slot across regions (`assignLanes`), so it keeps every toucher
  // ordered.
  for (const MemUnit &m : dp.mems) {
    SmallVector<MemTouch, 8> touch;
    auto note = [&](RegionId region, bool writes, std::optional<int64_t> bank) {
      RegionId t = dp.topRegionOf(region);
      for (MemTouch &e : touch)
        if (e.node == t && e.bank == bank) {
          e.writes |= writes;
          return;
        }
      touch.push_back({t, writes, bank});
    };
    for (const MemUnit::Access &a : m.accesses)
      note(a.region, a.isWrite,
           !m.layout.skew() && a.staticBank
               ? std::optional<int64_t>(*a.staticBank)
               : std::nullopt);
    for (const CallUnit &cu : dp.calls)
      for (const CallUnit::MemArg &ma : cu.memArgs)
        if (ma.mem == m.id)
          note(cu.region, ma.isWrite, std::nullopt);
    llvm::stable_sort(touch, [](const MemTouch &a, const MemTouch &b) {
      return a.node < b.node; // program order (a top's id orders it)
    });
    if (m.layout.skew()) {
      for (unsigned j = 1; j < touch.size(); ++j)
        addPred(touch[j - 1].node, touch[j].node);
      continue;
    }
    hazardEdges(touch, [&](unsigned p, unsigned c) { addPred(p, c); });
  }

  // (2) A shared channel: a FIFO is one port carrying the program's token
  // order, so two regions touching it must run in sequence, else they drive it
  // together and (for two gets) pop the same token twice.
  for (const StreamChannel &s : dp.streams) {
    // The top-level regions touching this channel, chained in program order so
    // the rest follows transitively.
    SmallVector<RegionId, 4> tops;
    for (const StreamChannel::Access &a : s.accesses) {
      RegionId t = dp.topRegionOf(a.region);
      if (!llvm::is_contained(tops, t))
        tops.push_back(t);
    }
    llvm::sort(tops);
    for (unsigned j = 1; j < tops.size(); ++j)
      addPred(tops[j - 1], tops[j]);
  }

  // (3) Cross-region SSA edges: an op in one top-level region uses a value
  // produced in an earlier one (a scalar survivor). SSA dominance guarantees
  // the producer precedes the consumer in program order.
  func.walk([&](Operation *o) {
    auto uit = opTop.find(o);
    if (uit == opTop.end())
      return;
    RegionId consumer = uit->second;
    for (Value v : o->getOperands()) {
      auto *def = v.getDefiningOp();
      if (!def)
        continue;
      if (auto dit = opTop.find(def); dit != opTop.end()) {
        if (dit->second != consumer)
          addPred(dit->second, consumer);
        continue;
      }
      // A def no region owns binds no hardware and so orders nothing:
      // `enumerateRegions` is total over a block, and the only ops the reify
      // leaves outside a region are declarations.
      assert(isDeclarationOp(def) &&
             "a computing op outside every region drives a region's input");
    }
  });

  if (logging::detail::enabled(Level::Debug))
    diffSiblingDeps(dp);
}

//===----------------------------------------------------------------------===//
// Driver.
//===----------------------------------------------------------------------===//

void DatapathBuilder::build() {
  collectConstants();

  // dcp region ops in program order. Pre-order so an enclosing container is
  // processed first: the parent/child linkage and the outer-index counter
  // attribution rely on parent-before-child.
  SmallVector<Operation *> regionOps;
  func.walk<WalkOrder::PreOrder>([&](dcp::DCPathRegionOpInterface region) {
    regionOps.push_back(region);
  });

  // Scalar-argument IO ports: one of the maps `resolveValue` reads, so every
  // pass below sees a scalar func arg as an IO source.
  bindIOArgs();

  // Every array the function touches, before the walk so a binding looks its
  // memory up rather than deciding anything about it.
  collectStorageFacts(regionOps);

  for (unsigned ridx = 0, e = regionOps.size(); ridx < e; ++ridx) {
    Operation *regionOp = regionOps[ridx];
    auto rb = addRegion(regionOp, ridx);
    forEachBodyOp(regionOp, [&](Operation *op) { bindResource(op, rb); });
    dp.regions.push_back(std::move(rb));
  }
  recordForwards();

  // The allocation, settled here and not later: every pass below resolves
  // Values to Sources against the unit table (see `allocateUnits`).
  allocateUnits(
      policy.plan(dp, {cycleTime, dev.operators})); // trivial => a no-op
  assert(llvm::all_of(dp.units,
                      [](const FuncUnit &u) { return !u.boundOps.empty(); }) &&
         "the unit table is the allocation: a unit exists because ops are "
         "bound to it");

  deriveShapes();       // controller discriminant (needs every child)
  deriveCounterTypes(); // counter width (each loop's own range)
  dp.countedDelayCycles = countedDelayThreshold(dev.operators);
  // Everything below resolves Values to Sources, and so runs here rather than
  // during the walk: `resolveValue` needs the complete region model.
  // Every op the reify leaves in the module body binds no hardware:
  // `enumerateRegions` is total over a block, so anything that computes is
  // inside a region.
#ifndef NDEBUG
  for (Operation &op : func.getBody().front())
    assert((isa<dcp::DCPathRegionOpInterface, dcp::DCPathOutputOp>(&op) ||
            isDeclarationOp(&op)) &&
           "an operation outside every region reached the datapath");
#endif
  recordRegionResults(); // per-region results/recurrence + predicate
  recordCallScalars();   // each dcp.instance's scalar operand drivers
  recordCallDeps();      // composition DAG on the instance substrate
  recordRegionBounds();  // induction bounds, at that width
  recordResults();       // scalar func-result output ports
  // What every input slot reads and how late it needs it, then what carries it
  // there: scaled counters, delay chains, muxes.
  resolveEdges();
  recordDrainStages(); // when each region finishes (needs the final stages)
  // The top-level composition DAG, before the ports: `portGraph` reads two
  // accesses under different top-level ancestors as unable to issue together,
  // and this pass is what makes that true.
  recordSiblingDeps();
  // Ports before the delays, and the two are independent: an access holds a
  // port once it knows which bank it commits to and which skew lane it holds,
  // both settled by the region walk, and no pass here reads a Register or a
  // Mux.
  assignLanes();
  planAccessPorts();
  bindMemoryPorts();
  enumerateBoundaryPorts();
  measurePorts(); // what the ports cost against what the schedule asks
  // The delays the edges owe: the address reduction first, since a term it
  // folds into a scaled counter withdraws its edge, then the chains over what
  // is left, then the muxes that pick between them. Last of the derivations, so
  // no pass may resolve a Source after it.
  planAddressGenerators();
  insertRegisters();
  verifyBinding(dp); // MRT legality: no unit shared by conflicting ops
}

} // namespace mlir::allo::uarch
