/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Microarch/BindingPolicy.h"
#include "allo/Microarch/DatapathBuilder.h"

#include "allo/IR/AlloOps.h"
#include "allo/Scheduling/OperatorLibrary.h" // unit input delay

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::allo;

namespace mlir::allo::uarch {

//===----------------------------------------------------------------------===//
// Timing readers over the scheduled dcp IR. One definition of the schedule
// cycle, the operator latency, and the derived result-ready cycle.
//===----------------------------------------------------------------------===//

unsigned dcpStart(Operation *op) {
  return cast<IntegerAttr>(op->getAttr("start")).getInt();
}

unsigned dcpLatency(Operation *op) {
  // An OPERATOR latency: the cycles between an op's issue and its result
  // landing. A region's `latency` is its whole start->done span, not an
  // operator delay.
  assert(!isa<dcp::DCPathRegionOpInterface>(op) &&
         "a region's `latency` is its whole span, not an operator latency");
  if (auto l = dyn_cast<dcp::DCPathLoadOp>(op))
    return static_cast<unsigned>(l.getLatency());
  // An IP compute takes its latency from the `dcp.operator` it names, which
  // outlives emission for this reason; a combinational one issues and lands in
  // the same cycle.
  if (auto comp = dyn_cast<dcp::DCPathComputeOp>(op)) {
    FlatSymbolRefAttr sym = comp.getOpTypeAttr();
    if (!sym)
      return 0;
    auto opr =
        SymbolTable::lookupNearestSymbolFrom<dcp::DCPathOperatorOp>(comp, sym);
    assert(opr && "a dcp.compute op_type must reference a live dcp.operator");
    return static_cast<unsigned>(opr.getLatency());
  }
  // A store and a call carry their own `latency`, each an ODS field of its op.
  if (auto lat = op->getAttrOfType<IntegerAttr>("latency"))
    return static_cast<unsigned>(lat.getInt());
  return 0;
}

unsigned readyCycleOf(Operation *op) { return dcpStart(op) + dcpLatency(op); }

llvm::StringRef shapeName(RegionBlock::Shape s) {
  switch (s) {
  case RegionBlock::Shape::Leaf:
    return "leaf";
  case RegionBlock::Shape::Container:
    return "container";
  case RegionBlock::Shape::Guard:
    return "guard";
  case RegionBlock::Shape::CallNode:
    return "callnode";
  }
  llvm_unreachable("unhandled RegionBlock::Shape");
}

unsigned storeDrainCycle(const MemUnit &m, const MemUnit::Access &acc) {
  assert(m.writeLatency >= 1 &&
         "a zero-cycle write has no commit edge for the done latch to ride; "
         "`assertModelInvariants` holds the device row to that");
  return acc.stage + m.writeLatency - 1;
}

RegionId Datapath::topRegionOf(RegionId r) const {
  while (regions[r].parent)
    r = *regions[r].parent;
  return r;
}

std::optional<int64_t> Datapath::constantOf(const Source &s) const {
  if (s.kind != Source::Kind::Const)
    return std::nullopt;
  auto ia = dyn_cast<IntegerAttr>(consts[s.id].value);
  return ia ? std::optional<int64_t>(ia.getInt()) : std::nullopt;
}

unsigned Datapath::readyCycle(const Source &s) const {
  switch (s.kind) {
  // A call result does not land at `stage + latency`: it lands at its
  // region-relative issue plus the callee's whole start->done depth.
  // Indeterminate calls are guarded earlier.
  case Source::Kind::Call: {
    const CallUnit &cu = calls[s.id];
    assert(cu.latency && "readyCycle of an indeterminate call result");
    return cu.start + static_cast<unsigned>(*cu.latency);
  }
  case Source::Kind::Unit: {
    const FuncUnit &u = units[s.id];
    return u.boundOps[s.outPort].stage + u.latency;
  }
  case Source::Kind::Mem: {
    const MemUnit &m = mems[s.id];
    return m.accesses[s.outPort].stage + m.readLatency;
  }
  // A get is a combinational front-read of the FIFO, so it lands at issue.
  case Source::Kind::Stream:
    return streams[s.id].accesses[s.outPort].stage;
  // A held source has no landing stage: a literal is constant, an IO port
  // stable for the whole kernel, and a counter or survivor a register settled
  // by the time the region reading it issues.
  case Source::Kind::Const:
  case Source::Kind::IO:
  case Source::Kind::Counter:
  case Source::Kind::Survivor:
    return 0;
  case Source::Kind::Reg:
  case Source::Kind::Mux:
  case Source::Kind::None:
    break;
  }
  llvm_unreachable("readyCycle only models a producing or held Source");
}

Datapath::Datapath(dcp::DCPathModuleOp func, const BindingPolicy &policy,
                   const DeviceModel &dev, float cycleTime,
                   const CalleeCtx &callees, bool isTop) {
  this->func = func;
  atTop = isTop;
  DatapathBuilder builder(*this, func, policy, dev, cycleTime, callees);
  builder.build();
}

//===----------------------------------------------------------------------===//
// The model visitor.
//===----------------------------------------------------------------------===//

std::string SourceSite::describe() const {
  auto idx = [&](const char *noun) {
    return std::string(noun) + " " + std::to_string(index);
  };
  switch (slot) {
  case Slot::UnitInput:
    return idx("operand") + " of a compute unit";
  case Slot::UnitInit:
    return "the reduction identity of " + idx("operand");
  case Slot::RegisterInput:
    return "the input of a pipeline register";
  case Slot::MuxInput:
    return idx("arm") + " of a shared-unit mux";
  case Slot::MemAddress:
    return idx("address index") + " of a memory access";
  case Slot::MemWriteData:
    return "the data of a memory write";
  case Slot::StreamData:
    return "the token data of a stream put";
  case Slot::StreamPredicate:
    return "the predicate of a stream access";
  case Slot::CallScalarIn:
    return idx("scalar argument") + " of a sub-kernel call";
  case Slot::FuncResult:
    return idx("scalar function result");
  case Slot::RegionBound:
    return "a runtime loop bound";
  case Slot::RegionResult:
    return idx("result") + " of a region";
  case Slot::RegionResultInit:
    return "the loop-carried identity of " + idx("result");
  case Slot::RegionElseResult:
    return idx("else-branch result") + " of a guard";
  case Slot::RegionCondition:
    return "the control predicate of a region";
  }
  llvm_unreachable("unhandled SourceSite::Slot");
}

llvm::BitVector residualReads(const MemUnit::Access &acc) {
  llvm::BitVector read(acc.addr.size());
  unsigned numDims = acc.addrMap.getNumDims();
  for (AffineExpr e : {acc.offset.residual, acc.bank.residual}) {
    if (!e)
      continue;
    e.walk([&](AffineExpr x) {
      unsigned p;
      if (auto d = dyn_cast<AffineDimExpr>(x))
        p = d.getPosition();
      else if (auto s = dyn_cast<AffineSymbolExpr>(x))
        p = numDims + s.getPosition();
      else
        return;
      // Past the operand list: a digit `Reduced::reads` supplies instead.
      if (p < read.size())
        read.set(p);
    });
  }
  return read;
}

void forEachSource(
    const Datapath &dp,
    llvm::function_ref<void(const Source &, const SourceSite &)> fn) {
  using Slot = SourceSite::Slot;
  // `required` states whether a None source at that slot means "absent" or
  // "unresolved", so no consumer re-decides it.
  auto visit = [&](const Source &s, Slot slot, unsigned index, Operation *op,
                   bool required) {
    fn(s, SourceSite{slot, index, op, required});
  };

  for (const FuncUnit &u : dp.units) {
    for (auto [k, s] : llvm::enumerate(u.inputs))
      visit(s, Slot::UnitInput, k, u.repOp(), /*required=*/true);
    for (auto [k, inits] : llvm::enumerate(u.inputInits))
      for (const Source &s : inits)
        visit(s, Slot::UnitInit, k, u.repOp(), /*required=*/false);
  }
  for (const Register &r : dp.regs)
    visit(r.input, Slot::RegisterInput, r.id, nullptr, /*required=*/true);
  for (const Mux &x : dp.muxes) {
    assert(x.selectOps.size() == x.sources.size() &&
           "a mux's selects are parallel to its arms");
    for (auto [k, s] : llvm::enumerate(x.sources))
      visit(s, Slot::MuxInput, k, x.selectOps[k], /*required=*/true);
  }

  for (const MemUnit &m : dp.mems)
    for (const MemUnit::Access &acc : m.accesses) {
      llvm::BitVector read = residualReads(acc);
      for (auto [k, s] : llvm::enumerate(acc.addr))
        visit(s, Slot::MemAddress, k, acc.op, /*required=*/read[k]);
      // A load leaves `data` None by construction.
      visit(acc.data, Slot::MemWriteData, 0, acc.op, /*required=*/acc.isWrite);
    }
  for (const StreamChannel &ch : dp.streams)
    for (const StreamChannel::Access &acc : ch.accesses) {
      visit(acc.data, Slot::StreamData, 0, acc.op, /*required=*/acc.isPut);
      visit(acc.when, Slot::StreamPredicate, 0, acc.op, /*required=*/false);
    }
  for (const CallUnit &cu : dp.calls)
    for (auto [k, sa] : llvm::enumerate(cu.scalarIns))
      visit(sa.src, Slot::CallScalarIn, k, cu.invoke, /*required=*/true);
  for (auto [k, r] : llvm::enumerate(dp.results))
    visit(r.source, Slot::FuncResult, k, nullptr, /*required=*/true);

  for (const RegionBlock &rb : dp.regions) {
    // Set for a counted region, None for an acyclic one; `ubSource` is also
    // None for the one derived bound (`tripCount` over a runtime lb/step), so
    // none of the three is required.
    for (const Source &s : {rb.lbSource, rb.ubSource, rb.stepSource})
      visit(s, Slot::RegionBound, rb.id, nullptr, /*required=*/false);
    // Only a Container threads its recurrence through `setupCarriedIterArgs`,
    // where an unresolved init or next has nothing to latch. Elsewhere a result
    // may be untracked.
    bool threaded = rb.shape == RegionBlock::Shape::Container;
    for (auto [k, r] : llvm::enumerate(rb.results)) {
      visit(r.value, Slot::RegionResult, k, nullptr, threaded);
      visit(r.init, Slot::RegionResultInit, k, nullptr, threaded);
      visit(r.elseValue, Slot::RegionElseResult, k, nullptr,
            /*required=*/false);
    }
    // A while and a guard both need their predicate; a counted region has none.
    visit(rb.condition, Slot::RegionCondition, rb.id, nullptr,
          /*required=*/rb.conditional || rb.shape == RegionBlock::Shape::Guard);
  }
}

//===----------------------------------------------------------------------===//
// Textual dump.
//===----------------------------------------------------------------------===//

static void printValueName(Value v, raw_ostream &os) {
  if (auto arg = dyn_cast<BlockArgument>(v))
    os << "#arg" << arg.getArgNumber();
  else if (Operation *def = v.getDefiningOp())
    os << def->getName().getStringRef();
  else
    os << "<?>";
}

static void printSource(const Source &s, raw_ostream &os) {
  switch (s.kind) {
  case Source::Kind::None:
    os << "-";
    break;
  case Source::Kind::Unit:
    os << "u" << s.id << "#" << s.outPort;
    break;
  case Source::Kind::Reg:
    os << "r" << s.id << "@" << s.outPort;
    break;
  case Source::Kind::Mem:
    os << "m" << s.id << "#" << s.outPort;
    break;
  case Source::Kind::Mux:
    os << "x" << s.id;
    break;
  case Source::Kind::IO:
    os << "i" << s.id;
    break;
  case Source::Kind::Const:
    os << "c" << s.id;
    break;
  case Source::Kind::Counter:
    os << "iv" << s.id;
    break;
  case Source::Kind::Survivor:
    os << "sv" << s.id << "#" << s.outPort;
    break;
  case Source::Kind::Stream:
    os << "st" << s.id << "#" << s.outPort;
    break;
  case Source::Kind::Call:
    os << "call" << s.id << "#" << s.outPort;
    break;
  }
}

static void printSourceList(ArrayRef<Source> ss, raw_ostream &os) {
  os << "[";
  llvm::interleaveComma(ss, os, [&](const Source &s) { printSource(s, os); });
  os << "]";
}

// A start policy as one lower-case word, for the debug dump.
static llvm::StringRef startPolicyName(CallUnit::StartPolicy p) {
  switch (p) {
  case CallUnit::StartPolicy::Handshake:
    return "handshake";
  case CallUnit::StartPolicy::Broadcast:
    return "broadcast";
  case CallUnit::StartPolicy::TimeTriggered:
    return "timed";
  }
  llvm_unreachable("unhandled CallUnit::StartPolicy");
}

std::optional<double>
unitSlack(const FuncUnit &u, const OperatorLibrary &lib, float cycleTime,
          const llvm::DenseMap<Operation *, double> *sinkTails) {
  double slack = cycleTime;
  for (const FuncUnit::BoundOp &bo : u.boundOps) {
    if (!bo.z)
      return std::nullopt;
    // A same-cycle non-unit sink downstream takes its committed delay out of
    // this unit's slack, the same as the unit's own inDelay.
    double tail = sinkTails ? sinkTails->lookup(bo.op) : 0.0;
    slack = std::min(slack, cycleTime - *bo.z - u.inDelay - tail);
  }
  // The identity re-injection select `emitUnits` builds in front of a
  // recurrence port: one arm per early iteration plus the carried value. A
  // path enters through one port, so the deepest port's cone bounds them all.
  double cone = 0.0;
  for (auto [k, inits] : llvm::enumerate(u.inputInits))
    if (!inits.empty())
      cone = std::max(
          cone, muxCone(lib, inits.size() + 1,
                        datapathWidth(u.repOp()->getOperand(k).getType())));
  return slack - cone;
}

llvm::DenseMap<Operation *, double> sinkTails(const Datapath &dp) {
  llvm::DenseMap<Operation *, double> out;
  // Only a unit's own result carries a binding cone into the sink: other
  // producers are settled at cycle start, and an earlier-cycle producer hands
  // off through a register.
  auto credit = [&](Value v, Operation *sink, double d) {
    Operation *def = v.getDefiningOp();
    if (!def || d <= 0.0 || !dp.opToUnit.contains(def))
      return;
    if (dcpLatency(def) != 0 || dcpStart(def) != dcpStart(sink))
      return;
    double &tail = out[def];
    tail = std::max(tail, d);
  };
  for (const MemUnit &m : dp.mems)
    for (const MemUnit::Access &acc : m.accesses) {
      if (acc.isWrite)
        credit(cast<dcp::DCPathStoreOp>(acc.op).getValue(), acc.op,
               acc.portDelay);
      // An address in a delay register launches the port path from that
      // register, past any unit cone. addrDelay is unset until `resolveEdges`,
      // so this pre-binding read is conservative.
      if (acc.addrDelay == 0) {
        auto indices = acc.isWrite
                           ? cast<dcp::DCPathStoreOp>(acc.op).getIndices()
                           : cast<dcp::DCPathLoadOp>(acc.op).getIndices();
        for (Value v : indices)
          credit(v, acc.op, acc.inDelay);
      }
    }
  for (const StreamChannel &ch : dp.streams)
    for (const StreamChannel::Access &acc : ch.accesses) {
      if (!acc.isPut)
        continue;
      auto put = cast<StreamPutOp>(acc.op);
      credit(put.getValue(), acc.op, acc.inDelay);
      if (Value pred = put.getPred())
        credit(pred, acc.op, acc.inDelay);
    }
  return out;
}

// Does \p a reach \p b by walking `preds` backwards from \p b? Memoized in
// \p memo, whose keys are the pair asked about; the callers ask over a pair
// loop quadratic in the accesses.
static bool reachesMemo(
    unsigned a, unsigned b,
    llvm::DenseMap<std::pair<unsigned, unsigned>, bool> &memo,
    llvm::function_ref<void(unsigned, llvm::SmallVectorImpl<unsigned> &)>
        preds) {
  auto [it, isNew] = memo.try_emplace({a, b}, false);
  if (!isNew)
    return it->second;
  llvm::SmallVector<unsigned> work{b}, ps;
  llvm::SmallDenseSet<unsigned> seen{b};
  while (!work.empty()) {
    ps.clear();
    preds(work.pop_back_val(), ps);
    for (unsigned p : ps) {
      if (p == a)
        return it->second = true;
      if (seen.insert(p).second)
        work.push_back(p);
    }
  }
  return false;
}

Datapath::PortRelation Datapath::portGraph(MemId id,
                                           std::optional<bool> writes) const {
  const MemUnit &m = mems[id];
  PortRelation rel;
  // Does call \p a precede \p b transitively? A channel-joined pair in a
  // concurrent container is deliberately NOT ordered, and writes from such a
  // pair really are simultaneous. Memoized, since the pair loop below is
  // quadratic in the accesses.
  llvm::DenseMap<std::pair<CallId, CallId>, bool> precedes;
  auto callPrecedes = [&](CallId a, CallId b) {
    return reachesMemo(a, b, precedes,
                       [&](unsigned c, llvm::SmallVectorImpl<unsigned> &out) {
                         for (const CallUnit::Pred &p : calls[c].predecessors)
                           out.push_back(p.call);
                       });
  };

  // When each vertex runs, parallel to `rel.verts`, which holds its identity.
  struct When {
    RegionId top, region;
    unsigned residue;
  };
  // `staticBank` is empty under a skew, where two slots rotate onto one bank
  // and neither names the memory an access reaches.
  auto bankOf = [](std::optional<unsigned> b) { return b ? int(*b) : -1; };
  llvm::SmallVector<When> ws;
  auto add = [&](const When &w, const PortVertex &v) {
    rel.verts.push_back(v);
    ws.push_back(w);
  };
  // Writes before reads, and this function's own accesses before the ports its
  // children master: the order every caller writes its colouring back in.
  for (bool dir : {true, false}) {
    if (writes && *writes != dir)
      continue;
    for (auto [i, acc] : llvm::enumerate(m.accesses))
      if (acc.isWrite == dir) {
        unsigned ii = regions[acc.region].ii.value_or(0);
        unsigned start = acc.stage;
        add({topRegionOf(acc.region), acc.region, ii ? start % ii : start},
            {unsigned(i), -1, /*independent=*/false, dir,
             bankOf(acc.staticBank)});
      }
    for (const CallUnit &cu : calls)
      for (const CallUnit::MemArg &ma : cu.memArgs)
        if (ma.mem == id && ma.isWrite == dir)
          add({topRegionOf(cu.region), cu.region, 0},
              {kNoAccess, int(cu.id), ma.independent, dir, int(ma.bank)});
  }
  // Does top-level region \p a transitively precede \p b (a < b)? The sibling
  // DAG (`recordSiblingDeps`) orders hazard pairs only, so two tops with no
  // path between them really do overlap. Memoized, as `callPrecedes` is.
  llvm::DenseMap<std::pair<RegionId, RegionId>, bool> topReach;
  auto topPrecedes = [&](RegionId a, RegionId b) {
    return reachesMemo(a, b, topReach,
                       [&](unsigned r, llvm::SmallVectorImpl<unsigned> &out) {
                         for (RegionId p : regions[r].predecessors)
                           out.push_back(p);
                       });
  };

  // A container drives its children serially, so two accesses in different
  // regions under one top are ordered UNLESS a concurrent container is in the
  // chain, which places every child at 0.
  auto underConcurrent = [&](RegionId r) {
    for (;; r = *regions[r].parent) {
      if (regions[r].determinacy == DeterminacyEnum::Concurrent)
        return true;
      if (!regions[r].parent)
        return false;
    }
  };
  auto overlaps = [&](unsigned i, unsigned j) {
    const When &a = ws[i], &b = ws[j];
    const PortVertex &va = rel.verts[i], &vb = rel.verts[j];
    // A bank is its own `seq.hlmem`, so two accesses that name different ones
    // contend for nothing however they are scheduled.
    if (va.bank >= 0 && vb.bank >= 0 && va.bank != vb.bank)
      return false;
    // Under different top-level ancestors the sibling DAG decides: an ordered
    // pair hands off, an unordered one runs concurrently.
    if (a.top != b.top)
      return !topPrecedes(std::min(a.top, b.top), std::max(a.top, b.top));
    if (va.call >= 0 && vb.call >= 0) {
      if (callPrecedes(va.call, vb.call) || callPrecedes(vb.call, va.call))
        return false;
      // An unordered pair of one scheduled region still cannot meet when the
      // earlier contract drains before the later release: a TimeTriggered
      // child runs exactly [start, start+latency), and no child of such a
      // region is released before its placed cycle (`startForCall`).
      const CallUnit &x = calls[va.call], &y = calls[vb.call];
      auto apart = [](const CallUnit &p, const CallUnit &q) {
        return p.startPolicy == CallUnit::StartPolicy::TimeTriggered &&
               p.latency &&
               int64_t(p.start) + std::max<int64_t>(*p.latency, 1) <=
                   int64_t(q.start);
      };
      return a.region != b.region ||
             regions[a.region].determinacy == DeterminacyEnum::Concurrent ||
             (!apart(x, y) && !apart(y, x));
    }
    // Two direct accesses agree in one region only at the same residue; a mixed
    // pair runs beside the access there. Across regions a call's window lies
    // inside its region's span, so both kinds hand off under a serial top.
    if (a.region == b.region)
      return va.call >= 0 || vb.call >= 0 || a.residue == b.residue;
    return underConcurrent(a.region) || underConcurrent(b.region);
  };
  rel.adj.assign(ws.size(), llvm::BitVector(ws.size()));
  for (unsigned i = 0; i < ws.size(); ++i)
    for (unsigned j = i + 1; j < ws.size(); ++j)
      if (overlaps(i, j))
        rel.link(i, j);
  return rel;
}

unsigned Datapath::portConcurrency(MemId id, bool writes) const {
  PortRelation rel = portGraph(id, writes);
  unsigned n = rel.size();
  // Grow a clique from each vertex, always taking the lowest remaining
  // candidate. A vertex is never adjacent to itself, so intersecting with the
  // one just taken drops it from the candidate set.
  unsigned best = n ? 1 : 0;
  for (unsigned s = 0; s < n; ++s) {
    llvm::BitVector cand = rel.adj[s];
    unsigned size = 1;
    while (cand.any()) {
      cand &= rel.adj[cand.find_first()];
      ++size;
    }
    best = std::max(best, size);
  }
  return best;
}

void Datapath::dump(llvm::raw_ostream &os) const {
  auto func = this->func;
  os << "datapath @" << func.getSymName()
     << " countedDelay=" << countedDelayCycles << " {\n";

  // The controller discriminant as the emitter reads it: shape, then
  // termination class.
  for (const RegionBlock &rb : this->regions) {
    os << "  region " << rb.id << ": " << shapeName(rb.shape) << "/"
       << (rb.conditional                         ? "while"
           : rb.kind == RegionBlock::Kind::Cyclic ? "cyclic"
                                                  : "acyclic");
    if (rb.ii)
      os << " ii=" << *rb.ii;
    if (rb.tripCount)
      os << " trip=" << *rb.tripCount;
    os << " drain=" << rb.drainStage;
    if (!rb.predecessors.empty()) {
      os << " after=[";
      llvm::interleaveComma(rb.predecessors, os, [&](RegionId p) { os << p; });
      os << "]";
    }
    os << "\n";
    for (UnitId uid : rb.units) {
      const FuncUnit &u = this->units[uid];
      os << "    unit u" << uid << ": " << u.identity.realizationName()
         << " lat=" << u.latency << (u.pipelined ? " pipelined" : " sequential")
         << " : " << u.identity.resultType << "  [" << u.repOp()->getName()
         << " @" << u.boundOps.front().residue << "] <= ";
      printSourceList(u.inputs, os);
      // A recurrence input's reduction identities, one per early iteration.
      for (auto [k, inits] : llvm::enumerate(u.inputInits))
        if (!inits.empty()) {
          os << " init[" << k << "]=";
          printSourceList(inits, os);
        }
      os << "\n";
    }
    for (RegId rid : rb.regs) {
      const Register &r = this->regs[rid];
      os << "    reg r" << rid << ": depth=" << r.depth << " <= ";
      printSource(r.input, os);
      os << " : " << r.type << "\n";
    }
    for (MuxId xid : rb.muxes) {
      const Mux &x = this->muxes[xid];
      os << "    mux x" << xid << ": ";
      printSourceList(x.sources, os);
      // Per-arm select: the op's start cycle (the delayValid select stage),
      // suffixed by the iteration window a phased arm drives ('iN' the
      // reduction identity of iteration N, 'rN' the iterations from N on).
      os << " sel@[";
      for (auto [k, stage] : llvm::enumerate(x.selectStages)) {
        const Mux::Phase &ph = x.phases[k];
        os << (k ? ", " : "") << stage;
        if (ph.kind != Mux::Phase::Always)
          os << (ph.kind == Mux::Phase::At ? "i" : "r") << ph.iter;
      }
      os << "]\n";
    }
  }

  for (const MemUnit &m : this->mems) {
    os << "  mem m" << m.id << ": ";
    printValueName(m.memref, os);
    os << (m.external ? " external" : " internal") << " w=" << m.width
       << " depth=" << m.depthWords << " banks=" << m.numBanks
       << " storage=" << m.storage << "\n";
    for (const MemUnit::Access &acc : m.accesses) {
      os << "    " << (acc.isWrite ? "wr " : "rd ") << acc.op->getName()
         << " @r" << acc.region << " addr=";
      printSourceList(acc.addr, os);
      if (acc.isWrite) {
        os << " data=";
        printSource(acc.data, os);
      }
      os << "\n";
    }
  }

  for (const StreamChannel &s : this->streams) {
    os << "  chan s" << s.id << ": ";
    printValueName(s.stream, os);
    os << (s.internal  ? " internal"
           : s.isInput ? " in"
                       : " out")
       << " depth=" << s.depth;
    if (auto init = dyn_cast_or_null<ArrayAttr>(s.init))
      os << " init=" << init.size();
    for (const StreamChannel::CallEnd &e : s.callEnds)
      os << (this->calls[e.call].streamArgs[e.arg].isInput ? " get@k"
                                                           : " put@k")
         << e.call;
    os << "\n";
  }

  // The composition graph on the instance substrate: each child's start policy
  // inputs and the predecessors it waits for.
  for (const CallUnit &cu : this->calls) {
    os << "  call k" << cu.id << ": " << cu.callee << " @r" << cu.region
       << " start=" << cu.start << (cu.async ? " spawn" : "")
       << (cu.determinate ? " determinate" : " indeterminate") << " via "
       << startPolicyName(cu.startPolicy);
    if (!cu.predecessors.empty()) {
      os << " after=[";
      llvm::interleaveComma(cu.predecessors, os, [&](const CallUnit::Pred &p) {
        os << "k" << p.call << (p.viaResult ? "(result)" : "");
      });
      os << "]";
    }
    os << "\n";
  }

  for (const ConstCell &c : this->consts)
    os << "  const c" << c.id << ": " << c.value << "\n";

  for (const IOPort &io : this->ios)
    os << "  io i" << io.id << ": in " << io.type << "\n";

  // A region's results, each held for a sibling as a survivor (program order),
  // with the loop-carried identity / else-arm value where the regime has one.
  for (const RegionBlock &rb : this->regions) {
    if (rb.condition) {
      os << "  cond region " << rb.id << " <= ";
      printSource(rb.condition, os);
      os << "\n";
    }
    for (auto [k, r] : llvm::enumerate(rb.results)) {
      os << "  result region " << rb.id << "#" << k << " <= ";
      printSource(r.value, os);
      if (r.init) {
        os << " init=";
        printSource(r.init, os);
      }
      if (r.elseValue) {
        os << " else=";
        printSource(r.elseValue, os);
      }
      os << "\n";
    }
  }

  os << "}\n";
}

} // namespace mlir::allo::uarch
