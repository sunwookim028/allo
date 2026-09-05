/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// The latency arithmetic. Every cycle a region costs is charged here; the two
// structural walks that feed it (`Scheduler.cpp`, `PostConversion.cpp`) report
// shape.
//===----------------------------------------------------------------------===//

#include "allo/Scheduling/LatencyModel.h"

#include "allo/IR/AlloOps.h"
#include "allo/Scheduling/Footprint.h"    // summarizeOp / summarizeCall
#include "allo/Scheduling/MemoryAccess.h" // asMemAccess
#include "allo/Scheduling/MemoryModel.h"  // assignedBankOf, bankLayoutOf
#include "allo/Scheduling/RegionGraph.h"  // isSyncSubKernelCall
#include "allo/Support/AliasAnalysis.h"   // resolveRoot (storage identity)

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;
using namespace mlir::allo;
using namespace mlir::allo::dcp;

// An ODS `I64Attr` accessor hands back `uint64_t`; every count in this model is
// signed.
static std::optional<int64_t> asInt64(std::optional<uint64_t> v) {
  if (v)
    return static_cast<int64_t>(*v);
  return std::nullopt;
}

std::optional<int64_t> mlir::allo::composeSpan(const SpanNode &n) {
  // An instance is a whole start->done contract already; nothing of this func's
  // composes inside it.
  if (n.instance)
    return n.contract;
  // A stall shell stretches a run by whatever back-pressure costs it, so what
  // composes below is a floor and not a contract. Tested ahead of the bound
  // and of a guard's ceiling, a floor being the wrong direction either way.
  if (n.elastic)
    return std::nullopt;
  // A guard's span is a ceiling: it arms, runs whichever arm the predicate
  // takes in sequence, and latches `done`, so the deeper arm bounds both. An
  // empty arm composes to 0. `spanHoldsBound` keeps the result off any exact
  // contract.
  if (n.shape == RegionShape::Guard) {
    std::optional<int64_t> thenSpan = composeSequence(n.children);
    std::optional<int64_t> elseSpan = composeSequence(n.elseChildren);
    if (!thenSpan || !elseSpan)
      return std::nullopt;
    return kGuardBoundary.arm + std::max(*thenSpan, *elseSpan) +
           kDoneLatchCycles;
  }
  // A data-dependent trip has no composable span; a carried bound stands in
  // where the builder judged one usable.
  if (!n.trip)
    return n.assumedSpan;
  if (n.shape == RegionShape::Container || n.shape == RegionShape::CallNode) {
    // A DONE-PACED region runs no schedule of its own, so `drain`/`ii` do not
    // describe it: one pass is its body elements in sequence.
    std::optional<int64_t> body = composeSequence(n.children);
    if (!body)
      return std::nullopt;
    return containerSpan(n.shape == RegionShape::CallNode ? kCallNodeBoundary
                                                          : kContainerBoundary,
                         *n.trip, *body);
  }
  // A LEAF issues on its own controller's cadence and then drains.
  if (!n.drain || (!n.acyclic && !n.ii))
    return std::nullopt;
  const BoundaryCost &boundary =
      !n.acyclic ? kPipelinedBoundary : kAcyclicBoundary;
  return leafSpan(boundary, *n.trip, n.acyclic ? 0 : *n.ii, *n.drain);
}

std::optional<int64_t> mlir::allo::composeSequence(ArrayRef<SpanNode> nodes) {
  int64_t sum = 0;
  for (const SpanNode &n : nodes) {
    std::optional<int64_t> span = composeSpan(n);
    if (!span)
      return std::nullopt;
    sum += *span;
  }
  return sum;
}

bool mlir::allo::spanHoldsBound(const SpanNode &n) {
  // A guard is a ceiling by construction, so its arms need no visit.
  if (n.shape == RegionShape::Guard || n.contractBound ||
      (!n.trip && n.assumedSpan))
    return true;
  return llvm::any_of(n.children,
                      [](const SpanNode &c) { return spanHoldsBound(c); });
}

// Reads and writes of the array bound to operand \p argIdx of \p inv, from the
// reified callee's own accesses, nested instances included. Reification is
// bottom-up, so the callee module exists before any caller composes against it.
static std::pair<bool, bool>
dcpArgAccess(dcp::DCPathInstanceOp inv, unsigned argIdx,
             llvm::SmallPtrSetImpl<Operation *> &active) {
  auto mod = SymbolTable::lookupNearestSymbolFrom<dcp::DCPathModuleOp>(
      inv, inv.getCalleeAttr());
  assert(mod && "a dcp.instance names a reified callee (built bottom-up)");
  bool isNew = active.insert(mod).second;
  assert(isNew && "a dcp module cannot instantiate itself transitively");
  (void)isNew;
  Value arg = mod.getBody().front().getArgument(argIdx);
  bool reads = false, writes = false;
  mod.walk([&](Operation *op) {
    if (auto load = dyn_cast<dcp::DCPathLoadOp>(op)) {
      reads |= resolveRoot(load.getMemref()) == arg;
    } else if (auto store = dyn_cast<dcp::DCPathStoreOp>(op)) {
      writes |= resolveRoot(store.getMemref()) == arg;
    } else if (auto nested = dyn_cast<dcp::DCPathInstanceOp>(op)) {
      for (auto [k, operand] : llvm::enumerate(nested.getInputs()))
        if (isa<MemRefType>(operand.getType()) && resolveRoot(operand) == arg) {
          auto [r, w] = dcpArgAccess(nested, k, active);
          reads |= r;
          writes |= w;
        }
    }
  });
  active.erase(mod);
  return {reads, writes};
}

void mlir::allo::hazardEdges(ArrayRef<MemTouch> touch,
                             llvm::function_ref<void(unsigned, unsigned)> add) {
  auto emit = [&](unsigned p, unsigned c) {
    if (p != c)
      add(p, c);
  };
  llvm::SmallVector<int64_t, 4> banks;
  for (const MemTouch &t : touch)
    if (t.bank && !llvm::is_contained(banks, *t.bank))
      banks.push_back(*t.bank);
  // The reader-writer walk over one bank's touches: a bank-less touch joins
  // every bank's walk, so it pairs with everything.
  auto walk = [&](std::optional<int64_t> bank) {
    std::optional<unsigned> lastWriter;
    llvm::SmallVector<unsigned, 4> readers;
    for (const MemTouch &t : touch) {
      if (bank && t.bank && *t.bank != *bank)
        continue;
      if (lastWriter)
        emit(*lastWriter, t.node);
      if (!t.writes) {
        readers.push_back(t.node);
        continue;
      }
      for (unsigned r : readers)
        emit(r, t.node);
      readers.clear();
      lastWriter = t.node;
    }
  };
  if (banks.empty())
    return walk(std::nullopt);
  for (int64_t b : banks)
    walk(b);
}

// The touches of one sibling node, over either IR form: the scheduler's loops
// and calls, or the reifier's dcp regions and instances. An access op touches
// at the bank `assign-banks` resolved it to; a call or an unclassifiable op
// touches bank-less (it may reach any bank).
namespace {
struct NodeTouches {
  // root -> (bank, writes), merged per (root, bank)
  llvm::MapVector<Value,
                  llvm::SmallVector<std::pair<std::optional<int64_t>, bool>, 2>>
      mem;
  llvm::SmallVector<Value, 2> streams;
};
} // namespace

static NodeTouches nodeTouches(ArrayRef<Operation *> ops) {
  NodeTouches nt;
  auto mem = [&](Value root, std::optional<int64_t> bank, bool writes) {
    auto &list = nt.mem[root];
    for (auto &e : list)
      if (e.first == bank) {
        e.second |= writes;
        return;
      }
    list.push_back({bank, writes});
  };
  auto stream = [&](Value root) {
    if (!llvm::is_contained(nt.streams, root))
      nt.streams.push_back(root);
  };
  auto bankOf = [](Operation *o) -> std::optional<int64_t> {
    if (std::optional<unsigned> b = assignedBankOf(o))
      return int64_t(*b);
    return std::nullopt;
  };
  for (Operation *top : ops)
    top->walk([&](Operation *o) {
      if (auto inv = dyn_cast<dcp::DCPathInstanceOp>(o)) {
        for (auto [k, operand] : llvm::enumerate(inv.getInputs())) {
          if (isa<MemRefType>(operand.getType())) {
            llvm::SmallPtrSet<Operation *, 8> active;
            auto [r, w] = dcpArgAccess(inv, k, active);
            if (r || w)
              mem(resolveRoot(operand), std::nullopt, w);
          } else if (isa<StreamType>(operand.getType())) {
            stream(resolveRoot(operand));
          }
        }
        return;
      }
      if (auto load = dyn_cast<dcp::DCPathLoadOp>(o)) {
        mem(resolveRoot(load.getMemref()), bankOf(o), /*writes=*/false);
        return;
      }
      if (auto store = dyn_cast<dcp::DCPathStoreOp>(o)) {
        mem(resolveRoot(store.getMemref()), bankOf(o), /*writes=*/true);
        return;
      }
      if (auto a = asMemAccess(o)) {
        if (a->kind == AccessKind::Stream) {
          stream(a->root);
          return;
        }
        mem(a->root, bankOf(o), a->isWrite);
        return;
      }
      Summary s;
      if (!(isSyncSubKernelCall(o) && summarizeCall(cast<func::CallOp>(o), s)))
        summarizeOp(o, s);
      for (auto &kv : s.mem)
        if (kv.second.reads || kv.second.writes)
          mem(kv.first, std::nullopt, kv.second.writes);
      for (Value v : s.streams)
        stream(v);
    });
  return nt;
}

std::vector<llvm::SmallVector<unsigned, 2>>
mlir::allo::siblingPredecessors(ArrayRef<SmallVector<Operation *>> nodeOps) {
  // Which node owns each op, so a cross-node SSA use can name the producer's
  // node rather than the producing op.
  DenseMap<Operation *, unsigned> owner;
  for (auto [i, ops] : llvm::enumerate(nodeOps))
    for (Operation *top : ops)
      top->walk([&, i = i](Operation *o) { owner[o] = i; });

  std::vector<llvm::SmallVector<unsigned, 2>> preds(nodeOps.size());
  auto addPred = [&](unsigned p, unsigned c) {
    if (p < c && !llvm::is_contained(preds[c], p))
      preds[c].push_back(p);
  };

  // Every toucher of a shared resource, in node order, with the direction and
  // bank the node touches it at. A stream is a FIFO whose token order is the
  // program's, so its touchers are ordered regardless of direction.
  llvm::MapVector<Value, llvm::SmallVector<MemTouch, 4>> mems;
  llvm::MapVector<Value, llvm::SmallVector<unsigned, 4>> streams;
  for (auto [i, ops] : llvm::enumerate(nodeOps)) {
    NodeTouches nt = nodeTouches(ops);
    for (auto &kv : nt.mem)
      for (auto [bank, writes] : kv.second)
        mems[kv.first].push_back({unsigned(i), writes, bank});
    for (Value stream : nt.streams)
      streams[stream].push_back(i);
  }

  // A scalar survivor: SSA dominance already puts the producer first.
  for (auto [i, ops] : llvm::enumerate(nodeOps))
    for (Operation *top : ops)
      top->walk([&, i = i](Operation *o) {
        for (Value v : o->getOperands()) {
          Operation *def = v.getDefiningOp();
          if (!def)
            continue;
          auto it = owner.find(def);
          if (it != owner.end()) {
            if (it->second != i)
              addPred(it->second, i);
            continue;
          }
          // A def no node owns binds no hardware and so orders nothing: a
          // block partitions entirely into regions, and the only ops outside
          // one are declarations.
          assert(isDeclarationOp(def) &&
                 "a computing op outside every region drives a node's input");
        }
      });

  for (auto &entry : streams)
    for (unsigned j = 1; j < entry.second.size(); ++j)
      addPred(entry.second[j - 1], entry.second[j]);
  // A memref orders only its hazard pairs (`hazardEdges`); two nodes that only
  // read stay unordered, and `Datapath::portGraph` prices the separate ports
  // their overlap takes. A skewed layout is the exception: its lanes share a
  // port per slot across regions, so every toucher pair stays chained.
  for (auto &entry : mems) {
    auto &touch = entry.second;
    if (bankLayoutOf(entry.first).skew()) {
      for (unsigned j = 1; j < touch.size(); ++j)
        addPred(touch[j - 1].node, touch[j].node);
      continue;
    }
    hazardEdges(touch, [&](unsigned p, unsigned c) { addPred(p, c); });
  }
  // `addPred` orders by discovery; hand each list back in program order.
  for (auto &p : preds)
    llvm::sort(p);
  return preds;
}

std::optional<int64_t>
mlir::allo::composeDag(ArrayRef<SpanNode> nodes,
                       ArrayRef<llvm::SmallVector<unsigned, 2>> preds) {
  assert(nodes.size() == preds.size() && "one predecessor set per node");
  int64_t total = 0;
  llvm::SmallVector<int64_t> finish(nodes.size(), 0);
  for (auto [i, n] : llvm::enumerate(nodes)) {
    std::optional<int64_t> span = composeSpan(n);
    if (!span)
      return std::nullopt;
    // A region with no predecessors starts with the kernel; one with them waits
    // on the joined `done` of all of them.
    int64_t start = 0;
    for (unsigned p : preds[i])
      start = std::max(start, finish[p]);
    finish[i] = start + *span;
    total = std::max(total, finish[i]);
  }
  return total;
}

std::vector<SpanNode> mlir::allo::dcpSpanNodes(Block &block, bool topLevel) {
  std::vector<SpanNode> nodes;
  for (Operation &inner : block)
    // An instance inside a region is a body element like any other. At kernel
    // scope there are none, since the reify wraps every call into a region,
    // which keeps this list index-aligned with `siblingPredecessors`.
    if (isa<DCPathRegionOpInterface>(inner) ||
        (!topLevel && isa<DCPathInstanceOp>(inner)))
      nodes.push_back(dcpSpanNode(&inner, topLevel));
  return nodes;
}

SpanNode mlir::allo::dcpSpanNode(Operation *op, bool topLevel) {
  SpanNode n;
  if (auto inv = dyn_cast<DCPathInstanceOp>(op)) {
    // A callee's `latency` is already a start->done contract, counted to its
    // own `done` rising, and the one composed number this side cannot derive.
    // A callee that published only a ceiling (`latency_bound`) is not
    // counted_static, and its contract composes onward as a ceiling.
    n.instance = true;
    n.contract = asInt64(inv.getLatency());
    n.contractBound =
        n.contract && inv.getDeterminacy() != DeterminacyEnum::CountedStatic;
    return n;
  }
  auto region = cast<DCPathRegionOpInterface>(op);
  n.shape = dcpRegionShape(op);
  n.elastic = isElastic(op);
  if (n.shape == RegionShape::Guard) {
    // A `dcp.select`: each arm is a done-paced sequence of its own, composed
    // by `composeSpan`'s ceiling rule.
    auto sel = cast<DCPathSelectOp>(op);
    n.children = dcpSpanNodes(sel.getThenRegion().front(), /*topLevel=*/false);
    if (!sel.getElseRegion().empty())
      n.elseChildren =
          dcpSpanNodes(sel.getElseRegion().front(), /*topLevel=*/false);
    return n;
  }
  if (n.shape == RegionShape::Container || n.shape == RegionShape::CallNode) {
    n.trip = asInt64(region.getTrip());
    n.children = dcpSpanNodes(op->getRegion(0).front(), /*topLevel=*/false);
    return n;
  }
  // `drainTerms` prices a call into the drain from its contract, so a leaf
  // holding one without has no static span: its terminal cycle is a `done`. A
  // bounded contract makes the drain a ceiling instead. Only an acyclic leaf
  // holds a call at all, a cyclic one being a `CallNode`.
  bool waitsOnADone = false;
  for (Operation &inner : op->getRegion(0).front())
    if (auto inv = dyn_cast<DCPathInstanceOp>(&inner)) {
      waitsOnADone |= !inv.getLatency();
      n.contractBound |= inv.getLatency() &&
                         inv.getDeterminacy() != DeterminacyEnum::CountedStatic;
    }
  if (!waitsOnADone)
    n.drain = asInt64(region.getDrain());
  if (isa<DCPathSequentialOp>(op)) {
    n.acyclic = true;
    n.trip = 1;
    return n;
  }
  auto pipe = cast<DCPathPipelineOp>(op);
  if (!pipe.isWhileLoop()) {
    n.trip = asInt64(pipe.getTrip()); // a while leaves it unset: data-dependent
    n.ii = asInt64(pipe.getIi());
  }
  // A dynamic trip carries no `trip` but keeps the scheduler's assume-bounded
  // worst case, which this side cannot re-derive: reification keeps the loop's
  // runtime bound operand, not the assumption that bounded it.
  if (!n.trip && topLevel && region.getLatencyBound())
    n.assumedSpan = asInt64(region.getLatency());
  return n;
}

RegionTiming mlir::allo::dcpRegionTiming(Operation *regionOp) {
  RegionTiming t;
  // CONCURRENT: the region holds a child wired as a process. A concurrent child
  // belongs to its NEAREST enclosing region, the one whose composition operator
  // becomes the network.
  bool concurrent = false;
  regionOp->walk([&](DCPathInstanceOp inv) {
    if (!spawnsConcurrently(inv))
      return;
    Operation *p = inv->getParentOp();
    while (p && !isa<DCPathRegionOpInterface>(p))
      p = p->getParentOp();
    concurrent |= p == regionOp;
  });
  if (concurrent) {
    t.determinacy = DeterminacyEnum::Concurrent;
    return t;
  }
  // CONDITIONAL: a guard or a while. Its own control decides when it ends, so
  // no exact span describes it, though a guard whose arms both compose carries
  // the deeper arm's span as a ceiling.
  auto pipe = dyn_cast<DCPathPipelineOp>(regionOp);
  if (pipe && pipe.isWhileLoop()) {
    t.determinacy = DeterminacyEnum::Conditional;
    return t;
  }
  SpanNode n = dcpSpanNode(regionOp, /*topLevel=*/false);
  std::optional<int64_t> span = composeSpan(n);
  if (isa<DCPathSelectOp>(regionOp)) {
    t.determinacy = DeterminacyEnum::Conditional;
    t.boundedLatency = span;
    return t;
  }
  // COUNTED_STATIC when a span composes exactly, which is the contract a
  // container may time-trigger against. A span holding a ceiling (a guard, a
  // bounded contract) is only waitable, so the region stays INDETERMINATE, as
  // does one with no composable span at all.
  if (span && spanHoldsBound(n)) {
    t.boundedLatency = span;
    return t;
  }
  t.staticLatency = span;
  t.determinacy =
      span ? DeterminacyEnum::CountedStatic : DeterminacyEnum::Indeterminate;
  return t;
}
