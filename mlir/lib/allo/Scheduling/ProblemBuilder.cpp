/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Scheduling/ProblemBuilder.h"

#include "allo/IR/AlloOps.h"
#include "allo/Scheduling/Footprint.h"
#include "allo/Scheduling/MemoryAccess.h" // asMemAccess
#include "allo/Scheduling/OperatorLibrary.h"
#include "allo/Scheduling/RegionGraph.h" // blockHasSyncCall
#include "allo/Scheduling/Scheduler.h"
#include "allo/Support/Logging.h"

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::allo;
using namespace circt::analysis;
using namespace circt::scheduling;

namespace mlir::allo {

// Project a memory dependence's components onto the innermost scheduled loop
// (its last component), setting `drop` when an enclosing loop carries the
// dependence: that loop's sequential execution satisfies it, so it does not
// constrain the innermost modulo schedule.
static unsigned
innermostCarriedDistance(ArrayRef<affine::DependenceComponent> comps,
                         bool &drop) {
  bool valid = true;
  return static_cast<unsigned>(
      carriedDistanceAtLevel(comps, comps.size(), drop, valid));
}

// Whether a dependence is carried by some enclosing loop (a positive distance
// at any level), hence satisfied by that loop's sequential execution and not an
// ordering within a single straight-line instance. An acyclic span has no
// scheduled loop of its own, so it keeps only loop-independent edges.
static bool
isLoopCarriedDependence(ArrayRef<affine::DependenceComponent> comps) {
  for (const affine::DependenceComponent &c : comps)
    if (c.lb.value_or(0) > 0)
      return true;
  return false;
}

// Trace an iter_arg's incoming value to the operation that defines it,
// following any chain of iter_arg-to-iter_arg shifts (accumulator rotation) and
// counting one loop-carried distance per hop.
std::pair<Operation *, unsigned> iterArgSource(Block *body, Operation *yield,
                                               unsigned iterArg) {
  auto v = yield->getOperand(iterArg);
  unsigned distance = 0;
  llvm::SmallDenseSet<unsigned> seen;
  while (auto arg = dyn_cast<BlockArgument>(v)) {
    // iter_args are the body block arguments after the induction variable.
    if (arg.getOwner() != body || arg.getArgNumber() == 0 ||
        !seen.insert(arg.getArgNumber()).second)
      return {nullptr, 0};
    ++distance;
    v = yield->getOperand(arg.getArgNumber() - 1);
  }
  auto *definer = v.getDefiningOp();
  return definer ? std::make_pair(definer, distance + 1)
                 : std::make_pair<Operation *, unsigned>(nullptr, 0);
}

static bool isSyncCall(Operation *op);

// Anchor every remaining dependence-DAG sink to \p anchor with a
// loop-independent (distance-0) edge, making the anchor the unique sink the
// modulo scheduler requires: an unanchored sink is rejected by
// `ModuloSimplexScheduler::checkLastOp`. A sink here is a graph property, any
// op whose consumers are all loop-carried, or a result-less nested terminator.
template <class ProblemT>
static void anchorSinks(ProblemT &problem, Operation *anchor) {
  DenseSet<Operation *> sinks(problem.getOperations().begin(),
                              problem.getOperations().end());
  for (Operation *op : problem.getOperations())
    for (auto &dep : problem.getDependences(op))
      if (problem.getDistance(dep).value_or(0) == 0)
        sinks.erase(dep.getSource());
  // Collect in the problem's insertion order, not the hash set's, so the edges
  // and the solved schedule are deterministic. Snapshot before inserting:
  // `insertDependence` registers its endpoints into the set being iterated.
  SmallVector<Operation *> unanchored;
  for (Operation *op : problem.getOperations())
    if (op != anchor && sinks.contains(op))
      unanchored.push_back(op);
  for (Operation *op : unanchored)
    (void)problem.insertDependence(Problem::Dependence(op, anchor));
}

template <class ProblemT>
ProblemT buildCyclicProblem(LoopLikeOpInterface loop,
                            DependenceAnalysis &deps) {
  ProblemT problem(loop.getOperation());
  Block *body = &loop.getLoopRegions().front()->front();

  // Insert memory and stream dependences into the problem.
  body->walk([&](Operation *op) {
    problem.insertOperation(op);

    for (const MemoryDependence &memoryDep : deps.getDependences(op)) {
      if (!hasDependence(memoryDep.dependenceType))
        continue;

      // Only model dependences whose source is inside this loop: whole-func
      // analysis also surfaces pairs whose endpoints are scheduled elsewhere.
      if (!body->findAncestorOpInBlock(*memoryDep.source))
        continue;

      bool drop = false;
      unsigned distance =
          innermostCarriedDistance(memoryDep.dependenceComponents, drop);
      if (drop)
        continue;

      Problem::Dependence dep(memoryDep.source, op);
      auto depInserted = problem.insertDependence(dep);
      assert(succeeded(depInserted));
      (void)depInserted;

      // One pair may carry both an intra-iteration (dist 0) and a loop-carried
      // edge (`A[2*i]`/`A[i]` alias only at i == 0): keep the smallest distance
      // so the same-iteration ordering survives.
      unsigned cur = problem.getDistance(dep).value_or(distance);
      problem.setDistance(dep, std::min(cur, distance));
    }
  });

  // Insert conditional dependences into the problem.
  body->walk([&](Operation *op) {
    Block *thenBlock = nullptr;
    Block *elseBlock = nullptr;
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      thenBlock = ifOp.thenBlock();
      elseBlock = ifOp.elseBlock();
    } else if (auto ifOp = dyn_cast<AffineIfOp>(op)) {
      thenBlock = ifOp.getThenBlock();
      if (ifOp.hasElse())
        elseBlock = ifOp.getElseBlock();
    } else {
      return WalkResult::advance();
    }

    // No special handling required for control-only `if`s.
    if (op->getNumResults() == 0)
      return WalkResult::skip();

    // Model the implicit value flow from the `yield` to the `if`'s result(s).
    Problem::Dependence depThen(thenBlock->getTerminator(), op);
    auto depInserted = problem.insertDependence(depThen);
    assert(succeeded(depInserted));
    (void)depInserted;

    if (elseBlock) {
      Problem::Dependence depElse(elseBlock->getTerminator(), op);
      depInserted = problem.insertDependence(depElse);
      assert(succeeded(depInserted));
      (void)depInserted;
    }

    return WalkResult::advance();
  });

  // Side-effecting ops (stores, streams, a sync sub-kernel call) must be
  // scheduled before the loop terminator, making it the problem's unique sink.
  auto *anchor = body->getTerminator();
  body->walk([&](Operation *op) {
    if (!isa<AffineStoreOp, memref::StoreOp, StreamGetOp, StreamPutOp>(op) &&
        !isSyncCall(op))
      return;
    Problem::Dependence dep(op, anchor);
    auto depInserted = problem.insertDependence(dep);
    assert(succeeded(depInserted));
    (void)depInserted;
  });

  // Inter-iteration dependences from the definers of the iter_args (the
  // explicitly computed loop-carried values, excluding the induction variable)
  // to their users.
  if (unsigned nIterArgs = anchor->getNumOperands(); nIterArgs > 0) {
    auto iterArgs = loop.getRegionIterArgs();
    for (unsigned i = 0; i < nIterArgs; ++i) {
      auto [definer, distance] = iterArgSource(body, anchor, i);
      if (!definer)
        continue;

      for (Operation *iterArgUser : iterArgs[i].getUsers()) {
        Problem::Dependence dep(definer, iterArgUser);
        auto depInserted = problem.insertDependence(dep);
        assert(succeeded(depInserted));
        (void)depInserted;
        problem.setDistance(dep, distance);
      }
    }
  }

  // Every other sink joins the terminator too. Run last, so the sink set is
  // computed over the finished graph.
  anchorSinks(problem, anchor);

  return problem;
}

bool whileHasIdentityForwarding(scf::WhileOp w) {
  auto &before = w.getBefore().front();
  auto &after = w.getAfter().front();
  auto cond = w.getConditionOp();
  unsigned n = before.getNumArguments();
  if (cond.getArgs().size() != n || after.getNumArguments() != n ||
      w.getYieldOp().getNumOperands() != n)
    return false;
  for (auto [i, arg] : llvm::enumerate(cond.getArgs()))
    if (arg != before.getArgument(i))
      return false;
  return true;
}

bool conditionIsCombinational(scf::WhileOp w, const DeviceModel &dev) {
  // Combinational iff every op in the before region (except the pure-wire
  // `scf.condition`) is 0-latency.
  bool comb = true;
  auto *term = w.getConditionOp().getOperation();
  w.getBefore().walk([&](Operation *op) {
    // A sub-kernel call is timed by its callee and not by any row of `lib`,
    // which will not answer for one. It is fired and awaited over as many
    // cycles as the child takes, which is never the issue cycle.
    if (isSyncSubKernelCall(op)) {
      comb = false;
      return WalkResult::interrupt();
    }
    if (op == term)
      return WalkResult::advance();
    // An access is timed by its storage, everything else by an operator row.
    unsigned latency = asMemAccess(op)
                           ? dev.memory.timing(op).latency
                           : dev.operators.lookup(op).timing.latency;
    if (latency == 0)
      return WalkResult::advance();
    comb = false;
    return WalkResult::interrupt();
  });
  return comb;
}

bool whileFlushingPipelines(scf::WhileOp w, const DeviceModel &dev) {
  for (Region &r : w->getRegions())
    if (r.walk([](Operation *op) {
           return isa<affine::AffineForOp, scf::ForOp, scf::WhileOp>(op)
                      ? WalkResult::interrupt()
                      : WalkResult::advance();
         }).wasInterrupted())
      return false;
  return conditionIsCombinational(w, dev) &&
         !blockHasSyncCall(w.getAfter().front());
}

template <class ProblemT>
ProblemT buildWhileProblem(scf::WhileOp w, DependenceAnalysis &deps) {
  assert(whileHasIdentityForwarding(w) && "while must forward args 1:1");
  ProblemT problem(w.getOperation());
  auto &before = w.getBefore().front();
  auto &after = w.getAfter().front();
  auto condOp = w.getConditionOp();
  auto yieldOp = w.getYieldOp();
  auto *condProducer = condOp.getCondition().getDefiningOp();

  // Register every op in both regions first, so a later-walked back-edge
  // source still resolves. The before terminator (`scf.condition`) is a pure
  // forwarding wire; excluding it keeps `scf.yield` the unique sink.
  before.walk([&](Operation *op) {
    if (op != condOp.getOperation())
      problem.insertOperation(op);
  });
  after.walk([&](Operation *op) { problem.insertOperation(op); });

  // Memory / stream dependences over both regions (intra-`while` only; SSA
  // def-use is modeled implicitly by the problem).
  auto addMemDeps = [&](Block &blk) {
    blk.walk([&](Operation *op) {
      for (const MemoryDependence &memoryDep : deps.getDependences(op)) {
        if (!hasDependence(memoryDep.dependenceType))
          continue;
        if (!w->isProperAncestor(memoryDep.source))
          continue;
        bool drop = false;
        unsigned distance =
            innermostCarriedDistance(memoryDep.dependenceComponents, drop);
        if (drop)
          continue;
        Problem::Dependence dep(memoryDep.source, op);
        if (failed(problem.insertDependence(dep)))
          continue;
        // Keep the tightest distance when the pair also has another edge.
        unsigned cur = problem.getDistance(dep).value_or(distance);
        problem.setDistance(dep, std::min(cur, distance));
      }
    });
  };
  addMemDeps(before);
  addMemDeps(after);

  // Non-speculative condition gate at distance 0: the whole after body waits
  // for the condition, so the state recurrence runs through it (II >= t_cond).
  if (condProducer)
    after.walk([&](Operation *op) {
      (void)problem.insertDependence(Problem::Dependence(condProducer, op));
    });

  // Loop-carried state recurrence: yield operand `j` feeds back one iteration
  // later to the users of before-arg[j] and after-arg[j], the forwarding
  // terminators excluded.
  for (unsigned j = 0, n = before.getNumArguments(); j < n; ++j) {
    auto *definer = yieldOp.getOperand(j).getDefiningOp();
    if (!definer)
      continue; // block-arg / invariant: no recurrence
    SmallVector<Operation *> readers;
    for (Operation *u : before.getArgument(j).getUsers())
      if (u != condOp.getOperation())
        readers.push_back(u);
    for (Operation *u : after.getArgument(j).getUsers())
      if (u != yieldOp.getOperation())
        readers.push_back(u);
    for (Operation *r : readers) {
      Problem::Dependence dep(definer, r);
      if (succeeded(problem.insertDependence(dep)))
        problem.setDistance(dep, 1);
    }
  }

  // Side-effect anchor: stores / streams in the body precede the yield.
  auto *anchor = yieldOp.getOperation();
  after.walk([&](Operation *op) {
    if (isa<AffineStoreOp, memref::StoreOp, StreamGetOp, StreamPutOp>(op))
      (void)problem.insertDependence(Problem::Dependence(op, anchor));
  });

  // Every other sink joins the yield too. The `before` region normally reaches
  // the anchor through the condition gate, which is empty when the condition is
  // a block argument or the after region is bare.
  anchorSinks(problem, anchor);

  return problem;
}

// A plain (non-async) func.call, scheduled as an opaque fixed-latency node. An
// async call composes structurally as dataflow, ordered by its streams.
static bool isSyncCall(Operation *op) {
  return isa<func::CallOp>(op) && !op->hasAttr(kAlloAsyncAttr);
}

template <class ProblemT>
ProblemT buildAcyclicProblem(ArrayRef<Operation *> ops,
                             DependenceAnalysis &deps) {
  assert(!ops.empty() && "straight-line region must have at least one op");
  ProblemT problem(ops.front());

  // Collect the span's op set (all nested ops) for intra-span dep filtering.
  DenseSet<Operation *> spanOps;
  for (Operation *top : ops)
    top->walk([&](Operation *op) { spanOps.insert(op); });

  // Only loop-independent (distance-0) edges are modeled: this problem carries
  // no distance, so a carried edge would falsely close a cycle with the forward
  // edge and make the span infeasible.
  for (Operation *top : ops)
    top->walk([&](Operation *op) {
      problem.insertOperation(op);

      for (const MemoryDependence &memoryDep : deps.getDependences(op)) {
        if (!hasDependence(memoryDep.dependenceType))
          continue;
        // Only intra-span dependences belong to this problem.
        if (!spanOps.contains(memoryDep.source))
          continue;
        if (isLoopCarriedDependence(memoryDep.dependenceComponents))
          continue;
        Problem::Dependence dep(memoryDep.source, op);
        auto depInserted = problem.insertDependence(dep);
        assert(succeeded(depInserted));
        (void)depInserted;
      }
    });

  // DependenceAnalysis misses call ops, so sync calls are ordered by memory
  // footprint instead: a shared write serializes, disjoint or read-only does
  // not, and an opaque callee falls back to a conservative record.
  auto summarize = [](Operation *top, Summary &s) {
    top->walk([&](Operation *op) {
      if (isSyncCall(op) && summarizeCall(cast<func::CallOp>(op), s))
        return;
      summarizeOp(op, s);
    });
  };
  for (unsigned i = 0, e = ops.size(); i < e; ++i)
    for (unsigned j = i + 1; j < e; ++j) {
      if (!isSyncCall(ops[i]) && !isSyncCall(ops[j]))
        continue;
      Summary si, sj;
      summarize(ops[i], si);
      summarize(ops[j], sj);
      for (const auto &kv : si.mem) {
        auto it = sj.mem.find(kv.first);
        if (it != sj.mem.end() &&
            callFootprintConflict(kv.second, it->second) != Conflict::None)
          (void)problem.insertDependence(Problem::Dependence(ops[i], ops[j]));
      }
    }

  // Make the last program-order op a unique sink via auxiliary dependences, so
  // that minimizing its start time yields an ASAP schedule for the whole span.
  auto *sink = ops.back();
  for (Operation *op : problem.getOperations()) {
    if (op == sink)
      continue;
    // Two sync calls are already ordered by the footprint edges above; a
    // blanket edge here would falsely serialize data-independent calls.
    if (isSyncCall(op) && isSyncCall(sink))
      continue;
    (void)problem.insertDependence(Problem::Dependence(op, sink));
  }

  return problem;
}

// Explicit instantiations for the problem types the scheduler pass builds.
template ChainingModuloProblem
buildCyclicProblem<ChainingModuloProblem>(LoopLikeOpInterface,
                                          DependenceAnalysis &);
template ChainingModuloProblem
buildWhileProblem<ChainingModuloProblem>(scf::WhileOp, DependenceAnalysis &);
template ChainingSharedOperatorsProblem
buildAcyclicProblem<ChainingSharedOperatorsProblem>(ArrayRef<Operation *>,
                                                    DependenceAnalysis &);

} // namespace mlir::allo
