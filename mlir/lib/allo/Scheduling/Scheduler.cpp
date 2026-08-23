/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// The SDC scheduling engine: the two rungs this backend solves against, the
// chaining modulo and chaining shared-operators problems. The difference
// constraints are solved on their constraint graph (longest-path potentials,
// incremental under pins); the greedy resource placement around it (the MRT,
// the II growth) remains derived from CIRCT's SimplexSchedulers
// (externals/circt/lib/Scheduling/SimplexSchedulers.cpp), whose parametric
// simplex this engine replaced. The Problem data model and the chaining
// utilities stay CIRCT's.
//
// Portions derived from LLVM/CIRCT, Apache-2.0 WITH LLVM-exception.
//===----------------------------------------------------------------------===//

#include "allo/Scheduling/Scheduler.h"
#include "allo/Support/Logging.h"

#include "circt/Scheduling/Utilities.h"

#include "mlir/IR/Operation.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"

#include <algorithm>
#include <limits>

#define DEBUG_TYPE "allo-simplex-schedulers"

using namespace mlir;
using namespace circt;
using namespace circt::scheduling;
using namespace mlir::allo::logging;
using mlir::allo::ChainingModuloProblem;
using mlir::allo::ChainingSharedOperatorsProblem;
using mlir::allo::ModuloOccupancyProblem;
using mlir::allo::OccupancyProblem;

using llvm::dbgs;
using llvm::format;

namespace {

/// A dependence circuit that binds the initiation interval. `latency` sums each
/// edge's source latency plus the extra cycle a chain-breaking constraint adds;
/// `distance` sums the iterations the edges span.
struct Recurrence {
  SmallVector<Operation *> ops; // the circuit, in dependence order
  int64_t latency = 0;
  int64_t distance = 0;
  explicit operator bool() const { return !ops.empty(); }
};

/// One-line rendering: the circuit as an arrow chain closing on itself,
/// followed by its two sums. The II it forces is `ceil(latency / distance)`.
static std::string render(const Recurrence &rec) {
  std::string s;
  llvm::raw_string_ostream os(s);
  for (Operation *op : rec.ops)
    os << op->getName().getStringRef() << " -> ";
  os << rec.ops.front()->getName().getStringRef() << " (total latency "
     << rec.latency << " over distance " << rec.distance << ")";
  return s;
}

/// Solves the difference-constraint core of a scheduling problem on its
/// constraint graph. Every constraint is
/// `start(dst) - start(src) >= lat + extra - II*dist` (a dependence, a
/// chain break, or a pin), so the feasible set is a lattice: the
/// component-wise least solution exists, is unique, and is the longest-path
/// potential from a virtual origin. That least point simultaneously minimizes
/// the last operation's start and every other start, i.e. exactly the
/// lexicographic (latency, then sum-of-starts) optimum the simplex fork this
/// engine replaced steered to, so the two return the same schedule. The
/// tie-break matters: the emitter builds an operation's start pulse as
/// `delayValid(regionStart, t)`, one flip-flop per cycle of `t`, so a
/// slack-bearing node placed late costs registers for no latency.
///
/// Feasibility is the absence of a positive cycle. The initial solve grows the
/// II to the smallest feasible value (feasibility is monotone in the II: only
/// the `-II*dist` terms move with it); a positive cycle found on the way names
/// the recurrence and the least II it admits. A pin (`scheduleAt`) becomes a
/// virtual edge pair through the origin, applied by an incremental relaxation
/// with an undo log, so a failed pin rolls back in O(touched nodes).
class SDCSchedulerBase {
protected:
  /// The last operation, whose start time the least solution minimizes first.
  Operation *lastOp;

  /// The initiation interval the constraint weights are taken at. The initial
  /// solve grows it to the smallest feasible value; placement grows it more.
  int parameterT = 0;

  /// One difference constraint:
  /// `start(dst) >= start(src) + lat + extra - parameterT * dist`.
  /// `extra` is 1 on a chain-breaking constraint. The weight is derived at
  /// relaxation time, so growing the II rebuilds nothing.
  struct Edge {
    unsigned src, dst;
    int64_t lat;
    unsigned dist;
    int extra;
  };
  /// Dependence and chain-break edges. A pin is a VIRTUAL edge pair through
  /// the origin (`frozenVariables`), added and removed without touching this.
  SmallVector<Edge> edges;
  /// Out-adjacency into `edges` by variable. The origin's outgoing edges are
  /// the pins' lower bounds and the implicit `start >= 0`, which seeding every
  /// potential at zero already applies.
  SmallVector<SmallVector<unsigned, 4>> staticOut;

  /// The virtual origin variable (index == number of operations): the zero
  /// every bound is stated against.
  unsigned origin = 0;

  /// The least solution: longest path from the origin, kept current across
  /// pins.
  SmallVector<int64_t> potentials;
  /// Whether `potentials` reflect the current constraints; only a frozen
  /// variable may be read while this is false.
  bool potentialsCurrent = false;

  /// Pinned start times, in insertion order so a diagnostic's cycle is
  /// deterministic. A pin is the edge pair origin->v (weight t) and v->origin
  /// (weight -t).
  llvm::MapVector<unsigned, unsigned> frozenVariables;

  /// An operation's start time variable id, in problem order.
  DenseMap<Operation *, unsigned> startTimeVariables;
  /// The inverse, for naming a cycle's operations.
  SmallVector<Operation *> varOps;

  /// Allow subclasses to collect additional constraints that are not part of
  /// the input problem (the chain breaks), each one time step stronger than a
  /// plain dependence.
  SmallVector<Problem::Dependence> additionalConstraints;

  virtual Problem &getProblem() = 0;
  /// The problem as the resource layer sees it, where the forwarded-edge set
  /// lives; set by the concrete schedulers.
  OccupancyProblem *occupancy = nullptr;
  /// The latency a dependence separates its endpoints by: its source's, or
  /// zero for a forwarded store->load edge.
  int64_t sourceLatencyOf(Problem::Dependence dep);
  /// Iteration distance a dependence spans. The base answers 0 (the acyclic
  /// `distance == 0` special case); the cyclic subclass overrides.
  virtual unsigned distanceOf(Problem::Dependence dep);
  /// The dependence circuit that binds the II at \p ii: the constraints are
  /// `t_dst - t_src >= latency(src) + extra - ii*distance`, so a schedule
  /// exists iff no circuit's weights sum positive. A positive circuit forces
  /// `ii >= ceil(latency / distance)`, and one with `distance == 0` can never
  /// be satisfied. Empty when no circuit binds. O(|ops| * |deps|) Bellman-Ford.
  Recurrence bindingRecurrence(unsigned ii);
  /// Report a failed initial solve, naming the recurrence responsible.
  void reportInfeasible();
  virtual LogicalResult checkLastOp();

  /// Build the variables and edges off the problem + `additionalConstraints`.
  void buildGraph();

  /// Recompute the least solution from scratch. With \p allowRaise, a
  /// positive cycle carrying iteration distance raises the II to the least
  /// value it admits and retries (the parametric-T bump of the simplex);
  /// without it, or on a zero-distance cycle, the system is infeasible.
  LogicalResult solveGraph(bool allowRaise);

  /// Pin \p startTimeVariable to \p timeStep, updating the least solution by
  /// an incremental relaxation. Failure (a positive cycle through the pin)
  /// rolls everything back, leaving the previous solution standing.
  /// \p conflictPins, non-null, receives the other pins on the certifying
  /// cycle: the ones a placement repair would evict.
  LogicalResult scheduleAt(unsigned startTimeVariable, unsigned timeStep,
                           SmallVectorImpl<unsigned> *conflictPins = nullptr);

  unsigned getStartTime(unsigned startTimeVariable);

  /// ASAP is the maintained least solution; ALAP is the greatest solution
  /// with the last operation held at its already-minimal start, the pair the
  /// simplex read off its negated objective row. Both extremes of the lattice
  /// are unique, so the two engines' margins agree. The greatest solution is
  /// a longest path TO the origin over the same edges (a path i -> origin of
  /// weight W states `start(i) <= -W`), a temporary pin on the last operation
  /// supplying every unpinned chain's upper bound through the anchor.
  void computeMargins(SmallVectorImpl<unsigned> &asap,
                      SmallVectorImpl<unsigned> &alap);

  /// A restorable copy of the engine's mutable state for a speculative
  /// transform (the targeted II growth): the static edges never change, so
  /// the pins, the potentials and the interval are the whole of it.
  struct GraphState {
    llvm::MapVector<unsigned, unsigned> frozenVariables;
    SmallVector<int64_t> potentials;
    int parameterT;
    bool potentialsCurrent;
  };
  GraphState saveState();
  void restoreState(GraphState &saved);

private:
  /// A positive cycle: its constraint sums, and its operation nodes for the
  /// diagnostic (the origin, where a pin closes it, is skipped).
  struct FoundCycle {
    int64_t lat = 0;
    int64_t dist = 0;
    SmallVector<unsigned> nodes;
  };
  /// Monotone longest-path relaxation from the queued seeds over the static
  /// edges plus the pins' virtual edges. Returns the cycle that certifies the
  /// system infeasible at the current interval, or nullopt at the fixpoint.
  /// \p undo, non-null, records each first-touched potential for rollback.
  /// `resetScratch` must run first; a caller may seed pred entries after it.
  std::optional<FoundCycle>
  relaxCore(SmallVector<unsigned> queue,
            SmallVectorImpl<std::pair<unsigned, int64_t>> *undo);
  /// The cycle in the pred bookkeeping reachable from \p v: a proper cycle,
  /// or a walk off the origin that closes through an implicit zero edge.
  FoundCycle extractCycle(unsigned v);
  void resetScratch();

  /// Relaxation scratch: the last relaxing edge per node (its source, its
  /// `lat + extra`, its distance), the provenance chain length whose
  /// overflow past the node count certifies a positive cycle (a feasible
  /// least solution needs no chain longer than that; a longer one repeats a
  /// node, and the repeated segment sums positive), the queue membership,
  /// and the first-touch marks feeding an undo log.
  SmallVector<int> predNode;
  SmallVector<int64_t> predLat;
  SmallVector<unsigned> predDist;
  SmallVector<unsigned> chainLen;
  SmallVector<uint8_t> inQueue;
  SmallVector<uint8_t> touchedFlag;

public:
  explicit SDCSchedulerBase(Operation *lastOp) : lastOp(lastOp) {}
  virtual ~SDCSchedulerBase() = default;
  virtual LogicalResult schedule() = 0;
};


// This class solves acyclic, resource-constrained `OccupancyProblem` with
// LP-guided first-fit placement, after de Dinechin.
class SharedOperatorsSimplexScheduler : public SDCSchedulerBase {
private:
  OccupancyProblem &prob;

protected:
  Problem &getProblem() override { return prob; }

public:
  SharedOperatorsSimplexScheduler(OccupancyProblem &prob, Operation *lastOp)
      : SDCSchedulerBase(lastOp), prob(prob) {
    occupancy = &prob;
  }
  LogicalResult schedule() override;
};

/// What set the resource-min II: the pool that needed the most cycles, its
/// demand against its per-cycle limit, and one operation holding it, so a
/// diagnostic can point at source rather than at an internal resource key.
struct BindingResource {
  circt::scheduling::Problem::ResourceType rsrc;
  unsigned demand = 0, limit = 0;
  Operation *witness = nullptr;
};

// This class solves the `ModuloOccupancyProblem` with the iterative modulo
// heuristic of de Dinechin, "Fast Modulo Scheduling Under the Simplex
// Scheduling Framework", PRISM 1995.01, plus budgeted eviction after Rau's
// iterative modulo scheduling, which repairs a failed placement at the
// current II before any growth.
class ModuloSimplexScheduler : public SDCSchedulerBase {
private:
  struct MRT {
    ModuloSimplexScheduler &sched;

    // Modulo slot -> number of resource instances occupied there. A count (not
    // a set of ops) so a non-pipelined window wider than the II, which wraps
    // and lands in a slot more than once, contributes its true multiplicity.
    using TableType = SmallDenseMap<unsigned, unsigned>;
    using ReverseTableType = SmallDenseMap<Operation *, unsigned>;
    SmallDenseMap<Problem::ResourceType, TableType> tables;
    SmallDenseMap<Problem::ResourceType, ReverseTableType> reverseTables;

    explicit MRT(ModuloSimplexScheduler &sched) : sched(sched) {}
    LogicalResult enter(Operation *op, unsigned timeStep);
    void release(Operation *op);
    void clear() {
      tables.clear();
      reverseTables.clear();
    }
  };

  ModuloOccupancyProblem &prob;
  SmallVector<unsigned> asapTimes, alapTimes;
  SmallVector<Operation *> unscheduled, scheduled;
  MRT mrt;
  // Lower bound on the II from a pipeline directive. The search only ever grows
  // the II, so the achieved II is max(this, the natural minimum).
  unsigned minII = 1;
  // Sum of occupancies over limited ops; the II growth must converge within
  // this bound (all ops fit in disjoint windows by then).
  unsigned totalResourceCycles = 0;
  // The largest II any bound justifies before resources are placed: the
  // resource-min II, a loop-carried recurrence, and the pipeline directive's
  // floor, whichever is largest. Greedy placement can only grow the II past it.
  unsigned lowerBoundII = 1;
  // Whether the resource-free solve that settles `lowerBoundII` got that far.
  bool boundSettled = false;
  // Whether a caller places this region itself if the greedy cannot (see
  // `SimplexWarmStart`). It changes only what a placement failure is reported
  // AS, never what the placement does.
  bool placementAdvisory = false;
  // Placement repair bookkeeping: how often each operation was evicted, and
  // the evictions the region may still spend. Both caps keep the repair
  // finite; exhaustion falls back to growing the II.
  DenseMap<Operation *, unsigned> evictCount;
  unsigned evictionBudget = 0;
  static constexpr unsigned kMaxEvictionsPerOp = 6;
  static constexpr unsigned kMaxCommitAttempts = 16;
  static constexpr unsigned kMaxDepRounds = 6;
  static constexpr unsigned kMaxSpanRepairMoves = 64;

protected:
  Problem &getProblem() override { return prob; }
  unsigned distanceOf(Problem::Dependence dep) override {
    return prob.getDistance(dep).value_or(0);
  }
  LogicalResult checkLastOp() override;
  void updateMargins();
  LogicalResult scheduleOperation(Operation *n);
  LogicalResult scheduleWithEviction(Operation *n);
  LogicalResult growIIAndRestart(Operation *n);
  /// After placement, lower the region's span: relocate critical limited ops
  /// first fit seated at a free-but-high slot down to a lower class, evicting
  /// the holders in the way, committing only strictly-improving moves.
  void repairSpan();
  /// Try to reseat the pinned op \p stvX below \p oldPin by evicting the
  /// holders of a lower class; victims are re-placed by first fit. Returns true
  /// if the op ended up below oldPin with every victim replaced (the caller
  /// checks the span).
  bool trySeatLower(unsigned stvX, unsigned oldPin, unsigned &budget);
  /// The least start \p stv could take from its dependences alone, given the
  /// other pins: a pin above this was pushed up by first fit's slot choice.
  int64_t depAsapOf(unsigned stv);
  /// The fewest cycles one iteration's resource demand can be issued in.
  /// \p binding receives what set it, untouched where nothing does.
  unsigned computeResMinII(BindingResource &binding);

public:
  ModuloSimplexScheduler(ModuloOccupancyProblem &prob, Operation *lastOp,
                         unsigned minII = 1)
      : SDCSchedulerBase(lastOp), prob(prob), mrt(*this), minII(minII) {
    occupancy = &prob;
  }
  LogicalResult schedule() override;
  /// See `lowerBoundII`. Settled before placement, so it is meaningful even
  /// after `schedule` fails, but only once `hasLowerBound` holds.
  unsigned getLowerBoundII() const { return lowerBoundII; }
  bool hasLowerBound() const { return boundSettled; }
  void setPlacementAdvisory() { placementAdvisory = true; }
};

// This class solves the resource-constrained, cyclic, chaining-enabled
// `ChainingModuloProblem` on top of the `ModuloSimplexScheduler`: a pre-pass
// fills the chain-breaking dependences (consumed by `buildGraph`), and a
// post-pass fills the sub-cycle start times.
class ChainingModuloSimplexScheduler : public ModuloSimplexScheduler {
private:
  ChainingModuloProblem &prob;
  float cycleTime;
  float regFloor;

protected:
  Problem &getProblem() override { return prob; }
public:
  ChainingModuloSimplexScheduler(ChainingModuloProblem &prob, Operation *lastOp,
                                 float cycleTime, float regFloor,
                                 unsigned minII = 1)
      : ModuloSimplexScheduler(prob, lastOp, minII), prob(prob),
        cycleTime(cycleTime), regFloor(regFloor) {}
  LogicalResult schedule() override {
    if (failed(mlir::allo::computeChainBreaks(prob, cycleTime, regFloor,
                                              additionalConstraints)))
      return failure();
    if (!additionalConstraints.empty())
      info(Stage::Sched, prob.getContainingOp())
          << "Split " << additionalConstraints.size()
          << " combinational chain(s) to meet the " << format("%g", cycleTime)
          << " ns clock period (adding pipeline register stages / latency)";
    if (failed(ModuloSimplexScheduler::schedule()))
      return failure();
    return mlir::allo::computeStartTimesInCycle(prob, regFloor);
  }
};

// This class solves the resource-constrained, acyclic, chaining-enabled
// `ChainingSharedOperatorsProblem` on top of the
// `SharedOperatorsSimplexScheduler`. The acyclic mirror of
// `ChainingModuloSimplexScheduler`.
class ChainingSharedOperatorsSimplexScheduler
    : public SharedOperatorsSimplexScheduler {
private:
  ChainingSharedOperatorsProblem &prob;
  float cycleTime;
  float regFloor;

protected:
  Problem &getProblem() override { return prob; }
public:
  ChainingSharedOperatorsSimplexScheduler(ChainingSharedOperatorsProblem &prob,
                                          Operation *lastOp, float cycleTime,
                                          float regFloor)
      : SharedOperatorsSimplexScheduler(prob, lastOp), prob(prob),
        cycleTime(cycleTime), regFloor(regFloor) {}
  LogicalResult schedule() override {
    if (failed(mlir::allo::computeChainBreaks(prob, cycleTime, regFloor,
                                              additionalConstraints)))
      return failure();
    if (!additionalConstraints.empty())
      info(Stage::Sched, prob.getContainingOp())
          << "Split " << additionalConstraints.size()
          << " combinational chain(s) to meet the " << format("%g", cycleTime)
          << " ns clock period (adds pipeline register stages / latency)";
    if (failed(SharedOperatorsSimplexScheduler::schedule()))
      return failure();
    return mlir::allo::computeStartTimesInCycle(prob, regFloor);
  }
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Chain breaking
//===----------------------------------------------------------------------===//

LogicalResult mlir::allo::computeStartTimesInCycle(ChainingProblem &prob,
                                                   float regFloor) {
  prob.clearStartTimeInCycle();
  return handleOperationsInTopologicalOrder(prob, [&](Operation *op) {
    // The floor, not zero: an operand reaches `op` no earlier than its own
    // register can drive it.
    float startTimeInCycle = regFloor;
    unsigned startTime = *prob.getStartTime(op);

    for (auto dep : prob.getDependences(op)) {
      if (dep.isAuxiliary()) // carries no value
        continue;
      Operation *pred = dep.getSource();
      auto predStartTimeInCycle = prob.getStartTimeInCycle(pred);
      if (!predStartTimeInCycle)
        return failure(); // a predecessor is still pending

      auto predOpr = *prob.getLinkedOperatorType(pred);
      unsigned predEnd = *prob.getStartTime(pred) + *prob.getLatency(predOpr);
      if (predEnd < startTime)
        continue; // registered a whole step earlier

      // `pred` ends in the cycle `op` starts in. A multi-cycle producer
      // contributes only its outgoing delay, its last register stage being
      // what the cycle starts from.
      float predEndInCycle =
          (*prob.getStartTime(pred) == predEnd ? *predStartTimeInCycle : 0.0f) +
          *prob.getOutgoingDelay(predOpr);
      startTimeInCycle = std::max(predEndInCycle, startTimeInCycle);
    }

    prob.setStartTimeInCycle(op, startTimeInCycle);
    return success();
  });
}

LogicalResult
mlir::allo::computeChainBreaks(ChainingProblem &prob, float cycleTime,
                               float regFloor,
                               SmallVectorImpl<Problem::Dependence> &result) {
  // Every operator fits a cycle of its own: `runSDCScheduler` raises the
  // period to the least every row does before any problem is built, so a
  // violation here is an operation the derate walk did not price.
  assert(llvm::all_of(prob.getOperatorTypes(),
                      [&](Problem::OperatorType opr) {
                        return regFloor + *prob.getIncomingDelay(opr) <=
                                   cycleTime &&
                               *prob.getOutgoingDelay(opr) <= cycleTime;
                      }) &&
         "an operator exceeds the derated period; `minSchedulablePeriod` "
         "prices every operation a problem registers");

  // chains[v][u]: the delay arriving at `v` along the longest combinational
  // chain starting at `u`. A key is also the "handled" marker, so nothing is
  // written for an operation until every predecessor of it is complete.
  DenseMap<Operation *, SmallDenseMap<Operation *, float>> chains;

  // Problem order, which is the IR's. `chains` is keyed by pointer, so its
  // iteration order is one of ADDRESSES, and the edges below would otherwise
  // vary between two compiles of one kernel.
  DenseMap<Operation *, unsigned> order;
  for (Operation *op : prob.getOperations())
    order.try_emplace(op, order.size());

  return handleOperationsInTopologicalOrder(prob, [&](Operation *op) {
    for (auto dep : prob.getDependences(op))
      if (dep.isDefUse() && !chains.count(dep.getSource()))
        return failure(); // a predecessor is still pending; retry `op` later

    // `op` is the origin of its own chain, and every chain arriving at it is
    // one of its combinational predecessors' extended by that predecessor. A
    // chain starts at the floor, not at zero: its operands leave a register.
    chains[op][op] = regFloor;
    for (auto dep : prob.getDependences(op)) {
      if (!dep.isDefUse()) // an auxiliary edge transports no value
        continue;
      Operation *pred = dep.getSource();
      auto predOpr = *prob.getLinkedOperatorType(pred);
      float outgoing = *prob.getOutgoingDelay(predOpr);
      if (*prob.getLatency(predOpr) > 0) {
        // Registered: the chain restarts at `pred` carrying its output delay,
        // maxed against any longer chain that also reaches here through `pred`,
        // and against the floor, which is `pred`'s own clock-to-out.
        chains[op][pred] =
            std::max(chains[op][pred], std::max(regFloor, outgoing));
        continue;
      }
      for (auto [origin, delay] : chains[pred])
        chains[op][origin] = std::max(delay + outgoing, chains[op][origin]);
    }

    // Break every chain `op` cannot be appended to within the period. Erasing
    // it here is what keeps `op`'s successors from inheriting a chain the edge
    // has just cut.
    float incoming = *prob.getIncomingDelay(*prob.getLinkedOperatorType(op));
    SmallVector<Operation *, 4> tooLong;
    for (auto [origin, delay] : chains[op])
      if (delay + incoming > cycleTime)
        tooLong.push_back(origin);
    llvm::sort(tooLong, [&](Operation *a, Operation *b) {
      return order.at(a) < order.at(b);
    });
    for (Operation *origin : tooLong) {
      result.emplace_back(origin, op);
      chains[op].erase(origin);
    }
    return success();
  });
}

//===----------------------------------------------------------------------===//
// SDCSchedulerBase
//===----------------------------------------------------------------------===//

unsigned SDCSchedulerBase::distanceOf(Problem::Dependence) { return 0; }

int64_t SDCSchedulerBase::sourceLatencyOf(Problem::Dependence dep) {
  if (occupancy && occupancy->isForwarded(dep))
    return 0;
  auto &prob = getProblem();
  return *prob.getLatency(*prob.getLinkedOperatorType(dep.getSource()));
}

Recurrence SDCSchedulerBase::bindingRecurrence(unsigned ii) {
  auto &prob = getProblem();
  DenseMap<Operation *, unsigned> index;
  SmallVector<Operation *> nodes;
  for (auto *op : prob.getOperations()) {
    index[op] = nodes.size();
    nodes.push_back(op);
  }

  // One edge per constraint the engine builds, carrying the same
  // latency / distance / chain-break terms.
  struct Edge {
    unsigned src, dst;
    int64_t latency, distance;
  };
  SmallVector<Edge> edges;
  auto weightOf = [&](const Edge &e) {
    return e.latency - static_cast<int64_t>(ii) * e.distance;
  };
  auto addEdge = [&](Problem::Dependence dep, int extra) {
    auto srcIt = index.find(dep.getSource());
    auto dstIt = index.find(dep.getDestination());
    if (srcIt == index.end() || dstIt == index.end())
      return;
    int64_t latency = sourceLatencyOf(dep) + extra;
    edges.push_back({srcIt->second, dstIt->second, latency, distanceOf(dep)});
  };
  for (auto *op : prob.getOperations())
    for (auto &dep : prob.getDependences(op))
      addEdge(dep, /*extra=*/0);
  // A chain-breaking constraint costs one extra time step.
  for (auto &dep : additionalConstraints)
    addEdge(dep, /*extra=*/1);

  // Bellman-Ford for a positive circuit, every node a source (`dist` starts at
  // zero) so a circuit anywhere in the graph is found. Settling early means
  // there is none.
  SmallVector<int64_t> dist(nodes.size(), 0);
  SmallVector<int> pred(nodes.size(), -1), predEdge(nodes.size(), -1);
  int relaxed = -1;
  for (unsigned round = 0; round < nodes.size(); ++round) {
    relaxed = -1;
    for (auto [e, edge] : llvm::enumerate(edges))
      if (dist[edge.src] + weightOf(edge) > dist[edge.dst]) {
        dist[edge.dst] = dist[edge.src] + weightOf(edge);
        pred[edge.dst] = edge.src;
        predEdge[edge.dst] = e;
        relaxed = edge.dst;
      }
    if (relaxed < 0)
      return {}; // settled: every circuit's weights sum non-positive
  }

  // A node still relaxing after |ops| rounds is reachable from a positive
  // circuit; |ops| predecessor steps land inside the circuit itself.
  unsigned v = relaxed;
  for (unsigned i = 0; i < nodes.size(); ++i) {
    if (pred[v] < 0)
      return {};
    v = pred[v];
  }
  Recurrence rec;
  for (unsigned u = v;;) {
    rec.ops.push_back(nodes[u]);
    const Edge &in = edges[predEdge[u]];
    rec.latency += in.latency;
    rec.distance += in.distance;
    u = pred[u];
    if (u == v)
      break;
  }
  std::reverse(rec.ops.begin(), rec.ops.end());
  return rec;
}

void SDCSchedulerBase::reportInfeasible() {
  auto &prob = getProblem();
  // The initial solve grows the II freely, so failing it means no II works:
  // some circuit carries positive latency over zero distance. Search at an II
  // large enough that any distance-carrying circuit is comfortably negative.
  unsigned bigII = 1 + additionalConstraints.size();
  for (auto *op : prob.getOperations())
    if (auto opr = prob.getLinkedOperatorType(op))
      bigII += prob.getLatency(*opr).value_or(0);
  Recurrence rec = bindingRecurrence(bigII);
  auto diag =
      error(Stage::Sched, Code::DependenceInfeasible, prob.getContainingOp());
  if (!rec) {
    // No circuit binds, so the infeasibility comes from the constraints layered
    // on top of the dependences (a fixed start time, a resource reservation).
    diag << "Problem is infeasible: no dependence recurrence explains it, so a "
            "fixed start time or a resource reservation does";
    return;
  }
  diag << "Problem is infeasible: the dependence cycle " << render(rec)
       << " must complete within one iteration, but takes " << rec.latency
       << " cycle(s); break it with a loop-carried value (an iter-arg), a "
          "faster operator, or an allo.assume.nodep hint if the dependence is "
          "spurious";
}

LogicalResult SDCSchedulerBase::checkLastOp() {
  auto &prob = getProblem();
  if (!prob.hasOperation(lastOp)) {
    assert(false && "the scheduling problem does not include its last "
                    "operation; ProblemBuilder constructs both, so no input "
                    "can reach this");
    return failure();
  }
  return success();
}

void SDCSchedulerBase::buildGraph() {
  auto &prob = getProblem();
  for (auto *op : prob.getOperations()) {
    startTimeVariables[op] = varOps.size();
    varOps.push_back(op);
  }
  origin = varOps.size();
  auto addEdge = [&](Problem::Dependence dep, int extra) {
    // A self-arc stays: its row constrains no start, but a positive weight is
    // a one-node circuit forcing `II >= ceil(lat / dist)` like any other.
    edges.push_back({startTimeVariables[dep.getSource()],
                     startTimeVariables[dep.getDestination()],
                     sourceLatencyOf(dep), distanceOf(dep), extra});
  };
  for (auto *op : prob.getOperations())
    for (auto &dep : prob.getDependences(op))
      addEdge(dep, 0);
  for (auto &dep : additionalConstraints)
    addEdge(dep, 1);
  staticOut.assign(origin + 1, {});
  for (auto [i, e] : llvm::enumerate(edges))
    staticOut[e.src].push_back(i);
  potentials.assign(origin + 1, 0);
}

void SDCSchedulerBase::resetScratch() {
  unsigned nNodes = origin + 1;
  predNode.assign(nNodes, -1);
  predLat.assign(nNodes, 0);
  predDist.assign(nNodes, 0);
  chainLen.assign(nNodes, 0);
  inQueue.assign(nNodes, 0);
  touchedFlag.assign(nNodes, 0);
}

SDCSchedulerBase::FoundCycle SDCSchedulerBase::extractCycle(unsigned v) {
  SmallVector<uint8_t> stamp(origin + 1, 0);
  unsigned x = v;
  while (predNode[x] >= 0 && !stamp[x]) {
    stamp[x] = 1;
    x = unsigned(predNode[x]);
  }
  FoundCycle cyc;
  if (predNode[x] < 0) {
    // The chain runs off a zero-seeded node: only a walk started AT the
    // origin closes that way, through the implicit `start >= 0` edge. Sum
    // every walked node's incoming edge; the implicit closure adds nothing.
    assert(v == origin && "a chain-length trip walks into a proper cycle");
    for (unsigned y = v; predNode[y] >= 0; y = unsigned(predNode[y])) {
      cyc.lat += predLat[y];
      cyc.dist += predDist[y];
      if (y != origin)
        cyc.nodes.push_back(y);
    }
    if (x != origin)
      cyc.nodes.push_back(x);
    return cyc;
  }
  for (unsigned y = x;;) {
    cyc.lat += predLat[y];
    cyc.dist += predDist[y];
    if (y != origin)
      cyc.nodes.push_back(y);
    y = unsigned(predNode[y]);
    if (y == x)
      break;
  }
  return cyc;
}

std::optional<SDCSchedulerBase::FoundCycle>
SDCSchedulerBase::relaxCore(SmallVector<unsigned> queue,
                            SmallVectorImpl<std::pair<unsigned, int64_t>> *undo) {
  unsigned nNodes = origin + 1;
  std::optional<unsigned> trip;
  for (unsigned head = 0; head < queue.size() && !trip; ++head) {
    unsigned u = queue[head];
    inQueue[u] = 0;
    auto relax = [&](unsigned dst, int64_t lat, unsigned dist) {
      int64_t np = potentials[u] + lat - int64_t(parameterT) * dist;
      if (np <= potentials[dst])
        return;
      if (undo && !touchedFlag[dst]) {
        touchedFlag[dst] = 1;
        undo->push_back({dst, potentials[dst]});
      }
      potentials[dst] = np;
      predNode[dst] = int(u);
      predLat[dst] = lat;
      predDist[dst] = dist;
      chainLen[dst] = chainLen[u] + 1;
      // The origin is the zero every bound is stated against: raising it
      // closes a positive cycle through a pin. A provenance chain longer than
      // the node count certifies one anywhere else.
      if ((dst == origin && np > 0) || chainLen[dst] > nNodes) {
        trip = dst;
        return;
      }
      if (!inQueue[dst]) {
        inQueue[dst] = 1;
        queue.push_back(dst);
      }
    };
    for (unsigned ei : staticOut[u]) {
      const Edge &e = edges[ei];
      relax(e.dst, e.lat + e.extra, e.dist);
      if (trip)
        break;
    }
    if (trip)
      break;
    if (u == origin) {
      for (auto &[v, t] : frozenVariables) {
        relax(v, int64_t(t), 0);
        if (trip)
          break;
      }
    } else if (const auto *it = frozenVariables.find(u);
               it != frozenVariables.end()) {
      relax(origin, -int64_t(it->second), 0);
    }
  }
  if (!trip)
    return std::nullopt;
  return extractCycle(*trip);
}

LogicalResult SDCSchedulerBase::solveGraph(bool allowRaise) {
  potentialsCurrent = false;
  unsigned nNodes = origin + 1;
  // Past this interval every distance-carrying circuit sums negative, so a
  // still-infeasible system is a zero-distance circuit no interval fixes.
  int64_t latBound = 1;
  for (const Edge &e : edges)
    latBound += (e.lat < 0 ? -e.lat : e.lat) + e.extra;
  for (auto &[v, t] : frozenVariables)
    latBound += t;
  while (true) {
    potentials.assign(nNodes, 0);
    resetScratch();
    SmallVector<unsigned> queue;
    queue.reserve(nNodes);
    for (unsigned v = 0; v < nNodes; ++v) {
      queue.push_back(v);
      inQueue[v] = 1;
    }
    std::optional<FoundCycle> cyc = relaxCore(std::move(queue), nullptr);
    if (!cyc) {
      potentialsCurrent = true;
      return success();
    }
    int64_t cw = cyc->lat - int64_t(parameterT) * cyc->dist;
    if (!allowRaise || (cyc->dist == 0 && cw > 0) || parameterT > latBound)
      return failure();
    // Never past the least feasible interval: the trip certifies the current
    // one infeasible, and every real circuit independently requires
    // `ceil(lat / dist)`, so the climb lands exactly on the minimum.
    int64_t newT = parameterT + 1;
    if (cyc->dist > 0 && cyc->lat > 0)
      newT = std::max(newT, (cyc->lat + cyc->dist - 1) / cyc->dist);
    {
      auto diag = info(Stage::Sched, getProblem().getContainingOp());
      diag << "II=" << parameterT
           << " is not achievable: a loop-carried recurrence requires II>="
           << newT << ", increasing II to " << newT;
      if (!cyc->nodes.empty()) {
        Recurrence rec;
        for (unsigned v : llvm::reverse(cyc->nodes))
          rec.ops.push_back(varOps[v]);
        rec.latency = cyc->lat;
        rec.distance = cyc->dist;
        diag << "; the binding recurrence is " << render(rec);
      }
    }
    parameterT = int(newT);
  }
}

LogicalResult
SDCSchedulerBase::scheduleAt(unsigned startTimeVariable, unsigned timeStep,
                             SmallVectorImpl<unsigned> *conflictPins) {
  assert(startTimeVariable < origin);
  assert(!frozenVariables.count(startTimeVariable));
  assert(potentialsCurrent && "a pin lands on a solved system");
  int64_t t = timeStep;
  if (potentials[startTimeVariable] > t)
    return failure(); // already forced later than the pin allows
  frozenVariables.insert({startTimeVariable, timeStep});
  if (potentials[startTimeVariable] == t)
    return success(); // the solution already sits on the pin
  SmallVector<std::pair<unsigned, int64_t>> undo;
  undo.push_back({startTimeVariable, potentials[startTimeVariable]});
  potentials[startTimeVariable] = t;
  resetScratch();
  touchedFlag[startTimeVariable] = 1;
  predNode[startTimeVariable] = int(origin);
  predLat[startTimeVariable] = t;
  predDist[startTimeVariable] = 0;
  chainLen[startTimeVariable] = 1;
  inQueue[startTimeVariable] = 1;
  if (auto cyc = relaxCore({startTimeVariable}, &undo)) {
    for (auto &[node, old] : llvm::reverse(undo))
      potentials[node] = old;
    frozenVariables.pop_back();
    if (conflictPins)
      for (unsigned v : cyc->nodes)
        if (frozenVariables.count(v))
          conflictPins->push_back(v);
    return failure();
  }
  return success();
}

unsigned SDCSchedulerBase::getStartTime(unsigned startTimeVariable) {
  assert(startTimeVariable < origin);
  if (const auto *it = frozenVariables.find(startTimeVariable);
      it != frozenVariables.end())
    return it->second;
  assert(potentialsCurrent && "an unpinned start is read off a solved system");
  return unsigned(potentials[startTimeVariable]);
}

SDCSchedulerBase::GraphState SDCSchedulerBase::saveState() {
  return {frozenVariables, potentials, parameterT, potentialsCurrent};
}

void SDCSchedulerBase::restoreState(GraphState &saved) {
  frozenVariables = std::move(saved.frozenVariables);
  potentials = std::move(saved.potentials);
  parameterT = saved.parameterT;
  potentialsCurrent = saved.potentialsCurrent;
}

void SDCSchedulerBase::computeMargins(SmallVectorImpl<unsigned> &asap,
                                      SmallVectorImpl<unsigned> &alap) {
  assert(potentialsCurrent && "margins are read off a solved system");
  unsigned nNodes = origin + 1;
  for (unsigned stv = 0; stv < origin; ++stv)
    asap[stv] = unsigned(potentials[stv]);

  // g[i] = longest path i -> origin, i.e. `start(i) <= -g[i]`. Only upper
  // bounds shape the greatest element: an edge u -> v of weight w relaxes
  // g[u] against w + g[v], a pin's own upper edge starts its chain, and the
  // temporary pin holds the last operation at its minimal start.
  unsigned lastVar = startTimeVariables[lastOp];
  constexpr int64_t kUnreached = std::numeric_limits<int64_t>::min();
  SmallVector<int64_t> g(nNodes, kUnreached);
  SmallVector<SmallVector<std::pair<unsigned, int64_t>, 4>> gOut(nNodes);
  for (const Edge &e : edges)
    gOut[e.dst].push_back(
        {e.src, e.lat + e.extra - int64_t(parameterT) * e.dist});
  for (auto &[v, t] : frozenVariables)
    gOut[origin].push_back({v, -int64_t(t)});
  gOut[origin].push_back({lastVar, -potentials[lastVar]});

  g[origin] = 0;
  chainLen.assign(nNodes, 0);
  inQueue.assign(nNodes, 0);
  SmallVector<unsigned> queue{origin};
  inQueue[origin] = 1;
  for (unsigned head = 0; head < queue.size(); ++head) {
    unsigned u = queue[head];
    inQueue[u] = 0;
    for (auto [dst, w] : gOut[u]) {
      int64_t ng = g[u] + w;
      if (ng <= g[dst])
        continue;
      g[dst] = ng;
      chainLen[dst] = chainLen[u] + 1;
      assert(chainLen[dst] <= nNodes &&
             "the solved system has no positive circuit, so no chain repeats");
      if (!inQueue[dst]) {
        inQueue[dst] = 1;
        queue.push_back(dst);
      }
    }
  }
  for (unsigned stv = 0; stv < origin; ++stv) {
    assert(g[stv] != kUnreached &&
           "every operation is bounded above through the anchor");
    assert(-g[stv] >= potentials[stv] && "the two lattice extremes are ordered");
    alap[stv] = unsigned(-g[stv]);
  }
}


//===----------------------------------------------------------------------===//
// SharedOperatorsSimplexScheduler
//===----------------------------------------------------------------------===//

static bool isLimited(Operation *op, SharedOperatorsProblem &prob) {
  auto maybeRsrcs = prob.getLinkedResourceTypes(op);
  if (!maybeRsrcs)
    return false;
  return llvm::any_of(*maybeRsrcs, [&](Problem::ResourceType rsrc) {
    return prob.getLimit(rsrc).value_or(0) > 0;
  });
}

/// The limited units \p op holds, in link order. An operation takes all of them
/// at its start time and releases them together, so a cycle is feasible for it
/// only if every one has room. An unlimited link is dropped: it constrains
/// nothing, so no reservation table tracks it.
static SmallVector<Problem::ResourceType>
limitedUnits(SharedOperatorsProblem &prob, Operation *op) {
  auto maybeRsrcs = prob.getLinkedResourceTypes(op);
  assert(maybeRsrcs && "operation must have linked resource types");
  SmallVector<Problem::ResourceType> units;
  for (Problem::ResourceType rsrc : *maybeRsrcs)
    if (prob.getLimit(rsrc).value_or(0) > 0)
      units.push_back(rsrc);
  return units;
}

LogicalResult SharedOperatorsSimplexScheduler::schedule() {
  if (failed(checkLastOp()))
    return failure();

  parameterT = 0;
  buildGraph();

  if (failed(solveGraph(/*allowRaise=*/false))) {
    reportInfeasible();
    return failure();
  }

  // Heuristic phase: greedily fix start times for shared-operator ops within
  // allocation limits, the least solution updated with each pin. Each state
  // is optimal given prior fixes; overall optimality is not guaranteed.

  auto &ops = prob.getOperations();
  SmallVector<Operation *> limitedOps;
  for (auto *op : ops)
    if (isLimited(op, prob))
      limitedOps.push_back(op);

  // Placement order: earliest first, then the largest reservation first among
  // operations starting at the same time. Earliest-first is a topological
  // order, which keeps the acyclic problem feasible under pinning; the scan
  // below is first fit over rectangles, which needs largest-first to behave.
  //
  // Slack is not available as a further tie-break here: an ALAP would maximize
  // the start times, and with dependences the only constraints here an
  // operation without an outgoing one (any store) is unbounded above.
  auto rectangle = [&](Operation *op) {
    return prob.getResourceCycles(op) * prob.getResourceDemand(op);
  };
  llvm::stable_sort(limitedOps, [&](Operation *a, Operation *b) {
    unsigned ta = getStartTime(startTimeVariables[a]);
    unsigned tb = getStartTime(startTimeVariables[b]);
    if (ta != tb)
      return ta < tb;
    return rectangle(a) > rectangle(b);
  });

  // Store the number of operations using a resource type in a particular time
  // step.
  SmallDenseMap<Problem::ResourceType, SmallDenseMap<unsigned, unsigned>>
      reservationTable;

  for (auto *op : limitedOps) {
    SmallVector<Problem::ResourceType> units = limitedUnits(prob, op);
    assert(!units.empty() && "a limited operation holds a limited unit");

    // Find the first time step (from the current start time) where every unit
    // the op holds is free for its whole occupancy window (occ consecutive
    // cycles; occ == 1 when pipelined).
    unsigned occ = prob.getResourceCycles(op);
    unsigned slots = prob.getResourceDemand(op);
    unsigned startTimeVar = startTimeVariables[op];
    unsigned candTime = getStartTime(startTimeVar);
    auto hasRoom = [&](unsigned t) {
      for (Problem::ResourceType rsrc : units) {
        unsigned limit = *prob.getLimit(rsrc);
        for (unsigned i = 0; i < occ; ++i)
          if (reservationTable[rsrc].lookup(t + i) + slots > limit)
            return false;
      }
      return true;
    };
    while (!hasRoom(candTime))
      ++candTime;

    // Fix the start time. As explained above, this cannot make the problem
    // infeasible.
    auto fixed = scheduleAt(startTimeVar, candTime);
    assert(succeeded(fixed));
    (void)fixed;

    // Record the use of every unit across the occupancy window.
    for (Problem::ResourceType rsrc : units)
      for (unsigned i = 0; i < occ; ++i)
        reservationTable[rsrc][candTime + i] += slots;

  }

  assert(parameterT == 0);

  for (auto *op : ops)
    prob.setStartTime(op, getStartTime(startTimeVariables[op]));

  return success();
}

//===----------------------------------------------------------------------===//
// ModuloSimplexScheduler
//===----------------------------------------------------------------------===//

LogicalResult ModuloSimplexScheduler::checkLastOp() {
  if (!prob.hasOperation(lastOp)) {
    assert(false && "the scheduling problem does not include its last "
                    "operation; ProblemBuilder constructs both, so no input "
                    "can reach this");
    return failure();
  }

  // Determine which operations have no outgoing *intra*-iteration dependences.
  auto &ops = prob.getOperations();
  DenseSet<Operation *> sinks(ops.begin(), ops.end());
  for (auto *op : ops)
    for (auto &dep : prob.getDependences(op))
      if (prob.getDistance(dep).value_or(0) == 0)
        sinks.erase(dep.getSource());

  if (!sinks.contains(lastOp)) {
    assert(false && "the problem's last operation is not a sink; "
                    "ProblemBuilder anchors it, so no input can reach this");
    return failure();
  }
  if (sinks.size() > 1) {
    assert(false && "the problem has several sinks; ProblemBuilder anchors "
                    "exactly one, so no input can reach this");
    return failure();
  }

  return success();
}

LogicalResult ModuloSimplexScheduler::MRT::enter(Operation *op,
                                                 unsigned timeStep) {
  SmallVector<Problem::ResourceType> units = limitedUnits(sched.prob, op);
  assert(!units.empty() && "a limited operation holds a limited unit");

  // A non-pipelined op occupies `occ` consecutive modulo slots; a window wider
  // than II wraps, hitting one slot twice, which a per-slot set would hide. The
  // window is the same on every unit, all taken at the op's start time.
  unsigned occ = sched.prob.getResourceCycles(op);
  unsigned slots = sched.prob.getResourceDemand(op);
  unsigned base = timeStep % sched.parameterT;
  SmallDenseMap<unsigned, unsigned> want;
  for (unsigned i = 0; i < occ; ++i)
    want[(base + i) % sched.parameterT] += slots;

  // Admit only if every touched slot of every unit fits, then commit to all of
  // them: an op that fits in one unit but not another must leave no partial
  // reservation behind.
  for (Problem::ResourceType rsrc : units) {
    auto &table = tables[rsrc];
    for (const auto &[slot, cnt] : want)
      if (table.lookup(slot) + cnt > *sched.prob.getLimit(rsrc))
        return failure();
  }
  for (Problem::ResourceType rsrc : units) {
    auto &table = tables[rsrc];
    for (const auto &[slot, cnt] : want)
      table[slot] += cnt;
    auto &revTab = reverseTables[rsrc];
    assert(!revTab.count(op));
    revTab[op] = base;
  }
  return success();
}

void ModuloSimplexScheduler::MRT::release(Operation *op) {
  unsigned occ = sched.prob.getResourceCycles(op);
  unsigned slots = sched.prob.getResourceDemand(op);
  // Undo enter's per-slot increments on every unit it reserved, recomputed from
  // the stored base + occ so a wrapped slot is decremented once per lap. The
  // reverse tables record exactly the units entered, unlimited links skipped.
  bool held = false;
  for (auto &[rsrc, revTab] : reverseTables) {
    auto it = revTab.find(op);
    if (it == revTab.end())
      continue;
    auto &table = tables[rsrc];
    for (unsigned i = 0; i < occ; ++i) {
      unsigned &cnt = table[(it->second + i) % sched.parameterT];
      assert(cnt >= slots && "releasing an MRT slot that was never reserved");
      cnt -= slots;
    }
    revTab.erase(it);
    held = true;
  }
  assert(held && "releasing an operation that holds no unit");
  (void)held;
}

void ModuloSimplexScheduler::updateMargins() {
  computeMargins(asapTimes, alapTimes);
}

/// Tries `n` at its current time step and the II-1 slots after it. When none
/// admits it, repairs the placement by evicting blockers
/// (`scheduleWithEviction`); failure past that is the caller's cue to grow
/// the II and restart placement.
LogicalResult ModuloSimplexScheduler::scheduleOperation(Operation *n) {
  unsigned stvN = startTimeVariables[n];

  // Try the op's current time step in the partial solution and the II-1
  // following ones. A later step may increase the overall latency, but that is
  // preferred over incrementing the II to resolve resource conflicts.
  unsigned stN = getStartTime(stvN);
  unsigned ubN = stN + parameterT - 1;

  LLVM_DEBUG(dbgs() << "Attempting to schedule in [" << stN << ", " << ubN
                    << "]: " << *n << '\n');

  for (unsigned ct = stN; ct <= ubN; ++ct)
    if (succeeded(mrt.enter(n, ct))) {
      auto fixedN = scheduleAt(stvN, ct);
      if (succeeded(fixedN)) {
        LLVM_DEBUG(dbgs() << "Success at t=" << ct << " " << *n << '\n');
        return success();
      }
      // Problem became infeasible with `n` at `ct`, roll back the MRT
      // assignment. Also, no later time can be feasible, so stop the search
      // here.
      mrt.release(n);
      break;
    }

  // `n` does not fit at this II: repair the placement by evicting blockers.
  // Failure here means the repair budget could not buy a slot either.
  return scheduleWithEviction(n);
}

/// Repairs a failed placement at the current II: choose a start in the same
/// window first fit scanned, evict the placed operations blocking it (their
/// reservations first, then the pins a certifying cycle names), and pin `n`
/// there. Victims return to `unscheduled` to be placed again. The
/// per-operation cap and the region budget keep the repair finite; once they
/// are spent the caller grows the II exactly as without repair.
LogicalResult ModuloSimplexScheduler::scheduleWithEviction(Operation *n) {
  if (evictionBudget == 0)
    return failure();
  unsigned stvN = startTimeVariables[n];
  unsigned stN = getStartTime(stvN);
  SmallVector<Problem::ResourceType> unitsN = limitedUnits(prob, n);
  unsigned occN = prob.getResourceCycles(n);
  unsigned slotsN = prob.getResourceDemand(n);
  unsigned ii = parameterT;

  auto evictable = [&](Operation *op) {
    return evictCount.lookup(op) < kMaxEvictionsPerOp;
  };
  // How often `slot` falls inside a reservation window starting at `base`: a
  // window wider than the II wraps and can hold one slot more than once.
  auto covers = [&](unsigned base, unsigned occ, unsigned slot) {
    unsigned m = 0;
    for (unsigned i = 0; i < occ; ++i)
      if ((base + i) % ii == slot)
        ++m;
    return m;
  };

  // The holders of each unit `n` needs, least-evicted first with problem
  // order breaking ties, so victim choice is deterministic and a contested
  // pair runs out of budget instead of cycling.
  SmallDenseMap<Problem::ResourceType,
                SmallVector<std::pair<Operation *, unsigned>>>
      holders;
  for (Problem::ResourceType rsrc : unitsN) {
    auto &hs = holders[rsrc];
    hs.assign(mrt.reverseTables[rsrc].begin(), mrt.reverseTables[rsrc].end());
    llvm::sort(hs, [&](auto &a, auto &b) {
      unsigned ea = evictCount.lookup(a.first);
      unsigned eb = evictCount.lookup(b.first);
      if (ea != eb)
        return ea < eb;
      return startTimeVariables[a.first] < startTimeVariables[b.first];
    });
  }

  // For every start in the window, the placed operations whose reservations
  // must leave for `n`'s whole occupancy window to fit there. A start whose
  // deficit the evictable holders cannot cover is skipped.
  struct Candidate {
    unsigned ct;
    unsigned cost;
    SmallVector<Operation *> victims;
  };
  SmallVector<Candidate> cands;
  for (unsigned ct = stN; ct < stN + ii; ++ct) {
    SmallDenseMap<unsigned, unsigned> want;
    for (unsigned i = 0; i < occN; ++i)
      want[(ct + i) % ii] += slotsN;

    SmallVector<Operation *> victims;
    SmallPtrSet<Operation *, 8> taken;
    unsigned cost = 0;
    bool feasible = true;
    for (Problem::ResourceType rsrc : unitsN) {
      unsigned limit = *prob.getLimit(rsrc);
      auto &table = mrt.tables[rsrc];
      auto &revTab = mrt.reverseTables[rsrc];
      for (auto [slot, need] : want) {
        // What already-chosen victims free in this slot counts against the
        // deficit; a victim may hold several of `n`'s units at once.
        unsigned freed = 0;
        for (Operation *v : victims)
          if (auto it = revTab.find(v); it != revTab.end())
            freed += covers(it->second, prob.getResourceCycles(v), slot) *
                     prob.getResourceDemand(v);
        while (table.lookup(slot) + need > limit + freed) {
          Operation *pick = nullptr;
          unsigned mult = 0;
          for (auto &[op, base] : holders[rsrc]) {
            if (taken.count(op) || !evictable(op))
              continue;
            mult = covers(base, prob.getResourceCycles(op), slot);
            if (mult) {
              pick = op;
              break;
            }
          }
          if (!pick) {
            feasible = false;
            break;
          }
          taken.insert(pick);
          victims.push_back(pick);
          freed += mult * prob.getResourceDemand(pick);
          cost += 1 + evictCount.lookup(pick);
        }
        if (!feasible)
          break;
      }
      if (!feasible)
        break;
    }
    if (feasible && victims.size() <= evictionBudget)
      cands.push_back({ct, cost, std::move(victims)});
  }

  // Cheapest victim set first, earliest start breaking ties.
  llvm::stable_sort(cands, [](const Candidate &a, const Candidate &b) {
    return std::tie(a.cost, a.ct) < std::tie(b.cost, b.ct);
  });

  unsigned attempts = std::min<size_t>(cands.size(), kMaxCommitAttempts);
  for (unsigned c = 0; c < attempts; ++c) {
    Candidate &cand = cands[c];
    GraphState saved = saveState();
    auto savedTables = mrt.tables;
    auto savedReverse = mrt.reverseTables;
    SmallVector<Operation *> evicted;
    auto evict = [&](Operation *v) {
      mrt.release(v);
      frozenVariables.erase(startTimeVariables[v]);
      evicted.push_back(v);
    };
    auto resolve = [&] {
      LogicalResult r = solveGraph(/*allowRaise=*/false);
      assert(succeeded(r) && "removing pins keeps the system feasible");
      (void)r;
    };
    for (Operation *v : cand.victims)
      evict(v);
    if (!evicted.empty())
      resolve();
    LogicalResult entered = mrt.enter(n, cand.ct);
    assert(succeeded(entered) && "the victim set frees the whole window");
    (void)entered;

    // A pin failure certifies a cycle through other pins; evict those too and
    // retry, a bounded number of rounds.
    bool placed = false;
    for (unsigned round = 0; round < kMaxDepRounds; ++round) {
      SmallVector<unsigned> conflicts;
      if (succeeded(scheduleAt(stvN, cand.ct, &conflicts))) {
        placed = true;
        break;
      }
      if (conflicts.empty() ||
          evicted.size() + conflicts.size() > evictionBudget ||
          !llvm::all_of(conflicts,
                        [&](unsigned v) { return evictable(varOps[v]); }))
        break;
      for (unsigned v : conflicts)
        evict(varOps[v]);
      resolve();
    }
    if (placed) {
      for (Operation *v : evicted) {
        llvm::erase(scheduled, v);
        unscheduled.push_back(v);
        ++evictCount[v];
      }
      evictionBudget -= evicted.size();
      info(Stage::Sched, n)
          << "Placement repair at II=" << ii << ": evicted " << evicted.size()
          << " operation(s) so " << n->getName().getStringRef()
          << " can start at t=" << cand.ct;
      return success();
    }
    restoreState(saved);
    mrt.tables = std::move(savedTables);
    mrt.reverseTables = std::move(savedReverse);
  }
  return failure();
}


/// Grows the II by one and restarts placement from scratch at the larger
/// interval, after Rau: pins made at the smaller II would otherwise carry
/// their scars (evicted victims stranded on late slots) into the new one. The
/// caller's worklist loop then re-places every limited operation with a fresh
/// reservation table and repair budget.
LogicalResult ModuloSimplexScheduler::growIIAndRestart(Operation *n) {
  ++parameterT;
  // Every op fits in a disjoint window by II=totalResourceCycles; 2x+2 leaves
  // slack for cross-window fragmentation. Past that, growth is not
  // converging: a scheduler limit, not a fact about the kernel.
  if (parameterT > 2 * static_cast<int>(totalResourceCycles) + 2) {
    // Where the compile stops on the default path, and only advice when an
    // exact solver is going to place the region itself.
    auto d = placementAdvisory
                 ? warn(Stage::Sched, n)
                 : unsupported(Stage::Sched, Code::PlacementFailed, n);
    d << "The modulo scheduler could not place " << n->getName().getStringRef()
      << " at any initiation interval tried (up to II=" << parameterT
      << "): resource placement is greedy with a budgeted eviction repair, "
         "and neither found this operation a feasible cycle";
    if (placementAdvisory)
      d << "; the exact scheduler places the region instead";
    else
      d << ". Partitioning the array it contends for, or reducing how many "
           "times one iteration accesses that array, gives the placement "
           "room";
    return failure();
  }
  info(Stage::Sched, n) << "II=" << parameterT - 1 << " is not achievable for "
                        << n->getName().getStringRef()
                        << "; restarting placement at II=" << parameterT;
  frozenVariables.clear();
  mrt.clear();
  scheduled.clear();
  unscheduled.clear();
  for (auto *op : prob.getOperations())
    if (isLimited(op, prob))
      unscheduled.push_back(op);
  evictCount.clear();
  evictionBudget = kMaxEvictionsPerOp * unscheduled.size();
  LogicalResult solved = solveGraph(/*allowRaise=*/true);
  assert(succeeded(solved) &&
         "the pin-free system was feasible at a smaller II");
  return solved;
}

int64_t ModuloSimplexScheduler::depAsapOf(unsigned stv) {
  // The longest path from the origin to `stv` over the dependence edges alone
  // (its own pin excluded), read off the current potentials. A pin above this
  // was not forced by dependences but by first fit taking a free-but-high slot.
  int64_t d = 0;
  for (const Edge &e : edges)
    if (e.dst == stv) {
      int64_t c = potentials[e.src] + e.lat + e.extra -
                  int64_t(parameterT) * e.dist;
      d = std::max(d, c);
    }
  return d;
}

bool ModuloSimplexScheduler::trySeatLower(unsigned stvX, unsigned oldPin,
                                          unsigned &budget) {
  Operation *crit = varOps[stvX];
  SmallVector<Problem::ResourceType> unitsX = limitedUnits(prob, crit);
  unsigned occX = prob.getResourceCycles(crit);
  unsigned slotsX = prob.getResourceDemand(crit);
  unsigned ii = parameterT;

  // Release crit and let its start fall to the dependence bound.
  mrt.release(crit);
  frozenVariables.erase(stvX);
  {
    LogicalResult r = solveGraph(/*allowRaise=*/false);
    assert(succeeded(r) && "removing a pin keeps the system feasible");
    (void)r;
  }
  auto repin = [&] {
    LogicalResult e = mrt.enter(crit, oldPin);
    assert(succeeded(e) && "the slot crit just vacated still fits it");
    (void)e;
    LogicalResult p = scheduleAt(stvX, oldPin);
    assert(succeeded(p) && "crit's old pin is still feasible");
    (void)p;
  };
  unsigned lo = getStartTime(stvX);
  if (lo >= oldPin) {
    repin();
    return false;
  }

  auto covers = [&](unsigned base, unsigned occ, unsigned slot) {
    unsigned m = 0;
    for (unsigned i = 0; i < occ; ++i)
      if ((base + i) % ii == slot)
        ++m;
    return m;
  };

  // The lowest class in [lo, oldPin) crit fits after evicting the holders that
  // stand in its way.
  SmallVector<Operation *> victims;
  unsigned chosen = oldPin;
  for (unsigned ct = lo; ct < oldPin && chosen == oldPin; ++ct) {
    SmallDenseMap<unsigned, unsigned> want;
    for (unsigned i = 0; i < occX; ++i)
      want[(ct + i) % ii] += slotsX;
    SmallVector<Operation *> vic;
    SmallPtrSet<Operation *, 8> taken;
    bool feasible = true;
    for (Problem::ResourceType rsrc : unitsX) {
      unsigned limit = *prob.getLimit(rsrc);
      auto &table = mrt.tables[rsrc];
      auto &revTab = mrt.reverseTables[rsrc];
      // Victim order must be deterministic: the reverse table is keyed by
      // pointer, so problem order sorts the holders the way scheduleWithEviction
      // does before a scan.
      SmallVector<std::pair<Operation *, unsigned>> sortedHolders(revTab.begin(),
                                                                  revTab.end());
      llvm::sort(sortedHolders, [&](auto &a, auto &b) {
        return startTimeVariables[a.first] < startTimeVariables[b.first];
      });
      for (auto [slot, need] : want) {
        unsigned freed = 0;
        for (Operation *v : vic)
          if (auto it = revTab.find(v); it != revTab.end())
            freed += covers(it->second, prob.getResourceCycles(v), slot) *
                     prob.getResourceDemand(v);
        while (table.lookup(slot) + need > limit + freed) {
          Operation *pick = nullptr;
          unsigned mult = 0;
          for (auto &[op, base] : sortedHolders) {
            // Any evictable holder is a candidate victim; re-placing it may
            // land it in another low class at the same time (a swap), so the
            // final span check, not a slack pre-filter, is what rejects a move
            // that would raise the span.
            if (taken.count(op) ||
                evictCount.lookup(op) >= kMaxEvictionsPerOp)
              continue;
            mult = covers(base, prob.getResourceCycles(op), slot);
            if (mult) {
              pick = op;
              break;
            }
          }
          if (!pick) {
            feasible = false;
            break;
          }
          taken.insert(pick);
          vic.push_back(pick);
          freed += mult * prob.getResourceDemand(pick);
        }
        if (!feasible)
          break;
      }
      if (!feasible)
        break;
    }
    if (feasible && vic.size() <= budget) {
      chosen = ct;
      victims = std::move(vic);
    }
  }
  if (chosen == oldPin) {
    repin();
    return false;
  }

  // Commit: evict the victims, seat crit low, then re-place the victims by first
  // fit. Any failure leaves the state dirty for the caller to roll back.
  for (Operation *v : victims) {
    mrt.release(v);
    frozenVariables.erase(startTimeVariables[v]);
  }
  {
    LogicalResult r = solveGraph(/*allowRaise=*/false);
    assert(succeeded(r) && "removing pins keeps the system feasible");
    (void)r;
  }
  if (failed(mrt.enter(crit, chosen)))
    return false;
  if (failed(scheduleAt(stvX, chosen))) {
    mrt.release(crit);
    return false;
  }
  llvm::sort(victims, [&](Operation *a, Operation *b) {
    unsigned va = startTimeVariables[a], vb = startTimeVariables[b];
    return std::make_pair(getStartTime(va), va) <
           std::make_pair(getStartTime(vb), vb);
  });
  for (Operation *v : victims) {
    unsigned sv = getStartTime(startTimeVariables[v]);
    bool placed = false;
    for (unsigned cv = sv; cv < sv + ii; ++cv)
      if (succeeded(mrt.enter(v, cv))) {
        if (succeeded(scheduleAt(startTimeVariables[v], cv))) {
          placed = true;
          break;
        }
        mrt.release(v);
      }
    if (!placed)
      return false;
  }
  budget -= victims.size();
  ++evictCount[crit];
  return true;
}

/// Hill-climbs the region span down after the placement loop: each pass lowers
/// the highest critical op that first fit pushed above its dependence bound,
/// keeping the move only when the last operation's start strictly drops. An
/// already-span-optimal region has no such op, so the pass is a no-op and its
/// schedule is left byte-identical.
void ModuloSimplexScheduler::repairSpan() {
  unsigned lastVar = startTimeVariables[lastOp];
  unsigned budget = kMaxEvictionsPerOp * scheduled.size();
  for (unsigned moves = 0; moves < kMaxSpanRepairMoves && budget > 0; ++moves) {
    updateMargins();
    int64_t span = potentials[lastVar];

    // Critical pinned ops sitting above their dependence bound, highest first:
    // lowering the top of the critical path frees the most.
    SmallVector<std::pair<unsigned, unsigned>> cand; // (pin, stv)
    for (auto &[stv, pin] : frozenVariables) {
      if (alapTimes[stv] != asapTimes[stv])
        continue;
      if (depAsapOf(stv) < int64_t(pin))
        cand.push_back({pin, stv});
    }
    llvm::sort(cand, [](auto &a, auto &b) {
      return std::tie(b.first, a.second) < std::tie(a.first, b.second);
    });

    bool improved = false;
    for (auto [pin, stv] : cand) {
      GraphState saved = saveState();
      auto savedTables = mrt.tables;
      auto savedReverse = mrt.reverseTables;
      unsigned tryBudget = budget;
      if (trySeatLower(stv, pin, tryBudget) && potentials[lastVar] < span) {
        info(Stage::Sched, varOps[stv])
            << "Span repair at II=" << parameterT << ": moved "
            << varOps[stv]->getName().getStringRef() << " from t=" << pin
            << " to t=" << getStartTime(stv) << ", region span " << span
            << " -> " << potentials[lastVar];
        budget = tryBudget;
        improved = true;
        break;
      }
      restoreState(saved);
      mrt.tables = std::move(savedTables);
      mrt.reverseTables = std::move(savedReverse);
    }
    if (!improved)
      break;
  }
}

unsigned ModuloSimplexScheduler::computeResMinII(BindingResource &binding) {
  unsigned resMinII = 1;
  SmallDenseMap<Problem::ResourceType, unsigned> uses;
  SmallDenseMap<Problem::ResourceType, Operation *> witness;
  for (auto *op : prob.getOperations()) {
    auto maybeRsrcs = prob.getLinkedResourceTypes(op);
    if (!maybeRsrcs)
      continue;

    for (auto rsrc : *maybeRsrcs) {
      if (prob.getLimit(rsrc).value_or(0) > 0) {
        // occupancy: the whole window a non-pipelined unit is held for, times
        // the units the operation holds at once
        uses[rsrc] += prob.getResourceCycles(op) * prob.getResourceDemand(op);
        // The operation list is in a stable order, so the witness a diagnostic
        // points at is deterministic.
        witness.try_emplace(rsrc, op);
      }
    }
  }

  // Integer ceil: enough parallel units to cover total occupancy in one II.
  // (unsigned `a / b` floors, so an explicit integer ceil is needed once
  // limit >= 2.)
  for (auto pair : uses) {
    unsigned limit = *prob.getLimit(pair.first);
    unsigned need = (pair.second + limit - 1) / limit;
    if (need <= resMinII)
      continue;
    resMinII = need;
    binding = {pair.first, pair.second, limit, witness.lookup(pair.first)};
  }

  return resMinII;
}

/// Seeds the II at the larger of the resource-min II and the pipeline
/// directive's floor, then iteratively fixes limited operations to time steps
/// in earliest-first, least-slack-breaks-ties order. That order matters:
/// pinning a consumer caps how late its operands may issue, and once a
/// resource saturates at this II there is no cycle left for the last of them.
LogicalResult ModuloSimplexScheduler::schedule() {
  if (failed(checkLastOp()))
    return failure();

  // Seed the II at the resource-min II, but never below the pipeline
  // directive's target; the search only grows it from there.
  BindingResource binding;
  unsigned resMinII = computeResMinII(binding);
  parameterT = std::max(resMinII, minII);
  info(Stage::Sched, prob.getContainingOp())
      << "Initiation interval search seeded at II=" << parameterT
      << " (resource-min II=" << resMinII
      << ", pipeline-directive floor minII=" << minII << ")";
  LLVM_DEBUG(dbgs() << "ResMinII = " << parameterT << " (minII=" << minII
                    << ")\n");
  buildGraph();
  asapTimes.resize(startTimeVariables.size());
  alapTimes.resize(startTimeVariables.size());

  if (failed(solveGraph(/*allowRaise=*/true))) {
    reportInfeasible();
    return failure();
  }
  // The resource-free solve already raises the II to any loop-carried
  // recurrence's minimum, so `parameterT` here is the best lower bound anything
  // downstream can justify.
  lowerBoundII = parameterT;
  boundSettled = true;

  // Report what set the bound, so it can be acted on: banking or replicating an
  // array lowers a port-bound interval, reassociating a reduction lowers a
  // recurrence-bound one.
  if (lowerBoundII > 1) {
    if (lowerBoundII > std::max(resMinII, minII))
      info(Stage::Sched, prob.getContainingOp())
          << "II cannot go below " << lowerBoundII
          << " here: a loop-carried recurrence takes that long to come round";
    else if (resMinII >= minII && binding.witness)
      info(Stage::Sched, binding.witness)
          << "II cannot go below " << resMinII << " here: one iteration takes "
          << binding.demand << " slots of a resource serving " << binding.limit
          << " per cycle. Banking or replicating what this access reaches is "
             "what lowers that bound";
  }

  // Determine which operations are subject to resource constraints, and whether
  // any of them is non-pipelined (occupies its unit for more than one cycle).
  auto &ops = prob.getOperations();
  for (auto *op : ops)
    if (isLimited(op, prob)) {
      unscheduled.push_back(op);
      totalResourceCycles += prob.getResourceCycles(op);
    }
  evictionBudget = kMaxEvictionsPerOp * unscheduled.size();

  // Main loop: iteratively fix limited operations to time steps. An operation
  // that fits nowhere even after eviction repair grows the II and restarts
  // the placement from scratch at the larger interval.
  while (!unscheduled.empty()) {
    // ASAP/ALAP margins, refreshed against the operations pinned so far.
    updateMargins();

    // Earliest-first, least slack breaking the tie (see the doc comment above).
    auto priority = [&](Operation *op) {
      unsigned stv = startTimeVariables[op];
      return std::make_pair(asapTimes[stv], alapTimes[stv] - asapTimes[stv]);
    };
    auto *opIt = std::min_element(unscheduled.begin(), unscheduled.end(),
                                  [&](Operation *opA, Operation *opB) {
                                    return priority(opA) < priority(opB);
                                  });
    Operation *op = *opIt;
    unscheduled.erase(opIt);

    if (succeeded(scheduleOperation(op))) {
      scheduled.push_back(op);
      continue;
    }
    if (failed(growIIAndRestart(op)))
      return failure();
  }

  // Lower the span where first fit left a critical op above its dependence
  // bound. Only strictly-improving moves commit, so this never grows the II
  // or the latency, and a span-optimal region is untouched.
  repairSpan();

  // Resource placement is greedy, so an II above the LP's bound may be the
  // problem's real minimum or just what the heuristic cost; nothing here can
  // tell the two apart.
  if (parameterT > static_cast<int>(lowerBoundII))
    warn(Stage::Sched, prob.getContainingOp())
        << "Scheduled at II=" << parameterT
        << " against a lower bound of II=" << lowerBoundII
        << " (resource-min II=" << resMinII
        << "): resource placement is a greedy heuristic, so this gap is not "
           "known to be necessary";

  prob.setInitiationInterval(parameterT);
  for (auto *op : ops)
    prob.setStartTime(op, getStartTime(startTimeVariables[op]));

  return success();
}

namespace mlir::allo {

//===----------------------------------------------------------------------===//
// OccupancyProblem / ModuloOccupancyProblem (declared in Scheduler.h): CIRCT's
// resource problems with a per-operation occupancy window.
//===----------------------------------------------------------------------===//

LogicalResult OccupancyProblem::checkLatency(Operation *op) {
  // Deliberately NOT SharedOperatorsProblem::checkLatency, which rejects a
  // zero-latency operation on a limited resource. A combinational access holds
  // its port for the cycle it issues in and contends like any other.
  return Problem::checkLatency(op);
}

int64_t OccupancyProblem::latencyOf(Operation *op) {
  std::optional<OperatorType> opr = getLinkedOperatorType(op);
  assert(opr && "an operation the operator model never characterized");
  std::optional<unsigned> latency = getLatency(*opr);
  assert(latency && "an operator type with no latency");
  return *latency;
}

int64_t OccupancyProblem::scheduleDepth() {
  int64_t depth = 1;
  for (Operation *op : getOperations())
    if (std::optional<unsigned> start = getStartTime(op))
      depth = std::max(depth, static_cast<int64_t>(*start) +
                                  std::max<int64_t>(1, latencyOf(op)));
  return depth;
}

bool OccupancyProblem::holdsLimitedUnit(Operation *op) {
  auto linked = getLinkedResourceTypes(op);
  return linked && llvm::any_of(*linked, [&](ResourceType rsrc) {
           return getLimit(rsrc).value_or(0) > 0;
         });
}

bool OccupancyProblem::holdsAllocatableUnit(Operation *op) {
  auto linked = getLinkedResourceTypes(op);
  return linked && llvm::any_of(*linked, [&](ResourceType rsrc) {
           return getAllocatable(rsrc).has_value();
         });
}

SmallVector<Operation *> OccupancyProblem::usersOf(ResourceType rsrc) {
  SmallVector<Operation *> users;
  for (Operation *op : getOperations())
    if (usesResource(op, rsrc))
      users.push_back(op);
  llvm::stable_sort(users, [&](Operation *a, Operation *b) {
    return *getStartTime(a) < *getStartTime(b);
  });
  return users;
}

unsigned OccupancyProblem::demandFor(ResourceType rsrc, unsigned ii) {
  SmallDenseMap<unsigned, unsigned> used;
  unsigned peak = 0;
  for (Operation *op : getOperations()) {
    if (!usesResource(op, rsrc))
      continue;
    unsigned start = *getStartTime(op);
    unsigned slots = getResourceDemand(op);
    for (unsigned k = 0, occ = getResourceCycles(op); k < occ; ++k) {
      unsigned &cnt = used[ii ? (start + k) % ii : start + k];
      cnt += slots;
      peak = std::max(peak, cnt);
    }
  }
  return peak;
}

void OccupancyProblem::assignUnits(unsigned ii) {
  for (ResourceType rsrc : getResourceTypes()) {
    std::optional<unsigned> units = getAllocation(rsrc);
    if (!units)
      continue;
    SmallVector<Operation *> users = usersOf(rsrc);
    // Both rules round-robin over all the instances rather than packing into
    // the fewest that fit, so the count decided is the count built.
    unsigned cursor = 0;
    if (ii) {
      // Occupancy is one cycle here, so an instance is available iff it is
      // free in the operation's congruence class.
      llvm::DenseSet<std::pair<unsigned, unsigned>> taken;
      for (Operation *op : users) {
        unsigned cls = *getStartTime(op) % ii;
        unsigned k = cursor % *units;
        for (unsigned tried = 1; taken.count({k, cls}) && tried < *units;
             ++tried)
          k = (k + 1) % *units;
        assert(!taken.count({k, cls}) &&
               "the busiest congruence class needs more instances than the "
               "allocation decided");
        taken.insert({k, cls});
        assignedUnit[op] = k;
        cursor = k + 1;
      }
    } else {
      // First fit over occupancy windows in start order, rotating the instance
      // scanned first so the load spreads.
      SmallVector<unsigned> freeAt(*units, 0);
      for (Operation *op : users) {
        unsigned start = *getStartTime(op);
        unsigned k = cursor % *units;
        for (unsigned tried = 1; freeAt[k] > start && tried < *units; ++tried)
          k = (k + 1) % *units;
        assert(freeAt[k] <= start && "the busiest cycle needs more instances "
                                     "than the allocation decided");
        assignedUnit[op] = k;
        freeAt[k] = start + getResourceCycles(op);
        cursor = k + 1;
      }
    }
  }
}

LogicalResult OccupancyProblem::verifyAllocation(unsigned ii) {
  for (ResourceType rsrc : getResourceTypes()) {
    std::optional<unsigned> units = getAllocation(rsrc);
    if (!units)
      continue; // no solve decided one, so the trivial allocation stands
    // (instance, cycle) pairs already taken.
    llvm::DenseSet<std::pair<unsigned, unsigned>> busy;
    for (Operation *op : getOperations()) {
      if (!usesResource(op, rsrc))
        continue;
      std::optional<unsigned> unit = getAssignedUnit(op);
      if (!unit || *unit >= *units) {
        assert(false && "an operation on an allocated operator has no instance "
                        "to run on, or one past the count decided");
        return failure();
      }
      unsigned start = *getStartTime(op);
      for (unsigned k = 0, occ = getResourceCycles(op); k < occ; ++k)
        if (!busy.insert({*unit, ii ? (start + k) % ii : start + k}).second) {
          assert(false && "two operations share one operator instance in the "
                          "same cycle");
          return failure();
        }
    }
  }
  return success();
}

LogicalResult OccupancyProblem::verifyOccupancy(unsigned ii) {
  for (ResourceType rsrc : getResourceTypes()) {
    unsigned limit = getLimit(rsrc).value_or(0);
    if (limit && demandFor(rsrc, ii) > limit) {
      assert(false && "a resource is oversubscribed across its occupancy "
                      "windows; the reservation table admits an operation "
                      "only when every slot it touches fits, so a solved "
                      "schedule cannot reach this");
      return failure();
    }
  }
  return success();
}

LogicalResult ModuloOccupancyProblem::verifyPrecedence(Dependence dep) {
  if (!isForwarded(dep))
    return CyclicProblem::verifyPrecedence(dep);
  unsigned stI = *getStartTime(dep.getSource());
  unsigned stJ = *getStartTime(dep.getDestination());
  unsigned dist = getDistance(dep).value_or(0);
  if (stI <= stJ + dist * *getInitiationInterval())
    return success();
  return getContainingOp()->emitError()
         << "Precedence violated for a forwarded store->load dependence: the "
            "store issues after the load it forwards to";
}

LogicalResult ModuloOccupancyProblem::verify() {
  if (failed(ModuloProblem::verify()))
    return failure();
  unsigned ii = *getInitiationInterval();
  if (failed(verifyOccupancy(ii)))
    return failure();
  return verifyAllocation(ii);
}

//===----------------------------------------------------------------------===//
// ChainingModuloProblem (declared in Scheduler.h): the composition of CIRCT's
// ChainingProblem and ModuloOccupancyProblem.
//===----------------------------------------------------------------------===//

LogicalResult ChainingModuloProblem::checkDefUse(Dependence dep) {
  if (!dep.isAuxiliary() && (getDistance(dep).value_or(0) != 0)) {
    assert(false && "a def-use dependence carries a non-zero distance; the "
                    "edges are ours to insert, so no input can reach this");
    return failure();
  }
  return success();
}

LogicalResult ChainingModuloProblem::check() {
  for (auto *op : getOperations())
    for (auto &dep : getDependences(op))
      if (failed(checkDefUse(dep)))
        return failure();

  if (ChainingProblem::check().succeeded() &&
      ModuloProblem::check().succeeded())
    return success();
  return failure();
}

LogicalResult ChainingModuloProblem::verify() {
  if (ChainingProblem::verify().succeeded() &&
      ModuloOccupancyProblem::verify().succeeded())
    return success();
  return failure();
}

//===----------------------------------------------------------------------===//
// ChainingSharedOperatorsProblem (declared in Scheduler.h): the composition of
// CIRCT's ChainingProblem and OccupancyProblem. The acyclic twin of
// ChainingModuloProblem (no distance, so no def-use distance check).
//===----------------------------------------------------------------------===//

LogicalResult ChainingSharedOperatorsProblem::check() {
  if (ChainingProblem::check().succeeded() &&
      SharedOperatorsProblem::check().succeeded())
    return success();
  return failure();
}

LogicalResult ChainingSharedOperatorsProblem::verify() {
  if (ChainingProblem::verify().succeeded() &&
      SharedOperatorsProblem::verify().succeeded() &&
      verifyOccupancy(/*ii=*/0).succeeded() &&
      verifyAllocation(/*ii=*/0).succeeded())
    return success();
  return failure();
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

LogicalResult scheduleSimplex(ChainingModuloProblem &prob, Operation *lastOp,
                              float cycleTime, float regFloor, unsigned minII,
                              SimplexWarmStart *warm) {
  ChainingModuloSimplexScheduler simplex(prob, lastOp, cycleTime, regFloor,
                                         minII);
  if (warm)
    simplex.setPlacementAdvisory();
  LogicalResult scheduled = simplex.schedule();
  if (!warm)
    return scheduled;
  warm->lowerBoundII = simplex.getLowerBoundII();
  warm->placed = succeeded(scheduled);
  // A placement failure is the caller's to recover from; a resource-free one
  // means no II admits a schedule, and nothing downstream can repair that.
  return success(simplex.hasLowerBound());
}

LogicalResult scheduleSimplex(ChainingSharedOperatorsProblem &prob,
                              Operation *lastOp, float cycleTime,
                              float regFloor) {
  ChainingSharedOperatorsSimplexScheduler simplex(prob, lastOp, cycleTime,
                                                  regFloor);
  return simplex.schedule();
}

} // namespace mlir::allo
