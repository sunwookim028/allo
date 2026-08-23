/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_SCHEDULING_SCHEDULER_H
#define ALLO_SCHEDULING_SCHEDULER_H

#include "allo/Scheduling/ScheduleModel.h"

#include "circt/Scheduling/Problems.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

#include <optional>

namespace mlir::allo {

/// The device view, forward-declared here: `OperatorLibrary` is built on the
/// problems below, so the dependence runs one way.
class OperatorLibrary;

/// How one region's solve ran: `proven` = every deciding status OPTIMAL; a
/// `budgetExhausted` solve may ship a different schedule on the next run.
struct SolveTelemetry {
  bool proven = false;
  /// The span half's own verdict, which `proven` conflates with the area
  /// tie-break's: an exhausted budget with the span proven spent its time on
  /// area alone. Under the area objective, the drain-under-settled-area
  /// solve's verdict.
  bool spanProven = false;
  bool budgetExhausted = false;
  bool fallback = false; // shipped the heuristic's schedule instead
  /// The interval whose solve exhausted the budget, ending the cyclic search.
  std::optional<int64_t> exhaustedAtII;
};

/// A resource-constrained problem with per-operation occupancy windows, so a
/// synchronous call holding its callee's instance until done can be modeled
/// (`populateCallOccupancy`). An operation may hold several units at once
/// (`setLinkedResourceTypes`); a cycle is feasible only where every linked unit
/// has room across the whole window. A limited operation may have zero latency
/// (CIRCT requires non-zero): a combinational access still occupies its issue
/// cycle and contends like any other.
class OccupancyProblem
    : public virtual circt::scheduling::SharedOperatorsProblem {
public:
  static constexpr auto name = "OccupancyProblem";
  using circt::scheduling::SharedOperatorsProblem::SharedOperatorsProblem;

  /// Filled by the exact solver; read back by the solve report.
  SolveTelemetry telemetry;

protected:
  OccupancyProblem() = default;
  /// A limited operation may have zero latency (see the class comment).
  LogicalResult checkLatency(Operation *op) override;

public:
  /// Consecutive cycles \p op holds its resource unit from its start; one
  /// (fully pipelined) unless set.
  unsigned getResourceCycles(Operation *op) {
    return resourceCycles.lookup(op).value_or(1);
  }
  void setResourceCycles(Operation *op, unsigned cycles) {
    resourceCycles[op] = cycles;
  }

  /// Units of each linked resource \p op holds at once; one unless set (a write
  /// to a multi-copy array takes a port of each).
  unsigned getResourceDemand(Operation *op) {
    return resourceDemand.lookup(op).value_or(1);
  }
  void setResourceDemand(Operation *op, unsigned units) {
    resourceDemand[op] = units;
  }

  /// Cycles a dependent waits after \p op issues for its result: the latency of
  /// \p op's linked operator type. Every op carries one. Signed though latency
  /// is not, so composing it into a subtraction cannot underflow (a
  /// combinational `unsigned` `latencyOf(op) - 1` would be 2^32 - 1).
  int64_t latencyOf(Operation *op);

  /// Store->load dependences a forwarding network relaxes: the load may issue
  /// in the store's cycle, a shadow register supplying the datum on an address
  /// match. Set before solving; every solver and verifier weighs such an edge
  /// at latency zero instead of the store's write latency.
  void setForwarded(Dependence dep) { forwardedEdges.insert(dep); }
  bool isForwarded(Dependence dep) const {
    return forwardedEdges.contains(dep);
  }
  /// Forget every forwarded edge, for a re-solve of the unrelaxed problem.
  void clearForwarded() { forwardedEdges.clear(); }

  /// Schedule depth of a solved problem: the cycle by which every op has
  /// completed. Report only; a span composes from the drain instead, which may
  /// sit below the depth. A combinational op still occupies its issue cycle,
  /// hence the floor of one.
  int64_t scheduleDepth();

  /// Whether \p op holds at least one capped unit; an unlimited link constrains
  /// nothing and no reservation tracks it.
  bool holdsLimitedUnit(Operation *op);

  /// Whether \p op holds a unit of \p rsrc.
  bool usesResource(Operation *op, ResourceType rsrc) {
    auto linked = getLinkedResourceTypes(op);
    return linked && llvm::is_contained(*linked, rsrc);
  }

  /// Operations holding a unit of \p rsrc, earliest start first, so a derived
  /// assignment depends on the schedule not walk order. Every op must be
  /// scheduled.
  SmallVector<Operation *> usersOf(ResourceType rsrc);

  //===--------------------------------------------------------------------===//
  // Allocatable resources: how many units to build, not how many exist. An
  // allocatable resource carries no limit, so `holdsLimitedUnit` stays false
  // and no heuristic reservation table sees it.
  //===--------------------------------------------------------------------===//

  /// What one allocatable resource may cost and how many of it may exist.
  struct AllocatableUnit {
    /// Max units. The trivial allocation is one per linked op, so declaring a
    /// resource never makes a problem infeasible.
    unsigned ceiling = 0;
    /// Cost of building `n` instances, indexed by `n` over `[0, ceiling]`: the
    /// instances plus the muxes they put in front of the ops sharing each. A
    /// table, not a coefficient: a mux's per-bit cost rises in plateaus (a LUT6
    /// absorbs three source/select pairs), so the total is not monotone in `n`.
    llvm::SmallVector<int64_t> price;
    /// Delay (ns) of the select cone in front of the fullest instance at `n`
    /// instances, indexed like `price`; zero at the ceiling where nothing
    /// shares. Charged on every linked op's sub-cycle start, so a count shrinks
    /// only where the cone fits the slack the schedule leaves.
    llvm::SmallVector<double> headroomNs;
  };

  void setAllocatable(ResourceType rsrc, AllocatableUnit unit) {
    allocatable[rsrc] = unit;
  }
  std::optional<AllocatableUnit> getAllocatable(ResourceType rsrc) {
    return allocatable.lookup(rsrc);
  }

  /// How many units a solve decided to build. Absent until one does, leaving
  /// the trivial allocation in force.
  void setAllocation(ResourceType rsrc, unsigned units) {
    allocation[rsrc] = units;
  }
  std::optional<unsigned> getAllocation(ResourceType rsrc) {
    return allocation.lookup(rsrc);
  }

  /// Which instance of its allocatable operator \p op runs on: an index below
  /// `getAllocation`. Absent until `assignUnits` derives it, and for ops on
  /// nothing allocatable.
  std::optional<unsigned> getAssignedUnit(Operation *op) {
    return assignedUnit.lookup(op);
  }

  /// Record which instance \p op runs on, for a caller deriving the assignment
  /// itself rather than via `assignUnits`.
  void setAssignedUnit(Operation *op, unsigned index) {
    assignedUnit[op] = index;
  }

  /// Turn every decided count into an assignment of ops to instances, spread
  /// round-robin over the instances bought rather than packed into the fewest
  /// that fit. Valid at the offered occupancies: cyclic (\p ii > 0) occupancy
  /// is one cycle, so 0, 1, 2, ... within each congruence class fits its bound;
  /// acyclic (\p ii == 0) windows form an interval graph, so the busiest
  /// cycle's count suffices.
  void assignUnits(unsigned ii);

  /// Whether \p op contends for a resource whose count is being decided.
  bool holdsAllocatableUnit(Operation *op);

  /// The fewest units of \p rsrc the current schedule needs: the busiest cycle
  /// of its occupancy windows, or busiest congruence class at non-zero \p ii.
  /// Every op must be scheduled. The count `assignUnits` can place (windows on
  /// a line form an interval graph, first fit in start order colours it), and
  /// the same histogram `verifyOccupancy` checks against a limit.
  unsigned demandFor(ResourceType rsrc, unsigned ii);

  /// Whether \p op contends for anything: a capped or allocated unit. Such an
  /// op needs a congruence class in a modulo model.
  bool contendsForUnit(Operation *op) {
    return holdsLimitedUnit(op) || holdsAllocatableUnit(op);
  }

  /// No two ops on one instance contend for it in the same cycle, and no
  /// instance index exceeds the decided count. Vacuous where no solve set an
  /// allocation.
  LogicalResult verifyAllocation(unsigned ii);

  /// No limited resource is oversubscribed in any cycle. \p ii == 0 checks an
  /// acyclic schedule; non-zero \p ii checks the windows modulo the II. Not an
  /// override: the concrete problems below call it from their `verify`.
  LogicalResult verifyOccupancy(unsigned ii);

private:
  OperationProperty<unsigned> resourceCycles;
  OperationProperty<unsigned> resourceDemand;
  ResourceTypeProperty<AllocatableUnit> allocatable;
  ResourceTypeProperty<unsigned> allocation;
  OperationProperty<unsigned> assignedUnit;
  llvm::DenseSet<Dependence> forwardedEdges;
};

/// The cyclic twin: CIRCT's `ModuloProblem` with occupancy windows, i.e.
/// reservations that span several congruence classes modulo the II.
class ModuloOccupancyProblem : public virtual circt::scheduling::ModuloProblem,
                               public virtual OccupancyProblem {
public:
  static constexpr auto name = "ModuloOccupancyProblem";
  using circt::scheduling::ModuloProblem::ModuloProblem;

protected:
  ModuloOccupancyProblem() = default;
  /// A forwarded store->load edge needs only issue order (the shadow supplies
  /// the datum), so it is checked at latency zero.
  LogicalResult verifyPrecedence(Dependence dep) override;

public:
  LogicalResult verify() override;
};

/// A cyclic, resource-constrained, chaining-enabled problem: CIRCT's
/// `ChainingProblem` composed with `ModuloOccupancyProblem`. Solving yields an
/// integer II, integer and sub-cycle start times respecting the target period,
/// under modulo resource constraints.
class ChainingModuloProblem : public virtual circt::scheduling::ChainingProblem,
                              public virtual ModuloOccupancyProblem {
public:
  static constexpr auto name = "ChainingModuloProblem";
  using circt::scheduling::ChainingProblem::ChainingProblem;

protected:
  ChainingModuloProblem() = default;

public:
  LogicalResult checkDefUse(circt::scheduling::Problem::Dependence dep);
  LogicalResult check() override;
  LogicalResult verify() override;
};

/// An acyclic, resource-constrained, chaining-enabled problem: CIRCT's
/// `ChainingProblem` composed with `OccupancyProblem`. The straight-line twin
/// of `ChainingModuloProblem`, no II and no inter-iteration distance.
class ChainingSharedOperatorsProblem
    : public virtual circt::scheduling::ChainingProblem,
      public virtual OccupancyProblem {
public:
  static constexpr auto name = "ChainingSharedOperatorsProblem";
  using circt::scheduling::ChainingProblem::ChainingProblem;

protected:
  ChainingSharedOperatorsProblem() = default;

public:
  LogicalResult check() override;
  LogicalResult verify() override;
};

/// The chain-breaking edges \p prob needs to meet \p cycleTime: for every
/// combinational path whose accumulated delay would not fit the period, an
/// auxiliary edge from the path's ORIGIN to the operation, which both solvers
/// weigh one cycle more than a plain dependence. Schedule-independent, so a
/// caller may run it before or after solving.
///
/// The edges state the period exactly over integer start times: an
/// over-period pair must sit a cycle apart in any schedule (same-cycle
/// endpoints pull every zero-latency intermediate into their cycle), and a
/// schedule separating every such pair leaves no cycle holding an over-period
/// chain. Register placement inside a broken chain stays the solver's.
///
/// Visits operations in topological order and marks one "handled" only once
/// every predecessor's chain map is complete, so a successor never inherits a
/// half-built map.
/// \p regFloor is the earliest sub-cycle time any operation may start at, so
/// every chain begins having already spent it.
/// Every operator fits \p cycleTime on its own (asserted): `runSDCScheduler`
/// derates the period before any problem is built.
LogicalResult computeChainBreaks(
    circt::scheduling::ChainingProblem &prob, float cycleTime, float regFloor,
    SmallVectorImpl<circt::scheduling::Problem::Dependence> &result);

/// `circt::scheduling::computeStartTimesInCycle` with a floor: an operation's
/// sub-cycle start is at least \p regFloor, where CIRCT's takes zero. CIRCT
/// models an ideal register whose result is available at time 0.0 of the cycle
/// it is read in; a real one costs clock-to-out plus routing (0.419 ns on
/// xcu55c, against a 3.333 ns period). A chain from a registered node then
/// costs `max(regFloor, that node's outgoing delay)`.
LogicalResult computeStartTimesInCycle(circt::scheduling::ChainingProblem &prob,
                                       float regFloor);

//===----------------------------------------------------------------------===//
// SDC simplex schedulers.
//
// Fork of CIRCT's `scheduleSimplex` family (implementation in Scheduler.cpp).
// Call these via `solveSchedulingProblem` below or by fully-qualified name
// (`mlir::allo::scheduleSimplex`) to avoid ambiguity with the CIRCT overloads.
//
// Two entries, for the two problems this backend builds. The resource-free and
// non-chaining rungs of CIRCT's family have no caller here: every Allo region
// is solved against a clock period.
//===----------------------------------------------------------------------===//

/// What the SDC heuristic contributes to a solve that is not its own: the II
/// bound it settles before placing, and whether its greedy placement reached a
/// schedule. Passing one makes a placement failure advisory (the call succeeds
/// with `placed == false`); a failure in the exact resource-free LP below
/// placement still fails the call, since infeasible there means no schedule at
/// any II.
struct SimplexWarmStart {
  /// Largest II any bound justifies before placement: the max of the
  /// resource-min II, a loop-carried recurrence, and the pipeline floor. Where
  /// an exact II search starts.
  unsigned lowerBoundII = 1;
  /// Whether greedy placement reached a schedule: start times and an II now
  /// present.
  bool placed = false;
};

/// \p minII is a lower bound on the initiation interval (from a pipeline
/// directive); the achieved II is max(\p minII, the natural minimum). The
/// default 1 imposes no additional bound.
///
/// \p warm, when given, receives the warm start above and switches placement
/// failures to advisory.
LogicalResult scheduleSimplex(ChainingModuloProblem &prob, Operation *lastOp,
                              float cycleTime, float regFloor,
                              unsigned minII = 1,
                              SimplexWarmStart *warm = nullptr);
LogicalResult scheduleSimplex(ChainingSharedOperatorsProblem &prob,
                              Operation *lastOp, float cycleTime,
                              float regFloor);

//===----------------------------------------------------------------------===//
// What a solve is charged: the span objective.
//===----------------------------------------------------------------------===//

/// One region output's contribution to the drain: it commits at
/// `start(op) + offset`, plus the linked operator's latency where
/// `plusLatency`. Latency is read off the problem at composition time, not
/// baked into the offset, since which row produces the value may be a solver
/// decision.
struct DrainTerm {
  Operation *op;
  int64_t offset;
  bool plusLatency = false;
};

/// The drain of a solved problem: the cycle its deepest output commits.
inline int64_t drainOf(circt::scheduling::Problem &problem,
                       ArrayRef<DrainTerm> terms) {
  int64_t drain = 0;
  for (const DrainTerm &term : terms) {
    int64_t at =
        static_cast<int64_t>(*problem.getStartTime(term.op)) + term.offset;
    if (term.plusLatency)
      at += *problem.getLatency(*problem.getLinkedOperatorType(term.op));
    drain = std::max(drain, at);
  }
  return drain;
}

/// One value a region spends a delay register chain on. The chain is as long
/// as its deepest reader needs and costs the device's price for that many
/// stages at this width:
///
/// ```
/// depth(v) = max over reads ( t_read + ii * distance ) - ( t_def + latency )
/// cost(v)  = chainPrice( stages(v), width )
/// ```
///
/// No register is shared between values (`insertRegister` keys one chain per
/// value and region), so the cost is a sum over values linear in the schedule,
/// not a max-live coupled to an allocation, and an objective can carry it
/// directly. `stages(v)` is `depth(v)`, except at II > 1 where the emitter
/// folds the chain onto the phase: `depth` cycles are built from
/// `ceil(depth / ii)` registers (`EmitContext::foldedChain`). `latency` is the
/// definer's, read live off the model since which row realizes it may be a
/// solver decision.
struct RegisterTerm {
  Operation *def;
  /// Flip-flops one cycle of delay costs.
  int64_t width;
  /// Each reader, and the iteration distance its read spans.
  SmallVector<std::pair<Operation *, int64_t>> reads;
};

/// What a region's span is charged, and so what the exact scheduler minimizes:
/// `(trip - 1) * ii + drain`, the part of `leafSpan` a solve controls; area is
/// decided in a second solve under the span the first settles. The heuristic
/// ignores this and minimizes the anchor's start time instead.
struct SpanObjective {
  /// Read one region's charge off \p problem, which needs its operator types
  /// but not a solution: a term's cost is a property of the region, only where
  /// it lands a property of the schedule. \p results are the values escaping
  /// the region, \p carried the counted-loop body whose block arguments after
  /// the induction variable are its iter_args (null with no such recurrence: a
  /// straight-line span, a `while`), \p device what the area terms are priced
  /// against.
  SpanObjective(OccupancyProblem &problem, ValueRange results, Block *carried,
                std::optional<int64_t> trip, const OperatorLibrary &device);

  /// The region's outputs.
  SmallVector<DrainTerm> drain;
  /// The values it spends a delay register on.
  SmallVector<RegisterTerm> regs;
  /// Trip count when compile-time constant. Empty leaves the scheduler on the
  /// anchor-start objective, right wherever no span composes off this solve (a
  /// `while`, a dynamic bound) or iterations do not overlap and the trip
  /// multiplies schedule depth not the drain (`s.pipeline(ii=-1)`).
  std::optional<int64_t> trip;
  /// The device the area terms are priced against, so a register, a mux and an
  /// operator are comparable rather than flip-flops ranked against DSP slices.
  const OperatorLibrary &device;

  /// Where this region's deepest output commits in a solved \p problem.
  int64_t drainOf(circt::scheduling::Problem &problem) const {
    return mlir::allo::drainOf(problem, drain);
  }
};

//===----------------------------------------------------------------------===//
// CP-SAT exact schedulers.
//
// Which solver settles the RESOURCE half of a problem. The SDC simplex is exact
// for the difference constraints either way; only the resource placement
// differs, greedy over an MRT there and one constraint program here.
//===----------------------------------------------------------------------===//

enum class SchedulerKind {
  /// The SDC simplex plus greedy modulo / shared-operator placement.
  Heuristic,
  /// CP-SAT over the same problem: exact under the model. Chain breaks stay the
  /// pre-pass's (`computeChainBreaks`), which state the period exactly, so only
  /// resource placement differs from the heuristic. Where a device offers
  /// several usable rows for an op, which row realizes it is also this solver's
  /// decision; the heuristic keeps the library's pick.
  Exact,
};

/// Whether \p kind solves the resource half with CP-SAT.
inline bool usesExactScheduler(SchedulerKind kind) {
  return kind != SchedulerKind::Heuristic;
}

/// The direction a solve optimizes toward. Only an exact solve reads it.
enum class ScheduleObjective {
  /// Shortest span, with area breaking ties under it.
  Cycles,
  /// Smallest area under a span leash (no slower than the heuristic), span
  /// slack reclaimed under the settled area.
  Area,
};

/// Defaults for one solve. The budget is in OR-Tools deterministic time units
/// (roughly a core-second), charged per solve and shared by its span and area
/// passes, so a cyclic search spends it again at every II it probes.
/// Reproducibility comes from the fixed seed plus the interleaved portfolio
/// above one worker while `deterministic` holds; a budget-exhausted solve can
/// still differ run to run. The worker count is not only a speed knob: the same
/// budget buys more search, so a budget-limited region can settle differently
/// at a different count.
inline constexpr double kDefaultSolveBudget = 30.0;
inline constexpr int kDefaultSolveWorkers = 8;
inline constexpr int kDefaultSolveSeed = 0;

/// What the caller asked the scheduler for.
struct SchedulerOptions {
  SchedulerKind kind = SchedulerKind::Heuristic;
  ScheduleObjective objective = ScheduleObjective::Cycles;
  double budget = kDefaultSolveBudget;
  /// Whether to decide how many copies of each operator a region builds
  /// (`populateOperatorAllocation`) rather than one per op. Meaningful only
  /// with a binding that folds them (the trivial binding builds one per op
  /// anyway); the heuristic ignores it. An op whose realization the exact
  /// solver decides (`selectionCandidates`) is composed through a shared class,
  /// straight-line and modulo alike.
  bool allocate = false;
  int workers = kDefaultSolveWorkers;
  int seed = kDefaultSolveSeed;
  /// Whether workers advance in a fixed interleaved order, so two identical
  /// compiles emit identical RTL. Off, above one worker, they race, each held
  /// to budget / workers seconds of wall-clock; the optimum then depends on
  /// thread timing, so no exact solve is reproducible.
  bool deterministic = true;
  /// Span the area objective may pay beyond its leash, as a fraction of the
  /// reference span (the heuristic's, or the first solved interval the greedy
  /// did not place). Zero ships no slower than the heuristic. Buys interval
  /// room for unit folds alone: an interval the ungranted leash already admits
  /// keeps its tight drain bound.
  double areaSlack = 0.0;
  /// Register-to-register floor (ns): the earliest sub-cycle start any op may
  /// take. Combinational rows carry their measured delay less the floor, so a
  /// cycle pays it once however many operators chain.
  float regFloor = 0.0f;
};

/// \p name ("heuristic" / "exact") as a kind, or nullopt when it names
/// neither.
std::optional<SchedulerKind> parseSchedulerKind(StringRef name);

/// \p name ("cycles" / "area") as an objective, or nullopt when it names
/// neither.
std::optional<ScheduleObjective> parseScheduleObjective(StringRef name);

/// One region's operator-sharing problem, decided at bind time with the
/// schedule fixed: which same-class units to fold onto one instance. Numeric
/// throughout, so the emitter hands it over without this header knowing its
/// model. A shared instance grows one select per operand port, one arm per
/// member plus each member's re-injected recurrence identities; tables are per
/// port, indexed by arms, and a one-arm select is a wire (indices 0 and 1 are
/// zero).
struct SharingProblem {
  struct Port {
    /// The select at this port's own width, by arms.
    llvm::SmallVector<int64_t> muxPrice;
    /// Its delay in picoseconds, by arms.
    llvm::SmallVector<int64_t> conePicos;
  };
  struct UnitClass {
    /// One instance of the operator, in the device's currency.
    int64_t instancePrice = 0;
    llvm::SmallVector<Port, 0> ports;
  };
  struct Unit {
    unsigned cls = 0; // index into `classes`; only equal ones may fold
    /// The room the schedule left for this operation's whole input cone.
    int64_t slackPicos = 0;
    /// Same-cycle combinational producers, as (port, unit): a producer's cone
    /// arrives through the select of the port it drives.
    llvm::SmallVector<std::pair<unsigned, unsigned>, 2> preds;
    /// Per port: select arms past the data arm, one per recurrence identity the
    /// op re-injects there.
    llvm::SmallVector<unsigned, 2> initArms;
    /// Per port: a nonzero key marks a held operand (a wire at any issue
    /// cycle), equal keys equal values; members all sharing one key collapse to
    /// that wire and build no select. 0 marks a scheduled or carried operand,
    /// which never collapses.
    llvm::SmallVector<unsigned, 2> drivers;
  };
  llvm::SmallVector<UnitClass, 0> classes;
  llvm::SmallVector<Unit, 0> units;
  /// Same-class pairs `(i, j)`, `i < j`, that may not share an instance: their
  /// reservations collide.
  llvm::SmallVector<std::pair<unsigned, unsigned>> conflicts;
};

/// Decide the fold exactly: returns, for each unit, the unit it runs on (the
/// smallest member of its group; itself where unshared). Minimizes the
/// modelled area, instances plus multiplexers with fewer folds breaking ties,
/// holding every unit's input cone within `slackPicos` under the recursion the
/// emit gate walks (`AddedDelay`): a bin's select plus everything a same-cycle
/// producer's bin adds. \p hint seeds the search as an incumbent. \p anchor is
/// where diagnostics land. Returns nullopt when the budget expires with
/// nothing usable.
std::optional<SmallVector<unsigned>> solveSharing(SharingProblem &problem,
                                                  ArrayRef<unsigned> hint,
                                                  Operation *anchor);

/// Solve \p prob exactly with CP-SAT, minimizing \p span under the target clock
/// period \p cycleTime.
LogicalResult scheduleCPSAT(ChainingSharedOperatorsProblem &prob,
                            Operation *lastOp, float cycleTime,
                            const SpanObjective &span,
                            const SchedulerOptions &opts);
/// Cyclic twin; \p minII lower-bounds the initiation interval, and the search
/// over intervals is a branch and bound on \p span. \p maxII, nonzero, caps
/// the search under the area objective (an explicit pipeline directive's
/// ceiling, held no lower than the natural floor); the cycles objective
/// ignores it. \p slackGrant, nonzero, is composition slack the enclosing
/// kernel proved free of this region (off the sibling DAG's longest path). It
/// buys interval room alone under the area objective: intervals the ungranted
/// leash admits keep their own tight drain bound, wider ones are admitted with
/// what the grant leaves. The cycles objective ignores it.
LogicalResult scheduleCPSAT(ChainingModuloProblem &prob, Operation *lastOp,
                            float cycleTime, unsigned minII, unsigned maxII,
                            const SpanObjective &span,
                            const SchedulerOptions &opts,
                            int64_t slackGrant = 0);

/// Check, solve, and verify \p problem, minimizing the span \p span charges.
/// The chaining modulo variant, with a target-II lower bound (from a pipeline
/// directive): the achieved II is max(\p minII, the natural minimum). \p minII
/// == 1 imposes no additional bound. \p maxII, nonzero, is an explicit
/// directive's II ceiling, honored by the exact area objective alone. \p opts
/// selects the resource solver; both paths go through the same `check` and
/// `verify`.
inline LogicalResult
solveSchedulingProblem(ChainingModuloProblem &problem, Operation *anchor,
                       float cycleTime, unsigned minII,
                       const SchedulerOptions &opts, const SpanObjective &span,
                       unsigned maxII = 0, int64_t slackGrant = 0) {
  if (failed(problem.check()))
    return failure();
  if (usesExactScheduler(opts.kind)) {
    if (failed(scheduleCPSAT(problem, anchor, cycleTime, minII, maxII, span,
                             opts, slackGrant)))
      return failure();
  } else if (failed(mlir::allo::scheduleSimplex(problem, anchor, cycleTime,
                                                opts.regFloor, minII))) {
    return failure();
  }
  if (failed(problem.verify()))
    return failure();
  return success();
}

/// Acyclic twin of the variant above. A slack grant has no interval to buy in
/// a straight-line region, so none is taken.
inline LogicalResult solveSchedulingProblem(
    ChainingSharedOperatorsProblem &problem, Operation *anchor, float cycleTime,
    const SchedulerOptions &opts, const SpanObjective &span) {
  if (failed(problem.check()))
    return failure();
  if (usesExactScheduler(opts.kind)) {
    if (failed(scheduleCPSAT(problem, anchor, cycleTime, span, opts)))
      return failure();
  } else if (failed(mlir::allo::scheduleSimplex(problem, anchor, cycleTime,
                                                opts.regFloor))) {
    return failure();
  }
  if (failed(problem.verify()))
    return failure();
  return success();
}

/// Reject a kernel the backend cannot schedule at all: an unmodelled memory
/// effect, an unrealizable operator, an illegal channel or partition.
/// Everything here is a property of the input, so it is settled before a
/// single problem is built. Timing is not among the refusals: an operator or
/// address cone past the clock period derates the period at schedule time
/// instead.
LogicalResult runPreScheduleVerification(ModuleOp module, StringRef top);

/// Solve the schedule of every func reachable from \p top, recording it in
/// \p model. The IR is left in affine/scf form; nothing is materialized.
/// \p cycleTime is the RESOLVED target period in ns (the caller applies the
/// default). A target no single operator fits is raised to the least period
/// every device row does, with a warning naming the rows; the period the
/// schedule holds is published as `model.cycleTimeNs` either way.
LogicalResult runSDCScheduler(ModuleOp module, StringRef top, float cycleTime,
                              const SchedulerOptions &opts,
                              ScheduleModel &model);

/// Reify \p model onto the IR as `dcp.*` regions. It runs immediately after the
/// scheduler over the same module, which is what keeps the model's `Operation
/// *` keys valid; it also ADDS to the model, for the condition cones and
/// symbolic bounds it schedules itself.
void runPostScheduleConversion(ModuleOp module, ScheduleModel &model);

} // namespace mlir::allo

#endif // ALLO_SCHEDULING_SCHEDULER_H
