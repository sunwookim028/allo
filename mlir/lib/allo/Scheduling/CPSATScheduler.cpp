/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Scheduling/OperatorLibrary.h" // the device the area is priced at
#include "allo/Scheduling/Scheduler.h"
#include "allo/Support/Logging.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Format.h"

#include "ortools/sat/cp_model.h"
#include "ortools/sat/cp_model_solver.h"

#include <cmath>
#include <fstream>
#include <limits>
#include <type_traits>

using namespace mlir;
using namespace mlir::allo;
using namespace mlir::allo::logging;

std::optional<SchedulerKind> mlir::allo::parseSchedulerKind(StringRef name) {
  return llvm::StringSwitch<std::optional<SchedulerKind>>(name)
      .Case("heuristic", SchedulerKind::Heuristic)
      .Case("exact", SchedulerKind::Exact)
      .Default(std::nullopt);
}

using namespace circt::scheduling;
using namespace operations_research::sat;

namespace {

/// Solver configuration for one solve. A deterministic time limit under an
/// interleaved portfolio lets two identical compiles emit identical RTL; a
/// solve that exhausts the limit can still differ run to run.
SatParameters solverParameters(const SchedulerOptions &opts) {
  SatParameters params;
  params.set_num_workers(opts.workers);
  params.set_random_seed(opts.seed);
  params.set_max_deterministic_time(opts.budget);
  // Interleaved, the portfolio advances in a fixed order under the
  // deterministic limit, so the schedule and the proven optimum are
  // reproducible. Racing workers depend on thread timing, and each reports its
  // deterministic time only on return, so a wall-clock cap holds the race to
  // the budget's worth of core-seconds.
  if (opts.workers > 1 && opts.deterministic)
    params.set_interleave_search(true);
  else if (opts.workers > 1)
    params.set_max_time_in_seconds(opts.budget / opts.workers);
  // Solver progress log, for diagnosing where a budget goes on a model that
  // returns nothing.
  if (getenv("ALLO_CPSAT_LOG")) {
    params.set_log_search_progress(true);
    params.set_log_to_stdout(true);
  }
  return params;
}

/// Solve \p model under \p params. Below this variable count a single worker
/// runs, since such a model is solved before a portfolio loads; above it the
/// portfolio proves an area model that a lone worker would burn the budget on.
constexpr int kPortfolioFloorVars = 32;
CpSolverResponse solveBuilt(CpModelBuilder &model, SatParameters params) {
  // An empty solution hint still routes the interleaved workers down the
  // hinted path, so drop it outright when the model carries no hint.
  if (model.Proto().solution_hint().vars().empty())
    model.MutableProto()->clear_solution_hint();
  const CpModelProto &proto = model.Build();
  if (proto.variables_size() < kPortfolioFloorVars) {
    params.set_num_workers(1);
    params.set_interleave_search(false);
  }
  // Every model as built, numbered, for replaying one outside the compiler.
  if (const char *dir = getenv("ALLO_CPSAT_DUMP")) {
    static int serial = 0;
    std::ofstream out(std::string(dir) + "/model-" + std::to_string(serial++) +
                          ".pb",
                      std::ios::binary);
    proto.SerializeToOstream(&out);
  }
  return SolveWithParameters(proto, params);
}

/// What is left of the budget once \p spent's solve has taken its share.
static SchedulerOptions lessBudget(SchedulerOptions opts,
                                   const CpSolverResponse &spent) {
  opts.budget = std::max(opts.budget - spent.deterministic_time(), 0.0);
  return opts;
}

/// How the model states the clock period: the chain-breaking edges the
/// pre-pass computed, each costing a cycle on top of plain precedence. The
/// edges hold the period exactly over integer start times (see
/// `computeChainBreaks`), so no sub-cycle constraint is needed for
/// feasibility. They hold for the delays of the rows currently linked, so a
/// model that decides realizations itself uses the sub-cycle system instead
/// (`addSubCycleTimes` with selection).
template <class ProblemT>
SmallVector<Problem::Dependence> chainBreaksFor(ProblemT &prob, float cycleTime,
                                                float regFloor) {
  SmallVector<Problem::Dependence> breaks;
  auto broke =
      mlir::allo::computeChainBreaks(prob, cycleTime, regFloor, breaks);
  assert(succeeded(broke) && "chain breaking is a pure function of the problem "
                             "and the cycle time, and the heuristic just ran "
                             "it successfully");
  (void)broke;
  return breaks;
}

/// Sub-cycle time in picoseconds, rounded to nearest: CP-SAT is integer, and a
/// picosecond has enough resolution against delays given to a hundredth of a
/// nanosecond. Round-to-nearest rather than up, since a chain that fills the
/// period exactly is common and rounding up would reject it.
constexpr double kPicosPerNs = 1000.0;
int64_t picos(double ns) { return std::llround(ns * kPicosPerNs); }

//===----------------------------------------------------------------------===//
// Realization selection: which of several usable device rows an operation runs
// on, decided by the solve alongside its start time.
//===----------------------------------------------------------------------===//

/// One operation whose realization the solve decides, and the rows it may
/// choose among. Model-independent, so the cyclic search builds one model per
/// interval off the same choices.
struct SelectionChoice {
  Operation *op;
  SmallVector<OperatorChar, 2> cands;
  /// The candidate `lookup` resolves on its own: the warm-start hint, and the
  /// row every fallback path leaves linked.
  unsigned preferred = 0;
};

/// The operations whose realization a solve decides, each with its usable
/// rows (`selectionCandidates`, which also keeps such an operation out of
/// every allocation class), and the library's own pick located among them.
template <class ProblemT>
SmallVector<SelectionChoice, 0> selectionChoices(ProblemT &prob,
                                                 const OperatorLibrary &lib) {
  constexpr bool cyclic = std::is_base_of_v<CyclicProblem, ProblemT>;
  SmallVector<SelectionChoice, 0> choices;
  for (Operation *op : prob.getOperations()) {
    SmallVector<OperatorChar, 2> cands = selectionCandidates(op, lib, cyclic);
    if (cands.empty())
      continue;
    std::string own = lib.lookup(op).timing.typeName;
    const auto *pos = llvm::find_if(
        cands, [&](const OperatorChar &c) { return c.timing.typeName == own; });
    assert(pos != cands.end() &&
           "a non-empty candidate set holds the library's own pick");
    auto preferred = static_cast<unsigned>(pos - cands.begin());
    choices.push_back({op, std::move(cands), preferred});
  }
  return choices;
}

/// The one-hot decision per choice on one model, hinted at the library's own
/// pick. Aligned with the choices it was built from.
struct SelectionVars {
  ArrayRef<SelectionChoice> choices;
  SmallVector<SmallVector<BoolVar, 2>, 4> sel;
  DenseMap<Operation *, unsigned> index; // op -> position in `choices`

  bool empty() const { return choices.empty(); }
  /// The position of \p op's choice, nullopt where the solve does not decide
  /// its realization.
  std::optional<unsigned> of(Operation *op) const {
    auto it = index.find(op);
    return it == index.end() ? std::nullopt : std::optional(it->second);
  }
  /// The decided latency of choice \p i, as the one-hot weighted sum.
  LinearExpr latency(unsigned i) const {
    SmallVector<int64_t> lats;
    for (const OperatorChar &c : choices[i].cands)
      lats.push_back(c.timing.latency);
    return LinearExpr::WeightedSum(sel[i], lats);
  }
};

SelectionVars addSelection(CpModelBuilder &model,
                           ArrayRef<SelectionChoice> choices) {
  SelectionVars sels;
  sels.choices = choices;
  for (auto [i, choice] : llvm::enumerate(choices)) {
    SmallVector<BoolVar, 2> row;
    for (unsigned m = 0; m < choice.cands.size(); ++m) {
      row.push_back(model.NewBoolVar());
      model.AddHint(row.back(), m == choice.preferred);
    }
    model.AddExactlyOne(row);
    sels.sel.push_back(std::move(row));
    sels.index.try_emplace(choice.op, i);
  }
  return sels;
}

/// The latency extremes selection leaves each decided op, keyed by op. The
/// minimum keeps `drainFloor` a bound on every selection; the maximum keeps
/// the horizon covering one.
DenseMap<Operation *, std::pair<int64_t, int64_t>>
latencyRange(ArrayRef<SelectionChoice> choices) {
  DenseMap<Operation *, std::pair<int64_t, int64_t>> range;
  for (const SelectionChoice &choice : choices) {
    int64_t lo = std::numeric_limits<int64_t>::max(), hi = 0;
    for (const OperatorChar &c : choice.cands) {
      lo = std::min<int64_t>(lo, c.timing.latency);
      hi = std::max<int64_t>(hi, c.timing.latency);
    }
    range.try_emplace(choice.op, lo, hi);
  }
  return range;
}

/// The candidate index each choice settled on in \p response.
SmallVector<unsigned> readSelection(const CpSolverResponse &response,
                                    const SelectionVars &sels) {
  SmallVector<unsigned> chosen;
  for (const SmallVector<BoolVar, 2> &row : sels.sel) {
    unsigned m = 0;
    while (!SolutionBooleanValue(response, row[m]))
      ++m;
    chosen.push_back(m);
  }
  return chosen;
}

/// What the decided rows cost the device: the term the cyclic search adds to
/// an interval's allocation area when comparing intervals. A pair in
/// \p covered is priced by a shared class instead and skipped here.
int64_t selectionPrice(ArrayRef<SelectionChoice> choices,
                       ArrayRef<unsigned> chosen,
                       const DenseSet<std::pair<unsigned, unsigned>> &covered) {
  int64_t price = 0;
  for (auto [i, choice] : llvm::enumerate(choices))
    if (!covered.contains({static_cast<unsigned>(i), chosen[i]}))
      price += choice.cands[chosen[i]].price;
  return price;
}

/// Write the decided realization of every choice back onto the problem, so
/// everything downstream of the solve (the sub-cycle recompute, verification,
/// the reify) reads the decided rows. An IP row's operator type is keyed by
/// its symbol, so relinking is what carries the decision out.
void applySelection(circt::scheduling::ChainingProblem &prob,
                    ArrayRef<SelectionChoice> choices,
                    ArrayRef<unsigned> chosen) {
  unsigned moved = 0;
  for (auto [choice, m] : llvm::zip(choices, chosen)) {
    const OperatorChar &cand = choice.cands[m];
    Problem::OperatorType opr =
        prob.getOrInsertOperatorType(cand.timing.typeName);
    prob.setLatency(opr, cand.timing.latency);
    prob.setIncomingDelay(opr, cand.timing.inDelay);
    prob.setOutgoingDelay(opr, cand.timing.outDelay);
    prob.setLinkedOperatorType(choice.op, opr);
    moved += m != choice.preferred;
  }
  if (moved)
    info(Stage::Sched, prob.getContainingOp())
        << "Exact scheduling moved " << moved << " of " << choices.size()
        << " realization decisions off the library's own ranking";
}

//===----------------------------------------------------------------------===//
// Shared classes: the composition of selection and allocation on one model.
// One class per candidate row several operations could run on, its instance
// count decided with which operations select it. Membership is collected once
// per region; an acyclic model states the capacity as a cumulative over
// occupancy intervals, a modulo model as a per-slot sum of slot-and-selection
// products.
//===----------------------------------------------------------------------===//

/// One shared class. A static member (an operation whose realization is not a
/// solver decision) is always on it; a conditional member joins on its `sel`
/// literal, so the class prices what folding a converged selection saves.
struct SharedClassVar {
  Problem::ResourceType rsrc;
  /// One instance's price and its select shape, the inputs of the (members,
  /// units) tables below.
  int64_t unitPrice = 0, ports = 0, width = 1;
  /// Members reading a loop-carried operand, counted over everyone who could
  /// join: a shared unit re-injects such an operand on a select arm of its own
  /// (`populateOperatorAllocation`), charged conservatively since the cone is
  /// tabulated before the membership is decided. Zero on an acyclic region.
  unsigned carried = 0;
  /// Static members: the operation, its occupancy, and its incoming delay in
  /// picoseconds (what the headroom is held against).
  SmallVector<std::tuple<Operation *, unsigned, int64_t>> statics;
  /// Conditional members: the choice, the candidate naming this row, and that
  /// candidate's occupancy.
  struct Cond {
    unsigned choice, cand, occ;
  };
  SmallVector<Cond> conds;
  /// Whether `populateOperatorAllocation` declared this row, i.e. its static
  /// members are already linked to it; a row only selection reaches is
  /// written onto the problem at adoption.
  bool onProblem = false;
  /// The count to build, and how many members actually joined. `units <=
  /// members`: more instances than members is never cheaper and leaves no
  /// smaller cone, and the bound keeps the decided count inside the
  /// realizable table slice.
  IntVar units, members;
  IntVar price;
  std::optional<IntVar> headroom;
};

/// The classes of one model, with the (choice, candidate) pairs they price
/// (so the per-operation price terms skip them) and the rows they own (so
/// `allocationVars` does not price them a second time).
struct SharedClasses {
  SmallVector<SharedClassVar, 0> classes;
  DenseSet<std::pair<unsigned, unsigned>> covered;
  DenseSet<Problem::ResourceType> owned;
  bool empty() const { return classes.empty(); }
};

/// What \p n instances hosting \p k members of \p cls cost: the instances
/// plus the select in front of every port, members spread round-robin as
/// `assignUnits` spreads them (`k % n` instances host one more). The table
/// `populateOperatorAllocation` builds is this at k fixed to the ceiling;
/// here the membership is decided, so the price is a function of both.
int64_t sharedPrice(const OperatorLibrary &lib, const SharedClassVar &cls,
                    int64_t k, int64_t n) {
  if (n == 0)
    return 0;
  int64_t busy = k % n, share = k / n;
  auto armPrice = [&](int64_t arms) {
    return arms < 1 ? 0 : lib.muxPrice(arms, cls.width);
  };
  return n * cls.unitPrice + cls.ports * (busy * armPrice(share + 1) +
                                          (n - busy) * armPrice(share));
}

/// The fullest instance's select cone at (\p k, \p n), in ns: `ceil(k / n)`
/// source arms plus one re-injection arm per carried member among them. Zero
/// once every member has its own instance: nothing shares, so there is no
/// select to re-inject through either.
double sharedConeNs(const OperatorLibrary &lib, const SharedClassVar &cls,
                    int64_t k, int64_t n) {
  if (n == 0 || k <= n)
    return 0.0;
  auto fullest = static_cast<unsigned>((k + n - 1) / n);
  unsigned arms = fullest + std::min(fullest, cls.carried);
  return muxCone(lib, arms,
                 static_cast<unsigned>(std::max<int64_t>(1, cls.width)));
}

/// Collect the shared classes of one region: for every candidate row, who
/// could run on it and the shape of one instance. Metadata only, no model
/// variables, so the cyclic search collects once and builds each fixed-II
/// model on it. In a cyclic region a multi-cycle static stays out (a count
/// per congruence slot is not an assignment for it, the same refusal as
/// `populateOperatorAllocation`), and every conditional member occupies one
/// cycle (`selectionCandidates` keeps only pipelined rows there).
template <class ProblemT>
SharedClasses collectSharedClasses(ProblemT &prob, const OperatorLibrary &lib,
                                   ArrayRef<SelectionChoice> choices) {
  constexpr bool isCyclic =
      std::is_base_of_v<circt::scheduling::CyclicProblem, ProblemT>;
  SharedClasses shared;
  if (choices.empty())
    return shared;
  DenseMap<Operation *, unsigned> choiceOf;
  for (auto [i, choice] : llvm::enumerate(choices))
    choiceOf.try_emplace(choice.op, static_cast<unsigned>(i));
  // The loop whose carried values a shared unit re-injects; its own induction
  // variable is not carried.
  Operation *container = prob.getContainingOp();
  Value inductionVar;
  if (auto loop = dyn_cast<LoopLikeOpInterface>(container))
    if (auto iv = loop.getSingleInductionVar())
      inductionVar = *iv;
  auto readsCarried = [&](Operation *op) {
    return isCyclic && llvm::any_of(op->getOperands(), [&](Value v) {
             auto barg = dyn_cast<BlockArgument>(v);
             return barg && barg.getOwner()->getParentOp() == container &&
                    v != inductionVar;
           });
  };
  auto occOf = [](const OperatorChar &c) {
    return c.pipelined ? 1u : std::max(1u, c.timing.latency);
  };
  auto shapeOf = [](SharedClassVar &cls, Operation *op, const OperatorChar &c) {
    cls.unitPrice = c.price;
    cls.ports = op->getNumOperands();
    for (Type t : op->getOperandTypes())
      if (t.isIntOrFloat())
        cls.width = std::max<int64_t>(cls.width, t.getIntOrFloatBitWidth());
  };
  // Membership, keyed and sorted like `populateOperatorAllocation` keys its
  // classes, so two compiles declare the same model.
  std::map<std::string, SharedClassVar> byIdentity;
  for (Operation *op : prob.getOperations()) {
    if (auto it = choiceOf.find(op); it != choiceOf.end()) {
      bool carried = readsCarried(op);
      for (auto [m, cand] : llvm::enumerate(choices[it->second].cands)) {
        // A comb candidate builds no allocatable instance, so it joins no
        // class and is priced through the selection term alone.
        if (cand.identity.comb)
          continue;
        SharedClassVar &cls = byIdentity[cand.identity.key()];
        shapeOf(cls, op, cand);
        cls.conds.push_back(
            {it->second, static_cast<unsigned>(m), occOf(cand)});
        cls.carried += carried;
      }
      continue;
    }
    if (isSyncSubKernelCall(op) || asMemAccess(op))
      continue;
    OperatorChar c = lib.lookup(op);
    if (!c.identity.realized() || c.identity.comb)
      continue;
    unsigned occ = occOf(c);
    if (isCyclic && occ > 1)
      continue;
    SharedClassVar &cls = byIdentity[c.identity.key()];
    shapeOf(cls, op, c);
    cls.statics.push_back({op, occ, picos(c.timing.inDelay)});
    cls.carried += readsCarried(op);
  }

  for (auto &[key, cls] : byIdentity) {
    if (cls.statics.size() + cls.conds.size() < 2)
      continue;
    cls.rsrc = prob.getOrInsertResourceType(key);
    cls.onProblem = prob.getAllocatable(cls.rsrc).has_value();
    for (const SharedClassVar::Cond &cond : cls.conds)
      shared.covered.insert({cond.choice, cond.cand});
    shared.owned.insert(cls.rsrc);
    shared.classes.push_back(std::move(cls));
  }
  return shared;
}

/// The operations a class holds or could hold, static and conditional alike:
/// what a modulo model gives a congruence slot beyond what already contends.
DenseSet<Operation *> sharedMemberOps(SharedClasses &shared,
                                      ArrayRef<SelectionChoice> choices) {
  DenseSet<Operation *> members;
  for (SharedClassVar &cls : shared.classes) {
    for (auto [op, occ, in] : cls.statics)
      members.insert(op);
    for (const SharedClassVar::Cond &cond : cls.conds)
      members.insert(choices[cond.choice].op);
  }
  return members;
}

/// Declare each collected class's decision variables on one model: the count
/// and the joined membership, the price and select cone read off (members,
/// units) tables through one flattened element constraint each, and the cone
/// held against every member's sub-cycle start. Capacity is the caller's:
/// what fits `units` instances differs between a straight line and a modulo
/// table.
void addSharedClassVars(CpModelBuilder &model, const OperatorLibrary &lib,
                        SharedClasses &shared,
                        ArrayRef<SelectionChoice> choices,
                        const SelectionVars &sels,
                        DenseMap<Operation *, IntVar> &inCycle,
                        float cycleTime) {
  int64_t period = picos(cycleTime);
  for (SharedClassVar &cls : shared.classes) {
    auto ceiling = static_cast<int64_t>(cls.statics.size() + cls.conds.size());
    auto staticCount = static_cast<int64_t>(cls.statics.size());
    cls.units = model.NewIntVar(
        operations_research::Domain(staticCount ? 1 : 0, ceiling));
    cls.members =
        model.NewIntVar(operations_research::Domain(staticCount, ceiling));
    LinearExpr joined(staticCount);
    for (const SharedClassVar::Cond &cond : cls.conds)
      joined += sels.sel[cond.choice][cond.cand];
    model.AddEquality(cls.members, joined);
    model.AddLessOrEqual(cls.units, cls.members);

    // The (members, units) tables, flattened onto one index.
    std::vector<int64_t> prices((ceiling + 1) * (ceiling + 1));
    std::vector<int64_t> cones(prices.size());
    for (int64_t k = 0; k <= ceiling; ++k)
      for (int64_t n = 0; n <= ceiling; ++n) {
        prices[k * (ceiling + 1) + n] = sharedPrice(lib, cls, k, n);
        cones[k * (ceiling + 1) + n] = picos(sharedConeNs(lib, cls, k, n));
      }
    IntVar idx = model.NewIntVar(operations_research::Domain(
        0, static_cast<int64_t>(prices.size()) - 1));
    model.AddEquality(idx,
                      LinearExpr::Term(cls.members, ceiling + 1) + cls.units);
    cls.price = model.NewIntVar(
        operations_research::Domain(0, *llvm::max_element(prices)));
    model.AddElement(idx, prices, cls.price);
    if (int64_t top = *llvm::max_element(cones)) {
      cls.headroom = model.NewIntVar(operations_research::Domain(0, top));
      model.AddElement(idx, cones, *cls.headroom);
      for (auto [op, occ, in] : cls.statics)
        model.AddLessOrEqual(inCycle.at(op) + *cls.headroom, period - in);
      for (const SharedClassVar::Cond &cond : cls.conds) {
        const OperatorChar &cand = choices[cond.choice].cands[cond.cand];
        model
            .AddLessOrEqual(inCycle.at(choices[cond.choice].op) + *cls.headroom,
                            period - picos(cand.timing.inDelay))
            .OnlyEnforceIf(sels.sel[cond.choice][cond.cand]);
      }
    }

    // A feasible hint alongside the start and selection hints: the members
    // the hinted selection joins, each its own instance (a cone of nothing).
    int64_t hintMembers = staticCount;
    for (const SharedClassVar::Cond &cond : cls.conds)
      hintMembers += cond.cand == choices[cond.choice].preferred;
    model.AddHint(cls.units, hintMembers);
  }
}

/// Build the shared classes of an acyclic model: the collected membership,
/// its variables, and the capacity as a cumulative whose demands are a fixed
/// interval per static member and an optional one per conditional member.
///
/// Only in allocation mode (the caller gates): with the trivial binding the
/// emitter builds one unit per operation, so a fold priced here would never
/// be built.
SharedClasses
addSharedClasses(CpModelBuilder &model, ChainingSharedOperatorsProblem &prob,
                 const OperatorLibrary &lib, ArrayRef<SelectionChoice> choices,
                 const SelectionVars &sels,
                 DenseMap<Operation *, IntVar> &startVars,
                 DenseMap<Operation *, IntVar> &inCycle, float cycleTime) {
  SharedClasses shared = collectSharedClasses(prob, lib, choices);
  addSharedClassVars(model, lib, shared, choices, sels, inCycle, cycleTime);
  for (SharedClassVar &cls : shared.classes) {
    CumulativeConstraint cum = model.AddCumulative(cls.units);
    for (auto [op, occ, in] : cls.statics)
      cum.AddDemand(model.NewFixedSizeIntervalVar(startVars.at(op), occ), 1);
    for (const SharedClassVar::Cond &cond : cls.conds)
      cum.AddDemand(model.NewOptionalFixedSizeIntervalVar(
                        startVars.at(choices[cond.choice].op), cond.occ,
                        sels.sel[cond.choice][cond.cand]),
                    1);
  }
  return shared;
}

/// The period as a model constraint: one sub-cycle start time `z` per
/// operation, in picoseconds from the start of its cycle, matching what
/// `computeStartTimesInCycle` computes afterwards.
/// `z(v) <= P - inDelay(v)`, and where a def-use producer u ends in the cycle v
/// starts, `z(v) >= (lat(u) == 0 ? z(u) : 0) + outDelay(u)`.
///
/// Two callers. Without \p sels, redundant against the break edges for
/// feasibility, stated only so `addAllocationHeadroom` can hold a select cone
/// against sub-cycle slack. With \p sels it is the model's period statement:
/// the break edges hold only for the rows the library picked, so a model that
/// re-decides rows drops them and every delay here follows the row decided,
/// one conditional bound per candidate.
///
/// Precedence already forces `t_v - t_u >= lat(u)`, so gating on the `<=` half
/// alone (via `sameCycle`) detects "ends in the same cycle".
///
/// Only def-use edges carry a combinational path; an auxiliary edge (memory
/// order, stream order, loop-carried recurrence) always passes through a port
/// or register.
///
/// Returns the per-operation sub-cycle variables, for constraints stated on
/// top of the system (`addAllocationHeadroom`).
///
/// \p hintSchedule hints every variable created here off the schedule already
/// on \p prob. A partial hint on a model this size never completes.
template <class ProblemT>
DenseMap<Operation *, IntVar>
addSubCycleTimes(CpModelBuilder &model, ProblemT &prob,
                 DenseMap<Operation *, IntVar> &startVars, float cycleTime,
                 float regFloor, const SelectionVars *sels = nullptr,
                 bool hintSchedule = false) {
  int64_t period = picos(cycleTime);
  // Nothing in a cycle starts before its operands leave a register, so the
  // fabric floor is every `z`'s lower bound. A chain from a registered producer
  // then costs `max(floor, that producer's outgoing delay)`.
  int64_t floor = picos(regFloor);
  auto zHint = [&](Operation *op, int64_t hi) {
    return std::clamp(picos(*prob.getStartTimeInCycle(op)), floor, hi);
  };
  DenseMap<Operation *, IntVar> inCycle;
  for (Operation *op : prob.getOperations()) {
    if (std::optional<unsigned> i = sels ? sels->of(op) : std::nullopt) {
      // The op's own need follows the row decided: the domain takes the least
      // candidate, and the decided one is enforced alongside. A candidate the
      // picosecond grid leaves no cycle room for is pruned by that constraint
      // rather than asserted, the float fit test having admitted it.
      SmallVector<int64_t> ins;
      for (const OperatorChar &c : sels->choices[*i].cands)
        ins.push_back(picos(c.timing.inDelay));
      IntVar z = model.NewIntVar(
          operations_research::Domain(floor, period - *llvm::min_element(ins)));
      model.AddLessOrEqual(z + LinearExpr::WeightedSum(sels->sel[*i], ins),
                           period);
      if (hintSchedule)
        model.AddHint(z, zHint(op, period - *llvm::min_element(ins)));
      inCycle.try_emplace(op, z);
      continue;
    }
    int64_t in = picos(*prob.getIncomingDelay(*prob.getLinkedOperatorType(op)));
    assert(in + floor <= period &&
           "an operator whose own delay exceeds the period cannot reach a "
           "solve: `runSDCScheduler` derates the period to fit every row");
    IntVar z = model.NewIntVar(operations_research::Domain(floor, period - in));
    if (hintSchedule)
      model.AddHint(z, zHint(op, period - in));
    inCycle.try_emplace(op, z);
  }

  for (Operation *op : prob.getOperations())
    for (auto &dep : prob.getDependences(op)) {
      if (!dep.isDefUse())
        continue;
      if constexpr (std::is_base_of_v<CyclicProblem, ProblemT>)
        assert(prob.getDistance(dep).value_or(0) == 0 &&
               "a distance rides an AUXILIARY edge here (`ProblemBuilder` "
               "inserts every carried dependence as one), so a def-use edge is "
               "always intra-iteration and its endpoints share a cycle");
      Operation *src = dep.getSource();
      Problem::OperatorType srcOpr = *prob.getLinkedOperatorType(src);
      int64_t lat = *prob.getLatency(srcOpr);
      std::optional<unsigned> si = sels ? sels->of(src) : std::nullopt;
      LinearExpr srcLat = si ? sels->latency(*si) : LinearExpr(lat);
      LinearExpr separation = startVars.at(op) - startVars.at(src);
      BoolVar sameCycle = model.NewBoolVar();
      model.AddLessOrEqual(separation, srcLat).OnlyEnforceIf(sameCycle);
      model.AddGreaterOrEqual(separation, srcLat + 1)
          .OnlyEnforceIf(sameCycle.Not());
      if (hintSchedule)
        model.AddHint(sameCycle,
                      static_cast<int64_t>(*prob.getStartTime(op)) -
                              static_cast<int64_t>(*prob.getStartTime(src)) <=
                          lat);
      // A multi-cycle producer contributes only its outgoing delay: its last
      // register stage is what the cycle starts from.
      if (si) {
        for (auto [m, cand] : llvm::enumerate(sels->choices[*si].cands)) {
          int64_t out = picos(cand.timing.outDelay);
          LinearExpr ready = cand.timing.latency == 0 ? inCycle.at(src) + out
                                                      : LinearExpr(out);
          model.AddGreaterOrEqual(inCycle.at(op), ready)
              .OnlyEnforceIf({sameCycle, sels->sel[*si][m]});
        }
        continue;
      }
      int64_t out = picos(*prob.getOutgoingDelay(srcOpr));
      LinearExpr ready = lat == 0 ? inCycle.at(src) + out : LinearExpr(out);
      model.AddGreaterOrEqual(inCycle.at(op), ready).OnlyEnforceIf(sameCycle);
    }
  return inCycle;
}

/// State \p breaks on the model: a chain-breaking edge widens an existing
/// precedence by one cycle.
template <class ProblemT>
void addChainBreaks(CpModelBuilder &model, ProblemT &prob,
                    DenseMap<Operation *, IntVar> &startVars,
                    ArrayRef<Problem::Dependence> breaks) {
  for (const Problem::Dependence &dep : breaks)
    model.AddLessOrEqual(startVars.at(dep.getSource()) +
                             prob.latencyOf(dep.getSource()) + 1,
                         startVars.at(dep.getDestination()));
}

/// Every operation's inputs settle within the period.
[[maybe_unused]] bool chainsFitCycleTime(ChainingProblem &prob,
                                         float cycleTime) {
  // Slop of one picosecond, the model's own resolution, to absorb float error.
  constexpr float kSlop = 1e-3f;
  for (Operation *op : prob.getOperations()) {
    float in = *prob.getIncomingDelay(*prob.getLinkedOperatorType(op));
    if (*prob.getStartTimeInCycle(op) + in > cycleTime + kSlop)
      return false;
  }
  return true;
}

/// Derive the solved schedule's sub-cycle start times and check the period.
/// `ChainingProblem` does not carry the period itself, so this is the only
/// place that verifies chains fit it.
LogicalResult finishSchedule(ChainingProblem &prob, float cycleTime,
                             float regFloor) {
  if (failed(mlir::allo::computeStartTimesInCycle(prob, regFloor)))
    return failure();
  assert(chainsFitCycleTime(prob, cycleTime) &&
         "a combinational chain crosses more than one clock period");
  return success();
}

/// The region's drain as a variable: the max of each term's commit cycle over
/// the same terms `drainOf` maxes over, stated as lower bounds only, which is
/// tight since the objective minimizes it. A `plusLatency` term commits its
/// definer's latency later, which \p latencyOf reads as the model states it
/// (a decided realization's is the one-hot sum).
///
/// \p bound caps it at an incumbent's, so the solver only searches schedules
/// that would beat it; an INFEASIBLE result then means "nothing beats the
/// incumbent" rather than "the interval is impossible".
IntVar drainVariable(CpModelBuilder &model,
                     DenseMap<Operation *, IntVar> &startVars,
                     ArrayRef<DrainTerm> terms, int64_t horizon,
                     std::optional<int64_t> bound,
                     llvm::function_ref<LinearExpr(Operation *)> latencyOf,
                     std::optional<int64_t> hint = std::nullopt) {
  assert((!bound || *bound >= 0) && "incumbent cut before building the model");
  IntVar drain = model.NewIntVar(operations_research::Domain(
      0, bound ? std::min(*bound, horizon) : horizon));
  if (hint)
    model.AddHint(drain, *hint);
  for (const DrainTerm &term : terms) {
    LinearExpr commit = startVars.at(term.op) + term.offset;
    if (term.plusLatency)
      commit += latencyOf(term.op);
    model.AddLessOrEqual(commit, drain);
  }
  return drain;
}

/// One allocatable resource in the model: the unit count to decide, and what
/// building that many of it costs.
struct AllocationVar {
  Problem::ResourceType rsrc;
  IntVar units;
  /// Priced off `units` through the resource's own table, so the plateaus in
  /// what a fold saves and what it grows are in the model exactly.
  IntVar price;
  int64_t maxPrice = 0;
  /// The fullest instance's select-cone delay in picoseconds, `headroomNs`
  /// read at `units`; absent where no count this resource can take builds a
  /// select.
  std::optional<IntVar> headroom;
};

/// The tightest count of \p rsrc the schedule currently on \p prob admits: the
/// busiest-slot demand, opened until the select cone fits the sub-cycle slack
/// that schedule leaves the resource's operations. This count paired with those
/// start times satisfies the headroom constraint, so it is a feasible point as
/// a hint.
template <class ProblemT>
unsigned demandWithHeadroom(ProblemT &prob, Problem::ResourceType rsrc,
                            unsigned ii, float cycleTime) {
  auto unit = prob.getAllocatable(rsrc);
  double slack = cycleTime;
  for (Operation *op : prob.getOperations())
    if (prob.usesResource(op, rsrc))
      slack = std::min(
          slack, double(cycleTime) - *prob.getStartTimeInCycle(op) -
                     *prob.getIncomingDelay(*prob.getLinkedOperatorType(op)));
  unsigned n = prob.demandFor(rsrc, ii);
  while (n < unit->ceiling && unit->headroomNs[n] > slack)
    ++n;
  return n;
}

/// Declare `N_r` for every allocatable resource: how many copies of one
/// operator this region builds, in `[1, ceiling]`. The caller states the
/// capacity constraint against it.
///
/// \p hint says the heuristic's start times are being hinted too, and then the
/// count hinted is the tightest one those start times admit with the select
/// cone charged.
///
/// \p owned are the rows a shared class already carries (`addSharedClasses`),
/// skipped so one model never prices a row twice.
template <class ProblemT>
SmallVector<AllocationVar>
allocationVars(CpModelBuilder &model, ProblemT &prob, unsigned ii, bool hint,
               float cycleTime,
               const DenseSet<Problem::ResourceType> *owned = nullptr) {
  SmallVector<AllocationVar> allocs;
  for (Problem::ResourceType rsrc : prob.getResourceTypes()) {
    auto unit = prob.getAllocatable(rsrc);
    if (!unit)
      continue;
    if (owned && owned->contains(rsrc))
      continue;
    assert(unit->ceiling > 0 && "an allocatable resource with no operation");
    IntVar n = model.NewIntVar(operations_research::Domain(1, unit->ceiling));
    if (hint)
      model.AddHint(n, demandWithHeadroom(prob, rsrc, ii, cycleTime));
    std::vector<int64_t> table(unit->price.begin(), unit->price.end());
    int64_t hi = *llvm::max_element(unit->price);
    IntVar price = model.NewIntVar(
        operations_research::Domain(*llvm::min_element(table), hi));
    model.AddElement(n, table, price);
    AllocationVar alloc{rsrc, n, price, hi, std::nullopt};
    std::vector<int64_t> cone;
    cone.reserve(unit->headroomNs.size());
    for (double ns : unit->headroomNs)
      cone.push_back(picos(ns));
    if (int64_t top = *llvm::max_element(cone)) {
      alloc.headroom = model.NewIntVar(operations_research::Domain(0, top));
      model.AddElement(n, cone, *alloc.headroom);
    }
    allocs.push_back(alloc);
  }
  return allocs;
}

/// Hold every operation of an allocatable operator to the period with the
/// select cone its decided count implies: `z + inDelay + headroom(N) <=
/// period`. A count then only shrinks where its multiplexer fits the slack the
/// same solve leaves, which is what lets a `planned` binding realize the
/// allocation as built. A combinational member's cone also reaches its
/// same-cycle consumers, so each is held to the period against its own
/// incoming delay.
///
/// \p sharedInCycle is the sub-cycle system a selection model already built,
/// reused so one model never carries two; null creates one on demand. The
/// break edges already keep the plain system satisfiable at any placement, so
/// without selection this tightens the model only by the headroom itself.
template <class ProblemT>
void addAllocationHeadroom(CpModelBuilder &model, ProblemT &prob,
                           DenseMap<Operation *, IntVar> &startVars,
                           ArrayRef<AllocationVar> allocs, float cycleTime,
                           float regFloor,
                           DenseMap<Operation *, IntVar> *sharedInCycle,
                           bool hintSchedule = false) {
  if (llvm::none_of(allocs, [](const AllocationVar &a) {
        return a.headroom.has_value();
      }))
    return;
  DenseMap<Operation *, IntVar> own;
  if (!sharedInCycle) {
    own = addSubCycleTimes(model, prob, startVars, cycleTime, regFloor,
                           /*sels=*/nullptr, hintSchedule);
    sharedInCycle = &own;
  }
  DenseMap<Operation *, IntVar> &inCycle = *sharedInCycle;
  int64_t period = picos(cycleTime);
  for (const AllocationVar &alloc : allocs) {
    if (!alloc.headroom)
      continue;
    for (Operation *op : prob.getOperations()) {
      if (!prob.usesResource(op, alloc.rsrc))
        continue;
      int64_t in =
          picos(*prob.getIncomingDelay(*prob.getLinkedOperatorType(op)));
      model.AddLessOrEqual(inCycle.at(op) + *alloc.headroom, period - in);
    }
  }
  // The consumer half: covering a combinational producer's same-cycle consumer
  // covers a non-unit sink (a store's port and address setup, a stream port)
  // that uses no allocatable operator of its own.
  for (Operation *op : prob.getOperations()) {
    llvm::SmallDenseSet<Operation *, 4> seen;
    for (auto &dep : prob.getDependences(op)) {
      if (!dep.isDefUse())
        continue;
      Operation *src = dep.getSource();
      if (prob.latencyOf(src) != 0 || !seen.insert(src).second)
        continue;
      std::optional<BoolVar> same;
      for (const AllocationVar &alloc : allocs) {
        if (!alloc.headroom || !prob.usesResource(src, alloc.rsrc) ||
            prob.usesResource(op, alloc.rsrc))
          continue;
        if (!same) {
          LinearExpr separation = startVars.at(op) - startVars.at(src);
          same = model.NewBoolVar();
          model.AddLessOrEqual(separation, 0).OnlyEnforceIf(*same);
          model.AddGreaterOrEqual(separation, 1).OnlyEnforceIf(same->Not());
          if (hintSchedule)
            model.AddHint(*same,
                          *prob.getStartTime(op) == *prob.getStartTime(src));
        }
        int64_t in =
            picos(*prob.getIncomingDelay(*prob.getLinkedOperatorType(op)));
        model.AddLessOrEqual(inCycle.at(op) + *alloc.headroom, period - in)
            .OnlyEnforceIf(*same);
      }
    }
  }
}

/// Add \p price at \p size to a weighted sum, for a price tabulated at every
/// value the size can take, charged relative to `price[0]`. A convex table is
/// its first slope on the size, plus at every change of slope that change
/// charged on how far the size runs past the point it changes at:
/// `max(size - b, 0)`; every variable this adds is determined by the size
/// through a propagator, and every weight is nonnegative. A non-convex table
/// cannot ride that form (its negative weights void the objective's LP bound):
/// it is charged at its lower convex envelope, one supporting line per hull
/// segment on a minimized cost variable. The staircase above the envelope is
/// not modeled; the register-lifetime repair re-prices the shipped schedule at
/// the real table. An element lookup is out: or-tools 9.15's presolve crashes
/// crushing hints through an expanded element
/// (`TryToReplaceVariableByItsEncoding`).
void addPiecewiseCost(CpModelBuilder &model, IntVar size,
                      ArrayRef<int64_t> price, SmallVectorImpl<IntVar> &vars,
                      SmallVectorImpl<int64_t> &weights) {
  auto hi = static_cast<int64_t>(price.size()) - 1;
  if (hi < 1)
    return;
  bool convex = true;
  for (int64_t d = 2; d <= hi && convex; ++d)
    convex = price[d] - price[d - 1] >= price[d - 1] - price[d - 2];
  if (convex) {
    int64_t slope = price[1] - price[0];
    vars.push_back(size);
    weights.push_back(slope);
    for (int64_t d = 2; d <= hi; ++d) {
      int64_t next = price[d] - price[d - 1];
      if (next == slope)
        continue;
      IntVar over = model.NewIntVar(operations_research::Domain(0, hi - d + 1));
      model.AddMaxEquality(over, {LinearExpr(size) - (d - 1), LinearExpr(0)});
      vars.push_back(over);
      weights.push_back(next - slope);
      slope = next;
    }
    return;
  }
  std::vector<int64_t> table;
  table.reserve(price.size());
  for (int64_t p : price)
    table.push_back(p - price[0]);
  IntVar cost = model.NewIntVar(operations_research::Domain(
      *llvm::min_element(table), *llvm::max_element(table)));
  vars.push_back(cost);
  weights.push_back(1);
  SmallVector<int64_t> hull;
  for (int64_t d = 0; d <= hi; ++d) {
    while (hull.size() >= 2) {
      int64_t a = hull[hull.size() - 2], b = hull.back();
      if ((d - a) * (table[b] - table[a]) >= (b - a) * (table[d] - table[a]))
        hull.pop_back();
      else
        break;
    }
    hull.push_back(d);
  }
  for (size_t i = 0; i + 1 < hull.size(); ++i) {
    int64_t d1 = hull[i], d2 = hull[i + 1];
    int64_t p1 = table[d1], p2 = table[d2];
    model.AddGreaterOrEqual(LinearExpr::Term(cost, d2 - d1) -
                                LinearExpr::Term(size, p2 - p1),
                            (d2 - d1) * p1 - (p2 - p1) * d1);
  }
}

/// The region's area, every term of it in what the device spends, as one
/// expression the caller weights into an objective: the delay chain each value
/// carried across slack costs (`RegisterTerm`), the activation pulse chain,
/// and the table above per allocatable operator. A chain is not `width *
/// depth` flip-flops: neither run holds a reset, so past a measured depth the
/// synthesizer extracts a shift register and the cost stops rising with the
/// depth outside the steps a new site adds.
///
/// The pulse chain is shared: taps at every start offset ride one chain as
/// deep as the deepest (`EmitContext::delayValid`), so it is charged on the
/// maximum start, not the sum. A guard family's own chain rides other source
/// signals this approximation does not see.
///
/// A decided realization joins the area on both sides: its instance costs its
/// row's price, and a register chain from it starts its decided latency after
/// its issue, both read through \p sels / \p latencyOf. A candidate a shared
/// class prices (\p shared) is skipped here, its cost riding the class's
/// (members, units) table instead.
///
/// \p hintFrom, non-null, hints every variable created here off that problem's
/// solved schedule, completing the start/allocation/selection hints an
/// area-first solve carries. A partial hint the solver fails to complete
/// leaves a heavy model with no incumbent at all.
///
/// \p structuralOut, non-null, receives the structural part alone (instances,
/// their selects, the decided rows; no chains, no pulse), which an area solve
/// bootstraps on: the chain terms relax only to their convex envelope, so the
/// full expression still searches worse than structure alone.
LinearExpr areaTerms(CpModelBuilder &model, ArrayRef<IntVar> starts,
                     const SpanObjective &span,
                     DenseMap<Operation *, IntVar> &startVars,
                     ArrayRef<AllocationVar> allocs, int64_t ii,
                     int64_t horizon,
                     llvm::function_ref<LinearExpr(Operation *)> latencyOf,
                     const SelectionVars &sels, const SharedClasses &shared,
                     OccupancyProblem *hintFrom = nullptr,
                     LinearExpr *structuralOut = nullptr) {
  SmallVector<IntVar> vars;
  SmallVector<int64_t> weights;
  // At II > 1 the emitter folds every chain onto the region's phase, holding
  // `depth` taps in `ceil(depth / ii)` registers (`EmitContext::foldedChain`),
  // so the variable below counts registers built rather than cycles spanned and
  // the table is indexed by that count.
  int64_t fold = std::max<int64_t>(ii, 1);
  int64_t stages = (horizon + fold - 1) / fold;
  // One chain price table per width: a region carries many values of the same
  // type, and tabulating the device's cost is the expensive half.
  DenseMap<int64_t, SmallVector<int64_t>> chainPrices;
  for (const RegisterTerm &term : span.regs) {
    auto [entry, isNew] = chainPrices.try_emplace(term.width);
    if (isNew)
      for (int64_t n = 0; n <= stages; ++n)
        entry->second.push_back(span.device.chainPrice(n, term.width));
    ArrayRef<int64_t> table = entry->second;
    IntVar built = model.NewIntVar(operations_research::Domain(0, stages));
    IntVar def = startVars.at(term.def);
    // Only bounded from below. A chain price is nondecreasing in its length, so
    // a minimizing solve lands `built` on the fold of the deepest read.
    for (auto [reader, distance] : term.reads)
      model.AddLessOrEqual(startVars.at(reader) + distance * ii,
                           def + latencyOf(term.def) +
                               LinearExpr::Term(built, fold));
    if (hintFrom) {
      int64_t end = static_cast<int64_t>(*hintFrom->getStartTime(term.def)) +
                    hintFrom->latencyOf(term.def);
      int64_t depth = 0;
      for (auto [reader, distance] : term.reads)
        depth = std::max(depth,
                         static_cast<int64_t>(*hintFrom->getStartTime(reader)) +
                             distance * ii - end);
      model.AddHint(built, (depth + fold - 1) / fold);
    }
    addPiecewiseCost(model, built, table, vars, weights);
  }
  LinearExpr structural;
  for (const AllocationVar &alloc : allocs)
    structural += alloc.price;
  for (const SharedClassVar &cls : shared.classes)
    structural += cls.price;
  for (auto [i, choice] : llvm::enumerate(sels.choices))
    for (auto [m, cand] : llvm::enumerate(choice.cands))
      if (cand.price && !shared.covered.contains({static_cast<unsigned>(i),
                                                  static_cast<unsigned>(m)}))
        structural += LinearExpr::Term(sels.sel[i][m], cand.price);
  LinearExpr area = LinearExpr::WeightedSum(vars, weights) + structural;
  if (structuralOut)
    *structuralOut = structural;
  if (int64_t pulse = span.device.pulsePrice(); pulse && !starts.empty()) {
    IntVar deepest = model.NewIntVar(operations_research::Domain(0, horizon));
    for (IntVar start : starts)
      model.AddLessOrEqual(start, deepest);
    if (hintFrom) {
      int64_t top = 0;
      for (Operation *op : hintFrom->getOperations())
        top = std::max(top, static_cast<int64_t>(*hintFrom->getStartTime(op)));
      model.AddHint(deepest, top);
    }
    area += LinearExpr::Term(deepest, pulse);
  }
  return area;
}

/// Whether \p response carries a schedule to read.
bool solved(const CpSolverResponse &response) {
  return response.status() == CpSolverStatus::OPTIMAL ||
         response.status() == CpSolverStatus::FEASIBLE;
}

/// The unit counts one solve decided. Held apart from the problem because the
/// cyclic search runs many solves and only the adopted one's counts stand.
using Allocated = SmallVector<std::pair<Problem::ResourceType, unsigned>>;

Allocated readAllocation(const CpSolverResponse &response,
                         ArrayRef<AllocationVar> allocs) {
  Allocated decided;
  for (const AllocationVar &alloc : allocs)
    decided.push_back({alloc.rsrc, static_cast<unsigned>(SolutionIntegerValue(
                                       response, alloc.units))});
  return decided;
}

/// The instance count each shared class settled on in \p response, aligned
/// with `shared.classes`. Read apart from the writeback because the cyclic
/// search runs many solves and only the adopted one's counts stand.
SmallVector<unsigned> readSharedUnits(const CpSolverResponse &response,
                                      SharedClasses &shared) {
  SmallVector<unsigned> units;
  for (SharedClassVar &cls : shared.classes)
    units.push_back(
        static_cast<unsigned>(SolutionIntegerValue(response, cls.units)));
  return units;
}

/// What the decided classes cost at the membership \p chosen implies and the
/// counts \p classUnits record: the host-side mirror of the model's price
/// element, for the cyclic search to compare intervals with.
int64_t sharedAreaOf(const OperatorLibrary &lib, SharedClasses &shared,
                     ArrayRef<SelectionChoice> choices,
                     ArrayRef<unsigned> chosen, ArrayRef<unsigned> classUnits) {
  int64_t area = 0;
  for (auto [cls, units] : llvm::zip(shared.classes, classUnits)) {
    auto k = static_cast<int64_t>(cls.statics.size());
    for (const SharedClassVar::Cond &cond : cls.conds)
      k += chosen[cond.choice] == cond.cand;
    area += sharedPrice(lib, cls, k, units);
  }
  return area;
}

/// Write the adopted membership of every shared class back onto the problem:
/// link each decided member to the class resource, set the problem-side
/// allocatable view to the tables at the decided member count, and append the
/// decided instance counts (\p classUnits, aligned with the classes) for
/// `applyAllocation` to realize. A class fewer than two members joined
/// dissolves: each operation keeps its own instance.
void applySharedClasses(OccupancyProblem &prob, const OperatorLibrary &lib,
                        SharedClasses &shared,
                        ArrayRef<SelectionChoice> choices,
                        ArrayRef<unsigned> chosen,
                        ArrayRef<unsigned> classUnits, Allocated &decided) {
  auto link = [&](Operation *op, Problem::ResourceType rsrc, unsigned occ) {
    SmallVector<Problem::ResourceType> units;
    if (auto linked = prob.getLinkedResourceTypes(op))
      units.assign(linked->begin(), linked->end());
    units.push_back(rsrc);
    prob.setLinkedResourceTypes(op, units);
    prob.setResourceCycles(op, occ);
  };
  for (auto [cls, decidedUnits] : llvm::zip(shared.classes, classUnits)) {
    SmallVector<std::pair<Operation *, unsigned>> joined;
    for (const SharedClassVar::Cond &cond : cls.conds)
      if (chosen[cond.choice] == cond.cand)
        joined.push_back({choices[cond.choice].op, cond.occ});
    auto k = static_cast<int64_t>(cls.statics.size() + joined.size());
    if (k < 2)
      continue;
    auto units = static_cast<int64_t>(decidedUnits);
    assert(units >= 1 && units <= k &&
           "the model bounds the count by the membership it decided");
    if (!cls.onProblem)
      for (auto [op, occ, in] : cls.statics)
        link(op, cls.rsrc, occ);
    for (auto [op, occ] : joined)
      link(op, cls.rsrc, occ);
    OccupancyProblem::AllocatableUnit unit;
    unit.ceiling = static_cast<unsigned>(k);
    for (int64_t n = 0; n <= k; ++n) {
      unit.price.push_back(sharedPrice(lib, cls, k, n));
      unit.headroomNs.push_back(sharedConeNs(lib, cls, k, n));
    }
    prob.setAllocatable(cls.rsrc, unit);
    decided.push_back({cls.rsrc, static_cast<unsigned>(units)});
  }
}

/// Point the model's hints at the schedule \p from found, for the area solve
/// that runs on the same model next.
void rehintFrom(CpModelBuilder &model, ArrayRef<Operation *> ops,
                DenseMap<Operation *, IntVar> &startVars,
                ArrayRef<AllocationVar> allocs, const SelectionVars &sels,
                const SharedClasses &shared, const CpSolverResponse &from) {
  model.ClearHints();
  for (Operation *op : ops)
    model.AddHint(startVars.at(op),
                  SolutionIntegerValue(from, startVars.at(op)));
  for (const AllocationVar &alloc : allocs)
    model.AddHint(alloc.units, SolutionIntegerValue(from, alloc.units));
  for (const SmallVector<BoolVar, 2> &row : sels.sel)
    for (const BoolVar &var : row)
      model.AddHint(var, SolutionBooleanValue(from, var));
  for (const SharedClassVar &cls : shared.classes)
    model.AddHint(cls.units, SolutionIntegerValue(from, cls.units));
}

/// Point the model's hints at every variable of \p from, verbatim: the
/// complete warm start the next solve on the same builder resumes from.
/// Valid only while the builder has grown no variables since \p from was
/// solved (added constraints are fine).
void rehintAll(CpModelBuilder &model, const CpSolverResponse &from) {
  assert(model.Proto().variables_size() == from.solution_size() &&
         "a variable added after the solve has no value to hint");
  auto *hint = model.MutableProto()->mutable_solution_hint();
  hint->Clear();
  for (int i = 0; i < from.solution_size(); ++i) {
    hint->add_vars(i);
    hint->add_values(from.solution(i));
  }
}

/// Pin the structure \p from settled, counts and rows, so only placement
/// floats in the solves that follow. Pinning the aggregate alone would let
/// them spend register-chain savings on extra units, which the model prices as
/// a wash and the fabric does not.
void pinStructure(CpModelBuilder &model, const CpSolverResponse &from,
                  ArrayRef<AllocationVar> allocs, const SelectionVars &sels,
                  const SharedClasses &shared) {
  for (const AllocationVar &alloc : allocs)
    model.AddEquality(alloc.units, SolutionIntegerValue(from, alloc.units));
  for (const SmallVector<BoolVar, 2> &row : sels.sel)
    for (const BoolVar &var : row)
      model.AddEquality(LinearExpr(var),
                        SolutionBooleanValue(from, var) ? 1 : 0);
  for (const SharedClassVar &cls : shared.classes)
    model.AddEquality(cls.units, SolutionIntegerValue(from, cls.units));
}

/// Share of the budget the cycles order's area tie-break gets where a span
/// composes: the fold re-solve at the winning interval redoes the area on a
/// fresh budget, so an uncapped tie-break would spend the remainder without
/// proving anything the fold does not.
constexpr double kAreaTieBreakShare = 0.3;

/// The area fold minimizes in deterministic-time slices and releases the rest
/// of its budget once the area has stopped improving. The bound the solver
/// proves on this objective can sit at the chain terms' convex envelope, well
/// under the true staircase, so the stop reads the incumbent curve
/// instead: a slice improving the modeled area by less than `kFoldPlateauEps`
/// is a stall, and `kFoldPatience` consecutive stalls end the fold. A region
/// still paying for its budget runs the whole of it; a plateaued one does not.
constexpr double kFoldChunkShare = 0.25;
constexpr double kFoldPlateauEps = 0.01;
constexpr unsigned kFoldPatience = 2;

/// The fold also stops once the solver's bound certifies the incumbent within
/// this fraction of the model's optimum; further slices could recover at most
/// that. The plateau above stays the fallback where the bound lags the
/// incumbent.
constexpr double kFoldGapEps = 0.01;

/// The structural bootstrap runs on a capped share of the budget: it is
/// hinted and structure-only, so it searches well, and an uncapped run can
/// spend the whole budget proving structure while the area solve it exists to
/// seed never runs at all.
constexpr double kBootstrapShare = 0.25;

/// A span solve that finds a schedule but not the proof earns one retry at
/// this multiple of the budget: an unproven span ships a possibly far-off
/// incumbent, so the retry buys latency exactly where the certificate failed.
/// The retry's incumbent converges well before its certificate (which stays
/// flaky under the worker portfolio), so the multiple is sized for the former.
/// The rest of that solve's pipeline then runs on the escalated budget.
constexpr double kSpanEscalation = 10.0;

/// What \p decided costs the device: every resource, at the price of the count
/// it settled on.
int64_t areaOf(OccupancyProblem &prob, const Allocated &decided) {
  int64_t area = 0;
  for (auto [rsrc, units] : decided)
    area += prob.getAllocatable(rsrc)->price[units];
  return area;
}

/// Write \p decided onto the problem and derive which instance each operation
/// runs on. \p ii is 0 for a straight-line region.
void applyAllocation(OccupancyProblem &prob, const Allocated &decided,
                     unsigned ii) {
  if (decided.empty())
    return;
  int64_t built = 0, ops = 0;
  for (auto [rsrc, units] : decided) {
    prob.setAllocation(rsrc, units);
    built += units;
    ops += prob.getAllocatable(rsrc)->ceiling;
  }
  // Counts alone are not buildable; this derives the per-operation instance.
  prob.assignUnits(ii);
  info(Stage::Sched, prob.getContainingOp())
      << "Allocated: " << ops << " operations onto " << built
      << " instances of " << decided.size() << " shared operator types";
}

/// Fall back to a first-fit shared allocation over the schedule already on the
/// problem, for a solve that decided none. Operations whose realization was
/// the solver's to decide are grouped at the library's own pick, which is what
/// an undecided solve realizes them as. Bins grow member by member in start
/// order; joining one holds the class's select cone at the grown size against
/// every member's own sub-cycle slack, so one packed operation keeps its own
/// instance instead of opening the whole class to its ceiling.
template <class ProblemT>
void applyFallbackAllocation(ProblemT &prob, const OperatorLibrary &lib,
                             bool allocate, unsigned ii, float cycleTime) {
  if (allocate)
    populateOperatorAllocation(prob, lib, AllocationScope::Selecting);
  int64_t built = 0, ops = 0;
  unsigned classes = 0;
  for (Problem::ResourceType rsrc : prob.getResourceTypes()) {
    auto unit = prob.getAllocatable(rsrc);
    if (!unit)
      continue;
    unsigned ceiling = unit->ceiling;
    // The cone the model itself would charge a fullest instance of f members:
    // the table entry at the largest count whose fullest instance holds f.
    auto cone = [&](unsigned f) {
      return f <= 1 ? 0.0 : unit->headroomNs[(ceiling - 1) / (f - 1)];
    };
    struct Bin {
      SmallVector<Operation *, 2> members;
      llvm::SmallDenseSet<unsigned, 4> slots; // congruence classes, cyclic
      unsigned freeAt = 0;                    // next free cycle, acyclic
      double minSlack = std::numeric_limits<double>::infinity();
    };
    SmallVector<Bin> bins;
    for (Operation *op : prob.usersOf(rsrc)) {
      double slack = double(cycleTime) - *prob.getStartTimeInCycle(op) -
                     *prob.getIncomingDelay(*prob.getLinkedOperatorType(op));
      unsigned start = *prob.getStartTime(op);
      unsigned occ = prob.getResourceCycles(op);
      auto free = [&](Bin &bin) {
        if (!ii)
          return bin.freeAt <= start;
        for (unsigned k = 0; k < occ; ++k)
          if (bin.slots.count((start + k) % ii))
            return false;
        return true;
      };
      Bin *dest = nullptr;
      for (Bin &bin : bins)
        if (cone(bin.members.size() + 1) <= std::min(bin.minSlack, slack) &&
            free(bin)) {
          dest = &bin;
          break;
        }
      if (!dest)
        dest = &bins.emplace_back();
      dest->members.push_back(op);
      dest->minSlack = std::min(dest->minSlack, slack);
      if (ii)
        for (unsigned k = 0; k < occ; ++k)
          dest->slots.insert((start + k) % ii);
      else
        dest->freeAt = start + occ;
    }
    prob.setAllocation(rsrc, bins.size());
    for (auto [k, bin] : llvm::enumerate(bins))
      for (Operation *m : bin.members)
        prob.setAssignedUnit(m, static_cast<unsigned>(k));
    built += bins.size();
    ops += ceiling;
    ++classes;
  }
  if (classes)
    info(Stage::Sched, prob.getContainingOp())
        << "Allocated: " << ops << " operations onto " << built
        << " instances of " << classes << " shared operator types";
}

/// Report a solve that produced nothing usable and leave the heuristic's
/// schedule in place. A `warn`: the compile is still correct, it just did not
/// get a better schedule.
void reportUnsolved(Problem &prob, const CpSolverResponse &response,
                    double budget) {
  assert(response.status() != CpSolverStatus::INFEASIBLE &&
         response.status() != CpSolverStatus::MODEL_INVALID &&
         "the heuristic's schedule satisfies this encoding, so the model is "
         "satisfiable by construction");
  warn(Stage::Sched, prob.getContainingOp())
      << "Exact scheduling gave up after " << llvm::format("%g", budget)
      << " deterministic time units (solver status "
      << CpSolverStatus_Name(response.status())
      << "); keeping the heuristic schedule";
}

/// Lower bound on the drain of any schedule of \p prob.
///
/// Two facts bound where an output can commit. One is its own longest path. The
/// other is resource contention: for any set S of operations that must all pass
/// one capped resource before the output commits,
///
/// ```
/// start(v) >= minHead(S) + ceil( sum demand(u) / limit ) - 1 + minTail(S, v)
/// ```
///
/// since every member of S issues between the earliest head in it and
/// `start(v)` less the shortest path onward, a window whose capacity has to
/// cover them all. The longest path is this with S a singleton, where the
/// middle term vanishes.
///
/// Only intra-iteration edges count, so the bound holds at every initiation
/// interval: within one iteration a window of length L touches `min(L, ii)`
/// congruence classes, each admitting `limit` units from that iteration, and
/// work above `ii * limit` is an interval `computeResMinII` already ruled out.
///
/// \p latencyOf must under-approximate every latency the model can decide
/// (the least candidate of a decided realization), or the floor cuts the
/// optimum.
///
/// \p opFloors, non-null, receives a per-operation start floor: the longest
/// path in for every operation, raised by the same threshold-set bound for
/// operations holding a capped unit. True lower bounds on any schedule, fed
/// to the model as variable domains.
template <typename ProblemT>
int64_t drainFloor(ProblemT &prob, ArrayRef<Problem::Dependence> breaks,
                   ArrayRef<DrainTerm> terms,
                   llvm::function_ref<int64_t(Operation *)> latencyOf,
                   DenseMap<Operation *, int64_t> *opFloors = nullptr) {
  auto offsetOf = [&](const DrainTerm &term) {
    return term.offset + (term.plusLatency ? latencyOf(term.op) : 0);
  };

  // The edges the model imposes, weighted as it weights them. A forwarded
  // store->load edge separates by zero (its shadow supplies the datum), so it
  // must not tighten this floor.
  struct FloorEdge {
    Operation *src, *dst;
    int64_t weight;
  };
  SmallVector<FloorEdge> edges;
  for (Operation *op : prob.getOperations())
    for (auto &dep : prob.getDependences(op)) {
      int64_t dist = 0;
      if constexpr (std::is_same_v<ProblemT, ChainingModuloProblem>)
        dist = prob.getDistance(dep).value_or(0);
      if (dist != 0)
        continue;
      int64_t weight = prob.separationOf(dep);
      edges.push_back({dep.getSource(), op, weight});
    }
  // A chain break is intra-iteration whichever problem this is.
  for (const auto &dep : breaks)
    edges.push_back({dep.getSource(), dep.getDestination(),
                     latencyOf(dep.getSource()) + 1});

  unsigned nOps = prob.getOperations().size();

  // Longest paths from the zero floor over the intra-iteration DAG, by
  // relaxation to a fixpoint.
  DenseMap<Operation *, int64_t> heads;
  for (Operation *op : prob.getOperations())
    heads[op] = 0;
  bool changed = true;
  for (unsigned round = 0; changed && round <= nOps; ++round) {
    changed = false;
    for (const FloorEdge &e : edges) {
      int64_t reach = heads[e.src] + e.weight;
      if (reach > heads[e.dst]) {
        heads[e.dst] = reach;
        changed = true;
      }
    }
  }
  assert(!changed && "a positive cycle at the floor's interval");

  int64_t bound = 0;
  for (const DrainTerm &term : terms)
    bound = std::max(bound, heads.lookup(term.op) + offsetOf(term));

  if (opFloors)
    for (Operation *op : prob.getOperations())
      if (int64_t head = heads.lookup(op))
        (*opFloors)[op] = head;

  SmallVector<std::pair<Problem::ResourceType, int64_t>> capped;
  for (Problem::ResourceType rsrc : prob.getResourceTypes())
    if (unsigned limit = prob.getLimit(rsrc).value_or(0))
      capped.push_back({rsrc, limit});
  if (capped.empty())
    return bound;

  struct Contender {
    int64_t head, tail, demand;
  };
  // The strongest bound over the threshold sets of a group. At fixed
  // thresholds widening a set only adds work, so the maximum lies on one of
  // them and the subsets themselves need no enumerating.
  auto strongest = [](SmallVectorImpl<Contender> &group, int64_t limit) {
    llvm::sort(group, [](const Contender &a, const Contender &b) {
      return a.tail > b.tail;
    });
    int64_t best = 0;
    for (const Contender &first : group) {
      int64_t work = 0;
      for (const Contender &c : group) {
        if (c.head < first.head)
          continue;
        work += c.demand;
        best = std::max(best,
                        first.head + (work + limit - 1) / limit - 1 + c.tail);
      }
    }
    return best;
  };

  // Longest path on to \p sink by the reverse relaxation, absent for an
  // operation that cannot reach it. A carried path can weigh negative, which
  // stays a valid (if weak) tail.
  auto tailsTo = [&](Operation *sink) {
    DenseMap<Operation *, int64_t> tails;
    tails[sink] = 0;
    bool grew = true;
    for (unsigned round = 0; grew && round <= nOps; ++round) {
      grew = false;
      for (const FloorEdge &e : edges) {
        auto to = tails.find(e.dst);
        if (to == tails.end())
          continue;
        int64_t reach = e.weight + to->second;
        auto from = tails.find(e.src);
        if (from == tails.end()) {
          tails[e.src] = reach;
          grew = true;
        } else if (reach > from->second) {
          from->second = reach;
          grew = true;
        }
      }
    }
    assert(!grew && "a positive cycle at the floor's interval");
    return tails;
  };

  DenseSet<Operation *> feeding;
  // The capped-`rsrc` contenders reaching a sink along \p tails, each with its
  // head, tail and demand. Marks the members as feeding when asked: the
  // per-output pass does, the per-op floor pass does not.
  auto groupFor = [&](DenseMap<Operation *, int64_t> &tails,
                      Problem::ResourceType rsrc, bool markFeeding) {
    SmallVector<Contender> group;
    for (Operation *op : prob.getOperations()) {
      if (!prob.usesResource(op, rsrc))
        continue;
      auto it = tails.find(op);
      if (it == tails.end())
        continue;
      if (markFeeding)
        feeding.insert(op);
      group.push_back({heads.lookup(op), it->second,
                       static_cast<int64_t>(prob.getResourceDemand(op))});
    }
    return group;
  };

  for (const DrainTerm &term : terms) {
    DenseMap<Operation *, int64_t> tails = tailsTo(term.op);
    for (auto [rsrc, limit] : capped) {
      SmallVector<Contender> group =
          groupFor(tails, rsrc, /*markFeeding=*/true);
      if (!group.empty())
        bound = std::max(bound, strongest(group, limit) + offsetOf(term));
    }
  }

  // Every operation feeding any output issues by the drain whatever path it
  // takes there, which bounds the drain where no single output orders them all.
  for (auto [rsrc, limit] : capped) {
    SmallVector<Contender> group;
    for (Operation *op : feeding)
      if (prob.usesResource(op, rsrc))
        group.push_back({heads.lookup(op), 0,
                         static_cast<int64_t>(prob.getResourceDemand(op))});
    if (!group.empty())
      bound = std::max(bound, strongest(group, limit));
  }

  // The same threshold-set bound per contending operation: a set that must
  // all pass a capped resource before \p v starts pushes v's start the way it
  // pushes an output's commit, with v itself in the set at tail zero.
  if (opFloors)
    for (Operation *v : prob.getOperations()) {
      if (llvm::none_of(capped,
                        [&](auto &c) { return prob.usesResource(v, c.first); }))
        continue;
      DenseMap<Operation *, int64_t> tails = tailsTo(v);
      int64_t &floor = (*opFloors)[v];
      for (auto [rsrc, limit] : capped) {
        SmallVector<Contender> group =
            groupFor(tails, rsrc, /*markFeeding=*/false);
        if (!group.empty())
          floor = std::max(floor, strongest(group, limit));
      }
    }
  return bound;
}

} // namespace

//===----------------------------------------------------------------------===//
// The acyclic solve.
//===----------------------------------------------------------------------===//

/// Refines the heuristic's acyclic schedule to the CP-SAT optimum.
///
/// The heuristic runs first as a feasibility check and a warm-start hint: its
/// resource-free LP is the only thing that can fail, so a failure here is
/// fatal.
///
/// A straight-line region runs once, so its whole cost is its drain,
/// upper-bounded by the heuristic's own drain so the search prunes like a
/// branch and bound. Two solves on one model: the first minimizes the drain
/// alone, the second the area under the drain the first settled, so a budget
/// that runs short is spent proving the span before minimizing the area. Both
/// decide which row realizes an operation the device offers several for
/// (`selectionChoices`), alongside its start time.
LogicalResult mlir::allo::scheduleCPSAT(ChainingSharedOperatorsProblem &prob,
                                        Operation *lastOp, float cycleTime,
                                        const SpanObjective &span,
                                        const SchedulerOptions &opts) {
  prob.telemetry.cpsatRan = true;
  // First-fit placement here cannot fail (a cycle with room always exists),
  // so a failure is the resource-free LP declaring infeasibility, which no
  // exact solver repairs either.
  if (failed(
          mlir::allo::scheduleSimplex(prob, lastOp, cycleTime, opts.regFloor)))
    return failure();

  // Which row realizes each multi-candidate operation is this solve's decision
  // too, made alongside the start times. The comb row is not admitted: the span
  // order would always take its zero latency, which no price can veto.
  SmallVector<SelectionChoice, 0> choices = selectionChoices(prob, span.device);

  // The pre-pass is schedule-independent, so taking its edges hands CP-SAT the
  // chain breaks the heuristic just used. They state the period only for the
  // rows the library picked, so a model that re-decides rows drops them and
  // states the period through the sub-cycle system instead.
  SmallVector<Problem::Dependence> breaks;
  if (choices.empty())
    breaks = chainBreaksFor(prob, cycleTime, opts.regFloor);

  const auto &ops = prob.getOperations();

  DenseMap<Operation *, std::pair<int64_t, int64_t>> latRange =
      latencyRange(choices);
  auto minLat = [&](Operation *op) -> int64_t {
    auto it = latRange.find(op);
    return it != latRange.end() ? it->second.first : prob.latencyOf(op);
  };
  auto maxLat = [&](Operation *op) -> int64_t {
    auto it = latRange.find(op);
    return it != latRange.end() ? it->second.second : prob.latencyOf(op);
  };

  // The same entry cut the cyclic search takes, at the one interval a
  // straight-line region has: reaching `drainFloor` proves this schedule is as
  // short as the region gets and leaves only the area to decide.
  // `scheduleSimplex` has written the start times and their sub-cycle offsets,
  // so its schedule ships as is. An allocation or a realization still to
  // decide is worth the solve anyway.
  DenseMap<Operation *, int64_t> opFloors;
  int64_t floorDrain = drainFloor(prob, breaks, span.drain, minLat, &opFloors);
  bool allocates = false;
  for (Problem::ResourceType rsrc : prob.getResourceTypes())
    allocates |= prob.getAllocatable(rsrc).has_value();
  if (!allocates && choices.empty() && floorDrain >= span.drainOf(prob)) {
    // The floor proves the heuristic's drain minimal with nothing to solve.
    prob.telemetry.proven = true;
    prob.telemetry.spanProven = true;
    return success();
  }

  // Horizon: the whole region laid out end to end (each op after the previous
  // one's end at its longest realization, its occupancy window, plus a spare
  // cycle), wide enough that every precedence, chain break and reservation is
  // satisfiable whatever rows are decided.
  int64_t horizon = 0;
  for (Operation *op : ops)
    horizon += maxLat(op) + prob.getResourceCycles(op) + 1;

  CpModelBuilder model;
  DenseMap<Operation *, IntVar> startVars;
  // The same variables in problem order, for the objective; `ops` is a
  // SetVector so this order is stable across runs.
  SmallVector<IntVar> orderedStarts;
  orderedStarts.reserve(ops.size());
  for (Operation *op : ops) {
    IntVar var = model.NewIntVar(
        operations_research::Domain(opFloors.lookup(op), horizon));
    model.AddHint(var, *prob.getStartTime(op));
    startVars.try_emplace(op, var);
    orderedStarts.push_back(var);
  }
  SelectionVars sels = addSelection(model, choices);
  auto latExpr = [&](Operation *op) -> LinearExpr {
    if (std::optional<unsigned> i = sels.of(op))
      return sels.latency(*i);
    return LinearExpr(prob.latencyOf(op));
  };

  // Precedence, as `buildTableau` emits it: a dependence separates its
  // endpoints by the source's latency, decided or fixed.
  for (Operation *op : ops)
    for (auto &dep : prob.getDependences(op))
      model.AddLessOrEqual(startVars.at(dep.getSource()) +
                               latExpr(dep.getSource()),
                           startVars.at(dep.getDestination()));
  addChainBreaks(model, prob, startVars, breaks);
  DenseMap<Operation *, IntVar> inCycle;
  if (!sels.empty())
    inCycle = addSubCycleTimes(model, prob, startVars, cycleTime, opts.regFloor,
                               &sels, /*hintSchedule=*/true);

  // The composition of the two decisions: which row each operation runs on,
  // and how many instances of each row to build, priced together. Only in
  // allocation mode, where the binding realizes what is decided here.
  SharedClasses shared;
  if (opts.allocate)
    shared = addSharedClasses(model, prob, span.device, choices, sels,
                              startVars, inCycle, cycleTime);

  // An op occupies one instance of every unit it links to for its whole window,
  // so a cumulative constraint per resource matches `verifyOccupancy`. A
  // multi-unit op contributes the same window to each.
  auto cumulativeOn = [&](Problem::ResourceType rsrc, LinearExpr capacity) {
    CumulativeConstraint cumulative = model.AddCumulative(std::move(capacity));
    for (Operation *op : ops)
      if (prob.usesResource(op, rsrc))
        cumulative.AddDemand(model.NewFixedSizeIntervalVar(
                                 startVars.at(op), prob.getResourceCycles(op)),
                             prob.getResourceDemand(op));
  };
  for (Problem::ResourceType rsrc : prob.getResourceTypes())
    if (unsigned limit = prob.getLimit(rsrc).value_or(0))
      cumulativeOn(rsrc, limit);

  // An allocatable operator takes the same shape, with the count being decided
  // as the capacity. Occupancy windows on a line form an interval graph, so a
  // capacity is an assignment: `N` units suffice when no cycle needs more.
  SmallVector<AllocationVar> allocs = allocationVars(
      model, prob, /*ii=*/0, /*hint=*/true, cycleTime, &shared.owned);
  for (const AllocationVar &alloc : allocs)
    cumulativeOn(alloc.rsrc, alloc.units);
  addAllocationHeadroom(model, prob, startVars, allocs, cycleTime,
                        opts.regFloor, sels.empty() ? nullptr : &inCycle,
                        /*hintSchedule=*/true);

  // What the region is charged, bounded by what the heuristic already reached
  // and below by the floor, which speeds the span proof. Under the area
  // objective the heuristic's drain is a span leash instead: a constraint, not
  // what the first solve minimizes, widened by any composition slack the
  // kernel's sibling DAG granted this region.
  int64_t heuristicDrain = span.drainOf(prob);
  assert(heuristicDrain <= horizon &&
         "the horizon must cover the schedule the heuristic just found, or "
         "capping the drain variable at it cuts that schedule out and the "
         "solve comes back INFEASIBLE against a model that has one");
  assert(floorDrain <= heuristicDrain &&
         "the drain floor is a lower bound on every schedule, the heuristic's "
         "included");
  IntVar drain = drainVariable(model, startVars, span.drain, horizon,
                               heuristicDrain, latExpr, heuristicDrain);
  model.AddGreaterOrEqual(drain, floorDrain);

  auto ship = [&](const CpSolverResponse &pick) {
    for (Operation *op : ops)
      prob.setStartTime(op, SolutionIntegerValue(pick, startVars.at(op)));
    Allocated decided = readAllocation(pick, allocs);
    if (!sels.empty()) {
      SmallVector<unsigned> chosen = readSelection(pick, sels);
      applySelection(prob, choices, chosen);
      applySharedClasses(prob, span.device, shared, choices, chosen,
                         readSharedUnits(pick, shared), decided);
    }
    applyAllocation(prob, decided, /*ii=*/0);
    return finishSchedule(prob, cycleTime, opts.regFloor);
  };

  // The heuristic's schedule stands: this solve came back with none.
  auto giveUp = [&](const CpSolverResponse &response) {
    reportUnsolved(prob, response, opts.budget);
    prob.telemetry.fallback = true;
    prob.telemetry.budgetExhausted = true;
    applyFallbackAllocation(prob, span.device, opts.allocate, /*ii=*/0,
                            cycleTime);
    return success();
  };

  // The span solve, skipped where the floor already proves the heuristic's
  // drain minimal and only the area is left to decide.
  bool spanProven = floorDrain >= heuristicDrain;
  int64_t solvedDrain = heuristicDrain;
  CpSolverResponse first;
  bool ranFirst = false;
  SchedulerOptions live = opts;
  if (!spanProven) {
    model.Minimize(drain);
    first = solveBuilt(model, solverParameters(opts));
    if (!solved(first))
      return giveUp(first);
    ranFirst = true;
    spanProven = first.status() == CpSolverStatus::OPTIMAL;
    if (!spanProven) {
      // One escalated retry for the missing certificate (kSpanEscalation).
      live.budget = opts.budget * kSpanEscalation;
      CpSolverResponse retry = solveBuilt(model, solverParameters(live));
      if (solved(retry) && (retry.status() == CpSolverStatus::OPTIMAL ||
                            SolutionIntegerValue(retry, drain) <=
                                SolutionIntegerValue(first, drain))) {
        first = retry;
        spanProven = retry.status() == CpSolverStatus::OPTIMAL;
      }
    }
    solvedDrain = SolutionIntegerValue(first, drain);
    assert(solvedDrain <= heuristicDrain &&
           "the model bounds the drain by the heuristic's own");
  }

  // The area solve, under the span the first settled, on what is left of the
  // budget. A positive `areaSlack` widens the drain bound past the proven span,
  // trading that much latency for the smaller design the extra room admits
  // (deeper delay chains that spread operations onto fewer units).
  int64_t leashDrain =
      solvedDrain + static_cast<int64_t>(solvedDrain * opts.areaSlack);
  model.AddLessOrEqual(drain, leashDrain);
  LinearExpr area = areaTerms(model, orderedStarts, span, startVars, allocs,
                              /*ii=*/0, horizon, latExpr, sels, shared);
  model.Minimize(area);
  SchedulerOptions rest = live;
  if (ranFirst) {
    rehintFrom(model, ops.getArrayRef(), startVars, allocs, sels, shared,
               first);
    rest = lessBudget(live, first);
  }
  // Certified stop, as the fold's slices: within kFoldGapEps of the model's
  // optimum is close enough to end the solve with a certificate.
  SatParameters areaParams = solverParameters(rest);
  areaParams.set_relative_gap_limit(kFoldGapEps);
  CpSolverResponse second = solveBuilt(model, areaParams);
  assert(second.status() != CpSolverStatus::INFEASIBLE &&
         "the span solve's schedule satisfies the pinned model");
  if (!solved(second) && !ranFirst)
    return giveUp(second);
  if (solved(second)) {
    prob.telemetry.modelArea = SolutionIntegerValue(second, area);
    prob.telemetry.modelAreaBound = second.best_objective_bound();
  }
  const CpSolverResponse *pick = solved(second) ? &second : &first;

  // Reclaim the paid slack: with the area pinned at its minimum, minimize the
  // drain back down so the design ships the shortest span that costs no more.
  // Runs only where a positive slack widened the bound, since at the tight
  // bound the area solve already sits at the minimal span.
  CpSolverResponse third;
  if (opts.areaSlack > 0.0 && solved(second)) {
    model.AddLessOrEqual(area, SolutionIntegerValue(second, area));
    pinStructure(model, second, allocs, sels, shared);
    model.Minimize(drain);
    rehintAll(model, second);
    third = solveBuilt(model, solverParameters(lessBudget(rest, second)));
    assert(third.status() != CpSolverStatus::INFEASIBLE &&
           "the area solve's schedule satisfies the pinned model");
    if (solved(third))
      pick = &third;
  }

  if (!spanProven)
    warn(Stage::Sched, prob.getContainingOp())
        << "Exact scheduling ran out of budget before proving this region's "
           "span minimal; what ships is no worse than the heuristic's";
  else if (second.status() != CpSolverStatus::OPTIMAL)
    warn(Stage::Sched, prob.getContainingOp())
        << "Exact scheduling settled this region's span but ran out of budget "
           "minimizing the area under it, so its registers and instances are "
           "not known to be fewest";

  int64_t finalDrain = SolutionIntegerValue(*pick, drain);
  if (finalDrain < heuristicDrain)
    info(Stage::Sched, prob.getContainingOp())
        << "Exact scheduling shortened the region: its deepest output now "
           "commits at cycle "
        << finalDrain << " instead of " << heuristicDrain;

  prob.telemetry.proven =
      spanProven && second.status() == CpSolverStatus::OPTIMAL;
  prob.telemetry.spanProven = spanProven;
  prob.telemetry.budgetExhausted = !prob.telemetry.proven;
  return ship(*pick);
}

//===----------------------------------------------------------------------===//
// The cyclic solve: a branch and bound over initiation intervals.
//===----------------------------------------------------------------------===//

namespace {

/// What one fixed-II solve settled. `Infeasible` is a proof that the
/// initiation interval admits no schedule; `Exhausted` is the solver giving
/// up, which proves nothing.
enum class ModuloOutcome { Scheduled, Infeasible, Exhausted };

/// What one fixed-II solve writes back on a Scheduled outcome: the placement
/// and every decision made alongside it.
struct ModuloAttempt {
  DenseMap<Operation *, unsigned> starts;
  Allocated decided;
  SmallVector<unsigned> chosen, classUnits;
  bool spanProven = false, areaProven = false;
  int64_t drain = 0;
  /// The shipped schedule's modeled area, chains included. Absent where the
  /// cycles order's area solve found nothing under the settled span.
  std::optional<int64_t> modelArea;
  /// The area minimization's dual bound when its solve last returned; absent
  /// where no area solve ran (the structural bootstrap spent the budget).
  std::optional<double> modelAreaBound;
};

/// Solve \p prob at the fixed initiation interval \p ii, writing what it
/// settles into \p out when a schedule exists. Fixing the II keeps the model
/// linear:
/// `ii * distance` in a precedence edge and the modulo congruence below would
/// otherwise need a variable modulus.
///
/// \p hint is only valid when the greedy placement itself reached this II; at
/// any other II its start times are not a schedule. It is dropped where
/// \p drainBound cuts below the greedy schedule's own drain, since the hint
/// then violates the model and the interleaved portfolio check-fails on it.
///
/// Two solves on one model, span then area, sharing one budget (see the
/// acyclic entry). Where a span composes (\p span.trip) the area solve is held
/// to `kAreaTieBreakShare` of it, since the caller folds the winning interval
/// in the area order on a fresh budget; the area solve here only breaks the
/// interval tie and seeds that fold. `out.spanProven` and `out.areaProven` are
/// each solve's
/// OPTIMAL against FEASIBLE, which the II search cannot otherwise tell apart,
/// and an unproven placement's drain is still what the region's span gets
/// charged.
///
/// \p drainBound is the incumbent's, so INFEASIBLE here means nothing beats
/// the incumbent at this II rather than a proof the interval is impossible.
/// \p floorDrain is `drainFloor`'s bound, valid at every interval.
///
/// \p areaMode swaps the two solves into the area order: area first with the
/// drain held to \p drainBound (the caller's leash at this II) and the area
/// held under \p areaBound (the incumbent's), then the shortest drain the
/// settled area admits. `out.modelArea` receives the shipped schedule's
/// modeled area, the figure the caller compares intervals on and holds the
/// fold re-solve to; the host-side mirror below carries no register-chain
/// term and cannot serve there. The cycles order sets it only where its area
/// solve found a schedule.
///
/// \p choices are the realization decisions this model carries (empty
/// \p breaks then, the period stated through the sub-cycle system);
/// `out.chosen` receives the candidate settled per choice. \p sharedMeta is
/// the region's one collection of shared classes; this model builds its own
/// variables on it, and `out.classUnits` receives the counts settled, aligned
/// with it.
ModuloOutcome solveAtII(ChainingModuloProblem &prob, Operation *lastOp,
                        ArrayRef<Problem::Dependence> breaks,
                        ArrayRef<SelectionChoice> choices,
                        SharedClasses &sharedMeta, float cycleTime,
                        const SpanObjective &span, const SchedulerOptions &opts,
                        std::optional<int64_t> drainBound, int64_t floorDrain,
                        const DenseMap<Operation *, int64_t> &opFloors,
                        std::optional<int64_t> areaBound, bool areaMode,
                        unsigned ii, unsigned horizon, bool hint,
                        ModuloAttempt &out) {
  if (hint && drainBound && span.drainOf(prob) > *drainBound)
    hint = false;
  const auto &ops = prob.getOperations();

  CpModelBuilder model;
  DenseMap<Operation *, IntVar> startVars;
  SmallVector<IntVar> orderedStarts;
  unsigned anchorIndex = 0;
  orderedStarts.reserve(ops.size());
  for (Operation *op : ops) {
    IntVar var = model.NewIntVar(
        operations_research::Domain(opFloors.lookup(op), horizon));
    if (hint)
      model.AddHint(var, *prob.getStartTime(op));
    startVars.try_emplace(op, var);
    if (op == lastOp)
      anchorIndex = orderedStarts.size();
    orderedStarts.push_back(var);
  }
  SelectionVars sels = addSelection(model, choices);
  auto latExpr = [&](Operation *op) -> LinearExpr {
    if (std::optional<unsigned> i = sels.of(op))
      return sels.latency(*i);
    return LinearExpr(prob.latencyOf(op));
  };

  // Precedence. An edge spanning `distance` iterations is relaxed by one II
  // per iteration it spans, matching the cyclic constraint row `buildTableau`
  // emits; a chain-breaking edge is intra-iteration and carries no II term. A
  // forwarded store->load edge needs only issue order (its shadow supplies the
  // datum), so it separates by zero.
  for (Operation *op : ops)
    for (auto &dep : prob.getDependences(op)) {
      Operation *src = dep.getSource();
      LinearExpr sep =
          (prob.isForwarded(dep) || prob.hasSeparationOverride(dep))
              ? LinearExpr(prob.separationOf(dep))
              : latExpr(src);
      model.AddLessOrEqual(startVars.at(src) + sep -
                               static_cast<int64_t>(ii) *
                                   prob.getDistance(dep).value_or(0),
                           startVars.at(dep.getDestination()));
    }
  addChainBreaks(model, prob, startVars, breaks);
  DenseMap<Operation *, IntVar> inCycle;
  if (!sels.empty())
    inCycle = addSubCycleTimes(model, prob, startVars, cycleTime, opts.regFloor,
                               &sels, hint);
  // This model's own view of the collected classes; the copy carries the
  // per-model variables, the collection stays pristine for the writeback.
  SharedClasses shared = sharedMeta;
  addSharedClassVars(model, span.device, shared, choices, sels, inCycle,
                     cycleTime);
  DenseSet<Operation *> sharedMembers = sharedMemberOps(shared, choices);

  // One-hot congruence class per contending op and per shared-class member.
  // `t = ii*lap + sum(p*slot[p])` defines class and modulo at once with no
  // reification: slot[p] is membership in class p, which the sums below need.
  DenseMap<Operation *, SmallVector<BoolVar>> slotsOf;
  SmallVector<int64_t> classes(ii);
  for (unsigned p = 0; p < ii; ++p)
    classes[p] = p;
  for (Operation *op : ops) {
    if (!prob.contendsForUnit(op) && !sharedMembers.contains(op))
      continue;
    SmallVector<BoolVar> slots;
    slots.reserve(ii);
    for (unsigned p = 0; p < ii; ++p)
      slots.push_back(model.NewBoolVar());
    model.AddExactlyOne(slots);
    IntVar lap = model.NewIntVar(operations_research::Domain(0, horizon / ii));
    model.AddEquality(startVars.at(op),
                      lap * static_cast<int64_t>(ii) +
                          LinearExpr::WeightedSum(slots, classes));
    if (hint) {
      unsigned at = *prob.getStartTime(op);
      for (unsigned p = 0; p < ii; ++p)
        model.AddHint(slots[p], p == at % ii);
      model.AddHint(lap, at / ii);
    }
    slotsOf.try_emplace(op, std::move(slots));
  }

  // Modulo reservation: an op holding a unit for `occ` cycles wraps the II
  // table floor(occ/ii) times (every class) plus `occ % ii` more from its
  // own slot, exactly what `MRT::enter` counts, so the two models cross-check.
  auto usesIn = [&](Problem::ResourceType rsrc, unsigned slot) {
    LinearExpr used;
    for (Operation *op : ops) {
      if (!prob.usesResource(op, rsrc))
        continue;
      unsigned occ = prob.getResourceCycles(op);
      auto held = static_cast<int64_t>(prob.getResourceDemand(op));
      used += static_cast<int64_t>(occ / ii) * held;
      const SmallVector<BoolVar> &slots = slotsOf.at(op);
      for (unsigned k = 0, partial = occ % ii; k < partial; ++k)
        used += LinearExpr::Term(slots[(slot + ii - k) % ii], held);
    }
    return used;
  };
  for (Problem::ResourceType rsrc : prob.getResourceTypes()) {
    unsigned limit = prob.getLimit(rsrc).value_or(0);
    if (limit == 0)
      continue;
    for (unsigned slot = 0; slot < ii; ++slot)
      model.AddLessOrEqual(usesIn(rsrc, slot), static_cast<int64_t>(limit));
  }

  // A shared class's capacity, per congruence slot: a static member occupies
  // its slot outright, a conditional one only where its row is selected, as
  // the AND of the two literals. Every member occupies one cycle
  // (`collectSharedClasses` admits no other in a cyclic region), so no window
  // wraps the table, and `ii * units >= members` is the demand cut.
  for (SharedClassVar &cls : shared.classes) {
    model.AddGreaterOrEqual(
        LinearExpr::Term(cls.units, static_cast<int64_t>(ii)), cls.members);
    for (unsigned slot = 0; slot < ii; ++slot) {
      LinearExpr used;
      for (auto [op, occ, in] : cls.statics) {
        assert(occ == 1 && "a multi-cycle member joins no cyclic class");
        used += slotsOf.at(op)[slot];
      }
      for (const SharedClassVar::Cond &cond : cls.conds) {
        assert(cond.occ == 1 && "a cyclic candidate row is pipelined");
        const BoolVar &sat = slotsOf.at(choices[cond.choice].op)[slot];
        const BoolVar &sel = sels.sel[cond.choice][cond.cand];
        BoolVar joins = model.NewBoolVar();
        model.AddBoolAnd({sat, sel}).OnlyEnforceIf(joins);
        model.AddBoolOr({joins, sat.Not(), sel.Not()});
        used += joins;
      }
      model.AddLessOrEqual(used, cls.units);
    }
  }

  // The same sum against the count being decided. Allocatable operators occupy
  // one cycle here, so an op sits in one class and a per-class count is
  // realizable as an assignment. `N_r >= ceil(total/ii)` is implied, cut here.
  SmallVector<AllocationVar> allocs =
      allocationVars(model, prob, ii, hint, cycleTime, &shared.owned);
  for (const AllocationVar &alloc : allocs) {
    int64_t total = 0;
    for (Operation *op : ops)
      if (prob.usesResource(op, alloc.rsrc))
        total += prob.getResourceCycles(op);
    model.AddGreaterOrEqual(alloc.units, (total + ii - 1) / ii);
    for (unsigned slot = 0; slot < ii; ++slot)
      model.AddLessOrEqual(usesIn(alloc.rsrc, slot), alloc.units);
  }
  addAllocationHeadroom(model, prob, startVars, allocs, cycleTime,
                        opts.regFloor, sels.empty() ? nullptr : &inCycle, hint);

  // `(trip - 1) * ii` is constant at a fixed II, so minimizing the span here is
  // minimizing the drain; the outer search carries the II term. With no span to
  // compose, the anchor's start time takes the primary slot instead.
  std::optional<IntVar> drainVar;
  if (span.trip) {
    drainVar = drainVariable(
        model, startVars, span.drain, horizon, drainBound, latExpr,
        hint ? std::optional(span.drainOf(prob)) : std::nullopt);
    model.AddGreaterOrEqual(*drainVar, floorDrain);
  }
  IntVar primary = drainVar.value_or(orderedStarts[anchorIndex]);
  // A hint reaches the portfolio only on a model it satisfies: the interleaved
  // workers check-fail (or-tools 9.15, `ConfigureSearchHeuristics`) on a hinted
  // model infeasible at load, which a probe bounded by the incumbent can be. An
  // unhinted probe keeps none of the per-variable defaults the builders left
  // (selection, shared counts).
  if (!hint)
    model.ClearHints();

  if (areaMode) {
    assert(drainVar && "the area order runs only where a span composes");
    // Three solves on one model, sharing one budget. A structural bootstrap
    // first (instances, selects and rows), which searches well: the
    // heuristic's hint completes only at its own interval and carries its own
    // realization, which a packed schedule degenerates to the ceiling
    // allocation. Then the full area, complete-hinted from the bootstrap's
    // whole solution since its chain terms search too poorly to stand alone;
    // then the shortest drain the settled structure admits.
    LinearExpr structural;
    LinearExpr area =
        areaTerms(model, orderedStarts, span, startVars, allocs, ii, horizon,
                  latExpr, sels, shared, hint ? &prob : nullptr, &structural);
    model.Minimize(structural);
    SchedulerOptions bootOpts = opts;
    bootOpts.budget = opts.budget * kBootstrapShare;
    CpSolverResponse boot = solveBuilt(model, solverParameters(bootOpts));
    SchedulerOptions restArea = lessBudget(opts, boot);
    // A capped bootstrap that decided nothing gets the rest of the budget;
    // only a proof or a spent budget may end the solve.
    if (!solved(boot) && boot.status() != CpSolverStatus::INFEASIBLE) {
      boot = solveBuilt(model, solverParameters(restArea));
      restArea = lessBudget(restArea, boot);
    }
    if (boot.status() == CpSolverStatus::INFEASIBLE)
      return ModuloOutcome::Infeasible;
    if (!solved(boot)) {
      assert(boot.status() != CpSolverStatus::MODEL_INVALID &&
             "the encoding built an ill-formed model");
      info(Stage::Sched, prob.getContainingOp())
          << "Area bootstrap at II=" << ii << " found no schedule: status "
          << CpSolverStatus_Name(boot.status()) << " after "
          << llvm::format("%.1f", boot.deterministic_time()) << " of "
          << llvm::format("%g", opts.budget) << " deterministic units";
      return ModuloOutcome::Exhausted;
    }
    // The full-area solve. INFEASIBLE under \p areaBound means nothing here
    // beats the incumbent; a proven structural minimum already past the bound
    // says so without a solve, the full area being the structure plus
    // nonnegative chain and pulse terms. A budget-exhausted solve ships the
    // bootstrap, whose chain terms hold feasible but unminimized values, so its
    // recorded area overstates and its full area can exceed the bound added
    // below; the model then carries no hint, since a hint violating it
    // check-fails the portfolio.
    if (areaBound && boot.status() == CpSolverStatus::OPTIMAL &&
        SolutionIntegerValue(boot, structural) > *areaBound)
      return ModuloOutcome::Infeasible;
    if (!areaBound || SolutionIntegerValue(boot, area) <= *areaBound)
      rehintAll(model, boot);
    else
      model.ClearHints();
    if (areaBound)
      model.AddLessOrEqual(area, *areaBound);
    model.Minimize(area);
    // The area minimization runs in deterministic-time slices, each warm
    // started from the last incumbent, and stops once the area plateaus so the
    // rest of the budget (a proven span still reclaims its slack below) is
    // released instead of burned on a solve that no longer improves. What
    // ships is the best slice, not the last one: a restarted slice can return
    // a worse incumbent than its predecessor's.
    CpSolverResponse first;
    double areaSpent = 0.0;
    int64_t bestArea = 0, plateauRef = 0;
    bool haveBest = false;
    unsigned stalls = 0;
    while (areaSpent < restArea.budget) {
      SchedulerOptions slice = restArea;
      slice.budget =
          std::min(restArea.budget - areaSpent, opts.budget * kFoldChunkShare);
      SatParameters sliceParams = solverParameters(slice);
      sliceParams.set_relative_gap_limit(kFoldGapEps);
      CpSolverResponse r = solveBuilt(model, sliceParams);
      areaSpent += r.deterministic_time();
      if (r.status() == CpSolverStatus::INFEASIBLE) {
        if (!haveBest)
          return ModuloOutcome::Infeasible;
        break; // nothing beats the incumbent under the area bound
      }
      if (!solved(r))
        break; // this slice found nothing; the bootstrap stands
      int64_t a = SolutionIntegerValue(r, area);
      bool improved =
          !haveBest || plateauRef - a > plateauRef * kFoldPlateauEps;
      if (!haveBest || a <= bestArea) {
        first = r;
        bestArea = a;
      }
      if (improved) {
        plateauRef = a;
        haveBest = true;
        stalls = 0;
      } else if (++stalls >= kFoldPatience) {
        info(Stage::Sched, prob.getContainingOp())
            << "Area fold at II=" << ii << " plateaued after "
            << llvm::format("%.1f", areaSpent) << " of "
            << llvm::format("%g", restArea.budget)
            << " deterministic units; releasing the rest";
        break; // area has plateaued
      }
      if (r.status() == CpSolverStatus::OPTIMAL)
        break; // proven minimal
      if (double(a) - r.best_objective_bound() <= kFoldGapEps * double(a)) {
        info(Stage::Sched, prob.getContainingOp())
            << "Area fold at II=" << ii << " stopped certified: incumbent "
            << bestArea << " is within "
            << llvm::format("%.1f%%", kFoldGapEps * 100)
            << " of the model's optimum after "
            << llvm::format("%.1f", areaSpent) << " deterministic units";
        break;
      }
      rehintAll(model, r); // warm start the next slice from the incumbent
    }
    out.areaProven = first.status() == CpSolverStatus::OPTIMAL;
    if (solved(first))
      out.modelAreaBound = first.best_objective_bound();
    const CpSolverResponse *pick = solved(first) ? &first : &boot;
    CpSolverResponse second;
    if (solved(first)) {
      int64_t solvedArea = SolutionIntegerValue(first, area);
      model.AddLessOrEqual(area, solvedArea);
      pinStructure(model, first, allocs, sels, shared);
      model.Minimize(*drainVar);
      rehintAll(model, first);
      SchedulerOptions afterArea = restArea;
      afterArea.budget = std::max(restArea.budget - areaSpent, 0.0);
      second = solveBuilt(model, solverParameters(afterArea));
      assert(second.status() != CpSolverStatus::INFEASIBLE &&
             "the area solve's schedule satisfies the pinned model");
      out.spanProven = second.status() == CpSolverStatus::OPTIMAL;
      if (solved(second))
        pick = &second;
    }
    for (Operation *op : ops)
      out.starts[op] = SolutionIntegerValue(*pick, startVars.at(op));
    out.decided = readAllocation(*pick, allocs);
    out.chosen = readSelection(*pick, sels);
    out.classUnits = readSharedUnits(*pick, shared);
    out.drain = SolutionIntegerValue(*pick, *drainVar);
    out.modelArea = SolutionIntegerValue(*pick, area);
    return ModuloOutcome::Scheduled;
  }

  // The span skip, the cyclic mirror of the acyclic path's: at the greedy's
  // own interval a floor that reaches the heuristic's drain proves that
  // schedule span-optimal, so the span solve has nothing left to decide and
  // the budget goes to the area tie-break. Solving anyway is worse than
  // redundant: the presolve can break the completed hint, and on a saturated
  // packing the search then burns the whole budget failing to rebuild a
  // schedule the heuristic already holds.
  bool skipSpan = hint && drainVar && floorDrain >= span.drainOf(prob);
  CpSolverResponse first;
  SchedulerOptions live = opts;
  if (skipSpan) {
    out.spanProven = true;
  } else {
    model.Minimize(primary);
    first = solveBuilt(model, solverParameters(opts));
    if (first.status() == CpSolverStatus::INFEASIBLE)
      return ModuloOutcome::Infeasible;
    if (!solved(first)) {
      assert(first.status() != CpSolverStatus::MODEL_INVALID &&
             "the encoding built an ill-formed model");
      return ModuloOutcome::Exhausted;
    }
    out.spanProven = first.status() == CpSolverStatus::OPTIMAL;
    if (!out.spanProven) {
      // One escalated retry for the missing certificate (kSpanEscalation).
      live.budget = opts.budget * kSpanEscalation;
      CpSolverResponse retry = solveBuilt(model, solverParameters(live));
      if (solved(retry) && (retry.status() == CpSolverStatus::OPTIMAL ||
                            SolutionIntegerValue(retry, primary) <=
                                SolutionIntegerValue(first, primary))) {
        first = retry;
        out.spanProven = retry.status() == CpSolverStatus::OPTIMAL;
      }
    }
  }

  // The area solve, under the span the first settled (or the floor proved),
  // on what is left of the budget.
  model.AddLessOrEqual(primary, skipSpan
                                    ? span.drainOf(prob)
                                    : SolutionIntegerValue(first, primary));
  LinearExpr area = areaTerms(model, orderedStarts, span, startVars, allocs, ii,
                              horizon, latExpr, sels, shared);
  model.Minimize(area);
  SchedulerOptions rest = live;
  if (!skipSpan) {
    rehintFrom(model, ops.getArrayRef(), startVars, allocs, sels, shared,
               first);
    rest = lessBudget(live, first);
  }
  if (span.trip)
    rest.budget = std::min(rest.budget, opts.budget * kAreaTieBreakShare);
  // Certified stop, as the fold's slices: a tie-break certified within
  // kFoldGapEps reads areaProven, which also skips the fold re-solve; the
  // fold cannot beat a certificate by more than the certificate.
  SatParameters tieParams = solverParameters(rest);
  tieParams.set_relative_gap_limit(kFoldGapEps);
  CpSolverResponse second = solveBuilt(model, tieParams);
  assert(second.status() != CpSolverStatus::INFEASIBLE &&
         "the span solve's (or the hinted heuristic's) schedule satisfies the "
         "pinned model");
  out.areaProven = second.status() == CpSolverStatus::OPTIMAL;
  // With the span skipped there is no first solution to fall back on; the
  // heuristic's schedule stands and the caller's fallback ships it,
  // span-proven by the floor.
  if (skipSpan && !solved(second))
    return ModuloOutcome::Exhausted;
  const CpSolverResponse &pick = solved(second) ? second : first;

  for (Operation *op : ops)
    out.starts[op] = SolutionIntegerValue(pick, startVars.at(op));
  out.decided = readAllocation(pick, allocs);
  out.chosen = readSelection(pick, sels);
  out.classUnits = readSharedUnits(pick, shared);
  out.drain = drainVar ? SolutionIntegerValue(pick, *drainVar) : 0;
  if (solved(second)) {
    out.modelArea = SolutionIntegerValue(pick, area);
    out.modelAreaBound = second.best_objective_bound();
  }
  return ModuloOutcome::Scheduled;
}

/// The intervals to probe, ascending from the heuristic's II lower bound, the
/// caller's span cut breaking the scan.
static SmallVector<unsigned> intervalProbes(const SimplexWarmStart &warm,
                                            unsigned upperII) {
  SmallVector<unsigned> probes;
  for (unsigned ii = warm.lowerBoundII; ii <= upperII; ++ii)
    probes.push_back(ii);
  return probes;
}

} // namespace

/// Refines the heuristic's modulo (cyclic) schedule by searching fixed II
/// values from the heuristic's own II lower bound upward, as a branch and bound
/// on the region's span. Only that lower bound (from the resource-free LP) is
/// needed; the heuristic's placement is optional context (`SimplexWarmStart`).
///
/// The search cannot stop at the first feasible II: what the region is charged
/// is `(trip - 1) * ii + drain`, and a larger II can still win with a shorter
/// drain. It keeps the best span seen and cuts once an interval's II term
/// alone already reaches it.
LogicalResult mlir::allo::scheduleCPSAT(ChainingModuloProblem &prob,
                                        Operation *lastOp, float cycleTime,
                                        unsigned minII, unsigned maxII,
                                        const SpanObjective &span,
                                        const SchedulerOptions &opts) {
  prob.telemetry.cpsatRan = true;
  SimplexWarmStart warm;
  if (failed(mlir::allo::scheduleSimplex(prob, lastOp, cycleTime, opts.regFloor,
                                         minII, &warm)))
    return failure();

  unsigned greedyII = warm.placed ? *prob.getInitiationInterval() : 0;
  assert((!warm.placed || greedyII >= warm.lowerBoundII) &&
         "placement only ever grows the II");

  // Which row realizes each multi-candidate operation is this solve's decision
  // too, made alongside the placement at each interval. The comb row is not
  // admitted: the span order would always take its zero latency.
  SmallVector<SelectionChoice, 0> choices = selectionChoices(prob, span.device);

  // The heuristic ran the same pre-pass, so the schedule this falls back to
  // meets the period. The edges state the period only for the rows the
  // library picked, so a model that re-decides rows drops them and states the
  // period through the sub-cycle system instead.
  SmallVector<Problem::Dependence> breaks;
  if (choices.empty())
    breaks = chainBreaksFor(prob, cycleTime, opts.regFloor);

  // The classes selection and allocation compose over, collected once:
  // membership does not depend on the interval, so every fixed-II model
  // builds its variables on this one collection and the adopted counts are
  // written back through it. Only in allocation mode, where the binding
  // realizes what is decided here.
  SharedClasses sharedMeta;
  if (opts.allocate)
    sharedMeta = collectSharedClasses(prob, span.device, choices);
  DenseSet<Operation *> sharedMembers = sharedMemberOps(sharedMeta, choices);

  DenseMap<Operation *, std::pair<int64_t, int64_t>> latRange =
      latencyRange(choices);
  auto minLat = [&](Operation *op) -> int64_t {
    auto it = latRange.find(op);
    return it != latRange.end() ? it->second.first : prob.latencyOf(op);
  };

  // Window: region laid out end to end (satisfying precedence and chain
  // breaks, at each op's longest realization) plus one II per contending op
  // (a shared-class member contends too), widened to the heuristic's own
  // reach. Must be provably sufficient, since INFEASIBLE here counts as proof.
  const auto &ops = prob.getOperations();
  int64_t sequential = 0;
  int64_t greedyReach = 0;
  unsigned contending = 0;
  for (Operation *op : ops) {
    auto it = latRange.find(op);
    sequential +=
        (it != latRange.end() ? it->second.second : prob.latencyOf(op)) + 1;
    if (prob.contendsForUnit(op) || sharedMembers.contains(op))
      ++contending;
    if (warm.placed)
      greedyReach = std::max(greedyReach, int64_t(*prob.getStartTime(op)));
  }
  int64_t window = std::max(sequential, greedyReach);

  // Search bound: with a greedy incumbent, scan through its own II (the II
  // alone isn't sufficient; placement there must still be solved). With no
  // incumbent, bound by total occupancy, where every op gets its own slot.
  unsigned totalOccupancy = 0;
  for (Operation *op : ops)
    if (prob.holdsLimitedUnit(op))
      totalOccupancy += prob.getResourceCycles(op);
  unsigned upperII =
      warm.placed ? greedyII : std::max(warm.lowerBoundII, totalOccupancy);

  // The part of `leafSpan` this solve controls. With no trip there is no span
  // to compare across intervals, so the search takes the first feasible II,
  // placed as shallowly as the anchor objective can manage.
  bool bySpan = span.trip.has_value();
  int64_t iiWeight = bySpan ? *span.trip - 1 : 0;

  // The incumbent: bounds every model below, and is the fallback if none beats
  // it. Without it, a budget-limited placement at a new II is unbounded and can
  // ship a schedule worse than the heuristic's.
  std::optional<int64_t> heuristicSpan;
  if (bySpan && warm.placed)
    heuristicSpan = iiWeight * greedyII + span.drainOf(prob);
  std::optional<int64_t> best = heuristicSpan;
  DenseMap<Operation *, int64_t> opFloors;
  int64_t floorDrain =
      bySpan ? drainFloor(prob, breaks, span.drain, minLat, &opFloors) : 0;

  // Whether this region has an allocation to decide at all, which the cut
  // below admits a span tie for.
  bool allocates = false;
  for (Problem::ResourceType rsrc : prob.getResourceTypes())
    allocates |= prob.getAllocatable(rsrc).has_value();

  ModuloAttempt bestAttempt;
  int64_t bestArea = 0;
  unsigned bestII = 0;
  bool adopted = false;
  std::optional<unsigned> exhaustedAt;

  for (unsigned ii : intervalProbes(warm, upperII)) {
    // Cut: this interval's span already reaches the incumbent's before a single
    // operation is placed, and every later interval is worse; where an
    // allocation or a realization is decided, admit a tie since it can still
    // win on area.
    if (best && iiWeight * ii + floorDrain >=
                    *best + ((allocates || !choices.empty()) ? 1 : 0))
      break;
    std::optional<int64_t> drainBound;
    if (best)
      drainBound = *best - iiWeight * ii;

    ModuloAttempt attempt;
    ModuloOutcome outcome =
        solveAtII(prob, lastOp, breaks, choices, sharedMeta, cycleTime, span,
                  opts, drainBound, bySpan ? floorDrain : 0, opFloors,
                  /*areaBound=*/std::nullopt, /*areaMode=*/false, ii,
                  window + ii * contending,
                  /*hint=*/warm.placed && ii == greedyII, attempt);
    if (outcome == ModuloOutcome::Infeasible) {
      // INFEASIBLE is a proof only where nothing bounded the solve; under the
      // incumbent's bound it is the weaker "nothing here beats it".
      assert((!warm.placed || ii < greedyII || drainBound) &&
             "the greedy's own schedule satisfies this encoding at the II it "
             "achieved");
      continue;
    }
    if (outcome == ModuloOutcome::Exhausted) {
      // Stop rather than try a wider interval: the budget just proved this
      // problem hard.
      exhaustedAt = ii;
      break;
    }
    // Adopt on a strict improvement, or on the first exact schedule at all:
    // span first, the instances built and the rows decided breaking the tie.
    int64_t solved = iiWeight * ii + attempt.drain;
    int64_t area = areaOf(prob, attempt.decided) +
                   selectionPrice(choices, attempt.chosen, sharedMeta.covered) +
                   sharedAreaOf(span.device, sharedMeta, choices,
                                attempt.chosen, attempt.classUnits);
    bool adopt =
        !adopted || solved < *best || (solved == *best && area < bestArea);
    if (adopt) {
      best = solved;
      bestArea = area;
      bestII = ii;
      bestAttempt = std::move(attempt);
      adopted = true;
    }
    if (!bySpan)
      break;
  }

  if (!adopted) {
    if (!warm.placed) {
      auto d = unsupported(Stage::Sched, Code::PlacementFailed,
                           prob.getContainingOp());
      d << "Neither scheduler could place this region: the greedy modulo "
           "placement gave up, and ";
      if (exhaustedAt)
        d << "the exact one ran out of budget at II=" << *exhaustedAt
          << " without deciding it";
      else
        d << "every initiation interval from " << warm.lowerBoundII << " to "
          << upperII << " is infeasible";
      return failure();
    }
    // Both arms leave the problem exactly as the simplex left it.
    if (exhaustedAt)
      warn(Stage::Sched, prob.getContainingOp())
          << "Exact scheduling ran out of budget at II=" << *exhaustedAt
          << " without deciding it; falling back to the heuristic's schedule "
             "at II="
          << greedyII << ", which is therefore not known to be minimal";
    else
      info(Stage::Sched, prob.getContainingOp())
          << "Exact scheduling found nothing shorter than the heuristic's "
             "schedule at II="
          << greedyII << "; keeping it";
    prob.telemetry.fallback = true;
    // Every interval decided and none beat the incumbent: the heuristic's
    // schedule is thereby proven optimal. An exhaustion at the greedy's own
    // interval with the floor at the heuristic's drain spent the budget on
    // area alone; the span is still proven.
    prob.telemetry.proven = !exhaustedAt;
    prob.telemetry.spanProven =
        !exhaustedAt || (bySpan && *exhaustedAt == greedyII &&
                         floorDrain >= span.drainOf(prob));
    prob.telemetry.budgetExhausted = exhaustedAt.has_value();
    if (exhaustedAt)
      prob.telemetry.exhaustedAtII = (int64_t)*exhaustedAt;
    applyFallbackAllocation(prob, span.device, opts.allocate, greedyII,
                            cycleTime);
    return success();
  }

  // The span scan's per-interval area tie-break runs on a capped share of the
  // budget (kAreaTieBreakShare) and may not prove the area minimal, so a proven
  // span ships at the scan's incidental allocation. Re-solve in the area order
  // over the interval envelope the proven span admits, seeded from the
  // incumbent on a fresh budget released early once it stops improving, so the
  // fewest units hold. At `areaSlack` zero the envelope is the winning interval
  // alone; a positive slack opens every interval whose span stays within `best
  // * (1 + areaSlack)`, where a wider II folds operations onto fewer units. A
  // fold whose own area solve runs out of budget ships an unminimized
  // bootstrap, so it is adopted only where its modeled area beats the
  // incumbent's (or none was found), a tie going to the shorter span.
  bool foldedArea = false;
  bool hasAreaDecision = allocates || !choices.empty() || !span.regs.empty() ||
                         span.device.pulsePrice();
  if (bySpan && hasAreaDecision &&
      (opts.areaSlack > 0.0 || !bestAttempt.areaProven)) {
    // The proven span, fixed: the leash and the tight-bound test below are cut
    // from it, not from the incumbent `best`, which grows as the fold adopts a
    // wider interval.
    int64_t strictSpan = *best;
    int64_t leash =
        strictSpan + static_cast<int64_t>(strictSpan * opts.areaSlack);
    // The interval envelope: the winning interval at slack zero, else every
    // interval whose II term alone fits the leash, an explicit pipeline(ii=n)
    // directive capping it and the natural floor flooring it.
    unsigned foldLo = bestII, foldHi = bestII;
    if (opts.areaSlack > 0.0) {
      foldLo = warm.lowerBoundII;
      foldHi = iiWeight > 0
                   ? std::max<unsigned>(
                         bestII,
                         static_cast<unsigned>((leash - floorDrain) / iiWeight))
                   : bestII;
      if (maxII)
        foldHi = std::min(foldHi, std::max(maxII, warm.lowerBoundII));
    }
    // The interval whose schedule currently sits in `prob`, the only one the
    // incumbent's start times hint validly.
    unsigned incumbentII = bestII;
    std::optional<int64_t> foldArea = bestAttempt.modelArea;
    for (unsigned ii = foldLo; ii <= foldHi; ++ii) {
      if (iiWeight * ii + floorDrain > leash)
        continue;
      bool atIncumbent = ii == incumbentII;
      if (atIncumbent)
        for (Operation *op : ops)
          prob.setStartTime(op, bestAttempt.starts.at(op));
      // Where the strict span already admits this interval keep its own tight
      // bound, so the fold pays no register for drain the design will not use;
      // a wider interval spends the leash it opened.
      int64_t tight = strictSpan - iiWeight * ii;
      int64_t drainBound = tight >= floorDrain ? tight : leash - iiWeight * ii;
      ModuloAttempt folded;
      ModuloOutcome outcome =
          solveAtII(prob, lastOp, breaks, choices, sharedMeta, cycleTime, span,
                    opts, drainBound, floorDrain, opFloors,
                    opts.areaSlack > 0.0 ? foldArea : std::nullopt,
                    /*areaMode=*/true, ii, window + ii * contending,
                    /*hint=*/warm.placed && atIncumbent, folded);
      if (outcome != ModuloOutcome::Scheduled)
        continue;
      int64_t foldSpan = iiWeight * ii + folded.drain;
      if (!foldArea || *folded.modelArea < *foldArea ||
          (*folded.modelArea == *foldArea && foldSpan < *best)) {
        // The fold is leashed and floored like any schedule, so it cannot
        // regress the proven span past the leash; its own flag only says
        // whether the re-min under the pinned structure proved.
        folded.spanProven |= bestAttempt.spanProven;
        best = foldSpan;
        bestArea = *folded.modelArea;
        bestII = ii;
        incumbentII = ii;
        foldArea = folded.modelArea;
        bestAttempt = std::move(folded);
        foldedArea = true;
      }
    }
  }

  prob.setInitiationInterval(bestII);
  for (Operation *op : ops)
    prob.setStartTime(op, bestAttempt.starts.at(op));
  if (!choices.empty()) {
    applySelection(prob, choices, bestAttempt.chosen);
    applySharedClasses(prob, span.device, sharedMeta, choices,
                       bestAttempt.chosen, bestAttempt.classUnits,
                       bestAttempt.decided);
  }
  applyAllocation(prob, bestAttempt.decided, bestII);

  {
    auto d = info(Stage::Sched, prob.getContainingOp());
    d << "Exact scheduling placed the region at II=" << bestII;
    if (!warm.placed)
      d << ": the greedy placement could not place it at all";
    else if (bestII < greedyII)
      d << ", down from the heuristic's II=" << greedyII
        << ": the gap was greedy resource placement";
    else if (bestII > greedyII)
      d << ", up from the heuristic's II=" << greedyII
        << ": a wider interval bought area within the span leash";
    else
      d << ", the II the heuristic also reached";
    if (bySpan) {
      d << "; span " << *best;
      if (heuristicSpan)
        d << " against the heuristic's " << *heuristicSpan;
    }
    if (foldedArea)
      d << "; area " << bestArea;
  }
  // An exhausted budget leaves the primary objective's placement unproven,
  // and that placement is what the region is charged.
  if (foldedArea) {
    if (!bestAttempt.areaProven)
      warn(Stage::Sched, prob.getContainingOp())
          << "Exact scheduling ran out of budget placing the region at II="
          << bestII
          << " for area, so it shipped the best schedule it had found; it "
             "holds the span leash but is not known to be smallest";
    else if (!bestAttempt.spanProven)
      warn(Stage::Sched, prob.getContainingOp())
          << "Exact scheduling settled the region's area at II=" << bestII
          << " but ran out of budget shortening the span under it, so leash "
             "slack may remain unclaimed";
  } else if (!bestAttempt.spanProven) {
    warn(Stage::Sched, prob.getContainingOp())
        << "Exact scheduling ran out of budget placing the region at II="
        << bestII
        << ", so it shipped the best schedule it had found rather than the "
           "cheapest one; what it reached is no worse than the heuristic's but "
           "is not known to be minimal in span, registers or instances";
  } else if (!bestAttempt.areaProven) {
    warn(Stage::Sched, prob.getContainingOp())
        << "Exact scheduling settled the region's span at II=" << bestII
        << " but ran out of budget minimizing the area under it, so its "
           "registers and instances are not known to be fewest";
  }
  if (exhaustedAt)
    warn(Stage::Sched, prob.getContainingOp())
        << "Exact scheduling ran out of budget at II=" << *exhaustedAt
        << " without deciding it, so the search stopped there; what it kept is "
           "the best of the intervals it did decide";
  prob.telemetry.proven =
      bestAttempt.spanProven && bestAttempt.areaProven && !exhaustedAt;
  prob.telemetry.spanProven = bestAttempt.spanProven && !exhaustedAt;
  prob.telemetry.budgetExhausted = !prob.telemetry.proven;
  if (exhaustedAt)
    prob.telemetry.exhaustedAtII = (int64_t)*exhaustedAt;
  prob.telemetry.modelArea = bestAttempt.modelArea;
  prob.telemetry.modelAreaBound = bestAttempt.modelAreaBound;
  return finishSchedule(prob, cycleTime, opts.regFloor);
}

//===----------------------------------------------------------------------===//
// Exact operator sharing: one bind-time solve per region.
//===----------------------------------------------------------------------===//

/// Deterministic time budget for one region's sharing solve. Small next to a
/// schedule's: the model is a few booleans per same-class unit pair.
static constexpr double kSharingSolveBudget = 10.0;

std::optional<SmallVector<unsigned>>
mlir::allo::solveSharing(SharingProblem &problem, ArrayRef<unsigned> hint,
                         Operation *anchor) {
  auto n = static_cast<unsigned>(problem.units.size());
  llvm::DenseSet<uint64_t> collide;
  for (auto [a, b] : problem.conflicts)
    collide.insert(uint64_t(a) * n + b);
  // Who may fold onto whom: same class, no collision, onto a smaller index
  // only, so a group's representative is its first member.
  SmallVector<SmallVector<unsigned>> cands(n), joiners(n);
  SmallVector<unsigned> assign(n);
  bool foldable = false;
  for (unsigned i = 0; i < n; ++i) {
    assign[i] = i;
    for (unsigned j = 0; j < i; ++j)
      if (problem.units[i].cls == problem.units[j].cls &&
          !collide.contains(uint64_t(j) * n + i)) {
        cands[i].push_back(j);
        joiners[j].push_back(i);
        foldable = true;
      }
  }
  if (!foldable)
    return assign;

  CpModelBuilder model;
  SmallVector<BoolVar> rep(n);            // the unit keeps its own instance
  llvm::DenseMap<uint64_t, BoolVar> join; // j * n + i: unit i runs on unit j
  for (unsigned i = 0; i < n; ++i)
    rep[i] = model.NewBoolVar();
  for (unsigned i = 0; i < n; ++i) {
    SmallVector<BoolVar> choice{rep[i]};
    for (unsigned j : cands[i]) {
      BoolVar x = model.NewBoolVar();
      model.AddImplication(x, rep[j]);
      join[uint64_t(j) * n + i] = x;
      choice.push_back(x);
    }
    model.AddExactlyOne(choice);
  }
  auto lit = [&](unsigned i, unsigned j) {
    return i == j ? rep[i] : join.find(uint64_t(j) * n + i)->second;
  };
  // A colliding pair may not meet through a common representative either.
  for (auto [a, b] : problem.conflicts)
    for (unsigned j : cands[a])
      if (auto x = join.find(uint64_t(j) * n + b); x != join.end())
        model.AddAtMostOne({lit(a, j), x->second});

  // Per potential representative and operand port: the arms its select grew
  // (zero while it shares nothing), with the cone and price read off the port's
  // tables at that count. A port whose candidates all read one held value stays
  // a wire in every fold, which the emitter collapses, so it is skipped whole.
  int64_t horizon = 0;
  for (SharingProblem::Unit &u : problem.units)
    horizon = std::max(horizon, u.slackPicos);
  SmallVector<IntVar> arrive(n);
  for (unsigned j = 0; j < n; ++j)
    arrive[j] = model.NewIntVar(operations_research::Domain(0, horizon));
  llvm::DenseMap<std::pair<unsigned, unsigned>, IntVar> coneAt; // (host, port)
  // Area dominates; below it, fewer folds win ties, so a device that prices
  // everything at zero shares nothing.
  int64_t w = n + 1;
  LinearExpr objective;
  for (unsigned i = 0; i < n; ++i)
    objective += LinearExpr::Term(
        rep[i], problem.classes[problem.units[i].cls].instancePrice * w - 1);
  for (unsigned j = 0; j < n; ++j) {
    if (joiners[j].empty())
      continue;
    const SharingProblem::Unit &uj = problem.units[j];
    const SharingProblem::UnitClass &cls = problem.classes[uj.cls];
    BoolVar shared = model.NewBoolVar();
    SmallVector<BoolVar> in;
    for (unsigned i : joiners[j]) {
      in.push_back(lit(i, j));
      model.AddImplication(in.back(), shared);
    }
    model.AddBoolOr(in).OnlyEnforceIf(shared);
    for (unsigned p = 0, e = cls.ports.size(); p < e; ++p) {
      unsigned key = uj.drivers[p];
      if (key && llvm::all_of(joiners[j], [&](unsigned i) {
            return problem.units[i].drivers[p] == key;
          }))
        continue; // one held driver across every candidate: a wire
      int64_t maxArms = 1 + uj.initArms[p];
      LinearExpr arms = LinearExpr::Term(shared, 1 + uj.initArms[p]);
      for (unsigned i : joiners[j]) {
        unsigned add = 1 + problem.units[i].initArms[p];
        arms += LinearExpr::Term(lit(i, j), add);
        maxArms += add;
      }
      IntVar armCount =
          model.NewIntVar(operations_research::Domain(0, maxArms));
      model.AddEquality(armCount, arms);
      const SharingProblem::Port &port = cls.ports[p];
      std::vector<int64_t> cones(port.conePicos.begin(),
                                 port.conePicos.begin() + maxArms + 1);
      if (int64_t top = *llvm::max_element(cones)) {
        IntVar c = model.NewIntVar(operations_research::Domain(0, top));
        model.AddElement(armCount, cones, c);
        coneAt.try_emplace({j, p}, c);
        model.AddGreaterOrEqual(arrive[j], c);
      }
      std::vector<int64_t> prices(port.muxPrice.begin(),
                                  port.muxPrice.begin() + maxArms + 1);
      if (int64_t top = *llvm::max_element(prices)) {
        IntVar price = model.NewIntVar(operations_research::Domain(0, top));
        model.AddElement(armCount, prices, price);
        objective += LinearExpr::Term(price, w);
      }
    }
  }
  model.Minimize(objective);

  // The gate's recursion (`AddedDelay`), over bins instead of built sources:
  // a producer's cone arrives through the select of the port it drives, and
  // every member's slack must hold its whole bin's cone.
  for (unsigned y = 0; y < n; ++y)
    for (auto [port, p] : problem.units[y].preds) {
      SmallVector<unsigned> ys(cands[y]);
      ys.push_back(y);
      SmallVector<unsigned> ps(cands[p]);
      ps.push_back(p);
      for (unsigned jy : ys)
        for (unsigned jp : ps) {
          if (jy == jp)
            continue;
          LinearExpr reach = arrive[jp];
          if (auto c = coneAt.find({jy, port}); c != coneAt.end())
            reach += c->second;
          model.AddLessOrEqual(reach, arrive[jy])
              .OnlyEnforceIf({lit(y, jy), lit(p, jp)});
        }
    }
  for (unsigned i = 0; i < n; ++i) {
    model.AddLessOrEqual(arrive[i], problem.units[i].slackPicos)
        .OnlyEnforceIf(rep[i]);
    for (unsigned j : cands[i])
      model.AddLessOrEqual(arrive[j], problem.units[i].slackPicos)
          .OnlyEnforceIf(lit(i, j));
  }

  // The greedy plan seeds the search. Where its own cone test under-counted the
  // plan sits outside this model, which costs no more than the hint.
  for (unsigned i = 0; i < n; ++i) {
    model.AddHint(rep[i], hint[i] == i);
    if (hint[i] != i)
      model.AddHint(lit(i, hint[i]), true);
  }

  SchedulerOptions opts;
  opts.budget = kSharingSolveBudget;
  CpSolverResponse response = solveBuilt(model, solverParameters(opts));
  if (response.status() != CpSolverStatus::OPTIMAL &&
      response.status() != CpSolverStatus::FEASIBLE) {
    assert(response.status() != CpSolverStatus::INFEASIBLE &&
           response.status() != CpSolverStatus::MODEL_INVALID &&
           "every unit keeping its own instance satisfies this encoding, so "
           "the model is satisfiable by construction");
    warn(Stage::Emit, anchor)
        << "Exact sharing gave up after " << llvm::format("%g", opts.budget)
        << " deterministic time units (solver status "
        << CpSolverStatus_Name(response.status())
        << "); keeping the greedy plan";
    return std::nullopt;
  }
  unsigned folded = 0;
  for (unsigned i = 0; i < n; ++i)
    for (unsigned j : cands[i])
      if (SolutionBooleanValue(response, lit(i, j))) {
        assign[i] = j;
        ++folded;
        break;
      }
  if (response.status() != CpSolverStatus::OPTIMAL)
    warn(Stage::Emit, anchor)
        << "Exact sharing ran out of budget before proving this region's fold "
           "optimal; it shipped the best plan it had found";
  info(Stage::Emit, anchor)
      << "Exact sharing folded " << folded << " of " << n
      << " units away (spent "
      << llvm::format("%.3f", response.deterministic_time()) << " of "
      << llvm::format("%g", opts.budget) << " deterministic time units)";
  return assign;
}

//===----------------------------------------------------------------------===//
// Post-schedule register-lifetime repair.
//===----------------------------------------------------------------------===//

namespace {

/// Shared core of `repairRegisterLifetimes` (see Scheduler.h). Starts are
/// written as `sigma + M * lap` with `sigma` the solved start modulo \p M and
/// only the laps free, so a cyclic move can never change a congruence slot;
/// the acyclic case runs at M = 1 with sigma = 0. The system is dependences,
/// chain breaks, the drain and depth leashes, and the pins (all difference
/// constraints), under the linear width-weighted lifetime objective, which
/// CP-SAT settles at its LP root.
template <typename ProblemT>
void repairLifetimes(ProblemT &prob, Operation *anchor,
                     const SpanObjective &span, float cycleTime,
                     float regFloor) {
  int64_t pulse = span.device.pulsePrice();
  if (span.regs.empty() && !pulse)
    return;
  constexpr bool cyclic = std::is_same_v<ProblemT, ChainingModuloProblem>;
  int64_t modulus = 1;
  if constexpr (cyclic)
    modulus = static_cast<int64_t>(*prob.getInitiationInterval());

  const auto &ops = prob.getOperations();
  DenseMap<Operation *, int64_t> cur;
  int64_t depthCap = 1;
  for (Operation *op : ops) {
    int64_t t = static_cast<int64_t>(*prob.getStartTime(op));
    cur[op] = t;
    depthCap = std::max(depthCap, t + std::max<int64_t>(1, prob.latencyOf(op)));
  }
  int64_t curDrain = span.drainOf(prob);

  // The commit criterion: what a placement costs at the device's real chain
  // and pulse prices, folded onto the interval as the emitter builds it.
  auto price = [&](DenseMap<Operation *, int64_t> &at) {
    int64_t total = 0;
    for (const RegisterTerm &term : span.regs) {
      if (term.reads.empty())
        continue;
      int64_t end = at.lookup(term.def) + prob.latencyOf(term.def);
      int64_t depth = 0;
      for (auto [reader, dist] : term.reads)
        depth = std::max(depth, at.lookup(reader) + dist * modulus - end);
      total +=
          span.device.chainPrice((depth + modulus - 1) / modulus, term.width);
    }
    if (pulse) {
      int64_t deepest = 0;
      for (Operation *op : ops)
        deepest = std::max(deepest, at.lookup(op));
      total += pulse * deepest;
    }
    return total;
  };
  int64_t curPrice = price(cur);
  if (!curPrice)
    return;

  auto onAllocatable = [&](Operation *op) {
    auto linked = prob.getLinkedResourceTypes(op);
    if (!linked)
      return false;
    return llvm::any_of(*linked, [&](Problem::ResourceType rsrc) {
      return prob.getAllocatable(rsrc).has_value();
    });
  };

  CpModelBuilder model;
  DenseMap<Operation *, IntVar> laps;
  DenseMap<Operation *, int64_t> sigma;
  int64_t lapCap = depthCap / modulus + 1;
  for (Operation *op : ops) {
    int64_t s = cyclic ? cur[op] % modulus : 0;
    sigma[op] = s;
    bool pinned = op == anchor || onAllocatable(op) ||
                  (!cyclic && prob.holdsLimitedUnit(op));
    int64_t k0 = (cur[op] - s) / modulus;
    laps.try_emplace(op, model.NewIntVar(operations_research::Domain(
                             pinned ? k0 : 0, pinned ? k0 : lapCap)));
  }
  auto tOf = [&](Operation *op) {
    return LinearExpr::Term(laps.at(op), modulus) + sigma.lookup(op);
  };

  for (Operation *op : ops)
    for (auto &dep : prob.getDependences(op)) {
      Operation *src = dep.getSource();
      int64_t dist = 0;
      if constexpr (cyclic)
        dist = prob.getDistance(dep).value_or(0);
      int64_t w = prob.separationOf(dep);
      model.AddGreaterOrEqual(tOf(op) - tOf(src), w - modulus * dist);
    }
  for (const Problem::Dependence &dep :
       chainBreaksFor(prob, cycleTime, regFloor))
    model.AddGreaterOrEqual(tOf(dep.getDestination()) - tOf(dep.getSource()),
                            prob.latencyOf(dep.getSource()) + 1);
  for (const DrainTerm &term : span.drain) {
    int64_t off =
        term.offset + (term.plusLatency ? prob.latencyOf(term.op) : 0);
    model.AddLessOrEqual(tOf(term.op), curDrain - off);
  }
  for (Operation *op : ops)
    model.AddLessOrEqual(tOf(op),
                         depthCap - std::max<int64_t>(1, prob.latencyOf(op)));

  LinearExpr cost;
  for (const RegisterTerm &term : span.regs) {
    if (term.reads.empty())
      continue;
    int64_t maxDist = 0;
    for (auto [reader, dist] : term.reads)
      maxDist = std::max(maxDist, dist);
    IntVar last = model.NewIntVar(
        operations_research::Domain(0, depthCap + maxDist * modulus));
    for (auto [reader, dist] : term.reads)
      model.AddGreaterOrEqual(last, tOf(reader) + dist * modulus);
    cost += LinearExpr::Term(last, term.width) -
            LinearExpr::Term(laps.at(term.def), term.width * modulus);
  }
  if (pulse) {
    IntVar deepest = model.NewIntVar(operations_research::Domain(0, depthCap));
    for (Operation *op : ops)
      model.AddLessOrEqual(tOf(op), deepest);
    cost += LinearExpr::Term(deepest, pulse);
  }
  model.Minimize(cost);

  SatParameters params;
  params.set_num_workers(1);
  params.set_random_seed(0);
  params.set_max_deterministic_time(5.0);
  CpSolverResponse r = solveBuilt(model, params);
  if (r.status() != CpSolverStatus::OPTIMAL)
    return;

  DenseMap<Operation *, int64_t> moved;
  for (Operation *op : ops)
    moved[op] =
        sigma.lookup(op) + modulus * SolutionIntegerValue(r, laps.at(op));
  int64_t newPrice = price(moved);
  if (newPrice >= curPrice)
    return;
  for (Operation *op : ops)
    prob.setStartTime(op, static_cast<unsigned>(moved[op]));
  if (failed(finishSchedule(prob, cycleTime, regFloor))) {
    for (Operation *op : ops)
      prob.setStartTime(op, static_cast<unsigned>(cur[op]));
    (void)finishSchedule(prob, cycleTime, regFloor);
    return;
  }
  info(Stage::Sched, prob.getContainingOp())
      << "Lifetime repair re-placed the schedule's slack: modeled chain and "
         "pulse cost "
      << curPrice << " -> " << newPrice;
}

} // namespace

void mlir::allo::repairRegisterLifetimes(ChainingModuloProblem &prob,
                                         Operation *anchor,
                                         const SpanObjective &span,
                                         float cycleTime, float regFloor) {
  repairLifetimes(prob, anchor, span, cycleTime, regFloor);
}

void mlir::allo::repairRegisterLifetimes(ChainingSharedOperatorsProblem &prob,
                                         Operation *anchor,
                                         const SpanObjective &span,
                                         float cycleTime, float regFloor) {
  repairLifetimes(prob, anchor, span, cycleTime, regFloor);
}

namespace {

// The compile-time escalation oracle (`SchedulerOptions::escalate`): whether
// the heuristic's solved schedule provably leaves cycles behind. Either the
// modulo placement's gap warn survived the sigma/lap oracle, or the solved
// drain sits above the region's intra-iteration floor (a true lower bound,
// measured empirically exact bed-wide).
template <typename ProblemT>
bool scheduleGap(ProblemT &prob, const SpanObjective &span, float cycleTime,
                 float regFloor) {
  if (prob.telemetry.heuristicIIGap) {
    info(Stage::Sched, prob.getContainingOp())
        << "Escalating this region to the exact solver: the placement gap is "
           "not known to be necessary";
    return true;
  }
  if (span.drain.empty())
    return false;
  SmallVector<Problem::Dependence> breaks =
      chainBreaksFor(prob, cycleTime, regFloor);
  int64_t floor = drainFloor(prob, breaks, span.drain,
                             [&](Operation *op) { return prob.latencyOf(op); });
  int64_t have = span.drainOf(prob);
  if (have <= floor)
    return false;
  info(Stage::Sched, prob.getContainingOp())
      << "Escalating this region to the exact solver: the heuristic's drain "
      << have << " sits above the region's floor " << floor;
  return true;
}

} // namespace

bool mlir::allo::heuristicScheduleGap(ChainingModuloProblem &prob,
                                      const SpanObjective &span,
                                      float cycleTime, float regFloor) {
  return scheduleGap(prob, span, cycleTime, regFloor);
}

bool mlir::allo::heuristicScheduleGap(ChainingSharedOperatorsProblem &prob,
                                      const SpanObjective &span,
                                      float cycleTime, float regFloor) {
  return scheduleGap(prob, span, cycleTime, regFloor);
}
