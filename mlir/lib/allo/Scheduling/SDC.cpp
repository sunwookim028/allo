/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo-c/Schedule.h" // kPipelineIIAttr
#include "allo/IR/AlloOps.h"
#include "allo/Scheduling/AddressModel.h" // addressDelayOf
#include "allo/Scheduling/DependenceAnalysis.h"
#include "allo/Scheduling/LatencyModel.h"
#include "allo/Scheduling/MemoryAccess.h" // asMemAccess
#include "allo/Scheduling/MemoryModel.h"  // kIndexWidth
#include "allo/Scheduling/OperatorLibrary.h"
#include "allo/Scheduling/ProblemBuilder.h"
#include "allo/Scheduling/RegionGraph.h"
#include "allo/Scheduling/ScheduleModel.h"
#include "allo/Scheduling/Scheduler.h"
#include "allo/Support/Logging.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MathExtras.h"

#include <chrono>

using llvm::format;
using namespace mlir;
using namespace mlir::affine;
using namespace mlir::allo;
using namespace mlir::allo::dcp;
using namespace mlir::allo::logging;

// The maximal perfect band of counted loops (affine.for / scf.for) rooted at
// \p root: descend while a level's body is exactly { inner counted loop,
// terminator }. Returns [root, ..., innermost].
//
// Must run after `expand-region-bounds`, which places an inner loop's runtime
// bound arithmetic beside it, breaking the band there deliberately. The
// `allo.volatile` marker carrying that bound is not work, so it is stepped over
// and a loop whose bound map is trivial keeps its band.
static SmallVector<LoopLikeOpInterface> perfectNest(LoopLikeOpInterface root) {
  SmallVector<LoopLikeOpInterface> nest{root};
  while (true) {
    Block &body = nest.back().getLoopRegions().front()->front();
    Operation *first = &body.front();
    while (isa<VolatileOp>(first))
      first = first->getNextNode();
    if (first->getNextNode() != body.getTerminator())
      break; // the body holds more than just the inner loop
    auto inner = dyn_cast<LoopLikeOpInterface>(first);
    if (!inner || !isa<AffineForOp, scf::ForOp>(first))
      break;
    nest.push_back(inner);
  }
  return nest;
}

// The region's outputs, as the terms whose max is its terminal cycle: how long
// after its last issue pulse the deepest output commits. Kept as separate
// terms so the exact scheduler can bound a variable by each one and minimize
// the charged quantity; `drainOf` takes the max after the solve.
//
// Each output is charged at its commit cycle: a store commits `writeLatency`
// cycles after its start, a sync sub-kernel call the same way (its `done`
// rises at its start plus its contract), a stream put commits at its stage,
// and a value handed onward is latched the cycle it lands, one cycle above a
// store presented at the same depth.
//
// \p results are the values escaping the region. One only forwarded (a block
// argument, an earlier region's survivor, or a declaration) charges nothing:
// it is settled before the region starts or binds no hardware to wait on.
//
// An INDETERMINATE call is charged a floor of one cycle: it has no contract
// to place its `done` against, so the operator model gives it latency zero
// and the only static fact left is that it occupies the cycle it issues in.
static SmallVector<DrainTerm> drainTerms(OccupancyProblem &problem,
                                         ValueRange results) {
  SmallVector<DrainTerm> terms;
  for (Operation *op : problem.getOperations()) {
    if (isa<AffineStoreOp, memref::StoreOp>(op) || isSyncSubKernelCall(op))
      terms.push_back({op, std::max<int64_t>(problem.latencyOf(op), 1) - 1});
    else if (isa<StreamPutOp>(op))
      terms.push_back({op, 0});
  }
  for (Value v : results) {
    Operation *def = v.getDefiningOp();
    // A call's result is the one escaping value not read through a capture
    // register of this region: the region's `done` is the child's, charged by
    // the loop above, and the consumer's own arming cycle pays the latch.
    if (!def || isDeclarationOp(def) || isSyncSubKernelCall(def) ||
        !problem.hasOperation(def))
      continue;
    // The definer's latency is read live (`plusLatency`): which row realizes
    // it may itself be an exact solve's decision.
    terms.push_back({def, 0, /*plusLatency=*/true});
  }
  return terms;
}

// The cycle count one iteration paces by when iterations do not overlap: the
// schedule depth, with the anchor's cycle re-derived so a stream put commits at
// its stage (the rule `drainTerms` applies), since a put's write latency lands
// in the FIFO's own register. Every other anchor charge stands: a store's write
// must land, and a call's `done` needs its re-arm cycle.
static int64_t pacedDepth(ChainingModuloProblem &problem, Operation *anchor) {
  int64_t depth = 1;
  for (Operation *op : problem.getOperations()) {
    if (op == anchor)
      continue;
    if (std::optional<unsigned> start = problem.getStartTime(op))
      depth = std::max(depth, static_cast<int64_t>(*start) +
                                  std::max<int64_t>(1, problem.latencyOf(op)));
  }
  int64_t anchorAt = 0;
  for (auto &dep : problem.getDependences(anchor)) {
    if (problem.getDistance(dep).value_or(0) != 0)
      continue;
    Operation *src = dep.getSource();
    int64_t commit = isa<StreamPutOp>(src) ? 0 : problem.latencyOf(src);
    anchorAt = std::max(
        anchorAt, static_cast<int64_t>(*problem.getStartTime(src)) + commit);
  }
  return std::max(depth, anchorAt + 1);
}

// The flip-flops one cycle of delay on \p type costs, or 0 for a value not
// carried in a register at all (a memref, a stream). An index is charged at
// `kIndexWidth`, an upper bound since the emitter may build that address
// register narrower; charging it zero would let the solver lengthen an address
// chain for free.
static int64_t registerWidth(Type type) {
  if (auto i = dyn_cast<IntegerType>(type))
    return i.getWidth();
  if (auto f = dyn_cast<FloatType>(type))
    return f.getWidth();
  if (isa<IndexType>(type))
    return kIndexWidth;
  return 0;
}

// The values a region spends a delay register on, and what each one charges:
// mirrors `DatapathBuilder::resolveOperand` + `insertRegister`, so a solve
// minimizes the same quantity the emitter spends.
//
// Two kinds are charged: a scheduled producer read in the same region (a
// def-use edge), and a loop-carried read of an iter_arg (the same edge
// `distance` iterations back). A value held longer than the region (a
// survivor, an IO port, a literal) is defined by no op in the problem and is
// free. An enclosing loop's counter and the activation-pulse chain are not
// charged here, both left to the objective's sum-of-starts tie-break.
//
// \p carried is the counted-loop body whose block arguments after the
// induction variable are its iter_args, or null where there is no such
// recurrence to price (a straight-line span, a `while`).
static SmallVector<RegisterTerm> registerTerms(OccupancyProblem &problem,
                                               Block *carried) {
  SmallVector<RegisterTerm> terms;
  DenseMap<Value, unsigned> slotOf;
  auto readBy = [&](Value v, Operation *def, Operation *reader,
                    int64_t distance) {
    int64_t width = registerWidth(v.getType());
    if (width == 0)
      return;
    auto [slot, isNew] = slotOf.try_emplace(v, terms.size());
    if (isNew)
      terms.push_back({def, width, {}});
    terms[slot->second].reads.push_back({reader, distance});
  };

  for (Operation *reader : problem.getOperations()) {
    // A terminator takes no input register: the values it hands on are latched
    // by the region's completion, not delayed into it.
    if (reader->hasTrait<OpTrait::IsTerminator>())
      continue;
    for (auto &dep : problem.getDependences(reader))
      if (dep.isDefUse())
        readBy(dep.getSource()->getResult(*dep.getSourceIndex()),
               dep.getSource(), reader, /*distance=*/0);
  }

  if (!carried)
    return terms;
  Operation *yield = carried->getTerminator();
  for (unsigned i = 0, n = yield->getNumOperands(); i < n; ++i) {
    auto [def, distance] = iterArgSource(carried, yield, i);
    if (!def || !problem.hasOperation(def))
      continue;
    for (Operation *reader : carried->getArgument(i + 1).getUsers())
      if (problem.hasOperation(reader))
        readBy(def->getResult(0), def, reader, distance);
  }
  return terms;
}

SpanObjective::SpanObjective(OccupancyProblem &problem, ValueRange results,
                             Block *carried, std::optional<int64_t> trip,
                             const OperatorLibrary &device)
    : drain(drainTerms(problem, results)),
      regs(registerTerms(problem, carried)), trip(trip), device(device) {}

// Whether the problem carries a loop-carried recurrence (a dependence spanning
// >= 1 iteration), which can hold the modulo II above the resource bound.
static bool hasCarriedRecurrence(circt::scheduling::CyclicProblem &problem) {
  for (Operation *op : problem.getOperations())
    for (auto dep : problem.getDependences(op))
      if (problem.getDistance(dep).value_or(0) > 0)
        return true;
  return false;
}

// The values a straight-line span hands to something outside itself. Must match
// what the reify treats as escaping, so the two agree on what the region's
// completion waits to capture. A boundary value is one of them, its `volatile`
// marker sitting beside the anchor rather than in the span.
static SmallVector<Value> spanEscapingValues(ArrayRef<Operation *> ops) {
  llvm::SmallPtrSet<Operation *, 16> inSpan(ops.begin(), ops.end());
  SmallVector<Value> escaping;
  for (Operation *op : ops) {
    // A literal is hoisted out rather than yielded, so the span waits for
    // nothing on its account.
    if (isa<arith::ConstantOp>(op))
      continue;
    for (Value res : op->getResults())
      if (llvm::any_of(res.getUsers(),
                       [&](Operation *user) { return !inSpan.contains(user); }))
        escaping.push_back(res);
  }
  return escaping;
}

// A steady-clock stopwatch for timing one solve.
using Stopwatch = std::chrono::steady_clock::time_point;
static Stopwatch now() { return std::chrono::steady_clock::now(); }

namespace {

/// What the area objective's slack pass collects from the heuristic
/// pre-schedule: leash widenings for regions the kernel's composition proved
/// off its longest path (`grants`, keyed by the region's solve key), and
/// in-region float on a sync call, banked for the callee's own regions
/// (`calleeBudget`, single-site callees only).
struct SlackLedger {
  DenseMap<Operation *, int64_t> grants;
  DenseMap<Operation *, int64_t> calleeBudget;
  /// Sync call sites per callee, module-wide: a budget is only safe when one
  /// site holds the whole float.
  DenseMap<Operation *, unsigned> callSites;
};

/// Where a composition-slack grant on \p region lands, and what one granted
/// cycle costs there: the solve key of the region's own schedule, and the trip
/// product of the counted wrappers between the region's span and that schedule
/// (a container re-runs its body per iteration, so widening the body by one
/// widens the region by the product). nullopt where the region holds no
/// interval a grant could widen: a straight-line span, a container decomposed
/// into sub-regions, a call node, a while.
std::optional<std::pair<Operation *, int64_t>>
grantTarget(const SchedRegion &region, DependenceAnalysis &deps) {
  if (region.kind != allo::RegionKind::Loop)
    return std::nullopt;
  if (!isa<AffineForOp, scf::ForOp>(region.anchor()))
    return std::nullopt;
  SmallVector<LoopLikeOpInterface> band =
      perfectNest(cast<LoopLikeOpInterface>(region.anchor()));
  LoopLikeOpInterface innermost = band.back();
  if (countedLoopShape(innermost) != RegionShape::Leaf)
    return std::nullopt;
  int64_t divisor = 1;
  for (LoopLikeOpInterface level : ArrayRef(band).drop_back()) {
    std::optional<int64_t> t = deps.tripOf(level.getOperation()).count;
    if (!t || *t <= 0)
      return std::nullopt;
    divisor *= *t;
  }
  return std::pair{innermost.getOperation(), divisor};
}

/// Solves ONE function's schedule. Holds the analysis, device, model and
/// options every method needs, instead of threading them through each
/// signature.
///
/// One instance per function, no longer lived than the `DependenceAnalysis` it
/// is handed: the span composition reads that analysis after the solve.
class FuncScheduler {
public:
  FuncScheduler(DependenceAnalysis &deps, const DeviceModel &dev,
                ScheduleModel &model, float cycleTime,
                const SchedulerOptions &opts, SlackLedger *ledger = nullptr,
                const DenseMap<Operation *, int64_t> *grants = nullptr)
      : deps(deps), dev(dev), model(model), cycleTime(cycleTime), opts(opts),
        ledger(ledger), grants(grants) {
    assert(!(ledger && grants) &&
           "one pass collects slack, the other consumes it");
  }

  /// Consume this function's assumption hints, solve its regions, and publish
  /// what the whole kernel costs.
  LogicalResult run(func::FuncOp funcOp);

private:
  // The hints the analysis has already distilled, erased before any problem is
  // built.
  void eraseHint(RewriterBase &b, Operation *op);
  void consumeHints(func::FuncOp funcOp);

  // The expressions a region's BOUNDARY is, turned into operations the solve
  // can see and cut.

  // The IR walk: a block into regions, a region onto the problem that fits it.
  LogicalResult scheduleBlock(Block &block);
  LogicalResult scheduleRegion(const SchedRegion &region);

  // One region, one solve.
  LogicalResult scheduleCyclic(LoopLikeOpInterface body,
                               const SchedRegion &region, unsigned minII,
                               unsigned maxII, bool pipelined);
  LogicalResult scheduleWhile(scf::WhileOp w, const SchedRegion &region);
  LogicalResult scheduleWhileCondition(scf::WhileOp w);
  LogicalResult scheduleAcyclic(ArrayRef<Operation *> ops, bool ownsRegion);

  // What a solve leaves behind: the schedule, the allocation, the measurement.
  // A region's own `RegionSolution` is the caller's: it is what that path
  // decided, and one path decides none (see `scheduleAcyclic`).
  void annotateStarts(circt::scheduling::ChainingProblem &problem);
  void annotateAllocation(OccupancyProblem &problem);
  int64_t regionArea(OccupancyProblem &problem, const SpanObjective &span,
                     int64_t ii);
  void recordSolve(OccupancyProblem &problem, StringRef kind,
                   std::optional<unsigned> ii, Stopwatch since);

  // The second walk: the solved tree composed into one kernel span.
  std::optional<SpanNode> buildSpanNode(const SchedRegion &region);
  std::vector<SpanNode> buildSpanNodes(Block &body);
  void recordTripBounds(func::FuncOp funcOp);
  void publishKernelLatency(func::FuncOp funcOp);

  // The slack pass (area objective, pass 1 of two): what this func's
  // composition leaves free, banked into `ledger`.
  void collectCallSlack(ChainingSharedOperatorsProblem &problem,
                        ArrayRef<Operation *> ops);
  void collectSiblingSlack(ArrayRef<SpanNode> nodes,
                           ArrayRef<SchedRegion> regions,
                           ArrayRef<SmallVector<unsigned, 2>> preds);

  DependenceAnalysis &deps;
  const DeviceModel &dev;
  ScheduleModel &model;
  float cycleTime;
  const SchedulerOptions &opts;
  /// Pass 1: collect slack here. Pass 2: consume `grants`. Never both.
  SlackLedger *ledger;
  const DenseMap<Operation *, int64_t> *grants;
};

} // namespace

/// Record the solved schedule in \p model: every registered op's start cycle
/// and sub-cycle start, and any realization an exact solve moved off the
/// library's own pick. The decision travels as the op's linked operator type
/// (an IP row's type name is its symbol), so a linked name that differs from
/// what `lookup` resolves is the solver's selection.
void FuncScheduler::annotateStarts(
    circt::scheduling::ChainingProblem &problem) {
  for (Operation *op : problem.getOperations()) {
    std::optional<unsigned> start = problem.getStartTime(op);
    if (!start)
      continue;
    // A child loop is scheduled as its own region and a terminator carries no
    // compute. Neither is recorded, though both count toward the length.
    if (isa<AffineForOp, scf::ForOp, scf::WhileOp>(op) ||
        op->hasTrait<OpTrait::IsTerminator>())
      continue;
    model.setStart(op, *start);
    if (std::optional<float> z = problem.getStartTimeInCycle(op))
      model.setStartInCycle(op, *z);
    if (usesExactScheduler(opts.kind) && !isSyncSubKernelCall(op) &&
        !asMemAccess(op)) {
      StringRef linked = problem.getLinkedOperatorType(op)->getValue();
      if (linked != dev.operators.lookup(op).timing.typeName)
        model.setSelectedImpl(op, linked);
    }
  }
}

// Publish the solved allocation into \p model: one entry per instance the
// region builds, and the instance each operation runs on. Every operation on an
// allocated resource carries one: `applyAllocation` derives them alongside the
// counts it sets, and `verifyAllocation` has already failed the solve where one
// is missing.
void FuncScheduler::annotateAllocation(OccupancyProblem &problem) {
  for (circt::scheduling::Problem::ResourceType rsrc :
       problem.getResourceTypes()) {
    std::optional<unsigned> units = problem.getAllocation(rsrc);
    if (!units)
      continue;
    SmallVector<Operation *> users = problem.usersOf(rsrc);
    assert(!users.empty() && "an allocated resource nothing runs on");
    // One resource is one operator identity, so every operation on it names
    // the same `dcp.operator`. A member whose row the solve decided names it
    // through the recorded selection, not the library's own pick.
    Operation *first = users.front();
    const OpSchedule *at = model.scheduleOf(first);
    OperatorChar oc = at && !at->selectedImpl.empty()
                          ? dev.operators.lookup(first, at->selectedImpl)
                          : dev.operators.lookup(first);
    unsigned base = model.addUnits(oc.identity.ipSymbol, *units);
    for (Operation *op : users)
      model.setUnit(op, base + *problem.getAssignedUnit(op));
  }
}

// What one solved region costs in the device's currency: the area objective's
// own terms (`areaTerms`) evaluated on a settled schedule instead of built as
// an expression, plus the rows that objective drops as a within-period
// constant. Two probes of one kernel at different periods legalize to
// different operations on different rows, so every realized operation is
// priced here, not only the ones an allocation decides.
//
// \p ii is the interval the region runs at, which is what the emitter folds a
// delay chain onto; zero for a straight-line span.
int64_t FuncScheduler::regionArea(OccupancyProblem &problem,
                                  const SpanObjective &span, int64_t ii) {
  using circt::scheduling::Problem;
  const OperatorLibrary &lib = dev.operators;
  unsigned interval = static_cast<unsigned>(std::max<int64_t>(ii, 0));
  int64_t area = 0;
  // Instances and the muxes in front of them, at the count the solve decided.
  // A solve that decided none leaves the schedule's own demand: the busiest
  // congruence class, the floor sharing could reach at this interval.
  llvm::SmallPtrSet<Operation *, 32> shared;
  for (Problem::ResourceType rsrc : problem.getResourceTypes()) {
    std::optional<OccupancyProblem::AllocatableUnit> unit =
        problem.getAllocatable(rsrc);
    if (!unit)
      continue;
    SmallVector<Operation *> users = problem.usersOf(rsrc);
    shared.insert(users.begin(), users.end());
    unsigned units =
        problem.getAllocation(rsrc).value_or(problem.demandFor(rsrc, interval));
    assert(units <= unit->ceiling && "an allocation builds no more than one "
                                     "instance per operation");
    area += unit->price[units];
  }
  // Everything else costs one instance of the row it was realized on.
  for (Operation *op : problem.getOperations()) {
    if (shared.contains(op) || isSyncSubKernelCall(op) || asMemAccess(op))
      continue;
    const OpSchedule *at = model.scheduleOf(op);
    area += (at && !at->selectedImpl.empty() ? lib.lookup(op, at->selectedImpl)
                                             : lib.lookup(op))
                .price;
  }
  // The delay chain each value crosses its slack on, folded onto the region's
  // phase at II > 1 exactly as the emitter builds it.
  int64_t fold = std::max<int64_t>(ii, 1);
  for (const RegisterTerm &term : span.regs) {
    int64_t end = static_cast<int64_t>(*problem.getStartTime(term.def)) +
                  problem.latencyOf(term.def);
    int64_t depth = 0;
    for (auto [reader, distance] : term.reads)
      depth =
          std::max(depth, static_cast<int64_t>(*problem.getStartTime(reader)) +
                              distance * ii - end);
    area += lib.chainPrice(llvm::divideCeil(depth, fold), term.width);
  }
  // One activation pulse chain, as deep as the deepest start rides it.
  if (int64_t pulse = lib.pulsePrice()) {
    int64_t deepest = 0;
    for (Operation *op : problem.getOperations())
      if (std::optional<unsigned> t = problem.getStartTime(op))
        deepest = std::max(deepest, static_cast<int64_t>(*t));
    area += deepest * pulse;
  }
  return area;
}

// The pipeline directive on the loop (or an enclosing loop up to the region
// anchor), from `s.pipeline(ii=N)` -> `allo.pipeline.ii`:
//   >= 1  requested target II: a lower bound on the achieved II
//    0    auto: minimize the II (same as no directive)
//   -1    pipelining disabled: schedule the loop non-pipelined
// Absent => 0 (auto). The directive may sit on any level of a perfect nest.
static int64_t pipelineDirective(Operation *loop, Operation *anchor) {
  for (Operation *op = loop;; op = op->getParentOp()) {
    if (auto attr = op->getAttrOfType<IntegerAttr>(kPipelineIIAttr))
      return attr.getInt();
    if (op == anchor || !op->getParentOp())
      return 0;
  }
}

//===----------------------------------------------------------------------===//
// Store->load forwarding: relaxing the RAW round trip through storage.
//
// A store commits `writeLatency` cycles after it issues, so a RAW edge holds a
// dependent load that far behind it and a memory recurrence pins the II at the
// full storage round trip. A shadow register pair (the store's address compared
// against the load's at issue, the select and the store's datum registered to
// the read latency, a 2:1 mux at the load's data out) serves the one case the
// RAM cannot: the two issuing in the same cycle. The RAW edge then needs only
// issue order, latency zero; the WAR/WAW edges stay, and they exclude every
// collision the shadow must not serve.
//===----------------------------------------------------------------------===//

// The compare of the forward select: the two element addresses, at the
// address width the array carries. Marginal over the register floor, like
// every comb row.
static double forwardCmpDelay(const OperatorLibrary &lib, Value root) {
  auto shape = cast<MemRefType>(root.getType()).getShape();
  int64_t words = 1;
  for (int64_t s : shape)
    words *= std::max<int64_t>(1, s);
  int64_t width = std::max<int64_t>(1, llvm::Log2_64_Ceil(words));
  if (std::optional<double> d = lib.measuredCombDelay(OpKind::Cmp, width))
    return *d;
  return lib.measuredCombDelay(OpKind::Cmp, 32).value_or(0.0);
}

// The auxiliary store->load edges of \p problem a forwarding network could
// serve: both endpoints access one addressed, unskewed array whose write
// commits in one cycle and whose read is registered (the shadow select and
// datum ride that register), and both sit in one block, so they share a region
// and its stall shell. The select's compare (each address cone plus one
// equality, into the select register) must fit the period; the data mux is
// priced in `relaxForwardableEdges`, where the arm count is known.
static SmallVector<circt::scheduling::Problem::Dependence>
forwardableEdges(ChainingModuloProblem &problem, const DeviceModel &dev,
                 float cycleTime, float regFloor) {
  using Dependence = circt::scheduling::Problem::Dependence;
  SmallVector<Dependence> out;
  for (Operation *op : problem.getOperations()) {
    std::optional<MemAccess> load = asMemAccess(op);
    if (!load || load->isWrite || load->kind != AccessKind::Array)
      continue;
    MemoryChar ch = characterize(load->root, dev.memory);
    if (ch.unlimited() || ch.layout.skew())
      continue;
    MemKindTiming timing = dev.memory.timing(ch.storage);
    if (timing.latency.read < 1 || timing.latency.write != 1)
      continue;
    double cmp = forwardCmpDelay(dev.operators, load->root);
    if (regFloor + addressDelayOf(op, dev.operators) + cmp > cycleTime)
      continue;
    for (auto &dep : problem.getDependences(op)) {
      if (!dep.isAuxiliary())
        continue;
      Operation *src = dep.getSource();
      std::optional<MemAccess> store = asMemAccess(src);
      if (!store || !store->isWrite || store->root != load->root ||
          src->getBlock() != op->getBlock())
        continue;
      if (regFloor + addressDelayOf(src, dev.operators) + cmp > cycleTime)
        continue;
      out.push_back(dep);
    }
  }
  return out;
}

// The recurrence graph of a modulo problem: one node per operation, one edge
// per dependence, weighted by its source's latency and its iteration distance.
// Both II questions below are positive-circuit searches over it, and both
// weigh a forwarded edge at zero, so they share the weights by construction.
namespace {
struct RecurrenceGraph {
  using Dependence = circt::scheduling::Problem::Dependence;
  struct Edge {
    unsigned src, dst;
    int64_t lat, dist;
    Dependence dep;
  };

  DenseMap<Operation *, unsigned> index;
  SmallVector<Edge> edges;
  /// Edges weighed at `-window` latency (0 when no window map names them);
  /// `latSum` is taken over the set this was built with, and the walk below
  /// reads it as it stands.
  llvm::DenseSet<Dependence> zeroed;
  const llvm::DenseMap<Dependence, unsigned> *windows = nullptr;
  /// An II no circuit can exceed: the entry cut of every search below.
  int64_t latSum = 1;
  // The last relaxation's state, which the circuit walk reads.
  SmallVector<int64_t> dist;
  SmallVector<int> pred;
  int lastMoved = -1;

  RecurrenceGraph(ChainingModuloProblem &problem,
                  const llvm::DenseSet<Dependence> &relaxed,
                  const llvm::DenseMap<Dependence, unsigned> *windows = nullptr)
      : zeroed(relaxed), windows(windows) {
    for (Operation *op : problem.getOperations())
      index.try_emplace(op, index.size());
    for (Operation *op : problem.getOperations())
      for (auto &dep : problem.getDependences(op)) {
        int64_t lat = problem.latencyOf(dep.getSource());
        edges.push_back(
            {index[dep.getSource()], index[op], lat,
             static_cast<int64_t>(problem.getDistance(dep).value_or(0)), dep});
        latSum += zeroed.contains(dep) ? 0 : lat;
      }
  }

  int64_t weightOf(const Edge &e) const {
    if (!zeroed.contains(e.dep))
      return e.lat;
    return windows ? -static_cast<int64_t>(windows->lookup(e.dep)) : 0;
  }

  /// Whether the longest-path relaxation quiesces at \p ii, i.e. no positive
  /// circuit survives it.
  bool feasible(int64_t ii) {
    dist.assign(index.size(), 0);
    pred.assign(index.size(), -1);
    lastMoved = -1;
    for (unsigned round = 0; round <= index.size(); ++round) {
      bool moved = false;
      for (auto [i, e] : llvm::enumerate(edges)) {
        int64_t w = dist[e.src] + weightOf(e) - ii * e.dist;
        if (w > dist[e.dst]) {
          dist[e.dst] = w;
          pred[e.dst] = static_cast<int>(i);
          lastMoved = static_cast<int>(e.dst);
          moved = true;
        }
      }
      if (!moved)
        return true;
    }
    return false;
  }

  /// The smallest feasible II in [1, \p hi], binary searched.
  int64_t smallestFeasibleII(int64_t hi) {
    int64_t lo = 1;
    while (lo < hi) {
      int64_t mid = (lo + hi) / 2;
      if (feasible(mid))
        hi = mid;
      else
        lo = mid + 1;
    }
    return lo;
  }
};
} // namespace

// The smallest II no dependence circuit of \p problem excludes, with the edges
// in \p relaxed weighed at zero latency. Chain breaks are not built yet, so
// this is a floor, which is all the gate below compares.
static unsigned recurrenceMinII(
    ChainingModuloProblem &problem,
    const llvm::DenseSet<circt::scheduling::Problem::Dependence> &relaxed,
    const llvm::DenseMap<circt::scheduling::Problem::Dependence, unsigned>
        *windows = nullptr) {
  RecurrenceGraph graph(problem, relaxed, windows);
  // A zero-distance positive circuit is infeasible at every II; the solve will
  // fail and report it, so any answer here is moot.
  if (!graph.feasible(graph.latSum))
    return static_cast<unsigned>(graph.latSum);
  return static_cast<unsigned>(graph.smallestFeasibleII(graph.latSum));
}

// The resource-min II of \p problem:
// `ModuloSimplexScheduler::computeResMinII`'s arithmetic, asked before any
// scheduler exists.
static unsigned resourceMinII(ChainingModuloProblem &problem) {
  using P = circt::scheduling::Problem;
  unsigned resMinII = 1;
  DenseMap<P::ResourceType, unsigned> uses;
  for (Operation *op : problem.getOperations()) {
    auto rsrcs = problem.getLinkedResourceTypes(op);
    if (!rsrcs)
      continue;
    for (P::ResourceType rsrc : *rsrcs)
      if (problem.getLimit(rsrc).value_or(0) > 0)
        uses[rsrc] +=
            problem.getResourceCycles(op) * problem.getResourceDemand(op);
  }
  for (auto &[rsrc, demand] : uses) {
    unsigned limit = *problem.getLimit(rsrc);
    resMinII = std::max(resMinII, (demand + limit - 1) / limit);
  }
  return resMinII;
}

// One relaxation: the edges it forwarded, and each re-linked load's original
// operator type, so a failed solve can put everything back and run unrelaxed.
struct ForwardRelaxation {
  SmallVector<circt::scheduling::Problem::Dependence> edges;
  SmallVector<std::pair<Operation *, circt::scheduling::Problem::OperatorType>>
      originalTypes;
};

// The most pairs one problem relaxes outright. Every pair costs a compare, a
// select chain and a datum chain, and relaxing hundreds floods the modulo
// placement with same-cycle freedom the greedy search cannot place.
constexpr size_t kMaxForwardPairs = 16;

// The budget a body with more may-alias pairs than that spends on the circuits
// binding the II (`selectCriticalPairs`). The walk stops by itself once the
// recurrence no longer binds, so this is a backstop sized for an unrolled
// read-modify-write rather than a target.
constexpr size_t kMaxTargetedPairs = 64;

// The candidate pairs worth the budget: walk the circuit that binds the II,
// relax the candidate edges it carries, and repeat until the recurrence no
// longer binds, a binding circuit carries none (the bound cannot drop past
// that circuit whatever is forwarded), or the budget is spent.
static SmallVector<circt::scheduling::Problem::Dependence>
selectCriticalPairs(ChainingModuloProblem &problem,
                    ArrayRef<circt::scheduling::Problem::Dependence> cands,
                    unsigned floorII, size_t budget) {
  using Dependence = circt::scheduling::Problem::Dependence;
  RecurrenceGraph graph(problem, {});
  llvm::DenseSet<Dependence> candSet(cands.begin(), cands.end());
  if (!graph.feasible(graph.latSum))
    return {}; // a zero-distance positive circuit; the solve will report it
  int64_t hi = graph.latSum;
  while (graph.zeroed.size() < budget) {
    int64_t lo = graph.smallestFeasibleII(hi);
    if (lo <= floorII)
      break; // the recurrence no longer binds
    // One positive circuit at lo - 1: every node reached backward from the
    // last update has been updated itself, so the predecessor walk cannot
    // fall off and must revisit a node, closing the circuit.
    bool quiesced = graph.feasible(lo - 1);
    assert(!quiesced && graph.lastMoved >= 0 &&
           "the bound's own circuit is positive one step below it");
    (void)quiesced;
    SmallVector<unsigned> stamp(graph.index.size(), 0);
    int x = graph.lastMoved;
    while (!stamp[x]) {
      stamp[x] = 1;
      x = static_cast<int>(graph.edges[graph.pred[x]].src);
    }
    SmallVector<Dependence> take;
    int y = x;
    do {
      const RecurrenceGraph::Edge &e = graph.edges[graph.pred[y]];
      if (candSet.contains(e.dep) && !graph.zeroed.contains(e.dep))
        take.push_back(e.dep);
      y = static_cast<int>(e.src);
    } while (y != x);
    if (take.empty())
      break; // the bound runs through edges forwarding cannot serve
    for (Dependence dep : take) {
      if (graph.zeroed.size() >= budget)
        break;
      graph.zeroed.insert(dep);
    }
    hi = lo;
  }
  // In the candidates' own (deterministic) order.
  SmallVector<Dependence> out;
  for (Dependence dep : cands)
    if (graph.zeroed.contains(dep))
      out.push_back(dep);
  return out;
}

// Whether \p v is settled when a store issues: a loop-carried block argument,
// a region-invariant def, a constant, or a registered producer (latency >= 1).
// A value combinational in its issue cycle would chain its whole cone through
// a window arm's data mux into the load's consumers, a path the chain model
// does not price.
static bool registeredAtIssue(ChainingModuloProblem &problem, Value v,
                              Block *block) {
  if (isa<BlockArgument>(v))
    return true;
  Operation *def = v.getDefiningOp();
  if (!def || def->getBlock() != block || def->hasTrait<OpTrait::ConstantLike>())
    return true;
  auto opr = problem.getLinkedOperatorType(def);
  return opr && *problem.getLatency(*opr) >= 1;
}

// Relax the forwardable RAW edges of \p problem when, and only when, a storage
// recurrence binds the II and relaxing moves that bound; otherwise the
// schedule is unchanged and no shadow is built. Each forwarded load is
// re-linked onto a `.fwd` twin of its operator type whose outgoing delay
// carries the data mux (the RAM datum plus one arm per paired store); the
// compare ends in the select register and touches no port path, so nothing
// else is re-priced.
//
// A pair whose dependence distances are exact (polyhedral) and whose store
// data is settled at issue is granted a WINDOW of the read latency on top of
// the relaxation: the store may issue while the read is in flight, served by
// deeper shadow arms, taking the RAM round trip out of the recurrence
// entirely. Sound only when the solved II clears every window (an instance
// one iteration past the paired one must fall outside it), which the guard
// loop below enforces against the II floor. Returns the relaxed edges, empty
// when there are none.
static ForwardRelaxation relaxForwardableEdges(ChainingModuloProblem &problem,
                                               DependenceAnalysis &deps,
                                               const DeviceModel &dev,
                                               float cycleTime, float regFloor,
                                               unsigned minII) {
  using Dependence = circt::scheduling::Problem::Dependence;
  SmallVector<Dependence> cands =
      forwardableEdges(problem, dev, cycleTime, regFloor);
  if (cands.empty())
    return {};
  unsigned floorII = std::max(resourceMinII(problem), minII);
  unsigned recOrig = recurrenceMinII(problem, {});
  if (recOrig <= floorII)
    return {}; // the recurrence is not what binds the II
  if (cands.size() > kMaxForwardPairs) {
    cands = selectCriticalPairs(problem, cands, floorII, kMaxTargetedPairs);
    if (cands.empty())
      return {};
  }
  // The data-mux price per load, over its real arm count; a load whose bumped
  // output no longer fits the period drops its pairs (an operator must fit a
  // cycle of its own).
  llvm::MapVector<Operation *, unsigned> armsOf;
  for (Dependence dep : cands)
    ++armsOf[dep.getDestination()];
  llvm::DenseMap<Operation *, double> muxOf;
  for (auto &[load, arms] : armsOf) {
    double mux = muxCone(dev.operators, 1 + arms,
                         datapathWidth(load->getResult(0).getType()));
    NodeTiming t = accessCharacterization(load, dev.operators, dev.memory);
    if (t.outDelay + mux <= cycleTime)
      muxOf[load] = mux;
  }
  llvm::erase_if(cands, [&](Dependence dep) {
    return !muxOf.count(dep.getDestination());
  });
  if (cands.empty())
    return {};
  llvm::DenseSet<Dependence> relaxed(cands.begin(), cands.end());
  // Window grants: exact distances and settled store data (see the doc
  // comment above).
  llvm::DenseMap<Dependence, unsigned> windows;
  for (Dependence dep : cands) {
    Operation *store = dep.getSource(), *load = dep.getDestination();
    if (!deps.isExactPair(store, load))
      continue;
    Value data;
    if (auto st = dyn_cast<affine::AffineWriteOpInterface>(store))
      data = st.getValueToStore();
    else
      data = cast<memref::StoreOp>(store).getValueToStore();
    if (!registeredAtIssue(problem, data, store->getBlock()))
      continue;
    MemoryChar ch = characterize(asMemAccess(load)->root, dev.memory);
    unsigned rL = dev.memory.timing(ch.storage).latency.read;
    if (rL)
      windows[dep] = rL;
  }
  unsigned recRelaxed = recurrenceMinII(problem, relaxed, &windows);
  // A window is sound only when the solved II strictly clears it: an instance
  // one iteration past the paired one must fall outside it. The solver never
  // goes below max(floorII, the windowed recurrence floor), so shrink any
  // window that bound does not clear to the widest it does and re-settle.
  // Every step strictly shrinks a window, so this converges; shrinking only
  // raises the floor, which can only admit what already stands.
  while (!windows.empty()) {
    unsigned bound = std::max(floorII, recRelaxed);
    SmallVector<std::pair<Dependence, unsigned>> shrink;
    for (const auto &[dep, w] : windows)
      if (bound < w + 1)
        shrink.push_back({dep, bound - 1});
    if (shrink.empty())
      break;
    for (auto &[dep, w] : shrink) {
      if (w)
        windows[dep] = w;
      else
        windows.erase(dep);
    }
    recRelaxed = recurrenceMinII(problem, relaxed, &windows);
  }
  if (recRelaxed >= recOrig)
    return {}; // the bound runs through edges forwarding cannot serve
  info(Stage::Sched, problem.getContainingOp())
      << "Relaxing " << cands.size() << " store->load RAW edge(s) through a "
      << "forwarding shadow (" << windows.size()
      << " with an in-flight window): the recurrence floor drops from II="
      << recOrig << " to II=" << recRelaxed;
  ForwardRelaxation out;
  out.edges = std::move(cands);
  for (Dependence dep : out.edges)
    problem.setForwarded(dep, windows.lookup(dep));
  // In `armsOf`'s (insertion) order, so the operator types the twins mint are
  // created in a deterministic order.
  for (auto &[load, arms] : armsOf) {
    auto it = muxOf.find(load);
    if (it == muxOf.end())
      continue;
    auto opr = *problem.getLinkedOperatorType(load);
    // Keyed by the arm count too: two loads of one storage may fan differently.
    auto nw = problem.getOrInsertOperatorType(
        (opr.getValue() + ".fwd" + Twine(arms)).str());
    problem.setLatency(nw, *problem.getLatency(opr));
    problem.setIncomingDelay(nw, *problem.getIncomingDelay(opr));
    problem.setOutgoingDelay(nw, *problem.getOutgoingDelay(opr) + it->second);
    problem.setLinkedOperatorType(load, nw);
    out.originalTypes.push_back({load, opr});
  }
  return out;
}

// Put a relaxation back: the forwarded set cleared and every re-linked load
// returned to its original operator type, so a re-solve runs the unrelaxed
// problem.
static void undoForwardRelaxation(ChainingModuloProblem &problem,
                                  ForwardRelaxation &relax) {
  problem.clearForwarded();
  for (auto &[load, opr] : relax.originalTypes)
    problem.setLinkedOperatorType(load, opr);
  relax.edges.clear();
  relax.originalTypes.clear();
}

// Record the relaxed pairs the solved schedule collides on: the paired store
// instance issues inside the load's shadow window `[0, forwardWindow]`
// relative to the read issue, so the RAM alone would hand the load stale
// data. A pair whose instance lands before the read issue needs no shadow
// and none is built; nearer instances (below the pair's distance) never
// alias, which is what makes the offset of the PAIRED instance the only one
// to arm.
static void
recordForwards(ChainingModuloProblem &problem,
               ArrayRef<circt::scheduling::Problem::Dependence> edges,
               unsigned ii, ScheduleModel &model) {
  for (auto dep : edges) {
    Operation *store = dep.getSource(), *load = dep.getDestination();
    int64_t delta = static_cast<int64_t>(*problem.getStartTime(store)) -
                    static_cast<int64_t>(*problem.getStartTime(load));
    unsigned dist = problem.getDistance(dep).value_or(0);
    int64_t off = delta - static_cast<int64_t>(dist) * ii;
    assert(off <= static_cast<int64_t>(problem.forwardWindow(dep)) &&
           "a solved schedule respects the forwarded edge's window");
    if (off < 0)
      continue;
    model.addForward(load, store, off);
    info(Stage::Sched, load)
        << "Forwarding a store issued " << delta
        << " cycle(s) later into this load's data path (distance " << dist
        << ", II=" << ii << ", window offset " << off << ")";
  }
}

// Record what one region's solve cost, timed from \p since. Keyed by where the
// region is rather than by the op that owned it: the schedule report is built
// later off the reified dcp ops, by which time this problem's loop is gone.
//
// \p ii is what the solve decided, which for a non-pipelined loop is not the
// interval the region is reported to run at (that is `annotateRegion`'s).
void FuncScheduler::recordSolve(OccupancyProblem &problem, StringRef kind,
                                std::optional<unsigned> ii, Stopwatch since) {
  SolveReport s;
  Operation *containing = problem.getContainingOp();
  if (auto fn = containing->getParentOfType<func::FuncOp>())
    s.func = fn.getSymName().str();
  s.where = logging::detail::describe(containing);
  s.kind = kind.str();
  s.ops = (int64_t)problem.getOperations().size();
  for (Operation *op : problem.getOperations())
    if (problem.holdsLimitedUnit(op))
      ++s.limitedOps;
  if (ii)
    s.interval = (int64_t)*ii;
  s.millis = std::chrono::duration<double, std::milli>(now() - since).count();
  // Config from the options, outcome from the solver's telemetry, so a reader
  // can judge the result without re-solving.
  if (usesExactScheduler(opts.kind)) {
    s.solver = "cpsat";
    s.workers = opts.workers;
    s.seed = opts.seed;
    s.budgetSeconds = opts.budget;
    s.proven = problem.telemetry.proven;
    s.spanProven = problem.telemetry.spanProven;
    s.budgetExhausted = problem.telemetry.budgetExhausted;
    s.fallback = problem.telemetry.fallback;
    s.exhaustedAtII = problem.telemetry.exhaustedAtII;
    s.modelArea = problem.telemetry.modelArea;
    s.modelAreaBound = problem.telemetry.modelAreaBound;
    // One worker has nobody to race, so it stays reproducible under a held
    // budget regardless of the knob.
    s.deterministic = (opts.deterministic || opts.workers == 1) &&
                      !problem.telemetry.budgetExhausted;
  } else {
    s.solver = "simplex";
  }
  model.solves.push_back(std::move(s));
}

// Schedule one counted loop body (affine.for or scf.for) as a
// `ChainingModuloProblem` and annotate the result (start times, II, sub-cycle
// times). \p minII lower-bounds the II; \p maxII, nonzero, is an explicit
// directive's ceiling, honored by the exact area objective alone. When
// \p pipelined is false iterations do not overlap: the II is reported as the
// body length, so the region latency folds to `trip * depth`, and it still
// reifies to a dcp.pipeline.
LogicalResult FuncScheduler::scheduleCyclic(LoopLikeOpInterface body,
                                            const SchedRegion &region,
                                            unsigned minII, unsigned maxII,
                                            bool pipelined) {
  auto problem = buildCyclicProblem<ChainingModuloProblem>(body, deps);
  Block *bodyBlock = &body.getLoopRegions().front()->front();
  populateOperatorTypes(problem, dev.operators, dev.memory);
  // What contends, then how many of it to build: an occupancy window is a
  // physical property of the region and holds however the units are allocated.
  populateMemoryResources(problem, dev.memory);
  populateOperatorOccupancy(problem, dev.operators);
  populateCallOccupancy(problem);
  if (opts.allocate)
    populateOperatorAllocation(problem, dev.operators,
                               usesExactScheduler(opts.kind)
                                   ? AllocationScope::Static
                                   : AllocationScope::All,
                               opts.objective == ScheduleObjective::Area);
  // Overlapping iterations only: without overlap the RAW round trip costs
  // depth, not II, and a shadow would buy latency a mux is not worth.
  ForwardRelaxation relax;
  if (pipelined)
    relax = relaxForwardableEdges(problem, deps, dev, cycleTime, opts.regFloor,
                                  minII);
  Operation *anchor = bodyBlock->getTerminator();
  // The trip this solution records is the INNERMOST loop's, the one its solved
  // `length`/`ii` describe. Every level above drives its child as a container,
  // composed in `buildSpanNode`.
  LoopTrip trip = deps.tripOf(body.getOperation());
  // A counted loop hands its carried next-values on: the terminator's operands.
  // The trip is withheld where iterations do not overlap: `ii` is the body
  // depth there, so depth, not drain, is what the trip multiplies.
  SpanObjective span(problem, anchor->getOperands(), bodyBlock,
                     pipelined ? trip.count : std::nullopt, dev.operators);
  int64_t grant = grants ? grants->lookup(body.getOperation()) : 0;
  Stopwatch solveStart = now();
  if (failed(solveSchedulingProblem(problem, anchor, cycleTime, minII, opts,
                                    span, maxII, grant))) {
    if (relax.edges.empty())
      return failure();
    // The relaxed problem starts its II search lower, which can strand the
    // greedy placement where the unrelaxed search would not have gone. The
    // relaxation is an optimization, so put it back and solve without it.
    info(Stage::Sched, problem.getContainingOp())
        << "The relaxed problem did not place; retrying without the "
           "store->load forwarding relaxation";
    undoForwardRelaxation(problem, relax);
    if (failed(solveSchedulingProblem(problem, anchor, cycleTime, minII, opts,
                                      span, maxII, grant)))
      return failure();
  }
  std::optional<unsigned> solvedII = problem.getInitiationInterval();
  assert(solvedII && "a modulo problem that solved carries an interval");
  recordForwards(problem, relax.edges, *solvedII, model);
  recordSolve(problem, "cyclic", solvedII, solveStart);
  int64_t depth = pacedDepth(problem, anchor);
  // Iterations that do not overlap issue one body length apart, which is the
  // interval the region RUNS at whatever the solve settled on.
  unsigned ii = pipelined ? *solvedII : static_cast<unsigned>(depth);
  int64_t drain = span.drainOf(problem);
  // For the report only, through the arithmetic that composes it for real.
  SpanNode node;
  node.trip = trip.count;
  node.drain = drain;
  node.ii = ii;
  std::optional<int64_t> latency = composeSpan(node);

  {
    auto d = info(Stage::Sched, region.anchor());
    d << "Scheduled: II=" << ii;
    if (!pipelined)
      d << " (pipelining off, iterations run back-to-back)";
    else if (ii == 1)
      d << " (fully pipelined)";
    else if (hasCarriedRecurrence(problem))
      d << " (>1: a loop-carried recurrence and/or shared-resource limit)";
    else
      d << " (>1: a shared-resource limit, e.g. memory ports)";
    if (latency)
      d << ", latency = " << *latency
        << (trip.bounded ? " (assume-bounded worst case)" : "");
    else
      d << ", latency dynamic (trip not statically known)";
  }

  // A non-pipelined multi-cycle operator holds its unit for its whole latency,
  // so it caps iteration overlap. Name the dominant one to explain II > 1.
  if (pipelined && ii > 1) {
    Operation *blocking = nullptr;
    unsigned maxOcc = 1;
    for (Operation *op : problem.getOperations())
      if (unsigned occ = problem.getResourceCycles(op); occ > maxOcc) {
        maxOcc = occ;
        blocking = op;
      }
    if (blocking)
      info(Stage::Sched, blocking)
          << blocking->getName().getStringRef()
          << " is non-pipelined and holds its unit for " << maxOcc
          << " cycle(s), so no iteration may issue sooner";
  }

  annotateStarts(problem);
  // Every field is per-invocation: no composed total is stored.
  RegionSolution &sol = model.addRegion(body.getOperation());
  sol.ii = ii;
  sol.length = depth;
  sol.drain = drain;
  sol.trip = trip.count;
  sol.tripIsBound = trip.bounded;
  annotateAllocation(problem);
  model.modeledArea += regionArea(problem, span, ii);
  return success();
}

// Schedule an uncounted `scf.while` (before + after as one iteration) as a
// `ChainingModuloProblem`, the flushing-pipeline scheduling view. Its trip
// count is data-dependent, so no latency is reported.
LogicalResult FuncScheduler::scheduleWhile(scf::WhileOp w,
                                           const SchedRegion &region) {
  auto problem = buildWhileProblem<ChainingModuloProblem>(w, deps);
  populateOperatorTypes(problem, dev.operators, dev.memory);
  populateMemoryResources(problem, dev.memory);
  // A flushing while issues an iteration per II like a pipeline: a
  // non-pipelined operator bounds its interval the same way. No call occupancy
  // is needed, since `whileFlushingPipelines` rejects a body with a sync call.
  populateOperatorOccupancy(problem, dev.operators);
  if (opts.allocate)
    populateOperatorAllocation(problem, dev.operators,
                               usesExactScheduler(opts.kind)
                                   ? AllocationScope::Static
                                   : AllocationScope::All,
                               opts.objective == ScheduleObjective::Area);
  Operation *anchor = w.getYieldOp().getOperation();
  // Honor a requested target II (>=1) as a lower bound. `ii=-1` (pipelining
  // off) is not modeled for while loops.
  int64_t dir = pipelineDirective(w, region.anchor());
  unsigned minII = dir >= 1 ? static_cast<unsigned>(dir) : 1;
  // A while's carried state is not priced as a register: its values are not a
  // counted loop's iter_args, so no body is passed in to read them. No trip
  // either, so the objective is just the anchor's start time.
  SpanObjective span(problem, anchor->getOperands(), /*carried=*/nullptr,
                     /*trip=*/std::nullopt, dev.operators);
  Stopwatch solveStart = now();
  if (failed(solveSchedulingProblem(problem, anchor, cycleTime, minII, opts,
                                    span)))
    return failure();
  std::optional<unsigned> ii = problem.getInitiationInterval();
  assert(ii && "a modulo problem that solved carries an interval");
  recordSolve(problem, "while", ii, solveStart);
  info(Stage::Sched, w.getOperation())
      << "  -> While loop scheduled as a flushing pipeline: II=" << *ii
      << " (trip is data-dependent, so whole-loop latency is unknown)";
  annotateStarts(problem);
  // The trip is data-dependent, so it stays empty and no span composes off this
  // drain: both are recorded, like `ii`, as what the solve decided.
  RegionSolution &sol = model.addRegion(w.getOperation());
  sol.ii = *ii;
  sol.length = problem.scheduleDepth();
  sol.drain = span.drainOf(problem);
  annotateAllocation(problem);
  model.modeledArea += regionArea(problem, span, *ii);
  return success();
}

// The operations a sequential while's continue-condition reads, in program
// order. The CHECK region emits arithmetic and loads only, so any other
// producer FAILS here, reported against the op itself; leaving the cone
// unscheduled would only move the report to the emitter.
//
// An EMPTY cone is normal: the condition is settled before the loop starts,
// so there is nothing to time. A value defined outside the before block
// bounds the walk (an iter-arg survivor, an enclosing counter, or a literal).
static FailureOr<SmallVector<Operation *>> conditionCone(scf::WhileOp w) {
  Block &before = w.getBefore().front();
  llvm::SmallPtrSet<Operation *, 8> reads;
  SmallVector<Value> work{w.getConditionOp().getCondition()};
  while (!work.empty()) {
    Operation *def = work.pop_back_val().getDefiningOp();
    if (!def || def->getBlock() != &before || isa<arith::ConstantOp>(def))
      continue;
    if (!reads.insert(def).second)
      continue;
    if (!isa<AffineLoadOp, memref::LoadOp>(def) &&
        !isa<arith::ArithDialect>(def->getDialect())) {
      unsupported(Stage::Sched, Code::PredicateNotCombinational, def)
          << "The continue-condition of this while reads '"
          << def->getName().getStringRef()
          << "', which the sequential CHECK region cannot evaluate; it emits "
             "arithmetic and array reads only. Compute the value in the loop "
             "body and test a carried variable instead";
      return failure();
    }
    for (Value o : def->getOperands())
      if (!isa<MemRefType>(
              o.getType())) // the memref names storage, not a value
        work.push_back(o);
  }
  SmallVector<Operation *> cone;
  for (Operation &op : before.without_terminator())
    if (reads.contains(&op))
      cone.push_back(&op);
  return cone;
}

// A sequential (CHECK/RUN) while's own condition cone, solved as its own
// straight-line span. Its depth is the `tCond` the controller waits out before
// deciding, which the emitter reads back off these start times
// (`emitConditionRegion`), so cutting it here is what holds it to the clock.
//
// The BODY is a separate problem: `scheduleRegion` decomposes the after block
// into sub-regions, and the two never overlap, CHECK deciding before RUN
// issues.
LogicalResult FuncScheduler::scheduleWhileCondition(scf::WhileOp w) {
  FailureOr<SmallVector<Operation *>> cone = conditionCone(w);
  if (failed(cone))
    return failure();
  if (cone->empty()) // settled before the loop: the check waits out nothing
    return success();
  info(Stage::Sched, w.getOperation())
      << "Scheduling the while's own condition cone of " << cone->size()
      << " op(s): the CHECK controller waits out its depth each iteration";
  return scheduleAcyclic(*cone, /*ownsRegion=*/false);
}

// Schedule one straight-line region as a `ChainingSharedOperatorsProblem` and
// annotate the result.
//
// \p ownsRegion is false for a span that is TIMED but is no region of its own:
// a sequential while's condition cone, whose op start times the emitter reads
// back (`emitConditionRegion`) while the WHILE's controller paces it. Its
// `RegionSolution` would be read by nobody, and `publishKernelLatency` counts
// the ones that exist.
LogicalResult FuncScheduler::scheduleAcyclic(ArrayRef<Operation *> ops,
                                             bool ownsRegion) {
  ChainingSharedOperatorsProblem problem =
      buildAcyclicProblem<ChainingSharedOperatorsProblem>(ops, deps);
  populateOperatorTypes(problem, dev.operators, dev.memory);
  populateMemoryResources(problem, dev.memory);
  if (opts.allocate)
    populateOperatorAllocation(problem, dev.operators,
                               usesExactScheduler(opts.kind)
                                   ? AllocationScope::Static
                                   : AllocationScope::All,
                               opts.objective == ScheduleObjective::Area);
  // A straight-line region runs once, so its whole cost is its drain, and it
  // carries nothing between iterations it does not have.
  SpanObjective span(problem, spanEscapingValues(ops),
                     /*carried=*/nullptr, /*trip=*/1, dev.operators);
  Stopwatch solveStart = now();
  if (failed(
          solveSchedulingProblem(problem, ops.back(), cycleTime, opts, span)))
    return failure();
  recordSolve(problem, "acyclic", /*ii=*/std::nullopt, solveStart);
  if (ledger && ownsRegion)
    collectCallSlack(problem, ops);
  int64_t depth = problem.scheduleDepth();
  info(Stage::Sched, ops.front())
      << "Scheduled: depth = " << depth << " cycles";
  annotateStarts(problem);
  if (ownsRegion) {
    // A straight-line span issues once, so it carries no II and no trip. How
    // often its enclosing loops re-run it is charged where they are composed.
    RegionSolution &sol = model.addRegion(ops.front());
    sol.length = depth;
    sol.drain = span.drainOf(problem);
  }
  annotateAllocation(problem);
  model.modeledArea += regionArea(problem, span, /*ii=*/0);
  return success();
}

// The body elements of a container, in program order.
std::vector<SpanNode> FuncScheduler::buildSpanNodes(Block &body) {
  std::vector<SpanNode> nodes;
  for (const SchedRegion &child : enumerateRegions(body))
    if (std::optional<SpanNode> n = buildSpanNode(child))
      nodes.push_back(std::move(*n));
  return nodes;
}

// One scheduling region as the latency model sees it, walked over the
// affine/scf loops; `PostConversion.cpp` walks the dcp regions built from those
// same loops, and both feed `composeSpan`.
//
// Descends the loop nest, not the solution list: one solution covers a whole
// perfect band, while the emitter drives every loop above the innermost as a
// container with its own boundary cycles, which a flat walk of solutions has
// nowhere to charge.
//
// nullopt means the region occupies no cycles and forms no node (a
// straight-line span of nothing but declarations). A data-dependent region
// still forms a node, with the unknown left in its own fields.
std::optional<SpanNode>
FuncScheduler::buildSpanNode(const SchedRegion &region) {
  SpanNode n;
  // Driven by an enclosing region rather than by the func's own sequencer, the
  // same question the reify side asks of a dcp op's parents.
  n.elastic =
      llvm::any_of(region.ops, [](Operation *o) { return isElastic(o); });
  if (region.kind == allo::RegionKind::StraightLine) {
    if (!spanFormsRegion(region.ops))
      return std::nullopt;
    n.acyclic = true;
    n.trip = 1;
    if (RegionSolution *sol = model.regionOf(region.ops.front()))
      n.drain = sol->drain;
    return n;
  }
  Operation *anchor = region.anchor();
  // An `if` if-conversion left opaque runs under a predicate, which becomes a
  // `dcp.select` the reify side reads back as a Guard. Its branches hold the
  // arms' scheduled sub-regions, composed by `composeSpan`'s ceiling rule.
  if (isa<AffineIfOp, scf::IfOp>(anchor)) {
    n.shape = RegionShape::Guard;
    n.children = buildSpanNodes(anchor->getRegion(0).front());
    if (!anchor->getRegion(1).empty())
      n.elseChildren = buildSpanNodes(anchor->getRegion(1).front());
    return n;
  }
  if (!isa<AffineForOp, scf::ForOp>(anchor))
    return n; // a while: a data-dependent trip, so no static span
  auto loop = cast<LoopLikeOpInterface>(anchor);
  n.trip = deps.tripOf(anchor).count;
  n.shape = countedLoopShape(loop);
  Block &body = loop.getLoopRegions().front()->front();

  if (n.shape == RegionShape::CallNode) {
    // The body is one instance the controller re-fires per iteration, so a pass
    // costs the callee's own start to done contract and nothing else.
    for (Operation &op : body) {
      if (!isSyncSubKernelCall(&op))
        continue;
      Operation *callee = calleeOf(&op);
      SpanNode child;
      child.instance = true;
      child.contract = callee ? calleeStaticLatency(callee) : std::nullopt;
      n.children.push_back(std::move(child));
    }
    return n;
  }
  // A container owns no solution: it sequences the regions its body decomposed
  // into, and its span is composed from theirs.
  if (n.shape == RegionShape::Container) {
    n.children = buildSpanNodes(body);
    return n;
  }
  // A leaf nests no loop, so it is the op the solve was keyed by.
  if (RegionSolution *sol = model.regionOf(anchor)) {
    n.drain = sol->drain;
    n.ii = sol->ii;
  }
  return n;
}

// Record every counted loop whose iteration count only an `allo.assume.ssa`
// range bounds, for the reify to stamp as `trip_bound` and the emitter to size
// its counter by. This is the one fact reification cannot re-derive: the hint
// that bounded a symbolic trip is already consumed and erased by the time reify
// runs, unlike a loop's lb/step/constant trip, which stay on the loop.
void FuncScheduler::recordTripBounds(func::FuncOp funcOp) {
  funcOp.walk([&](Operation *op) {
    if (!isa<AffineForOp, scf::ForOp>(op))
      return;
    LoopTrip trip = deps.tripOf(op);
    if (trip.bounded && trip.count)
      model.setTripBound(op, *trip.count);
  });
}

// Compose the solved region tree into one whole-kernel span, and publish it.
// The only thing the scheduler writes to the IR, and the only thing a caller of
// this kernel sees. Sets the attribute only when every region has a known span.
//
// The span is the top-level regions composed over their dependence DAG, and
// must equal what the reify's `setDcpLatencies` composes off the dcp regions
// built from these. Independent siblings overlap, so it is the longest path and
// not the sum.
void FuncScheduler::publishKernelLatency(func::FuncOp funcOp) {
  Builder b(funcOp.getContext());

  // A callee whose own length is data-dependent leaves this kernel's unknown.
  // Must be asked here: the operator library prices an uncharacterized call at
  // zero, so the composition alone would omit it.
  bool callsKnown = true;
  funcOp.walk([&](func::CallOp call) {
    if (call->hasAttr(kAlloAsyncAttr))
      return;
    Operation *callee = calleeOf(call);
    if (!callee || !calleeStaticLatency(callee))
      callsKnown = false;
  });
  if (!callsKnown)
    return;

  std::vector<SpanNode> top;
  SmallVector<SmallVector<Operation *>> topOps;
  SmallVector<SchedRegion> topRegions;
  for (const SchedRegion &region : enumerateRegions(funcOp))
    if (std::optional<SpanNode> n = buildSpanNode(region)) {
      top.push_back(std::move(*n));
      topOps.emplace_back(region.ops.begin(), region.ops.end());
      topRegions.push_back(region);
    }
  // A func with no node to compose (an empty body, or nothing but declarations)
  // publishes nothing: composing over none reports zero, which a caller would
  // read as an exact zero-cycle contract.
  if (top.empty())
    return;
  std::vector<SmallVector<unsigned, 2>> preds = siblingPredecessors(topOps);
  std::optional<int64_t> total = composeDag(top, preds);
  if (!total)
    return; // a data-dependent region leaves the kernel total unknown
  if (ledger)
    collectSiblingSlack(top, topRegions, preds);

  // Only the number is published, not whether it is a bound: a bound is an
  // upper one, so a caller placing consumers against it is safe either way.
  funcOp->setAttr(kLatencyAttr, b.getI64IntegerAttr(*total));
}

// The sibling half of the slack pass: total float over the func's top-level
// DAG, spent whole on one region at a time (a thin slice cannot buy an II
// step). Reserving a node's full float re-times the DAG, so each round
// recomputes before granting the next; a node grants once.
void FuncScheduler::collectSiblingSlack(
    ArrayRef<SpanNode> nodes, ArrayRef<SchedRegion> regions,
    ArrayRef<SmallVector<unsigned, 2>> preds) {
  unsigned n = nodes.size();
  SmallVector<int64_t> spans(n);
  for (auto [i, node] : llvm::enumerate(nodes)) {
    std::optional<int64_t> s = composeSpan(node);
    assert(s && "the composed total exists, so every node span does");
    spans[i] = *s;
  }
  SmallVector<SmallVector<unsigned, 2>> succs(n);
  for (unsigned i = 0; i < n; ++i)
    for (unsigned p : preds[i])
      succs[p].push_back(i);
  SmallVector<std::optional<std::pair<Operation *, int64_t>>> targets(n);
  for (unsigned i = 0; i < n; ++i)
    targets[i] = grantTarget(regions[i], deps);
  for (unsigned round = 0; round < n; ++round) {
    // Program order is topological: a node's predecessors are earlier nodes.
    SmallVector<int64_t> est(n, 0), down(n, 0);
    for (unsigned i = 0; i < n; ++i)
      for (unsigned p : preds[i])
        est[i] = std::max(est[i], est[p] + spans[p]);
    int64_t total = 0;
    for (unsigned i = n; i-- > 0;) {
      down[i] = spans[i];
      for (unsigned s : succs[i])
        down[i] = std::max(down[i], spans[i] + down[s]);
      total = std::max(total, est[i] + down[i]);
    }
    int64_t bestFloat = 0;
    int best = -1;
    for (unsigned i = 0; i < n; ++i)
      if (targets[i])
        if (int64_t f = total - est[i] - down[i]; f > bestFloat) {
          bestFloat = f;
          best = static_cast<int>(i);
        }
    if (best < 0)
      break;
    auto [key, divisor] = *targets[best];
    if (int64_t g = bestFloat / divisor)
      ledger->grants[key] += g;
    spans[best] += bestFloat;
    targets[best].reset();
  }
}

// The call half: total float of a single-site sync call within its region's
// latency DAG. The callee may run that much longer without moving this
// region's span, whatever loop re-runs the region; the budget lands on the
// callee's own regions after the pass.
void FuncScheduler::collectCallSlack(ChainingSharedOperatorsProblem &problem,
                                     ArrayRef<Operation *> ops) {
  if (llvm::none_of(ops, [](Operation *op) { return isSyncSubKernelCall(op); }))
    return;
  auto latOf = [&](Operation *op) -> int64_t {
    return *problem.getLatency(*problem.getLinkedOperatorType(op));
  };
  DenseMap<Operation *, int64_t> asap, down;
  DenseMap<Operation *, SmallVector<Operation *, 4>> succs;
  for (Operation *op : ops) { // block order is topological over the deps
    int64_t t = 0;
    for (auto &dep : problem.getDependences(op)) {
      Operation *src = dep.getSource();
      t = std::max(t, asap.lookup(src) + latOf(src));
      succs[src].push_back(op);
    }
    asap[op] = t;
  }
  int64_t total = 0;
  for (Operation *op : llvm::reverse(ops)) {
    int64_t d = latOf(op);
    for (Operation *s : succs.lookup(op))
      d = std::max(d, latOf(op) + down.lookup(s));
    down[op] = d;
    total = std::max(total, asap.lookup(op) + d);
  }
  for (Operation *op : ops) {
    if (!isSyncSubKernelCall(op))
      continue;
    Operation *callee = calleeOf(op);
    if (!callee || ledger->callSites.lookup(callee) != 1)
      continue;
    if (int64_t f = total - asap.lookup(op) - down.lookup(op); f > 0)
      ledger->calleeBudget[callee] = f;
  }
}

// Schedule one region: a straight-line span as an acyclic problem, a counted
// loop as a cyclic problem. An imperfect counted nest, whose innermost band
// body still holds loops, is decomposed into per-body sub-regions, the band
// loops staying as wrapper loops that drive those sub-regions as containers.
LogicalResult FuncScheduler::scheduleRegion(const SchedRegion &region) {
  if (region.kind != allo::RegionKind::Loop) {
    // A span of nothing but declarations is a tie-off the reify leaves in
    // place; scheduling it would spuriously let a func with nothing else
    // publish a zero-cycle latency. Shared predicate with the reify.
    if (!spanFormsRegion(region.ops))
      return success();
    info(Stage::Sched, region.anchor())
        << "A straight-line span of " << region.ops.size()
        << " op(s), using acyclic scheduling";
    return scheduleAcyclic(region.ops, /*ownsRegion=*/true);
  }
  if (isa<AffineForOp, scf::ForOp>(region.anchor())) {
    SmallVector<LoopLikeOpInterface> band =
        perfectNest(cast<LoopLikeOpInterface>(region.anchor()));
    LoopLikeOpInterface innermost = band.back();
    int64_t dir = pipelineDirective(innermost.getOperation(), region.anchor());
    // The same shape query `buildSpanNode` composes through, so solving and
    // costing agree on which level drives children. Only a Container
    // decomposes; a CallNode and a Leaf run one flat cyclic problem.
    if (countedLoopShape(innermost) == RegionShape::Container) {
      // Fusing the level over its inner loops into one modulo problem is not
      // implemented: the container sequences its children and runs no schedule
      // of its own.
      if (dir >= 1) {
        model.unhonored.push_back(
            {"pipeline", logging::detail::describe(innermost.getLoc()),
             "imperfect_nest"});
        warn(Stage::Sched, innermost.getOperation())
            << "A pipeline directive on an imperfect nest is not honored yet; "
               "scheduling its body as sequential sub-regions. Leave "
               "`unroll_under_pipeline` at its default, which unrolls the "
               "inner loops into the pipelined level instead";
      }
      info(Stage::Sched, innermost.getOperation())
          << "Detected imperfect nest, decomposing into sub-regions "
             "scheduled in program order.";
      Block &body = innermost.getLoopRegions().front()->front();
      return scheduleBlock(body);
    }
    {
      auto d = info(Stage::Sched, innermost.getOperation());
      d << "Detected as a for-loop";
      if (band.size() > 1)
        d << " (perfect band of " << band.size() << " levels)";
      if (dir == -1)
        d << ", pipelining disabled";
      else if (dir >= 1)
        d << ", target II=" << dir;
      d << ", using modulo-scheduling in the innermost body";
    }
    unsigned target = dir >= 1 ? static_cast<unsigned>(dir) : 0;
    return scheduleCyclic(innermost, region, std::max(target, 1u),
                          /*maxII=*/target, /*pipelined=*/dir != -1);
  }
  // An uncounted while; counted ones are already scf.for.
  if (auto whileOp = dyn_cast<scf::WhileOp>(region.anchor())) {
    // A nested loop (data-dependent per-iteration length) or a condition not
    // settled at issue forces the sequential CHECK/RUN controller. The
    // reifier's routing shares `conditionIsCombinational`, so the two agree.
    if (!whileFlushingPipelines(whileOp, dev)) {
      info(Stage::Sched, whileOp)
          << "While loop cannot flushing-pipeline (nested loop, sub-kernel "
             "call, or non-combinational condition); decomposing its body "
             "into sub-regions scheduled in program order (the outer while "
             "runs sequentially, latency data-dependent)";
      if (failed(scheduleWhileCondition(whileOp)))
        return failure();
      return scheduleBlock(whileOp.getAfter().front());
    }
    // `verify-rtl-legality` rejects a flushing while that does not forward
    // 1:1, so `buildWhileProblem`'s slot alignment holds here.
    assert(whileHasIdentityForwarding(whileOp) &&
           "a flushing while reached scheduling without identity forwarding");
    info(Stage::Sched, whileOp.getOperation())
        << "Detected as a while-loop, using flushing-pipeline schedule";
    return scheduleWhile(whileOp, region);
  }
  // An `if` that `fold-if-statements` could not predicate stays a control
  // construct: decompose each branch into sub-regions and leave the `if` raw
  // around them.
  if (isa<AffineIfOp, scf::IfOp>(region.anchor())) {
    Operation *ifOp = region.anchor();
    info(Stage::Sched, ifOp)
        << "Detected a conditional left opaque by if-conversion; decomposing "
           "each branch into sub-regions and keeping the `if` as a guard";
    for (Region &branch : ifOp->getRegions())
      if (!branch.empty())
        if (failed(scheduleBlock(branch.front())))
          return failure();
    return success();
  }
  error(Stage::Sched, Code::RegionShapeNotScheduled, region.anchor())
      << "Loop not scheduled";
  return failure();
}

LogicalResult FuncScheduler::scheduleBlock(Block &block) {
  for (const SchedRegion &region : enumerateRegions(block))
    if (failed(scheduleRegion(region)))
      return failure();
  return success();
}

// Erase one consumed hint along with any operand-producing ops it leaves
// trivially dead. The assert guards against a freed op's address being reused
// by the next `create`, which would alias a stale key in the analysis's range
// map.
void FuncScheduler::eraseHint(RewriterBase &b, Operation *op) {
  SmallVector<Value, 4> operands(op->getOperands());
  b.eraseOp(op);
  for (Value v : operands)
    if (Operation *def = v.getDefiningOp())
      if (isOpTriviallyDead(def)) {
        assert(llvm::none_of(
                   def->getResults(),
                   [&](Value r) { return deps.getAssumedRanges().count(r); }) &&
               "erasing a value the assumed-range map is keyed by");
        eraseHint(b, def);
      }
}

// Erase the hints `deps` has already consumed: they carry no schedulable
// computation and would perturb the problem. Erasing them before the analysis
// was built would have dropped every assumption instead.
void FuncScheduler::consumeHints(func::FuncOp funcOp) {
  SmallVector<Operation *, 4> hints;
  funcOp.walk([&](Operation *op) {
    if (isa<AssumeNoDepOp, AssumeSSAOp>(op))
      hints.push_back(op);
  });
  IRRewriter rewriter(funcOp.getContext());
  for (Operation *op : hints)
    eraseHint(rewriter, op);
}

// Solve one function's schedule into `model`. The only IR this writes is the
// hints it consumes and the kernel latency it publishes; the schedule itself is
// materialized by a later pass off the model.
LogicalResult FuncScheduler::run(func::FuncOp funcOp) {
  consumeHints(funcOp);

  std::string infoStr = "-- Start scheduling for " + funcOp.getSymName().str();
  info(Stage::Sched) << std::string(infoStr.size() * 2, '-');
  info(Stage::Sched) << infoStr;
  info(Stage::Sched) << std::string(infoStr.size() * 2, '-');

  // Schedule the function body's regions, recursing into imperfect nests.
  if (failed(scheduleBlock(funcOp.getBody().front())))
    return failure();
  recordTripBounds(funcOp);
  publishKernelLatency(funcOp);
  return success();
}

static void loadDependentDialects(MLIRContext &context) {
  context.getOrLoadDialect<allo::AlloDialect>();
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<func::FuncDialect>();
  context.getOrLoadDialect<math::MathDialect>();
  context.getOrLoadDialect<affine::AffineDialect>();
  context.getOrLoadDialect<scf::SCFDialect>();
  context.getOrLoadDialect<memref::MemRefDialect>();
}

// The least clock period the solve can hold: every operation must fit a cycle
// of its own, `regFloor + inDelay` to reach its first register, `outDelay` to
// leave its last, and `minPeriod` for the internal stages the row is
// warranted at. A target below the result is raised rather than refused, so an
// over-period operator lowers the achieved frequency instead of failing the
// compile. Each offending row is named once, with what would shorten it.
//
// The walk prices through the same characterization `populateOperatorTypes`
// registers, over every op a problem can hold: a region op, a call and a
// declaration contribute no operator row. Selection ranks fit-first against
// the target here, so an op with any candidate inside the target derates
// nothing, and one with none is priced at its least-need candidate, the row
// the raised period re-selects.
static float minSchedulablePeriod(ArrayRef<func::FuncOp> funcs,
                                  const DeviceModel &dev, float target,
                                  float regFloor) {
  float least = target;
  llvm::StringSet<> named;
  for (func::FuncOp fn : funcs)
    fn.walk([&](Operation *op) {
      if (op->getNumRegions() || isa<func::CallOp>(op))
        return;
      bool access = asMemAccess(op).has_value();
      NodeTiming t = access
                         ? accessCharacterization(op, dev.operators, dev.memory)
                         : dev.operators.lookup(op).timing;
      float need = periodNeed(regFloor, t.inDelay, t.outDelay, t.minPeriod);
      if (need <= target)
        return;
      least = std::max(least, need);
      if (!named.insert(t.typeName).second)
        return;
      // An access is named by whichever of its two cones is the larger, since
      // that is the one worth shortening.
      const char *advice =
          isa<AffineApplyOp>(op)
              ? "Its whole map is one combinational cone; compute the "
                "expression in arithmetic ops so each step can be scheduled "
                "and registered"
          : !access ? "It is one operator, so no register can split it"
          : portSelectDelay(op, dev.operators) >
                  addressDelayOf(op, dev.operators)
              ? "The select over the accesses sharing this array's port is "
                "what costs; partition the array, or move accesses out of it, "
                "so fewer of them drive one bus"
              : "The address cone ahead of the port is what costs; compute "
                "the subscript into a variable so it becomes a schedulable "
                "value, or partition by a power of two so the bank digit is "
                "a mask rather than a divider";
      warn(Stage::Sched, op)
          << "'" << t.typeName << "' needs " << format("%.2f", need)
          << " ns for a cycle of its own, over the " << format("%.2f", target)
          << " ns clock period. " << advice;
    });
  return least;
}

// The area objective's slack pass: a cheap heuristic pre-schedule of the whole
// module, whose composition prices each region's float off the sibling DAG's
// longest path. What it proves free widens the real pass's leashes; the
// composed kernel span stays within the heuristic's by construction, since
// every float is charged once.
static DenseMap<Operation *, int64_t> collectCompositionSlack(
    ModuleOp module, func::FuncOp topFunc, ArrayRef<func::FuncOp> order,
    ArrayRef<std::unique_ptr<DependenceAnalysis>> depsFor,
    const DeviceModel &dev, float cycleTime, const SchedulerOptions &opts) {
  DenseMap<Operation *, int64_t> grants;
  SlackLedger ledger;
  module.walk([&](Operation *op) {
    if (isSyncSubKernelCall(op))
      if (Operation *callee = calleeOf(op))
        ++ledger.callSites[callee];
  });
  ScheduleModel probe;
  probe.cycleTimeNs = cycleTime;
  SchedulerOptions heur = opts;
  heur.kind = SchedulerKind::Heuristic;
  bool preScheduled = true;
  for (auto [fn, deps] : llvm::zip(order, depsFor)) {
    FuncScheduler sched(*deps, dev, probe, cycleTime, heur, &ledger);
    if (failed(sched.run(fn))) {
      preScheduled = false;
      break;
    }
  }
  if (preScheduled) {
    // A single-site callee's banked float lands on its one grantable top
    // region, divided by the wrappers between the callee's span and it.
    for (auto [calleeOp, budget] : ledger.calleeBudget) {
      auto fn = cast<func::FuncOp>(calleeOp);
      const auto *at = llvm::find(order, fn);
      assert(at != order.end() && "a called func is in the order");
      DependenceAnalysis &calleeDeps = *depsFor[at - order.begin()];
      std::optional<std::pair<Operation *, int64_t>> target;
      unsigned grantable = 0;
      for (const SchedRegion &r : enumerateRegions(fn))
        if (auto t = grantTarget(r, calleeDeps)) {
          ++grantable;
          target = t;
        }
      if (grantable != 1)
        continue;
      if (int64_t g = budget / target->second)
        grants[target->first] += g;
    }
    for (auto [key, g] : ledger.grants)
      grants[key] += g;
    if (!grants.empty()) {
      int64_t granted = 0;
      for (auto [key, g] : grants)
        granted += g;
      info(Stage::Sched, topFunc)
          << "Composition slack: " << grants.size()
          << " region leash(es) widened by " << granted << " cycle(s) in total";
    }
  } else {
    info(Stage::Sched, topFunc)
        << "The slack pre-schedule did not place; area leashes stay "
           "region-local";
  }
  return grants;
}

LogicalResult mlir::allo::runSDCScheduler(ModuleOp module, StringRef top,
                                          float cycleTime,
                                          const SchedulerOptions &opts,
                                          ScheduleModel &model) {
  loadDependentDialects(*module->getContext());
  // Timing characterization for every op (latency + delays), built from the
  // injected `dcp.device` + `dcp.operator` IR, once for scheduling and reify.
  auto loadedDev = DeviceModel::fromModule(module);

  // Callees before callers: a caller's own region partition asks whether each
  // call is indeterminate, which reads the callee's published latency.
  auto topFunc = module.lookupSymbol<func::FuncOp>(top);
  if (!topFunc) {
    error(Stage::Prep, Code::TopFunctionMissing, module)
        << "Top function '" << top << "' not found";
    return failure();
  }
  auto orderOr = callGraphPostOrder(topFunc);
  if (failed(orderOr))
    return failure();

  // The fabric floor: a source flip-flop's clock-to-out plus routing, what a
  // path with no operator in it already costs. It reaches the solve as the
  // earliest sub-cycle time any operation may start at, not as a per-row delay
  // (which would cost an N-deep chain N floors) and not off the period (which
  // would leave `unitSlack` disagreeing with the solve by one floor).
  float regFloor = loadedDev.operators.registerFloor();

  // Derate rather than refuse: a target no single operator fits is raised to
  // the least period every row does, once for the whole module (one clock
  // domain), and everything downstream holds the raised period. The miss is a
  // report; the schedule stays valid at the frequency actually achieved.
  loadedDev.operators.setSelectionPeriod(cycleTime);
  float scheduled =
      minSchedulablePeriod(*orderOr, loadedDev, cycleTime, regFloor);
  if (scheduled > cycleTime) {
    warn(Stage::Sched, topFunc)
        << "The requested " << format("%.2f", cycleTime) << " ns clock period ("
        << format("%.0f", 1000.0f / cycleTime)
        << " MHz) is not schedulable on this device; scheduling at "
        << format("%.2f", scheduled) << " ns ("
        << format("%.0f", 1000.0f / scheduled)
        << " MHz). The QoR report prices the design at the achieved period";
    // Re-rank at the achieved period: a row that missed the target may fit
    // the raised period and win back a shorter latency. Every op keeps a row
    // that fits, since the walk raised the period to its least need.
    loadedDev.operators.setSelectionPeriod(scheduled);
  }
  model.cycleTimeNs = scheduled;

  if (scheduled <= regFloor) {
    error(Stage::Sched, Code::OperatorOverPeriod, module)
        << "The requested clock period of " << scheduled
        << " ns is at or below this device's register-to-register floor of "
        << regFloor << " ns, so no cycle has room for any logic at all";
    return failure();
  }
  SchedulerOptions optsWithFloor = opts;
  optsWithFloor.regFloor = regFloor;

  // Whole-func memory + stream dependence analysis, refined by the
  // `allo.assume.*` hints. Built before any solve since the first run consumes
  // the hints from the IR, and outliving the solves: the span composition
  // reads its value ranges to bound a symbolic trip.
  SmallVector<std::unique_ptr<DependenceAnalysis>> depsFor;
  for (func::FuncOp fn : *orderOr)
    depsFor.push_back(std::make_unique<DependenceAnalysis>(fn));

  if (usesExactScheduler(opts.kind))
    info(Stage::Sched, module)
        << "Exact scheduler: " << opts.workers << " workers, seed " << opts.seed
        << ", budget " << format("%g", opts.budget)
        << " deterministic units per region"
        << (opts.deterministic || opts.workers == 1 ? "" : ", workers racing");

  DenseMap<Operation *, int64_t> grants;
  if (opts.objective == ScheduleObjective::Area &&
      usesExactScheduler(opts.kind))
    grants = collectCompositionSlack(module, topFunc, *orderOr, depsFor,
                                     loadedDev, scheduled, optsWithFloor);

  for (auto [fn, deps] : llvm::zip(*orderOr, depsFor)) {
    FuncScheduler sched(*deps, loadedDev, model, scheduled, optsWithFloor,
                        /*ledger=*/nullptr, &grants);
    if (failed(sched.run(fn)))
      return failure();
  }
  return success();
}
