/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_SCHEDULING_LATENCY_MODEL_H
#define ALLO_SCHEDULING_LATENCY_MODEL_H

#include "allo/IR/AlloAttrs.h"           // DeterminacyEnum
#include "allo/Scheduling/RegionGraph.h" // RegionShape

#include "mlir/IR/Block.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace mlir::allo {

/// The cycles a controller family spends at its region's boundaries, outside
/// the region's own schedule. Only structural constants live here; a
/// datapath-derived delay (a condition cone's `tCond`, a region's `drainStage`)
/// is passed in as a parameter.
struct BoundaryCost {
  /// `start` -> the first body pass issues.
  unsigned arm;
  /// A body pass completing -> the next one issues. Meaningless for a run-once
  /// family, which sets it equal to `arm`.
  unsigned reArm;
};

/// A `done` level is a latch, so a region's completion is visible one cycle
/// after the pulse that sets it; every family pays it toward its own `done`. A
/// successor of a conditional region instead starts on the completion pulse
/// itself (`HWEmitter::handoffSafe`), and no static span composes across such a
/// region, so the arithmetic below never describes that boundary.
inline constexpr unsigned kDoneLatchCycles = 1;

/// A container's children read its counter as their own bound and sample it at
/// their own start, so every launch is registered, the first one included.
/// A container whose first child is conditional instead relaunches on the last
/// child's completion pulse (`HWEmitter::chainsTurnover`) and composes no body
/// span, so this constant does not describe it.
inline constexpr BoundaryCost kContainerBoundary{/*arm=*/1, /*reArm=*/1};

/// A call region's first pass launches on `start` itself through a start-cycle
/// bypass. Advances still ride the settled register.
inline constexpr BoundaryCost kCallNodeBoundary{/*arm=*/0, /*reArm=*/1};

/// A sequential-wrapper while re-evaluates its condition on a check pulse one
/// cycle after `start` and after each body drain; the condition cone's own
/// `tCond` is added on top by the controller. A chained one drains on the
/// body's completion pulse, so its check follows the last commit by one cycle;
/// a while composes no span, so neither variant is priced.
inline constexpr BoundaryCost kCheckedBoundary{/*arm=*/1, /*reArm=*/1};

/// A guard checks its predicate one cycle after `start`. That also keeps a
/// skipped guard's completion pulse off the `done` latch's start-clear, which
/// would otherwise leave no rising edge.
inline constexpr BoundaryCost kGuardBoundary{/*arm=*/1, /*reArm=*/1};

/// An acyclic region arms on `start` directly, nested or top-level: a
/// container's counter and iter-args commit at least a cycle before the launch
/// pulse, and a top-level one has no outer counter to wait for.
inline constexpr BoundaryCost kAcyclicBoundary{/*arm=*/0, /*reArm=*/0};

/// A pipelined leaf issues its first iteration on `start` itself: the counter,
/// phase and scaled counters read their reload values through a start-cycle
/// bypass. Iterations then overlap at the solved `ii`, which `leafSpan` takes
/// as a parameter, so `reArm` does not describe this family. Only the rigid
/// counted family is built this way; a while and an elastic region compose no
/// static span, so this constant never describes them.
inline constexpr BoundaryCost kPipelinedBoundary{/*arm=*/0, /*reArm=*/0};

/// An empty region, a runtime zero trip or a static `lb >= ub`, never launches
/// a pass at all: `gateStart` masks the start launch and a register on `start
/// && isEmpty` feeds the `done` latch. Two cycles, whichever family drives the
/// region. A separate constant, not the arithmetic below at trip zero, which
/// describes the steady state and is written for `trip >= 1`.
inline constexpr int64_t kEmptyRegionCycles = 2;

/// A done-paced region's whole span, given what one pass of its body costs
/// (\p bodySpan, the sum of its children's spans). Evaluated identically by the
/// scheduler and the reifier.
inline int64_t containerSpan(const BoundaryCost &boundary, int64_t trip,
                             int64_t bodySpan) {
  if (trip == 0)
    return kEmptyRegionCycles;
  return boundary.arm + (trip - 1) * (boundary.reArm + bodySpan) + bodySpan +
         kDoneLatchCycles;
}

/// A leaf's whole span: it arms, issues \p trip iterations at its solved \p ii,
/// then drains.
///
/// \p drain is the terminal quantity, the cycles from the last issue pulse to
/// the deepest output committing, so `done` rises `drain + 1` cycles after that
/// pulse. It is not the schedule depth, above which the solver may leave slack.
inline int64_t leafSpan(const BoundaryCost &boundary, int64_t trip, int64_t ii,
                        int64_t drain) {
  if (trip == 0)
    return kEmptyRegionCycles;
  return boundary.arm + (trip - 1) * ii + drain + kDoneLatchCycles;
}

/// One region as the latency model sees it: enough to compose a span, and
/// nothing else. Built by two structural walks, the scheduler over affine/scf
/// loops and the reifier over the dcp regions built from them, that both feed
/// the composition arithmetic above.
struct SpanNode {
  RegionShape shape = RegionShape::Leaf;
  /// Iterations of this region's body. Empty when data-dependent (a `while`, a
  /// dynamic bound), which leaves every enclosing span unknown rather than
  /// guessed.
  std::optional<int64_t> trip;
  /// A leaf's own solved schedule: its issue cadence and its terminal cycle,
  /// the delay from the last issue pulse to the deepest output committing. `ii`
  /// stays empty for an acyclic leaf, which issues once. `drain`, not the
  /// schedule depth, since the solver may leave slack above the last commit.
  std::optional<int64_t> drain, ii;
  /// An instance element's whole start->done contract (see `instance`).
  std::optional<int64_t> contract;
  /// A worst case the scheduler bounded from an `allo.assume.ssa` range, for a
  /// node whose own `trip` is data-dependent. Usable only as a kernel's own
  /// `latency` (flagged `latency_bound`, so a caller waits it out), never as a
  /// container's body pass, which has to pace a real counter. Carried here
  /// because reification keeps the bounded latency but drops the assumed trip
  /// that produced it.
  std::optional<int64_t> assumedSpan;
  /// A straight-line span rather than a counted loop.
  bool acyclic = false;
  /// Paced by back-pressure rather than by its own schedule (`isElastic`), so
  /// it has no static span. Set on whichever node holds the stream access and
  /// on every node above it, though either alone would answer.
  bool elastic = false;
  /// An instance element rather than a region of this func: `contract` is the
  /// callee's whole start->done span, counted to its own `done` rising.
  bool instance = false;
  /// The contract composed here (an instance's, or one a leaf priced into its
  /// drain) is a ceiling (`latency_bound`), not an exact count.
  bool contractBound = false;
  /// Body elements of a done-paced region, in program order. `std::vector`
  /// rather than `SmallVector`: the element type is this one, still incomplete
  /// here, and only `std::vector` is specified to accept that.
  std::vector<SpanNode> children;
  /// A guard's else-arm elements, in program order; `children` holds its
  /// then-arm's. Empty for an absent or empty else, which completes in the
  /// arming cycle.
  std::vector<SpanNode> elseChildren;
};

/// The per-invocation span of \p n: its start pulse to its `done` rising,
/// including the node's own arming cost, so a composer only ever sums spans. A
/// leaf runs its own solved schedule and drains it; a done-paced region runs
/// its body elements in sequence, each handed to the next through its own
/// `done` latch; a guard composes the deeper arm, a ceiling rather than an
/// exact count (`spanHoldsBound`).
///
/// nullopt whenever any element is data-dependent, which leaves the enclosing
/// span unknown rather than guessed.
std::optional<int64_t> composeSpan(const SpanNode &n);

/// Whether the span composed of \p n is a ceiling rather than an exact count:
/// a guard runs whichever arm its predicate takes, an assumed trip is a worst
/// case, and a bounded contract hands its callee's ceiling on. Meaningful only
/// where `composeSpan` returned a value; the composition arithmetic is
/// monotone, so a ceiling composes into a ceiling.
bool spanHoldsBound(const SpanNode &n);

/// A run of nodes composed in program order, hence the sum of their spans: each
/// starts on its predecessor's `done` edge, which costs nothing (`startFor` is
/// a rising edge, not a register). Both compositions in the compiler are this
/// one function: a done-paced region's body pass, and a func's top-level
/// regions along one path of their DAG.
std::optional<int64_t> composeSequence(llvm::ArrayRef<SpanNode> nodes);

/// For each top-level node, in program order, the earlier nodes it must run
/// after. \p nodeOps gives the ops each node owns, which keeps this
/// IR-agnostic across the scheduler's and the reifier's regions.
///
/// Three signals, the same three the emitter composes on
/// (`DatapathBuilder::recordSiblingDeps`): a shared memref, a shared stream
/// channel, and a cross-region SSA use. A shared memref orders only its hazard
/// pairs (RAW / WAW / WAR): two nodes that only read it overlap, and
/// `Datapath::portGraph` prices the separate ports that takes. A stream's
/// token order is the program's, so its touchers are ordered regardless of
/// direction, and a skewed layout keeps every toucher ordered (its lanes share
/// a port per slot across regions). Everything else runs concurrently.
///
/// The edges must match what `recordSiblingDeps` builds, since the span is
/// published as an exact contract. `RegionGraph`'s polyhedral refinement is
/// therefore not used here: it drops edges the emitter keeps.
std::vector<llvm::SmallVector<unsigned, 2>>
siblingPredecessors(llvm::ArrayRef<llvm::SmallVector<Operation *>> nodeOps);

/// One node's (or region's) touch of a shared array, for `hazardEdges`.
struct MemTouch {
  unsigned node;
  bool writes;
  /// The statically-resolved bank, or nullopt for a touch that may reach any
  /// (a crossbar access, a child's mastered port).
  std::optional<int64_t> bank;
};

/// Invoke \p addPred(p, c) for each pair of \p touch (sorted by node) the
/// hazard rule orders: per bank, each reader after the last writer and each
/// writer after every reader since. A node paired with itself is skipped. A
/// bank is its own storage, so touches of different banks never pair; a
/// bank-less touch joins every bank. Read-read pairs stay unordered. This is
/// the one edge rule `siblingPredecessors` and
/// `DatapathBuilder::recordSiblingDeps` both compose on.
void hazardEdges(llvm::ArrayRef<MemTouch> touch,
                 llvm::function_ref<void(unsigned, unsigned)> addPred);

/// A func's top-level span: its regions composed over their dependence DAG.
///
/// The longest path, not the sum. Independent siblings overlap, so summing them
/// reports a kernel as slower than its own hardware. \p preds is
/// `siblingPredecessors`, indexed alongside \p nodes.
std::optional<int64_t>
composeDag(llvm::ArrayRef<SpanNode> nodes,
           llvm::ArrayRef<llvm::SmallVector<unsigned, 2>> preds);

/// One materialized dcp region as the latency model sees it: the reify-side
/// structural walk, over `dcp.pipeline` / `dcp.sequential` / `dcp.select` and
/// the `dcp.instance` elements they hold. `SDC.cpp` has the other, over the
/// affine/scf loops these were built from.
///
/// A region's span and its composition class are derived where they are used,
/// never read back off an attribute. For \p topLevel see `dcpSpanNodes`.
SpanNode dcpSpanNode(Operation *regionOp, bool topLevel);

/// The elements of a reified block, in program order.
///
/// The two scopes differ in what a span is for. \p topLevel is a func's entry
/// block, which composes an exported contract, so an assume-bounded region
/// contributes its bound, which a caller then waits out. A region body composes
/// one pass of a counter, where a bound cannot pace anything, so a bounded
/// child leaves the container done-paced instead.
std::vector<SpanNode> dcpSpanNodes(Block &block, bool topLevel);

/// How a materialized region is paced: which controller family drives it, and
/// the single-run span a container may time-trigger it against.
struct RegionTiming {
  DeterminacyEnum determinacy = DeterminacyEnum::Indeterminate;
  /// Present only for `counted_static`, and then it is exact.
  std::optional<int64_t> staticLatency;
  /// A ceiling on the span where one composes but is not exact
  /// (`spanHoldsBound`): a guard's deeper arm, or a bounded contract beneath.
  /// Never set beside `staticLatency`. Published as `latency_bound`, which a
  /// caller may wait out but nothing may time-trigger against.
  std::optional<int64_t> boundedLatency;
};

/// Derive \p regionOp's pacing from the region itself.
///
/// One definition, called twice: the reifier stamps `latency` / `latency_bound`
/// / `determinacy` from it, the emitter decides a controller family from it.
/// Those attributes are a report of this function, never an input to it.
///
/// Four classes, tested in order since each shadows the ones after it:
/// concurrent, conditional, counted_static, indeterminate.
RegionTiming dcpRegionTiming(Operation *regionOp);

/// Func-level: whole-kernel latency in cycles, the top-level regions composed
/// over their dependence DAG (`publishKernelLatency`). Set only when every
/// region has a composable span. Whether it is an exact count or an
/// assume-bounded worst case is not recorded: a bound is an upper one, so it
/// times a caller safely either way.
constexpr llvm::StringLiteral kLatencyAttr = "allo.sched.latency";

} // namespace mlir::allo

#endif // ALLO_SCHEDULING_LATENCY_MODEL_H
