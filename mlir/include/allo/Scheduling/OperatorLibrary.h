/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_SCHEDULING_OPERATORLIBRARY_H
#define ALLO_SCHEDULING_OPERATORLIBRARY_H

#include "allo/IR/AlloOps.h"                  // kAlloAsyncAttr
#include "allo/Scheduling/MemoryAccess.h"     // asMemAccess
#include "allo/Scheduling/MemoryModel.h"      // MemoryLibrary
#include "allo/Scheduling/OperatorIdentity.h" // OperatorIdentity
#include "allo/Scheduling/RegionGraph.h"      // calleeStaticLatency
#include "allo/Scheduling/Scheduler.h"

#include "circt/Scheduling/Problems.h"
#include "mlir/Dialect/Func/IR/FuncOps.h" // func::CallOp
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace mlir::allo {

//===----------------------------------------------------------------------===//
// Abstract operator vocabulary (hardware-facing, independent of MLIR op names).
//===----------------------------------------------------------------------===//

/// The abstract operator kind timing is characterized against, spelled in the
/// IR by `dcp.comb` and a built-in `dcp.operator` (see `OpKindEnum` in
/// `AlloAttrs.td`). `stringifyOpKindEnum` / `symbolizeOpKindEnum` convert, the
/// latter returning nullopt for an advanced mnemonic such as `sqrt`.
using OpKind = OpKindEnum;

OpKind classify(Operation *op);

/// The abstract kind a combinational realization is priced under, for a caller
/// holding the realization after the `arith` op is gone. Strictly coarser than
/// `classify`: signed and unsigned mnemonics share a row, as do the four
/// integer casts. `Unknown` for a realization no abstract row covers.
OpKind opKindOf(CombOpKindEnum kind);

/// The combinational realization kind of \p op, nullopt for an op with no comb
/// lowering (a float/cast IP, a memory access, an unrelated op). Every case
/// here has an `emitCompute` lowering in the emitter.
std::optional<CombOpKindEnum> combKindOf(Operation *op);

//===----------------------------------------------------------------------===//
// Library entries (built from the injected `dcp.operator` / `dcp.device` IR).
//===----------------------------------------------------------------------===//

/// One row of the operator library. A comb row (`comb == true`) matches by
/// `kind` + all-integer operands, at any width, except the `select` and `neg`
/// rows, which match any operand type: a mux over any datatype, a float sign
/// flip. An IP row matches by `kind` + an exact operand/result element-type
/// list; an advanced row additionally by `mlirOp`.
struct OperatorEntry {
  OpKind kind = OpKind::Unknown; // Unknown on an advanced row
  std::string mlirOp;            // advanced: raw MLIR op name, else empty
  bool comb = false;
  llvm::SmallVector<Type> argTypes; // IP/advanced: exact operand element types
  llvm::SmallVector<Type> resTypes; // IP/advanced: exact result element types

  uint32_t latency = 0;  // cycles
  bool pipelined = true; // accepts a new input every cycle
  double inDelay = 0.0;  // ns; IP rows: signature pins the width
  double outDelay = 0.0;
  /// Least clock period the row's internal stages are warranted at (ns); zero
  /// claims nothing, leaving the boundary cones the only gate.
  double minPeriod = 0.0;
  /// Nonzero when the row was measured with inputs extended from this many
  /// significant bits; a candidate only for operations proven that narrow.
  unsigned fedWidth = 0;
  /// Comb rows: delay as a function of operand width. Null on an IP row, whose
  /// signature fixes one width and so one delay.
  CostAttr delay;
  std::string symbol; // injected `dcp.operator` sym_name, IP rows only
  ArrayAttr uses;     // what one instance spends; null where the device is
                      // silent (`priceOf`)
};

/// The width one operator row is characterized at for \p op: its widest integer
/// or float operand, falling back to the result when it takes none. An `index`
/// answers `kIndexWidth`, the one width the datapath builds it at. `dcp.comb`
/// parameterizes on the operand width, so a 32-bit compare prices as 32 bits
/// rather than as its one-bit result.
int64_t combParamWidth(Operation *op);

/// Whether \p op costs no logic, so the schedule leaves it no delay: a bit
/// rename (`isBitRename`), or a resize between two types the datapath carries
/// at one width, which emits nothing at all.
///
/// Both places a datapath node is priced, the chaining solve and the binder's
/// slack, ask this, so they cannot disagree about what the schedule left.
bool isZeroDelay(Operation *op);

/// What a scheduling problem registers for one node: the operator type it runs
/// under and the timing that type carries. Common to the three kinds of node
/// (`populateOperatorTypes`): a device operator row, a callee's own schedule,
/// or a storage port.
///
/// `typeName` is the key, so two nodes under one name must carry the same
/// timing; anything that makes a node cost differently has to reach the name
/// too (an access's address cone does).
struct NodeTiming {
  std::string typeName;
  uint32_t latency = 0;
  double inDelay = 0.0;   // ns, from an operand to this node
  double outDelay = 0.0;  // ns, from this node to its consumer
  double minPeriod = 0.0; // ns, least period the unit behind it holds
};

/// The least period one row or node needs for a cycle of its own.
inline float periodNeed(float regFloor, double inDelay, double outDelay,
                        double minPeriod) {
  return std::max(
      {regFloor + (float)inDelay, (float)outDelay, (float)minPeriod});
}

/// What a lookup resolves for one operation: the timing row it is scheduled
/// under, and what the library knows about the unit behind it. Two keys: the
/// scheduling problem prices `timing.typeName`, while an allocation limit, the
/// binder's share test and the emitted module name all key on `identity`.
struct OperatorChar {
  NodeTiming timing; // one Problem::OperatorType per matched entry
  /// Whether one instance accepts a new input every cycle. False bounds a
  /// cyclic region's interval (`populateOperatorOccupancy`) and bars the
  /// operator from a cyclic allocation (`populateOperatorAllocation`).
  bool pipelined = true;
  /// What one instance of the matched row costs in the device's currency
  /// (`priceOf`), at this operation's width. Zero both where the device prices
  /// the row at nothing and where it prices it not at all.
  int64_t price = 0;
  OperatorIdentity identity; // empty for an op no functional unit is built for
};

/// The operator library, built from the injected device IR: comb rows from
/// `dcp.device.comb`, IP rows from `dcp.operator` symbols, storage timing from
/// `dcp.device.memory`.
class OperatorLibrary {
public:
  /// Build the library from a module's injected `dcp.device` + `dcp.operator`
  /// ops. A module with no `dcp.device` yields an empty (all-default) library.
  static OperatorLibrary fromModule(ModuleOp module);

  /// Resolve the characterization for \p op: the row `selectImplementation`
  /// picks out of the candidates matching it, else the default row. Pure: the
  /// answer depends only on the library (its selection period included) and on
  /// \p op's own name, types and attributes.
  ///
  /// A row the device never measured at \p op's width comes back unrealized,
  /// with the gap reported against \p op.
  OperatorChar lookup(Operation *op) const;

  /// The characterization of the specific candidate row \p symbol for \p op,
  /// for a caller holding a realization an exact solve decided
  /// (`OpSchedule::selectedImpl`) rather than the library's own pick.
  OperatorChar lookup(Operation *op, StringRef symbol) const;

  /// Every IP row that could realize \p op at the selection period, each as the
  /// characterization `lookup` would resolve had selection picked it: the
  /// non-comb `matchEntries` rows that fit the period and are measured at \p
  /// op's width, in declaration order. The row `lookup` resolves is among them
  /// whenever it is an IP.
  SmallVector<OperatorChar, 2> candidateChars(Operation *op) const;

  /// The clock period (ns) selection ranks IP rows against: a fitting row
  /// outranks a non-fitting one. Set when the module period resolves and again
  /// on a derate, so every lookup ranks against the period it schedules at.
  /// Unset ranks as if the period were unbounded.
  void setSelectionPeriod(float ns) { selectionPeriodNs = ns; }

  /// The symbol of every IP row that could realize \p op, whichever one the
  /// period makes `selectImplementation` pick. Held by the pre-schedule stall
  /// contract check, since selection settles only once the period does.
  SmallVector<StringRef, 2> candidateIPs(Operation *op) const;

  /// Whether \p op needs an IP realization (a float or advanced compute op) but
  /// its candidate set is empty, so the caller can report an error instead of
  /// scheduling it at the default zero latency.
  bool requiresUnmatchedIP(Operation *op) const;

  /// Whether the device provides a direct realization for \p op, i.e. its
  /// candidate set is non-empty. `legalize-arith` keeps a composite arith op
  /// (max/min/maxnum/minnum/ceildiv/floordiv) when this holds and expands it
  /// into primitive arith otherwise.
  bool hasDirectRealization(Operation *op) const;

  /// The fabric's register-to-register floor (ns): what a path with no operator
  /// in it costs. Every measured combinational delay includes it.
  double registerFloor() const { return regFloor; }

  /// The delay from a register through one instance of \p kind at \p width
  /// bits: the floor plus the operator's own, which is what a row contributes
  /// as an incoming delay. 0.0 when the device declares no row for \p kind.
  ///
  /// \p width here is a width the compiler picks for a structure it builds (the
  /// datapath's index width for address arithmetic, one bit for a multiplexer
  /// level), never a program's operand width, which `lookup` reads. Below the
  /// row's first measured point it reads that point; above the last measured
  /// point the value is unbounded.
  double combDelay(OpKind kind, int64_t width) const;

  /// What one instance of \p kind at \p width adds to a path that already left
  /// a register: \ref combDelay less the floor, never negative, which is what a
  /// row contributes as an outgoing delay.
  double combMarginalDelay(OpKind kind, int64_t width) const;

  /// \ref combDelay where the row measured \p width, nullopt above its last
  /// measured point or with no row at all.
  std::optional<double> measuredCombDelay(OpKind kind, int64_t width) const;

  /// The widest integer multiplier the device offers as a pipelined IP row,
  /// 0 with no such row.
  unsigned maxPipelinedMulWidth() const;

  /// The narrowest integer width of at least \p width an advanced row
  /// declares for the raw mnemonic \p mnem, or 0 when no row reaches it.
  unsigned smallestAdvancedRowWidth(llvm::StringRef mnem, unsigned width) const;

  /// Whether an advanced row declares mnemonic \p mnem at exactly \p args ->
  /// \p results.
  bool hasAdvancedRow(llvm::StringRef mnem, TypeRange args,
                      TypeRange results) const;

  /// The cheapest price among the advanced rows declaring \p mnem at exactly
  /// \p args -> \p results, at \p width; nullopt where none is priced there.
  std::optional<int64_t> advancedRowPrice(llvm::StringRef mnem, TypeRange args,
                                          TypeRange results,
                                          int64_t width) const;

  /// The same two, for a caller holding a reified realization (a
  /// `dcp.compute`'s `comb_kind`). Falls back to the default row, not to 0.0,
  /// so an `affine.apply` is priced the way it was scheduled. \p width is the
  /// realized operation's own, already inside the row's measured points.
  double combDelay(CombOpKindEnum kind, int64_t width) const;
  double combMarginalDelay(CombOpKindEnum kind, int64_t width) const;

  //===--------------------------------------------------------------------===//
  // Area, in the objective's currency.
  //
  // One unit of a resource costs `kPriceResolution` times the largest capacity
  // the device declares, over its own: a resource the part has less of costs
  // more, which is the only ranking a scheduler can have between a LUT and a
  // DSP slice. The scale itself is arbitrary and cancels, since every term of
  // the objective is in it; what it buys is resolution, the most plentiful
  // resource pricing at `kPriceResolution` and everything else rounding
  // against that.
  //
  // A capacity is a price input and NOT a budget: regions are solved
  // independently, so no single solve can see a whole-device total (see
  // `dcp.resource`).
  //===--------------------------------------------------------------------===//

  /// What `k` sources of `width` bits cost to select between, which is what
  /// sharing one functional unit puts in front of each of its operand ports.
  int64_t muxPrice(int64_t sources, int64_t width) const;

  /// What one instance of the realization \p identity names costs at \p width:
  /// its IP row, or the comb row of its kind. The bind-time twin of the price
  /// `lookup` resolves, for a caller not holding the original operation. Zero
  /// where the device declares no such row.
  int64_t instancePrice(const OperatorIdentity &identity, int64_t width) const;

  /// What carrying a `width`-bit value across `depth` cycles costs, in the
  /// reset-free form a value run is emitted in. Zero at depth 0, which is a
  /// wire and not a chain.
  int64_t chainPrice(int64_t depth, int64_t width) const;

  /// What one cycle of an activation pulse chain costs: one more stage of a
  /// one-bit chain, a flip-flop wherever the device says so.
  int64_t pulsePrice() const;

  /// The measured `dcp.mux` delay row over fan-in and its unitless width
  /// factor, null attrs on an uncharacterized device. Read by `muxCone`.
  CostAttr muxDelayRow() const { return muxDelay; }
  CostAttr muxDelayWidthRow() const { return muxDelayWidth; }

private:
  /// What \p uses spends at \p params, every resource at its price. Null
  /// \p uses is free, which is what a device saying nothing about a row means.
  /// Nullopt where a cost was not measured at its parameter.
  std::optional<int64_t> priceOf(ArrayAttr uses,
                                 ArrayRef<int64_t> params) const;

  /// The characterization \p op takes when the row \p e realizes it, at
  /// \p width bits: the tail of `lookup`, shared with `candidateChars`. The
  /// row's delay and price must be measured at \p width; `lookup` reports the
  /// gap and `candidateChars` skips the row.
  OperatorChar characterize(Operation *op, const OperatorEntry &e,
                            int64_t width) const;

  /// Which of \p candidates \p op is realized on, at \p width bits; null for an
  /// empty set. An IP outranks the combinational row whatever their latencies.
  /// Among IPs, one that fits the selection period outranks one that does not;
  /// among misses the least need wins, which is what the period derates to.
  /// Then the shortest, then the cheapest, then the first by symbol.
  const OperatorEntry *
  selectImplementation(ArrayRef<const OperatorEntry *> candidates,
                       int64_t width) const;

  /// The device's combinational row for \p kind, null when it declares none.
  /// The last declared wins, as in `selectImplementation`.
  const OperatorEntry *combEntry(OpKind kind) const;

  std::vector<OperatorEntry> advancedEntries; // rows keyed by raw MLIR name
  std::vector<OperatorEntry> entries;         // abstract rows
  OperatorEntry defaultEntry;
  llvm::StringMap<int64_t> resourcePrices; // one `dcp.resource`, priced
  ArrayAttr muxUses;                       // `dcp.mux`, over (k, width)
  CostAttr muxDelay;      // `dcp.mux` delay over fan-in (ns, at 32-bit width)
  CostAttr muxDelayWidth; // its unitless width factor; null with `muxDelay`
  ArrayAttr chainUses;    // `dcp.chain`, reset-free, over (depth, width)
  double regFloor = 0.0;  // `dcp.device`'s `reg_delay`
  float selectionPeriodNs = std::numeric_limits<float>::infinity();
};

/// The combinational depth, in LUT levels, of a select over \p sources
/// sources: `ceil(log2 k)`, the emitter building a one-hot AND-OR reduction
/// (`EmitContext::oneHotSelect`) whose every level halves the term count. Zero
/// for a single source, which is a wire.
unsigned muxLevels(unsigned sources);

/// A safety factor on the formula fallback below, sized from the gap a one-bit
/// OR row leaves on a wide select. Unused on a device with a measured
/// `dcp.mux` delay row.
inline constexpr double kMuxDelayMargin = 1.4;

/// The routed marginal delay of a one-hot select over \p sources arms of
/// \p width bits, in ns: the device's measured `dcp.mux` delay row, clamped to
/// its measured domain (fan-in past the sweep grows one LUT level per
/// several-fold, which the clamp under-counts slightly). A device without the
/// row is priced at `muxLevels` times a margined one-bit OR row, the
/// conservative direction. Zero for a single source, which is a wire.
double muxCone(const OperatorLibrary &lib, unsigned sources, unsigned width);

/// The device as the compiler reads it: what it can compute and what it can
/// store in. Two peer models of one `dcp.device`, neither part of the other,
/// carried together because scheduling or emitting a region needs both: an
/// operation is timed by an operator row, an access by its storage, and an
/// access's address is arithmetic timed by an operator row.
struct DeviceModel {
  OperatorLibrary operators;
  MemoryLibrary memory;

  static DeviceModel fromModule(ModuleOp module) {
    return {OperatorLibrary::fromModule(module),
            MemoryLibrary::fromModule(module)};
  }
};

/// What the most plentiful resource on a device prices at, and so how much
/// resolution every other price keeps. See the area block above.
inline constexpr int64_t kPriceResolution = 8;

//===----------------------------------------------------------------------===//
// Scheduled-call latency
//
// A plain (non-async) call to an already-scheduled callee is a fixed-latency
// node priced through `calleeStaticLatency`; nullopt for any other op.
//===----------------------------------------------------------------------===//
inline std::optional<std::pair<int64_t, std::string>>
scheduledCallLatency(Operation *op) {
  auto call = dyn_cast<func::CallOp>(op);
  if (!call || op->hasAttr(kAlloAsyncAttr))
    return std::nullopt;
  Operation *callee = calleeOf(op);
  if (!callee)
    return std::nullopt;
  std::optional<int64_t> lat = calleeStaticLatency(callee);
  if (!lat)
    return std::nullopt;
  return std::make_pair(*lat, ("call." + call.getCallee()).str());
}

//===----------------------------------------------------------------------===//
// Operator model: apply a library to a scheduling problem.
//===----------------------------------------------------------------------===//

/// Grid a cone delay is stated on, in ns. A cone is named into its consumer's
/// operator type, so quantize to this or two cones printing alike would cost
/// differently by registration order.
constexpr double kConeDelayQuantum = 0.01;

/// \p delay on that grid.
inline double quantizeCone(double delay) {
  return std::round(delay / kConeDelayQuantum) * kConeDelayQuantum;
}

/// The select an array access drives its port bus through, in ns: a one-hot
/// cone over the holders `recordPortSelectArms` recorded, at the wider of the
/// address and (for a write) the data path. Zero where nothing shares the bus.
/// Charged into the access's setup delay so the cut leaves room for the port
/// cone `bindMemoryPorts` grows after it.
double portSelectDelay(Operation *op, const OperatorLibrary &lib);

/// What one memory or stream access is worth to a schedule. A NodeTiming, not
/// an `OperatorChar`: an access builds no functional unit, so nothing to name,
/// price, or share. Its length and port delay come from the storage (\p
/// memLib); the address cone and port select feeding it are priced against \p
/// opLib's combinational rows, summed here where the scheduling problem needs
/// them.
NodeTiming accessCharacterization(Operation *op, const OperatorLibrary &opLib,
                                  const MemoryLibrary &memLib);

/// The rows an exact solve may choose among for \p op, plus the library's own
/// pick: at least two measured candidates that fit and differ somewhere a
/// schedule can see (latency, a delay, a price). Empty where the realization
/// is not a solver decision: a default realization, a zero-delay rename, a
/// single usable candidate, a zero-latency row with unequal delays
/// (`checkDelays` rejects it), or under \p cyclic a pick the pipelined-only
/// limit drops (an occupancy window varying with the decision would move the
/// interval bound the search starts from).
///
/// An operation with choices joins no static class:
/// `populateOperatorAllocation` skips it, and the exact solve folds it into the
/// class of the row it decides (a shared class), straight-line and modulo
/// alike.
SmallVector<OperatorChar, 2>
selectionCandidates(Operation *op, const OperatorLibrary &lib, bool cyclic);

/// Assign an operator type (latency + chaining delays) to every operation
/// \p problem holds. Three sources for the three kinds of node: an operator row
/// for a compute op, the callee's own schedule for a sync call, and the storage
/// plus its address cone for an access.
///
/// Over the problem's own operations, not a second walk of the IR: each builder
/// registers every op it walks and nothing changes that set in between.
template <class ProblemT>
void populateOperatorTypes(ProblemT &problem, const OperatorLibrary &lib,
                           const MemoryLibrary &memLib) {
  using namespace circt::scheduling;
  for (Operation *op : problem.getOperations()) {
    NodeTiming t;
    if (isSyncSubKernelCall(op)) {
      // Timed by its callee, between registered boundaries. An INDETERMINATE
      // callee has no length to charge, so the node takes zero and its region
      // waits on the child's `done` instead (`isIndeterminateCall`). A call
      // sits between registered boundaries, so it chains with nothing.
      std::optional<std::pair<int64_t, std::string>> cl =
          scheduledCallLatency(op);
      t.typeName =
          cl ? cl->second
             : ("call." + cast<func::CallOp>(op).getCallee() + ".open").str();
      t.latency = cl ? static_cast<uint32_t>(cl->first) : 0;
    } else if (asMemAccess(op)) {
      t = accessCharacterization(op, lib, memLib);
    } else {
      t = lib.lookup(op).timing;
    }
    Problem::OperatorType opr = problem.getOrInsertOperatorType(t.typeName);
    problem.setLatency(opr, t.latency);
    problem.setIncomingDelay(opr, t.inDelay);
    problem.setOutgoingDelay(opr, t.outDelay);
    problem.setLinkedOperatorType(op, opr);
  }
}

/// Reserve a limit-1 resource, held for `latency + 1` cycles, for every sync
/// sub-kernel call in a counted loop body: one child instance re-fired per
/// iteration, not a pipelined operator, with the loop controller starting the
/// next invocation on the previous one's `done` plus a cycle to re-arm. Keyed
/// per callsite, since distinct calls are distinct instances.
///
/// A straight-line region needs none: each callsite issues once, so there is no
/// second invocation for an occupancy window to hold off.
inline void populateCallOccupancy(ChainingModuloProblem &problem) {
  using P = circt::scheduling::Problem;
  unsigned idx = 0;
  for (Operation *op : problem.getOperations()) {
    std::optional<std::pair<int64_t, std::string>> cl =
        scheduledCallLatency(op);
    if (!cl)
      continue;
    P::ResourceType rsrc =
        problem.getOrInsertResourceType(cl->second + "#" + std::to_string(idx));
    problem.setLimit(rsrc, 1);
    problem.setLinkedResourceTypes(op, SmallVector<P::ResourceType>{rsrc});
    problem.setResourceCycles(op, cl->first + 1);
    ++idx;
  }
}

/// Reserve a private limit-1 resource, held for the operator's whole latency,
/// for every operation on a non-pipelined operator. Such a unit takes one input
/// per latency window, so a modulo schedule re-issuing the operation every II
/// cycles needs `II >= latency`. Without it the model lets a non-pipelined IP
/// run at II=1 and the emitter feeds it faster than it can accept.
///
/// The window is the latency itself, the span `reservationOf` marks the unit
/// busy for, so the interval bound here and the binder's unit check are one
/// number.
///
/// Private per operation, since this prices an operation against itself one
/// iteration on, holding for however many units the region builds. A unit
/// shared between two operations is `populateOperatorAllocation`'s, which
/// declines a non-pipelined operator in a cyclic region for want of a
/// circular-arc colouring, leaving every such operation the unit this bounds.
///
/// Only an IP row can be non-pipelined: comb and default rows are zero-latency
/// and pipelined, and a memory access is timed by its storage.
///
/// A straight-line region needs none: it issues each operation once, so there
/// is no second issue for a window to hold off.
inline void populateOperatorOccupancy(ChainingModuloProblem &problem,
                                      const OperatorLibrary &lib) {
  using P = circt::scheduling::Problem;
  unsigned idx = 0;
  for (Operation *op : problem.getOperations()) {
    if (isSyncSubKernelCall(op))
      continue; // a re-fired child instance, `populateCallOccupancy`'s window
    if (asMemAccess(op))
      continue; // a storage port, whose limit is `populateMemoryResources`'
    OperatorChar c = lib.lookup(op);
    // A one-cycle window is what a pipelined unit already holds, and bounds no
    // interval.
    if (c.pipelined || c.timing.latency < 2)
      continue;
    assert(
        c.identity.realized() &&
        "only an IP row is non-pipelined, and an IP row names a realization");
    P::ResourceType rsrc = problem.getOrInsertResourceType(
        c.identity.key() + "#" + std::to_string(idx));
    problem.setLimit(rsrc, 1);
    SmallVector<P::ResourceType> units;
    if (auto linked = problem.getLinkedResourceTypes(op))
      units.assign(linked->begin(), linked->end());
    units.push_back(rsrc);
    problem.setLinkedResourceTypes(op, units);
    problem.setResourceCycles(op, c.timing.latency);
    ++idx;
  }
}

//===----------------------------------------------------------------------===//
// Allocation model: how many copies of an operator a region builds. Keyed on
// the operator identity rather than the timing row, since only one physical
// operator can host two operations.
//===----------------------------------------------------------------------===//

/// Which operations `populateOperatorAllocation` folds onto shared units. It
/// declares one allocatable resource per operator identity a region could build
/// fewer copies of than it has operations, under these limits:
///
///   * IP identities only. Folding a combinational operator pays for a
///     multiplexer nearly as wide as the operator itself (a 32-bit adder is
///     ~32 LUTs against ~64 of mux).
///   * At least two operations, or there is nothing to fold.
///   * In a cyclic region, a one-cycle occupancy. Past one cycle the
///     reservation window wraps the II and a count per congruence class no
///     longer implies that many units suffice (circular-arc colouring). Acyclic
///     windows form an interval graph, where the count is the chromatic number,
///     so any occupancy is fine.
///
/// `n` instances cost what the device charges for `n`: `n` copies of the
/// measured core plus the multiplexer that many put in front of every operand
/// port, an upper bound since operations sharing a driver need no select and
/// the emitter builds one only where drivers differ.
///
/// `All` declares every operation at the library's own pick. `Static` skips an
/// operation with selection candidates, since the solve may move it off its
/// static identity; the exact model re-composes it into the class of the row it
/// decides, merging with the static members. `Selecting` takes only those
/// skipped operations, at the library's own pick.
enum class AllocationScope { All, Static, Selecting };

template <class ProblemT>
void populateOperatorAllocation(ProblemT &problem, const OperatorLibrary &lib,
                                AllocationScope scope) {
  using namespace circt::scheduling;
  constexpr bool isCyclic = std::is_base_of_v<CyclicProblem, ProblemT>;
  // The loop whose carried values a shared unit re-injects; its own induction
  // variable is not carried.
  Operation *container = problem.getContainingOp();
  Value inductionVar;
  if (auto loop = dyn_cast<LoopLikeOpInterface>(container))
    if (auto iv = loop.getSingleInductionVar())
      inductionVar = *iv;
  // One identity's operations, in problem order. Sorted keying, not insertion
  // order, so two compiles declare the resources in the same order.
  struct OperatorClass {
    llvm::SmallVector<Operation *> ops;
    unsigned occupancy = 1;
    int64_t unitPrice = 0;
    int64_t ports = 0;     // operand ports one instance multiplexes
    int64_t portWidth = 0; // bits each of them carries
    unsigned carried = 0;  // ops reading a loop-carried value (one extra arm)
  };
  std::map<std::string, OperatorClass> byIdentity;
  for (Operation *op : problem.getOperations()) {
    if (isSyncSubKernelCall(op))
      continue; // one child instance per callsite; no unit to fold it onto
    if (asMemAccess(op))
      continue; // a storage port; nothing an operator allocation can fold
    OperatorChar c = lib.lookup(op);
    if (!c.identity.realized() || c.identity.comb)
      continue;
    bool selects = scope != AllocationScope::All &&
                   !selectionCandidates(op, lib, isCyclic).empty();
    if (scope == AllocationScope::Static && selects)
      continue; // the realization is the solver's decision
    if (scope == AllocationScope::Selecting && !selects)
      continue; // already declared by the Static pass
    // A non-pipelined unit is busy for its whole latency; a pipelined one
    // contends only for its issue slot.
    unsigned occ = c.pipelined ? 1 : std::max(1u, c.timing.latency);
    if (isCyclic && occ > 1)
      continue; // a count alone is not sufficient modulo the II
    OperatorClass &cls = byIdentity[c.identity.key()];
    cls.ops.push_back(op);
    cls.occupancy = occ;
    cls.unitPrice = c.price;
    cls.ports = op->getNumOperands();
    cls.portWidth = 0;
    for (Type t : op->getOperandTypes())
      if (t.isIntOrFloat())
        cls.portWidth =
            std::max<int64_t>(cls.portWidth, t.getIntOrFloatBitWidth());
    // A shared unit re-injects a loop-carried operand (the reduction identity)
    // on a select arm of its own, so such an operation prices as two arms.
    if (isCyclic && llvm::any_of(op->getOperands(), [&](Value v) {
          auto barg = dyn_cast<BlockArgument>(v);
          return barg && barg.getOwner()->getParentOp() == container &&
                 v != inductionVar;
        }))
      ++cls.carried;
  }

  for (auto &[key, cls] : byIdentity) {
    if (cls.ops.size() < 2)
      continue;
    auto ceiling = static_cast<unsigned>(cls.ops.size());
    llvm::SmallVector<int64_t> price(ceiling + 1, 0);
    for (unsigned n = 1; n <= ceiling; ++n) {
      // Round-robin, the rule `assignUnits` hands the operations out by:
      // `ceiling % n` instances host one more than the rest.
      unsigned busy = ceiling % n, share = ceiling / n;
      price[n] = n * cls.unitPrice +
                 cls.ports * (busy * lib.muxPrice(share + 1, cls.portWidth) +
                              (n - busy) * lib.muxPrice(share, cls.portWidth));
    }
    // The delay of the same multiplexer, for the solve to hold against the
    // period: the fullest instance hosts `ceil(ceiling / n)` operations, plus
    // one re-injection arm per carried operand among them. Building every
    // instance shares nothing and charges nothing.
    llvm::SmallVector<double> headroom(ceiling + 1, 0.0);
    for (unsigned n = 1; n < ceiling; ++n) {
      unsigned members = (ceiling + n - 1) / n;
      unsigned arms = members + std::min(members, cls.carried);
      headroom[n] =
          muxCone(lib, arms,
                  static_cast<unsigned>(std::max<int64_t>(1, cls.portWidth)));
    }
    Problem::ResourceType rsrc = problem.getOrInsertResourceType(key);
    assert(!problem.getAllocatable(rsrc) && "one identity is declared once");
    problem.setAllocatable(
        rsrc, typename ProblemT::AllocatableUnit{ceiling, std::move(price),
                                                 std::move(headroom)});
    for (Operation *op : cls.ops) {
      llvm::SmallVector<Problem::ResourceType> units;
      if (auto linked = problem.getLinkedResourceTypes(op))
        units.assign(linked->begin(), linked->end());
      units.push_back(rsrc);
      problem.setLinkedResourceTypes(op, units);
      problem.setResourceCycles(op, cls.occupancy);
    }
  }
}

} // namespace mlir::allo

#endif // ALLO_SCHEDULING_OPERATORLIBRARY_H
