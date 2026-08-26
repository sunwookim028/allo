/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// Everything checked between the model being sealed and hardware being built,
// cut by who is at fault: the design (`checkStorageLegality`,
// `checkStallContracts`), this backend (`checkEmitterSubset`), or an upstream
// pass (`assertModelInvariants`).
//===----------------------------------------------------------------------===//

#include "allo/Microarch/Verification.h"

#include "allo-c/Schedule.h"             // kPartitionAttr
#include "allo/Microarch/Naming.h"       // operatorModuleName, memOwnerName
#include "allo/Microarch/Primitives.h"   // memAddrWidth
#include "allo/Microarch/Report.h"       // TimingPath
#include "allo/Scheduling/MemoryModel.h" // datapathWidth
#include "allo/Support/Logging.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h" // memref::GetGlobalOp
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Format.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::allo;
using namespace mlir::allo::logging;

namespace mlir::allo::uarch {

//===----------------------------------------------------------------------===//
// 1. What the design asks for and this device cannot give.
//===----------------------------------------------------------------------===//

namespace {

/// What the design asks of its arrays and this device cannot give, reported
/// against the user. `logging::error` and `logging::warn` only.
LogicalResult checkStorageLegality(dcp::DCPathModuleOp func,
                                   const Datapath &dp) {
  // A kernel with no schedulable region computes nothing.
  if (dp.regions.empty())
    warn(Stage::Emit, func)
        << "Kernel '" << func.getSymName()
        << "' has no schedulable region: it emits as hardware that does "
           "nothing and completes immediately";

  for (const MemUnit &m : dp.mems) {
    // A constant table is combinational logic with no port limit, so a
    // partition of one buys no bandwidth.
    auto gg = m.memref.getDefiningOp<memref::GetGlobalOp>();
    if (m.isRom && gg) {
      auto global = SymbolTable::lookupNearestSymbolFrom<memref::GlobalOp>(
          gg, gg.getNameAttr());
      if (global && global->hasAttr(kPartitionAttr))
        warn(Stage::Emit, func)
            << "Partition on the constant table '" << gg.getName()
            << "' buys nothing: a read-only table is realized as "
               "combinational logic, which reads through as many ports as "
               "the schedule asks for";
    }

    // The copies the scheduler priced the array at are the read bandwidth it
    // reserved, so a binding taking more has bought bandwidth no cycle was cut
    // for. Warned rather than refused, the schedule being already fixed. The
    // concurrency says which side is at fault: equal to the ports, the schedule
    // does ask for them all at once and the array wants partitioning or a wider
    // row; below them, the binding separated accesses that never meet.
    if (m.instances > m.ports.copies())
      logging::log(Level::Warn, Stage::Emit, m.memref.getLoc())
          << memOwnerName(dp, m) << ": " << m.readPortsBuilt
          << " read ports on " << m.storage << " take " << m.instances
          << " copies of it per bank, past the " << m.ports.copies()
          << " the schedule reserved (" << m.readConcurrency
          << " of its reads may issue in one cycle)";

    // One boundary group is one interface the caller has to build, and the
    // ports bound for the array are all this module can drive at once on a
    // bank. Only an addressed argument has a budget: an internal array
    // publishes no group, and a scattered one publishes cells rather than
    // buses. Warned rather than refused, the interface being the manifest the
    // caller was already compiled against.
    unsigned budget = m.numBanks * m.portsBuilt;
    if (budget && m.boundaryPorts > budget)
      logging::log(Level::Warn, Stage::Emit, m.memref.getLoc())
          << memOwnerName(dp, m) << ": the caller provides " << m.boundaryPorts
          << " interface groups for this argument, "
          << (m.boundaryPorts - budget)
          << " past what this module can drive at once (" << m.portsBuilt
          << " ports per bank, " << m.numBanks
          << " banks). Every accessor takes a group of its own, so a "
             "sub-kernel reaching the array adds one whether or not its port "
             "already shares a bus with another's";

    // Only a write set reaches this: every copy of a row needs every write, so
    // one instance's write ports are the ceiling however many copies are built,
    // and on a pooled row writes that fill the pool leave a read no port
    // anywhere. Reads alone never reach it, the copies being what serves them.
    //
    // `characterize` budgets a derived row one write short of its pool, which
    // leaves two ways in: a topology the user stated, and writers of concurrent
    // regions, which are billed apart and only meet here.
    if (m.realization() == MemUnit::Realization::Ram && !m.fitsStorage()) {
      error(Stage::Emit, Code::StoragePortsExceeded, m.memref.getDefiningOp())
          << "Array " << m.memref.getType() << " is built with "
          << m.writePortsBuilt << " concurrent write ports per bank, and one "
          << m.storage << " has " << m.ports.describe()
          << ", so no number of copies holds it. "
          << (m.ports.stated ? "Ask `bind_storage` for a topology with more "
                               "write ports, partition the array so the "
                               "writers land in different banks, or "
                             : "Partition the array so the writers land in "
                               "different banks, or ")
          << "let fewer of them issue at once. Only accesses the model proves "
             "never issue together share a port";
      return failure();
    }
  }
  return success();
}

/// `ce` is the only IP port ABI the emitter realizes. `free` has no enable, so
/// it keeps clocking and desynchronizes in a back-pressured region, but is
/// fine elsewhere; `elastic` is rejected before scheduling.
LogicalResult checkStallContracts(const Datapath &dp) {
  if (llvm::all_of(dp.units, [](const FuncUnit &u) {
        return u.identity.comb || u.stall == allo::StallContractEnum::Ce;
      }))
    return success();

  llvm::SmallDenseSet<unsigned> backPressured;
  for (const StreamChannel &s : dp.streams)
    for (const StreamChannel::Access &acc : s.accesses)
      backPressured.insert(acc.region);
  llvm::DenseMap<UnitId, unsigned> unitRegion;
  for (const RegionBlock &rb : dp.regions)
    for (UnitId uid : rb.units)
      unitRegion[uid] = rb.id;
  for (const FuncUnit &u : dp.units) {
    if (u.identity.comb || u.stall == allo::StallContractEnum::Ce)
      continue;
    if (backPressured.count(unitRegion.lookup(u.id))) {
      error(Stage::Emit, Code::StallContractUnusable, u.repOp())
          << "Operator IP '" << operatorModuleName(u)
          << "' is free-running (no clock enable) but sits in a stream region, "
             "whose datapath freezes under back-pressure; the IP would keep "
             "advancing and fold a stale result. Declare style='ce'";
      return failure();
    }
  }
  return success();
}

} // namespace

//===----------------------------------------------------------------------===//
// 2. What this emitter does not lower yet.
//===----------------------------------------------------------------------===//

namespace {

/// The multiplexer delay a shared binding adds to a unit's input cone,
/// propagated along the chains it lengthens. It is the most any branch adds,
/// not what the worst-arriving branch adds, a refusal having to bound every
/// branch.
///
/// The scheduler proved `z(op) + inDelay(op) <= period` over a datapath whose
/// unit inputs are all driven directly, and each addition shifts its consumer's
/// arrival by a constant. The delta is additive along a combinational path, so
/// propagating it alone against each op's remaining sub-cycle slack is exact.
struct AddedDelay {
  AddedDelay(const Datapath &dp, const OperatorLibrary &lib)
      : dp(dp), lib(lib) {}

  const Datapath &dp;
  const OperatorLibrary &lib; // prices each select cone (`muxCone`)
  llvm::DenseMap<UnitId, double> memo;

  /// What arrives at \p id's input ports, its own delay excluded.
  double ofUnit(UnitId id) {
    auto seen = memo.find(id);
    if (seen != memo.end())
      return seen->second;
    // Seeded before the walk, so a fused recurrence's self-referential input
    // terminates instead of recursing forever.
    memo[id] = 0.0;
    double added = 0.0;
    for (const Source &in : dp.units[id].inputs)
      added = std::max(added, ofSource(in));
    memo[id] = added;
    return added;
  }

  double ofSource(const Source &s) {
    if (s.kind == Source::Kind::Mux) {
      const Mux &m = dp.muxes[s.id];
      double in = 0.0;
      for (const Source &src : m.sources)
        in = std::max(in, ofSource(src));
      return in + muxCone(lib, m.sources.size(), datapathWidth(m.type));
    }
    // Anything else is held when the cycle starts: a register tap, a port, a
    // literal, a survivor, or a unit whose own output is registered.
    if (s.kind != Source::Kind::Unit || dp.units[s.id].latency)
      return 0.0;
    return ofUnit(s.id);
  }
};

/// When each value settles within its cycle, which input set it, and what the
/// cell producing it added. A reported path is one chain of this walk, so its
/// steps sum to its total.
///
/// Arrivals are recomposed from the same device rows the schedule was cut
/// against (`FuncUnit::inDelay` is marginal, the register floor charged once at
/// the start point) rather than read off `z`, which carries whatever slack the
/// solve left.
struct PathTrace {
  PathTrace(const Datapath &dp, const OperatorLibrary &lib)
      : dp(dp), lib(lib), floor(lib.registerFloor()) {}

  const Datapath &dp;
  const OperatorLibrary &lib;
  /// One register hop with no logic in it: clock-to-out at the launching end
  /// plus setup at the capturing one. Every path pays it once.
  double floor;

  struct Arrival {
    double at = 0.0;  // when the value settles, from the launching edge
    double own = 0.0; // what the cell producing it adds
    double cut = 0.0; // how much of `at` the cones grown after the cut added
    Source from;      // the input that set `at`; None at a start point
    std::string what; // the cell, as the report names it
  };

  /// The schedule's own view of \p s: what reaches it through cells the solve
  /// priced, the cones grown after it excluded.
  double scheduled(const Source &s) {
    Arrival a = of(s);
    return a.at - a.cut;
  }

  Arrival of(const Source &s) {
    if (!s)
      return {};
    uint64_t k = (uint64_t)s.kind << 48 | (uint64_t)s.id << 16 | s.outPort;
    if (auto it = memo.find(k); it != memo.end())
      return it->second;
    // Seeded before recursing, so a fused recurrence's self-referential input
    // terminates at its own pipeline register.
    memo[k] = launch("a recurrence register");
    Arrival a = derive(s);
    memo[k] = a;
    return a;
  }

  /// The steps into \p s, start point first, appended to \p out.
  void stepsInto(const Source &s, std::vector<TimingStep> &out) {
    Arrival a = of(s);
    if (a.from)
      stepsInto(a.from, out);
    out.push_back({a.what, a.own});
  }

private:
  llvm::DenseMap<uint64_t, Arrival> memo;

  Arrival launch(llvm::StringRef what) {
    return {floor, floor, 0.0, Source{}, ("launch: " + what).str()};
  }

  /// The input that settles last. A cell with none launches from the floor.
  Arrival worst(llvm::ArrayRef<Source> ins) {
    Arrival best;
    best.at = floor;
    for (const Source &in : ins) {
      if (!in)
        continue;
      Arrival a = of(in);
      if (!best.from || a.at > best.at) {
        best = a;
        best.from = in;
      }
    }
    return best;
  }

  Arrival derive(const Source &s) {
    switch (s.kind) {
    case Source::Kind::Unit: {
      const FuncUnit &u = dp.units[s.id];
      std::string what =
          (u.identity.realizationName() + " at " +
           llvm::Twine(datapathWidth(u.identity.resultType)) + " bits")
              .str();
      // A registered result launches its consumers rather than chaining into
      // them.
      if (u.latency)
        return launch(what + " (registered)");
      Arrival in = worst(u.inputs);
      return {in.at + u.inDelay, u.inDelay, in.cut, in.from, what};
    }
    case Source::Kind::Mux: {
      const Mux &m = dp.muxes[s.id];
      unsigned width = datapathWidth(m.type);
      double cone = muxCone(lib, m.sources.size(), width);
      Arrival in = worst(m.sources);
      return {in.at + cone, cone, in.cut + cone, in.from,
              ("a sharing multiplexer, " + llvm::Twine(m.sources.size()) +
               ":1 at " + llvm::Twine(width) + " bits")
                  .str()};
    }
    case Source::Kind::Mem: {
      const MemUnit &m = dp.mems[s.id];
      const MemUnit::Access &acc = m.accesses[s.outPort];
      std::string what = "a read of " + memOwnerName(dp, m);
      if (m.readLatency)
        return launch(what);
      // A ROM or a scattered array reads combinationally, so its own cone is
      // on the reader's path. `inDelay` covers the address arithmetic and the
      // read itself, undecomposed.
      Arrival in = worst(acc.addr);
      return {in.at + acc.inDelay, acc.inDelay, in.cut, in.from,
              what + " (combinational)"};
    }
    case Source::Kind::Stream:
      return launch("a stream read");
    case Source::Kind::Reg:
      return launch("a delay register");
    case Source::Kind::Counter:
      return launch("a loop counter");
    case Source::Kind::Survivor:
      return launch("a survivor register");
    case Source::Kind::Call:
      return launch("a sub-kernel result");
    case Source::Kind::IO:
      return launch("an input port");
    case Source::Kind::Const:
      // Wires from a tie-off, but the capture at the far end still costs
      // setup, charged here as `floor`.
      return launch("a constant");
    case Source::Kind::None:
      break;
    }
    return {};
  }
};

/// What each access ends its path with, on top of whatever reaches it: the
/// address arithmetic still on the setup path (`addrSetup`), the select its
/// (bank, port) colour carries (one arm per holder, this module's accesses and
/// its children's ports alike), and the port's own delay. A crossbar access is
/// alone on its colour, and its bank crossbar is not priced.
struct Tail {
  llvm::SmallVector<TimingStep, 3> addr, data;
  bool addrRegistered = false; // the address launches from its delay register
};

/// The tail of every memory and stream access, keyed by the op that issues it.
/// A read grows no data tail: only a write captures at a data port.
llvm::DenseMap<Operation *, Tail> accessTails(const Datapath &dp,
                                              const OperatorLibrary &lib) {
  llvm::DenseMap<std::tuple<MemId, unsigned, unsigned>, unsigned> holders;
  for (const MemUnit &m : dp.mems)
    for (const MemUnit::Access &acc : m.accesses)
      if (acc.plan == PortPlan::Coloured)
        ++holders[{m.id, acc.staticBank.value_or(0), acc.port}];
  for (const CallUnit &cu : dp.calls)
    for (const CallUnit::MemArg &ma : cu.memArgs)
      if (ma.plan == PortPlan::Coloured)
        ++holders[{ma.mem, ma.bank, ma.port}];
  llvm::DenseMap<Operation *, Tail> tails;
  for (const MemUnit &m : dp.mems) {
    std::string owner = memOwnerName(dp, m);
    for (const MemUnit::Access &acc : m.accesses) {
      Tail &t = tails[acc.op];
      // With the term sum landing in a delay register, the address path starts
      // at that register rather than at the operands feeding it.
      t.addrRegistered = acc.addrDelay > 0;
      if (acc.addrSetup > 0.0)
        t.addr.push_back({"the address arithmetic of " + owner, acc.addrSetup});
      if (acc.plan == PortPlan::Coloured) {
        unsigned k =
            holders.lookup({m.id, acc.staticBank.value_or(0), acc.port});
        double sel = muxCone(lib, k, memAddrWidth(m));
        if (sel > 0.0)
          t.addr.push_back(
              {("a shared-port address select, " + llvm::Twine(k) + ":1").str(),
               sel});
        double dsel = acc.isWrite ? muxCone(lib, k, m.width) : 0.0;
        if (dsel > 0.0)
          t.data.push_back(
              {("a shared-port data select, " + llvm::Twine(k) + ":1").str(),
               dsel});
        // The colouring against the ceiling the cut reserved for it
        // (`recordPortSelectArms`). It counts a child's port groups as one, so
        // an overrun is possible; the path report carries what it costs.
        if (double built = std::max(sel, dsel);
            built > acc.selectDelay + kConeDelayQuantum)
          debug(Stage::Emit, acc.op)
              << "the port colouring put " << k
              << " holders on this bus, worth " << llvm::format("%.2f", built)
              << " ns of select against the "
              << llvm::format("%.2f", acc.selectDelay)
              << " ns the schedule reserved";
      }
      TimingStep port{"the " + m.storage + " port of " + owner, acc.portDelay};
      t.addr.push_back(port);
      if (acc.isWrite)
        t.data.push_back(std::move(port));
    }
  }
  for (const StreamChannel &ch : dp.streams)
    for (const StreamChannel::Access &acc : ch.accesses)
      tails[acc.op].data.push_back({"a stream port", acc.inDelay});
  return tails;
}

/// The two arrival models checked against each other, and a warning when a
/// shared binding grew a multiplexer past the period the schedule was cut
/// against. The emitted RTL is functionally correct; only the target period is
/// at risk, and the binding is a choice the user can withdraw, so this reports
/// rather than refusing.
void checkBindingMuxHeadroom(const Datapath &dp, float cycleTime,
                             const OperatorLibrary &lib, bool plannedBinding,
                             PathTrace &trace) {
  // One picosecond of slop, the resolution the scheduler's own model carries.
  constexpr double kSlop = 1e-3;
  AddedDelay added(dp, lib);
  llvm::DenseMap<Operation *, double> sinks = sinkTails(dp);

  for (const FuncUnit &u : dp.units) {
    // Cross-check of the two arrival models: with the cones grown after the cut
    // excluded, an input recomposed from the device rows arrives no later than
    // the sub-cycle start the solve placed the op at. `FuncUnit::inDelay` is
    // the solve's own stamped number, so prices cannot disagree; structure can,
    // as with a recurrence loop the recomposition charges into one cycle where
    // the hardware splits it at the carry register. Hence a diagnostic, not an
    // assert.
    //
    // The tightest bound op is both the arrival to check against and the anchor
    // of the warning below.
    const FuncUnit::BoundOp *worst = &u.boundOps.front();
    for (const FuncUnit::BoundOp &bo : u.boundOps)
      if (bo.z && (!worst->z || *bo.z > *worst->z))
        worst = &bo;
    double placed = worst->z.value_or(0.0);
    for (const Source &in : u.inputs) {
      if (!in)
        continue;
      double sched = trace.scheduled(in);
      if (sched > placed + kSlop)
        debug(Stage::Emit, u.repOp())
            << "an input settles at " << llvm::format("%.2f", sched)
            << " ns, past the " << llvm::format("%.2f", placed)
            << " ns sub-cycle start the solve placed this operation at; the "
               "two arrival models disagree about this path's structure";
    }
    double mux = added.ofUnit(u.id);
    // No binding-grown cone reaches this unit: a structural overrun (a
    // recurrence-identity select on a tight schedule) has no binding remedy and
    // reports through the published paths instead.
    if (mux <= kSlop)
      continue;
    std::optional<double> slack = unitSlack(u, lib, cycleTime, &sinks);
    if (slack && mux <= *slack + kSlop)
      continue;
    // A planned fold realizes the solve's own allocation, which held
    // `z + inDelay + headroom(N) <= period` for every operation it folded, so
    // an overrun on a priced unit breaks that contract and stays an assert. An
    // unpriced unit the solve never placed only warns, below.
    assert(!(plannedBinding && slack) &&
           "a planned binding grew a select cone past the period the schedule "
           "solve reserved headroom for; the allocation headroom model "
           "(`addAllocationHeadroom`) and the emitted cone disagree");
    // `mux` covers the whole input cone, so it may come from a shared
    // predecessor rather than from a multiplexer on this unit.
    warn(Stage::Emit, worst->op)
        << "Binding put " << llvm::format("%.2f", mux)
        << " ns of multiplexer on the path reaching this operation (its unit "
           "is shared between "
        << u.boundOps.size() << " operations), which is "
        << llvm::format("%.2f", mux - slack.value_or(0.0))
        << " ns more than the schedule left it against a "
        << llvm::format("%.2f", cycleTime)
        << " ns clock. The schedule was cut before the multiplexer existed, so "
           "this would miss timing in silicon. Use binding='trivial' for this "
           "kernel, or raise the target period";
  }
}

/// Every capture point, appended to \p paths. Prefix-free by construction, so
/// no path is a piece of another: an interior combinational cell is not a
/// capture, and a unit's own input port is one only where the unit registers
/// it.
void appendCapturePaths(const Datapath &dp, float cycleTime,
                        const llvm::DenseMap<Operation *, Tail> &tails,
                        PathTrace &trace, std::vector<TimingPath> &paths) {
  forEachSource(dp, [&](const Source &s, const SourceSite &site) {
    // An absent driver hangs no path; the reduced-address case reads stride
    // registers instead, priced with them by `appendStridePaths`.
    if (!s)
      return;
    llvm::ArrayRef<TimingStep> tail;
    bool registered = false;
    switch (site.slot) {
    case SourceSite::Slot::UnitInit:
    case SourceSite::Slot::MuxInput:
      return; // the interior of a cone, ending at the slot that consumes it
    case SourceSite::Slot::UnitInput: {
      // A combinational unit hands its result on; only a registered one
      // captures here.
      auto it = dp.opToUnit.find(site.op);
      assert(it != dp.opToUnit.end() &&
             "a unit's rep op is registered in opToUnit");
      if (!dp.units[it->second].latency)
        return;
      break;
    }
    case SourceSite::Slot::MemAddress: {
      const Tail &t = tails.at(site.op);
      tail = t.addr;
      registered = t.addrRegistered;
      break;
    }
    case SourceSite::Slot::MemWriteData:
    case SourceSite::Slot::StreamData:
    case SourceSite::Slot::StreamPredicate:
      tail = tails.at(site.op).data;
      break;
    default:
      break; // a register, a survivor, a result or a boundary captures it
    }

    TimingPath p;
    p.endpoint = site.describe();
    if (site.op)
      p.where = logging::detail::describe(site.op->getLoc());
    else if (site.slot == SourceSite::Slot::RegisterInput &&
             dp.regs[site.index].value)
      // A register is a model cell and owns no op, so it is anchored on the
      // value it carries.
      p.where = logging::detail::describe(dp.regs[site.index].value.getLoc());
    if (registered)
      p.steps.push_back({"launch: the address delay register", trace.floor});
    else
      trace.stepsInto(s, p.steps);
    p.steps.insert(p.steps.end(), tail.begin(), tail.end());
    for (const TimingStep &st : p.steps)
      p.total += st.delay;
    p.slack = cycleTime - p.total;
    paths.push_back(std::move(p));
  });
}

/// The stride-register update, the one reg-to-reg cone with no scheduler
/// counterpart, appended to \p paths. Priced off the emitted shape: the step
/// add, the carry add, the wrap compare beside its fix, and the wrap plus
/// issue/running selects, each a marginal row over the one register floor the
/// hop already pays.
void appendStridePaths(const Datapath &dp, float cycleTime,
                       const OperatorLibrary &lib,
                       std::vector<TimingPath> &paths) {
  for (const RegionBlock &rb : dp.regions)
    for (const RegionBlock::AddrStride &st : rb.addrStrides) {
      double sel = lib.combMarginalDelay(OpKind::Select, st.width);
      TimingPath p;
      p.endpoint = "an address stride register";
      p.where = logging::detail::describe(rb.op->getLoc());
      p.steps.push_back(
          {"launch: an address stride register", lib.registerFloor()});
      if (st.step)
        p.steps.push_back(
            {"the stride step", lib.combMarginalDelay(OpKind::Add, st.width)});
      if (st.hasCarry)
        p.steps.push_back({"the carry from the digit below",
                           lib.combMarginalDelay(OpKind::Add, st.width)});
      if (st.wrap)
        p.steps.push_back(
            {"the wrap test and its fix",
             std::max(lib.combMarginalDelay(OpKind::Cmp, st.width),
                      lib.combMarginalDelay(OpKind::Sub, st.width)) +
                 sel});
      p.steps.push_back({"the issue and running selects", 2 * sel});
      for (const TimingStep &s : p.steps)
        p.total += s.delay;
      p.slack = cycleTime - p.total;
      paths.push_back(std::move(p));
    }
}

/// Every combinational path built after the cut still settles within the
/// period: the multiplexers a shared binding grew in front of the units, and
/// the select the port colouring grew in front of an access's bus. An overrun
/// is a quality-of-result finding, not a refusal: a unit overrun warns (the
/// binding is withdrawable) and every other slot is reported through \p paths.
void checkCombPathsMeetPeriod(const Datapath &dp, float cycleTime,
                              const OperatorLibrary &lib, bool plannedBinding,
                              std::vector<TimingPath> &paths) {
  PathTrace trace(dp, lib);
  checkBindingMuxHeadroom(dp, cycleTime, lib, plannedBinding, trace);
  llvm::DenseMap<Operation *, Tail> tails = accessTails(dp, lib);
  appendCapturePaths(dp, cycleTime, tails, trace, paths);
  appendStridePaths(dp, cycleTime, lib, paths);
}

/// Shapes this backend does not lower yet, including the one the clock rules
/// out: the schedule was cut against \p cycleTime (ns) over a datapath with no
/// sharing muxes and no port selects, so what grows them is measured here, at
/// every capture point `forEachSource` enumerates, and appended to \p paths.
/// \p lib prices the muxes and the units they feed. A binding-grown unit
/// overrun warns, the binding being withdrawable; every other path is reported
/// and not refused, missing a target period being a quality-of-result finding
/// rather than an illegal design. \p plannedBinding says the folds realize the
/// schedule solve's own allocation, which reserved headroom for every select it
/// bought, so a priced unit overrun there is a broken invariant (an assert),
/// not a warning.
LogicalResult checkEmitterSubset(dcp::DCPathModuleOp func, const Datapath &dp,
                                 float cycleTime, const OperatorLibrary &lib,
                                 bool plannedBinding,
                                 std::vector<TimingPath> &paths) {
  // An unresolved (None) Source is a cross-region SSA hand-off the builder
  // could not thread; reject it here rather than asserting in `resolveSource`.
  // `forEachSource` owns the slot list and which slots may be empty.
  bool found = false;
  SourceSite badSite{};
  forEachSource(dp, [&](const Source &s, const SourceSite &site) {
    if (found || !site.required || s)
      return;
    found = true;
    badSite = site;
  });
  if (found) {
    // Wording matches the builder's own hand-off rejection.
    unsupported(Stage::Emit, Code::CrossRegionHandOff,
                badSite.op ? badSite.op : func.getOperation())
        << "A cross-region value hand-off is not lowered yet: "
        << badSite.describe() << " is unresolved";
    return failure();
  }

  // A skew hands out one port per bank per lane, and a lane is assigned from
  // the accesses of this module, so a sub-kernel's port belongs to no lane.
  for (const CallUnit &cu : dp.calls)
    for (const CallUnit::MemArg &ma : cu.memArgs)
      if (ma.plan == PortPlan::Lane) {
        unsupported(Stage::Emit, Code::SkewedArgumentToCallee,
                    dp.mems[ma.mem].memref.getDefiningOp())
            << "Array " << dp.mems[ma.mem].memref.getType()
            << " is skew-partitioned and passed to a sub-kernel, which is not "
               "lowered yet: a skewed bank serves a whole lane from one port, "
               "and a child's port belongs to no lane. Drop the skew on this "
               "array, or inline the callee so its accesses take lanes of "
               "their own";
        return failure();
      }

  // Condition timing: a flushing leaf while or guard samples it in-cycle,
  // needing a stage-0 Unit or settled Survivor, while a sequential CHECK/RUN
  // while waits t_cond cycles. A `None` is rejected above.
  auto conditionOk = [&](const Source &s, bool sequential) {
    switch (s.kind) {
    // A scheduled prologue predicate is settled at the region start.
    case Source::Kind::Survivor:
      return true;
    case Source::Kind::Unit:
      return sequential || dp.units[s.id].boundOps[s.outPort].stage == 0;
    default:
      return false; // a memory / IP / raw driver
    }
  };
  for (const RegionBlock &rb : dp.regions) {
    // Which of the two while controllers runs is the stored shape: a Container
    // while is the sequential CHECK/RUN one, a Leaf while the flushing one.
    if (rb.conditional && !conditionOk(rb.condition,
                                       /*sequential=*/rb.shape ==
                                           RegionBlock::Shape::Container)) {
      unsupported(Stage::Emit, Code::PredicateNotCombinational, rb.op)
          << "A while loop with a non-combinational (memory-/IP-dependent) "
             "condition is not lowered yet";
      return failure();
    }
    if (rb.shape == RegionBlock::Shape::Guard &&
        !conditionOk(rb.condition, /*sequential=*/false)) {
      unsupported(Stage::Emit, Code::PredicateNotCombinational, rb.op)
          << "A guard with a non-combinational predicate is not lowered yet";
      return failure();
    }
  }
  // A leaf `while` with an in-loop store needs no check: emitWrites gates the
  // store's write-enable by `issue & cond`, so a doomed exit iteration commits
  // nothing.

  checkCombPathsMeetPeriod(dp, cycleTime, lib, plannedBinding, paths);
  return success();
}

} // namespace

//===----------------------------------------------------------------------===//
// 3. Invariants an upstream pass owns, asserted at this seam.
//===----------------------------------------------------------------------===//

// Whether a counted container holds no work of its own: the reifier gives every
// run of loose ops its own child region, so this checks what a unit READS
// rather than whether units exist. A conditional container is exempt.
[[maybe_unused]] static bool containerOwnsNoDatapath(const RegionBlock &rb,
                                                     const Datapath &dp) {
  if (!rb.memAccesses.empty() || !rb.streamAccesses.empty() ||
      !rb.callUnits.empty())
    return false;
  for (UnitId uid : rb.units)
    for (const Source &s : dp.units[uid].inputs)
      if (s.kind == Source::Kind::Survivor &&
          llvm::is_contained(rb.children, s.id))
        return false;
  return true;
}

/// Invariants an upstream pass owns, asserted at this seam. `assert` only, so
/// this compiles away in a release build.
static void assertModelInvariants(const Datapath &dp) {
#ifndef NDEBUG
  // Memory rows the scheduler honors: a structure realizing them differently
  // would place every consumer on the wrong cycle.
  for (const MemUnit &m : dp.mems) {
    assert((!m.romInit || m.numBanks == 1) &&
           "`PreVerification` refuses a banked array declared with contents");
    assert(m.writeLatency >= 1 && "a 0-cycle write port reached emission");
    // The scatter realization is registers, and the emitter builds them at
    // exactly that timing: a read is a combinational select over the cells and
    // a store lands on the next edge, neither carrying a delay to absorb one.
    assert((!m.scattered || (m.readLatency == 0 && m.writeLatency == 1)) &&
           "a scattered memory must be timed as registers");
  }

  // `elastic` is rejected before scheduling, and every cell reaching the
  // datapath is placed by a solve, which stamps the sub-cycle start it proved.
  for (const FuncUnit &u : dp.units) {
    assert(u.stall != allo::StallContractEnum::Elastic &&
           "an elastic IP reached emission");
    // Operator realizability is settled before scheduling: an op with neither
    // an IP row nor a `combKindOf` lowering never becomes a `dcp.compute`.
    assert(u.identity.realized() &&
           "an unrealizable operator reached emission");
    for (const FuncUnit::BoundOp &bo : u.boundOps)
      assert(bo.z &&
             "a cell reached the datapath the scheduling stage never placed");
  }

  // Region shapes, mirroring `emitRegion`'s dispatch on the same stored
  // discriminant.
  for (const RegionBlock &rb : dp.regions) {
    // The op verifier already enforces that a counted `dcp.pipeline` carries
    // its trip either as the `trip` attribute or as the `dynamicBound` operand.
    assert((rb.kind != RegionBlock::Kind::Cyclic || rb.conditional ||
            rb.tripCount || rb.ubSource) &&
           "a counted cyclic region reached emission with neither a constant "
           "nor a dynamic trip; the reifier owns that");
    // `emitLoopCall` advances on the child's `done`, so it would silently drop
    // a second child or any loose datapath.
    assert(
        (rb.shape != RegionBlock::Shape::CallNode ||
         (rb.callUnits.size() <= 1 && rb.units.empty() && rb.regs.empty())) &&
        "a loop body holding a sub-kernel call alongside other work reached "
        "the leaf loop-over-calls controller; the scheduler must decompose "
        "it into sub-regions");
    assert((rb.shape != RegionBlock::Shape::Container || rb.conditional ||
            containerOwnsNoDatapath(rb, dp)) &&
           "a counted container reached emission carrying work of its own; the "
           "reifier gives every run of loose ops a child region");
    // A container's own units are the gating logic its children read, not a
    // datapath: it has no per-iteration issue pulse to time a recurrence
    // identity against. A counted container's predicate is sampled in-cycle,
    // where a conditional one's condition cone may take `t_cond` cycles and so
    // may be an IP.
    if (rb.shape != RegionBlock::Shape::Container)
      continue;
    for (UnitId uid : rb.units) {
      const FuncUnit &u = dp.units[uid];
      assert(llvm::all_of(
                 u.inputInits,
                 [](llvm::ArrayRef<Source> inits) { return inits.empty(); }) &&
             "a container's own unit carries a recurrence init, which it has "
             "no issue pulse to time");
      assert((u.identity.comb || rb.conditional) &&
             "a counted container's own unit must be native (comb)");
    }
  }

  // `verify-rtl-legality` owns the shapes a CONCURRENT container admits and the
  // caller/callee partition agreement, both settled before scheduling.

  // Stream protocol: a channel's {data,valid,ready} triple is time-shared by
  // all its accesses, sound only if the scheduler keeps them ordered and
  // non-overlapping. Ends are checked pre-schedule; timing is checked here.
  for (const StreamChannel &s : dp.streams)
    // Distinct cycles in program order within a region, spanning under one II.
    // Per DIRECTION, since that is what shares a wire: a put drives
    // {data, valid} and a get {ready}, so a local channel's ends may coincide.
    for (const RegionBlock &rb : dp.regions)
      for (bool put : {false, true}) {
        const StreamChannel::Access *first = nullptr, *prev = nullptr;
        for (AccRef r : rb.streamAccesses) {
          if (r.id != s.id)
            continue;
          const StreamChannel::Access &acc = s.accesses[r.idx];
          if (acc.isPut != put)
            continue;
          assert((!prev || acc.stage > prev->stage) &&
                 "two accesses to one stream are scheduled on the same cycle, "
                 "or out of program order; they share a single handshake, so "
                 "their token order is lost. The scheduler owns this");
          prev = &acc;
          first = first ? first : &acc;
        }
        assert((!prev || !rb.ii || prev->stage - first->stage < *rb.ii) &&
               "accesses to one stream span a whole initiation interval, so "
               "successive iterations overlap on its handshake. The scheduler "
               "owns this");
      }

  // Every access is listed by exactly the region that issues it, and exactly
  // the EXTERNAL accesses hold a boundary port slot.
  unsigned listed = 0;
  for (const RegionBlock &rb : dp.regions) {
    listed += rb.memAccesses.size();
    for (AccRef r : rb.memAccesses)
      assert(dp.mems[r.id].accesses[r.idx].region == rb.id &&
             "a region lists an access another region issues");
  }
  for (const MemUnit &m : dp.mems) {
    // A scattered argument's ports are per element, so its accesses hold no
    // port slot: each reads every element port and selects. An internal
    // scattered array holds the same cells as registers, which reach no
    // boundary.
    assert(m.elemPorts.size() ==
               (m.scattered && m.external ? m.depthWords : 0) &&
           "element ports belong to exactly the scattered arguments, one per "
           "element");
    for (const MemUnit::Access &acc : m.accesses) {
      --listed;
      bool hasPort = acc.portIdx != MemUnit::Access::kNoPort;
      assert(hasPort == (m.external && !m.scattered) &&
             "a boundary port slot is held by exactly the addressed external "
             "accesses");
      assert((!hasPort || (acc.isWrite ? dp.writePorts : dp.readPorts).size() >
                              acc.portIdx) &&
             "an access's port slot is out of its boundary port list");
      // An argument is never a constant table, and `assignLanes` skips it.
      assert((!m.external ||
              (acc.plan != PortPlan::Table && acc.plan != PortPlan::Lane)) &&
             "an argument array is neither a constant table nor skewed");
    }
  }
  assert(listed == 0 && "every memory access belongs to exactly one region");
  // An indeterminate call finishes at a data-dependent cycle, so nothing
  // statically scheduled may share its region; `enumerateRegions` isolates it.
  // A CONCURRENT container is exempt: nothing in it is placed against a child.
  for (const RegionBlock &rb : dp.regions) {
    if (rb.determinacy == DeterminacyEnum::Concurrent)
      continue;
    if (llvm::none_of(rb.callUnits,
                      [&](CallId cid) { return !dp.calls[cid].latency; }))
      continue;
    bool alone = rb.callUnits.size() == 1 && rb.units.empty() &&
                 rb.regs.empty() && rb.memAccesses.empty() &&
                 rb.streamAccesses.empty();
    assert(alone && "an indeterminate call shares its region with statically-"
                    "scheduled work; the region partitioner must isolate it");
  }
  // A constant table has no write port for anyone to master. A child may READ
  // one, but a writing port group would have nowhere to land.
  for (const CallUnit &cu : dp.calls)
    for (const CallUnit::MemArg &ma : cu.memArgs)
      assert(!(dp.mems[ma.mem].isRom && ma.isWrite) &&
             "a sub-kernel writes the ports of a constant table");
#else
  (void)dp;
#endif
}

/// How many of a module's worst paths the report keeps.
constexpr unsigned kReportedPaths = 3;

FailureOr<std::vector<TimingPath>>
validateDatapath(dcp::DCPathModuleOp func, const Datapath &dp, float cycleTime,
                 const OperatorLibrary &lib, bool plannedBinding) {
  // The builder already reported the offending edge, and the depths it left
  // are placeholders, so nothing below would measure the design as asked for.
  if (dp.infeasible)
    return failure();
  assertModelInvariants(dp);
  // The design's own faults are reported before what this backend has not
  // built yet. The period check measures while it checks, so the paths come out
  // of that traversal.
  std::vector<TimingPath> paths;
  if (failed(checkStorageLegality(func, dp)) ||
      failed(checkStallContracts(dp)) ||
      failed(
          checkEmitterSubset(func, dp, cycleTime, lib, plannedBinding, paths)))
    return failure();

  // A hundredth of a nanosecond, the grid the schedule's own delays are given
  // on: a path missing by less than that misses by nothing the model can see.
  constexpr double kQuantum = 0.01;
  unsigned missed = llvm::count_if(
      paths, [&](const TimingPath &p) { return p.slack < -kQuantum; });
  llvm::stable_sort(paths, [](const TimingPath &a, const TimingPath &b) {
    return a.total > b.total;
  });
  // One path per source anchor, so a store whose address and data both miss
  // does not spend every reported slot on one operation.
  llvm::SmallDenseSet<llvm::StringRef> seen;
  llvm::erase_if(paths, [&](const TimingPath &p) {
    return !p.where.empty() && !seen.insert(p.where).second;
  });
  paths.resize(std::min<size_t>(paths.size(), kReportedPaths));
  // A module with no datapath at all still holds the controller's own
  // registers, so the shortest path any design has is one register hop.
  if (paths.empty())
    paths.push_back({lib.registerFloor(),
                     cycleTime - lib.registerFloor(),
                     "a control register",
                     "",
                     {{"launch: a control register", lib.registerFloor()}}});

  if (missed)
    logging::log(Level::Warn, Stage::Emit, func)
        << missed << " combinational path(s) of this kernel miss the "
        << llvm::format("%.2f", cycleTime) << " ns clock. The worst takes "
        << llvm::format("%.2f", paths.front().total) << " ns ("
        << llvm::format("%+.2f", paths.front().slack) << " ns slack) reaching "
        << paths.front().endpoint
        << (paths.front().where.empty() ? "" : " at " + paths.front().where)
        << "; the design builds and simulates, and it is the part that may not "
           "hold the clock. The QoR report's `fmax` and its critical paths say "
           "where the time goes";
  return paths;
}

} // namespace mlir::allo::uarch
