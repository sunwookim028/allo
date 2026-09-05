/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_MICROARCH_REPORT_H
#define ALLO_MICROARCH_REPORT_H

#include "allo/Microarch/RegLedger.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace mlir::allo::uarch {

struct Datapath;

/// What the microarchitecture stage decided, as data: a projection of the
/// `Datapath` taken once per emitted module, read back through JSON by Python.
/// Nothing here duplicates the schedule report, which publishes op start
/// cycles, region trip counts and realizations joined on (func, region order);
/// only the binding and what the emitter built downstream of it live here.
/// Write-only: no pass reads it back.

/// One functional-unit instance. `boundOps > 1` is a sharing decision; the
/// trivial binding leaves every operation its own unit.
struct UnitReport {
  std::string identity; // `OperatorIdentity::key()`; sharing equivalence class
  std::string impl;     // the `dcp.operator` symbol; empty for a native unit
  std::string module;   // `operatorModuleName`; empty for a native unit
  unsigned width = 0;   // result width in bits
  unsigned latency = 0;
  unsigned boundOps = 1;
  bool comb = false; // native combinational vs an IP instance
  bool pipelined = true;
  // A standalone apply unit's cone (operator counts from `applyExprOf`) so the
  // estimator can price the map; zero for every other unit.
  unsigned adders = 0, multipliers = 0, dividers = 0;
};

/// One value delay chain, a `uarch::Register` projected. `rangeBits` is the
/// width of the value range a model-level interval walk proved, absent when
/// unproven.
struct ChainReport {
  int64_t region = 0;  // owning region's order (`RegionUarch::order`)
  unsigned width = 0;  // built carrier bits
  std::string carried; // the held type spelled ("index", "i32", "f32")
  unsigned depth = 0;  // chain length in cycles
  unsigned ii = 1;     // owning region's interval; folds registers at > 1
  unsigned taps = 0;   // distinct consumer read depths
  std::string source;  // driving cell class, or a unit's mnemonic / IP symbol
  std::optional<unsigned> rangeBits;
};

/// A class of multiplexer: `count` of them, each `fanin` sources wide at
/// `width` bits.
struct MuxClass {
  unsigned fanin = 0, width = 0, count = 0;
};

/// What the cost model needs of one array and no reader does: the ports it was
/// bound with, and who drives them.
struct MemCost {
  // Ports a child drives into this array; several children writing makes a
  // banking problem.
  unsigned callReads = 0, callWrites = 0;
  // Ports one bank is built with (`MemUnit::readPortsBuilt` and twins). `ports`
  // is not their sum: a pooled port may carry a read and a write that never
  // issue together.
  unsigned readPorts = 0, writePorts = 0, ports = 0;
  // Storage-row instances each bank is held in (`MemUnit::instances`); the cost
  // model multiplies by this.
  unsigned instances = 1;
  // Instances the schedule reserved against (`StoragePorts::copies`);
  // `instances` may exceed it when the binding replicates further.
  unsigned copiesBudget = 1;
  // Read/write ports one row instance provides, 0 for no limit; multiplied by
  // `instances`.
  unsigned rowReads = 0, rowWrites = 0;
  // Lower bound on what one cycle asks of this bank, per direction
  // (`MemUnit::readConcurrency`). Zero for a ROM or scattered array, neither
  // addressed.
  unsigned readConcurrency = 0, writeConcurrency = 0;
  // Module interface groups this array contributes (`MemUnit::boundaryPorts`).
  unsigned boundaryPorts = 0;
};

/// One array, and the storage decision taken for it.
struct MemReport {
  std::string owner;          // the name its ports are spelled from
  std::vector<int64_t> shape; // element shape
  unsigned width = 0;         // element bits
  unsigned banks = 1;
  std::string layout;      // "none", "cyclic", "block", "skew" or "complete"
  std::string storage;     // the resolved `dcp.storage` realization
  unsigned depthWords = 0; // elements per bank
  unsigned readLatency = 0, writeLatency = 1;
  unsigned reads = 0, writes = 0; // accesses bound in this module
  MemCost cost;
  bool external = false, scattered = false, writesIndependent = false;
  bool rom = false, skewed = false;
  /// What the module built to hold it (`MemUnit::Realization`, spelled
  /// "boundary" / "rom" / "scatter" / "ram").
  std::string realization;
  /// Whether the partition bought the bandwidth it costs: every access reaches
  /// one bank. An unresolved access takes a port on every bank, so a partition
  /// resolving none is N memories at the bandwidth of one. True for an
  /// unpartitioned array, which has nothing to resolve.
  bool partitionResolved = true;
};

/// One FIFO channel.
struct StreamReport {
  std::string owner;
  unsigned width = 0, depth = 0;
  bool crossesCall = false; // an end of it is a child port, not a local access
  bool internal = false;    // created in this body: its `seq.fifo` lives here
};

/// Sub-kernel invocations of one callee.
struct CallReport {
  std::string callee;
  unsigned count = 0;
  unsigned spawns = 0;            // of those, `await` spawns rather than calls
  std::optional<int64_t> latency; // the child's declared span, when static
  /// How those calls are released (`CallUnit::StartPolicy`), counted.
  unsigned handshake = 0, broadcast = 0, timed = 0;
};

/// One address stride register beside the counter: its width and which update
/// cells it builds (step adder; carry adder with its select; wrap compare with
/// its fix adder and select). `isCounter` names the stride that is the counter
/// itself, which builds no register.
struct StrideCost {
  unsigned width = 0;
  bool step = false, carry = false, wrap = false, isCounter = false;
};

/// What the cost model needs of one region and no reader does. Grouped apart
/// for the same reason as `MemCost`.
struct RegionCost {
  // Mux totals the allocation charges: inputs across every mux, and
  // 2:1-equivalent bits (a k:1 mux costs about (k-1) 2:1 muxes per bit).
  unsigned muxInputs = 0, muxBits = 0;
  // Control plane: iteration-counter width, phase-counter width (a pipelined
  // leaf at II>1), and the address strides beside them.
  unsigned counterWidth = 0, phaseWidth = 0, addrStrides = 0;
  std::vector<StrideCost> strides;
};

/// One step of a combinational path: what the signal passes through and what it
/// spends there. A step is one model cell, which may be a lump the model prices
/// without decomposing (an address cone, a select).
struct TimingStep {
  std::string what;
  double delay = 0.0; // ns
};

/// One combinational path, its steps ordered from the launching register or
/// port to the capture at `endpoint`. `total` is the sum of the step delays.
struct TimingPath {
  double total = 0.0; // ns
  double slack = 0.0; // period - total; negative means it misses the clock
  std::string endpoint;
  std::string where; // source anchor of the endpoint, when it has one
  std::vector<TimingStep> steps;
};

/// One region's allocation. `order` is the join key to the schedule report's
/// `RegionReport::order`: both are program order within the func.
struct RegionUarch {
  int64_t order = 0;
  std::string shape; // Leaf / Container / Guard / CallNode
  std::string kind;  // "cyclic" or "acyclic"
  std::optional<int64_t> interval;
  unsigned computeOps = 0; // operations bound to a unit in this region
  std::vector<UnitReport> units;
  std::vector<MuxClass> muxes;
  RegionCost cost;
};

/// One emitted module.
struct FuncUarch {
  std::string func;   // the `dcp` module symbol; joins to `FuncReport::name`
  std::string module; // the emitted `hw.module` name; joins to `Interfaces`
  bool top = false;
  std::vector<RegionUarch> regions;
  // Module-wide register runs: a run belongs to the value it carries, not a
  // region, counted where it is built.
  std::vector<RegClass> regs;
  // Every value delay chain the model holds (`dp.regs`), one row each. The
  // ledger's value-role classes also count chains built outside the model
  // (read-data alignment, stall holds), so these sum to less.
  std::vector<ChainReport> chains;
  // The select cones the emission built around storage, disjoint from each
  // region's allocation muxes.
  std::vector<MuxCone> muxCones;
  std::vector<MemReport> mems;
  std::vector<StreamReport> streams;
  std::vector<CallReport> calls;
  unsigned readPorts = 0, writePorts = 0; // boundary port groups
  /// This module's worst combinational paths, longest first, as
  /// `validateDatapath` measured them.
  std::vector<TimingPath> criticalPaths;

  /// The longest path's total, in ns. Zero only on a default-constructed
  /// report: every emitted module holds at least one register hop.
  double criticalPath() const {
    return criticalPaths.empty() ? 0.0 : criticalPaths.front().total;
  }

  /// Project \p dp, plus the registers and select cones its emission built and
  /// the paths `validateDatapath` measured.
  FuncUarch(const Datapath &dp, llvm::StringRef symbol, llvm::StringRef module,
            const RegLedger &ledger, const MuxLedger &muxes,
            std::vector<TimingPath> criticalPaths);
  FuncUarch() = default;
};

/// One emission: every module it built, in emit order (callees before callers).
struct MicroarchReport {
  /// Schema version, bumped on a breaking rename: a baseline persisted to disk
  /// by a comparison tool is refused rather than silently compared against a
  /// later schema. In-process the producer and consumer share a build.
  static constexpr unsigned kVersion = 1;

  std::string binding; // the sharing policy this emission ran under
  float cycleTime = 0; // ns; the period the schedule was cut to
  std::vector<FuncUarch> funcs;

  /// The report as the JSON document Python parses. Absent optionals are
  /// omitted rather than null, as in the schedule report and interface
  /// manifest, so a consumer tests for the field it needs.
  std::string toJSON() const;
};

} // namespace mlir::allo::uarch

#endif // ALLO_MICROARCH_REPORT_H
