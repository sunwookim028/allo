/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_MICROARCH_REGLEDGER_H
#define ALLO_MICROARCH_REGLEDGER_H

#include "llvm/ADT/StringRef.h"

#include <cassert>
#include <map>
#include <tuple>
#include <vector>

namespace mlir::allo::uarch {

/// Why a register exists. The emitter knows this where it builds the register
/// and nowhere later: a reader of the emitted design sees a `seq.compreg` and
/// can recover the reason only from the name it carries, a convention
/// `Naming.h` owns.
enum class RegRole {
  Value,    // a value delay chain: one datum carried across cycle boundaries
  Pulse,    // an activation chain: a region's issue delayed to an op's stage
  Counted,  // the counter a deep pulse delay is built as instead of a chain
  Survivor, // a region result, or a loop-carried iter-arg latch
  Counter,  // an iteration counter, or one of its address strides
  Control,  // run / phase / pending / done, and the rest of the control plane
  Storage,  // one element of an array scattered into registers
};

llvm::StringRef roleName(RegRole role);

/// One class of register run: `count` runs of `depth` registers in series,
/// `width` bits each, all built for the same reason. A lone register is a run
/// of depth 1, so the design's flip-flop count is `sum(width * depth * count)`.
///
/// The run, not the register, is the cost unit: past the synthesizer's
/// shift-register extraction threshold a run stops costing flip-flops per
/// stage. A multi-tapped chain is charged as one run per maximal inter-tap
/// segment. `reset` blocks extraction and pays fabric per bit; `enable` is
/// free; a cost model needs both to pick its characterization row.
struct RegClass {
  RegRole role = RegRole::Control;
  unsigned width = 0, depth = 0, count = 0;
  bool reset = true;   // holds a synchronous reset to its reset value
  bool enable = false; // samples only under a condition (d = ce ? in : q)
};

/// Every register one module's emission built. Filled at the one point that
/// creates a `seq.compreg` (`EmitContext::reg`) plus the chain builders, which
/// charge a whole run at once, so the total is a count, not an estimate.
class RegLedger {
public:
  /// Charge one run of \p depth registers of \p width bits. A depth of zero is
  /// no run at all (a chain a consumer reads at tap 0 builds nothing).
  void add(RegRole role, unsigned width, unsigned depth, bool reset = true,
           bool enable = false) {
    assert(width && "a register holds at least one bit");
    if (depth)
      ++runs[{role, width, depth, reset, enable}];
  }

  /// Re-charge a run extended in place from \p from to \p to stages deep: the
  /// old run's charge is replaced, not added to.
  void extend(RegRole role, unsigned width, unsigned from, unsigned to,
              bool reset = true, bool enable = false) {
    assert(from < to && "an extension adds at least one stage");
    if (from) {
      auto it = runs.find({role, width, from, reset, enable});
      assert(it != runs.end() && it->second && "extending a run never charged");
      if (!--it->second)
        runs.erase(it);
    }
    add(role, width, to, reset, enable);
  }

  /// Every class, in a deterministic order, so a report built from this does
  /// not reorder between two runs of the same compile.
  std::vector<RegClass> classes() const;

  /// Flip-flops across every class.
  unsigned bits() const;

private:
  std::map<std::tuple<RegRole, unsigned, unsigned, bool, bool>, unsigned> runs;
};

/// Why a select cone exists. These are the interconnect built around storage
/// after the binding, disjoint from the allocation's own mux cells, which the
/// region report prices from the model.
enum class MuxRole {
  Address,  // a shared port's one-hot address select
  Commit,   // a shared write or held read port's priority commit chain
  Crossbar, // routing a run-time-banked or scattered access
};

llvm::StringRef muxRoleName(MuxRole role);

/// One class of select cone: `count` cones, each `fanin` sources wide at
/// `width` bits. A k:1 cone costs about (k-1) 2:1 muxes per bit.
struct MuxCone {
  MuxRole role = MuxRole::Address;
  unsigned fanin = 0, width = 0, count = 0;
};

/// Every select cone the emission builds around storage, charged where it is
/// built, mirroring RegLedger.
class MuxLedger {
public:
  /// Charge one cone of \p fanin sources at \p width bits. Fewer than two
  /// sources select nothing and charge nothing.
  void add(MuxRole role, unsigned fanin, unsigned width) {
    if (fanin >= 2 && width)
      ++cones[{role, fanin, width}];
  }

  std::vector<MuxCone> classes() const;

private:
  std::map<std::tuple<MuxRole, unsigned, unsigned>, unsigned> cones;
};

} // namespace mlir::allo::uarch

#endif // ALLO_MICROARCH_REGLEDGER_H
