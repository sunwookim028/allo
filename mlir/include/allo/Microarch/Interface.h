/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_MICROARCH_INTERFACE_H
#define ALLO_MICROARCH_INTERFACE_H

#include "allo/Microarch/Naming.h" // uarch::Datapath

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <optional>
#include <string>
#include <vector>

namespace mlir::allo::iface {

/// A FIFO channel interface. Input (a `get`): the module reads `data` when
/// `valid`, drives `ready`. Output (a `put`): the module drives `data`/`valid`,
/// reads `ready`.
struct FIFO {
  int arg;      // kernel block-argument index (-1 if not an argument)
  bool isInput; // get (input) vs put (output)
  int depth;
  unsigned width; // payload bit width
  std::string base, data, valid, ready;
};

/// One physical interface to an argument array (a single bank of it when the
/// argument is cyclically partitioned). A read exposes `{addr(out), data(in)}`;
/// a write `{addr, data, we}` (all out, `we` empty for a read).
struct Memory {
  /// One partitioned axis of the argument, mirroring `allo::BankLayout::Axis`:
  /// the host reproduces the same element-space decomposition the RTL addresses
  /// with, which `bank`/`factor` alone cannot express.
  struct Axis {
    int dim;
    int64_t factor;
    /// `BankLayout::Kind` as the manifest spells it: "cyclic", "block" or
    /// "skew".
    std::string kind;
  };
  int arg;
  bool write;
  /// This module's write interfaces on \p arg never collide: two may be enabled
  /// in one cycle, but only where the scheduler proved they address different
  /// words. A caller may then give each its own `always` block and infer a true
  /// dual port; without it every write shares one block and the array becomes a
  /// register file. Always false on a read.
  bool independent;
  int bank, factor; // the bank this interface serves / total physical banks
  unsigned width;   // element bit width
  unsigned latency; // access latency
  std::string base, addr, data, we;
  std::vector<int64_t> shape; // the argument's element shape
  std::vector<Axis> axes;     // partitioned axes, mixed-radix order (empty when
                              // unbanked)
};

/// A completely-partitioned argument array, crossing the boundary as one port
/// per element rather than as an addressed interface (`MemUnit::scattered`).
/// `elements` holds the port names flat in row-major order, so the host drives
/// element k of the flattened argument onto `elements[k]`.
struct RegisterFile {
  /// One element's ports: `in` is where it arrives, `out`/`we` where it leaves.
  /// A direction the kernel does not use is empty, and the naming follows it:
  /// `A_k` when one direction is live, `A_k_in`/`A_k_out` when both are.
  struct Element {
    std::string in, out, we;
  };
  int arg;
  unsigned width;             // element bit width
  std::vector<int64_t> shape; // the argument's element shape
  std::vector<Element> elements;
};

/// A scalar input argument (one port, no suffix).
struct Scalar {
  int arg;
  unsigned width;
  std::string name;
};

/// A scalar function result (one output port, driven at `done`).
struct Result {
  unsigned width;
  std::string name;
};

/// One extern operator module this module instantiates, with the port shape it
/// was declared with. The simulation-model generator builds its behavioral
/// module from this entry and joins to the device operator on `impl` +
/// `predicate`.
struct Operator {
  /// What a port is for, so a consumer classifies structurally rather than by
  /// name. `Ce` decides whether the behavioral model gates on a clock enable.
  enum class Role { Data, Clk, Ce, Out };
  struct Port {
    std::string name;
    unsigned width;
    Role role;
    bool isInput() const { return role != Role::Out; }
  };
  std::string module;    // the extern module's RTL name
  std::string impl;      // the device operator's sym_name
  std::string predicate; // compare predicate; empty for everything else
  std::vector<Port> ports;
};

/// The whole boundary of one module. `reads`/`writes` group by access (an inner
/// vector is the access's per-bank interfaces: one entry unbanked, N when a
/// data-dependent access spans every bank).
struct ModuleInterface {
  // The emitted RTL module name and the MLIR symbol it came from; they differ
  // when the symbol needed legalizing (`top.child` -> `top_child`), and only
  // the former reaches the simulator.
  std::string module, symbol;
  /// This module's start->done contract, republished from the `dcp.module`:
  /// `latency` in cycles, absent when the span is data-dependent;
  /// `latencyBound` marks it a worst case rather than an exact count;
  /// `determinacy` is the class a caller composes against.
  std::optional<int64_t> latency;
  bool latencyBound = false;
  std::string determinacy;
  std::vector<Scalar> scalars;
  std::vector<FIFO> streams;
  std::vector<std::vector<Memory>> reads;
  std::vector<std::vector<Memory>> writes;
  std::vector<RegisterFile> registers;
  std::vector<Result> results;
  std::vector<Operator> operators;

  ModuleInterface() = default;
  /// Build the complete boundary from \p dp, whose `readPorts` / `writePorts`
  /// are the one enumeration of its external memory accesses. Nothing is filled
  /// in afterwards: the port declarations, the extern operator modules and the
  /// cosim model are all derived from this object.
  explicit ModuleInterface(const uarch::Datapath &dp);

  /// Every memory interface of argument \p arg, reads before writes and flat
  /// across access groups. One argument can have several port groups (accessed
  /// at several points) and several interfaces per group (one per bank).
  llvm::SmallVector<const Memory *, 2> portsForArg(int arg) const;
  /// The scalar input port of argument \p arg, or null if \p arg is not one.
  const Scalar *scalarForArg(int arg) const;
  /// The stream interface of argument \p arg, or null if \p arg is not one. A
  /// stream argument is single-ended within a module (one `get` side or one
  /// `put` side), so it has exactly one interface.
  const FIFO *streamForArg(int arg) const;

  /// Serialize the model to a compact JSON object.
  std::string toJSON() const;
};

} // namespace mlir::allo::iface

#endif // ALLO_MICROARCH_INTERFACE_H
