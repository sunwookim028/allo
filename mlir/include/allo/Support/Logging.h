/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_SUPPORT_LOGGING_H
#define ALLO_SUPPORT_LOGGING_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <utility>

namespace mlir {
class Operation;
class Location;
} // namespace mlir

namespace mlir::allo::logging {

// Severity, ascending. `Error` (an illegal program) and `Unsupported` (a legal
// one this backend does not lower yet) are siblings, both fatal; `Unsupported`
// sits last only so the ascending threshold never filters it.
enum class Level { Debug, Info, Warn, Error, Unsupported };

// Compiler stage printed in the second bracket. Extend as new stages log.
enum class Stage { Prep, Sched, Dcp, Emit };

/// The reason a compile is refused, and the only stable token a diagnostic
/// carries: message wording is free to change, so a caller recognizing a
/// refusal matches the code instead.
///
/// One code per reason, never per call site: two sites refusing for the same
/// reason share one, and a code is retired rather than reused when its reason
/// goes away. The `E` series is an illegal program, the `N` series a legal one
/// this backend does not lower yet, so a code names its level too; only `error`
/// and `unsupported` take one, a non-fatal line reporting a decision that
/// belongs in a report.
enum class Code {
  // Illegal input (`error`).
  TopFunctionMissing,         // the named top function is not there
  UnknownOption,              // an option value the backend does not offer
  OperatorNotRealized,        // the device covers the op with nothing
  DeviceDeclarationInvalid,   // a device row the backend cannot read
  StallContractUnusable,      // an IP port contract that cannot drive it
  ArrayLayoutConflict,        // one array, two layouts that disagree
  StorageNotDeclared,         // no storage row for what the array needs
  StorageTimingUnrealizable,  // a port latency no clock edge serves
  SpawnInLoop,                // a dataflow process spawned per iteration
  WhileForwardingNotIdentity, // carried values not forwarded 1:1
  ChannelEndMissing,          // a stream channel is missing an end
  DataflowCycleUnseeded,      // a feedback cycle with no initial tokens
  RecursiveCallGraph,         // the call graph is cyclic
  OperatorOverPeriod,         // a clock period with no room for any logic
  DependenceInfeasible,       // no schedule satisfies the dependences
  RegionShapeNotScheduled,    // no scheduling regime for this region
  CompilerInconsistency,      // our own model disagrees with itself
  StorageConflict,            // one array, two storage rows that disagree
  StoragePortsExceeded,       // more concurrent ports than any structure holds

  // A backend gap (`unsupported`).
  OperationNotModelled,        // this stage models no such operation
  MemrefResult,                // an array returned, not written through
  CrossRegionHandOff,          // a value the reading region cannot see
  ScatteredArgumentToCallee,   // a scattered array crossing into a call
  PartitionedViewArgument,     // a partitioned array passed as a view
  PredicateNotCombinational,   // a control predicate we cannot test
  ContainerWithDatapath,       // a dataflow container with loose compute
  ChannelMultiProducer,        // several writers on one channel
  StreamArgumentBidirectional, // a stream argument read and written
  // ALLO-N0010 is retired and is not reused.
  // ALLO-N0011 is retired and is not reused.
  // ALLO-N0012 is retired and is not reused.
  PlacementFailed, // no feasible cycle for an operation
  // ALLO-N0014 is retired and is not reused.
  PartitionedInitializedArray, // banked contents, realized as one bank
  SkewedArgumentToCallee,      // a skewed array crossing into a call
  AffineDivisionUnsupported,   // a floordiv/mod no unsigned lowering serves
  IndexWidthExceeded,          // a compile-time value past the index carrier
};

/// The one table: a code's stable spelling.
constexpr const char *codeTag(Code code) {
  switch (code) {
  case Code::TopFunctionMissing:
    return "ALLO-E0001";
  case Code::UnknownOption:
    return "ALLO-E0002";
  case Code::OperatorNotRealized:
    return "ALLO-E0003";
  case Code::DeviceDeclarationInvalid:
    return "ALLO-E0004";
  case Code::StallContractUnusable:
    return "ALLO-E0005";
  case Code::ArrayLayoutConflict:
    return "ALLO-E0006";
  case Code::StorageNotDeclared:
    return "ALLO-E0007";
  case Code::StorageTimingUnrealizable:
    return "ALLO-E0008";
  case Code::SpawnInLoop:
    return "ALLO-E0009";
  case Code::WhileForwardingNotIdentity:
    return "ALLO-E0010";
  case Code::ChannelEndMissing:
    return "ALLO-E0011";
  case Code::DataflowCycleUnseeded:
    return "ALLO-E0012";
  case Code::RecursiveCallGraph:
    return "ALLO-E0013";
  case Code::OperatorOverPeriod:
    return "ALLO-E0014";
  case Code::DependenceInfeasible:
    return "ALLO-E0015";
  case Code::RegionShapeNotScheduled:
    return "ALLO-E0016";
  case Code::CompilerInconsistency:
    return "ALLO-E0017";
  case Code::StorageConflict:
    return "ALLO-E0018";
  case Code::StoragePortsExceeded:
    return "ALLO-E0019";
  case Code::OperationNotModelled:
    return "ALLO-N0001";
  case Code::MemrefResult:
    return "ALLO-N0002";
  case Code::CrossRegionHandOff:
    return "ALLO-N0003";
  case Code::ScatteredArgumentToCallee:
    return "ALLO-N0004";
  case Code::PartitionedViewArgument:
    return "ALLO-N0005";
  case Code::PredicateNotCombinational:
    return "ALLO-N0006";
  case Code::ContainerWithDatapath:
    return "ALLO-N0007";
  case Code::ChannelMultiProducer:
    return "ALLO-N0008";
  case Code::StreamArgumentBidirectional:
    return "ALLO-N0009";
  case Code::PlacementFailed:
    return "ALLO-N0013";
  case Code::PartitionedInitializedArray:
    return "ALLO-N0015";
  case Code::SkewedArgumentToCallee:
    return "ALLO-N0016";
  case Code::AffineDivisionUnsupported:
    return "ALLO-N0017";
  case Code::IndexWidthExceeded:
    return "ALLO-N0018";
  }
  return "";
}

namespace detail {
// Format `LEVEL[CODE]: [STAGE] message[ (at where)]` and route to the backend
// (`code` is empty on a non-fatal line, which carries none). A fatal level with
// a non-null `subject` additionally emits an MLIR error diagnostic on it,
// carrying the code, so the message both logs and fails the pass.
void emit(Level level, Stage stage, llvm::StringRef code, llvm::StringRef where,
          llvm::StringRef message, mlir::Operation *subject);
// Whether `level` passes the threshold (skip building dropped lines). A fatal
// level is never filtered.
bool enabled(Level level);
// Concise source anchor for an op / location (symbolic name + file:line:col).
std::string describe(mlir::Operation *op);
std::string describe(const mlir::Location &loc, bool withFile = true);
} // namespace detail

// RAII stream proxy: accumulate a message with `<<`, emit it on destruction.
// Created through the factories below, which rely on C++17 guaranteed copy
// elision, so no move or copy constructor is needed.
class Diagnostic {
public:
  Diagnostic(Level level, Stage stage, const char *code, std::string where,
             mlir::Operation *subject)
      : level(level), stage(stage), active(detail::enabled(level)), code(code),
        subject(subject), where(std::move(where)), stream(message) {}
  ~Diagnostic() {
    if (active) {
      stream.flush();
      detail::emit(level, stage, code ? code : "", where, message, subject);
    }
  }

  Diagnostic(const Diagnostic &) = delete;
  Diagnostic &operator=(const Diagnostic &) = delete;

  template <typename T> Diagnostic &operator<<(T &&value) {
    if (active)
      stream << std::forward<T>(value);
    return *this;
  }

private:
  Level level;
  Stage stage;
  bool active;
  const char *code;
  mlir::Operation *subject;
  std::string where;
  std::string message;
  llvm::raw_string_ostream stream;
};

// Factories. The op/location overloads render a source anchor (a null op omits
// it). `log` builds the non-fatal levels, which carry no code; a fatal one goes
// through `error` / `unsupported` below, which demand one.
inline Diagnostic log(Level level, Stage stage) {
  return Diagnostic(level, stage, nullptr, std::string(), nullptr);
}
inline Diagnostic log(Level level, Stage stage, mlir::Operation *subject) {
  return Diagnostic(level, stage, nullptr, detail::describe(subject), subject);
}
inline Diagnostic log(Level level, Stage stage, const mlir::Location &loc) {
  return Diagnostic(level, stage, nullptr, detail::describe(loc), nullptr);
}

inline Diagnostic debug(Stage stage, mlir::Operation *op = nullptr) {
  return log(Level::Debug, stage, op);
}
inline Diagnostic info(Stage stage, mlir::Operation *op = nullptr) {
  return log(Level::Info, stage, op);
}
inline Diagnostic warn(Stage stage, mlir::Operation *op = nullptr) {
  return log(Level::Warn, stage, op);
}
// An illegal program: fatal, and `code` names the reason it is refused. Pass
// the subject op so the failure propagates to the caller.
inline Diagnostic error(Stage stage, Code code, mlir::Operation *op = nullptr) {
  return Diagnostic(Level::Error, stage, codeTag(code), detail::describe(op),
                    op);
}
// A legal program this backend does not lower yet: fatal like `error`, tagged
// `NYI`, with a message naming the missing compiler feature.
inline Diagnostic unsupported(Stage stage, Code code,
                              mlir::Operation *op = nullptr) {
  return Diagnostic(Level::Unsupported, stage, codeTag(code),
                    detail::describe(op), op);
}

// Runtime configuration (the threshold is also seeded from the ALLO_LOG_LEVEL
// environment variable on first use).
void setLevel(Level level);
Level getLevel();
void setColor(bool enable);

} // namespace mlir::allo::logging

#endif // ALLO_SUPPORT_LOGGING_H
