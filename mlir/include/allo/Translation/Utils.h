/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_TRANSLATION_UTILS_H
#define ALLO_TRANSLATION_UTILS_H

#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"

#include "allo/Dialect/AlloDialect.h"
#include "allo/Dialect/AlloOps.h"

using namespace mlir;
using namespace allo;

//===----------------------------------------------------------------------===//
// Base Classes
//===----------------------------------------------------------------------===//

/// This class maintains the mutable state that cross-cuts and is shared by the
/// various emitters.
class AlloEmitterState {
public:
  explicit AlloEmitterState(raw_ostream &os) : os(os) {}

  // The stream to emit to.
  raw_ostream &os;

  bool encounteredError = false;
  unsigned currentIndent = 0;

  // This table contains all declared values.
  DenseMap<Value, SmallString<8>> nameTable;
  std::map<std::string, int> nameConflictCnt;

  // Every identifier handed out so far, whatever produced it: explicit names
  // (loop_name, function inputs/outputs), default "v%d" names, and symbols
  // reserved by the emitter (globals). One namespace, so the two generators
  // cannot hand the same identifier to two different values.
  llvm::StringSet<> usedNames;
  // Counter for default names. Unlike nameTable.size() this only ever grows,
  // so a value that is renamed rather than newly declared cannot make the
  // next value reuse a name.
  unsigned nextDefaultName = 0;

  // Configuration flags
  bool linearize_pointers = false;

  // When set (SystemC/Catapult-native flow), emit f16 scalar constants as an
  // explicit `half(<v>f)` construction -- ac_ieee_float<binary16> has no
  // implicit double/float assignment. Left false for Vivado/Vitis (which uses
  // hls::half, accepts bare literals) so that output stays byte-identical.
  bool acFloatConstCtor = false;

  // When set (SystemC clocked-thread flow), a wait() is emitted at the end of
  // each scf.while body. A `while not S.try_put(x): pass` busy-wait spins on a
  // non-blocking op; without a clock advance per iteration it retries in ZERO
  // simulation time -- an infinite combinational loop that hangs RTL cosim (the
  // FIFO can't advance its handshake without a clock edge). The wait() lets the
  // clock tick so the retry can succeed. A successful iteration breaks BEFORE the
  // wait (the scf.condition break precedes the body), so it costs nothing there;
  // on any other while loop a wait() only adds a cycle, never changes a value.
  // Left false for Vivado/Vitis (no clocked threads).
  bool scfWhileWait = false;

  // Track which values are top-level function arguments (for linearization)
  DenseSet<Value> topLevelFunctionArgs;

private:
  AlloEmitterState(const AlloEmitterState &) = delete;
  void operator=(const AlloEmitterState &) = delete;
};

/// This is the base class for all of the HLSCpp Emitter components.
class AlloEmitterBase {
public:
  explicit AlloEmitterBase(AlloEmitterState &state)
      : state(state), os(state.os) {}

  InFlightDiagnostic emitError(Operation *op, const Twine &message) {
    state.encounteredError = true;
    return op->emitError(message);
  }

  raw_ostream &indent() { return os.indent(state.currentIndent); }

  void addIndent() { state.currentIndent += 2; }
  void reduceIndent() { state.currentIndent -= 2; }

  // All of the mutable state we are maintaining.
  AlloEmitterState &state;

  // The stream to emit to.
  raw_ostream &os;

  /// Value name management methods.
  SmallString<8> addName(Value val, bool isPtr = false, std::string name = "");

  /// Reserve `name` so no identifier generated later can collide with it.
  /// Used for symbols the emitter prints directly (globals), which otherwise
  /// bypass the name table entirely.
  void reserveName(StringRef name) { state.usedNames.insert(name); }

  /// Bind `val` to an existing, externally-owned identifier. Unlike addName
  /// this never disambiguates: every value denoting the same symbol has to
  /// print the same identifier, or the reference is undeclared.
  void bindName(Value val, StringRef symbol) {
    reserveName(symbol);
    state.nameTable[val] = SmallString<8>(symbol);
  }

  SmallString<8> getName(Value val);

  bool isDeclared(Value val) {
    if (getName(val).empty()) {
      return false;
    } else
      return true;
  }

private:
  AlloEmitterBase(const AlloEmitterBase &) = delete;
  void operator=(const AlloEmitterBase &) = delete;
};

void fixUnsignedType(Value &result, bool isUnsigned);
void fixUnsignedType(memref::GlobalOp &op, bool isUnsigned);

#endif // ALLO_TRANSLATION_UTILS_H