/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Translation/Utils.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace allo;

// Hands out a C identifier for `val`. Explicit names (loop_name, function
// inputs/outputs) and default "v%d" names share a single reserved-name set, so
// neither generator can hand the same identifier to two different values: a
// local that shadows a parameter does not compile (HLS 207-3746), and a
// reference that drifts away from its declaration is an undeclared identifier.
SmallString<8> AlloEmitterBase::addName(Value val, bool isPtr,
                                        std::string name) {
  assert(!isDeclared(val) && "has been declared before.");

  std::string candidate;
  if (name != "") {
    // Use the requested name verbatim the first time, then keep bumping the
    // suffix until we land on one nobody holds.
    candidate = name;
    int &cnt = state.nameConflictCnt[name];
    while (!state.usedNames.insert(candidate).second)
      candidate = name + std::to_string(++cnt);
  } else {
    // A monotonic counter, skipping anything already taken.
    do {
      candidate = "v" + std::to_string(state.nextDefaultName++);
    } while (!state.usedNames.insert(candidate).second);
  }

  SmallString<8> valName;
  if (isPtr)
    valName += "*";
  valName += candidate;
  state.nameTable[val] = valName;

  return valName;
};

SmallString<8> AlloEmitterBase::getName(Value val) {
  // For constant scalar operations, the constant number will be returned
  // rather than the value name.
  if (auto defOp = val.getDefiningOp()) {
    if (auto constOp = dyn_cast<arith::ConstantOp>(defOp)) {
      auto constAttr = constOp.getValue();

      if (auto boolAttr = llvm::dyn_cast<BoolAttr>(constAttr)) {
        return SmallString<8>(std::to_string(boolAttr.getValue()));

      } else if (auto floatAttr = llvm::dyn_cast<FloatAttr>(constAttr)) {
        // Emit float literals with 'f' suffix for f32 to avoid implicit
        // double-to-float or double-to-ac_ieee_float<binary32> conversions.
        int bitwidth =
            llvm::dyn_cast<FloatType>(floatAttr.getType()).getWidth();
        auto value = floatAttr.getValueAsDouble();
        if (std::isfinite(value)) {
          if (bitwidth == 32)
            return SmallString<8>(std::to_string((float)value) + "f");
          else
            return SmallString<8>(std::to_string(value));
        } else if (value > 0)
          return SmallString<8>("INFINITY");
        else
          return SmallString<8>("-INFINITY");

      } else if (auto intAttr = llvm::dyn_cast<IntegerAttr>(constAttr)) {
        auto value = intAttr.getInt();
        return SmallString<8>(std::to_string(value));
      }
    }
  }
  return state.nameTable.lookup(val);
};

Type getUnsignedTypeFromSigned(Type type) {
  if (auto intType = llvm::dyn_cast<IntegerType>(type)) {
    return IntegerType::get(type.getContext(), intType.getWidth(),
                            IntegerType::SignednessSemantics::Unsigned);
  } else if (auto memrefType = llvm::dyn_cast<MemRefType>(type)) {
    Type elt = getUnsignedTypeFromSigned(memrefType.getElementType());
    return MemRefType::get(memrefType.getShape(), elt, memrefType.getLayout(),
                           memrefType.getMemorySpace());
  } else if (auto streamType = llvm::dyn_cast<StreamType>(type)) {
    Type elt = getUnsignedTypeFromSigned(streamType.getBaseType());
    return StreamType::get(type.getContext(), elt, streamType.getDepth());
  }
  return type;
}

void fixUnsignedType(Value &result, bool isUnsigned) {
  if (isUnsigned) {
    result.setType(getUnsignedTypeFromSigned(result.getType()));
  }
}

void fixUnsignedType(memref::GlobalOp &op, bool isUnsigned) {
  if (isUnsigned) { // unsigned type
    auto type = op.getTypeAttr().getValue();
    op.setTypeAttr(TypeAttr::get(getUnsignedTypeFromSigned(type)));
  }
}