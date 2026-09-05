/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo-c/AlloTypes.h"

#include "allo/IR/AlloTypes.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "llvm/ADT/ArrayRef.h"

using namespace mlir;

bool alloTypeIsAStream(MlirType type) {
  return isa<allo::StreamType>(unwrap(type));
}

MlirType alloStreamTypeGet(MlirContext ctx, MlirType baseType, uint64_t depth,
                           intptr_t rank, const int64_t *shape) {
  return wrap(allo::StreamType::get(unwrap(ctx), unwrap(baseType),
                                    static_cast<std::size_t>(depth),
                                    llvm::ArrayRef<int64_t>(shape, rank)));
}

MlirType alloStreamTypeGetBaseType(MlirType type) {
  return wrap(cast<allo::StreamType>(unwrap(type)).getBaseType());
}

uint64_t alloStreamTypeGetDepth(MlirType type) {
  return cast<allo::StreamType>(unwrap(type)).getDepth();
}

intptr_t alloStreamTypeGetRank(MlirType type) {
  return static_cast<intptr_t>(
      cast<allo::StreamType>(unwrap(type)).getShape().size());
}

int64_t alloStreamTypeGetDimSize(MlirType type, intptr_t pos) {
  return cast<allo::StreamType>(unwrap(type)).getShape()[pos];
}

MlirTypeID alloStreamTypeGetTypeID(void) {
  return wrap(allo::StreamType::getTypeID());
}
