/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Hand-written verify/parse/print for the ISA buffer/descriptor/state types.
// The type classes themselves are generated into AlloTypes.cpp.inc (included by
// AlloOps.cpp); this file only defines their member functions.

#include "allo/IR/AlloTypes.h"

using namespace mlir;
using namespace mlir::allo;

//===----------------------------------------------------------------------===//
// Buffer element types
//===----------------------------------------------------------------------===//

LogicalResult
ScalarBufferType::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                         Type elementType) {
  if (!elementType.isIntOrIndexOrFloat())
    return emitError() << "expected int, index or float type";
  return success();
}

LogicalResult
VectorBufferType::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                         Type elementType, ArrayRef<int64_t> shape) {
  if (!elementType.isIntOrIndexOrFloat())
    return emitError() << "expected int, index or float type";
  if (shape.size() != 1)
    return emitError() << "expected 1D shape";
  return success();
}

void VectorBufferType::print(AsmPrinter &p) const {
  p << "<" << getShape().front() << "x" << getElementType() << ">";
}

Type VectorBufferType::parse(AsmParser &p) {
  if (p.parseLess())
    return {};
  SmallVector<int64_t, 4> shape;
  if (p.parseDimensionList(shape, /*allowDynamic=*/false,
                           /*withTrailingX=*/true))
    return {};
  if (shape.size() != 1)
    return {};
  int64_t lanes = shape[0];
  Type elementType;
  if (p.parseType(elementType) || p.parseGreater())
    return {};
  return get(p.getContext(), elementType, lanes);
}

LogicalResult
TileBufferType::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                       Type elementType, ArrayRef<int64_t> shape) {
  if (!elementType.isIntOrIndexOrFloat())
    return emitError() << "expected int, index or float type";
  if (shape.size() != 2)
    return emitError() << "expected 2D shape";
  return success();
}

void TileBufferType::print(AsmPrinter &p) const {
  p << "<";
  p.printDimensionList(getShape());
  p << "x" << getElementType() << ">";
}

Type TileBufferType::parse(AsmParser &p) {
  if (p.parseLess())
    return {};
  SmallVector<int64_t, 4> shape;
  if (p.parseDimensionList(shape, /*allowDynamic=*/false,
                           /*withTrailingX=*/true))
    return {};
  Type elementType;
  if (p.parseType(elementType) || p.parseGreater())
    return {};
  return get(p.getContext(), elementType, shape);
}

//===----------------------------------------------------------------------===//
// Descriptor and state types
//===----------------------------------------------------------------------===//

LogicalResult
DescriptorType::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                       ArrayRef<StringAttr> fields, ArrayRef<Type> fieldTypes) {
  if (fields.size() != fieldTypes.size())
    return emitError() << "expected same number of fields and field types";
  for (auto type : fieldTypes) {
    if (!type.isIntOrIndexOrFloat())
      return emitError() << "expected int, index or float type";
  }
  return success();
}

void DescriptorType::print(AsmPrinter &p) const {
  p << "<[";
  llvm::interleaveComma(llvm::zip(getFields(), getFieldTypes()), p,
                        [&](auto pair) {
                          auto [field, type] = pair;
                          p << field << ":" << type;
                        });
  p << "]>";
}

Type DescriptorType::parse(AsmParser &p) {
  if (p.parseLess() || p.parseLSquare())
    return {};
  SmallVector<StringAttr, 4> fields;
  SmallVector<Type, 4> fieldTypes;
  if (p.parseCommaSeparatedList([&]() -> ParseResult {
        std::string field;
        Type type;
        if (p.parseString(&field) || p.parseColon() || p.parseType(type))
          return failure();
        fields.push_back(StringAttr::get(p.getContext(), field));
        fieldTypes.push_back(type);
        return success();
      }))
    return {};

  if (p.parseRSquare() || p.parseGreater())
    return {};
  return get(p.getContext(), fields, fieldTypes);
}

LogicalResult
StateType::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                  Type elementType) {
  if (!elementType.isInteger())
    return emitError() << "expected integer type";
  return success();
}
