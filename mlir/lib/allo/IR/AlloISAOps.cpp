/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Hand-written method bodies for the ISA ops (resource decls, access patterns,
// instructions). The op classes themselves are generated into
// AlloISAOps.cpp.inc (included by AlloOps.cpp); this file only defines their
// member functions.

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

#include "allo/IR/AlloOps.h"

using namespace mlir;
using namespace mlir::allo;

//===----------------------------------------------------------------------===//
// Resource declaration ops
//===----------------------------------------------------------------------===//

void DeclareBufferOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printSymbolName(getSymName());
  p << " extents(";
  llvm::interleaveComma(getExtents(), p, [&](int64_t e) { p << e; });
  p << ") ";
  p.printOptionalAttrDict(getOperation()->getAttrs(),
                          /*elidedAttrs=*/{SymbolTable::getSymbolAttrName(),
                                           getExtentsAttrName(),
                                           getBufferTypeAttrName()});
  p << ": " << getBufferType();
}

ParseResult DeclareBufferOp::parse(OpAsmParser &parser,
                                   OperationState &result) {
  StringAttr symName;
  SmallVector<int64_t, 4> extents;
  Type bufferType;
  if (parser.parseSymbolName(symName) || parser.parseKeyword("extents") ||
      parser.parseCommaSeparatedList(AsmParser::Delimiter::Paren,
                                     [&]() {
                                       int64_t extent;
                                       if (parser.parseInteger(extent))
                                         return failure();
                                       extents.push_back(extent);
                                       return success();
                                     }) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(bufferType))
    return failure();
  result.addAttribute(SymbolTable::getSymbolAttrName(), symName);
  result.addAttribute(getExtentsAttrName(result.name),
                      parser.getBuilder().getDenseI64ArrayAttr(extents));
  result.addAttribute(getBufferTypeAttrName(result.name),
                      TypeAttr::get(bufferType));
  return success();
}

LogicalResult DeclareBufferOp::verify() {
  if (getExtents().empty())
    return emitError() << "a buffer must have at least one extent";
  for (int64_t extent : getExtents())
    if (extent <= 0)
      return emitError() << "extents must be positive, got " << extent;
  return success();
}

LogicalResult DeclareStateOp::verify() {
  auto defaultOr = getDefaultState();
  if (!defaultOr)
    return success();
  auto enums = getEnums().getAsRange<StringAttr>();
  if (llvm::find(enums, StringAttr::get(getContext(), *defaultOr)) ==
      enums.end())
    return emitError() << "default state must be one of the enumerated states";
  return success();
}

LogicalResult
WriteStateOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto state = symbolTable.lookupNearestSymbolFrom<DeclareStateOp>(
      *this, getStateAttr());
  if (!state)
    return emitError() << "referred state '" << getState()
                       << "' does not exist";
  auto enums = state.getEnums().getAsRange<StringAttr>();
  if (llvm::find(enums, getValueAttr()) == enums.end())
    return emitError() << "value must be one of the enumerated states";
  return success();
}

LogicalResult
ReadStateOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto state = symbolTable.lookupNearestSymbolFrom<DeclareStateOp>(
      *this, getStateAttr());
  if (!state)
    return emitError() << "referred state '" << getState()
                       << "' does not exist";
  auto eltType = state.getStateType().getElementType();
  if (eltType != getType())
    return emitError() << "expected type '" << eltType << "' but got '"
                       << getType() << "'";
  return success();
}

LogicalResult
WriteDescFieldOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto desc =
      symbolTable.lookupNearestSymbolFrom<DeclareDescOp>(*this, getDescAttr());
  if (!desc)
    return emitError() << "referred descriptor '" << getDesc()
                       << "' does not exist";
  DescriptorType descType = desc.getDescType();
  auto fields = descType.getFields();
  auto *it = llvm::find(fields, getFieldAttr());
  if (it == fields.end())
    return emitError() << "field must be one of the descriptor fields";
  auto fieldType = descType.getFieldTypes()[std::distance(fields.begin(), it)];
  if (fieldType != getValue().getType())
    return emitError() << "expected type '" << fieldType << "' but got '"
                       << getValue().getType() << "'";
  return success();
}

LogicalResult
ReadDescFieldOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto desc =
      symbolTable.lookupNearestSymbolFrom<DeclareDescOp>(*this, getDescAttr());
  if (!desc)
    return emitError() << "referred descriptor '" << getDesc()
                       << "' does not exist";
  DescriptorType descType = desc.getDescType();
  auto fields = descType.getFields();
  auto *it = llvm::find(fields, getFieldAttr());
  if (it == fields.end())
    return emitError() << "field must be one of the descriptor fields";
  auto fieldType = descType.getFieldTypes()[std::distance(fields.begin(), it)];
  if (fieldType != getType())
    return emitError() << "expected type '" << fieldType << "' but got '"
                       << getType() << "'";
  return success();
}

//===----------------------------------------------------------------------===//
// Access-pattern ops: strided / tiled
//===----------------------------------------------------------------------===//

LogicalResult StridedOp::verify() {
  if (getStaticStrides().size() != getStaticCounts().size() ||
      getStaticStrides().size() != getStaticBasis().size())
    return emitError()
           << "basis, counts and strides must have the same number of "
              "elements";
  for (auto stride : getStaticStrides()) {
    if (ShapedType::isStatic(stride) && stride <= 0)
      return emitError() << "strides must be positive";
  }
  for (auto count : getStaticCounts()) {
    if (ShapedType::isStatic(count) && count <= 0)
      return emitError() << "counts must be positive";
  }
  for (auto base : getStaticBasis()) {
    if (ShapedType::isStatic(base) && base < 0)
      return emitError() << "basis must be non-negative";
  }
  return success();
}

// Shared by the access-pattern ops: an access carries one component per address
// extent, and each component must stay inside its extent. The footprint along
// an axis is `basis + stride * (count - 1)`, plus whatever extra reach the
// pattern itself has (a tile's own width).
static LogicalResult
verifyAgainstExtents(Operation *op, ArrayRef<int64_t> extents,
                     ArrayRef<int64_t> basis, ArrayRef<int64_t> strides,
                     ArrayRef<int64_t> counts, ArrayRef<int64_t> reach = {}) {
  if (counts.size() != extents.size())
    return op->emitError() << "access pattern has " << counts.size()
                           << " dimension(s) but the buffer is addressed by "
                           << extents.size()
                           << " (one access component per buffer extent)";
  for (unsigned i = 0, e = counts.size(); i < e; ++i) {
    if (!ShapedType::isStatic(strides[i]) || !ShapedType::isStatic(counts[i]))
      continue;
    int64_t maxIndex = strides[i] * (counts[i] - 1);
    if (ShapedType::isStatic(basis[i]))
      maxIndex += basis[i];
    if (!reach.empty()) {
      if (!ShapedType::isStatic(reach[i]))
        continue;
      maxIndex += reach[i] - 1;
    }
    if (maxIndex >= extents[i])
      return op->emitError()
             << "access out of bounds in dimension " << i << ": max index is "
             << maxIndex << " but the buffer's extent is " << extents[i];
  }
  return success();
}

LogicalResult StridedOp::verifyCompatibility(BufferTypeInterface,
                                             ArrayRef<int64_t> extents) {
  // The slot type is deliberately unused: an access addresses *slots*, so what
  // one slot holds is none of its business. That separation is what lets the
  // same rule cover a flat register file and a row-major off-chip array.
  return verifyAgainstExtents(getOperation(), extents, getStaticBasis(),
                              getStaticStrides(), getStaticCounts());
}

// How many of an access's dimensions vanish from the tensor the compute region
// sees. A count of exactly 1 *selects* one slot along that axis rather than
// spanning a range, so — like numpy's `a[3]` versus `a[3:4]` — it contributes
// no tensor dimension: `vld vr[d], vmem[s]` reads one slot of a vector register
// file and hands the compute region the lanes, not a 1 x lanes tensor. The
// Python frontend's `PatternExpr.visible_shape` mirrors this exactly; if the
// two ever disagree the inlined semantics get an operand of the wrong rank.
static unsigned rankReduction(ArrayRef<int64_t> counts) {
  return llvm::count_if(counts, [](int64_t count) {
    return ShapedType::isStatic(count) && count == 1;
  });
}

FailureOr<Value> StridedOp::materialize(OpBuilder &builder, Location loc,
                                        Value buffer) {
  MLIRContext *ctx = builder.getContext();
  auto mixedBasis = getMixedValues(getStaticBasis(), getBasis(), ctx);
  auto mixedStrides = getMixedValues(getStaticStrides(), getStrides(), ctx);
  auto mixedCounts = getMixedValues(getStaticCounts(), getCounts(), ctx);
  auto shaped = cast<ShapedType>(buffer.getType());
  unsigned currRank = mixedBasis.size();
  // align the dimensions
  assert(shaped.getRank() >= currRank);
  unsigned diff = shaped.getRank() - currRank;
  mixedBasis.append(diff, builder.getI64IntegerAttr(0));
  mixedStrides.append(diff, builder.getI64IntegerAttr(1));
  for (unsigned i = 0; i < diff; ++i)
    mixedCounts.push_back(
        builder.getI64IntegerAttr(shaped.getDimSize(i + currRank)));
  // Drop the selected (count == 1) address dims; the slot's own dims all stay.
  auto resultTy = tensor::ExtractSliceOp::inferCanonicalRankReducedResultType(
      shaped.getRank() - rankReduction(getStaticCounts()),
      cast<RankedTensorType>(shaped), mixedCounts);
  Value extracted = tensor::ExtractSliceOp::create(
      builder, loc, resultTy, buffer, mixedBasis, mixedCounts, mixedStrides);
  return extracted;
}

FailureOr<Value> StridedOp::materialize(OpBuilder &builder, Location loc,
                                        Value value, Value buffer) {
  MLIRContext *ctx = builder.getContext();
  auto mixedBasis = getMixedValues(getStaticBasis(), getBasis(), ctx);
  auto mixedStrides = getMixedValues(getStaticStrides(), getStrides(), ctx);
  auto mixedCounts = getMixedValues(getStaticCounts(), getCounts(), ctx);
  auto shaped = cast<ShapedType>(buffer.getType());
  unsigned currRank = mixedBasis.size();
  assert(shaped.getRank() >= currRank);
  unsigned diff = shaped.getRank() - currRank;
  // align the dimensions
  mixedBasis.append(diff, builder.getI64IntegerAttr(0));
  mixedStrides.append(diff, builder.getI64IntegerAttr(1));
  for (unsigned i = 0; i < diff; ++i)
    mixedCounts.push_back(
        builder.getI64IntegerAttr(shaped.getDimSize(i + currRank)));
  Value inserted = tensor::InsertSliceOp::create(
      builder, loc, value, buffer, mixedBasis, mixedCounts, mixedStrides);
  return inserted;
}

LogicalResult TiledOp::verify() {
  if (getStaticStrides().size() != getStaticCounts().size() ||
      getStaticStrides().size() != getStaticBasis().size() ||
      getStaticStrides().size() != getStaticTileSizes().size())
    return emitError() << "basis, counts, strides and tile_sizes must "
                          "have the same number of elements";
  for (auto stride : getStaticStrides())
    if (ShapedType::isStatic(stride) && stride <= 0)
      return emitError() << "strides must be positive";
  for (auto count : getStaticCounts())
    if (ShapedType::isStatic(count) && count <= 0)
      return emitError() << "counts must be positive";
  for (auto base : getStaticBasis())
    if (ShapedType::isStatic(base) && base < 0)
      return emitError() << "basis must be non-negative";
  for (auto tileSize : getStaticTileSizes())
    if (ShapedType::isStatic(tileSize) && tileSize <= 0)
      return emitError() << "tile_sizes must be positive";
  // Tile size must not exceed stride — otherwise tiles overlap.
  auto strides = getStaticStrides();
  auto tileSizes = getStaticTileSizes();
  for (unsigned i = 0; i < strides.size(); ++i)
    if (ShapedType::isStatic(strides[i]) &&
        ShapedType::isStatic(tileSizes[i]) && tileSizes[i] > strides[i])
      return emitError() << "tile_size " << tileSizes[i] << " exceeds stride "
                         << strides[i] << " in dimension " << i
                         << ": tiles would overlap";
  return success();
}

LogicalResult TiledOp::verifyCompatibility(BufferTypeInterface,
                                           ArrayRef<int64_t> extents) {
  // Same rank + bounds rule as StridedOp, with the tile's own width counted
  // into the footprint. The old "tile_size must evenly divide the buffer
  // element size" rule is gone with the on-chip/HBM split it belonged to: it
  // compared the tile along dimension 0 against the *slot*, which was only
  // meaningful while dimension 0 necessarily indexed slots of a 1-D buffer.
  // What a tiled access means relative to a slot has to be re-derived when
  // something needs it — this op has no `materialize` and no frontend user (the
  // Python `access.tiled` was deleted), so guessing now would only bake in a
  // rule nobody has tested.
  return verifyAgainstExtents(getOperation(), extents, getStaticBasis(),
                              getStaticStrides(), getStaticCounts(),
                              getStaticTileSizes());
}

// Tiled materialize is not implemented yet (deferred until a workload needs
// it).
FailureOr<Value> TiledOp::materialize(OpBuilder &, Location, Value) {
  return failure();
}

FailureOr<Value> TiledOp::materialize(OpBuilder &, Location, Value, Value) {
  return failure();
}

//===----------------------------------------------------------------------===//
// Relayout ops: expand_shape / collapse_shape / transpose
//===----------------------------------------------------------------------===//

LogicalResult CollapseShapeOp::verify() {
  if (llvm::any_of(getReassociationIndices(),
                   [](ReassociationIndices &group) { return group.empty(); })) {
    return emitError("reassociation indices must not be empty");
  }
  return success();
}

LogicalResult ExpandShapeOp::verifyCompatibility(BufferTypeInterface,
                                                 ArrayRef<int64_t>) {
  return success(); // base access op handles buffer-level checks
}

LogicalResult CollapseShapeOp::verifyCompatibility(BufferTypeInterface,
                                                   ArrayRef<int64_t>) {
  return success();
}

LogicalResult TransposeOp::verifyCompatibility(BufferTypeInterface,
                                               ArrayRef<int64_t>) {
  return success();
}

FailureOr<Value> ExpandShapeOp::materialize(OpBuilder &builder, Location loc,
                                            Value buffer) {
  auto sourceOp = getSource().getDefiningOp<BufferAccessOpInterface>();
  if (!sourceOp)
    return emitError() << "source must be a buffer access op";
  auto baseSlice = sourceOp.materialize(builder, loc, buffer);
  if (failed(baseSlice))
    return failure();

  auto reassoc = getReassociationIndices();
  auto mixedOutputShape =
      getMixedValues(getStaticOutputShape(), getOutputShape(), builder);
  SmallVector<OpFoldResult> outputShape(mixedOutputShape.begin(),
                                        mixedOutputShape.end());
  auto srcType = cast<RankedTensorType>((*baseSlice).getType());
  SmallVector<int64_t> resultDims;
  for (auto ofr : outputShape) {
    if (auto attr = dyn_cast<Attribute>(ofr))
      resultDims.push_back(cast<IntegerAttr>(attr).getInt());
    else
      resultDims.push_back(ShapedType::kDynamic);
  }
  auto resultType = RankedTensorType::get(resultDims, srcType.getElementType());
  Value expanded = tensor::ExpandShapeOp::create(
      builder, loc, resultType, *baseSlice, reassoc, outputShape);
  return expanded;
}

FailureOr<Value> ExpandShapeOp::materialize(OpBuilder &builder, Location loc,
                                            Value value, Value buffer) {
  auto sourceOp = getSource().getDefiningOp<BufferAccessOpInterface>();
  if (!sourceOp)
    return emitError() << "source must be a buffer access op";
  auto reassoc = getReassociationIndices();
  Value collapsed =
      tensor::CollapseShapeOp::create(builder, loc, value, reassoc);
  return sourceOp.materialize(builder, loc, collapsed, buffer);
}

FailureOr<Value> CollapseShapeOp::materialize(OpBuilder &builder, Location loc,
                                              Value buffer) {
  auto sourceOp = getSource().getDefiningOp<BufferAccessOpInterface>();
  if (!sourceOp)
    return emitError() << "source must be a buffer access op";
  auto baseSlice = sourceOp.materialize(builder, loc, buffer);
  if (failed(baseSlice))
    return failure();

  auto reassoc = getReassociationIndices();
  Value collapsed =
      tensor::CollapseShapeOp::create(builder, loc, *baseSlice, reassoc);
  return collapsed;
}

FailureOr<Value> CollapseShapeOp::materialize(OpBuilder &builder, Location loc,
                                              Value value, Value buffer) {
  auto sourceOp = getSource().getDefiningOp<BufferAccessOpInterface>();
  if (!sourceOp)
    return emitError() << "source must be a buffer access op";
  auto baseSlice = sourceOp.materialize(builder, loc, buffer);
  if (failed(baseSlice))
    return failure();
  auto expandedType = cast<RankedTensorType>((*baseSlice).getType());
  SmallVector<OpFoldResult> outputShape;
  for (int64_t i = 0; i < expandedType.getRank(); ++i) {
    if (expandedType.isDynamicDim(i))
      outputShape.push_back(
          tensor::DimOp::create(builder, loc, *baseSlice, i).getResult());
    else
      outputShape.push_back(
          builder.getI64IntegerAttr(expandedType.getDimSize(i)));
  }
  auto reassoc = getReassociationIndices();
  Value expanded = tensor::ExpandShapeOp::create(builder, loc, expandedType,
                                                 value, reassoc, outputShape);
  return sourceOp.materialize(builder, loc, expanded, buffer);
}

FailureOr<Value> TransposeOp::materialize(OpBuilder &builder, Location loc,
                                          Value buffer) {
  auto sourceOp = getSource().getDefiningOp<BufferAccessOpInterface>();
  if (!sourceOp)
    return emitError() << "source must be a buffer access op";
  auto baseSlice = sourceOp.materialize(builder, loc, buffer);
  if (failed(baseSlice))
    return failure();

  auto srcType = cast<RankedTensorType>((*baseSlice).getType());
  auto perm = getPermutation();
  SmallVector<int64_t> transposedShape;
  SmallVector<Value> dynamicDims;
  for (int64_t p : perm) {
    transposedShape.push_back(srcType.getDimSize(p));
    if (srcType.isDynamicDim(p))
      dynamicDims.push_back(tensor::DimOp::create(builder, loc, *baseSlice, p));
  }
  auto destType =
      RankedTensorType::get(transposedShape, srcType.getElementType());
  Value empty = tensor::EmptyOp::create(builder, loc, destType, dynamicDims);
  auto transposeOp =
      linalg::TransposeOp::create(builder, loc, *baseSlice, empty, perm);
  return transposeOp->getResult(0);
}

FailureOr<Value> TransposeOp::materialize(OpBuilder &builder, Location loc,
                                          Value value, Value buffer) {
  auto sourceOp = getSource().getDefiningOp<BufferAccessOpInterface>();
  if (!sourceOp)
    return emitError() << "source must be a buffer access op";
  auto invPerm = getInversePermutation();
  auto valType = cast<RankedTensorType>(value.getType());
  SmallVector<int64_t> invShape;
  SmallVector<Value> dynamicDims;
  for (int64_t p : invPerm) {
    invShape.push_back(valType.getDimSize(p));
    if (valType.isDynamicDim(p))
      dynamicDims.push_back(tensor::DimOp::create(builder, loc, value, p));
  }
  auto invType = RankedTensorType::get(invShape, valType.getElementType());
  Value empty = tensor::EmptyOp::create(builder, loc, invType, dynamicDims);
  auto transposeOp =
      linalg::TransposeOp::create(builder, loc, value, empty, invPerm);
  Value invTransposed = transposeOp->getResult(0);
  return sourceOp.materialize(builder, loc, invTransposed, buffer);
}

//===----------------------------------------------------------------------===//
// Instruction ops: define / emit / sequence / launch
//===----------------------------------------------------------------------===//

void DefineOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printSymbolName(getSymName());
  p << " {";
  p.increaseIndent();
  p.printNewline();
  p << "src(";
  llvm::interleaveComma(
      getSources().getAsRange<FlatSymbolRefAttr>(), p,
      [&](FlatSymbolRefAttr src) { p.printSymbolName(src.getValue()); });
  p << ") dst(";
  llvm::interleaveComma(
      getDestinations().getAsRange<FlatSymbolRefAttr>(), p,
      [&](FlatSymbolRefAttr dst) { p.printSymbolName(dst.getValue()); });
  p << ")";
  p.printNewline();
  p << "addr(";
  llvm::interleaveComma(getAccessBlock().getArguments(), p,
                        [&](BlockArgument arg) { p.printRegionArgument(arg); });
  p << ") ";
  p.printRegion(getAccess(), /*printEntryBlockArgs=*/false);
  p.printNewline();
  p << "compute(";
  llvm::interleaveComma(getSemanticsBlock().getArguments(), p,
                        [&](BlockArgument arg) { p.printRegionArgument(arg); });
  p << ")";
  p.printRegion(getSemantics(), /*printEntryBlockArgs=*/false);
  p.decreaseIndent();
  p.printNewline();
  p << '}';
  p.printOptionalAttrDict(getOperation()->getAttrs(),
                          /*elidedAttrs=*/{SymbolTable::getSymbolAttrName(),
                                           getSourcesAttrName(),
                                           getDestinationsAttrName()});
}

ParseResult DefineOp::parse(OpAsmParser &parser, OperationState &result) {
  StringAttr symName;
  SmallVector<Attribute, 4> srcs, dsts;
  if (parser.parseSymbolName(symName) || parser.parseLBrace() ||
      parser.parseKeyword("src") || parser.parseLParen() ||
      parser.parseCommaSeparatedList([&]() {
        FlatSymbolRefAttr src;
        if (parser.parseAttribute(src))
          return failure();
        srcs.push_back(src);
        return success();
      }) ||
      parser.parseRParen() || parser.parseKeyword("dst") ||
      parser.parseLParen() || parser.parseCommaSeparatedList([&]() {
        FlatSymbolRefAttr dst;
        if (parser.parseAttribute(dst))
          return failure();
        dsts.push_back(dst);
        return success();
      }) ||
      parser.parseRParen())
    return failure();
  auto builder = parser.getBuilder();
  result.addAttribute(SymbolTable::getSymbolAttrName(), symName);
  result.addAttribute(getSourcesAttrName(result.name),
                      builder.getArrayAttr(srcs));
  result.addAttribute(getDestinationsAttrName(result.name),
                      builder.getArrayAttr(dsts));
  Region *accessRegion = result.addRegion();
  Region *semanticsRegion = result.addRegion();
  SmallVector<OpAsmParser::Argument, 4> addrArgs, computeArgs;
  if (parser.parseKeyword("addr") ||
      parser.parseArgumentList(addrArgs, AsmParser::Delimiter::Paren,
                               /*allowType=*/true, /*allowAttrs=*/false))
    return failure();
  // parse access region
  if (parser.parseRegion(*accessRegion, addrArgs))
    return failure();
  if (parser.parseKeyword("compute") ||
      parser.parseArgumentList(computeArgs, AsmParser::Delimiter::Paren,
                               /*allowType=*/true, /*allowAttrs=*/false))
    return failure();
  // parse semantics region
  if (parser.parseRegion(*semanticsRegion, computeArgs))
    return failure();
  if (parser.parseRBrace())
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  return success();
}

LogicalResult DefineOp::verify() {
  Block &access = getAccessBlock();
  if (access.empty() || !isa<YieldOp>(access.back()))
    return emitError() << "access region must end with a yield op";
  Operation &accYield = access.back();
  unsigned nBuffers = getSources().size() + getDestinations().size();
  if (accYield.getNumOperands() != nBuffers)
    return emitError()
           << "access region must yield the same number of buffer access "
              "patterns as the number of source and destination buffers";
  if (llvm::any_of(accYield.getOperands(), [](Value v) {
        return v.getDefiningOp<BufferAccessOpInterface>() == nullptr;
      }))
    return emitError()
           << "access region must yield only buffer access patterns";

  if (llvm::any_of(access.getArgumentTypes(),
                   [](Type t) { return !t.isIndex(); }))
    return emitError() << "access region arguments must all be index type";

  Block &semantics = getSemanticsBlock();
  if (semantics.empty() || !isa<YieldOp>(semantics.back()))
    return emitError() << "semantics region must end with a yield op";
  Operation &semYield = semantics.back();
  if (semYield.getNumOperands() < getDestinations().size())
    return emitError() << "semantics region must yield the same number of "
                          "values as the number of destination buffers";
  if (semantics.getNumArguments() < nBuffers)
    return emitError()
           << "number of semantics region arguments must be at least the total "
              "number of source and destination buffers";

  auto bufferTys = llvm::drop_end(semantics.getArgumentTypes(),
                                  semantics.getNumArguments() - nBuffers);
  auto computeParamTys =
      llvm::drop_begin(semantics.getArgumentTypes(), nBuffers);
  for (auto argTy : bufferTys) {
    if (!isa<RankedTensorType>(argTy))
      return emitError()
             << "semantics region arguments must all be ranked tensors."
             << "use 0-d tensors for scalars";
  }
  for (auto argTy : computeParamTys) {
    if (!argTy.isIntOrIndex())
      return emitError() << "compute parameters must be int or index "
                         << "types";
  }

  // The two regions are written independently, so nothing but this ties them
  // together: each yielded semantics value must have the type of the block
  // argument standing for the destination it is written into -- i.e. the shape
  // the destination's access pattern writes. A dynamic dim on either side is a
  // parameter resolved per emit, so it matches anything.
  unsigned nSources = getSources().size();
  for (unsigned k = 0, e = getDestinations().size(); k < e; ++k) {
    Type yieldedTy = semYield.getOperand(k).getType();
    auto got = dyn_cast<RankedTensorType>(yieldedTy);
    auto want =
        cast<RankedTensorType>(semantics.getArgument(nSources + k).getType());
    if (!got || got.getElementType() != want.getElementType() ||
        got.getRank() != want.getRank())
      return emitError() << "semantics region yields " << yieldedTy
                         << " for destination #" << k
                         << " but its access pattern writes " << want;
    for (int64_t dim = 0; dim < got.getRank(); ++dim) {
      int64_t g = got.getDimSize(dim), w = want.getDimSize(dim);
      if (!ShapedType::isDynamic(g) && !ShapedType::isDynamic(w) && g != w)
        return emitError() << "semantics region yields " << got
                           << " for destination #" << k
                           << " but its access pattern writes " << want
                           << " (dim " << dim << ": " << g << " vs " << w
                           << ")";
    }
  }
  return success();
}

LogicalResult DefineOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // The extents are ArrayRefs into each declaration's own attribute storage, so
  // they stay valid for the whole check.
  SmallVector<std::pair<BufferTypeInterface, ArrayRef<int64_t>>, 4> bufferArgs;
  for (auto source : getSources().getAsRange<FlatSymbolRefAttr>()) {
    auto sourceOp =
        symbolTable.lookupNearestSymbolFrom<DeclareBufferOp>(*this, source);
    if (!sourceOp)
      return emitError() << "referred source buffer '" << source
                         << "' does not exist";
    bufferArgs.push_back({sourceOp.getBufferType(), sourceOp.getExtents()});
  }
  for (auto dest : getDestinations().getAsRange<FlatSymbolRefAttr>()) {
    auto destOp =
        symbolTable.lookupNearestSymbolFrom<DeclareBufferOp>(*this, dest);
    if (!destOp)
      return emitError() << "referred destination buffer '" << dest
                         << "' does not exist";
    bufferArgs.push_back({destOp.getBufferType(), destOp.getExtents()});
  }
  // Walk relayout chains to find base access ops for buffer compatibility
  SmallVector<BufferAccessOpInterface, 4> basePatterns;
  for (auto operand : getAccessBlock().getTerminator()->getOperands()) {
    auto pattern = operand.getDefiningOp<BufferAccessOpInterface>();
    assert(pattern);
    Operation *curr = pattern.getOperation();
    while (auto relayout = dyn_cast<BufferRelayoutOpInterface>(curr))
      curr = relayout.getSource().getDefiningOp();
    basePatterns.push_back(cast<BufferAccessOpInterface>(curr));
  }
  for (auto [bufferArg, pattern] : llvm::zip(bufferArgs, basePatterns)) {
    auto [slotType, extents] = bufferArg;
    if (failed(pattern.verifyCompatibility(slotType, extents)))
      return emitError() << "buffer access pattern is not compatible with the "
                            "referred buffer";
  }
  return success();
}

LogicalResult EmitOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto defineOp = symbolTable.lookupNearestSymbolFrom<DefineOp>(
      *this, getInstructionAttr());
  if (!defineOp)
    return emitError() << "referred instruction '" << getInstruction()
                       << "' does not exist";
  if (defineOp.getAccessBlock().getNumArguments() !=
      getStaticAddrParams().size())
    return emitError() << "number of address parameters must match the number "
                          "of access region arguments";
  if (defineOp.getExtraComputeArgs().size() != getStaticComputeParams().size())
    return emitError() << "number of compute parameters must match the number "
                          "of semantics region arguments";
  return success();
}

static bool isUnsupportedSequenceControlOp(Operation *op) {
  return isa<LoopLikeOpInterface, BranchOpInterface, RegionBranchOpInterface>(
             op) ||
         isa<affine::AffineIfOp, scf::IfOp>(op);
}

LogicalResult SequenceOp::verify() {
  Region &body = getBody();
  if (body.empty())
    return emitError() << "sequence body must contain an entry block";
  if (body.front().getNumArguments() != 0)
    return emitError() << "sequence entry block must not have arguments";

  WalkResult walkResult = body.walk([&](Operation *op) {
    if (!isUnsupportedSequenceControlOp(op))
      return WalkResult::advance();
    op->emitError() << "control flow is not supported in allo.sequence";
    return WalkResult::interrupt();
  });
  return success(!walkResult.wasInterrupted());
}

LogicalResult LaunchOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto seqOp =
      symbolTable.lookupNearestSymbolFrom<SequenceOp>(*this, getSequenceAttr());
  if (!seqOp)
    return emitError() << "referred sequence '" << getSequence()
                       << "' does not exist";
  return success();
}
