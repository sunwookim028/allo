/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/IR/AlloOps.h"
#include "allo/IR/AlloAttrs.h"
#include "allo/IR/AlloTypes.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Format.h"

#include <cmath>

#include "allo/IR/AlloDialect.cpp.inc"

#include "allo/IR/AlloEnums.cpp.inc"

// Op interfaces must precede the op/type classes that implement them.
#include "allo/IR/AlloOpInterfaces.cpp.inc"

#define GET_OP_CLASSES
#include "allo/IR/AlloOps.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "allo/IR/AlloAttrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "allo/IR/AlloTypes.cpp.inc"

#include "mlir/Interfaces/FunctionImplementation.h"

using namespace mlir;
using namespace mlir::allo;

LogicalResult
StreamType::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                   Type baseType, std::size_t depth, ArrayRef<int64_t> shape) {
  if (!baseType)
    return emitError() << "expected stream base type";
  if (depth == 0)
    return emitError() << "stream depth must be positive";
  for (int64_t dim : shape) {
    if (dim < 0)
      return emitError() << "stream shape dimensions must be non-negative";
  }
  return success();
}

Type StreamType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return {};

  Type baseType;
  uint64_t depth = 0;
  SmallVector<int64_t> shape;
  if (parser.parseType(baseType) || parser.parseComma() ||
      parser.parseInteger(depth) || parser.parseComma() ||
      parser.parseLSquare())
    return {};

  if (failed(parser.parseOptionalRSquare())) {
    do {
      int64_t dim = 0;
      if (parser.parseInteger(dim))
        return {};
      shape.push_back(dim);
    } while (succeeded(parser.parseOptionalComma()));
    if (parser.parseRSquare())
      return {};
  }

  if (parser.parseGreater())
    return {};
  return parser.getChecked<StreamType>(
      parser.getCurrentLocation(), parser.getContext(), baseType, depth, shape);
}

void StreamType::print(AsmPrinter &printer) const {
  printer << "<" << getBaseType() << "," << getDepth() << ",[";
  for (auto [idx, dim] : llvm::enumerate(getShape())) {
    if (idx != 0)
      printer << ",";
    printer << dim;
  }
  printer << "]>";
}

void KernelOp::print(OpAsmPrinter &p) {
  p << ' ';
  auto op = llvm::cast<FunctionOpInterface>(getOperation());
  StringRef visibilityAttrName = SymbolTable::getVisibilityAttrName();
  if (auto visibilty = op->getAttrOfType<StringAttr>(visibilityAttrName)) {
    p << visibilty.getValue() << ' ';
  }
  auto kName = getSymNameAttr().getValue();
  p.printSymbolName(kName);
  function_interface_impl::printFunctionSignature(p, op, getArgumentTypes(),
                                                  false, getResultTypes());
  p << " mapping=";
  p.printStrippedAttrOrType(getMappingAttr());
  function_interface_impl::printFunctionAttributes(
      p, op,
      {
          SymbolTable::getVisibilityAttrName(),
          getFunctionTypeAttrName(),
          getArgAttrsAttrName(),
          getMappingAttrName(),

      });
  Region &body = getRegion();
  if (!body.empty()) {
    p << ' ';
    p.printRegion(body, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
  }
}

ParseResult KernelOp::parse(OpAsmParser &p, OperationState &result) {
  SmallVector<OpAsmParser::Argument> entryArgs;
  SmallVector<DictionaryAttr> resAttrs;
  SmallVector<Type> resTypes;
  auto &builder = p.getBuilder();

  (void)impl::parseOptionalVisibilityKeyword(p, result.attributes);

  StringAttr nameAttr;
  if (p.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                        result.attributes))
    return failure();

  bool isVariadic = false;
  if (function_interface_impl::parseFunctionSignatureWithArguments(
          p, false, entryArgs, isVariadic, resTypes, resAttrs))
    return failure();
  SmallVector<Type> argTypes;
  for (auto &arg : entryArgs)
    argTypes.push_back(arg.type);
  FunctionType type = builder.getFunctionType(argTypes, resTypes);
  result.addAttribute(getFunctionTypeAttrName(result.name),
                      TypeAttr::get(type));

  if (p.parseKeyword("mapping") || p.parseEqual())
    return failure();
  DenseI32ArrayAttr mapping;
  if (p.parseCustomAttributeWithFallback(mapping, Type()))
    return failure();
  result.addAttribute(getMappingAttrName(result.name), mapping);

  NamedAttrList parsedAttributes;
  if (p.parseOptionalAttrDictWithKeyword(parsedAttributes))
    return failure();

  result.attributes.append(parsedAttributes);

  assert(resAttrs.size() == resTypes.size());
  call_interface_impl::addArgAndResultAttrs(
      builder, result, entryArgs, resAttrs, getArgAttrsAttrName(result.name),
      getResAttrsAttrName(result.name));

  // The printer omits an empty body, so the parser must reject one to keep the
  // round-trip.
  auto *body = result.addRegion();
  SMLoc loc = p.getCurrentLocation();
  OptionalParseResult parseResult =
      p.parseOptionalRegion(*body, entryArgs,
                            /*enableNameShadowing=*/false);
  if (parseResult.has_value()) {
    if (failed(*parseResult))
      return failure();
    if (body->empty())
      return p.emitError(loc, "expected non-empty function body");
  }
  return success();
}

void KernelOp::build(OpBuilder &builder, OperationState &state, StringRef name,
                     FunctionType type, ArrayRef<int32_t> mapping,
                     ArrayRef<NamedAttribute> attrs,
                     ArrayRef<DictionaryAttr> argAttrs) {
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute(getFunctionTypeAttrName(state.name), TypeAttr::get(type));
  state.addAttribute(getMappingAttrName(state.name),
                     builder.getDenseI32ArrayAttr(mapping));
  state.attributes.append(attrs);
  state.addRegion();

  if (argAttrs.empty())
    return;
  assert(type.getNumInputs() == argAttrs.size());
  call_interface_impl::addArgAndResultAttrs(
      builder, state, argAttrs, /*resultAttrs=*/{},
      getArgAttrsAttrName(state.name), getResAttrsAttrName(state.name));
}

LogicalResult ReturnOp::verify() {
  auto kernel = cast<KernelOp>(this->getParentOp());
  auto results = kernel.getFunctionType().getResults();
  if (results.size() != getNumOperands())
    return emitOpError("has ")
           << getNumOperands() << " operands, but enclosing function (@"
           << kernel.getName() << ") returns " << results.size();

  for (unsigned i = 0, e = results.size(); i != e; ++i)
    if (getOperand(i).getType() != results[i])
      return emitError() << "type of return operand " << i << " ("
                         << getOperand(i).getType()
                         << ") doesn't match function result type ("
                         << results[i] << ")"
                         << " in kernel @" << kernel.getSymName();
  return success();
}

LogicalResult InvokeOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto fnAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!fnAttr)
    return emitOpError("requires a 'callee' symbol reference attribute");
  KernelOp fn = symbolTable.lookupNearestSymbolFrom<KernelOp>(*this, fnAttr);
  if (!fn)
    return emitOpError() << "'" << fnAttr.getValue()
                         << "' does not reference a valid kernel";

  auto fnType = fn.getFunctionType();
  if (fnType.getNumInputs() != getNumOperands())
    return emitOpError("incorrect number of operands for callee");

  for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i)
    if (getOperand(i).getType() != fnType.getInput(i))
      return emitOpError("operand type mismatch: expected operand type ")
             << fnType.getInput(i) << ", but provided "
             << getOperand(i).getType() << " for operand number " << i;

  if (fnType.getNumResults() != getNumResults())
    return emitOpError("incorrect number of results for callee");

  for (unsigned i = 0, e = fnType.getNumResults(); i != e; ++i)
    if (getResult(i).getType() != fnType.getResult(i)) {
      auto diag = emitOpError("result type mismatch at index ") << i;
      diag.attachNote() << "    op result types: " << getResultTypes();
      diag.attachNote() << "kernel result types: " << fnType.getResults();
      return diag;
    }

  return success();
}

void AlloDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "allo/IR/AlloAttrs.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "allo/IR/AlloTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "allo/IR/AlloOps.cpp.inc"
      >();
}

LogicalResult
PartitionAxisAttr::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                          PartitionKindEnum kind, int64_t factor,
                          int64_t dims) {
  if (kind == PartitionKindEnum::CompletePartition && factor != 0) {
    return emitError() << "partition factor must be 0 for complete partition";
  }
  if (kind != PartitionKindEnum::CompletePartition && !(factor > 1)) {
    return emitError() << "partition factor must be greater than 1 for "
                          "non-complete partition";
  }
  if (dims < 0) {
    return emitError() << "dimension index must be non-negative";
  }
  // `dim == 0` means "every dimension" for block and cyclic. A skew already
  // reads every subscript, so its `dim` names the single one it divides down.
  if (kind == PartitionKindEnum::SkewPartition && dims == 0) {
    return emitError() << "skew partition must name its distribution dimension";
  }
  return success();
}

LogicalResult
PartitionAttr::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                      ArrayRef<PartitionAxisAttr> partitions) {
  DenseSet<int64_t> seen;
  for (auto &axi : partitions) {
    seen.insert(axi.getDim());
  }
  if (seen.size() < partitions.size()) {
    return emitError() << "duplicate partition axis detected";
  }
  // A skew reads every subscript, so composing it with another axis would ask
  // one subscript to serve two digits.
  if (partitions.size() > 1 &&
      llvm::any_of(partitions, [](PartitionAxisAttr a) {
        return a.getKind() == PartitionKindEnum::SkewPartition;
      })) {
    return emitError() << "a skew partition must be an array's only axis";
  }
  return success();
}

LogicalResult StreamGetOp::verify() {
  auto streamTy = cast<StreamType>(getStream().getType());
  auto valueTy = getValue().getType();
  if (streamTy.getBaseType() != valueTy) {
    return emitOpError() << "stream type " << streamTy
                         << " does not match value type " << valueTy;
  }
  auto srcRank = streamTy.getShape().size();
  auto dstRank = getIndices().size();
  if (srcRank != dstRank) {
    return emitOpError() << "rank of stream (" << srcRank
                         << ") does not match number of indices (" << dstRank
                         << ")";
  }
  return success();
}

LogicalResult StreamPutOp::verify() {
  auto streamTy = cast<StreamType>(getStream().getType());
  auto valueTy = getValue().getType();
  if (streamTy.getBaseType() != valueTy) {
    return emitOpError() << "stream type " << streamTy
                         << " does not match value type " << valueTy;
  }
  auto dstRank = streamTy.getShape().size();
  auto srcRank = getIndices().size();
  if (srcRank != dstRank) {
    return emitOpError() << "rank of stream (" << dstRank
                         << ") does not match number of indices (" << srcRank
                         << ")";
  }
  return success();
}

LogicalResult AssumeNoDepOp::verify() {
  // A distance is an inter-iteration notion.
  if (getDistanceAttr() && getDepType() == AssumeDepTypeEnum::Intra)
    return emitOpError() << "'distance' is only meaningful for an inter-"
                            "iteration dependence (dep_type = inter)";
  return success();
}

//===----------------------------------------------------------------------===//
// Data & Control Path (dcp) operations
//===----------------------------------------------------------------------===//

namespace mlir::allo::dcp {

LogicalResult
DCPathUnitOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  if (!symbolTable.lookupNearestSymbolFrom<DCPathOperatorOp>(*this,
                                                             getOpTypeAttr()))
    return emitOpError("references unknown operator type '")
           << getOpType() << "'";
  return success();
}

LogicalResult
DCPathComputeOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  if (FlatSymbolRefAttr opType = getOpTypeAttr())
    if (!symbolTable.lookupNearestSymbolFrom<DCPathOperatorOp>(*this, opType))
      return emitOpError("references unknown operator type '")
             << opType.getValue() << "'";
  if (FlatSymbolRefAttr unit = getUnitAttr())
    if (!symbolTable.lookupNearestSymbolFrom<DCPathUnitOp>(*this, unit))
      return emitOpError("references unknown unit '") << unit.getValue() << "'";
  return success();
}

LogicalResult DCPathComputeOp::verify() {
  if (getStart() < 0)
    return emitOpError("start cycle must be non-negative");
  // Exactly one realization path: a combinational kind or an operator symbol.
  if (getCombKindAttr() && getOpTypeAttr())
    return emitOpError("has both 'comb_kind' and 'op_type'; set exactly one");
  if (!getCombKindAttr() && !getOpTypeAttr())
    return emitOpError(
        "has neither 'comb_kind' nor 'op_type'; set exactly one");
  return success();
}

LogicalResult DCPathDeviceOp::verify() {
  // One row per kind: the library keeps the LAST match, so a duplicate is a
  // declaration silently overriding another.
  llvm::DenseSet<OpKindEnum> seen;
  for (DCPathCombOp comb : getBody().getOps<DCPathCombOp>())
    if (!seen.insert(comb.getKind()).second)
      return emitOpError("declares combinational kind '")
             << stringifyOpKindEnum(comb.getKind()) << "' twice";
  // The two whole-device settings, for the same reason: a second one would
  // silently win over the first.
  auto tooMany = [](auto range) {
    return !range.empty() && !llvm::hasSingleElement(range);
  };
  if (llvm::count_if(getBody().getOps<DCPathStorageOp>(),
                     [](DCPathStorageOp s) { return s.getIsDefault(); }) > 1)
    return emitOpError("marks more than one dcp.storage `default`");
  if (llvm::count_if(getBody().getOps<DCPathStorageOp>(),
                     [](DCPathStorageOp s) { return s.getIsScatter(); }) > 1)
    return emitOpError("marks more than one dcp.storage `scatter`");
  if (tooMany(getBody().getOps<DCPathStreamTimingOp>()))
    return emitOpError("declares more than one dcp.stream_timing");
  if (tooMany(getBody().getOps<DCPathMuxOp>()))
    return emitOpError("declares more than one dcp.mux");
  if (tooMany(getBody().getOps<DCPathChainOp>()))
    return emitOpError("declares more than one dcp.chain");
  return success();
}

// Each `uses` entry is one product TERM, carrying one cost factor per parameter
// of the realization's kind (\p arity of them, spelled by \p params for the
// diagnostic). A wrong count would otherwise reach the evaluator, which zips
// factors against parameters and cannot tell a missing one from a whole tuple.
//
// A resource may appear in several entries, which is what makes the cost a sum
// of products rather than one product: `2*width + depth - 1` is a real measured
// shape and no single product is a sum. The price of that is the check this
// used to make, that a resource is named ONCE, which caught a typo repeating a
// row. There is nothing left to distinguish that typo from a second term.
static LogicalResult verifyResourceUses(Operation *op, ArrayAttr uses,
                                        unsigned arity, StringRef params) {
  if (!uses)
    return success();
  for (Attribute use : uses) {
    auto ru = dyn_cast<ResourceUseAttr>(use);
    if (!ru)
      return op->emitOpError(
          "'uses' holds an entry that is not an #allo.res_use");
    ArrayRef<CostAttr> factors = ru.getFactors();
    // One factor per parameter, or the single `tiled` that reads them together.
    if (factors.size() == arity)
      continue;
    if (factors.size() == 1 && factors.front().getForm() == CostFormEnum::Tiled)
      continue;
    return op->emitOpError("is characterized by ")
           << params << ", so its cost of '" << ru.getResource() << "' takes "
           << arity << " factor(s) or one 'tiled', not " << factors.size();
  }
  return success();
}

// Every resource a realization spends must be one this device declares: the
// symbol is what turns a misspelling into an error instead of a free row.
static LogicalResult verifyUsesResolve(Operation *op, ArrayAttr uses,
                                       SymbolTableCollection &symbolTable) {
  if (!uses)
    return success();
  for (Attribute use : uses) {
    auto ru = cast<ResourceUseAttr>(use);
    if (!symbolTable.lookupNearestSymbolFrom<DCPathResourceOp>(
            op, ru.getResource()))
      return op->emitOpError("spends '")
             << ru.getResource() << "', which is not a dcp.resource";
  }
  return success();
}

LogicalResult DCPathCombOp::verify() {
  // Sampled at representative widths rather than checked symbolically: a cost
  // form is piecewise, so non-negativity is not a property of the
  // coefficients. A width the row was not measured at is skipped.
  for (int64_t w : {1, 8, 16, 32, 64}) {
    std::optional<double> d = getDelay().evaluate(w);
    if (d && *d < 0.0)
      return emitOpError("delay must be non-negative, but is ")
             << *d << " ns at width " << w;
  }
  return verifyResourceUses(*this, getUsesAttr(), 1,
                            "one parameter (an operand width)");
}

LogicalResult
DCPathCombOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyUsesResolve(*this, getUsesAttr(), symbolTable);
}

// A 0-cycle write is not checked here even though no array can be realized at
// one: `PreVerification` reports it against the array that resolved to this
// row, and a row nothing resolves to costs nothing.
LogicalResult DCPathStorageOp::verify() {
  if (getRdDelay().convertToDouble() < 0.0 ||
      getWrDelay().convertToDouble() < 0.0)
    return emitOpError("delay must be non-negative");
  if (getIsScatter() && getIsTable())
    return emitOpError("a row is one structure: `scatter` is a cell per "
                       "element and `table` a constant lookup");
  if (getIsTable() && getNoInit())
    return emitOpError("a `table` row holds compile-time contents, so it "
                       "cannot be one that powers up undefined");
  std::optional<uint64_t> pool = getInstPorts();
  for (std::optional<uint64_t> limit :
       {getInstReads(), getInstWrites(), pool}) {
    if (limit && *limit < 1)
      return emitOpError("a port limit must be at least one port");
    if (limit && (getIsScatter() || getIsTable()))
      return emitOpError("a `scatter` or `table` row is not addressed and so "
                         "has no port limit to declare");
    if (limit && pool && *limit > *pool)
      return emitOpError("a direction's port limit exceeds `inst_p`, but an "
                         "access of either direction takes one port of the "
                         "pool");
  }
  return verifyResourceUses(*this, getUsesAttr(), 2,
                            "two parameters (depth, width)");
}

LogicalResult
DCPathStorageOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyUsesResolve(*this, getUsesAttr(), symbolTable);
}

LogicalResult DCPathMuxOp::verify() {
  return verifyResourceUses(*this, getUsesAttr(), 2,
                            "two parameters (fan-in, width)");
}

LogicalResult
DCPathMuxOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyUsesResolve(*this, getUsesAttr(), symbolTable);
}

LogicalResult DCPathChainOp::verify() {
  return verifyResourceUses(*this, getUsesAttr(), 2,
                            "two parameters (depth, width)");
}

LogicalResult
DCPathChainOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyUsesResolve(*this, getUsesAttr(), symbolTable);
}

// An IP core's area is a function of its operand width, the same one parameter
// a `dcp.comb` carries, even though `signature` already fixes that width.
LogicalResult DCPathOperatorOp::verify() {
  return verifyResourceUses(*this, getUsesAttr(), 1,
                            "one parameter (an operand width)");
}

// This op is at module scope and the resources it spends are the device's, so
// the reference has to name the device it reaches through (`@u55c::@lut`);
// `verifyUsesResolve` walks it like any other nested symbol.
LogicalResult
DCPathOperatorOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyUsesResolve(*this, getUsesAttr(), symbolTable);
}

//===----------------------------------------------------------------------===//
// dcp.operator custom assembly
//===----------------------------------------------------------------------===//

// Print a delay as a compact decimal (e.g. `0.5`, `1.2`), instead of the
// exponent form MLIR uses for float attributes (`5.000000e-01`).
static void printNum(OpAsmPrinter &p, double v) { p << llvm::format("%g", v); }

// Parse a delay, accepting both a float literal and a whole number (which the
// compact printer above emits without a trailing dot, e.g. `0` or `2`).
static ParseResult parseNum(OpAsmParser &p, double &v) {
  APInt whole;
  if (OptionalParseResult r = p.parseOptionalInteger(whole); r.has_value()) {
    if (failed(*r))
      return failure();
    v = whole.getSExtValue();
    return success();
  }
  return p.parseFloat(v);
}

// Parse the optional determinacy keyword a scheduling region prints just before
// its attr-dict. Any bare keyword in that position is a determinacy class, so
// an unknown one is an error.
static ParseResult parseOptionalDeterminacy(OpAsmParser &p,
                                            OperationState &result,
                                            StringAttr attrName) {
  StringRef kw;
  if (failed(p.parseOptionalKeyword(&kw)))
    return success();
  std::optional<DeterminacyEnum> d = symbolizeDeterminacyEnum(kw);
  if (!d)
    return p.emitError(p.getNameLoc(), "unknown determinacy '") << kw << "'";
  result.addAttribute(attrName,
                      DeterminacyEnumAttr::get(result.getContext(), *d));
  return success();
}

void DCPathModuleOp::build(OpBuilder &b, OperationState &state, StringRef name,
                           FunctionType type, DeterminacyEnum determinacy) {
  state.addAttribute(SymbolTable::getSymbolAttrName(), b.getStringAttr(name));
  state.addAttribute(getFunctionTypeAttrName(state.name), TypeAttr::get(type));
  state.addAttribute(getDeterminacyAttrName(state.name),
                     DeterminacyEnumAttr::get(b.getContext(), determinacy));
  state.addRegion();
}

// The kernel's timing contract. `counted_static` licenses placing a consumer at
// a fixed offset from the call, so it is the one class that must be backed by
// an exact latency, and the only one (besides `concurrent`) that may carry one.
LogicalResult DCPathModuleOp::verify() {
  std::optional<int64_t> lat = getLatency();
  if (getLatencyBound() && !lat)
    return emitOpError("latency_bound requires latency");
  bool exact = lat && !getLatencyBound();
  DeterminacyEnum d = getDeterminacy();
  if (d == DeterminacyEnum::CountedStatic && !exact)
    return emitOpError("a counted_static kernel needs an exact latency, not a ")
           << (lat ? "bounded one" : "missing one");
  // A concurrent container's span is a completion floor over self-timed
  // processes, so it promises a caller no offset.
  if (exact && d != DeterminacyEnum::CountedStatic &&
      d != DeterminacyEnum::Concurrent)
    return emitOpError("an exact latency contradicts determinacy ")
           << stringifyDeterminacyEnum(d);
  return success();
}

void DCPathModuleOp::print(OpAsmPrinter &p) {
  p << ' ';
  auto op = llvm::cast<FunctionOpInterface>(getOperation());
  if (auto vis =
          op->getAttrOfType<StringAttr>(SymbolTable::getVisibilityAttrName()))
    p << vis.getValue() << ' ';
  p.printSymbolName(getSymName());
  function_interface_impl::printFunctionSignature(p, op, getArgumentTypes(),
                                                  false, getResultTypes());
  if (IntegerAttr lat = getLatencyAttr()) {
    p << " lat=" << lat.getInt();
    if (getLatencyBound())
      p << " bound";
  }
  p << ' ' << stringifyDeterminacyEnum(getDeterminacy());
  function_interface_impl::printFunctionAttributes(
      p, op,
      {SymbolTable::getVisibilityAttrName(), getFunctionTypeAttrName(),
       getArgAttrsAttrName(), getResAttrsAttrName(), getLatencyAttrName(),
       getLatencyBoundAttrName(), getDeterminacyAttrName()});
  p << ' ';
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
}

ParseResult DCPathModuleOp::parse(OpAsmParser &p, OperationState &result) {
  SmallVector<OpAsmParser::Argument> entryArgs;
  SmallVector<DictionaryAttr> resAttrs;
  SmallVector<Type> resTypes;
  Builder &b = p.getBuilder();

  (void)impl::parseOptionalVisibilityKeyword(p, result.attributes);
  StringAttr nameAttr;
  if (p.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                        result.attributes))
    return failure();

  bool isVariadic = false;
  if (function_interface_impl::parseFunctionSignatureWithArguments(
          p, false, entryArgs, isVariadic, resTypes, resAttrs))
    return failure();
  SmallVector<Type> argTypes;
  for (auto &arg : entryArgs)
    argTypes.push_back(arg.type);
  result.addAttribute(getFunctionTypeAttrName(result.name),
                      TypeAttr::get(b.getFunctionType(argTypes, resTypes)));

  if (succeeded(p.parseOptionalKeyword("lat"))) {
    int64_t latency;
    if (p.parseEqual() || p.parseInteger(latency))
      return failure();
    result.addAttribute(getLatencyAttrName(result.name),
                        b.getI64IntegerAttr(latency));
    if (succeeded(p.parseOptionalKeyword("bound")))
      result.addAttribute(getLatencyBoundAttrName(result.name),
                          b.getUnitAttr());
  }
  if (parseOptionalDeterminacy(p, result, getDeterminacyAttrName(result.name)))
    return failure();

  NamedAttrList parsed;
  if (p.parseOptionalAttrDictWithKeyword(parsed))
    return failure();
  result.attributes.append(parsed);
  call_interface_impl::addArgAndResultAttrs(b, result, entryArgs, resAttrs,
                                            getArgAttrsAttrName(result.name),
                                            getResAttrsAttrName(result.name));

  return p.parseRegion(*result.addRegion(), entryArgs,
                       /*enableNameShadowing=*/false);
}

LogicalResult DCPathOutputOp::verify() {
  auto mod = cast<DCPathModuleOp>((*this)->getParentOp());
  ArrayRef<Type> results = mod.getResultTypes();
  if (results.size() != getNumOperands())
    return emitOpError("has ")
           << getNumOperands() << " operands, but @" << mod.getSymName()
           << " returns " << results.size();
  for (unsigned i = 0, e = results.size(); i != e; ++i)
    if (getOperand(i).getType() != results[i])
      return emitOpError() << "operand " << i << " has type "
                           << getOperand(i).getType() << ", but @"
                           << mod.getSymName() << " result " << i << " is "
                           << results[i];
  return success();
}

void DCPathOperatorOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printSymbolName(getSymName());
  p << getSignature();
  p << " kind=" << getKind();
  p << " latency " << getLatency();
  p << " in_delay ";
  printNum(p, getInDelayAttr().getValueAsDouble());
  p << " out_delay ";
  printNum(p, getOutDelayAttr().getValueAsDouble());
  if (double mp = getMinPeriod().convertToDouble()) {
    p << " min_period ";
    printNum(p, mp);
  }
  if (getPipelined())
    p << " pipelined";
  p << ' ' << stringifyStallContractEnum(getStall());
  if (std::optional<int64_t> fw = getFedWidth())
    p << " fed " << *fw;
  if (ArrayAttr uses = getUsesAttr())
    p << " uses " << uses;
}

ParseResult DCPathOperatorOp::parse(OpAsmParser &p, OperationState &result) {
  Builder &b = p.getBuilder();
  StringAttr symName;
  StringRef kind, stall;
  Type sig;
  int64_t latency;
  double inDelay, outDelay;
  if (p.parseSymbolName(symName, getSymNameAttrName(result.name),
                        result.attributes) ||
      p.parseType(sig) || p.parseKeyword("kind") || p.parseEqual() ||
      p.parseKeyword(&kind) || p.parseKeyword("latency") ||
      p.parseInteger(latency) || p.parseKeyword("in_delay") ||
      parseNum(p, inDelay) || p.parseKeyword("out_delay") ||
      parseNum(p, outDelay))
    return failure();
  auto fnTy = dyn_cast<FunctionType>(sig);
  if (!fnTy)
    return p.emitError(p.getNameLoc(), "expected a function-type signature");
  double minPeriod = 0.0;
  if (succeeded(p.parseOptionalKeyword("min_period")) && parseNum(p, minPeriod))
    return failure();
  bool pipelined = succeeded(p.parseOptionalKeyword("pipelined"));
  if (p.parseKeyword(&stall))
    return failure();
  std::optional<StallContractEnum> s = symbolizeStallContractEnum(stall);
  if (!s)
    return p.emitError(p.getNameLoc(), "unknown stall contract '")
           << stall << "'";
  result.addAttribute(getKindAttrName(result.name), b.getStringAttr(kind));
  result.addAttribute(getSignatureAttrName(result.name), TypeAttr::get(fnTy));
  result.addAttribute(getLatencyAttrName(result.name),
                      b.getI64IntegerAttr(latency));
  result.addAttribute(getInDelayAttrName(result.name),
                      b.getF32FloatAttr(inDelay));
  result.addAttribute(getOutDelayAttrName(result.name),
                      b.getF32FloatAttr(outDelay));
  result.addAttribute(getMinPeriodAttrName(result.name),
                      b.getF32FloatAttr(minPeriod));
  result.addAttribute(getPipelinedAttrName(result.name),
                      b.getBoolAttr(pipelined));
  result.addAttribute(getStallAttrName(result.name),
                      StallContractEnumAttr::get(b.getContext(), *s));
  if (succeeded(p.parseOptionalKeyword("fed"))) {
    int64_t fw;
    if (p.parseInteger(fw))
      return failure();
    result.addAttribute(getFedWidthAttrName(result.name),
                        b.getI64IntegerAttr(fw));
  }
  if (succeeded(p.parseOptionalKeyword("uses"))) {
    ArrayAttr uses;
    if (p.parseAttribute(uses))
      return failure();
    result.addAttribute(getUsesAttrName(result.name), uses);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// dcp.pipeline / dcp.sequential custom assembly
//===----------------------------------------------------------------------===//

void DCPathPipelineOp::print(OpAsmPrinter &p) {
  Block &body = getBody().front();
  int64_t lb = getLb().value_or(0), step = getStep().value_or(1);
  // A runtime bound carries its type: an affine bound is reified at the
  // datapath's index width, an scf one stays `index`.
  auto bound = [&](Value v) { p << v << " : " << v.getType(); };
  p << ' ' << body.getArgument(0) << " = ";
  if (Value l = getLbBound())
    bound(l); // a runtime lower bound (data-dependent range start)
  else
    p << lb;
  p << " to ";
  if (std::optional<int64_t> t = getTrip())
    p << (lb + *t * step); // the derived upper bound (ub = lb + trip*step)
  else if (Value b = getDynamicBound())
    bound(b); // a runtime upper bound (dynamic trip)
  else
    p << '?'; // a while loop (termination by dcp.condition)
  if (Value s = getStepBound()) {
    p << " step ";
    bound(s); // a runtime stride
  } else if (step != 1) {
    p << " step " << step;
  }
  if (IntegerAttr tb = getTripBoundAttr())
    p << " trip_bound=" << tb.getInt();
  if (IntegerAttr ii = getIiAttr())
    p << " ii=" << ii.getInt();
  if (IntegerAttr l = getLengthAttr())
    p << " length=" << l.getInt();
  if (IntegerAttr d = getDrainAttr())
    p << " drain=" << d.getInt();
  if (IntegerAttr lat = getLatencyAttr()) {
    p << " lat=" << lat.getInt();
    if (getLatencyBound())
      p << " bound";
  }
  if (!getInits().empty()) {
    p << " iter_args(";
    for (unsigned i = 0, e = getInits().size(); i < e; ++i) {
      if (i)
        p << ", ";
      p << body.getArgument(i + 1) << " = " << getInits()[i];
    }
    p << ")";
  }
  if (getNumResults()) {
    p << " -> (";
    for (unsigned i = 0, e = getNumResults(); i < e; ++i) {
      if (i)
        p << ", ";
      p << getResult(i).getType();
    }
    p << ")";
  }
  p << ' ';
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
  if (std::optional<DeterminacyEnum> d = getDeterminacy())
    p << ' ' << stringifyDeterminacyEnum(*d);
  p.printOptionalAttrDict(
      (*this)->getAttrs(),
      /*elidedAttrs=*/{
          getTripAttrName(), getTripBoundAttrName(), getLbAttrName(),
          getStepAttrName(), getIiAttrName(), getLengthAttrName(),
          getDrainAttrName(), getLatencyAttrName(), getLatencyBoundAttrName(),
          getDeterminacyAttrName(), getOperandSegmentSizesAttrName()});
}

ParseResult DCPathPipelineOp::parse(OpAsmParser &p, OperationState &result) {
  Builder &b = p.getBuilder();
  OpAsmParser::Argument iv;
  iv.type = b.getIndexType();
  int64_t lb = 0, ii;
  if (p.parseArgument(iv) || p.parseEqual())
    return failure();
  // Lower bound after `=`: an SSA `%operand` (the runtime `lbBound`) or an
  // integer (a compile-time `lb`, default 0). Resolved first so it leads the
  // operand segments in the declared order lbBound, dynamicBound, stepBound.
  bool hasLb = false;
  {
    OpAsmParser::UnresolvedOperand lbOp;
    OptionalParseResult res = p.parseOptionalOperand(lbOp);
    if (res.has_value()) {
      Type ty;
      if (failed(*res) || p.parseColonType(ty) ||
          p.resolveOperand(lbOp, ty, result.operands))
        return failure();
      hasLb = true;
    } else if (p.parseInteger(lb)) {
      return failure();
    }
  }
  if (p.parseKeyword("to"))
    return failure();
  // Termination bound after `to`: `?` for a while loop, an SSA `%operand` for a
  // runtime `dynamicBound`, or an integer `ub` from which `trip` is derived.
  bool hasBound = false, hasUb = false;
  int64_t ub = 0;
  if (succeeded(p.parseOptionalQuestion())) {
    // while loop: leave trip / dynamicBound / lb / step unset
  } else {
    OpAsmParser::UnresolvedOperand boundOp;
    OptionalParseResult res = p.parseOptionalOperand(boundOp);
    if (res.has_value()) {
      Type ty;
      if (failed(*res) || p.parseColonType(ty) ||
          p.resolveOperand(boundOp, ty, result.operands))
        return failure();
      hasBound = true; // resolved first, so it precedes inits in the segments
    } else {
      if (p.parseInteger(ub))
        return failure();
      hasUb = true;
    }
  }
  // Optional `step` (default 1): an SSA `%operand` (a runtime `stepBound`) or
  // an integer. `lb` and `step` become attributes only when compile-time and
  // non-default, so the common `lb=0`/`step=1` form round-trips unchanged.
  int64_t step = 1;
  bool hasStep = false;
  if (succeeded(p.parseOptionalKeyword("step"))) {
    OpAsmParser::UnresolvedOperand stepOp;
    OptionalParseResult res = p.parseOptionalOperand(stepOp);
    if (res.has_value()) {
      Type ty;
      if (failed(*res) || p.parseColonType(ty) ||
          p.resolveOperand(stepOp, ty, result.operands))
        return failure();
      hasStep = true;
    } else if (p.parseInteger(step)) {
      return failure();
    }
  }
  if (!hasLb && lb != 0)
    result.addAttribute(getLbAttrName(result.name), b.getI64IntegerAttr(lb));
  if (!hasStep && step != 1)
    result.addAttribute(getStepAttrName(result.name),
                        b.getI64IntegerAttr(step));
  // `trip` is ceil((ub-lb)/step), derived only when every bound is
  // compile-time.
  if (hasUb && !hasLb && !hasStep)
    result.addAttribute(getTripAttrName(result.name),
                        b.getI64IntegerAttr(std::max<int64_t>(
                            0, llvm::divideCeilSigned(ub - lb, step))));
  if (succeeded(p.parseOptionalKeyword("trip_bound"))) {
    int64_t tripBound;
    if (p.parseEqual() || p.parseInteger(tripBound))
      return failure();
    result.addAttribute(getTripBoundAttrName(result.name),
                        b.getI64IntegerAttr(tripBound));
  }
  // `ii` is optional: absent for a data-dependent sequential wrapper.
  if (succeeded(p.parseOptionalKeyword("ii"))) {
    if (p.parseEqual() || p.parseInteger(ii))
      return failure();
    result.addAttribute(getIiAttrName(result.name), b.getI64IntegerAttr(ii));
  }
  if (succeeded(p.parseOptionalKeyword("length"))) {
    int64_t length;
    if (p.parseEqual() || p.parseInteger(length))
      return failure();
    result.addAttribute(getLengthAttrName(result.name),
                        b.getI64IntegerAttr(length));
  }
  if (succeeded(p.parseOptionalKeyword("drain"))) {
    int64_t drain;
    if (p.parseEqual() || p.parseInteger(drain))
      return failure();
    result.addAttribute(getDrainAttrName(result.name),
                        b.getI64IntegerAttr(drain));
  }
  if (succeeded(p.parseOptionalKeyword("lat"))) {
    int64_t latency;
    if (p.parseEqual() || p.parseInteger(latency))
      return failure();
    result.addAttribute("latency", b.getI64IntegerAttr(latency));
    if (succeeded(p.parseOptionalKeyword("bound")))
      result.addAttribute(getLatencyBoundAttrName(result.name),
                          b.getUnitAttr());
  }

  SmallVector<OpAsmParser::Argument> regionArgs{iv};
  SmallVector<OpAsmParser::UnresolvedOperand> inits;
  if (succeeded(p.parseOptionalKeyword("iter_args"))) {
    SmallVector<OpAsmParser::Argument> iterArgs;
    if (p.parseAssignmentList(iterArgs, inits))
      return failure();
    regionArgs.append(iterArgs.begin(), iterArgs.end());
  }

  SmallVector<Type> resultTypes;
  if (succeeded(p.parseOptionalArrow()))
    if (p.parseLParen() || p.parseTypeList(resultTypes) || p.parseRParen())
      return failure();
  if (resultTypes.size() != inits.size())
    return p.emitError(p.getNameLoc(), "expected one result type per iter-arg");
  result.addTypes(resultTypes);
  for (unsigned i = 0, e = inits.size(); i < e; ++i)
    regionArgs[i + 1].type = resultTypes[i];
  if (p.resolveOperands(inits, resultTypes, p.getCurrentLocation(),
                        result.operands))
    return failure();
  // AttrSizedOperandSegments: the three optional bound operands (lbBound,
  // dynamicBound, stepBound, each 0 or 1) precede the inits, resolved above in
  // that declared order.
  result.addAttribute(
      getOperandSegmentSizesAttrName(result.name),
      b.getDenseI32ArrayAttr({hasLb ? 1 : 0, hasBound ? 1 : 0, hasStep ? 1 : 0,
                              static_cast<int32_t>(inits.size())}));

  Region *region = result.addRegion();
  if (p.parseRegion(*region, regionArgs) ||
      parseOptionalDeterminacy(p, result,
                               getDeterminacyAttrName(result.name)) ||
      p.parseOptionalAttrDict(result.attributes))
    return failure();
  // Default to an unconditional terminator when the body has none; a while
  // pipeline prints its dcp.condition explicitly. Stands in for the
  // SingleBlockImplicitTerminator hook, which this op does not use.
  Block &blk = region->front();
  if (blk.empty() || !blk.back().hasTrait<OpTrait::IsTerminator>()) {
    OpBuilder tb = OpBuilder::atBlockEnd(&blk);
    DCPathUnconditionOp::create(tb, result.location);
  }
  return success();
}

LogicalResult DCPathSequentialOp::verify() {
  if (getLatencyBound() && !getLatencyAttr())
    return emitOpError("latency_bound requires latency");
  return success();
}

void DCPathSequentialOp::print(OpAsmPrinter &p) {
  if (IntegerAttr l = getLengthAttr())
    p << " length=" << l.getInt();
  if (IntegerAttr d = getDrainAttr())
    p << " drain=" << d.getInt();
  if (IntegerAttr lat = getLatencyAttr()) {
    p << " lat=" << lat.getInt();
    if (getLatencyBound())
      p << " bound";
  }
  if (getNumResults()) {
    p << " -> (";
    for (unsigned i = 0, e = getNumResults(); i < e; ++i) {
      if (i)
        p << ", ";
      p << getResult(i).getType();
    }
    p << ")";
  }
  p << ' ';
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
  if (std::optional<DeterminacyEnum> d = getDeterminacy())
    p << ' ' << stringifyDeterminacyEnum(*d);
  p.printOptionalAttrDict(
      (*this)->getAttrs(),
      /*elidedAttrs=*/{getLengthAttrName(), getDrainAttrName(),
                       getLatencyAttrName(), getLatencyBoundAttrName(),
                       getDeterminacyAttrName()});
}

ParseResult DCPathSequentialOp::parse(OpAsmParser &p, OperationState &result) {
  Builder &b = p.getBuilder();
  if (succeeded(p.parseOptionalKeyword("length"))) {
    int64_t length;
    if (p.parseEqual() || p.parseInteger(length))
      return failure();
    result.addAttribute(getLengthAttrName(result.name),
                        b.getI64IntegerAttr(length));
  }
  if (succeeded(p.parseOptionalKeyword("drain"))) {
    int64_t drain;
    if (p.parseEqual() || p.parseInteger(drain))
      return failure();
    result.addAttribute(getDrainAttrName(result.name),
                        b.getI64IntegerAttr(drain));
  }
  if (succeeded(p.parseOptionalKeyword("lat"))) {
    int64_t latency;
    if (p.parseEqual() || p.parseInteger(latency))
      return failure();
    result.addAttribute("latency", b.getI64IntegerAttr(latency));
    if (succeeded(p.parseOptionalKeyword("bound")))
      result.addAttribute(getLatencyBoundAttrName(result.name),
                          b.getUnitAttr());
  }
  SmallVector<Type> resultTypes;
  if (succeeded(p.parseOptionalArrow()))
    if (p.parseLParen() || p.parseTypeList(resultTypes) || p.parseRParen())
      return failure();
  result.addTypes(resultTypes);
  Region *region = result.addRegion();
  if (p.parseRegion(*region) ||
      parseOptionalDeterminacy(p, result,
                               getDeterminacyAttrName(result.name)) ||
      p.parseOptionalAttrDict(result.attributes))
    return failure();
  ensureTerminator(*region, b, result.location);
  return success();
}

//===----------------------------------------------------------------------===//
// dcp.select custom assembly
//===----------------------------------------------------------------------===//

// One branch of a dcp.select must end with a dcp.uncondition yielding one value
// per select result. \p required rejects an empty branch.
static LogicalResult verifySelectBranch(DCPathSelectOp op, Region &r,
                                        bool required, StringRef which) {
  if (r.empty()) {
    if (required)
      return op.emitOpError() << which << " branch must be present";
    return success();
  }
  Block &blk = r.front();
  if (blk.empty() || !blk.back().hasTrait<OpTrait::IsTerminator>())
    return op.emitOpError() << which << " branch must end with a terminator";
  auto term = dyn_cast<DCPathUnconditionOp>(blk.getTerminator());
  if (!term)
    return op.emitOpError() << which << " branch must end with dcp.uncondition";
  if (term.getOperands().size() != op.getNumResults())
    return op.emitOpError()
           << which << " branch must yield one value per select result";
  return success();
}

LogicalResult DCPathSelectOp::verify() {
  if (getLatencyBound() && !getLatencyAttr())
    return emitOpError("latency_bound requires latency");
  if (failed(verifySelectBranch(*this, getThenRegion(), /*required=*/true,
                                "then")))
    return failure();
  // The else branch is required exactly when results are yielded, since the
  // derived result-mux needs a value from both paths.
  return verifySelectBranch(*this, getElseRegion(),
                            /*required=*/getNumResults() > 0, "else");
}

void DCPathSelectOp::print(OpAsmPrinter &p) {
  p << ' ' << getCondition();
  if (IntegerAttr lat = getLatencyAttr()) {
    p << " lat=" << lat.getInt();
    if (getLatencyBound())
      p << " bound";
  }
  if (getNumResults()) {
    p << " -> (";
    for (unsigned i = 0, e = getNumResults(); i < e; ++i) {
      if (i)
        p << ", ";
      p << getResult(i).getType();
    }
    p << ")";
  }
  p << ' ';
  p.printRegion(getThenRegion(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
  if (!getElseRegion().empty()) {
    p << " else ";
    p.printRegion(getElseRegion(), /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
  }
  if (std::optional<DeterminacyEnum> d = getDeterminacy())
    p << ' ' << stringifyDeterminacyEnum(*d);
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{getLatencyAttrName(),
                                           getLatencyBoundAttrName(),
                                           getDeterminacyAttrName()});
}

ParseResult DCPathSelectOp::parse(OpAsmParser &p, OperationState &result) {
  Builder &b = p.getBuilder();
  OpAsmParser::UnresolvedOperand cond;
  if (p.parseOperand(cond))
    return failure();
  if (succeeded(p.parseOptionalKeyword("lat"))) {
    int64_t latency;
    if (p.parseEqual() || p.parseInteger(latency))
      return failure();
    result.addAttribute("latency", b.getI64IntegerAttr(latency));
    if (succeeded(p.parseOptionalKeyword("bound")))
      result.addAttribute(getLatencyBoundAttrName(result.name),
                          b.getUnitAttr());
  }
  SmallVector<Type> resultTypes;
  if (succeeded(p.parseOptionalArrow()))
    if (p.parseLParen() || p.parseTypeList(resultTypes) || p.parseRParen())
      return failure();
  if (p.resolveOperand(cond, b.getI1Type(), result.operands))
    return failure();
  result.addTypes(resultTypes);
  Region *thenRegion = result.addRegion();
  Region *elseRegion = result.addRegion();
  if (p.parseRegion(*thenRegion))
    return failure();
  if (succeeded(p.parseOptionalKeyword("else")))
    if (p.parseRegion(*elseRegion))
      return failure();
  if (parseOptionalDeterminacy(p, result, getDeterminacyAttrName(result.name)))
    return failure();
  return p.parseOptionalAttrDict(result.attributes);
}

// Hold `dcp.instance`'s local copy of the callee's `latency` and `determinacy`
// to what the callee publishes. After emit the callee is an `hw.module`, which
// publishes no contract, so only the symbol must resolve.
LogicalResult
DCPathInstanceOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  Operation *sym = symbolTable.lookupNearestSymbolFrom(*this, getCalleeAttr());
  if (!sym)
    return emitOpError("references unknown callee '") << getCallee() << "'";
  auto callee = dyn_cast<DCPathModuleOp>(sym);
  if (!callee)
    return success();

  // Operands are the callee's arguments in order.
  FunctionType sig = callee.getFunctionType();
  if (sig.getNumInputs() != getInputs().size())
    return emitOpError("passes ")
           << getInputs().size() << " operand(s) to @" << callee.getSymName()
           << ", which takes " << sig.getNumInputs();
  for (auto [i, t] : llvm::enumerate(sig.getInputs()))
    if (getInputs()[i].getType() != t)
      return emitOpError("operand ")
             << i << " has type " << getInputs()[i].getType() << ", but @"
             << callee.getSymName() << " takes " << t;
  if (sig.getNumResults() != getResults().size())
    return emitOpError("takes ")
           << getResults().size() << " result(s) from @" << callee.getSymName()
           << ", which returns " << sig.getNumResults();
  for (auto [i, t] : llvm::enumerate(sig.getResults()))
    if (getResults()[i].getType() != t)
      return emitOpError("result ")
             << i << " has type " << getResults()[i].getType() << ", but @"
             << callee.getSymName() << " returns " << t;

  if (getLatency() != callee.getLatency())
    return emitOpError("declares latency ")
           << (getLatency() ? std::to_string(*getLatency()) : "none")
           << ", but @" << callee.getSymName() << " publishes "
           << (callee.getLatency() ? std::to_string(*callee.getLatency())
                                   : "none");
  if (getDeterminacy() != callee.getDeterminacy())
    return emitOpError("declares determinacy ")
           << stringifyDeterminacyEnum(getDeterminacy()) << ", but @"
           << callee.getSymName() << " publishes "
           << stringifyDeterminacyEnum(callee.getDeterminacy());
  return success();
}

LogicalResult DCPathInstanceOp::verify() {
  if (getStart() < 0)
    return emitOpError("start cycle must be non-negative");
  return success();
}

LogicalResult DCPathPipelineOp::verify() {
  // `ii` is optional (absent for a data-dependent sequential wrapper); when
  // present it must be a positive initiation interval.
  if (std::optional<int64_t> ii = getIi(); ii && *ii < 1)
    return emitOpError("ii must be >= 1");
  if (std::optional<int64_t> s = getStep(); s && *s <= 0)
    return emitOpError("step must be > 0"); // termination is iv+step >= ub
  // A bound is either compile-time (attribute) or runtime (operand), never
  // both.
  if (getLbBound() && getLbAttr())
    return emitOpError("lb given as both an operand and an attribute");
  if (getStepBound() && getStepAttr())
    return emitOpError("step given as both an operand and an attribute");
  if (getLatencyBound() && !getLatencyAttr())
    return emitOpError("latency_bound requires latency");
  if (getTripBoundAttr() && getTripAttr())
    return emitOpError("trip_bound is the worst case of a trip that is not "
                       "compile-time; it cannot accompany an exact trip");
  Block &body = getBody().front();
  if (body.getNumArguments() != 1 + getInits().size())
    return emitOpError(
        "body must have one induction argument plus one argument "
        "per iter-arg");
  if (!body.getArgument(0).getType().isIndex())
    return emitOpError(
        "the first body argument (induction variable) must have index type");

  // The terminator determines the loop kind: dcp.uncondition (counted) or
  // dcp.condition (while). Either carries one value per iter-arg.
  if (body.empty() || !body.back().hasTrait<OpTrait::IsTerminator>())
    return emitOpError("body must end with a terminator");
  Operation *term = body.getTerminator();
  if (auto cond = dyn_cast<DCPathConditionOp>(term)) {
    if (getTripAttr())
      return emitOpError(
          "a while pipeline (dcp.condition terminator) must not have a trip");
    if (cond.getCarried().size() != getInits().size())
      return emitOpError("dcp.condition must carry one value per iter-arg");
  } else if (auto y = dyn_cast<DCPathUnconditionOp>(term)) {
    if (y.getOperands().size() != getInits().size())
      return emitOpError("dcp.uncondition must yield one value per iter-arg");
    // A counted loop terminates on its bound, so it must carry one: the `trip`
    // attribute or the runtime `dynamicBound`.
    if (!getTripAttr() && !getDynamicBound())
      return emitOpError("a counted pipeline (dcp.uncondition terminator) "
                         "needs a trip attribute or a dynamicBound operand");
  } else {
    return emitOpError("body must end with dcp.uncondition or dcp.condition");
  }
  return success();
}

bool DCPathPipelineOp::isWhileLoop() {
  return isa<DCPathConditionOp>(getBody().front().getTerminator());
}

DCPathConditionOp DCPathPipelineOp::getConditionOp() {
  return dyn_cast<DCPathConditionOp>(getBody().front().getTerminator());
}

DCPathUnconditionOp DCPathPipelineOp::getUnconditionOp() {
  return dyn_cast<DCPathUnconditionOp>(getBody().front().getTerminator());
}

Value DCPathPipelineOp::getConditionValue() {
  DCPathConditionOp c = getConditionOp();
  return c ? c.getCondition() : Value();
}

OperandRange DCPathPipelineOp::getCarriedValues() {
  if (DCPathConditionOp c = getConditionOp())
    return c.getCarried();
  return getUnconditionOp().getOperands();
}

} // namespace mlir::allo::dcp

//===----------------------------------------------------------------------===//
// Resource cost attributes
//===----------------------------------------------------------------------===//

// How many coefficients each form carries. A table is [p0, v0, p1, v1, ...],
// so it is even and non-empty rather than a fixed count. Only `piecewise`
// carries arms.
LogicalResult
CostAttr::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                 CostFormEnum form, DenseF64ArrayAttr coeffsAttr,
                 llvm::ArrayRef<CostAttr> arms) {
  if (!coeffsAttr)
    return emitError() << "a cost needs its coefficients";
  llvm::ArrayRef<double> coeffs = coeffsAttr.asArrayRef();
  auto need = [&](size_t n) -> LogicalResult {
    if (coeffs.size() == n)
      return success();
    return emitError() << stringifyCostFormEnum(form) << " takes " << n
                       << " coefficient(s), got " << coeffs.size();
  };
  size_t wantArms = form == CostFormEnum::Piecewise ? 2 : 0;
  if (arms.size() != wantArms)
    return emitError() << stringifyCostFormEnum(form) << " takes " << wantArms
                       << " arm(s), got " << arms.size();
  switch (form) {
  case CostFormEnum::Const:
  case CostFormEnum::Quadratic:
    return need(1);
  case CostFormEnum::Tiled:
    if (failed(need(1)))
      return failure();
    // The tile is a divisor, so a non-positive one is a division by zero
    // dressed as a declaration.
    if (coeffs[0] <= 0.0)
      return emitError() << "tiled needs a positive number of bits per tile";
    return success();
  case CostFormEnum::Linear:
    return need(2);
  case CostFormEnum::Step:
    return need(3);
  case CostFormEnum::Piecewise:
    // The one coefficient is the breakpoint; the arms carry the shapes.
    return need(1);
  case CostFormEnum::Table:
  case CostFormEnum::Interp:
    if (coeffs.empty() || coeffs.size() % 2 != 0)
      return emitError() << stringifyCostFormEnum(form)
                         << " takes [point, value] pairs, got " << coeffs.size()
                         << " coefficient(s)";
    for (size_t i = 2; i < coeffs.size(); i += 2)
      if (coeffs[i] <= coeffs[i - 2])
        return emitError() << stringifyCostFormEnum(form)
                           << " points must ascend";
    return success();
  }
  llvm_unreachable("unhandled CostFormEnum");
}

CostAttr CostAttr::unmeasuredAt(int64_t param) const {
  llvm::ArrayRef<double> c = getCoeffs().asArrayRef();
  double p = static_cast<double>(param);
  switch (getForm()) {
  case CostFormEnum::Table:
  case CostFormEnum::Interp:
    // Only above the last point: below the first, the narrowest measurement
    // over-states rather than guesses.
    return p > c[c.size() - 2] ? *this : CostAttr();
  case CostFormEnum::Piecewise:
    return getArms()[p < c[0] ? 0 : 1].unmeasuredAt(param);
  default:
    // Every other form is a shape, and a shape holds at every parameter.
    return CostAttr();
  }
}

std::pair<int64_t, int64_t> CostAttr::measuredDomain() const {
  assert(
      (getForm() == CostFormEnum::Table || getForm() == CostFormEnum::Interp) &&
      "only measured points carry a domain");
  llvm::ArrayRef<double> c = getCoeffs().asArrayRef();
  return {static_cast<int64_t>(c[0]), static_cast<int64_t>(c[c.size() - 2])};
}

std::optional<double> CostAttr::evaluate(int64_t param) const {
  llvm::ArrayRef<double> c = getCoeffs().asArrayRef();
  double p = static_cast<double>(param);
  switch (getForm()) {
  case CostFormEnum::Const:
    return c[0];
  case CostFormEnum::Linear:
    return c[0] + c[1] * p;
  case CostFormEnum::Quadratic:
    return c[0] * p * p;
  case CostFormEnum::Step:
    return p < c[0] ? c[1] * p : c[2];
  case CostFormEnum::Table: {
    // The table is ascending, so the last point at or under `p` is the row.
    if (unmeasuredAt(param))
      return std::nullopt;
    double v = c[1];
    for (size_t i = 0; i < c.size(); i += 2)
      if (c[i] <= p)
        v = c[i + 1];
    return v;
  }
  case CostFormEnum::Interp:
    if (unmeasuredAt(param))
      return std::nullopt;
    if (p <= c[0])
      return c[1]; // the first measurement stands below itself
    for (size_t i = 2; i < c.size(); i += 2)
      if (p <= c[i])
        return c[i - 1] +
               (p - c[i - 2]) / (c[i] - c[i - 2]) * (c[i + 1] - c[i - 1]);
    return c[1]; // a single measured point, read at that point
  case CostFormEnum::Tiled:
    // One parameter's worth of tiles. The whole-tuple reading, where the
    // product sits inside the ceiling, is `evaluateResourceUse`'s.
    return std::ceil(p / c[0]);
  case CostFormEnum::Piecewise:
    return getArms()[p < c[0] ? 0 : 1].evaluate(param);
  }
  llvm_unreachable("unhandled CostFormEnum");
}

LogicalResult
ResourceUseAttr::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                        SymbolRefAttr resource,
                        llvm::ArrayRef<CostAttr> factors) {
  if (factors.empty())
    return emitError() << "a resource use needs one cost per parameter";
  return success();
}

CostAttr mlir::allo::unmeasuredUse(ArrayAttr uses,
                                   llvm::ArrayRef<int64_t> params) {
  if (uses)
    for (Attribute use : uses) {
      llvm::ArrayRef<CostAttr> factors =
          cast<ResourceUseAttr>(use).getFactors();
      if (factors.size() != params.size())
        continue; // a lone `tiled`, which is a shape over the whole tuple
      for (auto [factor, param] : llvm::zip(factors, params))
        if (CostAttr bad = factor.unmeasuredAt(param))
          return bad;
    }
  return CostAttr();
}

std::optional<llvm::SmallVector<std::pair<SymbolRefAttr, int64_t>>>
mlir::allo::evaluateResourceUse(ArrayAttr uses,
                                llvm::ArrayRef<int64_t> params) {
  // One running total per resource, in the order the resources first appear.
  llvm::SmallVector<std::pair<SymbolRefAttr, double>> totals;
  if (uses)
    for (Attribute use : uses) {
      auto ru = cast<ResourceUseAttr>(use);
      llvm::ArrayRef<CostAttr> factors = ru.getFactors();
      double term = 1.0;
      if (factors.size() == 1 &&
          factors.front().getForm() == CostFormEnum::Tiled) {
        // A lone `tiled` reads the whole tuple, so the product sits inside the
        // ceiling. One among a full set of factors tiles its own parameter.
        assert(!params.empty() &&
               "a tiled cost needs a parameter tuple to tile");
        double bits = 1.0;
        for (int64_t p : params)
          bits *= static_cast<double>(p);
        term = std::ceil(bits / factors.front().getCoeffs().asArrayRef()[0]);
      } else {
        assert(factors.size() == params.size() &&
               "a resource cost carries one factor per parameter of its kind");
        for (auto [factor, param] : llvm::zip(factors, params)) {
          std::optional<double> v = factor.evaluate(param);
          if (!v)
            return std::nullopt;
          term *= *v;
        }
      }
      auto *it = llvm::find_if(
          totals, [&](auto &total) { return total.first == ru.getResource(); });
      if (it == totals.end())
        totals.emplace_back(ru.getResource(), term);
      else
        it->second += term;
    }
  // Rounded ONCE per resource, after every term is in: rounding a factor or a
  // term would make the answer depend on how the cost happened to be written.
  llvm::SmallVector<std::pair<SymbolRefAttr, int64_t>> spent;
  spent.reserve(totals.size());
  for (auto &[resource, amount] : totals)
    spent.emplace_back(resource, static_cast<int64_t>(std::llround(amount)));
  return spent;
}
