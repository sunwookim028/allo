/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo-c/AlloAttrs.h"

#include "allo/IR/AlloAttrs.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

using namespace mlir;

//===----------------------------------------------------------------------===//
// PartitionAxisAttr
//===----------------------------------------------------------------------===//

bool alloAttributeIsAPartitionAxis(MlirAttribute attr) {
  return isa<allo::PartitionAxisAttr>(unwrap(attr));
}

MlirAttribute alloPartitionAxisAttrGet(MlirContext ctx, uint32_t kind,
                                       int64_t factor, int64_t dim) {
  return wrap(allo::PartitionAxisAttr::get(
      unwrap(ctx), static_cast<allo::PartitionKindEnum>(kind), factor, dim));
}

uint32_t alloPartitionAxisAttrGetKind(MlirAttribute attr) {
  return static_cast<uint32_t>(
      cast<allo::PartitionAxisAttr>(unwrap(attr)).getKind());
}

int64_t alloPartitionAxisAttrGetFactor(MlirAttribute attr) {
  return cast<allo::PartitionAxisAttr>(unwrap(attr)).getFactor();
}

int64_t alloPartitionAxisAttrGetDim(MlirAttribute attr) {
  return cast<allo::PartitionAxisAttr>(unwrap(attr)).getDim();
}

MlirTypeID alloPartitionAxisAttrGetTypeID(void) {
  return wrap(allo::PartitionAxisAttr::getTypeID());
}

//===----------------------------------------------------------------------===//
// PartitionAttr
//===----------------------------------------------------------------------===//

bool alloAttributeIsAPartition(MlirAttribute attr) {
  return isa<allo::PartitionAttr>(unwrap(attr));
}

MlirAttribute alloPartitionAttrGet(MlirContext ctx, intptr_t nAxes,
                                   MlirAttribute const *axes) {
  llvm::SmallVector<allo::PartitionAxisAttr> partitions;
  partitions.reserve(nAxes);
  for (intptr_t i = 0; i < nAxes; ++i)
    partitions.push_back(cast<allo::PartitionAxisAttr>(unwrap(axes[i])));
  return wrap(allo::PartitionAttr::get(unwrap(ctx), partitions));
}

intptr_t alloPartitionAttrGetNumAxes(MlirAttribute attr) {
  return static_cast<intptr_t>(
      cast<allo::PartitionAttr>(unwrap(attr)).getPartitions().size());
}

MlirAttribute alloPartitionAttrGetAxis(MlirAttribute attr, intptr_t pos) {
  return wrap(cast<allo::PartitionAttr>(unwrap(attr)).getPartitions()[pos]);
}

MlirTypeID alloPartitionAttrGetTypeID(void) {
  return wrap(allo::PartitionAttr::getTypeID());
}

//===----------------------------------------------------------------------===//
// Enum-backed attributes: generate the five accessors isa / get(value) /
// getByName / getValue / getTypeID from (CApiName, C++ attr, C++ enum).
//===----------------------------------------------------------------------===//

#define ALLO_ENUM_ATTR_CAPI(CApiName, CppAttr, CppEnum)                        \
  bool alloAttributeIsA##CApiName(MlirAttribute attr) {                        \
    return isa<allo::CppAttr>(unwrap(attr));                                   \
  }                                                                            \
  MlirAttribute allo##CApiName##AttrGet(MlirContext ctx, uint32_t value) {     \
    return wrap(                                                               \
        allo::CppAttr::get(unwrap(ctx), static_cast<allo::CppEnum>(value)));   \
  }                                                                            \
  MlirAttribute allo##CApiName##AttrGetByName(MlirContext ctx,                 \
                                              MlirStringRef name) {            \
    std::optional<allo::CppEnum> value =                                       \
        allo::symbolize##CppEnum(unwrap(name));                                \
    return wrap(value ? allo::CppAttr::get(unwrap(ctx), *value)                \
                      : Attribute());                                          \
  }                                                                            \
  uint32_t allo##CApiName##AttrGetValue(MlirAttribute attr) {                  \
    return static_cast<uint32_t>(                                              \
        cast<allo::CppAttr>(unwrap(attr)).getValue());                         \
  }                                                                            \
  MlirTypeID allo##CApiName##AttrGetTypeID(void) {                             \
    return wrap(allo::CppAttr::getTypeID());                                   \
  }

ALLO_ENUM_ATTR_CAPI(AssumeDepType, AssumeDepTypeEnumAttr, AssumeDepTypeEnum)
ALLO_ENUM_ATTR_CAPI(AssumeDepDir, AssumeDepDirEnumAttr, AssumeDepDirEnum)
ALLO_ENUM_ATTR_CAPI(MemoryKind, MemoryKindEnumAttr, MemoryKindEnum)
ALLO_ENUM_ATTR_CAPI(Determinacy, DeterminacyEnumAttr, DeterminacyEnum)
ALLO_ENUM_ATTR_CAPI(OpKind, OpKindEnumAttr, OpKindEnum)
ALLO_ENUM_ATTR_CAPI(CombOpKind, CombOpKindEnumAttr, CombOpKindEnum)
ALLO_ENUM_ATTR_CAPI(StallContract, StallContractEnumAttr, StallContractEnum)
ALLO_ENUM_ATTR_CAPI(CostForm, CostFormEnumAttr, CostFormEnum)

#undef ALLO_ENUM_ATTR_CAPI

//===----------------------------------------------------------------------===//
// CostAttr
//===----------------------------------------------------------------------===//

bool alloAttributeIsACost(MlirAttribute attr) {
  return isa<allo::CostAttr>(unwrap(attr));
}

MlirAttribute alloCostAttrGet(MlirContext ctx, uint32_t form, intptr_t nCoeffs,
                              const double *coeffs, intptr_t nArms,
                              MlirAttribute const *arms) {
  llvm::SmallVector<allo::CostAttr> armAttrs;
  armAttrs.reserve(nArms);
  for (intptr_t i = 0; i < nArms; ++i)
    armAttrs.push_back(cast<allo::CostAttr>(unwrap(arms[i])));
  return wrap(allo::CostAttr::get(
      unwrap(ctx), static_cast<allo::CostFormEnum>(form),
      DenseF64ArrayAttr::get(
          unwrap(ctx),
          llvm::ArrayRef<double>(coeffs, static_cast<size_t>(nCoeffs))),
      armAttrs));
}

uint32_t alloCostAttrGetForm(MlirAttribute attr) {
  return static_cast<uint32_t>(cast<allo::CostAttr>(unwrap(attr)).getForm());
}

intptr_t alloCostAttrGetNumCoeffs(MlirAttribute attr) {
  return static_cast<intptr_t>(
      cast<allo::CostAttr>(unwrap(attr)).getCoeffs().size());
}

double alloCostAttrGetCoeff(MlirAttribute attr, intptr_t pos) {
  return cast<allo::CostAttr>(unwrap(attr)).getCoeffs()[pos];
}

intptr_t alloCostAttrGetNumArms(MlirAttribute attr) {
  return static_cast<intptr_t>(
      cast<allo::CostAttr>(unwrap(attr)).getArms().size());
}

MlirAttribute alloCostAttrGetArm(MlirAttribute attr, intptr_t pos) {
  return wrap(cast<allo::CostAttr>(unwrap(attr)).getArms()[pos]);
}

MlirTypeID alloCostAttrGetTypeID(void) {
  return wrap(allo::CostAttr::getTypeID());
}

MlirAttribute alloCostAttrUnmeasuredAt(MlirAttribute attr, int64_t param) {
  return wrap(cast<allo::CostAttr>(unwrap(attr)).unmeasuredAt(param));
}

bool alloCostAttrGetMeasuredDomain(MlirAttribute attr, int64_t *first,
                                   int64_t *last) {
  auto cost = cast<allo::CostAttr>(unwrap(attr));
  if (cost.getForm() != allo::CostFormEnum::Table &&
      cost.getForm() != allo::CostFormEnum::Interp)
    return false;
  auto [lo, hi] = cost.measuredDomain();
  *first = lo;
  *last = hi;
  return true;
}

//===----------------------------------------------------------------------===//
// ResourceUseAttr
//===----------------------------------------------------------------------===//

bool alloAttributeIsAResourceUse(MlirAttribute attr) {
  return isa<allo::ResourceUseAttr>(unwrap(attr));
}

MlirAttribute alloResourceUseAttrGet(MlirContext ctx, MlirAttribute resource,
                                     intptr_t nFactors,
                                     MlirAttribute const *factors) {
  llvm::SmallVector<allo::CostAttr> factorAttrs;
  factorAttrs.reserve(nFactors);
  for (intptr_t i = 0; i < nFactors; ++i)
    factorAttrs.push_back(cast<allo::CostAttr>(unwrap(factors[i])));
  return wrap(allo::ResourceUseAttr::get(
      unwrap(ctx), cast<SymbolRefAttr>(unwrap(resource)), factorAttrs));
}

MlirAttribute alloResourceUseAttrGetResource(MlirAttribute attr) {
  return wrap(cast<allo::ResourceUseAttr>(unwrap(attr)).getResource());
}

intptr_t alloResourceUseAttrGetNumFactors(MlirAttribute attr) {
  return static_cast<intptr_t>(
      cast<allo::ResourceUseAttr>(unwrap(attr)).getFactors().size());
}

MlirAttribute alloResourceUseAttrGetFactor(MlirAttribute attr, intptr_t pos) {
  return wrap(cast<allo::ResourceUseAttr>(unwrap(attr)).getFactors()[pos]);
}

MlirTypeID alloResourceUseAttrGetTypeID(void) {
  return wrap(allo::ResourceUseAttr::getTypeID());
}

//===----------------------------------------------------------------------===//
// Resource cost evaluation
//===----------------------------------------------------------------------===//

bool alloEvaluateResourceUse(MlirAttribute uses, intptr_t nParams,
                             const int64_t *params,
                             AlloResourceUseCallback callback, void *userData) {
  auto spent = allo::evaluateResourceUse(
      dyn_cast_or_null<ArrayAttr>(unwrap(uses)),
      llvm::ArrayRef<int64_t>(params, static_cast<size_t>(nParams)));
  if (!spent)
    return false;
  for (auto [resource, amount] : *spent)
    callback(wrap(resource.getLeafReference().getValue()), amount, userData);
  return true;
}

bool alloEvaluateCost(MlirAttribute cost, int64_t param, double *value) {
  auto attr = dyn_cast_or_null<allo::CostAttr>(unwrap(cost));
  std::optional<double> v = attr ? attr.evaluate(param) : 0.0;
  if (!v)
    return false;
  *value = *v;
  return true;
}
