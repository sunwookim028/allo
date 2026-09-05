/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * C API for the Allo dialect's custom attributes, used by the Python bindings.
 */

#ifndef ALLO_C_ALLOATTRS_H
#define ALLO_C_ALLOATTRS_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// PartitionAxisAttr  (#allo.part_axis(dim, kind, factor))
//
// `kind` mirrors `allo::PartitionKindEnum`: 0 = Complete, 1 = Block,
// 2 = Cyclic, 3 = Skew.
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool alloAttributeIsAPartitionAxis(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute alloPartitionAxisAttrGet(MlirContext ctx,
                                                          uint32_t kind,
                                                          int64_t factor,
                                                          int64_t dim);

MLIR_CAPI_EXPORTED uint32_t alloPartitionAxisAttrGetKind(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t alloPartitionAxisAttrGetFactor(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t alloPartitionAxisAttrGetDim(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirTypeID alloPartitionAxisAttrGetTypeID(void);

//===----------------------------------------------------------------------===//
// PartitionAttr  (#allo.partition<[ axes... ]>)
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool alloAttributeIsAPartition(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute alloPartitionAttrGet(
    MlirContext ctx, intptr_t nAxes, MlirAttribute const *axes);

MLIR_CAPI_EXPORTED intptr_t alloPartitionAttrGetNumAxes(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute alloPartitionAttrGetAxis(MlirAttribute attr,
                                                          intptr_t pos);
MLIR_CAPI_EXPORTED MlirTypeID alloPartitionAttrGetTypeID(void);

//===----------------------------------------------------------------------===//
// Enum-backed attributes. Every one gets the same five accessors.
//
// `Get` takes the underlying I32 enum case, which `GetValue` round-trips.
// `GetByName` takes the case's mnemonic instead, the spelling that appears in
// the assembly (`mul` of `#allo<op_kind mul>`), and returns a null attribute
// for a name the enum does not have.
//===----------------------------------------------------------------------===//

#define ALLO_ENUM_ATTR_CAPI_DECL(Name)                                         \
  MLIR_CAPI_EXPORTED bool alloAttributeIsA##Name(MlirAttribute attr);          \
  MLIR_CAPI_EXPORTED MlirAttribute allo##Name##AttrGet(MlirContext ctx,        \
                                                       uint32_t value);        \
  MLIR_CAPI_EXPORTED MlirAttribute allo##Name##AttrGetByName(                  \
      MlirContext ctx, MlirStringRef name);                                    \
  MLIR_CAPI_EXPORTED uint32_t allo##Name##AttrGetValue(MlirAttribute attr);    \
  MLIR_CAPI_EXPORTED MlirTypeID allo##Name##AttrGetTypeID(void);

// #allo<dep_type inter|intra>
ALLO_ENUM_ATTR_CAPI_DECL(AssumeDepType)
// #allo<dep_dir raw|war|waw>
ALLO_ENUM_ATTR_CAPI_DECL(AssumeDepDir)
// #allo<mem_kind ram|rom>
ALLO_ENUM_ATTR_CAPI_DECL(MemoryKind)
// #allo<determinacy counted_static|conditional|indeterminate|concurrent>
ALLO_ENUM_ATTR_CAPI_DECL(Determinacy)
// #allo<op_kind add|sub|mul|...>, the abstract operator vocabulary
ALLO_ENUM_ATTR_CAPI_DECL(OpKind)
// #allo<comb_kind addi|subi|muli|...>, the comb op a dcp.compute realizes
ALLO_ENUM_ATTR_CAPI_DECL(CombOpKind)
// #allo<stall ce|free|elastic>
ALLO_ENUM_ATTR_CAPI_DECL(StallContract)
// #allo<cost_form const|linear|...>, the shapes CostAttr below takes
ALLO_ENUM_ATTR_CAPI_DECL(CostForm)

#undef ALLO_ENUM_ATTR_CAPI_DECL

//===----------------------------------------------------------------------===//
// CostAttr  (#allo.cost<form, [coeffs], [arms]>)
//
// `form` is an `allo::CostFormEnum` case. `arms` holds the two arms of a
// `piecewise` and is empty for every other form.
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool alloAttributeIsACost(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute alloCostAttrGet(MlirContext ctx, uint32_t form,
                                                 intptr_t nCoeffs,
                                                 const double *coeffs,
                                                 intptr_t nArms,
                                                 MlirAttribute const *arms);

MLIR_CAPI_EXPORTED uint32_t alloCostAttrGetForm(MlirAttribute attr);
MLIR_CAPI_EXPORTED intptr_t alloCostAttrGetNumCoeffs(MlirAttribute attr);
MLIR_CAPI_EXPORTED double alloCostAttrGetCoeff(MlirAttribute attr,
                                               intptr_t pos);
MLIR_CAPI_EXPORTED intptr_t alloCostAttrGetNumArms(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute alloCostAttrGetArm(MlirAttribute attr,
                                                    intptr_t pos);
MLIR_CAPI_EXPORTED MlirTypeID alloCostAttrGetTypeID(void);

/// The `table` or `interp` inside `attr` whose measured points do not cover
/// `param`, and a null attribute wherever `alloEvaluateCost` answers.
MLIR_CAPI_EXPORTED MlirAttribute alloCostAttrUnmeasuredAt(MlirAttribute attr,
                                                          int64_t param);

/// The first and last point a `table` or an `interp` was measured at. False,
/// leaving both alone, for every other form.
MLIR_CAPI_EXPORTED bool alloCostAttrGetMeasuredDomain(MlirAttribute attr,
                                                      int64_t *first,
                                                      int64_t *last);

//===----------------------------------------------------------------------===//
// ResourceUseAttr  (#allo.res_use<@resource, [factors]>)
//
// One term of what a realization spends of one resource: `factors` holds one
// `#allo.cost` per parameter of the realization's kind, and the term is their
// product. `resource` is a `SymbolRefAttr` naming a `dcp.resource`.
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool alloAttributeIsAResourceUse(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute
alloResourceUseAttrGet(MlirContext ctx, MlirAttribute resource,
                       intptr_t nFactors, MlirAttribute const *factors);

MLIR_CAPI_EXPORTED MlirAttribute
alloResourceUseAttrGetResource(MlirAttribute attr);
MLIR_CAPI_EXPORTED intptr_t
alloResourceUseAttrGetNumFactors(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
alloResourceUseAttrGetFactor(MlirAttribute attr, intptr_t pos);
MLIR_CAPI_EXPORTED MlirTypeID alloResourceUseAttrGetTypeID(void);

//===----------------------------------------------------------------------===//
// Resource cost evaluation.
//
// The one evaluator, reachable from Python. A second implementation of
// `CostAttr::evaluate` is the state the resource model exists to end, so the
// scorer in `benchmark/area.py` prices through this rather than through its own
// copy of the shapes.
//===----------------------------------------------------------------------===//

/// Called once per resource `uses` spends, with the resource's LEAF name (the
/// `lut` of `@u55c::@lut`) and how many of it.
typedef void (*AlloResourceUseCallback)(MlirStringRef resource, int64_t amount,
                                        void *userData);

/// Evaluates `uses`, an `#allo.res_use` array, at the `nParams` parameters of
/// its realization's kind (an operand width; a multiplexer's fan-in and width;
/// a chain's or a storage's depth and width). A null `uses` spends nothing.
/// False, with no call made, when a cost was not measured at its parameter.
MLIR_CAPI_EXPORTED bool
alloEvaluateResourceUse(MlirAttribute uses, intptr_t nParams,
                        const int64_t *params, AlloResourceUseCallback callback,
                        void *userData);

/// Evaluates one `#allo.cost` at `param` into `*value`, unrounded, unlike the
/// array evaluator above. False, leaving `*value` alone, when `param` falls
/// outside the cost's measured points.
MLIR_CAPI_EXPORTED bool alloEvaluateCost(MlirAttribute cost, int64_t param,
                                         double *value);

#ifdef __cplusplus
}
#endif

#endif // ALLO_C_ALLOATTRS_H
