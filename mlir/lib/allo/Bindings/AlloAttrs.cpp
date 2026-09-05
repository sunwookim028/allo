/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * CRTP nanobind subclasses of `allo._mlir.ir.Attribute` for the Allo dialect's
 * custom attributes (see mlir/examples/standalone for the upstream pattern),
 * funnelled through the Allo CAPI.
 */

#include "AlloBindings.h"

#include "allo-c/AlloAttrs.h"

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/Bindings/Python/IRCore.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

#include "nanobind/stl/optional.h"
#include "nanobind/stl/pair.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/vector.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;
namespace mpx = mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN;

namespace {

/// PartitionAxisAttr: #allo.part_axis(dim, kind, factor)
///   kind: 0 = Complete, 1 = Block, 2 = Cyclic.
struct PyPartitionAxisAttr : mpx::PyConcreteAttribute<PyPartitionAxisAttr> {
  static constexpr IsAFunctionTy isaFunction = alloAttributeIsAPartitionAxis;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      alloPartitionAxisAttrGetTypeID;
  static constexpr const char *pyClassName = "PartitionAxisAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](uint32_t kind, int64_t factor, int64_t dim,
           mpx::DefaultingPyMlirContext ctx) {
          return PyPartitionAxisAttr(
              ctx->getRef(),
              alloPartitionAxisAttrGet(ctx.get()->get(), kind, factor, dim));
        },
        nb::arg("kind"), nb::arg("factor"), nb::arg("dim"),
        nb::arg("context").none() = nb::none());
    c.def_prop_ro("kind", [](PyPartitionAxisAttr &self) {
      return alloPartitionAxisAttrGetKind(self);
    });
    c.def_prop_ro("factor", [](PyPartitionAxisAttr &self) {
      return alloPartitionAxisAttrGetFactor(self);
    });
    c.def_prop_ro("dim", [](PyPartitionAxisAttr &self) {
      return alloPartitionAxisAttrGetDim(self);
    });
  }
};

/// PartitionAttr: #allo.partition<[ axes... ]>
struct PyPartitionAttr : mpx::PyConcreteAttribute<PyPartitionAttr> {
  static constexpr IsAFunctionTy isaFunction = alloAttributeIsAPartition;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      alloPartitionAttrGetTypeID;
  static constexpr const char *pyClassName = "PartitionAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](std::vector<MlirAttribute> axes, mpx::DefaultingPyMlirContext ctx) {
          return PyPartitionAttr(
              ctx->getRef(),
              alloPartitionAttrGet(ctx.get()->get(),
                                   static_cast<intptr_t>(axes.size()),
                                   axes.data()));
        },
        nb::arg("axes"), nb::arg("context").none() = nb::none());
    c.def_prop_ro("num_axes", [](PyPartitionAttr &self) {
      return alloPartitionAttrGetNumAxes(self);
    });
    c.def(
        "axis",
        [](PyPartitionAttr &self, intptr_t pos) {
          return alloPartitionAttrGetAxis(self, pos);
        },
        nb::arg("pos"));
  }
};

/// Enum-backed attributes all share the same `get` / `value` shape, so
/// generate a CRTP subclass per attr from its CAPI hooks. `get` takes either
/// the enum case or its mnemonic.
#define ALLO_ENUM_ATTR(PyClass, PyName, IsAFn, GetFn, GetByNameFn, GetValueFn, \
                       GetIdFn)                                                \
  struct PyClass : mpx::PyConcreteAttribute<PyClass> {                         \
    static constexpr IsAFunctionTy isaFunction = IsAFn;                        \
    static constexpr GetTypeIDFunctionTy getTypeIdFunction = GetIdFn;          \
    static constexpr const char *pyClassName = PyName;                         \
    using PyConcreteAttribute::PyConcreteAttribute;                            \
    static void bindDerived(ClassTy &c) {                                      \
      c.def_static(                                                            \
          "get",                                                               \
          [](uint32_t value, mpx::DefaultingPyMlirContext ctx) {               \
            return PyClass(ctx->getRef(), GetFn(ctx.get()->get(), value));     \
          },                                                                   \
          nb::arg("value"), nb::arg("context").none() = nb::none());           \
      c.def_static(                                                            \
          "get",                                                               \
          [](const std::string &name, mpx::DefaultingPyMlirContext ctx) {      \
            MlirAttribute attr =                                               \
                GetByNameFn(ctx.get()->get(),                                  \
                            mlirStringRefCreate(name.data(), name.size()));    \
            if (mlirAttributeIsNull(attr))                                     \
              throw nb::value_error(                                           \
                  (PyName " has no case named '" + name + "'").c_str());       \
            return PyClass(ctx->getRef(), attr);                               \
          },                                                                   \
          nb::arg("name"), nb::arg("context").none() = nb::none());            \
      c.def_prop_ro("value", [](PyClass &self) { return GetValueFn(self); });  \
    }                                                                          \
  };

ALLO_ENUM_ATTR(PyAssumeDepTypeAttr, "AssumeDepTypeAttr",
               alloAttributeIsAAssumeDepType, alloAssumeDepTypeAttrGet,
               alloAssumeDepTypeAttrGetByName, alloAssumeDepTypeAttrGetValue,
               alloAssumeDepTypeAttrGetTypeID)
ALLO_ENUM_ATTR(PyAssumeDepDirAttr, "AssumeDepDirAttr",
               alloAttributeIsAAssumeDepDir, alloAssumeDepDirAttrGet,
               alloAssumeDepDirAttrGetByName, alloAssumeDepDirAttrGetValue,
               alloAssumeDepDirAttrGetTypeID)
ALLO_ENUM_ATTR(PyMemoryKindAttr, "MemoryKindAttr", alloAttributeIsAMemoryKind,
               alloMemoryKindAttrGet, alloMemoryKindAttrGetByName,
               alloMemoryKindAttrGetValue, alloMemoryKindAttrGetTypeID)
ALLO_ENUM_ATTR(PyDeterminacyAttr, "DeterminacyAttr",
               alloAttributeIsADeterminacy, alloDeterminacyAttrGet,
               alloDeterminacyAttrGetByName, alloDeterminacyAttrGetValue,
               alloDeterminacyAttrGetTypeID)
ALLO_ENUM_ATTR(PyOpKindAttr, "OpKindAttr", alloAttributeIsAOpKind,
               alloOpKindAttrGet, alloOpKindAttrGetByName,
               alloOpKindAttrGetValue, alloOpKindAttrGetTypeID)
ALLO_ENUM_ATTR(PyCombOpKindAttr, "CombOpKindAttr", alloAttributeIsACombOpKind,
               alloCombOpKindAttrGet, alloCombOpKindAttrGetByName,
               alloCombOpKindAttrGetValue, alloCombOpKindAttrGetTypeID)
ALLO_ENUM_ATTR(PyStallContractAttr, "StallContractAttr",
               alloAttributeIsAStallContract, alloStallContractAttrGet,
               alloStallContractAttrGetByName, alloStallContractAttrGetValue,
               alloStallContractAttrGetTypeID)
ALLO_ENUM_ATTR(PyCostFormAttr, "CostFormAttr", alloAttributeIsACostForm,
               alloCostFormAttrGet, alloCostFormAttrGetByName,
               alloCostFormAttrGetValue, alloCostFormAttrGetTypeID)

#undef ALLO_ENUM_ATTR

/// The two structural attributes below bind authoring and evaluation only; the
/// field accessors the CAPI carries for them stay unbound.

/// CostAttr: #allo.cost<form, [coeffs], [arms]>
///   form: the CostFormEnum case; arms: the two arms of a `piecewise`, and
///   empty for every other form.
struct PyCostAttr : mpx::PyConcreteAttribute<PyCostAttr> {
  static constexpr IsAFunctionTy isaFunction = alloAttributeIsACost;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      alloCostAttrGetTypeID;
  static constexpr const char *pyClassName = "CostAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static PyCostAttr build(uint32_t form, std::vector<double> &coeffs,
                          std::vector<MlirAttribute> &arms,
                          mpx::DefaultingPyMlirContext ctx) {
    return PyCostAttr(
        ctx->getRef(),
        alloCostAttrGet(ctx.get()->get(), form,
                        static_cast<intptr_t>(coeffs.size()), coeffs.data(),
                        static_cast<intptr_t>(arms.size()), arms.data()));
  }

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](uint32_t form, std::vector<double> coeffs,
           std::vector<MlirAttribute> arms, mpx::DefaultingPyMlirContext ctx) {
          return build(form, coeffs, arms, ctx);
        },
        nb::arg("form"), nb::arg("coeffs"),
        nb::arg("arms") = std::vector<MlirAttribute>(),
        nb::arg("context").none() = nb::none());
    c.def_static(
        "get",
        [](const std::string &form, std::vector<double> coeffs,
           std::vector<MlirAttribute> arms, mpx::DefaultingPyMlirContext ctx) {
          MlirAttribute formAttr = alloCostFormAttrGetByName(
              ctx.get()->get(), mlirStringRefCreate(form.data(), form.size()));
          if (mlirAttributeIsNull(formAttr))
            throw nb::value_error(
                ("there is no cost form named '" + form + "'").c_str());
          return build(alloCostFormAttrGetValue(formAttr), coeffs, arms, ctx);
        },
        nb::arg("form"), nb::arg("coeffs"),
        nb::arg("arms") = std::vector<MlirAttribute>(),
        nb::arg("context").none() = nb::none());
    c.def(
        "evaluate",
        [](PyCostAttr &self, int64_t param) -> std::optional<double> {
          double value = 0.0;
          if (!alloEvaluateCost(self, param, &value))
            return std::nullopt;
          return value;
        },
        nb::arg("param"),
        "This cost at one parameter, unrounded: what a `dcp.comb` row's delay "
        "is at an operand width. None outside its measured points.");
  }
};

/// ResourceUseAttr: #allo.res_use<@resource, [factors]>
struct PyResourceUseAttr : mpx::PyConcreteAttribute<PyResourceUseAttr> {
  static constexpr IsAFunctionTy isaFunction = alloAttributeIsAResourceUse;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      alloResourceUseAttrGetTypeID;
  static constexpr const char *pyClassName = "ResourceUseAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](MlirAttribute resource, std::vector<MlirAttribute> factors,
           mpx::DefaultingPyMlirContext ctx) {
          return PyResourceUseAttr(
              ctx->getRef(),
              alloResourceUseAttrGet(ctx.get()->get(), resource,
                                     static_cast<intptr_t>(factors.size()),
                                     factors.data()));
        },
        nb::arg("resource"), nb::arg("factors"),
        nb::arg("context").none() = nb::none());
    c.def_static(
        "evaluate_all",
        [](MlirAttribute uses, const std::vector<int64_t> &params)
            -> std::optional<std::vector<std::pair<std::string, int64_t>>> {
          std::vector<std::pair<std::string, int64_t>> spent;
          if (!alloEvaluateResourceUse(
                  uses, static_cast<intptr_t>(params.size()), params.data(),
                  [](MlirStringRef resource, int64_t amount, void *userData) {
                    static_cast<std::vector<std::pair<std::string, int64_t>> *>(
                        userData)
                        ->emplace_back(
                            std::string(resource.data, resource.length),
                            amount);
                  },
                  &spent))
            return std::nullopt;
          return spent;
        },
        nb::arg("uses"), nb::arg("params"),
        "What an array of these spends at a realization's parameter tuple, as "
        "(resource, amount) pairs, and None where a cost was not measured at "
        "its parameter. Static because a resource may be named by several "
        "entries and what it spends is their sum, rounded once the sum is "
        "complete, so no one entry carries the answer.");
  }
};

} // namespace

void allo::populateAlloAttrs(nb::module_ &m) {
  PyPartitionAxisAttr::bind(m);
  PyPartitionAttr::bind(m);
  PyAssumeDepTypeAttr::bind(m);
  PyAssumeDepDirAttr::bind(m);
  PyMemoryKindAttr::bind(m);
  PyDeterminacyAttr::bind(m);
  PyOpKindAttr::bind(m);
  PyCombOpKindAttr::bind(m);
  PyStallContractAttr::bind(m);
  PyCostFormAttr::bind(m);
  PyCostAttr::bind(m);
  PyResourceUseAttr::bind(m);
}
