/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * CRTP nanobind subclasses of `allo._mlir.ir.Type` for the Allo dialect's
 * custom types (see mlir/examples/standalone for the upstream pattern).
 * Construction/introspection is funnelled through the Allo CAPI so the
 * extension links no MLIR C++ statically.
 */

#include "AlloBindings.h"

#include "allo-c/AlloTypes.h"

#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/IRCore.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

#include "nanobind/stl/vector.h"

#include <cstdint>
#include <vector>

namespace nb = nanobind;
namespace mpx = mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN;

namespace {
/// StreamType: !allo.stream<baseType, depth, [shape...]>
struct PyStreamType : mpx::PyConcreteType<PyStreamType> {
  static constexpr IsAFunctionTy isaFunction = alloTypeIsAStream;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      alloStreamTypeGetTypeID;
  static constexpr const char *pyClassName = "StreamType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](mpx::PyType &baseType, uint64_t depth, std::vector<int64_t> shape) {
          return PyStreamType(
              baseType.getContext(),
              alloStreamTypeGet(mlirTypeGetContext(baseType), baseType, depth,
                                static_cast<intptr_t>(shape.size()),
                                shape.data()));
        },
        "Build an !allo.stream type carrying `base_type` with the given "
        "buffer `depth` and array `shape`.",
        nb::arg("base_type"), nb::arg("depth"), nb::arg("shape"));
    c.def_prop_ro("base_type", [](PyStreamType &self) {
      return alloStreamTypeGetBaseType(self);
    });
    c.def_prop_ro("depth", [](PyStreamType &self) {
      return alloStreamTypeGetDepth(self);
    });
    c.def_prop_ro("shape", [](PyStreamType &self) {
      std::vector<int64_t> shape;
      for (intptr_t i = 0, e = alloStreamTypeGetRank(self); i < e; ++i)
        shape.push_back(alloStreamTypeGetDimSize(self, i));
      return shape;
    });
  }
};
} // namespace

void allo::populateAlloTypes(nb::module_ &m) { PyStreamType::bind(m); }
