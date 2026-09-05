/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "AlloBindings.h"

#include "allo-c/AlloAttrs.h"
#include "allo-c/IRUtils.h"
#include "allo-c/Passes.h"
#include "allo-c/Registration.h"
#include "allo-c/Schedule.h"

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#include "mlir/Bindings/Python/NanobindAdaptors.h"

#include "nanobind/nanobind.h"
#include "nanobind/stl/optional.h"
#include "nanobind/stl/pair.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/vector.h"

#include "llvm-c/ErrorHandling.h"
#include "llvm/Support/Signals.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace {
/// MlirStringCallback that appends every chunk to a std::string (userData).
void appendToString(MlirStringRef chunk, void *userData) {
  static_cast<std::string *>(userData)->append(chunk.data, chunk.length);
}
} // namespace

NB_MODULE(_allo, m) {
  m.doc() = "Allo Python Native Extension";
  llvm::sys::PrintStackTraceOnErrorSignal("");
  LLVMEnablePrettyStackTrace();

  //===--------------------------------------------------------------------===//
  // allo: dialect / pass / extension registration
  //===--------------------------------------------------------------------===//
  auto allo = m.def_submodule("allo", "allo dialect registration");
  allo.def(
      "register_dialect",
      [](MlirContext context) { alloMlirRegisterAllDialects(context); },
      nb::arg("context"),
      "Register and load every dialect Allo needs (including allo).");
  allo.def(
      "register_extensions",
      [](MlirContext context) { alloMlirRegisterAllExtensions(context); },
      nb::arg("context"),
      "Register and load the transform dialect + Allo transform extensions.");
  allo.def(
      "register_passes", []() { alloMlirRegisterAllPasses(); },
      "Register all Allo passes with the global pass registry.");
  alloMlirRegisterAllPasses();
  allo.attr("SIGNED_ATTR_NAME") = kAlloSignedAttr;
  allo.attr("LAZY_ATTR_NAME") = kAlloLazyAttr;

  //===--------------------------------------------------------------------===//
  // emit: Vivado HLS translation. General/Allo pass pipelines (incl. the
  // registered `allo-lower-to-llvm` pipeline) are driven through upstream
  // `mlir.passmanager.PassManager`, so no pass wrappers live here.
  //===--------------------------------------------------------------------===//
  allo.def(
      "emit_vivado_hls",
      [](MlirModule module, bool enableApFloat, const std::string &top,
         unsigned indexWidth, bool withLocation) -> std::optional<std::string> {
        std::string out;
        if (mlirLogicalResultIsFailure(alloEmitVivadoHLS(
                module, enableApFloat, indexWidth, withLocation,
                mlirStringRefCreate(top.data(), top.size()), appendToString,
                &out)))
          return std::nullopt;
        return out;
      },
      nb::arg("module"), nb::arg("enable_apfloat"), nb::arg("top") = "",
      nb::arg("index_width") = 32, nb::arg("with_location") = true);
  allo.def(
      "emit_verilog",
      [](MlirModule module) -> std::optional<std::string> {
        std::string out;
        if (mlirLogicalResultIsFailure(
                alloEmitVerilog(module, appendToString, &out)))
          return std::nullopt;
        return out;
      },
      nb::arg("module"));
  allo.def(
      "emit_datapath_to_hw",
      [](MlirModule module, const std::string &binding, const std::string &top,
         double cycle_time) -> std::optional<std::string> {
        std::string out;
        if (mlirLogicalResultIsFailure(alloEmitDatapathToHW(
                module, mlirStringRefCreate(binding.data(), binding.size()),
                mlirStringRefCreate(top.data(), top.size()), cycle_time,
                appendToString, &out)))
          return std::nullopt;
        return out;
      },
      nb::arg("module"), nb::arg("binding"), nb::arg("top"),
      nb::arg("cycle_time"));
  allo.def(
      "emit_split_verilog",
      [](MlirModule module, const std::string &directory) -> bool {
        return mlirLogicalResultIsSuccess(alloEmitSplitVerilog(
            module, mlirStringRefCreate(directory.data(), directory.size())));
      },
      nb::arg("module"), nb::arg("directory"));

  allo.def(
      "run_sdc_scheduling",
      [](MlirModule module, const std::string &top, float cycleTime,
         const std::string &scheduler, double budget, bool allocate,
         int workers, int seed, bool deterministic, double areaSlack,
         bool escalate) -> std::optional<std::string> {
        std::string out;
        if (mlirLogicalResultIsFailure(alloRunSDCSchedulingPipeline(
                module, mlirStringRefCreate(top.data(), top.size()), cycleTime,
                mlirStringRefCreate(scheduler.data(), scheduler.size()), budget,
                allocate, workers, seed, deterministic, areaSlack, escalate,
                appendToString, &out)))
          return std::nullopt;
        return out;
      },
      nb::arg("module"), nb::arg("top"), nb::arg("cycle_time"),
      nb::arg("scheduler") = "heuristic", nb::arg("budget") = 0.0,
      nb::arg("allocate") = false, nb::arg("workers") = 0, nb::arg("seed") = 0,
      nb::arg("deterministic") = true, nb::arg("area_slack") = 0.0,
      nb::arg("escalate") = true);

  //===--------------------------------------------------------------------===//
  // schedule
  //===--------------------------------------------------------------------===//
  auto schedule = m.def_submodule("schedule", "schedule analysis");
  schedule.def(
      "annotate_schedule_ids",
      [](MlirModule module) { alloAnnotateScheduleIds(module); },
      nb::arg("module"));
  schedule.def(
      "cleanup_schedule_ids",
      [](MlirModule module) { alloCleanupScheduleIds(module); },
      nb::arg("module"));
  schedule.def(
      "collect_schedule_snapshot_json",
      [](MlirModule module) {
        std::string out;
        alloCollectScheduleSnapshotJSON(module, appendToString, &out);
        return out;
      },
      nb::arg("module"),
      "Return the schedule snapshot as a JSON document (parse on the Python "
      "side).");
  schedule.attr("SCHEDULE_ID_ATTR_NAME") = kScheduleIdAttr;
  schedule.attr("SCHEDULE_NAME_ATTR_NAME") = kScheduleNameAttr;
  schedule.attr("PIPELINE_II_ATTR_NAME") = kPipelineIIAttr;
  schedule.attr("DATAFLOW_ATTR_NAME") = kDataflowAttr;
  schedule.attr("UNROLL_FACTOR_ATTR_NAME") = kUnrollFactorAttr;
  schedule.attr("PARTITION_ATTR_NAME") = kPartitionAttr;
  schedule.attr("PIPELINE_REWIND_ATTR_NAME") = kPipelineRewindAttr;

  //===--------------------------------------------------------------------===//
  // ir_ext: block surgery + KernelOp helpers not exposed by upstream
  // bindings
  //===--------------------------------------------------------------------===//
  auto ir_ext = m.def_submodule("ir_ext", "IR mutation helpers");
  ir_ext.def(
      "erase_block", [](MlirBlock block) { alloBlockErase(block); },
      nb::arg("block"));
  ir_ext.def(
      "merge_block_before",
      [](MlirBlock src, MlirBlock dst) { alloBlockMergeBefore(src, dst); },
      nb::arg("src"), nb::arg("dst"));
  ir_ext.def(
      "clone_module",
      [](MlirModule module) { return alloCloneModuleOp(module); },
      nb::arg("module"), "Return a clone of the given module.");

  //===--------------------------------------------------------------------===//
  // Allo dialect types / attributes (CRTP subclasses of
  // allo._mlir.ir.Type/Attribute)
  //===--------------------------------------------------------------------===//
  allo::populateAlloTypes(m);
  allo::populateAlloAttrs(m);
}
