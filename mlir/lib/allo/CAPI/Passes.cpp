/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo-c/Passes.h"
#include "allo/Microarch/EmitDriver.h"
#include "allo/Microarch/Report.h"
#include "allo/Scheduling/MemoryModel.h"
#include "allo/Scheduling/Scheduler.h"
#include "allo/Support/Logging.h"

#include "allo/Translation/VerilogEmitter.h"
#include "allo/Translation/VivadoHLSEmitter.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Utils.h"

#include "llvm/ADT/StringMap.h"

using namespace mlir;

MlirLogicalResult alloEmitVivadoHLS(MlirModule module, bool enableApFloat,
                                    unsigned indexWidth, bool withLocation,
                                    MlirStringRef top,
                                    MlirStringCallback callback,
                                    void *userData) {
  mlir::detail::CallbackOstream stream(callback, userData);
  return wrap(allo::emitVivadoHLS(unwrap(module), stream, enableApFloat,
                                  indexWidth, withLocation, unwrap(top)));
}

MlirLogicalResult alloEmitVerilog(MlirModule module,
                                  MlirStringCallback callback, void *userData) {
  mlir::detail::CallbackOstream stream(callback, userData);
  return wrap(allo::emitVerilog(unwrap(module), stream));
}

MlirLogicalResult alloEmitSplitVerilog(MlirModule module,
                                       MlirStringRef directory) {
  return wrap(allo::emitSplitVerilog(unwrap(module), unwrap(directory)));
}

MlirLogicalResult alloEmitDatapathToHW(MlirModule module, MlirStringRef binding,
                                       MlirStringRef top, double cycleTime,
                                       MlirStringCallback callback,
                                       void *userData) {
  llvm::StringMap<std::string> interfaces;
  allo::uarch::MicroarchReport report;
  if (failed(allo::uarch::emitDatapathToHW(unwrap(module), unwrap(binding),
                                           unwrap(top), (float)cycleTime,
                                           interfaces, report)))
    return mlirLogicalResultFailure();
  // One envelope carrying both of the emission's documents: the per-module
  // port manifests keyed by module name, and the allocation report. Every
  // value is already valid JSON and a module name is a plain identifier, so
  // nothing here needs escaping.
  std::string out = "{\"version\":1,\"interfaces\":{";
  bool first = true;
  for (const auto &kv : interfaces) {
    if (!first)
      out += ',';
    first = false;
    out += '"';
    out += kv.first();
    out += "\":";
    out += kv.second;
  }
  out += "},\"microarch\":";
  out += report.toJSON();
  out += '}';
  callback(MlirStringRef{out.data(), out.size()}, userData);
  return mlirLogicalResultSuccess();
}

MlirLogicalResult alloRunSDCSchedulingPipeline(
    MlirModule module, MlirStringRef top, float cycleTime,
    MlirStringRef scheduler, MlirStringRef objective, double budget,
    bool allocate, int workers, int seed, bool deterministic, double areaSlack,
    MlirStringCallback callback, void *userData) {
  ModuleOp mod = unwrap(module);
  StringRef topName = unwrap(top);
  StringRef schedulerName = unwrap(scheduler);
  std::optional<allo::SchedulerKind> kind =
      allo::parseSchedulerKind(schedulerName);
  if (!kind) {
    allo::logging::error(allo::logging::Stage::Sched,
                         allo::logging::Code::UnknownOption, mod)
        << "Unknown scheduler '" << schedulerName
        << "'; expected \"heuristic\" or \"exact\"";
    return mlirLogicalResultFailure();
  }
  StringRef objectiveName = unwrap(objective);
  std::optional<allo::ScheduleObjective> obj =
      allo::parseScheduleObjective(objectiveName);
  if (!obj) {
    allo::logging::error(allo::logging::Stage::Sched,
                         allo::logging::Code::UnknownOption, mod)
        << "Unknown objective '" << objectiveName
        << "'; expected \"cycles\" or \"area\"";
    return mlirLogicalResultFailure();
  }
  // The target clock period, the exact-solve budget and its worker count take
  // the option, else the default, resolved once here so no second copy exists
  // downstream. A seed of zero is itself the default and passes through.
  float cycleTimeNs = cycleTime > 0.0f ? cycleTime : 5.0f;
  allo::SchedulerOptions opts{*kind,
                              *obj,
                              budget > 0.0 ? budget : allo::kDefaultSolveBudget,
                              allocate,
                              workers > 0 ? workers
                                          : allo::kDefaultSolveWorkers,
                              seed,
                              deterministic,
                              areaSlack > 0.0 ? areaSlack : 0.0};
  // The storage decision, taken once and recorded on every array before any
  // layer below reads it.
  allo::recordArrayStorage(mod, allo::MemoryLibrary::fromModule(mod));
  if (failed(allo::runPreScheduleVerification(mod, topName)))
    return mlirLogicalResultFailure();
  // The solved schedule travels between the two halves in memory, not as IR
  // attributes, so its `Operation *` keys cannot outlive the ops they name.
  allo::ScheduleModel model;
  if (failed(allo::runSDCScheduler(mod, topName, cycleTimeNs, opts, model)))
    return mlirLogicalResultFailure();
  allo::runPostScheduleConversion(mod, model);
  std::string report = model.toJSON();
  callback(MlirStringRef{report.data(), report.size()}, userData);
  return mlirLogicalResultSuccess();
}
