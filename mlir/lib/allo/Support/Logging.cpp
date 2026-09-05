/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Support/Logging.h"

#include "allo-c/Schedule.h" // kScheduleNameAttr

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/Path.h"

// The project builds with exceptions disabled; make spdlog abort instead of
// throwing (this is the only TU that includes it).
#define SPDLOG_NO_EXCEPTIONS
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

#include <atomic>
#include <cassert>
#include <cstdlib>
#include <memory>
#include <mutex>

using namespace mlir;

namespace mlir::allo::logging {
namespace {

// The threshold below which messages are dropped, and a hard mute switch.
std::atomic<int> gThreshold{static_cast<int>(Level::Info)};
std::atomic<bool> gMuted{false};

std::shared_ptr<spdlog::sinks::stderr_color_sink_mt> gSink;

// A level that fails the compilation: never filtered, and raising an MLIR
// diagnostic on its subject.
bool isFatal(Level level) {
  return level == Level::Error || level == Level::Unsupported;
}

spdlog::level::level_enum toSpdlog(Level level) {
  switch (level) {
  case Level::Debug:
    return spdlog::level::debug;
  case Level::Info:
    return spdlog::level::info;
  case Level::Warn:
    return spdlog::level::warn;
  case Level::Error:
  case Level::Unsupported:
    return spdlog::level::err;
  }
  return spdlog::level::info;
}

const char *levelTag(Level level) {
  switch (level) {
  case Level::Debug:
    return "DEBUG";
  case Level::Info:
    return "INFO";
  case Level::Warn:
    return "WARN";
  case Level::Error:
    return "ERROR";
  case Level::Unsupported:
    return "NYI";
  }
  return "INFO";
}

const char *stageTag(Stage stage) {
  switch (stage) {
  case Stage::Prep:
    return "PREP";
  case Stage::Sched:
    return "SCHED";
  case Stage::Dcp:
    return "DCP";
  case Stage::Emit:
    return "EMIT";
  }
  return "PREP";
}

// Seed the threshold from ALLO_LOG_LEVEL (debug/info/warn/error, or off).
void applyEnvLevel(StringRef value) {
  std::string v = value.lower();
  if (v == "off" || v == "none" || v == "silent")
    gMuted.store(true);
  else if (v == "debug" || v == "trace")
    gThreshold.store(static_cast<int>(Level::Debug));
  else if (v == "info")
    gThreshold.store(static_cast<int>(Level::Info));
  else if (v == "warn" || v == "warning")
    gThreshold.store(static_cast<int>(Level::Warn));
  else if (v == "error" || v == "err")
    gThreshold.store(static_cast<int>(Level::Error));
}

// Apply ALLO_LOG_LEVEL exactly once, before the first threshold read. An
// explicit setLevel() call afterwards overrides it.
void initFromEnv() {
  static std::once_flag once;
  std::call_once(once, [] {
    if (const char *env = std::getenv("ALLO_LOG_LEVEL"))
      applyEnvLevel(env);
  });
}

// Diagnostics go to stderr so they never corrupt tool output on stdout. The
// text carries its own `[LEVEL][STAGE]` prefix, and filtering is done by our
// own threshold, not spdlog's.
spdlog::logger &logger() {
  static std::shared_ptr<spdlog::logger> instance = [] {
    gSink = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
    auto lg = std::make_shared<spdlog::logger>("allo", gSink);
    lg->set_pattern("%^%v%$");
    lg->set_level(spdlog::level::trace);
    lg->flush_on(spdlog::level::trace);
    return lg;
  }();
  return *instance;
}

// The first symbolic name and first file:line:col reachable from `loc`.
void walkLocation(Location loc, std::string &name, std::string &pos) {
  if (auto named = dyn_cast<NameLoc>(loc)) {
    if (name.empty())
      name = named.getName().getValue().str();
    walkLocation(named.getChildLoc(), name, pos);
  } else if (auto file = dyn_cast<FileLineColLoc>(loc)) {
    if (pos.empty()) {
      StringRef base = llvm::sys::path::filename(file.getFilename().getValue());
      pos = (base + ":" + Twine(file.getLine()) + ":" + Twine(file.getColumn()))
                .str();
    }
  } else if (auto fused = dyn_cast<FusedLoc>(loc)) {
    for (Location child : fused.getLocations()) {
      walkLocation(child, name, pos);
      if (!name.empty() && !pos.empty())
        break;
    }
  } else if (auto call = dyn_cast<CallSiteLoc>(loc)) {
    walkLocation(call.getCallee(), name, pos);
  }
}

std::string render(StringRef name, bool isLoop, StringRef pos) {
  std::string out;
  if (!name.empty()) {
    out += isLoop ? "loop '" : "'";
    out.append(name.data(), name.size());
    out += '\'';
  }
  if (!pos.empty()) {
    if (!out.empty())
      out += " ";
    out.append(pos.data(), pos.size());
  }
  return out;
}

} // namespace

void detail::emit(Level level, Stage stage, StringRef code, StringRef where,
                  StringRef message, Operation *subject) {
  assert(isFatal(level) == !code.empty() &&
         "a refusal carries a code and nothing else does");
  assert((code.empty() || code[5] == (level == Level::Error ? 'E' : 'N')) &&
         "the code's series names the level it is reported at");
  std::string line;
  line.reserve(message.size() + where.size() + code.size() + 24);
  line += levelTag(level);
  if (!code.empty()) {
    line += '[';
    line.append(code.data(), code.size());
    line += ']';
  }
  line += ": [";
  line += stageTag(stage);
  line += "] ";
  line.append(message.data(), message.size());
  if (!where.empty()) {
    line += " (at ";
    line.append(where.data(), where.size());
    line += ')';
  }
  // Pass the line as an argument, never as the format string, so any braces in
  // the message (attributes, types) are treated as data.
  logger().log(toSpdlog(level), "{}", line);

  // A fatal message also emits an MLIR diagnostic so the pass fails and the
  // message surfaces to the caller. The diagnostic attaches the op's location,
  // and leads with the code: this is the one path by which the stable token
  // reaches a caller, which sees the diagnostic text and never this log.
  if (isFatal(level) && subject)
    subject->emitError() << "[" << code << "] " << message;
}

bool detail::enabled(Level level) {
  if (isFatal(level))
    return true; // a fatal message is never filtered
  initFromEnv();
  return !gMuted.load(std::memory_order_relaxed) &&
         static_cast<int>(level) >= gThreshold.load(std::memory_order_relaxed);
}

std::string detail::describe(Operation *op) {
  if (!op)
    return {};
  std::string name, pos;
  bool isLoop = false;
  if (auto attr = op->getAttrOfType<StringAttr>(kScheduleNameAttr)) {
    name = attr.getValue().str();
    isLoop = true;
  }
  walkLocation(op->getLoc(), name, pos);
  return render(name, isLoop, pos);
}

std::string detail::describe(const Location &loc, bool withFile) {
  std::string name, pos;
  walkLocation(loc, name, pos);
  if (withFile)
    return render(name, /*isLoop=*/false, pos);
  return name;
}

void setLevel(Level level) {
  initFromEnv(); // so a later first-use env read cannot clobber this
  gThreshold.store(static_cast<int>(level));
  gMuted.store(false);
}

Level getLevel() {
  initFromEnv();
  return static_cast<Level>(gThreshold.load());
}

void setColor(bool enable) {
  logger(); // ensure the sink exists
  if (gSink)
    gSink->set_color_mode(enable ? spdlog::color_mode::always
                                 : spdlog::color_mode::never);
}

} // namespace mlir::allo::logging
