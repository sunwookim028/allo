/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Microarch/EmitDriver.h"

#include "allo/IR/AlloOps.h"
#include "allo/Microarch/BindingPolicy.h"
#include "allo/Microarch/HWEmitter.h" // HWEmitter
#include "allo/Microarch/Interface.h"
#include "allo/Microarch/Report.h"
#include "allo/Microarch/Verification.h" // validateDatapath
#include "allo/Scheduling/OperatorLibrary.h"
#include "allo/Support/Logging.h"

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/BackedgeBuilder.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/StringMap.h"

using namespace mlir;
using namespace mlir::allo;
using namespace mlir::allo::logging;
using namespace circt;

#define DEBUG_TYPE "hw-emitter"

namespace mlir::allo::uarch {

//===----------------------------------------------------------------------===//
// emitModule: interface (ports, extern operator modules) + validation.
//===----------------------------------------------------------------------===//

// Declare one extern `hw.module` per operator the manifest lists, deduplicated
// across the whole module (`opModules`, which the emitter then looks a unit's
// module up in by name). The manifest entry is the source of the port shape,
// shared with the simulation model built from it.
//
// The module name stems from the `dcp.operator`'s own `sym_name`, and that
// declaration stays live until every kernel has emitted, so the symbol is
// briefly duplicated. `SymbolTable::lookupSymbolIn` returns the first match in
// block order, so the `dcp.operator` has to stay ahead of these declarations.
static void declareOperatorModules(ArrayRef<iface::Operator> operators,
                                   OpBuilder &b, Location loc,
                                   llvm::StringMap<Operation *> &opModules) {
  auto *ctx = b.getContext();
  using Dir = hw::ModulePort::Direction;
  for (const iface::Operator &o : operators) {
    Operation *&mod = opModules[o.module];
    if (mod)
      continue;
    SmallVector<hw::PortInfo> ep;
    for (const iface::Operator::Port &p : o.ports)
      ep.push_back({{StringAttr::get(ctx, p.name), b.getIntegerType(p.width),
                     p.isInput() ? Dir::Input : Dir::Output}});
    mod = hw::HWModuleExternOp::create(b, loc, StringAttr::get(ctx, o.module),
                                       hw::ModulePortInfo(ep));
  }
}

/// Flip-flops in \p mod's own body, which is what the ledger claims to count.
/// A child instance's registers live in the child's body and are not walked.
[[maybe_unused]] static unsigned compRegBits(hw::HWModuleOp mod) {
  unsigned bits = 0;
  mod.walk([&](Operation *op) {
    if (isa<seq::CompRegOp, seq::CompRegClockEnabledOp>(op))
      bits += datapathWidth(op->getResult(0).getType());
  });
  return bits;
}

// Emit an hw.module for one scheduled function's datapath. Returns failure with
// a diagnostic if the datapath is outside the supported subset
// (validateDatapath). `opModules` caches extern operator modules across
// functions.
static FailureOr<std::pair<hw::HWModuleOp, iface::ModuleInterface>>
emitModule(const uarch::Datapath &dp, OpBuilder &b,
           llvm::StringMap<Operation *> &opModules, float cycleTime,
           const OperatorLibrary &lib, bool plannedBinding,
           MicroarchReport &report, const uarch::CalleeCtx &callees) {
  auto *ctx = b.getContext();
  dcp::DCPathModuleOp func = dp.func;
  Location loc = func.getLoc();
  FailureOr<std::vector<uarch::TimingPath>> criticalPaths =
      validateDatapath(func, dp, cycleTime, lib, plannedBinding);
  if (failed(criticalPaths))
    return failure();

  Type i1 = b.getI1Type();
  Type i32 = b.getIntegerType(32);

  // The single source for every boundary port name, shared by declaration,
  // manifest and cosim harness; it also carries the extern operator modules
  // this kernel instantiates. Complete once constructed.
  iface::ModuleInterface model(dp);
  declareOperatorModules(model.operators, b, loc, opModules);
  hw::ModulePortInfo portInfo(declareModulePorts(model, b));
  StringAttr modName = StringAttr::get(ctx, model.module);

  RegLedger ledger;
  MuxLedger muxLedger;
  auto hwMod = hw::HWModuleOp::create(
      b, loc, modName, portInfo,
      [&](OpBuilder &ib, hw::HWModulePortAccessor &pa) {
        BackedgeBuilder bb(ib, loc);
        HWEmitter e(ib, loc, dp, pa, opModules, bb, i1, i32, callees);
        e.emit();
        ledger = std::move(e.ctx.ledger);
        muxLedger = std::move(e.ctx.muxLedger);
      });
  // Every register came through `EmitContext::reg`, so the ledger is the
  // emitted design's own flip-flop count and not a model of it. Checked here
  // rather than in one test, so every emission the suite runs holds it.
  assert(compRegBits(hwMod) == ledger.bits() &&
         "a register was built outside EmitContext::reg, so the ledger is no "
         "longer a count of the emitted design");
  report.funcs.emplace_back(dp, model.symbol, model.module, ledger, muxLedger,
                            std::move(*criticalPaths));

  // The caller derives the cosim manifest JSON from this port model and threads
  // it back in as a callee model.
  return std::make_pair(hwMod, std::move(model));
}

static void cleanupDcpOps(ModuleOp module) {
  // cleanup non-hw ops to avoid Verilog export errors
  for (dcp::DCPathModuleOp f :
       llvm::make_early_inc_range(module.getOps<dcp::DCPathModuleOp>()))
    f.erase();
  for (memref::GlobalOp g :
       llvm::make_early_inc_range(module.getOps<memref::GlobalOp>()))
    g.erase();
  // Spent declarations, dropped last: a `dcp.compute` reads its timing off the
  // `dcp.operator` it names, and dropping them leaves each extern operator
  // module sole owner of its `sym_name`.
  SmallVector<Operation *> spent;
  module.walk([&](Operation *op) {
    if (isa<dcp::DCPathOperatorOp, dcp::DCPathDeviceOp, dcp::DCPathUnitOp>(op))
      spent.push_back(op);
  });
  for (Operation *op : spent)
    op->erase();
}

LogicalResult emitDatapathToHW(ModuleOp module, StringRef binding,
                               StringRef top, float cycleTime,
                               llvm::StringMap<std::string> &interfaces,
                               MicroarchReport &report) {
  report.binding = binding.str();
  report.cycleTime = cycleTime;
  // Called directly (not via the pass manager), so load the dialects this
  // emits, the ones the pass declares as dependent, into the context.
  auto *ctx = module.getContext();
  ctx->getOrLoadDialect<hw::HWDialect>();
  ctx->getOrLoadDialect<comb::CombDialect>();
  ctx->getOrLoadDialect<seq::SeqDialect>();

  // Storage and comb timing have no per-access carrier, so they thread into the
  // datapath builder as a library; an IP's timing rides the `dcp.operator` its
  // `dcp.compute` names, which stays live for the whole of emission.
  DeviceModel dev = DeviceModel::fromModule(module);

  auto policy = bindingPolicyFor(binding);
  if (!policy) {
    error(Stage::Emit, Code::UnknownOption, module)
        << "Unknown binding policy '" << binding
        << "'; the policies are 'trivial', 'exact-share' and 'planned'";
    return failure();
  }

  // Bottom-up over the call DAG: a container always finds its children already
  // registered.
  llvm::StringMap<dcp::DCPathModuleOp> byName;
  for (dcp::DCPathModuleOp f : module.getOps<dcp::DCPathModuleOp>())
    byName[f.getSymName()] = f;
  dcp::DCPathModuleOp topFunc = byName.lookup(top);
  if (!topFunc) {
    error(Stage::Emit, Code::TopFunctionMissing, module)
        << "Top function '" << top << "' is not a scheduled function";
    return failure();
  }

  OpBuilder b(module.getBodyRegion());
  llvm::StringMap<Operation *> opModules;
  // Callee tables, keyed by symbol name: leaf kernels plus the containers
  // emitted so far, which compose exactly like a leaf.
  llvm::StringMap<hw::HWModuleOp> modules;
  llvm::StringMap<iface::ModuleInterface> ifaceModels;
  llvm::StringSet<> visited;

  auto registerModule = [&](StringRef name, hw::HWModuleOp mod,
                            iface::ModuleInterface model) {
    // The callee tables key on the func symbol, which a callsite names; the
    // manifest keys on the emitted module name, which the simulator names.
    interfaces[mod.getModuleName()] = model.toJSON();
    modules[name] = mod;
    ifaceModels[name] = std::move(model);
  };

  // Post-order over the call DAG, which is acyclic: the frontend rejects
  // recursion.
  auto emitOne = [&](auto &self, dcp::DCPathModuleOp f) -> LogicalResult {
    if (!visited.insert(f.getSymName()).second)
      return success(); // a shared callee already emitted
    // Children first: a `dcp.instance` is the only way a kernel reaches another
    // one, so it is the only edge to recurse on, and a leaf call misses
    // `byName`.
    WalkResult wr = f.walk([&](dcp::DCPathInstanceOp inv) -> WalkResult {
      auto it = byName.find(inv.getCallee());
      if (it != byName.end() && failed(self(self, it->second)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (wr.wasInterrupted())
      return failure();

    // One emission path, whichever way the function composes: leaf, sequential
    // container and dataflow differ only in the start policy they pick. The
    // post-order walk above means `cc` names every module emitted so far.
    const uarch::CalleeCtx cc{modules, ifaceModels};
    const Datapath dp(f, *policy, dev, cycleTime, cc, /*isTop=*/f == topFunc);
#ifndef NDEBUG
    // A determinate call is released by a static offset with no handshake, so
    // the latency it declares must equal the callee's own whole-kernel
    // contract. The two are stamped independently and compared nowhere else.
    for (const uarch::CallUnit &cu : dp.calls) {
      dcp::DCPathModuleOp callee = byName.lookup(cu.callee);
      if (!cu.determinate || !callee)
        continue;
      assert(cu.latency && callee.getLatency() &&
             *cu.latency == static_cast<int64_t>(*callee.getLatency()) &&
             "a determinate call's declared latency diverges from its "
             "callee's contract; a consumer released at that offset samples "
             "the wrong cycle");
    }
#endif
    LLVM_DEBUG({
      llvm::dbgs() << "// datapath for @" << f.getSymName() << "\n";
      dp.dump(llvm::dbgs());
    });
    b.setInsertionPoint(f);
    auto pairOr = emitModule(dp, b, opModules, cycleTime, dev.operators,
                             policy->realizesSolvePlan(), report, cc);
    if (failed(pairOr))
      return failure();
    registerModule(f.getSymName(), pairOr->first, std::move(pairOr->second));
    return success();
  };

  if (failed(emitOne(emitOne, topFunc)))
    return failure();

  cleanupDcpOps(module);
  return success();
}

} // namespace mlir::allo::uarch
