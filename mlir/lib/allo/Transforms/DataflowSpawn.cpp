/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Conversion/Passes.h"
#include "allo/Transforms/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/STLExtras.h"

namespace mlir::allo {
#define GEN_PASS_DEF_DATAFLOWSPAWNPASS
#include "allo/Transforms/Passes.h.inc"
} // namespace mlir::allo

using namespace mlir;
using namespace mlir::allo;

namespace {
constexpr llvm::StringLiteral kAlloDataflowSpawnFnName = "allo_df_spawn";
constexpr llvm::StringLiteral kAlloDataflowJoinFnName = "allo_df_join";
constexpr llvm::StringLiteral kAlloDataflowOpenFnName = "allo_df_open";
constexpr llvm::StringLiteral kAlloDataflowCloseFnName = "allo_df_close";
// Rewrites the (already LLVM-lowered) sequential calls to dataflow PEs into
// concurrent fiber spawns onto the marl runtime: open a scheduler, store each
// PE's operands into an alloca'd context, spawn a thunk per PE, then join and
// close. lower-dataflow tags the PE callees with `allo.dataflow.pe`. PEs in a
// region may have different signatures, so every fiber captures its own call's
// operands into a dedicated context; the allocas live on the launcher frame,
// which allo_df_join keeps alive until the fibers finish.
struct DataflowSpawnPass
    : public allo::impl::DataflowSpawnPassBase<DataflowSpawnPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();
    IRRewriter r(module.getContext());
    auto funcs = llvm::to_vector<8>(module.getOps<LLVM::LLVMFuncOp>());
    for (LLVM::LLVMFuncOp func : funcs) {
      if (func.isExternal())
        continue;
      SmallVector<LLVM::CallOp> peCalls;
      func.walk([&](LLVM::CallOp call) {
        std::optional<StringRef> callee = call.getCallee();
        if (!callee)
          return;
        auto fn = module.lookupSymbol<LLVM::LLVMFuncOp>(*callee);
        if (fn && fn->hasAttr(kAlloDataflowPEAttrName))
          peCalls.push_back(call);
      });
      if (!peCalls.empty())
        lowerRegion(r, module, peCalls);
    }
    for (LLVM::LLVMFuncOp func : funcs)
      func->removeAttr(kAlloDataflowPEAttrName);
  }

  LLVM::LLVMFuncOp getRuntimeFunc(ModuleOp module, StringRef name,
                                  LLVM::LLVMFunctionType type) {
    if (auto fn = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
      return fn;
    OpBuilder b(module.getBody(), module.getBody()->begin());
    return LLVM::LLVMFuncOp::create(b, module.getLoc(), name, type);
  }

  // The fiber entry point for \p callee: reloads the operands from the context
  // struct and calls it. One thunk per callee, shared by its spawns.
  LLVM::LLVMFuncOp getOrCreateThunk(ModuleOp module, LLVM::LLVMFuncOp callee,
                                    LLVM::LLVMStructType ctxTy,
                                    ArrayRef<Type> opTypes, Type ptrTy,
                                    Type voidTy) {
    std::string name = ("__allo_pe_thunk_" + callee.getSymName()).str();
    if (auto fn = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
      return fn;
    OpBuilder b(module.getBody(), module.getBody()->end());
    Location loc = callee.getLoc();
    auto thunk = LLVM::LLVMFuncOp::create(
        b, loc, name, LLVM::LLVMFunctionType::get(voidTy, {ptrTy}),
        LLVM::Linkage::Internal);
    Block *entry = thunk.addEntryBlock(b);
    b.setInsertionPointToStart(entry);
    Value ctxPtr = entry->getArgument(0);
    SmallVector<Value> args;
    for (auto [i, ty] : llvm::enumerate(opTypes)) {
      Value gep = LLVM::GEPOp::create(
          b, loc, ptrTy, ctxTy, ctxPtr,
          ArrayRef<LLVM::GEPArg>{LLVM::GEPArg(0),
                                 LLVM::GEPArg(static_cast<int32_t>(i))});
      args.push_back(LLVM::LoadOp::create(b, loc, ty, gep));
    }
    LLVM::CallOp::create(b, loc, callee, args);
    LLVM::ReturnOp::create(b, loc, ValueRange{});
    return thunk;
  }

  void lowerRegion(IRRewriter &r, ModuleOp module,
                   ArrayRef<LLVM::CallOp> peCalls) {
    OpBuilder::InsertionGuard g(r);

    MLIRContext *ctx = module.getContext();
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto i64Ty = IntegerType::get(ctx, 64);
    auto voidTy = LLVM::LLVMVoidType::get(ctx);
    auto openFn = getRuntimeFunc(module, kAlloDataflowOpenFnName,
                                 LLVM::LLVMFunctionType::get(ptrTy, {i64Ty}));
    auto spawnFn = getRuntimeFunc(
        module, kAlloDataflowSpawnFnName,
        LLVM::LLVMFunctionType::get(voidTy, {ptrTy, ptrTy, ptrTy}));
    auto joinFn = getRuntimeFunc(module, kAlloDataflowJoinFnName,
                                 LLVM::LLVMFunctionType::get(voidTy, {ptrTy}));
    auto closeFn = getRuntimeFunc(module, kAlloDataflowCloseFnName,
                                  LLVM::LLVMFunctionType::get(voidTy, {ptrTy}));

    LLVM::CallOp first = peCalls.front();
    LLVM::CallOp last = peCalls.back();

    // Open one scheduler for the whole region, before the first PE call; a
    // worker count of 0 asks the runtime for one worker per core.
    r.setInsertionPoint(first);
    Value sched = LLVM::CallOp::create(
                      r, first.getLoc(), openFn,
                      ValueRange{LLVM::ConstantOp::create(
                          r, first.getLoc(), i64Ty, r.getI64IntegerAttr(0))})
                      .getResult();

    for (LLVM::CallOp call : peCalls) {
      OpBuilder::InsertionGuard g(r);
      r.setInsertionPoint(call);
      Location loc = call.getLoc();
      SmallVector<Value> operands(call.getArgOperands());
      SmallVector<Type> opTypes(llvm::map_range(
          operands, [](Value v) -> Type { return v.getType(); }));
      auto ctxTy = LLVM::LLVMStructType::getLiteral(ctx, opTypes);
      auto callee = module.lookupSymbol<LLVM::LLVMFuncOp>(*call.getCallee());
      LLVM::LLVMFuncOp thunk =
          getOrCreateThunk(module, callee, ctxTy, opTypes, ptrTy, voidTy);

      Value one =
          LLVM::ConstantOp::create(r, loc, i64Ty, r.getI64IntegerAttr(1));
      Value ctxPtr = LLVM::AllocaOp::create(r, loc, ptrTy, ctxTy, one);
      for (auto [i, operand] : llvm::enumerate(operands)) {
        Value gep = LLVM::GEPOp::create(
            r, loc, ptrTy, ctxTy, ctxPtr,
            ArrayRef<LLVM::GEPArg>{LLVM::GEPArg(0),
                                   LLVM::GEPArg(static_cast<int32_t>(i))});
        LLVM::StoreOp::create(r, loc, operand, gep);
      }
      Value fp = LLVM::AddressOfOp::create(r, loc, thunk);
      LLVM::CallOp::create(r, loc, spawnFn, ValueRange{sched, fp, ctxPtr});
    }

    // Block until every fiber has finished, then tear the scheduler down.
    r.setInsertionPointAfter(last);
    LLVM::CallOp::create(r, last.getLoc(), joinFn, ValueRange{sched});
    LLVM::CallOp::create(r, last.getLoc(), closeFn, ValueRange{sched});

    for (LLVM::CallOp call : llvm::reverse(peCalls))
      r.eraseOp(call);
  }
};
} // namespace
