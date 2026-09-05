/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo-c/Schedule.h"

#include "mlir/CAPI/IR.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <unordered_set>

using namespace mlir;

// Keep in sync with the ScheduleOpTrait IntFlag mirrored on the Python side.
enum ScheduleOpTrait : uint64_t {
  ScheduleOpTraitLoopLike = 1ULL << 0,
  ScheduleOpTraitAffineLoop = 1ULL << 1,
  ScheduleOpTraitScfLoop = 1ULL << 2,
  ScheduleOpTraitRegionBranch = 1ULL << 3,
  ScheduleOpTraitFunctionLike = 1ULL << 4,
  ScheduleOpTraitSymbol = 1ULL << 5,
  ScheduleOpTraitMemoryAllocate = 1ULL << 6,
  ScheduleOpTraitMemoryFree = 1ULL << 7,
  ScheduleOpTraitMemoryRead = 1ULL << 8,
  ScheduleOpTraitMemoryWrite = 1ULL << 9,
  ScheduleOpTraitAffineFor = 1ULL << 10,
};

static bool isScheduled(Operation *op) {
  return op != nullptr && !op->hasTrait<OpTrait::IsTerminator>();
}

static void collectScheduledOps(Operation *op,
                                SmallVectorImpl<Operation *> &out) {
  if (!isScheduled(op))
    return;
  out.push_back(op);
  for (Region &region : op->getRegions())
    for (Block &block : region)
      for (Operation &nested : block)
        collectScheduledOps(&nested, out);
}

static std::string freshScheduleId(uint64_t &counter,
                                   std::unordered_set<std::string> &used) {
  while (true) {
    std::string candidate = "s" + std::to_string(counter++);
    if (used.insert(candidate).second)
      return candidate;
  }
}

static std::string requireScheduleId(Operation *op) {
  auto attr = op->getAttrOfType<StringAttr>(kScheduleIdAttr);
  assert(attr && "operation must be annotated before snapshot collection");
  return attr.str();
}

static std::optional<std::string> nameFromLoc(Location loc) {
  if (auto nameLoc = dyn_cast<NameLoc>(loc))
    return nameLoc.getName().str();
  return std::nullopt;
}

static std::optional<std::string> bestOperationName(Operation *op) {
  if (auto attr = op->getAttrOfType<StringAttr>(kScheduleNameAttr))
    return attr.str();
  if (auto attr =
          op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()))
    return attr.str();
  return nameFromLoc(op->getLoc());
}

static std::optional<std::string> bestValueName(Value value, Operation *owner) {
  if (auto name = nameFromLoc(value.getLoc()))
    return name;
  if (owner)
    return bestOperationName(owner);
  return std::nullopt;
}

static std::optional<FileLineColLoc> findFileLineCol(Location loc) {
  if (auto fileLoc = dyn_cast<FileLineColLoc>(loc))
    return fileLoc;
  if (auto nameLoc = dyn_cast<NameLoc>(loc))
    return findFileLineCol(nameLoc.getChildLoc());
  if (auto callLoc = dyn_cast<CallSiteLoc>(loc)) {
    if (auto calleeLoc = findFileLineCol(callLoc.getCallee()))
      return calleeLoc;
    return findFileLineCol(callLoc.getCaller());
  }
  if (auto fusedLoc = dyn_cast<FusedLoc>(loc)) {
    for (Location child : fusedLoc.getLocations())
      if (auto childLoc = findFileLineCol(child))
        return childLoc;
  }
  return std::nullopt;
}

static llvm::json::Value locationJson(Location loc) {
  auto fileLoc = findFileLineCol(loc);
  if (!fileLoc)
    return nullptr;
  llvm::json::Object out;
  out["file"] = fileLoc->getFilename().str();
  out["line"] = (int64_t)fileLoc->getLine();
  out["col"] = (int64_t)fileLoc->getColumn();
  return out;
}

static std::string typeString(Type type) {
  std::string text;
  llvm::raw_string_ostream os(text);
  type.print(os);
  return os.str();
}

static bool isBufferLike(Value value) {
  return isa<BaseMemRefType, TensorType>(value.getType());
}

static std::string childSegment(Operation *op, unsigned loopIdx,
                                unsigned branchIdx, unsigned opIdx) {
  if (auto attr =
          op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()))
    return attr.str();
  if (isa<LoopLikeOpInterface>(op))
    return "L" + std::to_string(loopIdx);
  if (isa<RegionBranchOpInterface>(op))
    return "B" + std::to_string(branchIdx);
  return "O" + std::to_string(opIdx);
}

static uint64_t memoryEffectTraits(Operation *op) {
  uint64_t traits = 0;
  auto effects = getEffectsRecursively(op);
  if (!effects)
    return traits;
  for (const MemoryEffects::EffectInstance &effect : *effects) {
    if (isa<MemoryEffects::Allocate>(effect.getEffect()))
      traits |= ScheduleOpTraitMemoryAllocate;
    if (isa<MemoryEffects::Free>(effect.getEffect()))
      traits |= ScheduleOpTraitMemoryFree;
    if (isa<MemoryEffects::Read>(effect.getEffect()))
      traits |= ScheduleOpTraitMemoryRead;
    if (isa<MemoryEffects::Write>(effect.getEffect()))
      traits |= ScheduleOpTraitMemoryWrite;
  }
  return traits;
}

static uint64_t operationTraits(Operation *op) {
  uint64_t traits = 0;
  if (isa<LoopLikeOpInterface>(op))
    traits |= ScheduleOpTraitLoopLike;
  if (isa<affine::AffineForOp>(op))
    traits |= ScheduleOpTraitAffineFor;
  if (isa<affine::AffineForOp, affine::AffineParallelOp>(op))
    traits |= ScheduleOpTraitAffineLoop;
  if (isa<scf::ForOp, scf::ParallelOp>(op))
    traits |= ScheduleOpTraitScfLoop;
  if (isa<RegionBranchOpInterface>(op))
    traits |= ScheduleOpTraitRegionBranch;
  if (isa<FunctionOpInterface>(op))
    traits |= ScheduleOpTraitFunctionLike;
  if (isa<SymbolOpInterface>(op))
    traits |= ScheduleOpTraitSymbol;
  traits |= memoryEffectTraits(op);
  return traits;
}

static llvm::json::Object valueJson(Value value, Operation *owner,
                                    llvm::StringRef ownerId,
                                    llvm::StringRef ownerPath, unsigned number,
                                    llvm::StringRef source) {
  llvm::json::Object out;
  out["id"] =
      (llvm::Twine(ownerId) + ":" + source + std::to_string(number)).str();
  out["owner_id"] = ownerId.str();
  if (auto name = bestValueName(value, owner))
    out["name"] = *name;
  else
    out["name"] = nullptr;
  out["type"] = typeString(value.getType());
  out["number"] = (int64_t)number;
  out["source"] = source.str();
  out["path"] =
      (llvm::Twine(ownerPath) + ":" + source + std::to_string(number)).str();
  out["loc"] = locationJson(value.getLoc());
  return out;
}

static void collectValues(Operation *op, llvm::StringRef opId,
                          llvm::StringRef opPath, llvm::json::Array &values) {
  for (OpResult result : op->getResults()) {
    if (!isBufferLike(result))
      continue;
    values.push_back(
        valueJson(result, op, opId, opPath, result.getResultNumber(), "res"));
  }
  auto func = dyn_cast<FunctionOpInterface>(op);
  if (!func)
    return;
  for (BlockArgument arg : func.getArguments()) {
    if (!isBufferLike(arg))
      continue;
    values.push_back(
        valueJson(arg, op, opId, opPath, arg.getArgNumber(), "arg"));
  }
}

static void collectSnapshotNode(Operation *op, llvm::StringRef parentId,
                                llvm::StringRef path, llvm::json::Array &ops,
                                llvm::json::Array &values) {
  std::string id = requireScheduleId(op);
  llvm::json::Array childIds;
  SmallVector<std::pair<Operation *, std::string>> children;
  unsigned loopIdx = 0, branchIdx = 0, opIdx = 0;

  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (Operation &nested : block) {
        if (!isScheduled(&nested))
          continue;
        unsigned currentLoop = loopIdx, currentBranch = branchIdx,
                 currentOp = opIdx;
        if (isa<LoopLikeOpInterface>(nested))
          ++loopIdx;
        else if (isa<RegionBranchOpInterface>(nested))
          ++branchIdx;
        else
          ++opIdx;
        std::string segment =
            childSegment(&nested, currentLoop, currentBranch, currentOp);
        std::string childPath =
            path.empty() ? segment : (llvm::Twine(path) + "/" + segment).str();
        childIds.push_back(requireScheduleId(&nested));
        children.push_back({&nested, childPath});
      }
    }
  }

  llvm::json::Object record;
  record["id"] = id;
  record["kind"] = op->getName().getStringRef().str();
  if (auto name = bestOperationName(op))
    record["name"] = *name;
  else
    record["name"] = nullptr;
  record["path"] = path.str();
  if (parentId.empty())
    record["parent_id"] = nullptr;
  else
    record["parent_id"] = parentId.str();
  record["children"] = std::move(childIds);
  record["loc"] = locationJson(op->getLoc());
  record["traits"] = (int64_t)operationTraits(op);
  ops.push_back(std::move(record));
  collectValues(op, id, path, values);

  for (auto &[child, childPath] : children)
    collectSnapshotNode(child, id, childPath, ops, values);
}

void alloAnnotateScheduleIds(MlirModule module) {
  ModuleOp mod = unwrap(module);
  SmallVector<Operation *> ops;
  collectScheduledOps(mod.getOperation(), ops);

  llvm::DenseMap<Attribute, unsigned> counts;
  for (Operation *op : ops)
    if (auto attr = op->getAttrOfType<StringAttr>(kScheduleIdAttr))
      ++counts[attr];

  std::unordered_set<std::string> used;
  for (Operation *op : ops) {
    auto attr = op->getAttrOfType<StringAttr>(kScheduleIdAttr);
    if (attr && counts[attr] == 1)
      used.insert(attr.str());
  }

  uint64_t counter = 0;
  MLIRContext *ctx = mod.getContext();
  for (Operation *op : ops) {
    auto attr = op->getAttrOfType<StringAttr>(kScheduleIdAttr);
    if (attr && counts[attr] == 1)
      continue;
    op->setAttr(kScheduleIdAttr,
                StringAttr::get(ctx, freshScheduleId(counter, used)));
  }
}

void alloCleanupScheduleIds(MlirModule module) {
  unwrap(module)->walk([](Operation *op) { op->removeAttr(kScheduleIdAttr); });
}

void alloCollectScheduleSnapshotJSON(MlirModule module,
                                     MlirStringCallback callback,
                                     void *userData) {
  Operation *root = unwrap(module).getOperation();
  assert(root->getAttrOfType<StringAttr>(kScheduleIdAttr) &&
         "call alloAnnotateScheduleIds before collecting the snapshot");

  llvm::json::Array ops, values;
  collectSnapshotNode(root, "", "module", ops, values);

  llvm::json::Object out;
  out["root_id"] = requireScheduleId(root);
  out["ops"] = std::move(ops);
  out["values"] = std::move(values);

  std::string text;
  llvm::raw_string_ostream os(text);
  os << llvm::json::Value(std::move(out));
  os.flush();
  callback(MlirStringRef{text.data(), text.size()}, userData);
}
