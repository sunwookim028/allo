/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Transforms/Passes.h"

#include "allo/IR/AlloOps.h"
#include "allo/Support/TopologyGraph.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"

namespace mlir::allo {
#define GEN_PASS_DEF_MATERIALIZETOPOLOGYPASS
#include "allo/Transforms/Passes.h.inc"
} // namespace mlir::allo

using namespace mlir;
using namespace mlir::allo;

static bool isRankedStream(Type type) {
  auto stream = dyn_cast<StreamType>(type);
  return stream && !stream.getShape().empty();
}

namespace {
struct PortDemand {
  unsigned sourceArgNo = 0;
  SmallVector<int64_t, 4> lane;
};

struct PortInfo {
  unsigned sourceArgNo = 0;
  BlockArgument sourceArg;
  BlockArgument newArg;
  SmallVector<int64_t, 4> lane;
};

struct InputInfo {
  unsigned sourceArgNo;
  SmallVector<int64_t, 4> lane;
  bool scalarPort = false;
};

struct KernelMaterialization {
  DenseMap<BlockArgument, unsigned> originalArgNos;
  SmallVector<PortInfo, 8> ports;
  SmallVector<BlockArgument, 8> keptOriginalArgs;
};

using KernelPortPlan = SmallVector<PortDemand, 8>;
using KernelPortPlanMap = DenseMap<KernelOp, KernelPortPlan>;
using KernelMaterializationMap = DenseMap<KernelOp, KernelMaterialization>;
using KernelInfoMap = DenseMap<StringAttr, SmallVector<InputInfo, 8>>;
using BlockedStreamSet = DenseSet<Value>;
using RetainedArgsMap = DenseMap<KernelOp, DenseSet<unsigned>>;
} // namespace

static FailureOr<SmallVector<int64_t, 4>> getStaticLane(ValueRange indices) {
  SmallVector<int64_t, 4> lane;
  for (Value index : indices) {
    IntegerAttr::ValueType cst;
    if (!matchPattern(index, m_ConstantInt(&cst)))
      return failure();
    lane.push_back(cst.getSExtValue());
  }
  return lane;
}

static bool sameLane(ArrayRef<int64_t> lhs, ArrayRef<int64_t> rhs) {
  return lhs.size() == rhs.size() && llvm::equal(lhs, rhs);
}

static bool addPortDemand(KernelPortPlan &plan, unsigned sourceArgNo,
                          ArrayRef<int64_t> lane) {
  if (llvm::any_of(plan, [&](PortDemand &demand) {
        return demand.sourceArgNo == sourceArgNo && sameLane(demand.lane, lane);
      }))
    return false;

  PortDemand demand;
  demand.sourceArgNo = sourceArgNo;
  llvm::append_range(demand.lane, lane);
  plan.push_back(std::move(demand));
  return true;
}

static bool hasPortDemand(KernelPortPlanMap &plans, KernelOp kernel,
                          unsigned sourceArgNo, ArrayRef<int64_t> lane) {
  auto it = plans.find(kernel);
  if (it == plans.end())
    return false;
  return llvm::any_of(it->second, [&](PortDemand &demand) {
    return demand.sourceArgNo == sourceArgNo && sameLane(demand.lane, lane);
  });
}

static bool removePortDemand(KernelPortPlan &plan, PortDemand &demand) {
  auto oldSize = plan.size();
  llvm::erase_if(plan, [&](PortDemand &candidate) {
    return candidate.sourceArgNo == demand.sourceArgNo &&
           sameLane(candidate.lane, demand.lane);
  });
  return oldSize != plan.size();
}

static void sortPortPlan(KernelPortPlan &plan) {
  llvm::sort(plan, [](PortDemand &a, PortDemand &b) {
    return a.sourceArgNo < b.sourceArgNo ||
           (a.sourceArgNo == b.sourceArgNo && a.lane < b.lane);
  });
}

static bool isPublicKernel(KernelOp kernel) {
  auto visibility = kernel.getSymVisibility();
  return !visibility || *visibility == "public";
}

static bool canChangeAbi(KernelOp kernel, bool allowPublicAbiChange) {
  return allowPublicAbiChange || !isPublicKernel(kernel);
}

static bool hasRankedStreamArgs(KernelOp kernel) {
  return llvm::any_of(kernel.getFunctionType().getInputs(), isRankedStream);
}

static bool getStreamAndIndices(Operation *op, Value &stream,
                                ValueRange &indices) {
  if (auto get = dyn_cast<StreamGetOp>(op)) {
    stream = get.getStream();
    indices = get.getIndices();
    return true;
  }
  if (auto put = dyn_cast<StreamPutOp>(op)) {
    stream = put.getStream();
    indices = put.getIndices();
    return true;
  }
  return false;
}

static void collectBlockedStreams(ModuleOp module,
                                  BlockedStreamSet &blockedStreams) {
  module.walk([&](Operation *op) {
    Value stream;
    ValueRange indices;
    if (!getStreamAndIndices(op, stream, indices))
      return;
    if (!isRankedStream(stream.getType()))
      return;
    if (failed(getStaticLane(indices)))
      blockedStreams.insert(stream);
  });
}

// "Blocked" must be a property of the whole channel, not of a single accessor:
// if any process accesses a stream-array channel with a dynamic lane, every
// other accessor must keep the array too, otherwise the static-index siblings
// get fresh scalar streams and silently disconnect from it. The fixpoint gives
// a channel's `stream.create` and all its bound arguments one shared status.
static void propagateBlockedStreams(ModuleOp module,
                                    SymbolTableCollection &symbols,
                                    BlockedStreamSet &blockedStreams) {
  SmallVector<InvokeOp, 8> invokes;
  module.walk([&](InvokeOp invoke) { invokes.push_back(invoke); });

  bool changed = true;
  while (changed) {
    changed = false;
    for (InvokeOp invoke : invokes) {
      auto callee = symbols.lookupNearestSymbolFrom<KernelOp>(
          invoke, invoke.getCalleeAttr());
      if (!callee || callee.getBody().empty())
        continue;
      Block &entry = callee.getBody().front();
      unsigned numEdges =
          std::min<unsigned>(invoke->getNumOperands(), entry.getNumArguments());
      for (unsigned i = 0; i < numEdges; ++i) {
        Value operand = invoke->getOperand(i);
        if (!isRankedStream(operand.getType()))
          continue;
        BlockArgument calleeArg = entry.getArgument(i);
        if (!blockedStreams.contains(operand) &&
            !blockedStreams.contains(calleeArg))
          continue;
        changed |= blockedStreams.insert(operand).second;
        changed |= blockedStreams.insert(calleeArg).second;
      }
    }
  }
}

static LogicalResult collectDirectPortDemands(KernelOp kernel,
                                              KernelPortPlanMap &plans,
                                              BlockedStreamSet &blockedStreams,
                                              bool allowPublicAbiChange) {
  if (kernel.getBody().empty())
    return success();
  if (!canChangeAbi(kernel, allowPublicAbiChange))
    return success();

  kernel.walk([&](Operation *op) {
    Value stream;
    ValueRange indices;
    if (!getStreamAndIndices(op, stream, indices))
      return;

    auto arg = dyn_cast<BlockArgument>(stream);
    if (!arg || arg.getOwner()->getParentOp() != kernel)
      return;
    if (!isRankedStream(arg.getType()))
      return;
    if (blockedStreams.contains(arg))
      return;

    auto laneOr = getStaticLane(indices);
    if (failed(laneOr))
      return;
    addPortDemand(plans[kernel], arg.getArgNumber(), *laneOr);
  });
  return success();
}

static LogicalResult collectKernelPortPlans(ModuleOp module,
                                            SymbolTableCollection &symbols,
                                            KernelPortPlanMap &plans,
                                            BlockedStreamSet &blockedStreams,
                                            bool allowPublicAbiChange) {
  SmallVector<TopologyGraph, 8> graphs;
  for (auto kernel : module.getOps<KernelOp>()) {
    if (failed(collectDirectPortDemands(kernel, plans, blockedStreams,
                                        allowPublicAbiChange)))
      return failure();
    if (kernel.getBody().empty())
      continue;

    auto graphOr =
        buildTopologyGraph(kernel, symbols, /*skipDynamicLanes=*/true);
    if (failed(graphOr))
      return failure();
    graphs.push_back(std::move(*graphOr));
  }

  bool changed = true;
  while (changed) {
    changed = false;
    for (TopologyGraph &graph : graphs) {
      for (ProcessNode &node : graph.nodes) {
        auto it = plans.find(node.callee);
        if (it == plans.end())
          continue;

        SmallVector<PortDemand, 8> calleeDemands(it->second.begin(),
                                                 it->second.end());
        for (PortDemand &demand : calleeDemands) {
          assert(demand.sourceArgNo < node.invoke->getNumOperands() &&
                 "port demand must reference a callee argument");
          Value source = node.invoke->getOperand(demand.sourceArgNo);
          if (!isRankedStream(source.getType()))
            return node.invoke.emitError(
                "scalarized stream source must be a ranked stream");
          if (blockedStreams.contains(source))
            continue;

          if (auto arg = dyn_cast<BlockArgument>(source)) {
            if (arg.getOwner()->getParentOp() != graph.scope)
              return node.invoke.emitError(
                  "scalarized boundary stream source must be a kernel "
                  "argument");
            if (canChangeAbi(graph.scope, allowPublicAbiChange))
              changed |= addPortDemand(plans[graph.scope], arg.getArgNumber(),
                                       demand.lane);
            continue;
          }

          if (source.getDefiningOp<StreamCreateOp>())
            continue;

          return node.invoke.emitError(
              "scalarized stream source must be a kernel argument or "
              "stream.create");
        }
      }
    }
  }

  changed = true;
  while (changed) {
    changed = false;
    for (TopologyGraph &graph : graphs) {
      for (ProcessNode &node : graph.nodes) {
        auto it = plans.find(node.callee);
        if (it == plans.end())
          continue;

        SmallVector<PortDemand, 8> calleeDemands(it->second.begin(),
                                                 it->second.end());
        for (PortDemand &demand : calleeDemands) {
          assert(demand.sourceArgNo < node.invoke->getNumOperands() &&
                 "port demand must reference a callee argument");
          Value source = node.invoke->getOperand(demand.sourceArgNo);
          if (!isRankedStream(source.getType()))
            return node.invoke.emitError(
                "scalarized stream source must be a ranked stream");
          if (blockedStreams.contains(source)) {
            changed |= removePortDemand(it->second, demand);
            continue;
          }

          if (auto arg = dyn_cast<BlockArgument>(source)) {
            if (arg.getOwner()->getParentOp() != graph.scope)
              return node.invoke.emitError(
                  "scalarized boundary stream source must be a kernel "
                  "argument");
            if (!hasPortDemand(plans, graph.scope, arg.getArgNumber(),
                               demand.lane))
              changed |= removePortDemand(it->second, demand);
            continue;
          }

          if (source.getDefiningOp<StreamCreateOp>())
            continue;

          return node.invoke.emitError(
              "scalarized stream source must be a kernel argument or "
              "stream.create");
        }
      }
    }
  }

  for (auto &it : plans)
    sortPortPlan(it.second);
  return success();
}

static unsigned getOriginalArgNo(KernelMaterialization &state,
                                 BlockArgument arg) {
  auto it = state.originalArgNos.find(arg);
  assert(it != state.originalArgNos.end() &&
         "argument must belong to the original kernel signature");
  return it->second;
}

static LogicalResult collectRetainedRankedArgs(ModuleOp module,
                                               SymbolTableCollection &symbols,
                                               KernelPortPlanMap &plans,
                                               BlockedStreamSet &blockedStreams,
                                               RetainedArgsMap &retainedArgs) {
  for (auto kernel : module.getOps<KernelOp>()) {
    if (kernel.getBody().empty())
      continue;

    for (BlockArgument arg : kernel.getBody().front().getArguments()) {
      if (!isRankedStream(arg.getType()))
        continue;

      unsigned argNo = arg.getArgNumber();
      if (blockedStreams.contains(arg)) {
        retainedArgs[kernel].insert(argNo);
        continue;
      }

      for (OpOperand &use : arg.getUses()) {
        Operation *owner = use.getOwner();
        Value stream;
        ValueRange indices;
        if (getStreamAndIndices(owner, stream, indices) && stream == arg) {
          auto laneOr = getStaticLane(indices);
          if (failed(laneOr) || !hasPortDemand(plans, kernel, argNo, *laneOr))
            retainedArgs[kernel].insert(argNo);
          continue;
        }

        if (isa<InvokeOp>(owner))
          continue;
        retainedArgs[kernel].insert(argNo);
      }
    }
  }

  SmallVector<TopologyGraph, 8> graphs;
  for (auto kernel : module.getOps<KernelOp>()) {
    if (kernel.getBody().empty())
      continue;
    auto graphOr =
        buildTopologyGraph(kernel, symbols, /*skipDynamicLanes=*/true);
    if (failed(graphOr))
      return failure();
    graphs.push_back(std::move(*graphOr));
  }

  bool changed = true;
  while (changed) {
    changed = false;
    for (TopologyGraph &graph : graphs) {
      for (ProcessNode &node : graph.nodes) {
        auto it = retainedArgs.find(node.callee);
        if (it == retainedArgs.end())
          continue;

        SmallVector<unsigned, 8> argNos(it->second.begin(), it->second.end());
        for (unsigned argNo : argNos) {
          assert(argNo < node.invoke->getNumOperands() &&
                 "retained argument must reference a callee argument");
          Value source = node.invoke->getOperand(argNo);
          if (auto arg = dyn_cast<BlockArgument>(source)) {
            if (arg.getOwner()->getParentOp() != graph.scope)
              return node.invoke.emitError(
                  "retained boundary stream source must be a kernel argument");
            if (isRankedStream(arg.getType()))
              changed |=
                  retainedArgs[graph.scope].insert(arg.getArgNumber()).second;
            continue;
          }
        }
      }
    }
  }
  return success();
}

static PortInfo *findPort(SmallVectorImpl<PortInfo> &ports,
                          BlockArgument sourceArg, ArrayRef<int64_t> lane) {
  auto *it = llvm::find_if(ports, [&](PortInfo &port) {
    return port.sourceArg == sourceArg && sameLane(port.lane, lane);
  });
  if (it == ports.end())
    return nullptr;
  return it;
}

static PortInfo *findNewPort(SmallVectorImpl<PortInfo> &ports,
                             BlockArgument newArg) {
  auto *it = llvm::find_if(
      ports, [&](PortInfo &port) { return port.newArg == newArg; });
  if (it == ports.end())
    return nullptr;
  return it;
}

static bool isKeptOriginalArg(KernelMaterialization &state, BlockArgument arg) {
  return llvm::is_contained(state.keptOriginalArgs, arg);
}

static LogicalResult rewriteStreamOps(IRRewriter &rewriter, KernelOp kernel,
                                      SmallVectorImpl<PortInfo> &ports) {
  WalkResult walkResult = kernel->walk([&](Operation *op) {
    BlockArgument arg;
    ValueRange indices;
    if (auto get = dyn_cast<StreamGetOp>(op)) {
      arg = dyn_cast<BlockArgument>(get.getStream());
      indices = get.getIndices();
    } else if (auto put = dyn_cast<StreamPutOp>(op)) {
      arg = dyn_cast<BlockArgument>(put.getStream());
      indices = put.getIndices();
    } else {
      return WalkResult::advance();
    }
    if (!arg || arg.getOwner()->getParentOp() != kernel)
      return WalkResult::advance();
    if (!isRankedStream(arg.getType()))
      return WalkResult::advance();

    auto laneOr = getStaticLane(indices);
    if (failed(laneOr))
      return WalkResult::advance();
    PortInfo *port = findPort(ports, arg, *laneOr);
    if (!port)
      return WalkResult::advance();
    BlockArgument portArg = port->newArg;
    rewriter.setInsertionPoint(op);
    if (auto get = dyn_cast<StreamGetOp>(op)) {
      auto newGet = StreamGetOp::create(rewriter, op->getLoc(), portArg,
                                        ArrayRef<Value>{});
      rewriter.replaceOp(get, newGet);
      return WalkResult::advance();
    }
    auto put = cast<StreamPutOp>(op);
    StreamPutOp::create(rewriter, op->getLoc(), portArg, ValueRange{},
                        put.getValue());
    rewriter.eraseOp(put);
    return WalkResult::advance();
  });
  return failure(walkResult.wasInterrupted());
}

static void buildKernelInputInfo(KernelOp kernel, KernelMaterialization &state,
                                 DenseSet<unsigned> &retainedArgs,
                                 KernelInfoMap &infos) {
  SmallVector<InputInfo, 8> inputs;
  Block &entry = kernel.getBody().front();
  for (BlockArgument arg : entry.getArguments()) {
    if (PortInfo *port = findNewPort(state.ports, arg)) {
      InputInfo input;
      input.sourceArgNo = port->sourceArgNo;
      llvm::append_range(input.lane, port->lane);
      input.scalarPort = true;
      inputs.push_back(std::move(input));
      continue;
    }

    unsigned sourceArgNo = getOriginalArgNo(state, arg);
    if (isRankedStream(arg.getType()) && !retainedArgs.contains(sourceArgNo))
      continue;
    if (arg.use_empty())
      continue;
    state.keptOriginalArgs.push_back(arg);

    InputInfo input;
    input.sourceArgNo = sourceArgNo;
    inputs.push_back(std::move(input));
  }
  infos[kernel.getSymNameAttr()] = std::move(inputs);
}

static LogicalResult insertScalarPortsAndRewriteDirectUses(
    IRRewriter &rewriter, KernelOp kernel, KernelPortPlan &plan,
    DenseSet<unsigned> &retainedArgs, KernelMaterialization &state,
    KernelInfoMap &infos) {
  if (!hasRankedStreamArgs(kernel))
    return success();
  if (!llvm::all_of(kernel.getMapping(), [](int64_t x) { return x == 1; }))
    return kernel.emitError(
        "materialize-topology only supports kernels with identity mapping");

  Block &entry = kernel.getBody().front();
  for (BlockArgument arg : entry.getArguments())
    state.originalArgNos[arg] = arg.getArgNumber();

  for (PortDemand &demand : plan) {
    assert(demand.sourceArgNo < entry.getNumArguments() &&
           "port demand must reference an original kernel argument");
    BlockArgument sourceArg = entry.getArgument(demand.sourceArgNo);
    assert(isRankedStream(sourceArg.getType()) &&
           "scalarized port must originate from a ranked stream");

    PortInfo port;
    port.sourceArgNo = demand.sourceArgNo;
    port.sourceArg = sourceArg;
    llvm::append_range(port.lane, demand.lane);
    state.ports.push_back(std::move(port));
  }

  unsigned shift = 0;
  for (PortInfo &port : state.ports) {
    auto oldType = cast<StreamType>(port.sourceArg.getType());
    assert(!oldType.getShape().empty() &&
           "scalarized port must originate from a ranked stream");
    // Inherit the source ranked stream argument's location (its NameLoc) so
    // codegen keeps the name.
    auto scalarType = StreamType::get(
        kernel.getContext(), oldType.getBaseType(), oldType.getDepth(), {});
    port.newArg = entry.insertArgument(port.sourceArgNo + shift, scalarType,
                                       port.sourceArg.getLoc());
    shift++;
  }

  if (failed(rewriteStreamOps(rewriter, kernel, state.ports)))
    return failure();

  buildKernelInputInfo(kernel, state, retainedArgs, infos);
  return success();
}

static LogicalResult finalizeKernelSignature(IRRewriter &rewriter,
                                             KernelOp kernel,
                                             KernelMaterialization &state) {
  Block &entry = kernel.getBody().front();
  ArrayAttr oldArgAttrs = kernel.getArgAttrsAttr();
  // Rebuild the signedness marker in lockstep with the new argument list.
  auto markerAttr = kernel->getAttrOfType<StringAttr>(kAlloSignedAttr);
  StringRef oldMarker = markerAttr ? markerAttr.getValue() : StringRef();
  unsigned numOldInputs = kernel.getFunctionType().getNumInputs();
  auto markerCharAt = [&](unsigned i) {
    return i < oldMarker.size() ? oldMarker[i] : 'x';
  };
  std::string newMarker;

  BitVector toErase(entry.getNumArguments());
  SmallVector<Type, 8> newInputs;
  SmallVector<Attribute> newArgAttrs;
  for (BlockArgument arg : entry.getArguments()) {
    if (PortInfo *port = findNewPort(state.ports, arg)) {
      newInputs.push_back(arg.getType());
      if (oldArgAttrs)
        newArgAttrs.push_back(rewriter.getDictionaryAttr({}));
      newMarker.push_back(markerCharAt(port->sourceArgNo));
      continue;
    }

    if (isKeptOriginalArg(state, arg)) {
      newInputs.push_back(arg.getType());
      unsigned sourceArgNo = getOriginalArgNo(state, arg);
      if (oldArgAttrs) {
        assert(sourceArgNo < oldArgAttrs.size() &&
               "arg_attrs must match the old function type");
        newArgAttrs.push_back(oldArgAttrs[sourceArgNo]);
      }
      newMarker.push_back(markerCharAt(sourceArgNo));
      continue;
    }

    if (isRankedStream(arg.getType())) {
      if (!arg.use_empty())
        return kernel.emitError("failed to eliminate ranked stream argument");
      toErase.set(arg.getArgNumber());
      continue;
    }

    toErase.set(arg.getArgNumber());
  }

  entry.eraseArguments(toErase);
  auto newType = FunctionType::get(kernel.getContext(), newInputs,
                                   kernel.getFunctionType().getResults());
  kernel.setFunctionType(newType);
  if (oldArgAttrs)
    kernel->setAttr(kernel.getArgAttrsAttrName(),
                    rewriter.getArrayAttr(newArgAttrs));
  if (markerAttr) {
    if (numOldInputs <= oldMarker.size())
      newMarker.append(oldMarker.substr(numOldInputs).str());
    kernel->setAttr(kAlloSignedAttr, rewriter.getStringAttr(newMarker));
  }
  return success();
}

static std::string getLaneKey(ArrayRef<int64_t> lane) {
  std::string key;
  for (int64_t index : lane) {
    if (!key.empty())
      key += ".";
    key += std::to_string(index);
  }
  return key;
}

static Value getOrCreateScalarStream(
    Value rankedStream, ArrayRef<int64_t> lane, InvokeOp invoke,
    IRRewriter &rewriter,
    DenseMap<Value, llvm::StringMap<Value>> &scalarStreams) {
  auto &streamsByLane = scalarStreams[rankedStream];
  std::string key = getLaneKey(lane);
  if (auto it = streamsByLane.find(key); it != streamsByLane.end())
    return it->second;

  auto rankedType = cast<StreamType>(rankedStream.getType());
  auto scalarType = StreamType::get(
      invoke.getContext(), rankedType.getBaseType(), rankedType.getDepth(), {});

  OpBuilder::InsertionGuard guard(rewriter);
  auto sourceCreate = rankedStream.getDefiningOp<StreamCreateOp>();
  if (sourceCreate)
    rewriter.setInsertionPointAfter(sourceCreate);
  else
    rewriter.setInsertionPoint(invoke);
  // Inherit the ranked stream's location (its NameLoc) so the materialized
  // scalar channel keeps the source name in codegen.
  auto scalarCreate =
      StreamCreateOp::create(rewriter, rankedStream.getLoc(), scalarType);
  // The scalar lane carries the ranked stream's payload, hence its signedness.
  if (sourceCreate)
    if (auto sgn = sourceCreate->getAttrOfType<StringAttr>(kAlloSignedAttr))
      scalarCreate->setAttr(kAlloSignedAttr, sgn);
  Value scalarStream = scalarCreate.getResult();
  streamsByLane.insert({key, scalarStream});
  return scalarStream;
}

static LogicalResult rewriteMaterializedStreamUses(
    IRRewriter &rewriter, ModuleOp module,
    DenseMap<Value, llvm::StringMap<Value>> &scalarStreams) {
  SmallVector<Operation *, 8> streamOps;
  module.walk([&](Operation *op) {
    if (isa<StreamGetOp, StreamPutOp>(op))
      streamOps.push_back(op);
  });

  for (Operation *op : streamOps) {
    Value stream;
    ValueRange indices;
    if (auto get = dyn_cast<StreamGetOp>(op)) {
      stream = get.getStream();
      indices = get.getIndices();
    } else {
      auto put = cast<StreamPutOp>(op);
      stream = put.getStream();
      indices = put.getIndices();
    }

    if (!isRankedStream(stream.getType()))
      continue;
    auto streamsIt = scalarStreams.find(stream);
    if (streamsIt == scalarStreams.end())
      continue;

    auto laneOr = getStaticLane(indices);
    if (failed(laneOr))
      continue;

    auto scalarIt = streamsIt->second.find(getLaneKey(*laneOr));
    if (scalarIt == streamsIt->second.end())
      continue;

    Value scalarStream = scalarIt->second;
    rewriter.setInsertionPoint(op);
    if (auto get = dyn_cast<StreamGetOp>(op)) {
      auto newGet = StreamGetOp::create(rewriter, op->getLoc(), scalarStream,
                                        ArrayRef<Value>{});
      rewriter.replaceOp(get, newGet);
      continue;
    }

    auto put = cast<StreamPutOp>(op);
    StreamPutOp::create(rewriter, op->getLoc(), scalarStream, ValueRange{},
                        put.getValue());
    rewriter.eraseOp(put);
  }
  return success();
}

static bool isEmptyKernel(KernelOp kernel) {
  if (kernel.getFunctionType().getNumInputs() != 0 ||
      kernel.getFunctionType().getNumResults() != 0)
    return false;
  if (!llvm::hasSingleElement(kernel.getBody()))
    return false;
  Block &entry = kernel.getBody().front();
  return llvm::hasSingleElement(entry) && isa<ReturnOp>(entry.front());
}

static LogicalResult rewriteInvokes(ModuleOp module, KernelInfoMap &kernelInfos,
                                    KernelMaterializationMap &materializations,
                                    IRRewriter &rewriter) {
  DenseMap<Value, llvm::StringMap<Value>> scalarStreams;
  SmallVector<InvokeOp> invokes;
  module.walk([&](InvokeOp invoke) {
    if (kernelInfos.contains(invoke.getCalleeAttr().getAttr()))
      invokes.push_back(invoke);
  });

  for (InvokeOp invoke : invokes) {
    StringAttr callee = invoke.getCalleeAttr().getAttr();
    auto it = kernelInfos.find(callee);
    assert(it != kernelInfos.end() && "invoke must reference rewritten kernel");

    ArrayAttr oldArgAttrs = invoke.getArgAttrsAttr();
    SmallVector<Value> newOperands;
    SmallVector<Attribute> newArgAttrs;
    for (InputInfo &input : it->second) {
      assert(input.sourceArgNo < invoke->getNumOperands() &&
             "rewritten port must reference an old invoke operand");
      Value source = invoke->getOperand(input.sourceArgNo);
      if (!input.scalarPort) {
        newOperands.push_back(source);
        if (oldArgAttrs) {
          assert(input.sourceArgNo < oldArgAttrs.size() &&
                 "arg_attrs must match the old invoke operands");
          newArgAttrs.push_back(oldArgAttrs[input.sourceArgNo]);
        }
        continue;
      }

      if (!isRankedStream(source.getType()))
        return invoke.emitError("scalarized stream port source must be a "
                                "ranked stream operand");

      if (auto arg = dyn_cast<BlockArgument>(source)) {
        auto parent = invoke->getParentOfType<KernelOp>();
        assert(parent && "invoke must be nested in a kernel");
        if (arg.getOwner()->getParentOp() != parent)
          return invoke.emitError(
              "scalarized boundary stream source must be a kernel argument");

        auto materializedIt = materializations.find(parent);
        if (materializedIt == materializations.end())
          return invoke.emitError(
              "missing scalarized boundary stream port in caller");
        PortInfo *port =
            findPort(materializedIt->second.ports, arg, input.lane);
        if (!port)
          return invoke.emitError(
              "missing scalarized boundary stream lane in caller");
        newOperands.push_back(port->newArg);
      } else if (source.getDefiningOp<StreamCreateOp>()) {
        newOperands.push_back(getOrCreateScalarStream(
            source, input.lane, invoke, rewriter, scalarStreams));
      } else {
        return invoke.emitError("scalarized stream source must be a kernel "
                                "argument or stream.create");
      }
      if (oldArgAttrs)
        newArgAttrs.push_back(rewriter.getDictionaryAttr({}));
    }

    rewriter.modifyOpInPlace(invoke, [&]() {
      invoke->setOperands(newOperands);
      if (oldArgAttrs)
        invoke->setAttr(invoke.getArgAttrsAttrName(),
                        rewriter.getArrayAttr(newArgAttrs));
    });
  }

  if (failed(rewriteMaterializedStreamUses(rewriter, module, scalarStreams)))
    return failure();
  return success();
}

static void cleanupMaterializedTopology(ModuleOp module,
                                        KernelInfoMap &kernelInfos,
                                        IRRewriter &rewriter) {
  DenseSet<StringAttr> rewrittenKernels;
  for (auto &it : kernelInfos)
    rewrittenKernels.insert(it.first);

  DenseSet<StringAttr> emptyKernels;
  for (auto kernel : module.getOps<KernelOp>())
    if (rewrittenKernels.contains(kernel.getSymNameAttr()) &&
        isEmptyKernel(kernel))
      emptyKernels.insert(kernel.getSymNameAttr());

  SmallVector<InvokeOp> invokesToErase;
  module.walk([&](InvokeOp invoke) {
    if (emptyKernels.contains(invoke.getCalleeAttr().getAttr()))
      invokesToErase.push_back(invoke);
  });
  for (InvokeOp invoke : invokesToErase) {
    assert(invoke->getNumResults() == 0 &&
           "empty kernels cannot produce results");
    rewriter.eraseOp(invoke);
  }

  SmallVector<KernelOp> kernelsToErase;
  for (auto kernel : module.getOps<KernelOp>())
    if (emptyKernels.contains(kernel.getSymNameAttr()) &&
        kernel.symbolKnownUseEmpty(module))
      kernelsToErase.push_back(kernel);
  for (KernelOp kernel : kernelsToErase)
    rewriter.eraseOp(kernel);

  SmallVector<StreamCreateOp> streamsToErase;
  module.walk([&](StreamCreateOp create) {
    if (isRankedStream(create.getStream().getType()) &&
        create.getStream().use_empty())
      streamsToErase.push_back(create);
  });
  for (StreamCreateOp create : streamsToErase)
    rewriter.eraseOp(create);
}

namespace {
struct MaterializeTopologyPass
    : public allo::impl::MaterializeTopologyPassBase<MaterializeTopologyPass> {
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();
    SymbolTableCollection symbols;
    BlockedStreamSet blockedStreams;
    collectBlockedStreams(module, blockedStreams);
    propagateBlockedStreams(module, symbols, blockedStreams);

    KernelPortPlanMap portPlans;
    if (failed(collectKernelPortPlans(module, symbols, portPlans,
                                      blockedStreams, allowPublicAbiChange)))
      return signalPassFailure();

    RetainedArgsMap retainedArgs;
    if (failed(collectRetainedRankedArgs(module, symbols, portPlans,
                                         blockedStreams, retainedArgs)))
      return signalPassFailure();

    KernelInfoMap kernelInfos;
    KernelMaterializationMap materializations;
    IRRewriter rewriter(context);
    for (auto kernel : module.getOps<KernelOp>()) {
      if (!hasRankedStreamArgs(kernel))
        continue;
      auto &state = materializations[kernel];
      if (failed(insertScalarPortsAndRewriteDirectUses(
              rewriter, kernel, portPlans[kernel], retainedArgs[kernel], state,
              kernelInfos)))
        return signalPassFailure();
    }

    if (failed(rewriteInvokes(module, kernelInfos, materializations, rewriter)))
      return signalPassFailure();

    for (auto kernel : module.getOps<KernelOp>()) {
      auto it = materializations.find(kernel);
      if (it == materializations.end())
        continue;
      if (failed(finalizeKernelSignature(rewriter, kernel, it->second)))
        return signalPassFailure();
    }

    cleanupMaterializedTopology(module, kernelInfos, rewriter);
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, allo::AlloDialect>();
  }
};
} // namespace
