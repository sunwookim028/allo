/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_OPS_H
#define ALLO_OPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
// ISA relayout ops expose getReassociationIndices() (ReassociationIndices).
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"

#include "allo/IR/AlloAttrs.h"
#include "allo/IR/AlloTypes.h"
#include "llvm/ADT/TypeSwitch.h"

#include "allo/IR/AlloDialect.h.inc"

#include "allo/IR/AlloOpInterfaces.h.inc"
#include "allo/IR/AlloISAOpInterfaces.h.inc"

#define GET_OP_CLASSES
#include "allo/IR/AlloOps.h.inc"

#define GET_OP_CLASSES
#include "allo/IR/AlloISAOps.h.inc"

namespace mlir::allo {
constexpr llvm::StringLiteral kAlloSignedAttr = "allo.signed";
constexpr llvm::StringLiteral kAlloLazyAttr = "allo.lazy";
constexpr llvm::StringLiteral kAlloAsyncAttr = "allo.async";
constexpr llvm::StringLiteral kMemoryInitAttr = "allo.mem.init";
/// On a `seq.hlmem`: no two of its write ports ever write one word in the same
/// cycle, so each may be emitted in its OWN `always` block, the only shape a
/// true-dual-port block RAM infers from.
constexpr llvm::StringLiteral kIndependentWritesAttr =
    "allo.mem.independent_writes";
/// On a `seq.read`/`seq.write`: which physical port of the memory it drives.
/// Accesses sharing a port never issue in the same cycle and are emitted in one
/// `always` block, so a read and a write on one port take one port of a
/// dual-port RAM.
constexpr llvm::StringLiteral kMemPortAttr = "allo.mem.port";
/// On a `seq.hlmem`: the vendor attribute pinning the array to the structure
/// the device priced it as (`ram_style` on Xilinx parts).
constexpr llvm::StringLiteral kRamStyleAttr = "allo.mem.ram_style";
} // namespace mlir::allo

#endif // ALLO_OPS_H
