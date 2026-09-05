/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/InitAllDialects.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"

#include "allo/IR/AlloOps.h"

void mlir::allo::registerAllDialects(DialectRegistry &registry) {
  // clang-format off
  registry.insert<
    affine::AffineDialect,
    arith::ArithDialect,
    bufferization::BufferizationDialect,
    cf::ControlFlowDialect,
    func::FuncDialect,
    index::IndexDialect,
    linalg::LinalgDialect,
    LLVM::LLVMDialect,
    math::MathDialect,
    memref::MemRefDialect,
    scf::SCFDialect,
    tensor::TensorDialect,
    tosa::TosaDialect,
    vector::VectorDialect,
    transform::TransformDialect,
    circt::comb::CombDialect,
    circt::hw::HWDialect,
    circt::seq::SeqDialect,
    allo::AlloDialect
  >();
  // clang-format on
}
