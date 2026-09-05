# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

add_mlir_library(MLIRAlloTranslation
  VivadoHLSEmitter.cpp
  VerilogEmitter.cpp

  DEPENDS
  MLIRAlloIncGen

  LINK_LIBS PUBLIC
  MLIRAllo
  MLIRReconcileUnrealizedCasts
  CIRCTSV
  CIRCTSeqTransforms
  CIRCTSeqToSV
  CIRCTHWToSV
  CIRCTVerifToSV
  CIRCTExportVerilog
)
