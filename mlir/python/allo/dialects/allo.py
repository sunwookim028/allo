# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# ODS-generated op classes for the `allo` dialect, plus the dialect/pass
# registration helpers from the `_allo` CAPI extension.
from ._allo_ops_gen import *
from .._mlir_libs._allo.allo import *

# Custom Allo dialect types / attributes (nanobind subclasses of ir.Type /
# ir.Attribute, built through the Allo CAPI rather than textual parsing).
from .._mlir_libs._allo import (
    StreamType,
    PartitionAxisAttr,
    PartitionAttr,
    AssumeDepTypeAttr,
    AssumeDepDirAttr,
    MemoryKindAttr,
    DeterminacyAttr,
    OpKindAttr,
    CombOpKindAttr,
    StallContractAttr,
    CostFormAttr,
    CostAttr,
    ResourceUseAttr,
)
