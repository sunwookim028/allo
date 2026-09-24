# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""One module per unit. Each is importable on its own, declares the channels,
memories, parameters and ISA names it needs, and closes over nothing: an
architecture binds every free name in its body when it composes the region."""

from .accumulator import accu
from .dma_load import dma_ld
from .dma_store import dma_st
from .pe import pe
from .reduction_tree import reduce_tree
from .scratchpad import spm
from .sequencer import sequencer
from .vector_regs import vru
from .weight_loader import wld

__all__ = ["accu", "dma_ld", "dma_st", "pe", "reduce_tree", "spm", "sequencer",
           "vru", "wld"]
