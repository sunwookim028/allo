# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The accumulator file as an optional module (README D-19).

What the module brings: the accumulator unit (``accu``, frozen, reused), the
result path it feeds (``dma_st`` and the result matrix ``C``), the three
channels that exist only with it (the sequencer's two dispatch queues
``c_acc``/``c_dst`` and ``ac2sp``), its contract parameter ``AR_RAW_DIST``
(the dependence obligation ``accu``'s directive quotes), and the five
instruction slots that exist only with it: ``mm`` (its sink is the
accumulator file), ``vadd``, ``vrelu``, ``vaddrelu`` and ``mvout``.

``dma_st`` is part of the option and not of the base because without the
accumulator nothing produces ``ac2sp``: the base machine has no result path.
That is the design's fact, and the netlist rules say so -- composing the base
alone is refused naming ``c_acc`` (the sequencer dispatches to a unit the
instance does not have). MiniTPU's instance lacks this module: its MXU pops
results to the vector register file instead.
"""

from __future__ import annotations

from allo.compose import Channel, Memory, Option
from examples.tinytpu.ip.units.accumulator import accu
from examples.tinytpu.ip.units.dma_store import dma_st

#: The slots that exist only with the accumulator file, derived in
#: ``instance.compare_isa`` from the spec's own actions (every opcode with an
#: action at ``accu`` or ``dma_st``) and held to this tuple.
SLOTS = ("mm", "vadd", "vrelu", "mvout", "vaddrelu")


def option(ar_raw_dist: int) -> Option:
    return Option(
        name="accumulator",
        units=(accu, dma_st),
        channels=(
            Channel("c_acc", "UInt(64)", "QD", carries="sequencer -> accu"),
            Channel("c_dst", "UInt(64)", "QD", carries="sequencer -> dma_st"),
            Channel("ac2sp", "UInt(VW)", "QD", carries="accumulator -> dma_st"),
        ),
        memories=(Memory("C", "int8[MAXDIM * MAXDIM]"),),
        parameters={"AR_RAW_DIST": ar_raw_dist},
        isa=SLOTS,
    )
