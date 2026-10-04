# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Track B finding 3 as a Catapult form: the MXU (M1's front and PE grid) with
its output FIFOs as per-lane Streams -- ``gather_try`` ``try_put``\\ s each
finished group (refusal = the RTL's drop), ``pop_engine`` polls ``empty()`` and
``get``\\ s on a pop (``u3_track_b_2026-10-04/repros/mxu_streams_units.py``,
unchanged). ``make(n, w, inst)`` builds the probe's Architecture for one trace
length; ``run(mod, cmd, n, w)`` is the probe's call shape, returning the
contract ports (``input_ready_o``, ``input_accept_o``, ``output_valid_o``,
``output_data_o``) and keeping the per-cycle drop word in ``LAST_DROP``.
"""
import os, sys

import numpy as np

from allo.compose import Architecture, Memory

from examples.minitpu.harness import ref_mxu
from examples.minitpu.units import mxu as U
from examples.minitpu.units.mxu_pe_unit import pe_channels, pe_unit
from examples.minitpu.units.mxu_unit import mxu_front

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                "u3_track_b_2026-10-04", "repros"))
from mxu_streams_units import gather_try, pop_engine, probe_channels  # noqa: E402

LAST_DROP = None


def make(n, w=0, inst="dim2"):
    D = U.DIMS[inst]
    mems = (Memory("RST", "UInt(1)[N]"), Memory("PUSH", "UInt(1)[N]"), Memory("KIND", "UInt(1)[N]"),
            Memory("DATA", "UInt(16)[N * D]"), Memory("CMT", "UInt(1)[N]"), Memory("RDY", "UInt(1)[N]"),
            Memory("ACC", "UInt(1)[N]"), Memory("DROP", "UInt(16)[N]"), Memory("POP", "UInt(1)[N]"),
            Memory("VLD", "UInt(1)[N]"), Memory("ODATA", "UInt(16)[N * D * SUB]"))
    params = {"N": n, "D": D, "PE": ref_mxu.PE_LATENCY, "SUB": U.SUB, "ENTRIES": U.ENTRIES,
              "SKEW": (D - 1) * ref_mxu.PE_LATENCY + 1, "SPAN": ref_mxu.switch_span(D),
              "ROW0_PSUM_ZERO": 1, "EDGE_OUT": 0}
    arch = Architecture(name=f"mxu_streams_{inst}", parameters=params, memories=mems,
                        channels=probe_channels(pe_channels()), units=(mxu_front, pe_unit, gather_try, pop_engine))
    return arch.region()


def run(mod, cmd, n, w):
    global LAST_DROP
    D = w
    data = [(int(v) >> (16 * c)) & 0xFFFF for v in cmd["input_data_i"][:n] for c in range(D)]
    ins = [np.asarray(cmd["rst_ni"][:n], np.uint8), np.asarray(cmd["input_push_i"][:n], np.uint8),
           np.asarray(cmd["input_kind_i"][:n], np.uint8), np.asarray(data, np.uint16),
           np.asarray(cmd["weight_commit_i"][:n], np.uint8)]
    rdy, acc, vld = (np.zeros(n, np.uint8) for _ in range(3))
    drop = np.zeros(n, np.uint16)
    od = np.zeros(n * D * U.SUB, np.uint16)
    mod(*ins, rdy, acc, drop, np.asarray(cmd["output_pop_i"][:n], np.uint8), vld, od)
    LAST_DROP = drop
    k = D * U.SUB
    out = [sum(int(od[t * k + i]) << (16 * i) for i in range(k)) for t in range(n)]
    return {"input_ready_o": rdy, "input_accept_o": acc, "output_valid_o": vld, "output_data_o": out}
