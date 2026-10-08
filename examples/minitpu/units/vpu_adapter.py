# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U4: ``sequencer_vpu_adapter``, the producer of ``vpu_ctrl_t`` (combinational).

Decoded V, M and X slots plus ``issue_i`` (the sequencer's ``run_accept``) and
the resolved VMEM word (``sequencer_agu_resolve``) to the VPU's 85-bit control
struct. Every *valid* field is ANDed with ``issue_i``; every payload field
(register addresses, destinations, ops, the VMEM address and
``vmem_store_read_hint``) passes **ungated**, so the struct carries the live
fetch head's operands on stall cycles too (``ref_ctrl_decode.VPU_CTRL_SOURCE``).
``alu_op`` widens from 3 to 4 bits by a cast that relies on both enums
agreeing.

Stimulus: random slot vectors with ``issue_i`` both ways, and the decoded
slots of every bundle of the two board images. Seed:
``tb_bundle_vpu_adapter`` (whole sequencer; scope ``tb.dut.u_vpu_adapter``).
"""

import os
import sys

from examples.minitpu.harness import rtl
from examples.minitpu.harness import ref_ctrl_decode as R
from examples.minitpu.harness.traces import rng_for
from examples.minitpu.units.seq_decoder import image_words

HERE = os.path.dirname(os.path.abspath(__file__))
PKGS = ["src/pkg/minitpu_config_pkg.sv", "src/core/vpu/vpu_pkg.sv", "src/core/sequencer/sequencer_pkg.sv"]
SOURCES = PKGS + ["src/core/sequencer/sequencer_vpu_adapter.sv", os.path.join(HERE, "rtl", "u4_vpu_adapter.sv")]
W = {n: sum(w for _, w in lay) for n, lay in (("v", R.V_SLOT), ("m", R.M_SLOT), ("x", R.X_SLOT))}
INPUTS = [("v_i", W["v"]), ("m_i", W["m"]), ("x_i", W["x"]), ("issue_i", 1), ("x_resolved_row_i", 12)]
RTL = rtl.RtlUnit(top="u4_vpu_adapter", sources=SOURCES, inputs=INPUTS,
                  outputs=[("vpu_ctrl_o", R.VPU_CTRL_W, "pre")], shape="trace", clk="clk_i", rst_n="rst_n_unused")
INSTANCES = {"base": RTL}
DEFAULT = "base"
LATENCY_SOURCE = "combinational (sequencer_vpu_adapter.sv always_comb); vpu_ctrl_t is valid in the issue cycle"


def REF(inst, cmd):
    return R.vpu_adapter_trace(cmd)


def _slots(f, slot, lay):
    return R.pack(lay, {n: f[f"{slot}.{n}"] for n, _ in lay})


def from_images(rng):
    rows = {p: [] for p, _ in INPUTS}
    for w in image_words():
        f = R.decode(w)
        for issue in (1, 0):
            for p, v in zip(rows, (_slots(f, "v", R.V_SLOT), _slots(f, "m", R.M_SLOT), _slots(f, "x", R.X_SLOT),
                                   issue, rng.getrandbits(12))):
                rows[p].append(v)
    return rows


def random_rows(rng, n):
    return {p: [rng.getrandbits(w) for _ in range(n)] for p, w in INPUTS}


def traces(inst):
    rng = rng_for("vpu_adapter", inst)
    return [("images", from_images(rng), True), ("random", random_rows(rng, 20000), True)]


def seeds():
    srcs = PKGS + [f"src/core/sequencer/{f}" for f in (
        "sequencer_decoder.sv", "sequencer_iram.sv", "sequencer_fetch_queue.sv", "sequencer_loop_buffer.sv",
        "sequencer_loop_ctrl.sv", "sequencer_agu_resolve.sv", "sequencer_scalar_agu.sv", "dma_desc_adapter.sv",
        "sequencer_vpu_adapter.sv", "sequencer.sv")]
    rows = R.seed_rows("tb_bundle_vpu_adapter", srcs, "dut.u_vpu_adapter",
                       [p for p, _ in INPUTS] + ["vpu_ctrl_o"], clk_scope="dut", clk="clk")
    return [("tb_bundle_vpu_adapter", "base", {p: rows[p] for p, _ in INPUTS}, {"vpu_ctrl_o": rows["vpu_ctrl_o"]},
             True)]


def probes(inst):
    cmd = {"v_i": [0] * 16, "m_i": [0] * 16, "x_i": [0] * 16, "issue_i": [1] * 8 + [0] * 8,
           "x_resolved_row_i": [0] * 16}
    cmd["v_i"] = [R.pack(R.V_SLOT, {"alu_valid": 1})] * 16
    return [("issue_i -> vpu_ctrl_o.alu_valid (comb)", 0,
             rtl.probe_trace(RTL, {p: rtl.pack(cmd[p], w) for p, w in INPUTS}, "vpu_ctrl_o", 8))]


def table():
    """vpu_ctrl_t, field by field: width, source, gated by issue."""
    lines = []
    for n, w in R.VPU_CTRL:
        src, gated = R.VPU_CTRL_SOURCE[n]
        lines.append(f"{n:22s} {w:3d}  {'issue &' if gated else 'ungated'}  {src}")
    return "\n".join(lines)


VARIANTS = {}

if __name__ == "__main__":
    print(table())
    sys.exit(0)
