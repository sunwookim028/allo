# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Catapult's combinational-CCORE RTL of Allo's ``add_bits`` against MiniTPU's
``vpu_bf16_add`` (both in the harness's ``comb`` shape: no clock), full stimulus.

    $ALLO_PYTHON cmp_comb.py <catapult solution dir>   # .../Catapult/bf16_add_comb.v1
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.getcwd())
from examples.minitpu.harness import rtl  # noqa: E402
from examples.minitpu.units import bf16_add as u  # noqa: E402

v1 = os.path.abspath(sys.argv[1])
from examples.minitpu.harness import stimulus as stim_mod  # noqa: E402
stim = stim_mod.binary_bf16().astype(np.uint64)
t = time.time()
want, _ = rtl.run(u.RTL, stim)
cat = rtl.RtlUnit(top="bf16_add_comb", sources=[os.path.join(v1, "concat_sim_rtl.v")],
                  inputs=[("a_i", 16), ("b_i", 16)], outputs=[("result_o", 16)], shape="comb")
got, _ = rtl.run(cat, stim)
n = len(stim)
eq = int((got[:, 0] == want[:, 0]).sum())
print(f"CCORE-CMP bf16_add bits comb: {eq}/{n} bit-exact vs MiniTPU vpu_bf16_add (comb shape, no clock); {time.time() - t:.1f}s")
