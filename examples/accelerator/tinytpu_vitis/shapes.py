# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The five benchmark shapes, in order. The ONE definition.

Every consumer imports this: `bench_isa`, `stress_isa`, `cosim`,
`impact/bench_variant.py`, and -- through `chia_agent/evaluate.py`, which loads
this file by path -- `accept.py` and `test_harness.py`. It existed in seven
copies, and the order is load-bearing (the published cycle counts, the cosim
summary table and `accept.BASELINES` are all read positionally against it).

This module imports NOTHING, deliberately: the CHIA harness runs in a conda
env without `allo`, so it cannot import any module that pulls the compiler in.
"""

#: (M, K, N), in the order every published table uses.
SHAPES = [(4, 4, 4), (8, 8, 8), (12, 12, 12), (16, 16, 8), (16, 16, 16)]

#: The same five as the "MxKxN" strings the harness keys results on.
NAMES = [f"{M}x{K}x{N}" for (M, K, N) in SHAPES]
