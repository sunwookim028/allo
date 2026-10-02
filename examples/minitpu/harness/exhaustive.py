# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exhaustive check of a two-operand bf16 unit's reference: all 2^32 pairs.

    python -m examples.minitpu.harness.exhaustive bf16_add [--chunk-bits 24]

Streams the pairs through the RTL in chunks (``a`` major) and compares each
chunk with the unit's ``REF``; prints the mismatch count and the first few.
Set ``MINITPU_HARNESS_CACHE`` to local disk: a 2^24 chunk is 512 MB of I/O.
"""

import argparse
import importlib
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from examples.minitpu.harness import rtl  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("unit")
    ap.add_argument("--chunk-bits", type=int, default=24)
    args = ap.parse_args(argv)
    u = importlib.import_module(f"examples.minitpu.units.{args.unit}")
    per = 1 << (args.chunk_bits - 16)  # a values per chunk
    b = np.arange(1 << 16, dtype=np.uint64)
    bad, first, t = 0, [], time.time()
    for a0 in range(0, 1 << 16, per):
        a = np.repeat(np.arange(a0, a0 + per, dtype=np.uint64), 1 << 16)
        stim = np.stack([a, np.tile(b, per)], axis=1)
        got = rtl.run(u.RTL, stim)[0][:, 0].astype(np.int64)
        want = u.REF(stim[:, 0], stim[:, 1]).astype(np.int64)
        idx = np.flatnonzero(got != want)
        bad += len(idx)
        first += [(int(stim[i, 0]), int(stim[i, 1]), int(want[i]), int(got[i])) for i in idx[: 5 - len(first)]]
    n = 1 << 32
    tag = "EXHAUSTIVE-MATCH" if bad == 0 else "EXHAUSTIVE-DIFF "
    print(f"{tag} {args.unit} {n - bad}/{n} ({time.time() - t:.0f}s)")
    for a_, b_, w, g in first:
        print(f"    {a_:04x},{b_:04x}: ref {w:x} rtl {g:x}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
