# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""U4 track A: the harness side of wide control words (plan P-8 carried to U4).

The Allo side never has a port wider than 32 bits: a 128-bit bundle is
``UInt(32)[4]`` lanes, a slot record wider than 32 bits two lanes, lane 0 the
least significant (B6/S7/S8: no numpy dtype above 64 bits; csim reads a port
through ``long long``). These helpers split the harness's Python ints into
lane arrays and join them back, and give each packed-struct field its LSB so
a unit body can name ``F_<SLOT>_<FIELD>`` positions instead of literals.
"""

import numpy as np


def lsb(layout):
    """``[(name, width), ...]`` MSB first -> ``{name: (lsb, width)}``."""
    out, pos = {}, sum(w for _, w in layout)
    for n, w in layout:
        pos -= w
        out[n] = (pos, w)
    return out


def split(values, nlanes):
    """Ints -> ``uint32[n, nlanes]`` (lane 0 = bits 31:0)."""
    vals = [int(v) for v in values]
    return np.array([[(v >> (32 * k)) & 0xFFFFFFFF for k in range(nlanes)] for v in vals],
                    dtype=np.uint32).reshape(len(vals), nlanes)


def join(arr):
    """``uint32[n, k]`` -> ints."""
    a = np.asarray(arr, dtype=np.uint64)
    return [sum(int(a[t, k]) << (32 * k) for k in range(a.shape[1])) for t in range(a.shape[0])]


def col(cmd, port, n, dtype=np.uint32):
    return np.asarray([int(x) for x in cmd[port][:n]], dtype=dtype)
