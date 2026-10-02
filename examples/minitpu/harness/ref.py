# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bit-level numpy references for MiniTPU's arithmetic units.

Each reference states the RTL's semantics, not IEEE's: where the two differ,
the difference is named here and was measured against the RTL with
``harness/rtl.py``. The Allo side is compared with the RTL directly; these
references exist to *explain* a mismatch, not to replace the RTL as oracle.
"""

import numpy as np


def _bf16_to_f32(x):
    return (np.asarray(x, dtype=np.uint32) << 16).view(np.float32)


def f32_to_bf16_rne(f):
    """float32 bits -> bf16 bits, round to nearest even, NaN kept with sign."""
    f = np.asarray(f, dtype=np.float32).view(np.uint32)
    out = ((f + 0x7FFF + ((f >> 16) & 1)) >> 16).astype(np.uint16)
    nan = (((f >> 23) & 0xFF) == 0xFF) & ((f & 0x7FFFFF) != 0)
    out[nan] = (((f[nan] >> 31) << 15) | 0x7FC0).astype(np.uint16)
    return out


def ieee_bf16_add(a, b):
    """IEEE semantics: exact sum, one RNE rounding to bf16.

    fp32 has 24 significand bits >= 2*8+2, so rounding the exact sum to fp32
    first and then to bf16 is innocuous (no double-rounding error).
    """
    with np.errstate(all="ignore"):
        return f32_to_bf16_rne(_bf16_to_f32(a) + _bf16_to_f32(b))


def vpu_bf16_add(a, b):
    """``vpu_bf16_add.sv`` at b3ba0a4d: IEEE RNE add with subnormals, except

    * every NaN result is the positive canonical NaN ``0x7FC0`` (IEEE keeps a
      sign; MiniTPU's ``tb/tb_bf16_add.cpp`` skips specials, so its own test
      does not see this);
    * ``(+0) + (-0)`` is ``-0`` (IEEE RNE gives ``+0``). Measured on the RTL
      2026-10-02 over 251,936 vectors (corners crossed, ties, random): these
      two are the only differences.
    """
    a = np.asarray(a, dtype=np.uint16)
    b = np.asarray(b, dtype=np.uint16)
    out = ieee_bf16_add(a, b)
    nan = ((out & 0x7F80) == 0x7F80) & ((out & 0x7F) != 0)
    out[nan] = 0x7FC0
    zz = ((a & 0x7FFF) == 0) & ((b & 0x7FFF) == 0) & (a != b)
    out[zz] = b[zz]  # (+0)+(-0) -> -0 measured; (-0)+(+0) -> +0 measured
    return out
