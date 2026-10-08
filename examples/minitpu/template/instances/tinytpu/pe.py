# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""TinyTPU's processing element in the template's form (README D-15, D-17).

``examples/tinytpu/ip/units/pe.py`` line for line, with the MAC taken out of
the body: ``psum = psum_north + activation16 * weight16`` becomes
``MAC_ADD(MAC_MUL(activation, weight), psum_north)`` on the engine the
architecture binds to slot ``MAC``, and every literal lane width (``8 * i``,
``32 * j``, ``int8``, ``int32``, ``UInt(AW)``) becomes the engine's
(``MAC_IN_BITS``, ``MAC_OUT_BITS``, ``MAC_IN``, ``MAC_ACC``). Nothing in the
body names int8.

Two couplings the body cannot remove, both at the weight queue: ``wld``
(frozen, reused) packs the weight lane at ``pe_word[0:8]`` and the row count
at ``[8:20]``, so the weight is taken by typed assignment (the low
``MAC_IN_BITS`` bits; D-17's rule for a parameter in a slice bound) and the
row count at the literal position ``wld`` writes it. A wider engine needs a
``wld`` with an engine slot of its own.

The activation lane is taken by a constant shift and a typed assignment
rather than ``activation_word[MAC_IN_BITS * i : MAC_IN_BITS * (i + 1)]``:
the front end folds no parameter name in a slice bound (``infer.py``
``visit_symbol`` makes every name a symbol), defaults the slice to 32 bits,
and at T=4 that is the whole word, which lowers to ``trunci i32 -> i32`` and
fails. ``compose``'s slice-bound check does not catch it because ``i`` is a
bound name (the record's F1/F2).
"""

from __future__ import annotations

from allo.compose import unit


@unit(
    instances=("T", "T"),
    reads=("wq", "acol", "a_fwd", "p_fwd", "cw"),
    writes=("acol", "a_fwd", "p_fwd", "cw"),
    parameters=("T",),
    engines=("MAC_IN", "MAC_IN_BITS", "MAC_ACC", "MAC_OUT_BITS", "MAC_MUL",
             "MAC_ADD", "MAC_PACK"),
)
def pe():
    i, j = df.get_pid()
    trip_word: UInt(32) = wq[i, j].get()
    n_wavefront_row: int32 = trip_word[0:16]
    weight: MAC_IN = 0
    mm_rows: int32 = 0
    row: int32 = -1             # advanced at the top: hoisting this holds II=1
    for work in range(n_wavefront_row):
        row += 1
        if row >= mm_rows:
            pe_word: UInt(32) = wq[i, j].get()
            # wld packs the lane at [0:MAC_IN_BITS]: the typed assignment
            # keeps the low bits (README D-17, slice bounds).
            weight = pe_word
            mm_rows = pe_word[8:20]
            row = 0
        activation: MAC_IN = 0
        with allo.meta_if(j == 0):
            activation_word: UInt(T * MAC_IN_BITS) = acol[i].get()
            with allo.meta_if(i != T - 1):
                acol[i + 1].put(activation_word)
            # Lane i by a constant shift and a typed assignment (the low
            # MAC_IN_BITS bits), not by `[MAC_IN_BITS * i : ...]`: the front
            # end folds no parameter NAME in a slice bound (record F1), and
            # the 32-bit default it falls back to is the word itself at T=4.
            lane_shifted: UInt(T * MAC_IN_BITS) = activation_word >> (MAC_IN_BITS * i)
            activation = lane_shifted
        with allo.meta_else():
            activation = a_fwd[i, j - 1].get()
        psum_north: MAC_ACC = 0
        with allo.meta_if(i > 0):
            psum_north = p_fwd[i - 1, j].get()
        product: MAC_ACC = MAC_MUL(activation, weight)
        psum: MAC_ACC = MAC_ADD(product, psum_north)
        with allo.meta_if(i != T - 1):
            p_fwd[i, j].put(psum)
        with allo.meta_else():
            # The bottom row assembles the packed result word as it travels
            # east, so the accumulator sees whole words and there is no T-way
            # fan-in.
            result_word: UInt(T * MAC_OUT_BITS) = 0
            with allo.meta_if(j > 0):
                result_word = cw[j - 1].get()
            result_word[MAC_OUT_BITS * j : MAC_OUT_BITS * (j + 1)] = MAC_PACK(psum)
            cw[j].put(result_word)
        with allo.meta_if(j != T - 1):
            a_fwd[i, j].put(activation)
