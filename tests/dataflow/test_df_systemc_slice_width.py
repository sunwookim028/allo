# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""ac_int slices are sized by their bounds, not by the value's type (E2).

The builder types a bit slice whose bounds fold only after ``meta_for``
expansion with the word's width. The SystemC and Catapult emitters sized the
``ac_int`` slice by that type: ``word[16*c : 16*(c+1)] = <i32>`` became
``set_slc(48, ac_int<32, false>(v))`` on a 64-bit word -- ac_int's "Out of
bounds set_slc" assertion in csim, and the 16 bits above every lower lane
overwritten -- and ``lane: UInt(32) = word[16*c : 16*(c+1)]`` became
``slc<32>(lo)``, reading the next lane into the upper half. The MiniTPU bf16
matrix rigs at DIM=4 aborted on the first and, past it, read 7/32 right. See
``dev/records/limitations/u3_fixes_2026-10-04.rst``.
"""
import os
import re
import tempfile

import numpy as np
import pytest

import allo
import allo.dataflow as df
from allo.ir.types import UInt, Stream

needs_csim = pytest.mark.skipif(
    not (os.environ.get("MGC_HOME") and os.environ.get("SYSTEMC_HOME")),
    reason="SystemC csim needs Catapult (MGC_HOME) and SYSTEMC_HOME",
)

N = 3
D = 4
LANE = 16


def _region():
    @df.region()
    def top(A: UInt(32)[N * D], O: UInt(32)[N * D]):
        row: Stream[UInt(D * LANE), 2]

        @df.kernel(mapping=[1], args=[A])
        def pack(a: UInt(32)[N * D]):
            for t in range(N):
                w: UInt(D * LANE) = 0
                with allo.meta_for(D) as c:
                    v: UInt(32) = a[t * D + c]
                    w[LANE * c : LANE * (c + 1)] = v
                row.put(w)

        @df.kernel(mapping=[1], args=[O])
        def unpack(o: UInt(32)[N * D]):
            for t in range(N):
                w: UInt(D * LANE) = row.get()
                with allo.meta_for(D) as c:
                    lane: UInt(32) = w[LANE * c : LANE * (c + 1)]
                    o[t * D + c] = lane

    return top


def _inputs():
    # 32-bit values whose upper halves are non-zero: they must not leak into
    # the next lane, and each lane reads back as its low 16 bits.
    return (np.arange(N * D, dtype=np.uint64) * 0x9E3779B1 + 0xFFFF0001).astype(
        np.uint32
    )


def test_simulator_reference():
    a = _inputs()
    o = np.zeros(N * D, np.uint32)
    df.build(_region(), target="simulator")(a, o)
    np.testing.assert_array_equal(o, a & 0xFFFF)


def test_systemc_slices_sized_by_bounds():
    code = df.customize(_region()).build(target="systemc").hls_code
    assert not re.search(r"set_slc\(\d+, ac_int<32, false>", code)
    assert len(re.findall(r"set_slc\(\d+, ac_int<16, false>", code)) == D
    assert len(re.findall(r"= ac_int<16, false>\(_bs_\w+\.slc<16>\(", code)) == D
    assert ".slc<32>(" not in code


def test_catapult_slices_sized_by_bounds():
    def k(A: UInt(32)[D], O: UInt(32)[D]):
        w: UInt(D * LANE) = 0
        with allo.meta_for(D) as c:
            v: UInt(32) = A[c]
            w[LANE * c : LANE * (c + 1)] = v
        with allo.meta_for(D) as c:
            lane: UInt(32) = w[LANE * c : LANE * (c + 1)]
            O[c] = lane

    code = str(allo.customize(k).build(target="catapult"))
    assert len(re.findall(r"set_slc\(\d+, ac_int<16, false>", code)) == D
    assert ".slc<32>(" not in code


@needs_csim
def test_systemc_csim_packed_lanes():
    a = _inputs()
    o = np.zeros(N * D, np.uint32)
    with tempfile.TemporaryDirectory() as tmp:
        df.build(_region(), target="systemc", mode="csim", project=tmp)(a, o)
    np.testing.assert_array_equal(o, a & 0xFFFF)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
