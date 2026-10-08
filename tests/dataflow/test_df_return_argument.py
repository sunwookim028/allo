# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A helper that returns its scalar argument as is gets an output port (E3).

The HLS emitters name each result's output pointer after the value that
defines it and skip a returned argument (an array argument is returned in
place). ``def pack_int32(v: int32) -> int32: return v`` was emitted as
``void pack_int32(int32_t v0)`` while every call passed ``(v, &out)``: g++
"too many arguments" in SystemC csim, and the same text from the Vitis and
Catapult emitters. See ``dev/records/limitations/u3_fixes_2026-10-04.rst``.
"""
import os
import re
import tempfile

import numpy as np
import pytest

import allo
import allo.dataflow as df
from allo.ir.types import int32, UInt

needs_csim = pytest.mark.skipif(
    not (os.environ.get("MGC_HOME") and os.environ.get("SYSTEMC_HOME")),
    reason="SystemC csim needs Catapult (MGC_HOME) and SYSTEMC_HOME",
)

N = 4


def pack_int32(v: int32) -> int32:
    return v


def pass_u8(v: UInt(8)) -> UInt(8):
    return v


def first_of(v: int32, w: int32) -> int32:
    return v


def _region():
    @df.region()
    def top(A: int32[N], U: UInt(8)[N], B: int32[N], C: UInt(8)[N]):
        @df.kernel(mapping=[1], args=[A, U, B, C])
        def k(a: int32[N], u: UInt(8)[N], b: int32[N], c: UInt(8)[N]):
            for i in range(N):
                b[i] = pack_int32(a[i]) + first_of(a[i], 7)
                c[i] = pass_u8(u[i])

    return top


def _inputs():
    a = np.array([3, -5, 1 << 30, -(1 << 31)], dtype=np.int32)
    u = np.array([0, 1, 200, 255], dtype=np.uint8)
    return a, u, (a.astype(np.int64) * 2).astype(np.int32), u


def _helper_signature(code, name):
    m = re.search(r"void " + name + r"\(([^)]*)\)", code)
    assert m, f"{name} not emitted"
    return [p.strip() for p in m.group(1).split(",")]


@pytest.mark.parametrize("target", ["vhls", "catapult"])
def test_identity_helper_has_output_port(target):
    def kernel(A: int32[N], B: int32[N]):
        for i in range(N):
            B[i] = pack_int32(A[i]) + 1

    code = str(allo.customize(kernel).build(target=target))
    params = _helper_signature(code, "pack_int32")
    assert len(params) == 2 and params[1].startswith("int32_t *"), params
    out = params[1].split("*")[1]
    assert re.search(r"\*" + out + r" = v\d+;", code), code


def test_simulator_unchanged():
    a, u, want_b, want_c = _inputs()
    b, c = np.zeros(N, np.int32), np.zeros(N, np.uint8)
    df.build(_region(), target="simulator")(a, u, b, c)
    np.testing.assert_array_equal(b, want_b)
    np.testing.assert_array_equal(c, want_c)


def test_systemc_emits_both_ports():
    code = df.customize(_region()).build(target="systemc").hls_code
    assert len(_helper_signature(code, "pack_int32")) == 2
    assert len(_helper_signature(code, "first_of")) == 3
    assert len(_helper_signature(code, "pass_u8")) == 2


@needs_csim
def test_systemc_csim_identity_helpers():
    a, u, want_b, want_c = _inputs()
    b, c = np.zeros(N, np.int32), np.zeros(N, np.uint8)
    with tempfile.TemporaryDirectory() as tmp:
        df.build(_region(), target="systemc", mode="csim", project=tmp)(a, u, b, c)
    np.testing.assert_array_equal(b, want_b)
    np.testing.assert_array_equal(c, want_c)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
