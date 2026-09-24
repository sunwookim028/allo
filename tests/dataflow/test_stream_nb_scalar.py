# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Scalar-Stream Non-Blocking Op Regression Tests
==============================================
Companion to test_stream_nb_simple.py, which exercises the non-blocking stream
ops (empty/full/try_get/try_put) exclusively on INDEXED streams (Stream[T,d][1]
accessed as S[0]). This file exercises the same ops on SCALAR streams
(Stream[T,d] accessed as S directly).

Regression guard: the builder previously read node.func.value.slice
unconditionally in all four NB-op branches, which raised
"'Name' object has no attribute 'slice'" for a scalar (non-subscripted)
stream. The fix guards the index extraction with isinstance(..., ast.Subscript)
and passes indices=[] for scalar streams. These tests fail on the pre-fix code.
"""
from __future__ import annotations
import allo
from allo.ir.types import int32, int1, Stream
import allo.dataflow as df
import numpy as np


def test_scalar_empty_full_sim():
    """empty()/full() on a SCALAR depth-1 stream, before and after a try_put."""

    @df.region()
    def nb_status(out: int32[5]):
        S: Stream[int32, 1]  # scalar stream (no [1])

        @df.kernel(mapping=[1], args=[out])
        def test_logic(out_buf: int32[5]):
            e0: int1 = S.empty()  # expect True  (empty initially)
            f0: int1 = S.full()  # expect False
            out_buf[0] = 1 if e0 else 0
            out_buf[1] = 1 if f0 else 0

            success: int1 = S.try_put(42)  # fill the single slot
            out_buf[4] = 1 if success else 0  # expect 1

            e1: int1 = S.empty()  # expect False (has data)
            f1: int1 = S.full()  # expect True  (depth-1 now full)
            out_buf[2] = 1 if e1 else 0
            out_buf[3] = 1 if f1 else 0

    sim = df.build(nb_status, target="simulator")
    out = np.zeros(5, dtype=np.int32)
    sim(out)
    assert out[0] == 1, f"empty() should be True initially, got {out[0]}"
    assert out[1] == 0, f"full() should be False initially, got {out[1]}"
    assert out[2] == 0, f"empty() should be False after put, got {out[2]}"
    assert out[3] == 1, f"full() should be True after put, got {out[3]}"
    assert out[4] == 1, f"try_put() should succeed, got {out[4]}"
    print("test_scalar_empty_full_sim PASSED")

    mod_sc = df.build(nb_status, target="systemc", mode="cosim", project="test_stream_nb_scalar")
    out[...] = 0   # clear the simulator's result first
    mod_sc(out)
    assert out[4] == 1, f"try_put() should succeed, got {out[4]}"
    print("SystemC Cosim Passed!")


def test_scalar_try_put_try_get_sim():
    """try_put / try_get spin-until-success on a SCALAR stream (producer/consumer)."""

    @df.region()
    def top_nb(out: int32[4]):
        S: Stream[int32, 4]  # scalar stream

        @df.kernel(mapping=[1])
        def producer():
            for i in range(4):
                while not S.try_put(i * 10):
                    pass

        @df.kernel(mapping=[1], args=[out])
        def consumer(out_buf: int32[4]):
            for i in range(4):
                val: int32 = 0
                ok: int1 = 0
                while ok == 0:
                    val, ok = S.try_get()
                out_buf[i] = val

    sim = df.build(top_nb, target="simulator")
    out = np.zeros(4, dtype=np.int32)
    sim(out)
    np.testing.assert_array_equal(out, [0, 10, 20, 30])
    print("test_scalar_try_put_try_get_sim PASSED")

    mod_sc = df.build(top_nb, target="systemc", mode="cosim", project="test_stream_nb_scalar_2")
    out[...] = 0   # clear the simulator's result first
    mod_sc(out)
    np.testing.assert_array_equal(out, [0, 10, 20, 30])
    print("SystemC Cosim Passed!")


def test_scalar_nb_ops_hls_codegen():
    """Scalar NB ops lower to the correct Vitis HLS API (no [i] subscript)."""

    @df.region()
    def top_nb_hls():
        ctrl_in: Stream[int32, 2]
        ctrl_out: Stream[int32, 2]

        @df.kernel(mapping=[1])
        def relay():
            val: int32 = 0
            ok: int1 = 0
            val, ok = ctrl_in.try_get()
            if ok:
                sent: int1 = ctrl_out.try_put(val + 1)
            e: int1 = ctrl_in.empty()
            f: int1 = ctrl_out.full()
            if e or f:
                pass

    mod = allo.customize(top_nb_hls)
    hls_code = mod.build(target="vhls").hls_code
    assert ".read_nb(" in hls_code, "Expected .read_nb() for try_get"
    assert ".write_nb(" in hls_code, "Expected .write_nb() for try_put"
    assert ".empty()" in hls_code, "Expected .empty()"
    assert ".full()" in hls_code, "Expected .full()"
    print("test_scalar_nb_ops_hls_codegen PASSED")


if __name__ == "__main__":
    test_scalar_empty_full_sim()
    test_scalar_try_put_try_get_sim()
    test_scalar_nb_ops_hls_codegen()
    print("\nAll test_stream_nb_scalar tests PASSED!")
