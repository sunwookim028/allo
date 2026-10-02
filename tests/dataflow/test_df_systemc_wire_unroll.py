# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A constant-trip loop inside a Wire kernel's steady-state loop is unrolled
in the emitted SystemC (C-W1).

A Wire kernel's steady-state loop is the clock: one iteration, one sampled
step. A shift-register pipe ``for j in range(1, 3): p[3-j] = p[2-j]``
emitted rolled is merged by Catapult into the pipelined loop --
``Prescheduled LOOP '/wa_0/run/while' (2 c-steps) (SCHD-7)`` after ``Loop
'/wa_0/run/l_S_k_0_k' is left rolled. (LOOP-4)`` -- so the unit sampled its
Wire inputs every second cycle while reporting II=1 (MiniTPU word array:
19,502/67,717). ``s.unroll`` on the loop made the RTL cycle-exact; the
emitter now writes that pragma itself. See
``dev/records/limitations/uint_index_2026-10-02.rst``.
"""
import pytest

import allo.dataflow as df
from allo.ir.types import UInt, Wire

N = 4
W = UInt(16)


def _wire_pipe():
    @df.region()
    def top(A: W[N], B: W[N]):
        w_in: Wire[W]
        w_out: Wire[W]

        @df.kernel(mapping=[1], args=[A])
        def src(a: W[N]):
            for t in range(N):
                w_in.put(a[t])

        @df.kernel(mapping=[1], args=[])
        def k():
            p: W[3]
            for t in range(N):
                for j in range(1, 3):
                    p[3 - j] = p[2 - j]
                p[0] = w_in.get()
                w_out.put(p[2])

        @df.kernel(mapping=[1], args=[B])
        def snk(b: W[N]):
            for t in range(N):
                b[t] = w_out.get()

    return top


def _stream_pipe():
    """The same body with stream-ported arrays: no Wire, no pragma."""

    @df.region()
    def top(A: W[N], B: W[N]):
        @df.kernel(mapping=[1], args=[A, B])
        def k(a: W[N], b: W[N]):
            p: W[3]
            for t in range(N):
                for j in range(1, 3):
                    p[3 - j] = p[2 - j]
                p[0] = a[t]
                b[t] = p[2]

    return top


def _kernel_body(code, name):
    return code.split(f"SC_MODULE({name})")[1].split("SC_MODULE(")[0]


def test_wire_kernel_shift_loop_gets_hls_unroll():
    code = df.customize(_wire_pipe()).build(target="systemc").hls_code
    body = _kernel_body(code, "k_0")
    lines = [l.strip() for l in body.splitlines()]
    pragmas = [i for i, l in enumerate(lines) if l.startswith("#pragma hls_unroll")]
    assert len(pragmas) == 1, pragmas  # the steady-state body is emitted once
    # the pragma sits directly above the shift loop's header
    for i in pragmas:
        assert lines[i + 1].startswith("l_S_j_0_j: for ("), lines[i : i + 2]


def test_explicit_unroll_is_not_doubled():
    s = df.customize(_wire_pipe())
    s.unroll("k_0:j")
    code = s.build(target="systemc").hls_code
    body = _kernel_body(code, "k_0")
    pragmas = [l for l in body.splitlines() if "#pragma hls_unroll" in l]
    assert len(pragmas) == 1, pragmas


def test_connections_kernel_is_left_alone():
    code = df.customize(_stream_pipe()).build(target="systemc").hls_code
    assert "hls_unroll" not in _kernel_body(code, "k_0")


if __name__ == "__main__":
    pytest.main([__file__])
