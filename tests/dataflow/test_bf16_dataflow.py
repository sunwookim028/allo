# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""BF16 in the dataflow simulator, and the emitters that cannot take it.

``examples/minitpu`` is a BF16 machine, so it rests on three facts that
nothing else in the tree pins down:

1. ``bfloat16`` works end to end in the simulator -- as a ``Stream`` element,
   as an arithmetic type, and widened to ``float32`` and back.
2. A 24-bit float accumulator (1 sign + 8 exponent + 15 fraction, MiniTPU's
   ``MXU_ACC_W``) can be emulated with the ``float32``/``uint32`` bitcast pair,
   so a design does not need a new MLIR type to model one.
3. The same design **cannot be emitted**.  ``EmitVivadoHLS.cpp:115``,
   ``EmitCatapultHLS.cpp:102`` and ``EmitSystemC.cpp`` all reach
   ``assert(1 == 0 && "Got unsupported type.")`` on a ``bf16``, which is a
   SIGABRT and not a catchable exception -- so that one runs in a subprocess.

See ``docs/source/developer/limitations.rst``.
"""

import subprocess
import sys
import textwrap

import numpy as np
import ml_dtypes
import pytest

import allo.dataflow as df
from allo.ir.types import bfloat16, float32, uint32, Stream

bf16 = ml_dtypes.bfloat16
N = 8


def test_bf16_stream_and_arithmetic():
    """bf16 as a stream element and as an arithmetic type, in the simulator."""

    @df.region()
    def top(A: bfloat16[N], B: bfloat16[N], C: bfloat16[N]):
        pipe: Stream[bfloat16, 4]

        @df.kernel(mapping=[1], args=[A])
        def producer(a: bfloat16[N]):
            for i in range(N):
                pipe.put(a[i])

        @df.kernel(mapping=[1], args=[B, C])
        def consumer(b: bfloat16[N], c: bfloat16[N]):
            for i in range(N):
                v: bfloat16 = pipe.get()
                c[i] = v * b[i] + b[i]

    mod = df.build(top, target="simulator")
    a = np.arange(1, N + 1, dtype=np.float32).astype(bf16)
    b = (np.arange(N, dtype=np.float32) * 0.5 + 1).astype(bf16)
    c = np.zeros(N, dtype=bf16)
    mod(a, b, c)
    want = (a.astype(np.float32) * b.astype(np.float32) + b.astype(np.float32)).astype(
        bf16
    )
    np.testing.assert_array_equal(np.asarray(c), np.asarray(want))


def test_acc24_emulation_via_bitcast():
    """A 24-bit float accumulator out of float32 + uint32 bitcasts.

    ``MXU_ACC_W = 1 + 8 + 15`` is float32's exponent field with eight fewer
    fraction bits, so acc24 is float32 rounded to 15 fraction bits.  The RTL's
    rule (``mxu_acc24_add_pipe.sv``) is add-half-plus-lsb, i.e. nearest-even.

    A BF16 x BF16 product has a 16-bit significand, which acc24 holds exactly:
    this test pins that the emulation leaves such a product untouched.
    """

    @df.region()
    def top(A: bfloat16[N], B: bfloat16[N], D: float32[N]):
        pipe: Stream[float32, 4]

        @df.kernel(mapping=[1], args=[A, B])
        def mul(a: bfloat16[N], b: bfloat16[N]):
            for i in range(N):
                av: float32 = a[i]
                bv: float32 = b[i]
                prod: float32 = av * bv
                u: uint32 = prod.bitcast()
                lsb: uint32 = (u >> 8) & 1
                r: uint32 = u + 127 + lsb
                t: uint32 = r & 4294967040
                y: float32 = t.bitcast()
                pipe.put(y)

        @df.kernel(mapping=[1], args=[D])
        def sink(d: float32[N]):
            for i in range(N):
                d[i] = pipe.get()

    mod = df.build(top, target="simulator")
    a = np.array([1.5, 1.0009765625, 3.0, 1e-3, 7.0, 1.25, 2.5, 100.0], np.float32)
    b = np.array([1.5, 1.0009765625, 5.0, 3e-3, 9.0, 1.75, 2.5, 3.0], np.float32)
    a, b = a.astype(bf16), b.astype(bf16)
    d = np.zeros(N, dtype=np.float32)
    mod(a, b, d)
    exact = a.astype(np.float32) * b.astype(np.float32)
    np.testing.assert_array_equal(d, exact)


_EMIT_SRC = textwrap.dedent(
    """
    import sys
    import numpy  # before allo: MKL_THREADING_LAYER vs libgomp (see AGENTS.md)
    import allo.dataflow as df
    from allo.ir.types import Stream
    from allo.ir.types import {ty} as Ty

    @df.region()
    def top(A: Ty[4], B: Ty[4]):
        pipe: Stream[Ty, 4]

        @df.kernel(mapping=[1], args=[A])
        def p(a: Ty[4]):
            for i in range(4):
                pipe.put(a[i])

        @df.kernel(mapping=[1], args=[B])
        def c(b: Ty[4]):
            for i in range(4):
                b[i] = pipe.get()

    df.build(top, target=sys.argv[1], project=sys.argv[2])
    print("EMITTED")
    """
)


def _emit(ty, target, tmp_path):
    # A file, not ``python -c``: Allo traces a region with inspect.getsource.
    src = tmp_path / f"emit_{ty}.py"
    src.write_text(_EMIT_SRC.format(ty=ty))
    return subprocess.run(
        [sys.executable, str(src), target, str(tmp_path / f"{ty}.prj")],
        capture_output=True,
        text=True,
        timeout=600,
    )


@pytest.mark.parametrize("target", ["vitis_hls", "catapult", "systemc"])
def test_float16_emits_but_bfloat16_aborts(target, tmp_path):
    """The control emits; bf16 aborts the process.

    This is a characterisation test, not a wish: it fails the day an emitter
    learns ``bf16``, which is the day to delete it and emit the MiniTPU model.
    """
    ok = _emit("float16", target, tmp_path)
    assert (
        "EMITTED" in ok.stdout
    ), f"float16 should emit on {target}: {ok.stderr[-800:]}"

    bad = _emit("bfloat16", target, tmp_path)
    assert "EMITTED" not in bad.stdout
    assert bad.returncode != 0, f"bf16 unexpectedly emitted on {target}"
    assert (
        "Got unsupported type" in bad.stderr or bad.returncode < 0
    ), f"bf16 failed on {target} for an unexpected reason: {bad.stderr[-800:]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
