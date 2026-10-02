# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Running an HLS IP with ``hls::stream`` ports under CPU dataflow simulation.

``df.build(top, target="simulator")`` turns every Allo stream into a ring buffer
in shared memory and every kernel into an OpenMP thread. These tests check that a
hand-written HLS IP joins that scheme through Allo's ``hls::stream`` shim: the
IP's own ``read()`` / ``write()`` calls drive Allo's ring buffers directly.

See ``docs/IP_STREAM_SIM_SHIM.md`` for the design; ``test_stream_ip.py`` covers
the FPGA path.

Run with ``OMP_NUM_THREADS`` at least as large as the number of kernels in the
region (the simulator gives each kernel one ``omp.section``); ``OMP_NUM_THREADS=8``
is what the repo uses.
"""

import textwrap
from pathlib import Path

import numpy as np
import pytest

import allo
from allo.ir.types import int32, Stream
import allo.dataflow as df

N = 32
_IMPL = Path(__file__).resolve().parent / "vadd_stream.cpp"

# The IP source used by the parametric tests: same body as vadd_stream.cpp, but
# with the trip count and element type substituted in, so a test can pick a
# larger N (to provoke back-pressure) or a mismatching element type.
_IP_TEMPLATE = """
#include <hls_stream.h>
#include <stdint.h>

extern "C" {{

void vadd_stream(hls::stream<{ctype}> &A, hls::stream<{ctype}> &B,
                 hls::stream<{ctype}> &C) {{
  for (int i = 0; i < {count}; ++i) {{
#pragma HLS pipeline II = 1
    {ctype} a = A.read();
    {ctype} b = B.read();
    C.write(a + b);
  }}
}}

}} // extern "C"
"""


def _make_ip(impl=None):
    # link_hls=False: the shim replaces Vitis's hls_stream.h, so the CPU path
    # does not need vitis_hls on PATH at all.
    return allo.IPModule(
        top="vadd_stream",
        impl=_IMPL if impl is None else impl,
        link_hls=False,
        input_idx=[0, 1],
        output_idx=[2],
    )


def _write_ip(tmp_path, count, ctype="int32_t"):
    path = Path(tmp_path) / "vadd_stream_gen.cpp"
    path.write_text(textwrap.dedent(_IP_TEMPLATE).format(count=count, ctype=ctype))
    return path


def _build_region(vadd_stream, n=N, depth=4):
    """Feeders -> IP -> drain, all at region scope, one kernel each."""

    @df.region()
    def top(A: int32[n], B: int32[n], C: int32[n]):
        sA: Stream[int32, depth]
        sB: Stream[int32, depth]
        sC: Stream[int32, depth]

        @df.kernel(mapping=[1], args=[A])
        def feedA(a: int32[n]):
            for i in range(n):
                sA.put(a[i])

        @df.kernel(mapping=[1], args=[B])
        def feedB(b: int32[n]):
            for i in range(n):
                sB.put(b[i])

        @df.kernel(mapping=[1])
        def ip_wrap():
            vadd_stream(sA, sB, sC)

        @df.kernel(mapping=[1], args=[C])
        def drain(c: int32[n]):
            for i in range(n):
                c[i] = sC.get()

    return top


def _run(mod, n, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.integers(-1000, 1000, n).astype(np.int32)
    b = rng.integers(-1000, 1000, n).astype(np.int32)
    c = np.zeros(n, dtype=np.int32)
    mod(a, b, c)
    np.testing.assert_array_equal(c, a + b)


def test_stream_ip_sim():
    """End-to-end: the IP consumes and produces Allo's FIFOs, results correct."""
    mod = df.build(_build_region(_make_ip()), target="simulator")
    _run(mod, N)


def test_stream_ip_sim_repeated_runs():
    """The same compiled module can be invoked again.

    A run leaves head == tail on every FIFO, so the ring buffers are reusable;
    if an index were left dangling the second run would deadlock or mismatch.
    """
    mod = df.build(_build_region(_make_ip()), target="simulator")
    for seed in range(5):
        _run(mod, N, seed=seed)


def test_stream_ip_sim_backpressure(tmp_path):
    """Depth-2 FIFOs and a long run, so both spin paths are exercised.

    With 256 elements moving through 3-slot buffers, the feeders block on FULL
    and the IP/drain block on EMPTY thousands of times. This is the case that
    fails if the data store/load sits on the wrong side of the index publish.
    """
    n = 256
    ip = _make_ip(_write_ip(tmp_path, count=n))
    mod = df.build(_build_region(ip, n=n, depth=2), target="simulator")
    _run(mod, n, seed=7)


def test_stream_ip_sim_element_type_mismatch(tmp_path):
    """An IP whose stream element type differs from the Allo stream is refused.

    The IP writes Allo's buffer in place, so the two element types have to be
    the same; here the IP says ``hls::stream<int8_t>`` while the region declares
    ``Stream[int32, 4]``.
    """
    ip = _make_ip(_write_ip(tmp_path, count=N, ctype="int8_t"))
    with pytest.raises(TypeError, match="element types must match"):
        df.build(_build_region(ip), target="simulator")


def test_stream_ip_sim_wrapper_shape():
    """The generated wrapper must bind each port to Allo's ring buffer."""
    ip = _make_ip()
    src = Path(ip.generate_stream_sim_wrapper()).read_text()
    # The shim has to be included before the IP body.
    assert src.index("#include <hls_stream.h>") < src.index("vadd_stream.cpp")
    # Three ring-buffer fields per stream, reconstructed via CRunnerUtils.
    for i in range(3):
        for field in ("data", "head", "tail"):
            assert f"DynamicMemRefType<int32_t> s{i}_{field}" in src
        assert f"s{i}_fifo.cap = (int32_t)s{i}_data.sizes[0];" in src
        assert f"hls::stream<int32_t> s{i}(&s{i}_fifo);" in src
    assert "vadd_stream(s0, s1, s2);" in src


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
