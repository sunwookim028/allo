# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration of an HLS IP whose interface uses ``hls::stream<T>`` ports.

The IP under test is ``vadd_stream.cpp``: it reads one element from each of two
input streams, adds them, and writes the sum to an output stream. Around it we
build a dataflow region with feeder kernels (that put into the input streams), a
thin wrapper kernel that calls the IP, and a drain kernel (that gets from the
output stream). See ``docs/IP_STREAM_INTEGRATION.md`` for the design rationale.
"""

import tempfile
from pathlib import Path

import pytest

import allo
from allo.ir.types import int32, Stream
import allo.dataflow as df
import allo.backend.hls as hls

N = 32
_IMPL = Path(__file__).resolve().parent / "vadd_stream.cpp"


def _make_ip():
    # link_hls=False so the test does not require vitis_hls on PATH just to
    # construct the module. input_idx/output_idx declare stream direction:
    # A(0), B(1) are read by the IP; C(2) is written by it.
    return allo.IPModule(
        top="vadd_stream",
        impl=_IMPL,
        link_hls=False,
        input_idx=[0, 1],
        output_idx=[2],
    )


def _build_region(vadd_stream):
    @df.region()
    def top(A: int32[N], B: int32[N], C: int32[N]):
        sA: Stream[int32, 4]
        sB: Stream[int32, 4]
        sC: Stream[int32, 4]

        @df.kernel(mapping=[1], args=[A])
        def feedA(a: int32[N]):
            for i in range(N):
                sA.put(a[i])

        @df.kernel(mapping=[1], args=[B])
        def feedB(b: int32[N]):
            for i in range(N):
                sB.put(b[i])

        @df.kernel(mapping=[1])
        def ip_wrap():
            vadd_stream(sA, sB, sC)

        @df.kernel(mapping=[1], args=[C])
        def drain(c: int32[N]):
            for i in range(N):
                c[i] = sC.get()

    return top


def test_parser_recognizes_stream():
    """The C++ parser marks ``hls::stream<T> &`` ports with the STREAM sentinel."""
    from allo.backend.ip import parse_cpp_function, STREAM

    with open(_IMPL, "r", encoding="utf-8") as f:
        args = parse_cpp_function(f.read(), "vadd_stream")
    assert args is not None
    assert all(shape is STREAM for _, shape in args)
    assert all("hls::stream" in t for t, _ in args)


def test_stream_ip_codegen():
    """Emitting the HLS project does not require vitis_hls; check the C++ shape."""
    vadd_stream = _make_ip()
    top = _build_region(vadd_stream)

    with tempfile.TemporaryDirectory() as tmpdir:
        mod = df.build(top, target="vitis_hls", mode="csyn", project=tmpdir)
        code = mod.hls_code
        # The IP-calling kernel must expose the streams by reference, and the
        # call itself must appear.
        assert "hls::stream< int32_t >&" in code
        assert "vadd_stream(" in code
        # The IP source must be copied into the project, textually included in
        # kernel.cpp, and registered with the HLS tool.
        assert (Path(tmpdir) / "vadd_stream.cpp").exists()
        assert '#include "vadd_stream.cpp"' in (Path(tmpdir) / "kernel.cpp").read_text()
        assert "add_files vadd_stream.cpp" in (Path(tmpdir) / "run.tcl").read_text()


def test_stream_ip_simulator_rejected():
    """A stream IP cannot run on the CPU simulator; it must fail loudly."""
    vadd_stream = _make_ip()
    top = _build_region(vadd_stream)
    with pytest.raises(NotImplementedError):
        df.build(top, target="simulator")


@pytest.mark.skipif(not hls.is_available(), reason="vitis_hls not available")
def test_stream_ip_csynth():
    """End-to-end: synthesize the design (requires vitis_hls)."""
    vadd_stream = allo.IPModule(
        top="vadd_stream", impl=_IMPL, input_idx=[0, 1], output_idx=[2]
    )
    top = _build_region(vadd_stream)
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = df.build(top, target="vitis_hls", mode="csyn", project=tmpdir)
        mod()


if __name__ == "__main__":
    test_parser_recognizes_stream()
    print("Passed: parser recognizes stream")
    test_stream_ip_codegen()
    print("Passed: stream IP codegen")
    test_stream_ip_simulator_rejected()
    print("Passed: simulator rejected")
