# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vitis HLS backend regression tests.

Two layers:

* **Codegen** tests drive a kernel through the schedule interface
  (``kernel.schedule() -> s.<transform>() -> s.export("vitis").hls_code``) and
  assert on the emitted C++. They need no toolchain and run everywhere.
* **Synthesis / simulation** tests are gated on ``is_vitis_available()`` and
  invoke real Vitis HLS via ``s.export("vitis", part=...).synth()`` / csim.

Tests always go through the schedule interface, never a hand-built ``Vitis``.
"""

import re
import tempfile

import numpy as np
import pytest
import os

from allo.operators import math as m
from allo.lang.core import (
    range as arange,
    i32,
    f32,
    bf16,
    APInt,
    Stream,
    Stateful,
    Template,
)
from allo.lang.kernel import kernel
from allo.operators.arith import bitcast as allo_bitcast
from allo.schedule.errors import InvalidScheduleArgumentError
from allo.backend.vitis.utils import is_vitis_available
from allo.backend.vitis.csim import discover_csim
from pathlib import Path

u32 = APInt(32, signed=False)
u256 = APInt(256, signed=False)

_NP_LUT = np.array([10, 20, 30, 40], dtype=np.int32)

PART = "xcvu9p-flga2104-2-i"
requires_vitis = pytest.mark.skipif(
    not is_vitis_available(), reason="Vitis HLS toolchain not detected"
)


def _find_legacy_vitis() -> str | None:
    """A pre-2025.2 Vitis home whose csim routes to the legacy plain-g++ flow."""
    candidates = sorted(Path("/tools/Xilinx/Vitis").glob("*")) + sorted(
        Path("/opt/xilinx").glob("*/Vitis")
    )
    for home in candidates:
        try:
            if discover_csim(home).flavor == "legacy":
                return str(home)
        except Exception:
            continue
    return None


_LEGACY_VITIS = _find_legacy_vitis()
requires_legacy_vitis = pytest.mark.skipif(
    _LEGACY_VITIS is None, reason="No pre-2025.2 Vitis install for legacy csim"
)


def _hls(schedule, **export_kwargs) -> str:
    """Emit the HLS C++ for a scheduled kernel (no toolchain required)."""
    return schedule.export("vitis", **export_kwargs).hls_code


def _contains(code: str, *needles: str):
    for needle in needles:
        assert needle in code, f"expected to find {needle!r} in:\n{code}"


def _regex(code: str, *patterns: str):
    for pattern in patterns:
        assert re.search(pattern, code), f"no match for {pattern!r} in:\n{code}"


# ===========================================================================
# Codegen-text tests (no toolchain)
# ===========================================================================


def test_codegen_bf16():
    @kernel
    def add_bf16(A: bf16[16], B: bf16[16], C: bf16[16]):
        for i in arange(16, name="i"):
            C[i] = A[i] + B[i]

    # force enable ap_float support to test codegen
    os.environ["ALLO_ENABLE_VITIS_APFLOAT"] = "1"
    code = _hls(add_bf16.schedule())
    _contains(
        code,
        "ap_float<16,8> A[16]",
        "ap_float<16,8> B[16]",
        "ap_float<16,8> C[16]",
    )
    os.environ.pop("ALLO_ENABLE_VITIS_APFLOAT")


def test_codegen_vadd_pipeline():
    @kernel
    def vadd(A: f32[16], B: f32[16], C: f32[16]):
        for i in arange(16, name="i"):
            C[i] = A[i] + B[i]

    s = vadd.schedule()
    s.pipeline(s.loop("i"), ii=1)
    code = _hls(s)
    _contains(code, 'extern "C" void vadd(float ', "#pragma HLS pipeline II=1")
    _regex(code, r"= v\d+ \+ v\d+;")


def test_codegen_disable_pipeline():
    @kernel
    def vadd(A: f32[16], B: f32[16], C: f32[16]):
        for i in arange(16, name="i"):
            C[i] = A[i] + B[i]

    s = vadd.schedule()
    s.pipeline(s.loop("i"), ii=-1)
    code = _hls(s)
    _contains(code, 'extern "C" void vadd(float ')
    _contains(code, "#pragma HLS pipeline off")


def test_codegen_auto_pipeline_ii():
    @kernel
    def vadd(A: f32[16], B: f32[16], C: f32[16]):
        for i in arange(16, name="i"):
            C[i] = A[i] + B[i]

    s = vadd.schedule()
    s.pipeline(s.loop("i"), ii=0)
    code = _hls(s)
    assert "II" not in code


def test_codegen_vadd2_tile():
    @kernel
    def vadd2(A: f32[8, 8], B: f32[8, 8], C: f32[8, 8]):
        for i in arange(8, name="i"):
            for j in arange(8, name="j"):
                C[i, j] = A[i, j] + B[i, j]

    s = vadd2.schedule()
    i, j = s.loops("i", "j")
    s.tile((i, j), factors=[4, 4])
    code = _hls(s)
    _contains(code, "void vadd2(float A[8][8]")
    # 8 split by 4 -> a 2-iteration outer band over a 4-iteration inner band.
    _regex(code, r"< 2;", r"< 4;")
    assert code.count("for (") >= 4


def test_codegen_gemm_reorder_pipeline():
    M, K, N = Template("M"), Template("K"), Template("N")

    @kernel(M, K, N)
    def gemm(A: f32[M, K], B: f32[K, N], C: f32[M, N]):
        for i in arange(M, name="i"):
            for j in arange(N, name="j"):
                for k in arange(K, name="k"):
                    C[i, j] += A[i, k] * B[k, j]

    s = gemm[16, 16, 16].schedule()
    s.reorder((s.loop("k"), s.loop("j")))
    s.pipeline(s.loop("j"), ii=1)
    code = _hls(s)
    _contains(code, "void gemm(float A[16][16]", "#pragma HLS pipeline II=1")
    _regex(code, r"= v\d+ \* v\d+;")
    assert code.count("for (") >= 3


def test_codegen_reduction():
    @kernel
    def vsum(A: f32[16], out: f32[1]):
        for i in arange(16, name="i"):
            out[0] += A[i]

    s = vsum.schedule()
    s.pipeline(s.loop("i"), ii=1)
    code = _hls(s)
    _contains(code, "void vsum(float A[16], float out[1])", "#pragma HLS pipeline II=1")
    _regex(code, r"= v\d+ \+ v\d+;")


def test_codegen_stencil():
    @kernel
    def stencil(A: f32[18], B: f32[16]):
        for i in arange(16, name="i"):
            B[i] = A[i] + A[i + 1] + A[i + 2]

    s = stencil.schedule()
    s.pipeline(s.loop("i"), ii=1)
    code = _hls(s)
    _contains(code, "void stencil(float A[18], float B[16])", "#pragma HLS pipeline")
    # three taps summed -> at least two additions
    assert len(re.findall(r"= v\d+ \+ v\d+;", code)) >= 2


def test_codegen_wide_integer():
    @kernel
    def copy256(A: u256[8], B: u256[8]):
        for i in arange(8, name="i"):
            B[i] = A[i]

    code = _hls(copy256.schedule())
    _contains(code, "void copy256(ap_uint<256> A[8], ap_uint<256> B[8])")


def test_codegen_apint_csim_wrapper():
    i5 = APInt(5, signed=True)
    u5 = APInt(5, signed=False)

    @kernel
    def vadd5(A: i5[8], B: u5[8], C: i5[8]):
        for i in arange(8, name="i"):
            C[i] = A[i] + B[i]

    backend = vadd5.schedule().export("vitis")
    # The synthesizable interface keeps the real ap_int boundary.
    _contains(
        backend.hls_code,
        "void vadd5(ap_int<5> A[8], ap_uint<5> B[8], ap_int<5> C[8])",
    )
    # C simulation wraps it with a std-width interface around the renamed kernel,
    # so ctypes can call it (signedness preserved per operand).
    csim_cpp = backend._compile_for_csim().kernel_cpp
    _contains(
        csim_cpp,
        'extern "C" void vadd5(int8_t v0[8], uint8_t v1[8], int8_t v2[8])',
        "void vadd5__impl(ap_int<5>",
        "ap_int<5> v",  # signed temp matches the callee parameter
        "ap_uint<5> v",  # unsigned temp
    )


def test_codegen_bit_slice():
    @kernel
    def bits(x: u32, out: u32[1]):
        y: u32 = x
        y[0:4] = 5
        out[0] = y[4:8]

    code = _hls(bits.schedule())
    _contains(code, "& ~(0xfULL <<", "static_cast<uint32_t>", "ap_uint<4>")
    _regex(code, r">> v\d+\) & 0xfULL")


def test_codegen_signed_max_min():
    # Signless i32 values default to unsigned C++ types, so a signed maxsi/minsi
    # must read its operands as signed before std::max/std::min -- otherwise a
    # negative value compares as a huge unsigned and the result is wrong.
    @kernel
    def clamp(x: i32, out: i32[2]):
        out[0] = max(0, x)
        out[1] = min(0, x)

    code = _hls(clamp.schedule())
    _regex(
        code,
        r"std::max\(.*static_cast<int32_t>",
        r"std::min\(.*static_cast<int32_t>",
    )


@requires_vitis
def test_csim_signed_max():
    @kernel
    def clamp(x: i32, out: i32[1]):
        out[0] = max(0, x)

    out = np.zeros(1, dtype=np.int32)
    with tempfile.TemporaryDirectory() as project:
        clamp.schedule().export("vitis", project_path=project)(-5, out)
    assert int(out[0]) == 0


def test_codegen_bitcast():
    @kernel
    def reinterpret(x: f32, out: u32[1]):
        out[0] = allo_bitcast(x, u32)

    code = _hls(reinterpret.schedule())
    _contains(
        code,
        "template <typename To, typename From> inline To allo_bitcast",
        "allo_bitcast<uint32_t, float>(",
    )


def test_codegen_while_loop():
    @kernel
    def count(A: i32[1]):
        i: i32 = 0
        while i < 10:
            i = i + 1
        A[0] = i

    code = _hls(count.schedule())
    _contains(code, "while (true) {", "break;")
    # Loop-carried var declared and initialized before the loop...
    m = re.search(r"\w+ (v\d+) = \w+;\n\s*while \(true\) \{", code)
    assert m, f"loop var not declared before while in:\n{code}"
    loop_var = m.group(1)
    # ...the after-region yield assigns the next value back into it...
    _regex(code, r"if \(!\(", rf"{loop_var} = \w+;")
    # ...and the while result feeding the store aliases that same variable.
    _contains(code, f"A[0] = {loop_var};")


@requires_vitis
def test_csim_while_loop():
    @kernel
    def count(A: i32[1]):
        i: i32 = 0
        while i < 10:
            i = i + 1
        A[0] = i

    @kernel
    def fib(out: i32[1]):
        a: i32 = 0
        b: i32 = 1
        n: i32 = 0
        while n < 10:
            t: i32 = a + b
            a = b
            b = t
            n = n + 1
        out[0] = a

    out = np.zeros(1, dtype=np.int32)
    with tempfile.TemporaryDirectory() as project:
        count.schedule().export("vitis", project_path=project)(out)
    assert int(out[0]) == 10
    with tempfile.TemporaryDirectory() as project:
        fib.schedule().export("vitis", project_path=project)(out)
    assert int(out[0]) == 55


def test_codegen_match_case():
    @kernel
    def pick(sel: i32, out: i32[1]):
        match sel:
            case 0:
                out[0] = 10
            case 1:
                out[0] = 20
            case _:
                out[0] = 99

    code = _hls(pick.schedule())
    _contains(
        code,
        "switch (",
        "case 0: {",
        "case 1: {",
        "default: {",
        "break;",
    )
    # Each case body is brace-delimited and terminated by a break (no
    # fall-through), and the default arm is emitted exactly once.
    assert code.count("break;") == 3
    assert code.count("default: {") == 1


def test_codegen_match_case_phi():
    # A scalar reassigned across cases becomes an index_switch result: declared
    # once before the switch, assigned inside each arm, and read afterwards.
    @kernel
    def pick(sel: i32, out: i32[1]):
        acc: i32 = 0
        match sel:
            case 0:
                acc = 5
            case 1:
                acc = 7
            case _:
                acc = acc + 100
        out[0] = acc

    code = _hls(pick.schedule())
    _contains(code, "switch (", "case 0: {", "default: {")
    # The phi result is declared before the switch and assigned in each arm.
    m = re.search(r"\w+ (\w+);\n\s*switch \(", code)
    assert m, f"phi result not declared before switch in:\n{code}"
    acc_var = m.group(1)
    assert code.count(f"{acc_var} = ") == 3, f"expected one assign per arm:\n{code}"
    _contains(code, f"out[0] = {acc_var};")


@requires_vitis
def test_csim_match_case():
    @kernel
    def pick(sel: i32, out: i32[1]):
        match sel:
            case 0:
                out[0] = 10
            case 1:
                out[0] = 20
            case _:
                out[0] = 99

    @kernel
    def pick_phi(sel: i32, out: i32[1]):
        acc: i32 = 0
        match sel:
            case 0:
                acc = 5
            case 1:
                acc = 7
            case _:
                acc = acc + 100
        out[0] = acc

    out = np.zeros(1, dtype=np.int32)
    for sel, expected in ((0, 10), (1, 20), (7, 99)):
        with tempfile.TemporaryDirectory() as project:
            pick.schedule().export("vitis", project_path=project)(np.int32(sel), out)
        assert int(out[0]) == expected
    for sel, expected in ((0, 5), (1, 7), (9, 100)):
        with tempfile.TemporaryDirectory() as project:
            pick_phi.schedule().export("vitis", project_path=project)(
                np.int32(sel), out
            )
        assert int(out[0]) == expected


def test_codegen_block_stream_datamover():
    @kernel
    def dmover(inp: i32[4, 4], out: i32[1]):
        fifo: Stream[i32[4, 4]]

        @kernel
        def load(src: i32[4, 4], strm: Stream[i32[4, 4]]):
            strm.put(src)

        @kernel
        def compute(strm: Stream[i32[4, 4]], dst: i32[1]):
            blk = strm.get()
            dst[0] = blk[0, 0]

        load(inp, fifo)
        compute(fifo, out)

    code = _hls(dmover.schedule())
    # Block payload streams element-by-element through a scalar FIFO whose depth
    # is scaled by the block size (2 blocks x 4x4 = 32), not via stream_of_blocks.
    _contains(
        code,
        "hls::stream<int32_t> &",  # callee parameters
        "hls::stream<int32_t> fifo",  # local stream.create (named after `fifo`)
        ".write(",
        ".read()",
        "dmover_load",
        "dmover_compute",
    )
    assert "hls::stream<uint32_t>" not in code
    _regex(code, r"#pragma HLS stream variable=fifo depth=32")
    assert "stream_of_blocks" not in code
    assert "read_lock" not in code


def test_codegen_stream_signed_payload():
    @kernel
    def top(s_in: i32[8], u_in: u32[8], s_out: i32[8], u_out: u32[8]):
        s_fifo: Stream[i32]
        u_fifo: Stream[u32]

        @kernel
        def prod(
            s_src: i32[8],
            u_src: u32[8],
            s_strm: Stream[i32],
            u_strm: Stream[u32],
        ):
            for i in arange(8, name="i"):
                s_strm.put(s_src[i])
                u_strm.put(u_src[i])

        @kernel
        def cons(
            s_strm: Stream[i32],
            u_strm: Stream[u32],
            s_dst: i32[8],
            u_dst: u32[8],
        ):
            for i in arange(8, name="i"):
                s_dst[i] = s_strm.get()
                u_dst[i] = u_strm.get()

        prod(s_in, u_in, s_fifo, u_fifo)
        cons(s_fifo, u_fifo, s_out, u_out)

    code = _hls(top.schedule())
    _contains(
        code,
        "hls::stream<int32_t> &",  # signed callee parameter
        "hls::stream<int32_t> s_fifo",  # signed local stream.create
        "hls::stream<uint32_t> &",  # unsigned callee parameter
        "hls::stream<uint32_t> u_fifo",  # unsigned local stream.create
    )


def test_codegen_streamline_signed_boundary():
    # `streamline` converts a signed-int memref boundary into on-chip FIFOs and
    # generates a `tee` for the fan-out. The boundary's signedness must reach
    # every derived site: the local stream.create, the in-place converted
    # producer/consumer parameters, and the generated tee kernel's parameters --
    # all int32_t, or the dataflow wiring fails to compile.
    N = 8

    @kernel
    def src(X: i32[N, N], T: i32[N, N]):
        for i in arange(N, name="i"):
            for j in arange(N, name="j"):
                T[i, j] = X[i, j] + 1

    @kernel
    def c1(T: i32[N, N], O1: i32[N, N]):
        for i in arange(N, name="i"):
            for j in arange(N, name="j"):
                O1[i, j] = T[i, j] * 2

    @kernel
    def c2(T: i32[N, N], O2: i32[N, N]):
        for i in arange(N, name="i"):
            for j in arange(N, name="j"):
                O2[i, j] = T[i, j] + 3

    @kernel
    def top(X: i32[N, N], O1: i32[N, N], O2: i32[N, N]):
        T: i32[N, N]
        src(X, T)
        c1(T, O1)
        c2(T, O2)

    ts = top.schedule()
    ts.streamline("src", ["c1", "c2"])
    ts.dataflow()
    code = ts.export("vitis").hls_code
    _contains(
        code,
        "void streamline_tee(hls::stream<int32_t> &",  # generated tee kernel
        "hls::stream<int32_t> v",  # local stream.create boundaries
    )
    assert "hls::stream<uint32_t>" not in code


def test_codegen_maxi_interface():
    @kernel
    def axicopy(A: i32[64], B: i32[64]):
        for i in arange(64, name="i"):
            B[i] = A[i] + 1

    backend = axicopy.schedule().export("vitis", part=PART)
    backend.set_axi(0, offset="slave", bundle="gmem")
    backend.set_axi(1, offset="slave", bundle="gmem")
    code = backend.hls_code
    _contains(
        code,
        "#pragma HLS interface mode=m_axi port=A offset=slave bundle=gmem",
        "#pragma HLS interface mode=m_axi port=B offset=slave bundle=gmem",
    )


def test_codegen_fma():
    @kernel
    def fma(a: f32, b: f32, c: f32, d: f32[1]):
        d[0] = m.fma(a, b, c)

    backend = fma.schedule().export("vitis", part=PART)
    code = backend.hls_code
    _contains(code, "hls::fma(")


# ===========================================================================
# Synthesis / simulation tests (gated on a real Vitis HLS toolchain)
# ===========================================================================


@requires_vitis
def test_synth_gemm_tile_pipeline():
    M = N = K = 16

    @kernel
    def gemm(A: f32[M, K], B: f32[K, N], C: f32[M, N]):
        for i in arange(M, name="i"):
            for j in arange(N, name="j"):
                for k in arange(K, name="k"):
                    C[i, j] += A[i, k] * B[k, j]

    s = gemm.schedule()
    s.tile(("i", "j"), factors=[4, 4])
    s.pipeline(s.loop("k"), ii=1)
    with tempfile.TemporaryDirectory() as project:
        mod = s.export("vitis", part=PART, project_path=project)
        mod.synth()
        assert mod.synth_report.exists()


@requires_vitis
def test_synth_block_stream_datamover():
    @kernel
    def dmover(inp: i32[4, 4], out: i32[1]):
        fifo: Stream[i32[4, 4]]

        @kernel
        def load(src: i32[4, 4], strm: Stream[i32[4, 4]]):
            strm.put(src)

        @kernel
        def compute(strm: Stream[i32[4, 4]], dst: i32[1]):
            blk = strm.get()
            dst[0] = blk[0, 0]

        load(inp, fifo)
        compute(fifo, out)

    with tempfile.TemporaryDirectory() as project:
        mod = dmover.schedule().export("vitis", part=PART, project_path=project)
        mod.synth()
        assert mod.synth_report.exists()


@requires_vitis
def test_csim_vadd():
    @kernel
    def vadd(A: f32[16], B: f32[16], C: f32[16]):
        for i in arange(16, name="i"):
            C[i] = A[i] + B[i]

    s = vadd.schedule()
    s.pipeline(s.loop("i"), ii=1)
    a = np.random.rand(16).astype(np.float32)
    b = np.random.rand(16).astype(np.float32)
    c = np.zeros(16, dtype=np.float32)
    with tempfile.TemporaryDirectory() as project:
        backend = s.export("vitis", project_path=project)
        backend(a, b, c)
    np.testing.assert_allclose(c, a + b, rtol=1e-5)


@requires_vitis
def test_csim_bitcast():
    # Reinterpret bits between float and int in both directions; bitcast copies
    # the bit pattern verbatim rather than performing a numeric conversion.
    @kernel
    def reinterpret(x: f32, bits_out: u32[1], raw: u32, val_out: f32[1]):
        bits_out[0] = allo_bitcast(x, u32)
        val_out[0] = allo_bitcast(raw, f32)

    bits_out = np.zeros(1, dtype=np.uint32)
    val_out = np.zeros(1, dtype=np.float32)
    raw = np.uint32(0x40490FDB)  # IEEE-754 bits of 3.14159265f
    with tempfile.TemporaryDirectory() as project:
        backend = reinterpret.schedule().export("vitis", project_path=project)
        backend(np.float32(1.0), bits_out, raw, val_out)
    assert int(bits_out[0]) == 0x3F800000  # IEEE-754 bits of 1.0f
    assert val_out[0].view(np.uint32) == raw


@requires_vitis
def test_csim_apint():
    i5 = APInt(5, signed=True)
    u5 = APInt(5, signed=False)

    @kernel
    def addsub(A: i5[8], B: u5[8], C: i5[8]):
        for i in arange(8, name="i"):
            C[i] = A[i] + B[i]

    a = np.array([-4, -3, -2, -1, 0, 1, 2, 3], dtype=np.int8)
    b = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint8)
    c = np.zeros(8, dtype=np.int8)
    with tempfile.TemporaryDirectory() as project:
        backend = addsub.schedule().export("vitis", project_path=project)
        backend(a, b, c)
    # i5 result wraps modulo 2**5 with sign extension back to int8.
    expected = ((a.astype(np.int16) + b + 16) % 32 - 16).astype(np.int8)
    np.testing.assert_array_equal(c, expected)


@requires_legacy_vitis
def test_csim_legacy_apint():
    """Pre-2025.2 installs lack the -fhls-csim clang, so csim falls back to the
    legacy plain-g++ flow; the APInt std-width wrapper must still be bit-accurate."""
    i5 = APInt(5, signed=True)
    u5 = APInt(5, signed=False)

    @kernel
    def addsub(A: i5[8], B: u5[8], C: i5[8]):
        for i in arange(8, name="i"):
            C[i] = A[i] + B[i]

    a = np.array([-4, -3, -2, -1, 0, 1, 2, 3], dtype=np.int8)
    b = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint8)
    c = np.zeros(8, dtype=np.int8)
    with tempfile.TemporaryDirectory() as project:
        backend = addsub.schedule().export(
            "vitis", project_path=project, vitis_home=_LEGACY_VITIS
        )
        assert backend._get_csim_toolchain().flavor == "legacy"
        backend(a, b, c)
    expected = ((a.astype(np.int16) + b + 16) % 32 - 16).astype(np.int8)
    np.testing.assert_array_equal(c, expected)


# ===========================================================================
# Module-level globals: stateful variables + list-initialized constants
#
# Both lower to a mutable `memref.global` and must be emitted as a file-scope
# *definition* (with initializer), not an `extern` declaration -- otherwise the
# csim .so has an undefined symbol. The symbol name follows the unified
# `_allo_<kind>_<func>_<var>_l<line>c<col>` convention.
# ===========================================================================


def test_codegen_stateful_global_definition():
    @kernel
    def counter() -> i32:
        c: Stateful[i32] = 0
        c = c + 1
        return c

    code = _hls(counter.schedule())
    # A defined (not extern) file-scope global, carrying its initializer.
    _regex(code, r"static \w+ _allo_stateful_counter_c_l\d+c\d+ = 0u?;")
    assert "extern uint32_t _allo_stateful" not in code


def test_codegen_list_initialized_buffer_definition():
    @kernel
    def lut(idx: i32) -> i32:
        table: i32[4] = [10, 20, 30, 40]
        return table[idx]

    code = _hls(lut.schedule())
    _regex(
        code,
        r"static \w+ _allo_const_lut_table_l\d+c\d+\[4\] = \{10u?, 20u?, 30u?, 40u?\};",
    )


def test_codegen_numpy_initialized_buffer_definition():
    # A captured NumPy array lowers to the same file-scope constant array as a
    # list initializer.
    @kernel
    def lut(idx: i32) -> i32:
        table: i32[4] = _NP_LUT
        return table[idx]

    code = _hls(lut.schedule())
    _regex(
        code,
        r"static \w+ _allo_const_lut_table_l\d+c\d+\[4\] = \{10u?, 20u?, 30u?, 40u?\};",
    )


def test_codegen_bufferize_strided_slice():
    # `bufferize` emits a module-level private copy kernel (a plain function with
    # an affine.for) that is declared, defined and invoked from the caller.
    @kernel
    def slicecopy(A: i32[8], out: i32[4]):
        new = A.bufferize([1], [4], [2])
        for i in arange(4, name="i"):
            out[i] = new[i]

    code = _hls(slicecopy.schedule())
    _regex(
        code,
        r"void allo_bufferize_slicecopy_A_l\d+c\d+\(int32_t dst\[4\], int32_t src\[8\]\)",
    )
    _contains(
        code,
        "int32_t new_1 = src[((i0 * 2) + 1)];",
        "dst[i0] = new_1;",
        "#pragma HLS pipeline II=1 rewind",
    )


def test_codegen_auto_rewind():
    @kernel
    def copy(A: i32[8, 8], B: i32[8, 8]):
        for i in range(8):
            for j in range(8):
                B[i, j] = A[i, j]

    s = copy.schedule()
    s.pipeline("j", ii=1)
    code = _hls(s)
    _contains(code, "#pragma HLS pipeline II=1 rewind")

    @kernel
    def imperfect(A: i32[8, 8], B: i32[8, 8], out: i32[1]):
        for i in range(8):
            acc: i32 = 0
            for j in range(8):
                B[i, j] = A[i, j]
                acc += B[i, j]
            out[0] += acc

    s = imperfect.schedule()
    s.pipeline("i", ii=1)
    code = _hls(s)
    _contains(code, "#pragma HLS pipeline II=1 rewind")

    s = imperfect.schedule()
    s.pipeline("j", ii=1)
    code = _hls(s)
    assert "#pragma HLS pipeline II=1 rewind" not in code


# A scheduled `partition` on a global-backed buffer (stateful variable / list
# initializer) records the attribute on the file-scope `memref.global`. The
# `array_partition` pragma is function-scoped, so the emitter must re-emit it at
# the top of every function that reads the global -- not next to the file-scope
# static definition.


def test_codegen_partition_list_initialized_global():
    @kernel
    def lut(idx: i32) -> i32:
        table: i32[4] = [10, 20, 30, 40]
        return table[idx]

    s = lut.schedule()
    s.partition(s.buffer("table"), kind=s.Complete)
    code = _hls(s)
    _regex(
        code,
        r"#pragma HLS array_partition variable=_allo_const_lut_table_l\d+c\d+ "
        r"dim=0 complete",
    )
    # The pragma must sit inside the function body, after the inline pragma.
    body = code.split('extern "C" int32_t lut(int32_t idx) {', 1)[1]
    assert "array_partition" in body


def test_codegen_partition_stateful_array():
    @kernel
    def accbuf(idx: i32, x: i32) -> i32:
        st: Stateful[i32[8]] = 0
        st[idx] = st[idx] + x
        return st[idx]

    s = accbuf.schedule()
    s.partition(s.buffer("st"), dim=1, factor=4, kind=s.Block)
    code = _hls(s)
    _regex(
        code,
        r"#pragma HLS array_partition variable=_allo_stateful_accbuf_st_l\d+c\d+ "
        r"dim=1 block factor=4",
    )


# `bind_storage` mirrors `partition`: it stamps an `allo.bind.storage` attribute
# on a buffer's root carrier (function arg / local alloc / global), and the
# emitter turns it into a `#pragma HLS bind_storage`.


def test_codegen_bind_storage_local_buffer():
    @kernel
    def bufk(A: f32[16], C: f32[16]):
        buf: f32[16] = 0.0
        for i in arange(16, name="i0"):
            buf[i] = A[i] * 2.0
        for i in arange(16, name="i1"):
            C[i] = buf[i]

    s = bufk.schedule()
    s.bind_storage(s.buffer("buf"), impl=s.URAM, mem_type=s.RAM_2P)
    code = _hls(s)
    _regex(code, r"#pragma HLS bind_storage variable=buf type=ram_2p impl=uram")


def test_codegen_bind_storage_argument():
    @kernel
    def vadd(A: f32[16], B: f32[16], C: f32[16]):
        for i in arange(16, name="i"):
            C[i] = A[i] + B[i]

    s = vadd.schedule()
    s.bind_storage(s.buffer("A"), impl=s.LUTRAM, mem_type=s.RAM_1P)
    code = _hls(s)
    _contains(code, "#pragma HLS bind_storage variable=A type=ram_1p impl=lutram")


def test_codegen_bind_storage_global():
    @kernel
    def lut(idx: i32) -> i32:
        table: i32[4] = [10, 20, 30, 40]
        return table[idx]

    s = lut.schedule()
    s.bind_storage(s.buffer("table"), impl=s.BRAM, mem_type=s.ROM_1P)
    code = _hls(s)
    _regex(
        code,
        r"#pragma HLS bind_storage variable=_allo_const_lut_table_l\d+c\d+ "
        r"type=rom_1p impl=bram",
    )


@requires_vitis
def test_csim_partitioned_list_initialized_buffer():
    # Partitioning a global-backed buffer must not change its functional result.
    @kernel
    def lut(idx: i32) -> i32:
        table: i32[4] = [10, 20, 30, 40]
        return table[idx]

    s = lut.schedule()
    s.partition(s.buffer("table"), kind=s.Complete)
    with tempfile.TemporaryDirectory() as project:
        backend = s.export("vitis", project_path=project)
        assert [int(backend(i)) for i in range(4)] == [10, 20, 30, 40]


@requires_vitis
def test_csim_stateful_accumulator():
    """A stateful scalar must persist across csim calls on the same backend (the
    .so stays loaded), so the global accumulates rather than re-initializing."""

    @kernel
    def acc(x: i32) -> i32:
        s: Stateful[i32] = 0
        s = s + x
        return s

    with tempfile.TemporaryDirectory() as project:
        backend = acc.schedule().export("vitis", project_path=project)
        assert int(backend(5)) == 5
        assert int(backend(10)) == 15
        assert int(backend(3)) == 18


@requires_vitis
def test_csim_list_initialized_buffer():
    @kernel
    def lut(idx: i32) -> i32:
        table: i32[4] = [10, 20, 30, 40]
        return table[idx]

    with tempfile.TemporaryDirectory() as project:
        backend = lut.schedule().export("vitis", project_path=project)
        assert [int(backend(i)) for i in range(4)] == [10, 20, 30, 40]


@requires_vitis
def test_csim_numpy_initialized_buffer():
    @kernel
    def lut(idx: i32) -> i32:
        table: i32[4] = _NP_LUT
        return table[idx]

    with tempfile.TemporaryDirectory() as project:
        backend = lut.schedule().export("vitis", project_path=project)
        assert [int(backend(i)) for i in range(4)] == [10, 20, 30, 40]


@requires_vitis
def test_synth_stateful_counter():
    @kernel
    def counter() -> i32:
        c: Stateful[i32] = 0
        c = c + 1
        return c

    with tempfile.TemporaryDirectory() as project:
        mod = counter.schedule().export("vitis", part=PART, project_path=project)
        mod.synth()
        assert mod.synth_report.exists()
