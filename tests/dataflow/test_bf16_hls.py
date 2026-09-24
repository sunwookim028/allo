# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""bf16 through the HLS emitters.

bf16 has always worked in the simulator; until this suite existed it aborted
both C++ emitters with `assert(1 == 0 && "Got unsupported type.")`, a SIGABRT
that killed the interpreter. These tests pin down what each emitter now spells
bf16 as, and -- the part that matters -- that the emitted C++ computes the same
bits the simulator does.

Vitis HLS 2023.2 has no bfloat16 type of its own (no `hls::bfloat16`, no
`ap_bfloat16`, `__bf16` rejected by its clang, and `ap_float<16, 8>` segfaults
csynth), so the Vivado/Vitis emitter ships a shim struct in the generated
header. Catapult does have one: `ac::bfloat16` from ac_std_float.h.
See docs/source/developer/limitations.rst item H.
"""

import subprocess
import tempfile
import os
import shutil

import numpy as np
import ml_dtypes
import pytest

import allo
import allo.dataflow as df
from allo.ir.types import bfloat16, float16, float32, uint16, int32, Stream, Stateful


def _vitis_include():
    """The Vitis HLS include dir, if this host has one (only ap_int.h is used)."""
    for root in ("/opt/xilinx/Vitis_HLS",):
        if not os.path.isdir(root):
            continue
        for ver in sorted(os.listdir(root), reverse=True):
            inc = os.path.join(root, ver, "include")
            if os.path.isfile(os.path.join(inc, "ap_int.h")):
                return inc
    return None


def test_vhls_bf16_type_and_shim():
    """The Vivado/Vitis emitter spells bf16 as allo_bfloat16 and defines it."""

    def mul(A: bfloat16[8], B: bfloat16[8], C: bfloat16[8]):
        for i in range(8):
            C[i] = A[i] * B[i]

    code = str(allo.customize(mul).build(target="vhls"))
    # The type shows up in the signature and in every local declaration...
    assert "allo_bfloat16 v0[8]" in code
    assert "allo_bfloat16 v" in code
    # ...and the header carries the definition, so the kernel is self-contained.
    assert "struct allo_bfloat16 {" in code
    assert "__allo_bf16_round" in code and "__allo_bf16_widen" in code
    # Not the old, wrong answers.
    assert "std::bfloat16_t" not in code
    # ap_float<16, 8> is named in the shim's comment, so check the declarations.
    assert "ap_float<16, 8> v" not in code


def test_vhls_no_bf16_shim_when_unused():
    """A design without bf16 must not grow the shim (output stays identical)."""

    def add(A: int32[8], B: int32[8], C: int32[8]):
        for i in range(8):
            C[i] = A[i] + B[i]

    code = str(allo.customize(add).build(target="vhls"))
    assert "allo_bfloat16" not in code


def test_catapult_bf16_type():
    """Catapult has a real bf16: ac::bfloat16, from ac_std_float.h."""

    def mul(A: bfloat16[8], B: bfloat16[8], C: bfloat16[8]):
        for i in range(8):
            C[i] = A[i] * B[i]

    s = allo.customize(mul)
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="catapult", mode="csyn", project=tmpdir)
    code = mod.hls_code
    assert "ac::bfloat16 v0[8]" in code
    assert "#include <ac_std_float.h>" in code
    # ac::bfloat16's operators round toward zero by default; arith.addf/mulf
    # round to nearest-even, so the header must line the two up -- and the
    # #define has to come before the include that reads it.
    assert "AC_STD_FLOAT_BFLOAT16_ROUND_OVERRIDE AC_RND_CONV" in code
    assert code.index("AC_STD_FLOAT_BFLOAT16_ROUND_OVERRIDE") < code.index(
        "#include <ac_std_float.h>"
    )
    assert "std::bfloat16_t" not in code


def test_vhls_bf16_stream():
    """bf16 as a dataflow Stream element type reaches the emitter as hls::stream."""

    @df.region()
    def top(A: bfloat16[8], B: bfloat16[8]):
        S: Stream[bfloat16, 4]

        @df.kernel(mapping=[1], args=[A])
        def producer(local_A: bfloat16[8]):
            for i in range(8):
                S.put(local_A[i])

        @df.kernel(mapping=[1], args=[B])
        def consumer(local_B: bfloat16[8]):
            for i in range(8):
                local_B[i] = S.get()

    with tempfile.TemporaryDirectory() as tmpdir:
        code = df.build(
            top, target="vitis_hls", project=os.path.join(tmpdir, "top.prj")
        ).hls_code
    assert "hls::stream< allo_bfloat16 >" in code
    assert "struct allo_bfloat16 {" in code


def test_systemc_bf16():
    """SystemC gets ac::bfloat16 too, and needs no Wrapped<> of its own.

    f32 on this target needed a hand-written Connections `Wrapped<>`
    specialization (ac_ieee_float has none); bf16 does not, because
    ac_marshaller.h ships AC_SPECIAL_FLOAT_WRAPPER(ac::bfloat16, 16). What bf16
    does need is the float-shim set the emitter writes per float type: raw-bit
    access for bitcast/memory ports, text input, and a waveform trace.
    """

    @df.region()
    def top(A: bfloat16[8], B: bfloat16[8]):
        S: Stream[bfloat16, 4]

        @df.kernel(mapping=[1], args=[A])
        def producer(local_A: bfloat16[8]):
            for i in range(8):
                S.put(local_A[i] * local_A[i])

        @df.kernel(mapping=[1], args=[B])
        def consumer(local_B: bfloat16[8]):
            for i in range(8):
                local_B[i] = S.get()

    with tempfile.TemporaryDirectory() as tmpdir:
        code = df.build(
            top, target="systemc", mode="csim",
            project=os.path.join(tmpdir, "sc.prj"),
        ).hls_code

    assert "Connections::In< ac::bfloat16 >" in code
    assert "Connections::Fifo< ac::bfloat16, 4 >" in code
    assert "ac::bfloat16 v3 = v2 * v2;" in code
    # the per-float-type shims
    assert "_fbits(const ac::bfloat16 &v)" in code
    assert "operator>>(std::istream &is, ac::bfloat16 &h)" in code
    assert "sc_trace(sc_core::sc_trace_file *tf, const ac::bfloat16 &h," in code
    # the testbench must read bf16 as float text, not as `long long`
    assert "ac::bfloat16 _v; for (int f = 0" in code
    assert "AC_STD_FLOAT_BFLOAT16_ROUND_OVERRIDE AC_RND_CONV" in code


@pytest.mark.skipif(shutil.which("g++") is None, reason="needs g++")
@pytest.mark.skipif(_vitis_include() is None, reason="needs Vitis HLS headers")
def test_vhls_bf16_matches_simulator_bit_for_bit():
    """Compile the emitted kernel and compare its bits with the simulator's.

    This is the test that would have caught a plausible-looking but wrong C++
    type: `half` compiles and synthesizes fine and is silently a different
    number format, and a shim that rounds toward zero instead of to nearest-even
    passes every "does it emit" check while disagreeing here.
    """

    def gemm(A: bfloat16[8, 8], B: bfloat16[8, 8], C: bfloat16[8, 8]):
        for i, j, k in allo.grid(8, 8, 8):
            C[i, j] += A[i, k] * B[k, j]

    s = allo.customize(gemm)
    kernel_src = str(s.build(target="vhls"))

    rng = np.random.default_rng(0)
    A = rng.standard_normal((8, 8)).astype(ml_dtypes.bfloat16)
    B = rng.standard_normal((8, 8)).astype(ml_dtypes.bfloat16)
    C = np.zeros((8, 8), dtype=ml_dtypes.bfloat16)
    s.build()(A, B, C)

    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "kernel.cpp"), "w") as f:
            f.write(kernel_src)
        hdr = kernel_src.split("/// This is top function.", 1)[0]
        with open(os.path.join(d, "kernel_hdr.h"), "w") as f:
            f.write("#pragma once\n" + hdr)
        np.asarray(A).view(np.uint16).tofile(os.path.join(d, "a.bin"))
        np.asarray(B).view(np.uint16).tofile(os.path.join(d, "b.bin"))
        np.asarray(C).view(np.uint16).tofile(os.path.join(d, "c.bin"))
        with open(os.path.join(d, "tb.cpp"), "w") as f:
            f.write(
                """
#include "kernel_hdr.h"
#include <cstdio>
#include <cstdint>
#include <cstdlib>
void gemm(allo_bfloat16 A[8][8], allo_bfloat16 B[8][8], allo_bfloat16 C[8][8]);
static void rd(const char *p, uint16_t *b) {
  FILE *f = fopen(p, "rb");
  if (!f || fread(b, 2, 64, f) != 64) { perror(p); exit(2); }
  fclose(f);
}
int main(int argc, char **argv) {
  uint16_t ab[64], bb[64], cb[64];
  rd(argv[1], ab); rd(argv[2], bb); rd(argv[3], cb);
  allo_bfloat16 A[8][8], B[8][8], C[8][8];
  for (int i = 0; i < 8; ++i)
    for (int j = 0; j < 8; ++j) {
      A[i][j].__b = ab[i * 8 + j];
      B[i][j].__b = bb[i * 8 + j];
      C[i][j].__b = 0;
    }
  gemm(A, B, C);
  int bad = 0;
  for (int i = 0; i < 8; ++i)
    for (int j = 0; j < 8; ++j)
      if (C[i][j].__b != cb[i * 8 + j]) {
        printf("[%d][%d] emitted 0x%04x, simulator 0x%04x\\n", i, j,
               (unsigned)C[i][j].__b, (unsigned)cb[i * 8 + j]);
        ++bad;
      }
  printf("%d mismatches\\n", bad);
  return bad != 0;
}
"""
            )
        exe = os.path.join(d, "tb")
        build = subprocess.run(
            ["g++", "-std=c++14", f"-I{_vitis_include()}", f"-I{d}", "-o", exe,
             os.path.join(d, "tb.cpp"), os.path.join(d, "kernel.cpp")],
            capture_output=True, text=True,
        )
        assert build.returncode == 0, build.stderr
        run = subprocess.run(
            [exe, os.path.join(d, "a.bin"), os.path.join(d, "b.bin"),
             os.path.join(d, "c.bin")],
            capture_output=True, text=True,
        )
        assert run.returncode == 0, run.stdout + run.stderr
        assert "0 mismatches" in run.stdout


def test_bf16_bitcast_round_trip():
    """bf16 -> uint16 -> bf16. The unpack direction needs an explicit target.

    Without the argument the result type is derived from the bit count alone,
    and 16 bits is float16 by that rule -- a bf16 could be packed but never
    unpacked.
    """

    def rt(A: bfloat16[4], B: bfloat16[4]):
        for i in range(4):
            u: uint16 = A[i].bitcast()
            B[i] = u.bitcast(bfloat16)

    s = allo.customize(rt)
    assert "arith.bitcast" in str(s.module)
    assert "i16 to bf16" in str(s.module)

    a = np.array([1.5, -0.375, 3.25, -7.0], dtype=ml_dtypes.bfloat16)
    b = np.zeros(4, dtype=ml_dtypes.bfloat16)
    s.build()(a, b)
    assert np.array_equal(np.asarray(a).view(np.uint16), np.asarray(b).view(np.uint16))

    code = str(s.build(target="vhls"))
    assert "union { allo_bfloat16 from; uint16_t to;}" in code
    assert "union { uint16_t from; allo_bfloat16 to;}" in code


def test_bitcast_without_target_is_unchanged():
    """The no-argument form still means "guess from the bitwidth"."""

    def kernel(A: uint16[4], B: float32[4]):
        for i in range(4):
            B[i] = 0.0

    def pack(A: bfloat16[4], B: uint16[4]):
        for i in range(4):
            B[i] = A[i].bitcast()

    s = allo.customize(pack)
    assert "bf16 to i16" in str(s.module)


def test_bitcast_target_must_match_width():
    def bad(A: bfloat16[4], B: float32[4]):
        for i in range(4):
            B[i] = A[i].bitcast(float32)

    # Allo turns a type-inference error into a printed diagnostic + SystemExit.
    with pytest.raises(BaseException) as err:
        allo.customize(bad)
    assert isinstance(err.value, (SystemExit, RuntimeError))


# ---------------------------------------------------------------------------
# Narrow-float ARRAY INITIALIZERS (not the type table -- the value printer)
# ---------------------------------------------------------------------------
# The Vivado emitter's dense-initializer printer tested `type.isF32()` and then
# `type.isF64()`, and sent everything else to "array has unsupported element
# type." Any float narrower than f64 that is not exactly f32 -- so f16 AS WELL
# AS bf16 -- fell off that chain. It is reachable only through an initialized
# array, which in practice means a region-scope `@ Stateful` (a local array's
# init lowers to stores, not to a dense attribute), which is why the existing
# bf16 coverage above never hit it: it uses arguments and streams.
#
# This is emphatically NOT a bf16 bug. float16 is included as the control that
# proves it: before the fix BOTH failed and only float32 passed.
@pytest.mark.parametrize("dtype", [float32, float16, bfloat16])
@pytest.mark.parametrize("target", ["vitis_hls", "catapult", "systemc"])
def test_stateful_array_initializer_narrow_float(dtype, target):
    if target == "catapult" and dtype is float16:
        pytest.skip(
            "separate, unrelated defect: a float16 array init emits a bare "
            "`= 0.000000;` double literal, and ac_ieee_float<binary16> has no "
            "implicit conversion from double. Independent of the element-type "
            "chain this test covers -- bf16 and f32 both pass here."
        )
    T = dtype

    @df.region()
    def top(inp: T[4], out: T[4]):
        st: T[4] @ Stateful = 0.0

        @df.kernel(mapping=[1], args=[inp, out])
        def k(a: T[4], b: T[4]):
            for i in range(4):
                st[i] = a[i]
            for i in range(4):
                b[i] = st[i]

    # Emission alone is the assertion: the defect was a hard emit failure.
    df.build(top, target=target)



if __name__ == "__main__":
    pytest.main([__file__])
