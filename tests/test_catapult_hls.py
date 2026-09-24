# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
import pytest
import numpy as np
import allo
from allo.ir.types import int32, int8, int16, float32


def check_catapult():
    """Check if Catapult HLS is available via MGC_HOME environment variable."""
    if "MGC_HOME" not in os.environ:
        return False
    return True


# =============================================================================
# Code Generation Tests (no Catapult installation required)
# =============================================================================


def test_catapult_vvadd():
    """Test basic vector addition for Catapult HLS"""

    def vvadd(a: int32[100], b: int32[100]) -> int32[100]:
        c: int32[100]
        for i in range(100):
            c[i] = a[i] + b[i]
        return c

    s = allo.customize(vvadd)
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="catapult", mode="csyn", project=tmpdir)

        # Check Catapult-specific headers
        assert "#include <ac_int.h>" in mod.hls_code
        assert "#include <ac_fixed.h>" in mod.hls_code
        assert "#include <ac_channel.h>" in mod.hls_code

        # Check function is generated
        assert "void vvadd(" in mod.hls_code
        print("test_catapult_vvadd passed!")


def test_catapult_gemm():
    """Test matrix multiplication for Catapult HLS"""

    def gemm(A: int32[32, 32], B: int32[32, 32]) -> int32[32, 32]:
        C: int32[32, 32] = 0
        for i, j, k in allo.grid(32, 32, 32, name="C"):
            C[i, j] += A[i, k] * B[k, j]
        return C

    s = allo.customize(gemm)
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="catapult", mode="csyn", project=tmpdir)

        # Check function signature
        assert "void gemm(" in mod.hls_code
        # Check nested loops are generated
        assert "for (" in mod.hls_code
        print("test_catapult_gemm passed!")


def test_catapult_different_types():
    """Test various data types for Catapult HLS"""

    def type_test(a: int8[10], b: int16[10], c: int32[10]) -> int32[10]:
        d: int32[10]
        for i in range(10):
            d[i] = a[i] + b[i] + c[i]
        return d

    s = allo.customize(type_test)
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="catapult", mode="csyn", project=tmpdir)

        # Check that int8 and int16 types are used
        assert "int8_t" in mod.hls_code or "ac_int<8" in mod.hls_code
        assert "int16_t" in mod.hls_code or "ac_int<16" in mod.hls_code
        assert "int32_t" in mod.hls_code
        print("test_catapult_different_types passed!")


def test_catapult_float():
    """Test floating point operations for Catapult HLS"""

    def float_add(a: float32[10], b: float32[10]) -> float32[10]:
        c: float32[10]
        for i in range(10):
            c[i] = a[i] + b[i]
        return c

    s = allo.customize(float_add)
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="catapult", mode="csyn", project=tmpdir)

        # F32 must come out as ac_ieee_float<binary32>, NOT native `float`.
        # nangate-45nm_beh does not synthesize native C++ float -- Catapult
        # stops with CIN-291 ("Type 'float' is not synthesizable with library
        # 'nangate-45nm_beh'"), which docs/source/backends/catapult.rst has
        # described as the emitter's behaviour since before it was true. The
        # SystemC-emitter merge makes it true; this assertion used to accept
        # the unsynthesizable spelling.
        assert "ac_ieee_float<binary32>" in mod.hls_code
        assert "float " not in mod.hls_code
        print("test_catapult_float passed!")


def test_catapult_conditional():
    """Test conditional statements for Catapult HLS"""

    def conditional(a: int32[10], b: int32[10]) -> int32[10]:
        c: int32[10]
        for i in range(10):
            if a[i] > b[i]:
                c[i] = a[i]
            else:
                c[i] = b[i]
        return c

    s = allo.customize(conditional)
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="catapult", mode="csyn", project=tmpdir)

        # Check if-else is generated
        assert "if (" in mod.hls_code
        assert "else" in mod.hls_code
        print("test_catapult_conditional passed!")


def test_catapult_pipeline():
    """Test pipeline pragma for Catapult HLS"""

    def pipelined_loop(a: int32[100]) -> int32[100]:
        b: int32[100]
        for i in range(100):
            b[i] = a[i] * 2
        return b

    s = allo.customize(pipelined_loop)
    s.pipeline("i")

    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="catapult", mode="csyn", project=tmpdir)

        # Check Catapult-specific pipeline pragma
        assert "#pragma hls_pipeline_init_interval" in mod.hls_code
        # The pragma MUST precede the loop it binds to -- Catapult drops an in-body
        # pragma with CIN-319 ("cannot bind pragma"), silently disabling pipelining.
        lines = mod.hls_code.splitlines()
        idx = next(
            i for i, l in enumerate(lines) if "hls_pipeline_init_interval" in l
        )
        nxt = lines[idx + 1].lstrip()
        assert nxt.startswith(("l_", "for ", "while ")), (
            "hls_pipeline_init_interval must be emitted immediately before the loop "
            f"(Catapult binds to the following construct); next line was: {nxt!r}"
        )
        print("test_catapult_pipeline passed!")


def test_catapult_unroll():
    """Test unroll pragma for Catapult HLS"""

    def unrolled_loop(a: int32[10]) -> int32[10]:
        b: int32[10]
        for i in range(10):
            b[i] = a[i] + 1
        return b

    s = allo.customize(unrolled_loop)
    s.unroll("i")

    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="catapult", mode="csyn", project=tmpdir)

        # Check Catapult-specific unroll pragma
        assert "#pragma hls_unroll" in mod.hls_code
        print("test_catapult_unroll passed!")


def test_catapult_2d_array():
    """Test 2D array operations for Catapult HLS"""

    def matrix_add(A: int32[4, 4], B: int32[4, 4]) -> int32[4, 4]:
        C: int32[4, 4]
        for i, j in allo.grid(4, 4):
            C[i, j] = A[i, j] + B[i, j]
        return C

    s = allo.customize(matrix_add)
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="catapult", mode="csyn", project=tmpdir)

        # Check 2D array declaration
        assert "[4][4]" in mod.hls_code
        print("test_catapult_2d_array passed!")


def test_catapult_nested_function():
    """Test nested function calls for Catapult HLS"""

    def inner(a: int32, b: int32) -> int32:
        return a + b

    def outer(x: int32[10], y: int32[10]) -> int32[10]:
        z: int32[10]
        for i in range(10):
            z[i] = inner(x[i], y[i])
        return z

    s = allo.customize(outer)
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="catapult", mode="csyn", project=tmpdir)

        # Check both functions are generated
        assert "void inner(" in mod.hls_code or "int32_t inner(" in mod.hls_code
        assert "void outer(" in mod.hls_code
        print("test_catapult_nested_function passed!")


def test_catapult_tcl_generation():
    """Test TCL script generation for Catapult HLS"""

    def simple_add(a: int32[10], b: int32[10]) -> int32[10]:
        c: int32[10]
        for i in range(10):
            c[i] = a[i] + b[i]
        return c

    s = allo.customize(simple_add)
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="catapult", mode="csyn", project=tmpdir)

        # Check TCL file exists and has correct content
        tcl_path = os.path.join(tmpdir, "run.tcl")
        assert os.path.exists(tcl_path), "TCL file should be generated"

        with open(tcl_path, "r", encoding="utf-8") as f:
            tcl_content = f.read()

        # Check Catapult-specific TCL commands
        assert "directive set -DESIGN_HIERARCHY" in tcl_content
        assert "directive set -CLOCKS" in tcl_content
        assert "go analyze" in tcl_content
        assert "go compile" in tcl_content
        assert "go assembly" in tcl_content
        assert "go extract" in tcl_content
        print("test_catapult_tcl_generation passed!")


# =============================================================================
# Customization Tests (no Catapult installation required)
# =============================================================================


def test_catapult_partition():
    """Test array partitioning for Catapult HLS"""

    def partition_test(A: int32[10, 10]) -> int32[10, 10]:
        B: int32[10, 10]
        for i, j in allo.grid(10, 10):
            B[i, j] = A[i, j] + 1
        return B

    s = allo.customize(partition_test)
    s.partition(s.A, dim=1)
    s.partition(s.B, dim=1)

    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="catapult", mode="csyn", project=tmpdir)
        # Check that we do NOT emit Vivado HLS pragmas
        assert "#pragma HLS array_partition" not in mod.hls_code
        # For now, we assume implicit partitioning or handled via other means.
        # Ideally we should check if generated code handles parallel access if unrolled.
        # But here we just check we don't emit wrong pragmas.
        print("test_catapult_partition passed!")


def test_catapult_parallel():
    """Test parallel loop for Catapult HLS"""

    def parallel_test(A: int32[10]) -> int32[10]:
        B: int32[10]
        for i in allo.grid(10):
            B[i] = A[i] * 2
        return B

    s = allo.customize(parallel_test)
    s.parallel("i")

    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="catapult", mode="csyn", project=tmpdir)
        # Check for unroll pragma which is often used for parallel loops in HLS
        # Usually full unroll = parallel.
        assert "#pragma hls_unroll" in mod.hls_code
        print("test_catapult_parallel passed!")


# =============================================================================
# CSIM Tests (requires Catapult installation with MGC_HOME)
# =============================================================================


@pytest.mark.skipif(
    not check_catapult(), reason="Catapult is not installed (MGC_HOME not set)"
)
def test_catapult_csim():
    """Test csim flow using g++ for Catapult HLS"""

    def vvadd(a: int32[10], b: int32[10]) -> int32[10]:
        c: int32[10]
        for i in range(10):
            c[i] = a[i] + b[i]
        return c

    s = allo.customize(vvadd)
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="catapult", mode="csim", project=tmpdir)

        # Create input data
        a = np.random.randint(0, 100, (10,)).astype(np.int32)
        b = np.random.randint(0, 100, (10,)).astype(np.int32)
        c = np.zeros(10, dtype=np.int32)

        # Run csim
        mod(a, b, c)

        # Verify results
        expected = a + b
        np.testing.assert_array_equal(c, expected)
        print("test_catapult_csim passed!")


@pytest.mark.skipif(
    not check_catapult(), reason="Catapult is not installed (MGC_HOME not set)"
)
def test_catapult_csim_multiply():
    """Test csim flow with element-wise multiplication"""

    def vmul(a: int32[16], b: int32[16]) -> int32[16]:
        c: int32[16]
        for i in range(16):
            c[i] = a[i] * b[i]
        return c

    s = allo.customize(vmul)
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="catapult", mode="csim", project=tmpdir)

        # Create input data (use small values to avoid overflow)
        a = np.random.randint(0, 100, (16,)).astype(np.int32)
        b = np.random.randint(0, 100, (16,)).astype(np.int32)
        c = np.zeros(16, dtype=np.int32)

        # Run csim
        mod(a, b, c)

        # Verify results
        expected = a * b
        np.testing.assert_array_equal(c, expected)
        print("test_catapult_csim_multiply passed!")


# =============================================================================
# CSYNTH Tests (requires Catapult installation with MGC_HOME)
# =============================================================================


@pytest.mark.skipif(
    not check_catapult(), reason="Catapult is not installed (MGC_HOME not set)"
)
def test_catapult_csynth_vvadd():
    """Test csynth flow for basic vector addition with Catapult HLS"""

    def vvadd(a: int32[10], b: int32[10]) -> int32[10]:
        c: int32[10]
        for i in range(10):
            c[i] = a[i] + b[i]
        return c

    s = allo.customize(vvadd)
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="catapult", mode="csyn", project=tmpdir)

        # Run synthesis (no arguments for csyn mode)
        mod()

        # Check that synthesis outputs are generated
        # Catapult generates output in the project directory
        print("test_catapult_csynth_vvadd passed!")


@pytest.mark.skipif(
    not check_catapult(), reason="Catapult is not installed (MGC_HOME not set)"
)
def test_catapult_csynth_2d_array():
    """Test csynth flow for 2D array operations with Catapult HLS"""

    def matrix_add(A: int32[4, 4], B: int32[4, 4]) -> int32[4, 4]:
        C: int32[4, 4]
        for i, j in allo.grid(4, 4):
            C[i, j] = A[i, j] + B[i, j]
        return C

    s = allo.customize(matrix_add)
    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="catapult", mode="csyn", project=tmpdir)

        # Run synthesis
        mod()

        print("test_catapult_csynth_2d_array passed!")


@pytest.mark.skipif(
    not check_catapult(), reason="Catapult is not installed (MGC_HOME not set)"
)
def test_catapult_csynth_with_pipeline():
    """Test csynth flow with pipeline pragma"""

    def pipelined_add(a: int32[16], b: int32[16]) -> int32[16]:
        c: int32[16]
        for i in range(16):
            c[i] = a[i] + b[i]
        return c

    s = allo.customize(pipelined_add)
    s.pipeline("i")

    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="catapult", mode="csyn", project=tmpdir)

        # Verify pipeline pragma is present
        assert "#pragma hls_pipeline_init_interval" in mod.hls_code

        # Run synthesis
        mod()

        print("test_catapult_csynth_with_pipeline passed!")


@pytest.mark.skipif(
    not check_catapult(), reason="Catapult is not installed (MGC_HOME not set)"
)
def test_catapult_csynth_with_unroll():
    """Test csynth flow with unroll pragma"""

    def unrolled_add(a: int32[8], b: int32[8]) -> int32[8]:
        c: int32[8]
        for i in range(8):
            c[i] = a[i] + b[i]
        return c

    s = allo.customize(unrolled_add)
    s.unroll("i", factor=4)

    with tempfile.TemporaryDirectory() as tmpdir:
        mod = s.build(target="catapult", mode="csyn", project=tmpdir)

        # Verify unroll pragma is present
        assert "#pragma hls_unroll" in mod.hls_code

        # Run synthesis
        mod()

        print("test_catapult_csynth_with_unroll passed!")


# =============================================================================
# mode="ppa": the power flow (no Catapult installation required)
# =============================================================================

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POWER_RPT = os.path.join(
    REPO, "dev/records/catapult_handoff/zhang21_power_2026-09-24/power.rpt"
)


def _mac16():
    def mac16(a: int8[16], b: int8[16]) -> int32:
        acc: int32 = 0
        for i in range(16):
            acc += a[i] * b[i]
        return acc

    return allo.customize(mac16)


def test_ppa_requires_a_testbench():
    """No testbench, no activity, no power: this must fail at build(), loudly."""
    s = _mac16()
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="testbench"):
            s.build(target="catapult", mode="ppa", project=tmpdir)


def test_ppa_tcl_runs_the_power_steps():
    """The emitted tcl must reproduce the measured working sequence."""
    s = _mac16()
    with tempfile.TemporaryDirectory() as tmpdir:
        tb = os.path.join(tmpdir, "tb.cpp")
        with open(tb, "w", encoding="utf-8") as f:
            f.write("// tb\n")
        prj = os.path.join(tmpdir, "prj")
        s.build(
            target="catapult",
            mode="ppa",
            project=prj,
            configs={"testbench": tb, "ncsim_root": tmpdir, "clock_period": 5.0},
        )
        tcl = open(os.path.join(prj, "run.tcl"), encoding="utf-8").read()
        for line in (
            "flow package option set /SCVerify/USE_NCSIM true",
            f"flow package option set /NCSim/NC_ROOT {tmpdir}",
            "flow package option set /LowPower/SWITCHING_ACTIVITY_TYPE saif",
            'solution file add "$sfd/tb.cpp" -type C++ -exclude true',
            "go extract",
            "go switching",
            "flow run /PowerAnalysis/report_pre_pwropt_Verilog",
        ):
            assert line in tcl, line
        # the testbench is copied next to kernel.cpp, where run.tcl looks for it
        assert os.path.exists(os.path.join(prj, "tb.cpp"))
        # csyn must be unchanged: no power steps
        prj2 = os.path.join(tmpdir, "prj2")
        s.build(target="catapult", mode="csyn", project=prj2)
        assert "go switching" not in open(
            os.path.join(prj2, "run.tcl"), encoding="utf-8"
        ).read()


def test_ppa_refuses_to_emit_an_unresolved_ncsim_root(monkeypatch, tmp_path):
    """Nothing resolves -> raise. NEVER emit an empty /NCSim/NC_ROOT.

    On zhang-21 none of NC_ROOT / XCELIUM_HOME / CDS_INST_DIR are set and `xrun` is
    not on PATH, so this is the live path there, not a hypothetical one. An empty
    root would point Catapult at nothing, and a switching step that simulates nothing
    still "succeeds" -- the silent zero this mode exists to prevent.
    """
    import shutil as _shutil
    from allo.backend.catapult import resolve_ncsim_root

    for var in ("NC_ROOT", "XCELIUM_HOME", "CDS_INST_DIR"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(_shutil, "which", lambda name: None)

    for configs in ({}, {"ncsim_root": ""}, {"ncsim_root": "   "}, {"ncsim_root": None}):
        with pytest.raises(ValueError, match="ncsim_root"):
            resolve_ncsim_root(configs)

    s = _mac16()
    tb = tmp_path / "tb.cpp"
    tb.write_text("// tb\n")
    with pytest.raises(ValueError, match="ncsim_root"):
        s.build(
            target="catapult",
            mode="ppa",
            project=str(tmp_path / "prj"),
            configs={"testbench": str(tb)},
        )


def test_ncsim_root_from_xrun_skips_the_tools_component(monkeypatch, tmp_path):
    """Xcelium keeps binaries in <root>/tools/bin; NC_ROOT is the directory above."""
    import shutil as _shutil
    from allo.backend.catapult import resolve_ncsim_root

    for var in ("NC_ROOT", "XCELIUM_HOME", "CDS_INST_DIR"):
        monkeypatch.delenv(var, raising=False)
    xrun = tmp_path / "XCELIUM2403" / "tools" / "bin" / "xrun"
    xrun.parent.mkdir(parents=True)
    xrun.write_text("")
    monkeypatch.setattr(_shutil, "which", lambda name: str(xrun))
    assert resolve_ncsim_root({}) == str(tmp_path / "XCELIUM2403")


def test_parse_real_power_report():
    """Parse the report a real PowerPro run wrote (not an invented format)."""
    from allo.backend.catapult import parse_power_report

    pwr = parse_power_report(POWER_RPT)
    assert pwr["use_mode"] == "pre_pwropt_test_Verilog"
    assert pwr["total"]["total"] == 248.47
    assert pwr["dynamic"]["total"] == 229.67
    assert pwr["static"]["total"] == 18.80
    assert pwr["total"]["clock_network"] == 20.03
    assert pwr["annotation"]["flop_outputs_pct"] == 100.0
    assert pwr["instances"]["mac_core_inst"]["total"] == 247.87


def test_zero_power_is_not_a_result():
    """A report with no activity must raise, not print a zero."""
    import re as _re
    from allo.backend.catapult import assert_power_measured

    text = open(POWER_RPT, encoding="utf-8").read()
    zeroed = _re.sub(
        r"(Dynamic)((?:\s+[\d.]+){5})",
        lambda m: m.group(1) + "     0.00     0.00          0.00          0.00   0.00",
        text,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "power.rpt"), "w", encoding="utf-8") as f:
            f.write(zeroed)
        with pytest.raises(RuntimeError, match="ZERO dynamic power"):
            assert_power_measured(tmpdir)
        os.remove(os.path.join(tmpdir, "power.rpt"))
        with pytest.raises(RuntimeError, match="no usable power report"):
            assert_power_measured(tmpdir)


# =============================================================================
# A @df.region() top, which is what a real design is -- and the two things that
# have to hold for SCVerify to be able to drive one.
# =============================================================================


def _region():
    """A minimal `@df.region()`: a VOID top with array arguments, the shape
    every real design (TinyTPU included) has."""
    import allo.dataflow as df

    @df.region()
    def rtop(A: int32[16], B: int32[16], C: int32[16]):
        @df.kernel(mapping=[1], args=[A, B, C])
        def core(a: int32[16], b: int32[16], c: int32[16]):
            for i in range(16):
                c[i] = a[i] + b[i]

    return df.customize(rtop)


def test_region_top_is_shaped_for_scverify():
    """SCVerify wraps the function marked `#pragma hls_design top`. For a region
    that top is a VOID function with array parameters -- the same shape as the
    `mac16` top SCVerify wrapped on zhang-21 -- and `-DESIGN_HIERARCHY` must name
    it. If either stopped holding, the ppa handoff could not be driven."""
    s = _region()
    with tempfile.TemporaryDirectory() as tmpdir:
        tb = os.path.join(tmpdir, "tb.cpp")
        with open(tb, "w", encoding="utf-8") as f:
            f.write("// tb\n")
        prj = os.path.join(tmpdir, "prj")
        s.build(
            target="catapult",
            mode="ppa",
            project=prj,
            wrap_io=False,
            configs={"testbench": tb, "ncsim_root": tmpdir, "clock_period": 5.0},
        )
        code = open(os.path.join(prj, "kernel.cpp"), encoding="utf-8").read()
        assert "#pragma hls_design top\nvoid rtop(" in code
        assert "int32_t v" in code.split("void rtop(")[1].split(")")[0]
        tcl = open(os.path.join(prj, "run.tcl"), encoding="utf-8").read()
        assert "directive set -DESIGN_HIERARCHY rtop" in tcl
        # No CCS_BLOCK: `#pragma hls_design top` is what Allo emits, and that was
        # enough for SCVerify on the mac16 run.
        assert "USE_CCS_BLOCK" not in tcl


def test_catapult_header_linkage_matches_the_kernel():
    """kernel.h must declare the top the way kernel.cpp DEFINES it.

    The Catapult emitter writes a plain C++ `void <name>(...)`. A header saying
    `extern "C"` does not match it, and anything compiling the project as
    generated -- the emitted Makefile, Catapult's `solution app linkage` csim --
    fails to link. `__call__` used to paper over this by rewriting the header
    just before compiling."""
    s = _mac16()
    with tempfile.TemporaryDirectory() as tmpdir:
        s.build(target="catapult", mode="csyn", project=tmpdir)
        header = open(os.path.join(tmpdir, "kernel.h"), encoding="utf-8").read()
        code = open(os.path.join(tmpdir, "kernel.cpp"), encoding="utf-8").read()
        assert 'extern "C"' not in header
        assert "void mac16(" in header and "void mac16(" in code
        assert 'extern "C"' not in code


def test_ppa_accepts_a_testbench_already_in_the_project():
    """A generator that writes the testbench and the project into ONE handoff
    directory is the normal case; copying a file onto itself must not raise."""
    s = _mac16()
    with tempfile.TemporaryDirectory() as tmpdir:
        tb = os.path.join(tmpdir, "tb.cpp")
        with open(tb, "w", encoding="utf-8") as f:
            f.write("// tb\n")
        s.build(
            target="catapult",
            mode="ppa",
            project=tmpdir,          # the testbench is ALREADY in here
            configs={"testbench": tb, "ncsim_root": tmpdir, "clock_period": 5.0},
        )
        assert open(tb, encoding="utf-8").read() == "// tb\n"
        assert 'solution file add "$sfd/tb.cpp" -type C++ -exclude true' in open(
            os.path.join(tmpdir, "run.tcl"), encoding="utf-8"
        ).read()


# =============================================================================
# Bit ops must be emitted in ac_int, not ap_int -- and must still be RIGHT
#
# TinyTPU's power handoff died in Catapult's C++ front end with 100 errors
# because the Catapult emitter inherited Vitis's bit-op codegen: `ap_int<64> t
# = x; y = t(15, 0);` (CRD-20, 88 times) and implicit narrowing of a >64-bit
# ac_int to int (CRD-413, 12 times). Both reproduce with plain g++ against
# hlslibs ac_types, so both belong here rather than on the licence host. See
# dev/records/catapult_handoff/ppa_tinytpu/zhang21_run_2026-09-24/.
# =============================================================================

_N_BITS = 8


def _bitops_schedule():
    from allo.ir.types import uint64

    def bitops(A: uint64[8], B: int32[8], C: int32[8], D: int32[8]):
        for i in range(8):
            B[i] = A[i][0:16]  # get slice at offset 0
            C[i] = A[i][20:26]  # get slice at a nonzero offset
            x: uint64 = A[i]
            x[8:16] = 0xAB  # set slice
            D[i] = x[0:32] + A[i][40]  # set-slice readback, and a get bit

    return allo.customize(bitops)


def _emit_bitops(tmpdir):
    s = _bitops_schedule()
    s.build(target="catapult", mode="csyn", project=tmpdir)
    with open(os.path.join(tmpdir, "kernel.cpp"), encoding="utf-8") as f:
        return f.read()


def test_catapult_bit_ops_use_ac_int_not_ap_int():
    """The Vitis idiom is not merely unidiomatic here -- Catapult has no
    ap_int, so an emitted ap_int is a hard front-end error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        code = _emit_bitops(tmpdir)
    assert "ap_int" not in code and "ap_uint" not in code
    assert "ap_fixed" not in code
    # and the replacement is actually there
    assert ".slc<16>(0)" in code
    assert ".slc<6>(20)" in code
    assert ".set_slc(8," in code


def test_catapult_wide_ac_int_narrows_explicitly():
    """`int v = <ac_int<65>>;` is CRD-413. Every native-int assignment whose
    right-hand side is a >64-bit ac_int must carry an explicit .to_int64()."""
    import re as _re
    from allo.ir.types import int64

    def wide(A: int64[8], B: int64[8]):
        for i in range(8):
            # index arithmetic wide enough to exceed 64 bits before the cast
            B[i] = A[(i * 2 + 1) % 8] + 1

    s = allo.customize(wide)
    with tempfile.TemporaryDirectory() as tmpdir:
        s.build(target="catapult", mode="csyn", project=tmpdir)
        code = open(os.path.join(tmpdir, "kernel.cpp"), encoding="utf-8").read()

    # names declared as an ac_int wider than 64 bits
    wide_names = set()
    for m in _re.finditer(r"\bac_int<\s*(\d+)\s*,[^>]*>\s+(\w+)\s*=", code):
        if int(m.group(1)) > 64:
            wide_names.add(m.group(2))
    assert wide_names, "the probe kernel no longer produces a >64-bit ac_int"
    for line in code.splitlines():
        m = _re.match(
            r"\s*(?:int|unsigned|u?int\d+_t|long|short|char)\s+\w+\s*=\s*(\w+)\s*;",
            line,
        )
        if m and m.group(1) in wide_names:
            raise AssertionError(f"implicit >64-bit narrowing (CRD-413): {line!r}")


# --- the gate itself -------------------------------------------------------


def test_catapult_gate_rejects_the_vitis_idiom():
    """The text stage needs nothing installed, so it runs everywhere."""
    from allo.backend.catapult import check_emitted_cpp, CatapultEmitError

    with tempfile.TemporaryDirectory() as tmpdir:
        bad = os.path.join(tmpdir, "kernel.cpp")
        with open(bad, "w", encoding="utf-8") as f:
            f.write(
                "#include <ac_int.h>\n"
                "void f(unsigned long long x, unsigned short *y) {\n"
                "  ap_int<64> x_tmp = x;\n"
                "  *y = x_tmp(15, 0);\n"
                "}\n"
            )
        with pytest.raises(CatapultEmitError, match="CRD-20"):
            check_emitted_cpp(bad)


def test_catapult_gate_skips_loudly_without_ac_types(tmp_path, capsys, monkeypatch):
    """No ac_types on the host must be VISIBLE. A silent skip is how a whole
    handoff reached a licence host with 100 errors in it."""
    from allo.backend.catapult import check_emitted_cpp, CatapultEmitError

    monkeypatch.setenv("ALLO_AC_TYPES_INCLUDE", str(tmp_path / "nowhere"))
    monkeypatch.delenv("MGC_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))  # hides ~/.cache/allo
    monkeypatch.delenv("ALLO_AC_TYPES_HOME", raising=False)
    ok = os.path.join(str(tmp_path), "kernel.cpp")
    with open(ok, "w", encoding="utf-8") as f:
        f.write("int f() { return 0; }\n")

    assert check_emitted_cpp(ok, strict=False) is False
    err = capsys.readouterr().err
    assert "SKIPPED" in err and "ac_types" in err
    # ...and a handoff can demand it
    with pytest.raises(CatapultEmitError, match="SKIPPED"):
        check_emitted_cpp(ok, strict=True)


def _ac_types():
    from allo.backend.catapult import find_ac_types_include

    return find_ac_types_include()


@pytest.mark.skipif(_ac_types() is None, reason="no hlslibs ac_types on this host")
def test_catapult_emitted_kernel_compiles_against_ac_types():
    """The gate, on real emitter output. This is what `s.build(target=
    "catapult", ...)` now runs for every project it writes."""
    from allo.backend.catapult import check_emitted_cpp

    with tempfile.TemporaryDirectory() as tmpdir:
        _emit_bitops(tmpdir)
        assert check_emitted_cpp(os.path.join(tmpdir, "kernel.cpp")) is True


@pytest.mark.skipif(_ac_types() is None, reason="no hlslibs ac_types on this host")
def test_catapult_bit_ops_compute_the_same_values_as_the_simulator():
    """Compiling is not enough: a slice with the wrong width, offset or
    signedness compiles and returns the wrong number, which is worse than the
    crash it replaced. So RUN the emitted Catapult C++ against ac_types and
    compare every output with Allo's LLVM simulator, bit for bit."""
    import subprocess

    inc = _ac_types()
    s = _bitops_schedule()
    rng = np.random.default_rng(0)
    A = rng.integers(0, 2**64, size=8, dtype=np.uint64)
    gB, gC, gD = (np.zeros(8, np.int32) for _ in range(3))
    s.build()(A, gB, gC, gD)  # LLVM reference

    with tempfile.TemporaryDirectory() as tmpdir:
        s.build(target="catapult", mode="csyn", project=tmpdir)

        def lit(v, suffix=""):
            return ",".join(str(int(x)) + suffix for x in v)

        drv = (
            "#include <stdint.h>\n"
            '#include "kernel.h"\n'
            "#include <cstdio>\n"
            f"static uint64_t A[8] = {{{lit(A, 'ull')}}};\n"
            f"static int32_t gB[8] = {{{lit(gB)}}};\n"
            f"static int32_t gC[8] = {{{lit(gC)}}};\n"
            f"static int32_t gD[8] = {{{lit(gD)}}};\n"
            "int main() {\n"
            "  int32_t B[8] = {0}, C[8] = {0}, D[8] = {0};\n"
            "  bitops(A, B, C, D);\n"
            "  int bad = 0;\n"
            "  for (int i = 0; i < 8; i++) {\n"
            '    if (B[i] != gB[i]) { printf("B[%d] %d != %d\\n", i, B[i], gB[i]); bad++; }\n'
            '    if (C[i] != gC[i]) { printf("C[%d] %d != %d\\n", i, C[i], gC[i]); bad++; }\n'
            '    if (D[i] != gD[i]) { printf("D[%d] %d != %d\\n", i, D[i], gD[i]); bad++; }\n'
            "  }\n"
            '  printf("mismatches=%d\\n", bad);\n'
            "  return bad != 0;\n"
            "}\n"
        )
        with open(os.path.join(tmpdir, "drv.cpp"), "w", encoding="utf-8") as f:
            f.write(drv)
        cp = subprocess.run(
            ["g++", "-std=c++11", "-I", inc, "-o", "drv", "drv.cpp", "kernel.cpp"],
            cwd=tmpdir,
            capture_output=True,
            text=True,
            check=False,
        )
        assert cp.returncode == 0, cp.stderr
        run = subprocess.run(
            ["./drv"], cwd=tmpdir, capture_output=True, text=True, check=False
        )
        assert run.returncode == 0, run.stdout + run.stderr
        assert "mismatches=0" in run.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
