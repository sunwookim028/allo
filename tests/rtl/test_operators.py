# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Operator injection/characterization, arithmetic datapath binding (legalize-arith keep/expand, compare/select/shift), and reduction restructuring."""

import collections
import math
import os
import re
import shutil
import sys

import numpy as np
import pytest

from allo import kernel
from allo.lang import bf16, f16, f32, f64, i8, i16, i32, i64, u16, u32, KernelOptions
from allo.lang.core import APInt
from allo.lang.ip import operator_ip, OperatorType
from allo.operators import math as amath
from allo.operators import arith as allo_arith
from allo.backend.rtl.devices import default_device
from allo.backend.rtl.device import (
    CombKind,
    Const,
    Interp,
    Linear,
    Piecewise,
    Quadratic,
    Step,
    Table,
    Tiled,
)

sys.path.insert(0, os.path.dirname(__file__))
from _common import (
    Dcp,
    _sched,
    _to_rtl,
    period_need,
    _impls,
    _iis,
    _latency,
    _walk,
    FADD,
    FMUL,
    comb_ns,
    comb_step_ns,
    REG_NS,
    PERIOD_NS,
)  # noqa: E402

pytestmark = pytest.mark.skipif(
    shutil.which("verilator") is None, reason="verilator not available"
)


def _f32(*shape):
    return np.random.default_rng(0).random(shape, dtype=np.float32)


def _signed_f32(seed):
    return (np.random.default_rng(seed).random(16, dtype=np.float32) - 0.5) * 10


# --- what the device declares -------------------------------------------------


# A resource is the device's own vocabulary, so nothing in the compiler names
# `lut` or `dsp`: they are symbols a cost refers to, and the reference is what
# gets verified. A cost's SHAPE is structural and only its coefficients are
# measured, which is why the forms are a closed set and the resources are not.
def test_a_device_declares_its_resources_and_what_they_cost():
    @kernel
    def mac(A: i32[16], B: i32[16], out: i32[16]):
        for i in range(16):
            out[i] = A[i] * B[i] + 1

    dev = default_device.copy()
    lut = dev.resources["lut"]
    dsp = dev.resources["dsp"]
    # An N-bit AND is N LUT6s (linear), a divider is quadratic, and a
    # multiplier's DSP count was measured per width rather than fitted.
    dev.set_comb_delay(CombKind.ADD, 1.2, uses={lut: Linear(1.0)})
    dev.set_comb_delay(CombKind.DIV, 2.5, uses={lut: Quadratic(1.06)})
    dev.set_comb_delay(
        CombKind.MUL, 2.0, uses={lut: Const(15.0), dsp: Table({8: 0, 16: 1, 32: 3})}
    )

    text = _to_rtl(mac, device=dev).dcp
    assert "allo.dcp.resource @lut capacity = 1303680" in text
    assert "allo.dcp.resource @dsp capacity = 9024" in text
    # The cost rides the row it belongs to, referring to the resource by symbol.
    assert "@dsp" in text and "table" in text and "quadratic" in text


# A cost naming something that is not a resource is a verifier error, so a
# misspelled name fails loudly instead of becoming an absent row.
def test_a_cost_must_name_a_declared_resource():
    dev = default_device.copy()
    ghost = dev.add_resource("ghost", capacity=10)
    dev.set_comb_delay(CombKind.ADD, 1.2, uses={ghost: Const(1.0)})
    del dev.resources["ghost"]

    @kernel
    def k(A: i32[8], out: i32[8]):
        for i in range(8):
            out[i] = A[i] + 1

    with pytest.raises(RuntimeError):
        _to_rtl(k, device=dev).dcp


# A device cannot declare the same kind twice: the library keeps the last match,
# so a duplicate would be one declaration silently overriding another.
def test_a_device_declares_each_comb_kind_once():
    dev = default_device.copy()
    lut = dev.resources["lut"]
    dev.set_comb_delay(CombKind.ADD, 1.2, uses={lut: Linear(1.0)})
    dev.set_comb_delay(CombKind.ADD, 0.9)  # overwrites rather than duplicating

    @kernel
    def k(A: i32[8], out: i32[8]):
        for i in range(8):
            out[i] = A[i] + 1

    text = _to_rtl(k, device=dev).dcp
    assert text.count("allo.dcp.comb add delay") == 1


# A multiplexer and a delay chain are structures the emitter builds and nothing
# chooses between, so each is one whole-device row. Both carry TWO parameters,
# and a cost with the wrong number of factors is a verifier error rather than a
# product the evaluator zips short.
def test_a_device_prices_its_multiplexers_and_delay_chains():
    @kernel
    def k(A: i32[8], out: i32[8]):
        for i in range(8):
            out[i] = A[i] + 1

    dev = default_device.copy()
    lut, ff = dev.resources["lut"], dev.resources["ff"]
    dev.set_mux_uses({lut: (Linear(0.4), Linear(1.0))})
    dev.set_chain_uses_norst({ff: (Step(4, 1.0, 2.0), Linear(1.0))})

    text = _to_rtl(k, device=dev).dcp
    assert "allo.dcp.mux uses" in text and "allo.dcp.chain uses" in text
    with pytest.raises(ValueError):
        dev.set_mux_uses({lut: Linear(0.4)})


# An IP core's area rides its own declaration, over the one parameter every
# realization of its kind carries. The resources are the DEVICE's and the
# operator is not in the device's symbol table, so the reference reaches through
# the device symbol and resolves from where it is written.
def test_an_operator_declares_what_its_core_spends():
    @kernel
    def addk(a: f32, b: f32) -> f32:
        return a + b

    text = _to_rtl(addk).dcp
    core = f"add_f32_f32_f32_l{FADD}"
    assert f"allo.dcp.operator @{core}" in text
    scope = default_device.name  # the device the reference reaches through
    # The count is read back off the device rather than restated: what this pins
    # is that the reference resolves through the device symbol.
    luts = dict(default_device.operator_uses[core])["lut"][0].coeffs[0]
    assert f"#allo.res_use<@{scope}::@lut, [<const, [{luts:.6e}]>]>" in text


# A cost is a sum of product terms, so a measured shape that is a sum can be
# declared: an extracted chain's flip-flops are a per-bit term plus a per-stage
# one. The sum is taken before rounding, so the factoring cannot change the
# answer.
def test_a_cost_sums_the_terms_that_name_one_resource():
    @kernel
    def k(A: i32[8], out: i32[8]):
        for i in range(8):
            out[i] = A[i] + 1

    dev = default_device.copy()
    ff = dev.resources["ff"]
    dev.set_chain_uses_norst(
        {ff: [(Const(2.0), Linear(1.0)), (Linear(1.0, base=-1.0), Const(1.0))]}
    )
    assert dev.price(dev.chain_uses_norst, (64, 32))["ff"] == 2 * 32 + 64 - 1
    # Both terms ride one `uses`, naming `@ff` twice.
    chain = [l for l in _to_rtl(k, device=dev).dcp.splitlines() if "dcp.chain" in l]
    assert len(chain) == 1 and chain[0].count("allo.res_use<@ff") == 2


# The device's own evaluator, reached from Python: one implementation of the
# measured shapes, not two. `allo/backend/rtl/qor.py` estimates through this.
def test_the_device_prices_a_realization_through_the_compiler():
    dev = default_device
    # 3 LUTs per bit of a 6-source select, over 32 bits.
    assert dev.price(dev.mux_uses, (6, 32)) == {"lut": 96}
    # A chain past the extraction cliff is SRLs plus a head and tail stage per
    # bit, not `depth * width` flip-flops.
    assert dev.price(dev.chain_uses, (64, 32))["ff"] == 2 * 32 + 64 - 1
    assert dev.price(dev.chain_uses, (2, 32))["ff"] == 64
    # A carry chain is a ceiling: a 17-bit adder takes three CARRY8s. Below 16
    # bits there is no carry chain at all and the adder is three LUTs more.
    assert dev.price(dev.comb_uses["add"], (17,)) == {"lut": 17, "carry8": 3}
    assert dev.price(dev.comb_uses["add"], (9,)) == {"lut": 12, "carry8": 0}
    # A block RAM tile holds 36864 bits however the array is cut.
    assert dev.price(dev.storage["bram"].uses, (1024, 32)) == {"bram36": 1}


def test_interp_interpolates_the_points_a_table_would_hold_flat():
    dev = default_device.copy()
    lut = dev.resources["lut"]
    points = {8: 10.0, 32: 34.0}
    dev.set_comb_delay(CombKind.ADD, 1.2, uses={lut: Interp(points)})
    dev.set_comb_delay(CombKind.SUB, 1.2, uses={lut: Table(points)})
    interp, table = dev.comb_uses["add"], dev.comb_uses["sub"]
    for w, v in points.items():
        assert dev.price(interp, (w,)) == dev.price(table, (w,)) == {"lut": v}
    # Between them the staircase still reads the lower row.
    assert dev.price(interp, (20,)) == {"lut": 22}
    assert dev.price(table, (20,)) == {"lut": 10}


# Below the first point the narrowest measurement bounds the structure asked
# about, so it stands; above the last point there is no such bound.
def test_a_measured_cost_is_bounded_below_and_refused_above():
    dev = default_device.copy()
    lut = dev.resources["lut"]
    points = {8: 10.0, 32: 34.0}
    dev.set_comb_delay(CombKind.ADD, 1.2, uses={lut: Interp(points)})
    dev.set_comb_delay(CombKind.SUB, 1.2, uses={lut: Table(points)})
    for uses in (dev.comb_uses["add"], dev.comb_uses["sub"]):
        assert dev.price(uses, (4,)) == {"lut": 10}
        with pytest.raises(ValueError, match=r"measured over 8\.\.32"):
            dev.price(uses, (40,))
    # A delay row is read the same way, and names the width it was asked at.
    dev.set_comb_delay(CombKind.MUL, Interp({8: 1.0, 64: 4.0}))
    assert dev.comb_delay(CombKind.MUL, 1) == 1.0
    with pytest.raises(ValueError, match=r"measured over 8\.\.64.*at 128 bits"):
        dev.comb_delay(CombKind.MUL, 128)


# An add past the widest measured width is a path that cannot be shown to close
# the period it was scheduled under.
def test_an_operation_outside_the_measured_widths_is_refused():
    i256 = APInt(256, signed=True)

    @kernel
    def wide(a: i256, b: i256) -> i256:
        return a + b

    with pytest.raises(RuntimeError, match=r"'arith.addi' is 256 bits wide"):
        _to_rtl(wide).schedule()
    with pytest.raises(RuntimeError, match=r"'add' row is measured over 8\.\.128"):
        _to_rtl(wide).schedule()


def test_piecewise_selects_an_arm_of_any_shape():
    dev = default_device.copy()
    lut = dev.resources["lut"]
    dev.set_comb_delay(
        CombKind.ADD, 1.2, uses={lut: Piecewise(16.0, Linear(2.0), Quadratic(1.0))}
    )
    uses = dev.comb_uses["add"]
    assert dev.price(uses, (8,)) == {"lut": 16}
    assert dev.price(uses, (16,)) == {"lut": 256}
    assert dev.price(uses, (20,)) == {"lut": 400}


# Which of the two spellings applies follows from the number of factors the
# term carries.
def test_tiled_tiles_one_parameter_or_the_whole_tuple():
    dev = default_device.copy()
    # One factor per parameter: the ceiling is taken on the depth alone, which
    # is `ceil(depth/32)` shift-register sites per bit of width.
    per_param = (("ff", (Tiled(32), Linear(1.0))),)
    assert dev.price(per_param, (256, 8)) == {"ff": math.ceil(256 / 32) * 8}
    # One factor at arity two: the product sits inside the ceiling instead.
    whole_tuple = (("ff", (Tiled(1024),)),)
    assert dev.price(whole_tuple, (256, 8)) == {"ff": math.ceil(256 * 8 / 1024)}
    # The op verifier takes both spellings.
    _parse_device(
        "allo.dcp.chain uses [#allo.res_use<@ff, "
        "[#allo.cost<tiled, [32.0]>, #allo.cost<linear, [0.0, 1.0]>]>]"
    )
    _parse_device(
        "allo.dcp.chain uses [#allo.res_use<@ff, [#allo.cost<tiled, [1024.0]>]>]"
    )


# A cost the evaluator cannot read is reported where it is declared.
def test_a_cost_that_cannot_be_evaluated_is_rejected():
    # `piecewise` chooses between exactly two arms.
    with pytest.raises(Exception, match="piecewise takes 2 arm"):
        _parse_device(
            "allo.dcp.chain uses [#allo.res_use<@ff, [#allo.cost<piecewise, [16.0], "
            "[#allo.cost<const, [1.0]>]>, #allo.cost<const, [1.0]>]>]"
        )
    # Only `tiled` reads a two-parameter tuple from one factor.
    with pytest.raises(Exception, match="takes 2 factor\\(s\\) or one 'tiled'"):
        _parse_device(
            "allo.dcp.chain uses [#allo.res_use<@ff, [#allo.cost<const, [1.0]>]>]"
        )


def _parse_device(body: str):
    """One `dcp.device` body through the compiler's own parser and verifiers."""
    from allo._mlir.ir import Module
    from allo.backend.rtl.device import _scratch_context

    with _scratch_context():
        return Module.parse(
            f"allo.dcp.device @dev {{\n  allo.dcp.resource @ff capacity = 100\n"
            f"  {body}\n}}"
        )


# --- operator injection ------------------------------------------------------


# The same kernel schedules once the operator is characterized via `@operator_ip`.
def test_ip_characterizes_math_op():
    @operator_ip(optype="sqrt", latency=7, pipelined=True, style="ce")
    def fsqrt(a: f32) -> f32: ...

    @kernel
    def sqrtk2(A: f32[8]):
        for i in range(8):
            A[i] = amath.sqrt(A[i])

    dev = default_device.copy()
    dev.add_operator(fsqrt)
    res = _sched(sqrtk2, device=dev)
    assert res.func("sqrtk2").latency is not None


# Integer arithmetic is natively combinational: it needs no `@operator_ip` and no
# library row, so the fail-loud check never fires on it.
def test_integer_ops_never_error():
    @kernel
    def intk(A: i32[8]):
        for i in range(8):
            A[i] = A[i] * 3 + 1

    res = _sched(intk)
    assert res.func("intk").latency is not None


# A custom fast fadd (latency 1) injects as a dcp.operator, is referenced
# by the reifier, and beats the shallowest built-in fadd; the default path is
# untouched (a separate export never sees the IP).
def test_operator_ip_overlay_shifts_schedule():
    @kernel
    def addk(a: f32, b: f32, c: f32) -> f32:
        return a + b + c

    r0 = addk.schedule().export("rtl")
    lat0 = r0.schedule().func("addk").latency

    @operator_ip(
        optype=OperatorType.ADD,
        mnemonic="add_fast",
        latency=1,
        in_delay_ns=0.5,
        pipelined=True,
        style="ce",
    )
    def fadd_fast(a: f32, b: f32) -> f32: ...

    @kernel
    def addk2(a: f32, b: f32, c: f32) -> f32:
        return a + b + c

    dev = default_device.copy()
    dev.add_operator(fadd_fast)
    r1 = addk2.schedule().export("rtl", device=dev)
    lat1 = r1.schedule().func("addk2").latency

    # The fabric's own ladder reaches latency 1 for slower clocks, so the
    # overlay is named apart by its mnemonic and is the faster candidate here.
    assert fadd_fast.symbol not in Dcp(r0).attrs("allo.dcp.operator", "sym_name")
    assert fadd_fast.symbol in Dcp(r1).attrs("allo.dcp.operator", "sym_name")
    assert fadd_fast.symbol in _impls(r1.schedule())
    assert lat0 is not None and lat1 is not None


# Two f32 adders of the same signature are both candidates; `mnemonic` is what
# gives the second core a symbol of its own.
def test_the_shorter_of_two_candidates_is_selected():
    @operator_ip(
        optype=OperatorType.ADD,
        mnemonic="add_dsp",
        latency=FADD - 1,
        in_delay_ns=0.5,
        pipelined=True,
        style="ce",
    )
    def add_dsp(a: f32, b: f32) -> f32: ...

    @kernel
    def addk(A: f32[8], B: f32[8], C: f32[8]):
        for i in range(8):
            C[i] = A[i] + B[i]

    dev = default_device.copy()
    dev.add_operator(add_dsp)
    assert add_dsp.symbol == f"add_dsp_f32_f32_f32_l{FADD - 1}"
    impls = _impls(_to_rtl(addk, device=dev).schedule())
    assert add_dsp.symbol in impls
    assert f"add_f32_f32_f32_l{FADD}" not in impls


def test_a_longer_candidate_does_not_displace_the_builtin():
    @operator_ip(
        optype=OperatorType.ADD,
        mnemonic="add_slow",
        latency=FADD + 2,
        in_delay_ns=0.5,
        pipelined=True,
        style="ce",
    )
    def add_slow(a: f32, b: f32) -> f32: ...

    @kernel
    def addk(A: f32[8], B: f32[8], C: f32[8]):
        for i in range(8):
            C[i] = A[i] + B[i]

    dev = default_device.copy()
    dev.add_operator(add_slow)
    impls = _impls(_to_rtl(addk, device=dev).schedule())
    assert f"add_f32_f32_f32_l{FADD}" in impls
    assert add_slow.symbol not in impls


# An integer multiply matches two rows: the measured DSP core and the
# combinational multiply row at latency 0. An IP outranks a combinational
# realization whatever their latencies compare to.
def test_integer_multiply_binds_its_ip_not_the_comb_row():
    @kernel
    def imulk(x: i32[8], y: i32[8], out: i32[8]):
        for i in range(8):
            out[i] = x[i] * y[i]

    # The fabric offers the multiply at more than one depth; fit against the
    # clock decides which, but whichever depth wins is an IP core, never the
    # combinational row.
    impls = _impls(_to_rtl(imulk).schedule())
    assert any(s.startswith("mul_i32_") for s in impls), impls


def test_advanced_math_sqrt_cosim():
    # A math.sqrt characterized by a unary @ip emits a single-input extern
    # operator and cosims against numpy.sqrt: the operator emit + behavioral
    # model are arity-general, not binary-only.
    N = 16

    @operator_ip(optype="sqrt", latency=5, pipelined=True, style="ce")
    def fsqrt(a: f32) -> f32: ...

    @kernel
    def sqrtk(A: f32[N], B: f32[N]):
        for i in range(N):
            B[i] = amath.sqrt(A[i])

    dev = default_device.copy()
    dev.add_operator(fsqrt)
    rng = np.random.default_rng(0)
    A = rng.random(N, dtype=np.float32).astype(np.float32)  # non-negative
    B = np.zeros(N, np.float32)
    _to_rtl(sqrtk, device=dev).cosim(A, B)
    np.testing.assert_allclose(B, np.sqrt(A), rtol=1e-5, atol=1e-6)


def test_non_pipelined_ip_bounds_the_initiation_interval():
    # A non-pipelined unit takes one input per latency window, so a loop that
    # re-issues it every II cycles needs II >= latency. Nothing else here bounds
    # the interval (two arrays, two ports each, no carried recurrence), so the
    # pipelined twin of the same IP runs at II=1 and the `pipelined` flag alone
    # is the difference.
    #
    # The behavioral model an `@operator_ip` emits accepts an input every cycle
    # whatever the flag says, so the cosim below passes either way. Only the II
    # catches a datapath that would feed a real unit faster than it accepts.
    N, LAT = 16, 3

    def _dev(pipelined):
        @operator_ip(
            optype="sqrt",
            latency=LAT,
            in_delay_ns=0.5,
            pipelined=pipelined,
            # A non-pipelined IP declares no stall style; it takes the ce default.
            style="ce" if pipelined else None,
        )
        def fsqrt(a: f32) -> f32: ...

        dev = default_device.copy()
        dev.add_operator(fsqrt)
        return dev

    @kernel
    def sqrtk4(A: f32[N], B: f32[N]):
        for i in range(N):
            B[i] = amath.sqrt(A[i])

    assert _iis(_sched(sqrtk4, device=_dev(True)).func("sqrtk4").regions) == [1]
    assert _iis(_sched(sqrtk4, device=_dev(False)).func("sqrtk4").regions) == [LAT]

    rng = np.random.default_rng(11)
    A = rng.random(N, dtype=np.float32).astype(np.float32)  # non-negative
    B = np.zeros(N, np.float32)
    _to_rtl(sqrtk4, device=_dev(False)).cosim(A, B)
    np.testing.assert_allclose(B, np.sqrt(A), rtol=1e-5, atol=1e-6)


def test_float_negate_cosim():
    # arith.negf (a float unary minus) lowers to a native comb sign-bit flip
    # with no IP, and cosims bit-exactly against -A.
    N = 16

    @kernel
    def negk(A: f32[N], B: f32[N]):
        for i in range(N):
            B[i] = -A[i]

    rng = np.random.default_rng(1)
    A = (rng.random(N, dtype=np.float32) - np.float32(0.5)).astype(np.float32)
    B = np.zeros(N, np.float32)
    _to_rtl(negk).cosim(A, B)
    np.testing.assert_array_equal(B, -A)  # exact: a sign-bit flip


def test_int_to_float_cast_cosim():
    # An int->float cast (arith.sitofp) is a unary IP: the built-in core emits a
    # single-input extern and cosims against a signed conversion.
    N = 16

    @kernel
    def castk(A: i32[N], B: f32[N]):
        for i in range(N):
            x: f32 = A[i]
            B[i] = x

    rng = np.random.default_rng(2)
    A = rng.integers(-1000, 1000, N).astype(np.int32)
    B = np.zeros(N, np.float32)
    _to_rtl(castk).cosim(A, B)
    np.testing.assert_array_equal(B, A.astype(np.float32))


def test_free_running_operator_cosim():
    # A style='free' operator emits a ce-less extern (a, b, clk) -> y. In
    # a non-back-pressured pipeline (where a ce operator's ce is a constant 1
    # anyway) it cosims identically to the clock-enabled default.
    N = 16

    @operator_ip(
        optype=OperatorType.ADD,
        mnemonic="add_free",
        latency=FADD - 1,
        in_delay_ns=0.5,
        pipelined=True,
        style="free",
    )
    def fadd_free(a: f32, b: f32) -> f32: ...

    @kernel
    def addk(A: f32[N], B: f32[N], C: f32[N]):
        for i in range(N):
            C[i] = A[i] + B[i]

    dev = default_device.copy()
    # Shorter than any of the fabric's own f32 adders, so this is the candidate
    # the operation binds.
    dev.add_operator(fadd_free)
    rtl = _to_rtl(addk, device=dev)
    # The manifest declares each instantiated operator's realized port shape.
    ops = [o for i in rtl.interfaces.values() for o in i.operators]
    free = [o for o in ops if o.module == fadd_free.symbol]
    assert free, "the free operator was not instantiated"
    names = [p.name for p in free[0].ports]
    assert "ce" not in names, f"a free-running extern must carry no ce: {names}"

    rng = np.random.default_rng(3)
    A = (rng.random(N, dtype=np.float32) - np.float32(0.5)).astype(np.float32)
    B = (rng.random(N, dtype=np.float32) - np.float32(0.5)).astype(np.float32)
    C = np.zeros(N, np.float32)
    rtl.cosim(A, B, C)
    np.testing.assert_allclose(C, A + B, rtol=1e-6, atol=1e-6)


def test_custom_c_model_for_uncharacterized_kind_cosim():
    # math.erf has no built-in behavioral model, so a device operator for it
    # needs a user C expression (add_c_model); once supplied, the operator is
    # fully characterized and cosims against a scalar math.erf golden, since the
    # C model is the sole behavior source.
    N = 16

    @kernel
    def erfk(A: f32[N], B: f32[N]):
        for i in range(N):
            B[i] = amath.erf(A[i])

    A = (np.random.default_rng(4).random(N, dtype=np.float32) - np.float32(0.5)).astype(
        np.float32
    )

    @operator_ip(optype="erf", latency=6, pipelined=True, style="ce")
    def ferf(a: f32) -> f32: ...

    ferf.add_c_model("std::erf(a)")
    dev = default_device.copy()
    dev.add_operator(ferf)
    B = np.zeros(N, np.float32)
    _to_rtl(erfk, device=dev).cosim(A, B)
    golden = np.array([math.erf(float(x)) for x in A], np.float32)
    np.testing.assert_allclose(B, golden, rtol=1e-4, atol=1e-6)


# Nothing stalls outside a stream region, so a free-style IP is emitted
# as declared: a plain extern instance with no ce port at all.
def test_free_running_ip_outside_stream_region_emits():
    @operator_ip(
        optype="mul",
        mnemonic="mul_free",
        latency=FMUL - 1,
        in_delay_ns=0.5,
        pipelined=True,
        style="free",
    )
    def freemul(a: f32, b: f32) -> f32: ...

    dev = default_device.copy()
    dev.add_operator(freemul)

    @kernel
    def scale(A: f32[8], B: f32[8]):
        for i in range(8):
            B[i] = A[i] * 2.0

    v = _to_rtl(scale, device=dev).verilog
    assert freemul.symbol in v
    # No `ce` port on a free-running instance: it is the whole difference.
    inst = [ln for ln in v.splitlines() if ".ce" in ln and freemul.symbol in ln]
    assert not inst, inst


# --- legalize-arith: keep vs. expand ------------------------------------------
# The RTL prepare pipeline runs `legalize-arith` (not the device-blind
# `arith-expand`): a composite arith op the device provides an operator IP for is
# KEPT for the scheduler to bind; the rest are EXPANDED into primitive arith.
# Integer max/min are native comb ops and are left untouched either way.


def test_int_max_min_native_comb_cosim():
    # Integer arith.maxsi/minsi are native combinational ops (no IP):
    # legalize-arith leaves them untouched, they schedule at latency 0, and cosim
    # bit-exactly against numpy.maximum/minimum.
    N = 16

    @kernel
    def imaxmin(A: i32[N], B: i32[N], mx: i32[N], mn: i32[N]):
        for i in range(N):
            mx[i] = allo_arith.max(A[i], B[i])
            mn[i] = allo_arith.min(A[i], B[i])

    rtl = _to_rtl(imaxmin)
    kinds = {o.kind for r in rtl.schedule().func("imaxmin").regions for o in r.ops}
    assert {"maxsi", "minsi"} <= kinds  # kept as-is, not expanded

    rng = np.random.default_rng(5)
    A = rng.integers(-50, 50, N).astype(np.int32)
    B = rng.integers(-50, 50, N).astype(np.int32)
    mx = np.zeros(N, np.int32)
    mn = np.zeros(N, np.int32)
    rtl.cosim(A, B, mx, mn)
    np.testing.assert_array_equal(mx, np.maximum(A, B))
    np.testing.assert_array_equal(mn, np.minimum(A, B))


@pytest.mark.parametrize("propagate_nan", [True, False])
def test_float_max_no_ip_expands_cosim(propagate_nan):
    # A float max/min on a device WITHOUT a max/min IP is expanded by
    # legalize-arith into cmpf+select (the compare binds the built-in fcmp IP,
    # the select a comb mux). Both the NaN-propagating (maximumf) and
    # NaN-avoiding (maxnumf) variants expand and cosim bit-exactly.
    N = 16

    @kernel
    def fmaxmin(A: f32[N], B: f32[N], mx: f32[N], mn: f32[N]):
        for i in range(N):
            mx[i] = allo_arith.max(A[i], B[i], propagate_nan=propagate_nan)
            mn[i] = allo_arith.min(A[i], B[i], propagate_nan=propagate_nan)

    rtl = _to_rtl(fmaxmin)
    kinds = {o.kind for r in rtl.schedule().func("fmaxmin").regions for o in r.ops}
    assert "cmpf" in kinds and "select" in kinds  # expanded, not a bare max/min
    assert not (kinds & {"maximumf", "minimumf", "maxnumf", "minnumf"})

    rng = np.random.default_rng(7)
    A = (rng.random(N, dtype=np.float32) - np.float32(0.5)).astype(np.float32)
    B = (rng.random(N, dtype=np.float32) - np.float32(0.5)).astype(np.float32)
    mx = np.zeros(N, np.float32)
    mn = np.zeros(N, np.float32)
    rtl.cosim(A, B, mx, mn)
    np.testing.assert_allclose(mx, np.maximum(A, B), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(mn, np.minimum(A, B), rtol=1e-6, atol=1e-6)


def test_float_max_with_ip_kept_cosim():
    # A float max on a device WITH a matching max IP is KEPT by legalize-arith
    # (not expanded) and bound to that IP, one operator instead of cmp+select.
    # `max` is an OperatorType, so the built-in model table supplies its
    # behavior and no add_c_model is needed.
    N = 16

    @operator_ip(
        optype=OperatorType.MAX, latency=2, in_delay_ns=0.5, pipelined=True, style="ce"
    )
    def fmax_ip(a: f32, b: f32) -> f32: ...

    @kernel
    def fmax_keep(A: f32[N], B: f32[N], out: f32[N]):
        for i in range(N):
            out[i] = allo_arith.max(A[i], B[i], propagate_nan=True)

    dev = default_device.copy()
    dev.add_operator(fmax_ip)
    rtl = _to_rtl(fmax_keep, device=dev)
    kinds = {o.kind for r in rtl.schedule().func("fmax_keep").regions for o in r.ops}
    assert not (kinds & {"cmpf", "select"})  # kept as one op, not expanded
    assert fmax_ip.symbol in Dcp(rtl).attrs("allo.dcp.operator", "sym_name")
    assert fmax_ip.symbol in _impls(rtl.schedule())

    rng = np.random.default_rng(6)
    A = (rng.random(N, dtype=np.float32) - np.float32(0.5)).astype(np.float32)
    B = (rng.random(N, dtype=np.float32) - np.float32(0.5)).astype(np.float32)
    out = np.zeros(N, np.float32)
    rtl.cosim(A, B, out)
    np.testing.assert_allclose(out, np.maximum(A, B), rtol=1e-6, atol=1e-6)


def _wide_add_ip(width):
    wide = APInt(width, signed=True)

    @operator_ip(
        optype=OperatorType.ADD, latency=3, in_delay_ns=0.5, pipelined=True, style="ce"
    )
    def wadd(a: wide, b: wide) -> wide: ...

    return wide, wadd


def test_wide_int_operator_ip_cosim():
    # An operator core at a width the C types have no name for. Its model is
    # native RTL, which carries the port's own 48 bits, so the accumulator wraps
    # in simulation exactly where the declared type says it does.
    from allo.backend.rtl.device import operator_descs
    from allo.backend.rtl.sim import ip_models

    i48, wadd = _wide_add_ip(48)

    @kernel
    def dot(x: i32[8], y: i32[8]) -> i48:
        acc: i48 = 0
        for k in range(8, name="k"):
            acc = acc + x[k] * y[k]
        return acc

    dev = default_device.copy()
    dev.add_operator(wadd)
    # The product narrows to 48 bits, and a 48-bit combinational multiply
    # measures past the default clock's period on this part.
    rtl = _to_rtl(dot, device=dev, freq_mhz=200)
    ops = [o for i in rtl.interfaces.values() for o in i.operators]
    assert [p.width for p in ops[0].ports if p.role == "data"] == [48, 48]
    # No DPI at all: an integer core needs no C, so nothing caps it at 64 bits.
    assert ip_models.dpi_c(rtl.interfaces, operator_descs(dev.operators)) == ""

    rng = np.random.default_rng(0)
    x = rng.integers(-(2**31), 2**31, size=8, dtype=np.int64).astype(np.int32)
    y = rng.integers(-(2**31), 2**31, size=8, dtype=np.int64).astype(np.int32)
    exact = sum(int(a) * int(b) for a, b in zip(x, y))
    assert abs(exact) > 2**47, "inputs must overflow i48 for the wrap to matter"
    assert int(rtl.cosim(x, y).result) == ((exact + 2**47) % 2**48) - 2**47


def test_operator_over_period_derates_the_clock():
    # An operator whose own delay exceeds the target period is not refused:
    # the scheduler lowers the clock to the least period every row fits and
    # reports it. The schedule, the emitted design and the QoR all hold the
    # achieved period; the target stays on the compiler's account of itself.
    wide = APInt(48, signed=True)

    @operator_ip(
        optype=OperatorType.ADD, latency=3, in_delay_ns=9.0, pipelined=True, style="ce"
    )
    def slow_add(a: wide, b: wide) -> wide: ...

    @kernel
    def dot(x: i32[8], y: i32[8]) -> wide:
        acc: wide = 0
        for k in range(8, name="k"):
            acc = acc + x[k] * y[k]
        return acc

    dev = default_device.copy()
    dev.add_operator(slow_add)
    rtl = _to_rtl(dot, device=dev, freq_mhz=200)
    sched = rtl.schedule()
    # 9 ns into the adder's first register, plus the fabric's register floor.
    assert sched.cycle_ns is not None and 9.0 < sched.cycle_ns < 10.0
    assert sched.compiler.options.cycle_ns == pytest.approx(5.0)
    assert rtl.estimation.fmax_target == pytest.approx(1000.0 / sched.cycle_ns)

    rng = np.random.default_rng(0)
    x = rng.integers(-(2**31), 2**31, size=8, dtype=np.int64).astype(np.int32)
    y = rng.integers(-(2**31), 2**31, size=8, dtype=np.int64).astype(np.int32)
    exact = sum(int(a) * int(b) for a, b in zip(x, y))
    assert int(rtl.cosim(x, y).result) == ((exact + 2**47) % 2**48) - 2**47


# Selection ranks period fit above latency: a shallow core whose own delay
# misses the clock loses to a deeper one that holds it, and the clock is not
# derated. At a clock both fit, the shallow one wins back.
def test_a_core_that_misses_the_clock_loses_to_one_that_holds_it():
    wide = APInt(48, signed=True)

    @operator_ip(
        optype=OperatorType.ADD,
        mnemonic="add_shallow",
        latency=2,
        in_delay_ns=9.0,
        pipelined=True,
        style="ce",
    )
    def add_shallow(a: wide, b: wide) -> wide: ...

    @operator_ip(
        optype=OperatorType.ADD,
        mnemonic="add_deep",
        latency=6,
        in_delay_ns=1.0,
        pipelined=True,
        style="ce",
    )
    def add_deep(a: wide, b: wide) -> wide: ...

    @kernel
    def dot(x: i32[8], y: i32[8]) -> wide:
        acc: wide = 0
        for k in range(8, name="k"):
            acc = acc + x[k] * y[k]
        return acc

    dev = default_device.copy()
    dev.add_operator(add_shallow)
    dev.add_operator(add_deep)

    fast = _to_rtl(dot, device=dev, freq_mhz=200).schedule()
    assert add_deep.symbol in _impls(fast)
    assert fast.cycle_ns == pytest.approx(5.0)

    slow = _to_rtl(dot, device=dev, freq_mhz=50).schedule()
    assert add_shallow.symbol in _impls(slow)
    assert slow.cycle_ns == pytest.approx(20.0)


# When another operator derates the clock, selection re-ranks at the achieved
# period: a core that missed the target fits the raised period and wins back
# its shorter latency.
def test_selection_reranks_at_the_derated_period():
    w48 = APInt(48, signed=True)
    w40 = APInt(40, signed=True)

    @operator_ip(
        optype=OperatorType.ADD,
        mnemonic="add_shallow",
        latency=2,
        in_delay_ns=8.0,
        pipelined=True,
        style="ce",
    )
    def add_shallow(a: w48, b: w48) -> w48: ...

    @operator_ip(
        optype=OperatorType.ADD,
        mnemonic="add_deep",
        latency=6,
        in_delay_ns=1.0,
        pipelined=True,
        style="ce",
    )
    def add_deep(a: w48, b: w48) -> w48: ...

    @operator_ip(
        optype=OperatorType.ADD,
        mnemonic="add_slow",
        latency=2,
        in_delay_ns=9.0,
        pipelined=True,
        style="ce",
    )
    def add_slow(a: w40, b: w40) -> w40: ...

    @kernel
    def dot(x: i32[8], y: i32[8]) -> w48:
        acc: w48 = 0
        pre: w40 = 0
        for k in range(8, name="k"):
            pre = pre + x[k] * y[k]
            acc = acc + pre
        return acc

    dev = default_device.copy()
    dev.add_operator(add_shallow)
    dev.add_operator(add_deep)
    dev.add_operator(add_slow)

    sched = _to_rtl(dot, device=dev, freq_mhz=200).schedule()
    # The 40-bit add's only core derates the clock past the 48-bit shallow
    # core's need, so the shallow core is selected, not the deep one.
    assert sched.cycle_ns is not None and 9.0 < sched.cycle_ns < 10.0
    assert add_slow.symbol in _impls(sched)
    assert add_shallow.symbol in _impls(sched)
    assert add_deep.symbol not in _impls(sched)


# A row's warranted period gates it like its boundary cones: the depth-2 float
# adder is a candidate only below its own floor, and a clock past every row's
# warranty derates to the fastest warranted period, where selection re-ranks.
def test_selection_honors_a_rows_warranted_period():
    @kernel
    def axpy(x: f32[8], y: f32[8]):
        for k in range(8):
            y[k] = x[k] + y[k]

    at_default = _to_rtl(axpy).schedule()
    assert "add_f32_f32_f32_l2" in _impls(at_default)

    @kernel
    def ratio(x: f32[8], y: f32[8]):
        for k in range(8):
            y[k] = x[k] / y[k]

    # 450 MHz is past both dividers' warranties: the clock derates to what the
    # 12-cycle row needs for a cycle of its own, where that row wins back over
    # the 10-cycle one.
    derated = _to_rtl(ratio, freq_mhz=450).schedule()
    assert derated.cycle_ns == pytest.approx(period_need("div", "float32", 12))
    assert "div_f32_f32_f32_l12" in _impls(derated)


# The stall contract holds for every candidate core, not only the one selection
# ranks first: which core wins is settled only once the period is.
def test_an_elastic_candidate_is_refused_whichever_core_ranks_first():
    wide = APInt(48, signed=True)

    @operator_ip(
        optype=OperatorType.ADD,
        mnemonic="add_ce",
        latency=2,
        in_delay_ns=0.5,
        pipelined=True,
        style="ce",
    )
    def add_ce(a: wide, b: wide) -> wide: ...

    @operator_ip(
        optype=OperatorType.ADD,
        mnemonic="add_elastic",
        latency=6,
        in_delay_ns=0.5,
        pipelined=True,
        style="elastic",
    )
    def add_elastic(a: wide, b: wide) -> wide: ...

    @kernel
    def addk(x: wide[8], y: wide[8], out: wide[8]):
        for i in range(8):
            out[i] = x[i] + y[i]

    dev = default_device.copy()
    dev.add_operator(add_ce)
    dev.add_operator(add_elastic)
    with pytest.raises(RuntimeError, match="elastic"):
        _to_rtl(addk, device=dev).schedule()


def _selection_cores(wide):
    """Three cores for the solver-side selection tests: a short expensive add,
    a deep cheap add, and a deep multiply to make slack beside itself."""

    @operator_ip(
        optype=OperatorType.ADD,
        mnemonic="add_fast",
        latency=1,
        in_delay_ns=1.0,
        pipelined=True,
        style="ce",
    )
    def add_fast(a: wide, b: wide) -> wide: ...

    @operator_ip(
        optype=OperatorType.ADD,
        mnemonic="add_cheap",
        latency=3,
        in_delay_ns=1.0,
        pipelined=True,
        style="ce",
    )
    def add_cheap(a: wide, b: wide) -> wide: ...

    @operator_ip(
        optype=OperatorType.MUL,
        mnemonic="mul_deep",
        latency=6,
        in_delay_ns=1.0,
        pipelined=True,
        style="ce",
    )
    def mul_deep(a: wide, b: wide) -> wide: ...

    dev = default_device.copy()
    dev.add_operators(add_fast, add_cheap, mul_deep)
    dev.set_operator_uses(add_fast, {dev.resources["lut"]: Const(400.0)})
    dev.set_operator_uses(add_cheap, {dev.resources["lut"]: Const(10.0)})
    return dev, add_fast, add_cheap, mul_deep


# Which core realizes an operation is the exact solver's own decision, made
# with the schedule: the span is settled first, so the add on the drain path
# keeps the short core, and the add whose latency hides in the slack beside
# the deep multiply takes the cheap one. The library's own ranking is latency
# first, so the heuristic ships the short core for both.
def test_exact_selection_spends_slack_on_the_cheaper_core():
    wide = APInt(48, signed=True)
    dev, add_fast, add_cheap, mul_deep = _selection_cores(wide)

    @kernel
    def mix(a: i32[1], b: i32[1], c: i32[1], d: i32[1]) -> wide:
        p: wide = a[0]
        p = p * b[0]
        u: wide = c[0]
        u = u + d[0]
        return p + u

    assert add_cheap.symbol not in _impls(_to_rtl(mix, device=dev).schedule())

    exact = _to_rtl(mix, device=dev).set_scheduler_opt(scheduler="exact")
    impls = _impls(exact.schedule())
    assert {add_fast.symbol, add_cheap.symbol, mul_deep.symbol} <= impls

    vals = [np.array([v], np.int32) for v in (312, -75, 4444, 9)]
    expect = 312 * -75 + (4444 + 9)
    assert int(exact.cosim(*vals).result) == expect


# The same decision inside a pipeline: the carried accumulate cannot take the
# deep add at II=1 (its latency would not close the recurrence), the join on
# the drain path keeps the short core with it, and only the slack add beside
# the multiply goes cheap.
def test_exact_selection_in_a_pipeline_leaves_the_recurrence_its_short_core():
    wide = APInt(48, signed=True)
    dev, add_fast, add_cheap, mul_deep = _selection_cores(wide)

    @kernel
    def loopy(a: i32[8], b: i32[8], c: i32[8]) -> wide:
        acc: wide = 0
        for k in range(8, name="k"):
            p: wide = a[k]
            p = p * p
            u: wide = b[k]
            u = u + c[k]
            acc = acc + (p + u)
        return acc

    exact = _to_rtl(loopy, device=dev).set_scheduler_opt(scheduler="exact")
    sched = exact.schedule()
    impls = _impls(sched)
    assert {add_fast.symbol, add_cheap.symbol, mul_deep.symbol} <= impls

    rng = np.random.default_rng(2)
    a, b, c = (rng.integers(-500, 500, size=8, dtype=np.int32) for _ in range(3))
    expect = sum(int(x) * int(x) + int(y) + int(z) for x, y, z in zip(a, b, c))
    assert int(exact.cosim(a, b, c).result) == expect


# Selection and allocation composed on the same ops: two slack adds behind a
# deep multiply chain converge on ONE row and fold onto ONE instance, because
# the model prices what a converged selection saves (one core plus a select
# against two cores). The multiplies stay unfolded: their core is declared
# free, so folding them would buy nothing and cost the select.
def test_converged_selections_fold_onto_one_instance():
    wide = APInt(48, signed=True)

    @operator_ip(
        optype=OperatorType.ADD,
        mnemonic="add_fast",
        latency=1,
        in_delay_ns=1.0,
        pipelined=True,
        style="ce",
    )
    def add_fast(a: wide, b: wide) -> wide: ...

    @operator_ip(
        optype=OperatorType.ADD,
        mnemonic="add_cheap",
        latency=3,
        in_delay_ns=1.0,
        pipelined=True,
        style="ce",
    )
    def add_cheap(a: wide, b: wide) -> wide: ...

    @operator_ip(
        optype=OperatorType.MUL,
        mnemonic="mul_deep",
        latency=6,
        in_delay_ns=1.0,
        pipelined=True,
        style="ce",
    )
    def mul_deep(a: wide, b: wide) -> wide: ...

    @kernel
    def mix(a: i32[1], b: i32[1], c: i32[1], d: i32[1], e: i32[1], f: i32[1]) -> wide:
        u: wide = a[0]
        u = u + b[0]
        v: wide = c[0]
        v = v + d[0]
        p: wide = e[0]
        p = p * f[0]
        return p * u * v

    dev = default_device.copy()
    dev.add_operators(add_fast, add_cheap, mul_deep)
    # Equal, expensive cores: whichever row wins, converging on it and sharing
    # one instance beats two instances of anything.
    dev.set_operator_uses(add_fast, {dev.resources["lut"]: Const(5000.0)})
    dev.set_operator_uses(add_cheap, {dev.resources["lut"]: Const(5000.0)})

    exact = _to_rtl(mix, device=dev).set_scheduler_opt(scheduler="exact")
    winners = _impls(exact.schedule()) & {add_fast.symbol, add_cheap.symbol}
    assert len(winners) == 1
    winner = winners.pop()
    loser = ({add_fast.symbol, add_cheap.symbol} - {winner}).pop()
    # One module definition plus one instantiation: the two adds share it.
    assert exact.mlir.count(winner) == 2
    assert exact.mlir.count(loser) == 0

    vals = [np.array([x], np.int32) for x in (3, 4, 5, 6, 7, 8)]
    assert int(exact.cosim(*vals).result) == (7 * 8) * (3 + 4) * (5 + 6)


# The same composition inside a pipeline: the recurrence multiply holds the
# loop at II=6, the three slack adds converge on one row, and the modulo model
# folds them onto ONE instance by spreading them over distinct congruence
# slots. At 100 MHz the three-arm select cone fits the period outright, so the
# full fold is the unambiguous optimum; at a tight clock the same model backs
# off to more instances instead.
def test_converged_selections_fold_in_a_pipeline():
    wide = APInt(48, signed=True)
    dev, add_fast, add_cheap, mul_deep = _selection_cores(wide)
    # Equal, expensive cores, as in the acyclic fold test.
    dev.set_operator_uses(add_fast, {dev.resources["lut"]: Const(5000.0)})
    dev.set_operator_uses(add_cheap, {dev.resources["lut"]: Const(5000.0)})

    @kernel
    def prodsum(a: i32[12]) -> wide:
        acc: wide = 1
        for k in range(4, name="k"):
            x: wide = a[3 * k]
            y: wide = a[3 * k + 1]
            z: wide = a[3 * k + 2]
            u: wide = x + y
            v: wide = y + z
            acc = acc * (u + v)
        return acc

    exact = _to_rtl(prodsum, device=dev, freq_mhz=100).set_scheduler_opt(
        scheduler="exact"
    )
    winners = _impls(exact.schedule()) & {add_fast.symbol, add_cheap.symbol}
    assert len(winners) == 1
    winner = winners.pop()
    loser = ({add_fast.symbol, add_cheap.symbol} - {winner}).pop()
    # One module definition plus one instantiation: the three adds share it.
    assert exact.mlir.count(winner) == 2
    assert exact.mlir.count(loser) == 0

    a = np.arange(1, 13, dtype=np.int32)
    expect = 1
    for k in range(4):
        x, y, z = (int(a[3 * k + j]) for j in range(3))
        expect *= x + 2 * y + z
    assert int(exact.cosim(a).result) == expect


def test_behavior_language_follows_the_domain():
    # A core's behavior language follows from the core: an integer one is native
    # RTL (exact at any width), a float one is C over the DPI, and a user
    # `add_c_model` is C whatever the domain.
    from allo.backend.rtl.device import operator_descs
    from allo.backend.rtl.sim import ip_models

    i128, wadd = _wide_add_ip(128)

    # Shorter than the built-in 64-bit multiply core, so this is the candidate
    # the library selects.
    @operator_ip(optype=OperatorType.MUL, latency=1, pipelined=True, style="ce")
    def imul(a: i64, b: i64) -> i64: ...

    imul.add_c_model("a * b")

    @kernel
    def wide(x: i64[4], y: i64[4], out: i64[4], z: f32[4]) -> i128:
        acc: i128 = 0
        for k in range(4, name="k"):
            out[k] = x[k] * y[k]
            z[k] = z[k] + z[k]
            acc = acc + out[k]
        return acc

    dev = default_device.copy()
    dev.add_operators(wadd, imul)
    rtl = _to_rtl(wide, device=dev)
    descs = operator_descs(dev.operators)
    sv = ip_models.sv_models(rtl.interfaces, descs)
    c = ip_models.dpi_c(rtl.interfaces, descs)
    # The 128-bit add is a wire in RTL; the user-modelled multiply and the float
    # add are the only two that reach C.
    assert f"module {wadd.symbol}(" in sv and "wire [127:0] f = a + b;" in sv
    assert wadd.symbol not in c
    assert f"allo_op_{imul.symbol}(" in c
    assert "allo_ld_f32(p0)" in c


def test_max_maxnum_split_binds_distinctly():
    # The Max / MaxNum op-kind split keeps NaN semantics correct: a device that
    # provides a max IP (maximumf, NaN-propagating) binds arith.maximumf but
    # NOT arith.maxnumf (maxNum, returns the non-NaN operand). The latter has
    # no matching IP, so legalize-arith expands it rather than silently computing
    # it with the wrong operator.
    N = 8

    @operator_ip(
        optype=OperatorType.MAX, latency=2, in_delay_ns=0.5, pipelined=True, style="ce"
    )
    def fmax_ip(a: f32, b: f32) -> f32: ...

    dev = default_device.copy()
    dev.add_operator(fmax_ip)

    @kernel
    def kmaximumf(A: f32[N], B: f32[N], o: f32[N]):
        for i in range(N):
            o[i] = allo_arith.max(A[i], B[i], propagate_nan=True)  # arith.maximumf

    @kernel
    def kmaxnumf(A: f32[N], B: f32[N], o: f32[N]):
        for i in range(N):
            o[i] = allo_arith.max(A[i], B[i], propagate_nan=False)  # arith.maxnumf

    assert fmax_ip.symbol in _impls(_to_rtl(kmaximumf, device=dev).schedule())  # bound
    maxnum = _to_rtl(kmaxnumf, device=dev)
    assert fmax_ip.symbol not in _impls(maxnum.schedule())  # NOT bound to the max IP
    k = {o.kind for r in maxnum.schedule().func("kmaxnumf").regions for o in r.ops}
    assert "cmpf" in k and "select" in k  # expanded instead


# --- arithmetic datapath: compare, select, shift ------------------------------


# Reductions and matmuls over the float and integer datapaths: the float ops
# are multi-cycle IP instances, the int add is combinational.
def test_float_and_int_arithmetic():
    @kernel
    def dotp(A: f32[8], B: f32[8], out: f32[1]):
        acc: f32 = 0.0
        for k in range(8):
            acc = acc + A[k] * B[k]
        out[0] = acc

    A, B = _f32(8), _f32(8)
    out = np.zeros(1, np.float32)
    _to_rtl(dotp).cosim(A, B, out)
    assert np.allclose(out[0], A @ B, rtol=1e-4, atol=1e-5)

    @kernel
    def mm(A: f32[4, 4], B: f32[4, 4], C: f32[4, 4]):
        for i in range(4):
            for j in range(4):
                acc: f32 = 0.0
                for k in range(4):
                    acc = acc + A[i, k] * B[k, j]
                C[i, j] = acc

    A, B = _f32(4, 4), _f32(4, 4)
    C = np.zeros((4, 4), np.float32)
    _to_rtl(mm).cosim(A, B, C, timeout=20000)
    assert np.allclose(C, A @ B, rtol=1e-4, atol=1e-5)

    A16 = (np.arange(16, dtype=np.int32) * 7 + 13) & 0xFF

    @kernel
    def isum(A: i32[16], out: i32[1]):
        acc: i32 = 0
        for i in range(16):
            acc = acc + A[i]
        out[0] = acc

    out = np.zeros(1, np.int32)
    _to_rtl(isum).cosim(A16, out)
    assert out[0] == int(A16.astype(np.int64).sum())

    @kernel
    def imm(A: i32[4, 4], B: i32[4, 4], C: i32[4, 4]):
        for i in range(4):
            for j in range(4):
                acc: i32 = 0
                for k in range(4):
                    acc = acc + A[i, k] * B[k, j]
                C[i, j] = acc

    rng = np.random.default_rng(1)
    Ai = rng.integers(-8, 8, size=(4, 4), dtype=np.int32)
    Bi = rng.integers(-8, 8, size=(4, 4), dtype=np.int32)
    Ci = np.zeros((4, 4), np.int32)
    _to_rtl(imm).cosim(Ai, Bi, Ci, timeout=20000)
    assert np.array_equal(Ci, (Ai @ Bi).astype(np.int32))


def test_exact_share_folds_profitable_ip():
    # Two chained float multiplies (fmul latency 2) issue at disjoint cycles,
    # so the MRT lets them share one physical unit. 'exact-share' minimizes
    # modelled area under the clock, and saving an fmul against a 2:1 mux per
    # port sits in that optimum, so one instance drops and the shared datapath
    # computes what the trivially-bound one does. Integer multiply is
    # combinational, with no instance to share, so a float IP operator is used.
    @kernel
    def chain(A: f32[1], B: f32[1], C: f32[1], o: f32[1]):
        o[0] = A[0] * B[0] * C[0]

    # At a slower clock, because the trade has to be legal to be taken: the
    # operands arrive from array ports, and a port's read cone plus a select
    # cone plus the multiply's own input cone do not fit the default period.
    a, b, c = (np.array([v], np.float32) for v in (7, 6, 5))
    shared = _to_rtl(chain, freq_mhz=150.0)
    assert shared.mlir.count("hw.instance") < _to_rtl(
        chain, freq_mhz=150.0
    ).use_trivial_binding().mlir.count("hw.instance")
    o = np.zeros(1, np.float32)
    shared.cosim(a, b, c, o)
    assert np.array_equal(o, np.array([7 * 6 * 5], np.float32))


def test_bit_slice_lowers_to_arithmetic():
    # No phase below the frontend models a bit field, so `legalize-arith`
    # expands it into the integer arithmetic the operator library prices, before
    # the schedule is cut so the chaining solve sees the field access at its real
    # combinational depth. Covers every shape the frontend admits: constant and
    # dynamic offset, read and write, and the width-one slice a bare `x[k]` is.
    @kernel
    def fields(A: u32[16], B: u32[16], C: u32[16], D: u32[16]):
        for i in range(16):
            B[i] = A[i][8:16]  # constant offset, read
            w: u32 = 0
            w[0:8] = A[i][0:8]  # constant offset, write
            w[8:16] = A[i][24:32]
            C[i] = w
            v: u32 = 0
            v[i : i + 8] = A[i][i : i + 8]  # dynamic offset, both ways
            v[3] = A[i][0]  # width-one slice
            D[i] = v

    A = np.random.default_rng(11).integers(0, 2**32, 16, dtype=np.uint64)
    A = A.astype(np.uint32)
    idx = np.arange(16, dtype=np.uint32)
    want_c = (A & 0xFF) | (((A >> 24) & 0xFF) << 8)
    want_d = ((((A >> idx) & 0xFF) << idx) & ~np.uint32(1 << 3)) | ((A & 1) << 3)

    mod = _to_rtl(fields)
    # Nothing of the allo dialect survives into the datapath.
    assert "allo.bit" not in mod.dcp_module.operation.get_asm()
    # A constant offset is a bit selection, not a shifter: CIRCT folds a shift
    # by a literal back into extract / concat. A dynamic offset cannot fold,
    # `comb.extract` taking its low bit as an attribute, so it keeps a shifter.
    assert ">>" not in mod.verilog.split("B_wr0_data")[1].split(";")[0]

    B, C, D = (np.zeros(16, np.uint32) for _ in range(3))
    mod.cosim(A.copy(), B, C, D)
    assert np.array_equal(B, (A >> 8) & 0xFF)
    assert np.array_equal(C, want_c)
    assert np.array_equal(D, want_d.astype(np.uint32))


def _op_kinds(fn):
    """How many of each operation the schedule placed in the leaf region."""
    ops = _sched(fn).func(fn.__name__).regions[0].ops
    return collections.Counter(o.kind for o in ops)


def test_bit_field_write_drops_redundant_masks():
    # Splicing a field masks the hole it fills, and the splices chain, so four
    # field writes put four AND-OR pairs on one combinational path. Where the
    # bits a mask clears are ones the value provably never sets (every field of
    # a word that started at zero) the mask computes nothing and the forward bit
    # walk in `narrow-demanded-bits` removes it, leaving the concatenation the
    # write really is.
    @kernel
    def pack(A: u32[16], B: u32[16]):
        for i in range(16):
            w: u32 = 0
            w[0:8] = A[i][0:8]
            w[8:16] = A[i][8:16]
            w[16:24] = A[i][16:24]
            w[24:32] = A[i][24:32]
            B[i] = w

    @kernel
    def copy(A: u32[16], B: u32[16]):
        for i in range(16):
            B[i] = A[i]

    # A mask over a field that already holds data is load-bearing and stays an
    # AND. A low mask over a signed shift is load-bearing too (the high bits are
    # the replicated sign), but a low mask is a zero-extended truncation, so it
    # survives as the casts, which are wiring, not as an AND unit.
    @kernel
    def overwrite(A: u32[16], V: u32[16], B: u32[16]):
        for i in range(16):
            w: u32 = A[i]
            w[8:16] = V[i][0:8]
            B[i] = w

    @kernel
    def signed_mask(A: i32[16], B: i32[16]):
        for i in range(16):
            s: i32 = A[i] >> 4
            B[i] = s & 65535

    assert _op_kinds(pack)["andi"] == 0
    assert _op_kinds(overwrite)["andi"] == 1
    assert _op_kinds(signed_mask)["andi"] == 0
    # The payoff: a word rebuilt field by field costs no more than copying it.
    assert _latency(pack) == _latency(copy)

    rng = np.random.default_rng(9)
    A = rng.integers(0, 2**32, 16, dtype=np.uint64).astype(np.uint32)
    V = rng.integers(0, 2**32, 16, dtype=np.uint64).astype(np.uint32)
    Ai = rng.integers(-(2**31), 2**31, 16).astype(np.int32)

    B = np.zeros(16, np.uint32)
    _to_rtl(pack).cosim(A.copy(), B)
    assert np.array_equal(B, A)
    B = np.zeros(16, np.uint32)
    _to_rtl(overwrite).cosim(A.copy(), V.copy(), B)
    assert np.array_equal(B, (A & np.uint32(0xFFFF00FF)) | ((V & 0xFF) << 8))
    B = np.zeros(16, np.int32)
    _to_rtl(signed_mask).cosim(Ai.copy(), B)
    assert np.array_equal(B, (Ai >> 4) & 0xFFFF)


def test_disjoint_or_is_a_concatenation():
    # Two values sharing no set bit concatenate rather than combine: every result
    # bit takes one side while the other contributes a constant zero. Disjoint
    # ORs chained cost nothing and settle at one sub-cycle position, where
    # overlapping ones spread a gate delay apart. The forward bit walk in
    # `narrow-demanded-bits` is what tells them apart.
    @kernel
    def disjoint(A: u32[64], B: u32[64], C: u32[64]):
        for i in range(64):
            lo: u32 = A[i][0:8]
            hi: u32 = B[i][0:8]
            a: u32 = lo | (hi << 8)
            b: u32 = a | (lo << 16)
            C[i] = b | (hi << 24)

    # The same fields placed to overlap: each OR now merges bits and costs a gate.
    @kernel
    def overlapping(A: u32[64], B: u32[64], C: u32[64]):
        for i in range(64):
            lo: u32 = A[i][0:8]
            hi: u32 = B[i][0:8]
            a: u32 = lo | (hi << 4)
            b: u32 = a | (lo << 2)
            C[i] = b | (hi << 6)

    def _or_offsets(fn):
        ops = _sched(fn).func(fn.__name__).regions[0].ops
        return sorted({round(o.z, 3) for o in ops if o.kind == "ori"})

    assert len(_or_offsets(disjoint)) == 1
    # Overlapping ORs each cost a gate, so they spread across more than one
    # sub-cycle position, a gate delay apart (the scheduler may pack ORs from
    # different pipeline stages onto a shared offset, so the count is not fixed).
    spaced = _or_offsets(overlapping)
    assert len(spaced) > 1
    assert all(
        b - a == pytest.approx(comb_step_ns("or"), abs=1e-3)
        for a, b in zip(spaced, spaced[1:])
    )

    rng = np.random.default_rng(2)
    A = rng.integers(0, 2**32, 64, dtype=np.uint64).astype(np.uint32)
    B = rng.integers(0, 2**32, 64, dtype=np.uint64).astype(np.uint32)
    lo, hi = A & 0xFF, B & 0xFF
    C = np.zeros(64, np.uint32)
    _to_rtl(disjoint).cosim(A.copy(), B.copy(), C)
    assert np.array_equal(C, lo | (hi << 8) | (lo << 16) | (hi << 24))
    C = np.zeros(64, np.uint32)
    _to_rtl(overlapping).cosim(A.copy(), B.copy(), C)
    assert np.array_equal(C, lo | (hi << 4) | (lo << 2) | (hi << 6))


def test_literal_shift_is_wiring():
    # A shift by a literal renames bits: `comb` folds it into an extract /
    # concat, so it costs no logic. The device's shift row prices a barrel
    # shifter, which is what a runtime amount pays for. The two kernels have the
    # same operation count and memory traffic and differ only in the shift
    # amount, so any gap is the shifter's delay alone.
    @kernel
    def literal(A: u32[16], C: u32[16], B: u32[16]):
        for i in range(16):
            a: u32 = (A[i] << 3) ^ C[i]
            b: u32 = (a << 3) ^ C[i]
            c: u32 = (b << 3) ^ C[i]
            B[i] = (c << 3) ^ C[i]

    @kernel
    def runtime(A: u32[16], C: u32[16], B: u32[16]):
        for i in range(16):
            a: u32 = (A[i] << C[i]) ^ C[i]
            b: u32 = (a << C[i]) ^ C[i]
            c: u32 = (b << C[i]) ^ C[i]
            B[i] = (c << C[i]) ^ C[i]

    assert len(_sched(literal).func("literal").regions[0].ops) == len(
        _sched(runtime).func("runtime").regions[0].ops
    )
    # The premise, read off the device rather than assumed: a runtime shift
    # costs a barrel shifter's step and forces cuts a literal one does not.
    assert comb_step_ns("shl") > 0
    assert _latency(literal) < _latency(runtime)

    rng = np.random.default_rng(4)
    A = rng.integers(0, 2**32, 16, dtype=np.uint64).astype(np.uint32)
    C = (rng.integers(0, 2**32, 16, dtype=np.uint64).astype(np.uint32),)[0]
    want = A
    for _ in range(4):
        want = ((want << np.uint32(3)) ^ C).astype(np.uint32)
    B = np.zeros(16, np.uint32)
    _to_rtl(literal).cosim(A.copy(), C.copy(), B)
    assert np.array_equal(B, want)


def _shared_units(mod):
    """How many operations each shared unit carries, across every region."""
    return sorted(
        u.bound_ops for r in mod.microarch.top.regions for u in r.shared_units
    )


def _mux_fanins(mod):
    """Every multiplexer's fan-in, one entry per mux."""
    return sorted(
        f
        for r in mod.microarch.top.regions
        for m in r.muxes
        for f in [m.fanin] * m.count
    )


def test_shared_reduction_reinjects_its_identity():
    # A loop-carried accumulator may share its adder with ordinary ops: the
    # identity it re-injects on the first iteration rides an arm of that unit's
    # input mux (`Mux::Phase`), since a time-shared port has no cycle of its own
    # to time a 2:1 mux against. The identity is non-zero, so an arm that never
    # fires, or fires on the wrong iteration, shows up in the sum.
    @kernel
    def fred(A: f32[64], B: f32[64], out: f32[1]):
        s: f32 = 1.5
        for i in range(0, 64, 4):
            s = s + A[i]
            B[i] = A[i] + A[i + 1]
            B[i + 1] = A[i + 1] + A[i + 2]
            B[i + 2] = A[i + 2] + A[i + 3]
            B[i + 3] = A[i + 3] + A[i]
        out[0] = s

    A = np.random.default_rng(5).standard_normal(64).astype(np.float32)
    want = np.float32(1.5)
    for i in range(0, 64, 4):
        want = np.float32(want + A[i])

    shared = _to_rtl(fred)
    assert _shared_units(shared), "the reduction's adder was not shared at all"
    # The recurrence port carries one arm per bound op plus the identity's.
    assert max(_mux_fanins(shared)) > max(u for u in _shared_units(shared))

    for mod in (_to_rtl(fred).use_trivial_binding(), shared):
        B, out = np.zeros(64, np.float32), np.zeros(1, np.float32)
        mod.cosim(A.copy(), B, out)
        assert abs(out[0] - want) < 1e-3
        assert np.allclose(B[0::4], A[0::4] + A[1::4], rtol=1e-5)


def test_planned_allocation_is_never_looser_than_exact_share():
    # The exact scheduler decides how many copies of each operator a region
    # builds and 'planned' builds exactly that. Its search starts from, and
    # falls back on, the tightest count its own schedule admits, so it uses no
    # more instances than binding-time exact sharing does.
    @kernel
    def chain(A: f32[1], B: f32[1], C: f32[1], D: f32[1], o: f32[1]):
        o[0] = A[0] * B[0] * C[0] * D[0]

    args = [np.array([v], np.float32) for v in (7, 6, 5, 2)]
    ref = np.array([7 * 6 * 5 * 2], np.float32)
    shared = _to_rtl(chain)
    planned = _to_rtl(chain).set_scheduler_opt(scheduler="exact")
    assert planned.mlir.count("hw.instance") <= shared.mlir.count("hw.instance")
    o = np.zeros(1, np.float32)
    planned.cosim(*args, o)
    assert np.array_equal(o, ref)


# If-conversion over both datapaths: an int compare lowers to native
# comb.icmp, a float compare to an fcmp IP instance, both feeding a comb.mux.
# Shifts lower to native comb.shl / comb.shr.
def test_compare_select_and_shift():
    @kernel
    def relu(A: i32[16], out: i32[16]):
        for i in range(16):
            if A[i] > 0:
                out[i] = A[i]
            else:
                out[i] = 0

    A = np.random.default_rng(0).integers(-50, 50, size=16, dtype=np.int32)
    mod = _to_rtl(relu)
    assert "comb.icmp" in mod.mlir and "comb.mux" in mod.mlir
    out = np.zeros(16, np.int32)
    mod.cosim(A, out)
    assert np.array_equal(out, np.maximum(A, 0))

    # A second predicate (`<=` -> sle) exercises the arith->comb predicate map.
    @kernel
    def sel(A: i32[16], B: i32[16], out: i32[16]):
        for i in range(16):
            if A[i] <= B[i]:
                out[i] = A[i]
            else:
                out[i] = B[i]

    rng = np.random.default_rng(1)
    A = rng.integers(-40, 40, size=16, dtype=np.int32)
    B = rng.integers(-40, 40, size=16, dtype=np.int32)
    out = np.zeros(16, np.int32)
    _to_rtl(sel).cosim(A, B, out)
    assert np.array_equal(out, np.minimum(A, B))

    A16 = (np.arange(16, dtype=np.int32) * 7 + 13) & 0xFF

    @kernel
    def sh(A: i32[16], out: i32[16]):
        for i in range(16):
            out[i] = (A[i] << 2) >> 1

    mod = _to_rtl(sh)
    assert "comb.shl" in mod.mlir and "comb.shr" in mod.mlir
    out = np.zeros(16, np.int32)
    mod.cosim(A16, out)
    assert np.array_equal(out, (A16 << 2) >> 1)

    @kernel
    def frelu(A: f32[16], out: f32[16]):
        for i in range(16):
            if A[i] > 0.0:
                out[i] = A[i]
            else:
                out[i] = 0.0

    Af = _signed_f32(0)
    mod = _to_rtl(frelu)
    fcmp = next(o for o in default_device.operators if o.optype is OperatorType.CMP)
    # The predicate rides the module name, so the extern is the symbol plus it.
    assert f"hw.module.extern @{fcmp.symbol}_ogt" in mod.mlir
    assert "comb.mux" in mod.mlir
    outf = np.zeros(16, np.float32)
    mod.cosim(Af, outf)
    assert np.allclose(outf, np.maximum(Af, 0.0), rtol=1e-5)

    # A second float predicate (`<=` -> ole) + a select over both operands.
    @kernel
    def fmax(A: f32[16], B: f32[16], out: f32[16]):
        for i in range(16):
            if A[i] <= B[i]:
                out[i] = B[i]
            else:
                out[i] = A[i]

    Af, Bf = _signed_f32(1), _signed_f32(2)
    outf = np.zeros(16, np.float32)
    _to_rtl(fmax).cosim(Af, Bf, outf)
    assert np.allclose(outf, np.maximum(Af, Bf), rtol=1e-5)


# --- reduction restructuring ---------------------------------------------------


# Rotating a float reduction across N accumulators turns its distance-1
# recurrence (II == add latency) into a distance-N one: II == ceil(L/N).
def test_rotate_reduction_scales_ii():
    def ii(n):
        @kernel
        def red(x: f32[256]) -> f32:
            acc: f32 = 0.0
            for i in range(256, name="i"):
                acc += x[i]
            return acc

        res = _to_rtl(red).set_scheduler_opt(accumulators=n).schedule()
        return res.cyclic()[0].interval

    assert ii(0) == FADD  # unrotated
    assert ii(FADD) == 1  # N == latency -> II 1
    assert ii(2) == math.ceil(FADD / 2)

    # bf16 inputs with an f32 accumulator (the common ML pattern): the cast sits
    # on the leaf, not around the operator, so rotation works unchanged.
    def mixed_ii(n):
        @kernel
        def red(x: bf16[64]) -> f32:
            acc: f32 = 0.0
            for i in range(64, name="i"):
                acc += x[i]
            return acc

        res = _to_rtl(red).set_scheduler_opt(accumulators=n).schedule()
        return res.cyclic()[0].interval

    assert mixed_ii(0) == FADD
    assert mixed_ii(FADD) == 1


# `accumulators=-1` is auto: the pass reads the reduction operator's latency L
# at the target clock and rotates on exactly L accumulators, the least count
# that brings II to 1. A fixed count cannot track L, which deepens with the
# clock; auto does. The unrotated II (accumulators=0) is L itself, so the auto
# result is checked against it rather than a hard-coded latency.
def test_rotate_reduction_auto_tracks_operator_latency():
    @kernel
    def red(x: f32[256]) -> f32:
        acc: f32 = 0.0
        for i in range(256, name="i"):
            acc += x[i]
        return acc

    def ii(freq, n):
        rtl = _to_rtl(red, freq_mhz=freq) if freq else _to_rtl(red)
        return rtl.set_scheduler_opt(accumulators=n).schedule().cyclic()[0].interval

    # A faster clock deepens the adder; auto reaches II=1 at every clock, and by
    # rotating on exactly L == the unrotated II, never over-provisioning.
    for freq in (None, 450):
        latency = ii(freq, 0)
        assert ii(freq, -1) == 1
        assert ii(freq, latency) == 1  # forcing N == L also reaches 1
        assert ii(freq, -1) == ii(freq, latency)  # auto picks exactly L
        if latency > 2:
            # A fixed count of 2 under-provisions the deepened adder.
            assert ii(freq, 2) == math.ceil(latency / 2)


# One loop carrying several independent reductions (a complex accumulate's real
# and imaginary parts) rotates each on its own iter_arg, so every one reaches
# II=1. Rotation reassociates the sums, so the float result holds to tolerance.
def test_rotate_multiple_reductions_in_one_loop():
    @kernel
    def red(x: f32[128], y: f32[128], out: f32[2]):
        a: f32 = 0.0
        b: f32 = 0.0
        for i in range(128, name="i"):
            a += x[i]
            b += y[i] * 2.0
        out[0] = a
        out[1] = b

    def ii(n):
        res = _to_rtl(red).set_scheduler_opt(accumulators=n).schedule()
        return res.cyclic()[0].interval

    assert ii(0) == FADD  # both carried at the add latency
    assert ii(FADD) == 1  # each reduction rotated independently to II=1

    rng = np.random.default_rng(0)
    x = rng.uniform(-1.0, 1.0, 128).astype(np.float32)
    y = rng.uniform(-1.0, 1.0, 128).astype(np.float32)
    out = np.zeros(2, np.float32)
    _to_rtl(red).set_scheduler_opt(accumulators=FADD).cosim(x, y, out)
    assert np.allclose(out[0], x.sum(), rtol=1e-3, atol=1e-3)
    assert np.allclose(out[1], (y * 2.0).sum(), rtol=1e-3, atol=1e-3)


# Only a LEAF reduction rotates: the emitter builds the rotated shift register on
# a childless modulo loop. A container reduction (`total += inner_sum`) is left
# unrotated -- rotating it would double-count, so cosim pins the nest's sum.
def test_rotate_leaves_container_reduction_alone():
    @kernel
    def nred(A: f32[8, 8], out: f32[1]):
        total: f32 = 0.0
        for i in range(8, name="i"):
            partial: f32 = 0.0
            for j in range(8, name="j"):
                partial += A[i, j]
            total += partial
        out[0] = total

    iis = _iis(_to_rtl(nred).set_scheduler_opt(accumulators=FADD).schedule().cyclic())
    assert 1 in iis  # the inner leaf reduction rotated to II=1

    rng = np.random.default_rng(1)
    A = rng.uniform(-1.0, 1.0, (8, 8)).astype(np.float32)
    out = np.zeros(1, np.float32)
    _to_rtl(nred).set_scheduler_opt(accumulators=FADD).cosim(A, out)
    assert np.allclose(out[0], A.sum(), rtol=1e-3, atol=1e-3)


# Integer reductions rebalance unconditionally (integer arithmetic is exactly
# associative mod 2^w), cutting an unrolled chain's recurrence to one operator.
def test_reassociate_int_reduction_recurrence():
    # Unrolling threads the carried accumulator through four widened multiplies;
    # folding it in last makes the recurrence one (widened, combinational)
    # multiply rather than a chain of four. The rebalance is what this test
    # pins. The resulting II is NOT evidence for it: with a factor-4 unroll the
    # II is the resource bound (four loads over the port budget), so it would
    # read the same whether the chain was rebalanced or not.
    @kernel
    def red(x: i32[32]) -> i32:
        acc: i32 = 1
        for i in range(32, name="i"):
            acc *= x[i]
        return acc

    s = red.schedule()
    s.unroll("i", factor=4)
    region = s.export("rtl").schedule().cyclic()[0]
    # The five terms are the four unrolled multiplies plus the accumulator,
    # folded in last, so the carried path is one multiply rather than the four a
    # chain would leave on it. The recurrence bounds the II, so a tree fits in a
    # span a chain could not. Measured against the device rather than pinned.
    assert region.interval * PERIOD_NS < REG_NS + 4 * comb_step_ns("mul")


# `tree-height-reduction` rebalances a subtraction-rooted datapath tap: the
# additive family treats `sub` as a sign flip, so `a - b + c - d` (linear
# `((a-b)+c)-d`, depth 3) collapses to `(a+c) - (b+d)` (depth 2). The op mix is
# the witness: two subtracts and one add become one subtract and two adds.
def test_tree_height_reduction_balances_mixed_sign():
    from allo.backend.base import run_pipeline
    from allo.backend.rtl.schedule import RTL_PREPARE_PIPELINE
    from allo.compiler.mlir_codegen import compile as compile_kernel

    @kernel
    def tap(a: f32[8], b: f32[8], c: f32[8], d: f32[8], out: f32[8]):
        for i in range(8, name="i"):
            out[i] = a[i] - b[i] + c[i] - d[i]

    module = compile_kernel(tap)
    run_pipeline(module, RTL_PREPARE_PIPELINE)
    before = str(module)
    assert before.count("arith.subf") == 2 and before.count("arith.addf") == 1

    run_pipeline(
        module, "builtin.module(func.func(tree-height-reduction{enable-fp=true}))"
    )
    after = str(module)
    assert after.count("arith.addf") == 2 and after.count("arith.subf") == 1


# The rebalanced tree is numerically correct end to end (within the fp-relaxed
# tolerance reassociation runs under).
def test_tree_height_reduction_cosim_mixed_sign():
    @kernel
    def tap(a: f32[16], b: f32[16], c: f32[16], d: f32[16], out: f32[16]):
        for i in range(16, name="i"):
            out[i] = a[i] - b[i] + c[i] - d[i]

    rng = np.random.default_rng(0)
    arrs = [rng.standard_normal(16).astype(np.float32) for _ in range(4)]
    out = np.zeros(16, np.float32)
    _to_rtl(tap).cosim(*arrs, out)
    np.testing.assert_allclose(
        out, arrs[0] - arrs[1] + arrs[2] - arrs[3], rtol=1e-5, atol=1e-5
    )


# A mixed-sign carried reduction `acc = acc + a - b`: neither predecessor pass
# handled it (reassociate never matched `sub`, and the split THR skipped carried
# trees). The merged pass folds the carried accumulator at the root so the
# recurrence spans one operator, and the result stays numerically correct.
def test_tree_height_reduction_carried_mixed_sign():
    @kernel
    def r(a: f32[64], b: f32[64]) -> f32:
        acc: f32 = 0.0
        for i in range(64, name="i"):
            acc = acc + a[i] - b[i]
        return acc

    rng = np.random.default_rng(1)
    a = rng.standard_normal(64).astype(np.float32)
    b = rng.standard_normal(64).astype(np.float32)
    res = _to_rtl(r).cosim(a, b)
    assert abs(float(res.result) - float((a - b).sum())) < 1e-2


# Bit growth types an expression at its natural width and applies the declared
# type as a trailing truncation, so every operator in between is built at a
# width nothing reads. `narrow-demanded-bits` sinks that truncation onto the
# leaves, where it collapses into the extends bit growth put there.
def test_narrow_demanded_bits_widths():
    from allo.backend.base import run_pipeline
    from allo.backend.rtl.schedule import RTL_PREPARE_PIPELINE
    from allo.compiler.mlir_codegen import compile as compile_kernel

    i48 = APInt(48, signed=True)

    @kernel
    def mac(b: i32, c: i32, d: i32) -> i48:
        a: i48 = b * c + d
        return a

    module = compile_kernel(mac)
    run_pipeline(module, RTL_PREPARE_PIPELINE)
    # The natural widths: a 64-bit product feeding a 65-bit add, then truncated
    # to the 48 bits the declaration asked for.
    before = str(module)
    assert "i64" in before and "i65" in before and "arith.trunci" in before

    run_pipeline(module, "builtin.module(func.func(narrow-demanded-bits))")
    after = str(module)
    assert "arith.muli" in after and "arith.addi" in after
    assert "i64" not in after and "i65" not in after
    # Nothing is discarded any more, so the truncation is gone rather than moved.
    assert "arith.trunci" not in after


# The narrowing is bit-exact: an i48 accumulator wraps identically whether its
# adder is 48 or 65 bits wide. The inputs are sized so the exact sum overflows.
def test_narrow_demanded_bits_wraps_exactly():
    i48 = APInt(48, signed=True)

    @kernel
    def dot(x: i32[8], y: i32[8]) -> i48:
        acc: i48 = 0
        for k in range(8, name="k"):
            acc = acc + x[k] * y[k]
        return acc

    rng = np.random.default_rng(0)
    x = rng.integers(-(2**31), 2**31, size=8, dtype=np.int64).astype(np.int32)
    y = rng.integers(-(2**31), 2**31, size=8, dtype=np.int64).astype(np.int32)
    exact = sum(int(a) * int(b) for a, b in zip(x, y))
    assert abs(exact) > 2**47, "inputs must overflow i48 for the wrap to matter"
    wrapped = ((exact + 2**47) % 2**48) - 2**47

    # The narrowed multiply is 48 bits wide, which measures past the default
    # clock's period on this part.
    r = _to_rtl(dot, freq_mhz=200).cosim(x, y)
    assert int(r.result) == wrapped


def _run_narrow(ir: str) -> str:
    """Parse hand-written IR and run `narrow-demanded-bits` over it alone."""
    from allo._mlir.ir import Module
    from allo.backend.base import run_pipeline
    from allo.backend.rtl.device import _scratch_context

    with _scratch_context():
        module = Module.parse(ir)
        run_pipeline(module, "builtin.module(func.func(narrow-demanded-bits))")
        return str(module)


# The backward truncation sink crosses bitwise operators and a constant-amount
# left shift, not just the ring ops: a whole `(a & M) | (b ^ c) + (d << 2)` cone
# rebuilds at the i8 its consumer reads, and the truncation is gone.
def test_narrow_trunc_crosses_bitwise_and_shift():
    after = _run_narrow(
        """
        func.func @cone(%a: i32, %b: i32, %c: i32, %d: i32) -> i8 {
          %m = arith.constant 255 : i32
          %k = arith.constant 2 : i32
          %0 = arith.andi %a, %m : i32
          %1 = arith.xori %b, %c : i32
          %2 = arith.ori %0, %1 : i32
          %3 = arith.shli %d, %k : i32
          %4 = arith.addi %2, %3 : i32
          %5 = arith.trunci %4 : i32 to i8
          return %5 : i8
        }
        """
    )
    # The whole cone was rebuilt at i8: every arithmetic op landed narrow, and
    # the truncation reached the leaves, where it only truncates the block args.
    narrowed = [
        line
        for line in after.splitlines()
        if any(
            op in line for op in ("arith.ori", "arith.xori", "arith.shli", "arith.addi")
        )
    ]
    assert narrowed
    assert all(": i8" in line for line in narrowed), after
    for line in after.splitlines():
        if "arith.trunci" in line:
            assert ": i32 to i8" in line and "%arg" in line, line


# A right shift sinks through a truncation only when the operand's high bits are
# known: masked to the low byte, `shrui` narrows to i8; with unknown high bits it
# must stay wide and the truncation remains.
def test_narrow_trunc_shift_guard():
    guarded = _run_narrow(
        """
        func.func @g(%a: i32) -> i8 {
          %m = arith.constant 255 : i32
          %one = arith.constant 1 : i32
          %0 = arith.andi %a, %m : i32
          %1 = arith.shrui %0, %one : i32
          %2 = arith.trunci %1 : i32 to i8
          return %2 : i8
        }
        """
    )
    assert re.search(r"arith\.shrui[^\n]*: i8", guarded)
    assert not re.search(r"arith\.shrui[^\n]*: i32", guarded)

    unguarded = _run_narrow(
        """
        func.func @u(%a: i32) -> i8 {
          %one = arith.constant 1 : i32
          %0 = arith.shrui %a, %one : i32
          %1 = arith.trunci %0 : i32 to i8
          return %1 : i8
        }
        """
    )
    assert re.search(r"arith\.shrui[^\n]*: i32", unguarded)
    assert "arith.trunci" in unguarded


# The narrowed bitwise + shift cone is bit-exact: computing
# `(a & 0xFF) | ((b ^ c) << 1)` at i8 wraps identically to computing it wide and
# truncating.
def test_narrow_trunc_bitwise_cosim():
    @kernel
    def cone(A: i32[16], B: i32[16], C: i32[16], out: i8[16]):
        for i in range(16):
            out[i] = (A[i] & 255) | ((B[i] ^ C[i]) << 1)

    rng = np.random.default_rng(5)
    A = rng.integers(-(2**31), 2**31, 16).astype(np.int32)
    B = rng.integers(-(2**31), 2**31, 16).astype(np.int32)
    C = rng.integers(-(2**31), 2**31, 16).astype(np.int32)
    out = np.zeros(16, np.int8)
    _to_rtl(cone).cosim(A.copy(), B.copy(), C.copy(), out)
    wide = (A.astype(np.int64) & 255) | ((B.astype(np.int64) ^ C.astype(np.int64)) << 1)
    ref = (wide & 0xFF).astype(np.uint8).astype(np.int8)
    assert np.array_equal(out, ref)


def _run_f2i(ir: str) -> str:
    """Parse hand-written IR and run `float-to-int` over it alone."""
    from allo._mlir.ir import Module
    from allo.backend.base import run_pipeline
    from allo.backend.rtl.device import _scratch_context

    with _scratch_context():
        module = Module.parse(ir)
        run_pipeline(module, "builtin.module(func.func(float-to-int))")
        return str(module)


# An integer-valued float cone (int cast in, arithmetic, cast/compare out) whose
# range fits the mantissa is rewritten to integer ops: the casts become resizes,
# the arithmetic becomes `addi`/`subi`, and no float op remains.
def test_float_to_int_demotes_a_small_range_cone():
    after = _run_f2i(
        """
        func.func @cone(%a: i8, %b: i8) -> i16 {
          %fa = arith.sitofp %a : i8 to f32
          %fb = arith.sitofp %b : i8 to f32
          %s = arith.addf %fa, %fb : f32
          %n = arith.negf %s : f32
          %r = arith.fptosi %n : f32 to i16
          return %r : i16
        }
        """
    )
    for gone in ("arith.sitofp", "arith.addf", "arith.negf", "arith.fptosi"):
        assert gone not in after, after
    assert "arith.addi" in after and "arith.subi" in after, after
    assert "arith.extsi" in after, after


# The demotion is exact only while every value stays inside the float type's
# exactly representable integer band, so the guard is the mantissa width: an
# `i16 * i16` product overflows f32's 24-bit band and stays float, but fits
# f64's 53-bit band and demotes. The guard reads the cone's own float type.
def test_float_to_int_guard_is_mantissa_bound():
    kept = _run_f2i(
        """
        func.func @wide(%a: i16, %b: i16) -> i32 {
          %fa = arith.sitofp %a : i16 to f32
          %fb = arith.sitofp %b : i16 to f32
          %m = arith.mulf %fa, %fb : f32
          %r = arith.fptosi %m : f32 to i32
          return %r : i32
        }
        """
    )
    assert "arith.mulf" in kept and "arith.muli" not in kept, kept

    demoted = _run_f2i(
        """
        func.func @wide(%a: i16, %b: i16) -> i32 {
          %fa = arith.sitofp %a : i16 to f64
          %fb = arith.sitofp %b : i16 to f64
          %m = arith.mulf %fa, %fb : f64
          %r = arith.fptosi %m : f64 to i32
          return %r : i32
        }
        """
    )
    assert "arith.muli" in demoted and "arith.mulf" not in demoted, demoted


# A float compare of integer-valued floats becomes a signed integer compare; the
# ordered and unordered forms of a predicate both map to it (the operands are
# never NaN).
def test_float_to_int_demotes_a_compare():
    after = _run_f2i(
        """
        func.func @cmp(%a: i8, %b: i8) -> i1 {
          %fa = arith.sitofp %a : i8 to f32
          %fb = arith.sitofp %b : i8 to f32
          %c = arith.cmpf ogt, %fa, %fb : f32
          return %c : i1
        }
        """
    )
    assert "arith.cmpi sgt" in after, after
    assert "arith.cmpf" not in after and "arith.sitofp" not in after, after


# The demotion fires only where it is exact: a divide has no exact integer image,
# a fractional constant is not an integer, and a loop-carried accumulator leaves
# the cone open. All three keep their float ops.
def test_float_to_int_leaves_inexact_cones_float():
    div = _run_f2i(
        """
        func.func @d(%a: i8, %b: i8) -> i16 {
          %fa = arith.sitofp %a : i8 to f32
          %fb = arith.sitofp %b : i8 to f32
          %q = arith.divf %fa, %fb : f32
          %r = arith.fptosi %q : f32 to i16
          return %r : i16
        }
        """
    )
    assert "arith.divf" in div and "arith.divi" not in div, div

    frac = _run_f2i(
        """
        func.func @f(%a: i8) -> i16 {
          %fa = arith.sitofp %a : i8 to f32
          %k = arith.constant 1.5 : f32
          %m = arith.mulf %fa, %k : f32
          %r = arith.fptosi %m : f32 to i16
          return %r : i16
        }
        """
    )
    assert "arith.mulf" in frac, frac

    carried = _run_f2i(
        """
        func.func @c(%n: index, %a: i8) -> i32 {
          %c0 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          %fa = arith.sitofp %a : i8 to f32
          %init = arith.constant 0.0 : f32
          %sum = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> f32 {
            %next = arith.addf %acc, %fa : f32
            scf.yield %next : f32
          }
          %r = arith.fptosi %sum : f32 to i32
          return %r : i32
        }
        """
    )
    assert "arith.addf" in carried and "arith.addi" not in carried, carried


# End to end: a kernel that casts int to float, adds, and stores back to an int
# array demotes to integer arithmetic (no float op survives) and cosims
# bit-exact against the integer reference.
def test_float_to_int_cosim():
    N = 8

    @kernel
    def demote(A: i8[N], B: i8[N], out: i16[N]):
        for i in range(N):
            x: f32 = A[i]
            y: f32 = B[i]
            out[i] = x + y

    kinds = _op_kinds(demote)
    assert kinds["addi"] >= 1
    assert kinds["addf"] == 0 and kinds["sitofp"] == 0 and kinds["fptosi"] == 0

    rng = np.random.default_rng(0)
    A = rng.integers(-128, 128, size=N, dtype=np.int8)
    B = rng.integers(-128, 128, size=N, dtype=np.int8)
    out = np.zeros(N, np.int16)
    _to_rtl(demote).cosim(A, B, out)
    assert np.array_equal(out, A.astype(np.int16) + B.astype(np.int16))


# Float min and max of integer-valued (never-NaN) operands demote to signed
# integer min/max; all four NaN variants collapse to one op. It fires end to end
# even though min/max lower to compare+select, because the demotion runs first.
def test_float_to_int_demotes_min_max():
    after = _run_f2i(
        """
        func.func @mm(%a: i8, %b: i8) -> i16 {
          %fa = arith.sitofp %a : i8 to f32
          %fb = arith.sitofp %b : i8 to f32
          %x = arith.maxnumf %fa, %fb : f32
          %y = arith.minimumf %x, %fa : f32
          %r = arith.fptosi %y : f32 to i16
          return %r : i16
        }
        """
    )
    assert "arith.maxsi" in after and "arith.minsi" in after, after
    for gone in ("arith.maxnumf", "arith.minimumf", "arith.sitofp", "arith.fptosi"):
        assert gone not in after, after

    N = 8

    @kernel
    def relu(A: i8[N], out: i16[N]):
        for i in range(N):
            x: f32 = A[i]
            out[i] = max(x, 0.0)

    assert _op_kinds(relu)["maxsi"] >= 1 and _op_kinds(relu)["maxnumf"] == 0
    A = np.random.default_rng(4).integers(-128, 128, size=N, dtype=np.int8)
    out = np.zeros(N, np.int16)
    _to_rtl(relu).cosim(A, out)
    assert np.array_equal(out, np.maximum(A.astype(np.int16), 0))


# A cone that changes float width demotes as long as each op stays exact in its
# OWN type: extf is always exact and vanishes; truncf demotes only when the value
# fits the narrower type, so an f64 product that overflows f32 keeps its floats.
def test_float_to_int_crosses_float_widths():
    widened = _run_f2i(
        """
        func.func @w(%a: i8, %b: i8) -> i16 {
          %fa = arith.sitofp %a : i8 to f32
          %fb = arith.sitofp %b : i8 to f32
          %s = arith.addf %fa, %fb : f32
          %e = arith.extf %s : f32 to f64
          %r = arith.fptosi %e : f64 to i16
          return %r : i16
        }
        """
    )
    assert "arith.addi" in widened, widened
    for gone in ("arith.extf", "arith.addf", "arith.sitofp"):
        assert gone not in widened, widened

    exact = _run_f2i(
        """
        func.func @te(%a: i8, %b: i8) -> i16 {
          %fa = arith.sitofp %a : i8 to f64
          %fb = arith.sitofp %b : i8 to f64
          %s = arith.addf %fa, %fb : f64
          %t = arith.truncf %s : f64 to f32
          %r = arith.fptosi %t : f32 to i16
          return %r : i16
        }
        """
    )
    assert "arith.addi" in exact and "arith.truncf" not in exact, exact

    rounds = _run_f2i(
        """
        func.func @tr(%a: i16, %b: i16) -> i32 {
          %fa = arith.sitofp %a : i16 to f64
          %fb = arith.sitofp %b : i16 to f64
          %m = arith.mulf %fa, %fb : f64
          %t = arith.truncf %m : f64 to f32
          %r = arith.fptosi %t : f32 to i32
          return %r : i32
        }
        """
    )
    assert "arith.mulf" in rounds and "arith.truncf" in rounds, rounds
    assert "arith.muli" not in rounds, rounds


# A loop-carried integer scalar is built at its recurrence envelope, not at its
# declared carrier: a counter stepping by 2 over 50 trips stays within [0, 100]
# and an argmax-style position within the IV's range. A data-stepped
# accumulator has no envelope and keeps the carrier.
def test_a_carried_scalar_narrows_to_its_recurrence_envelope():
    N = 50

    @kernel
    def track(A: i32[N], out: i32[3]):
        t: i32 = 0
        idx: i32 = 0
        s: i32 = 0
        for i in range(N):
            if A[i] > 0:
                t = t + 2
                idx = i
            s = s + A[i]
        out[0] = t
        out[1] = idx
        out[2] = s

    rtl = _to_rtl(track)
    m = rtl.mlir
    widths = {
        name: width
        for name, width in re.findall(r"%(r\d+_sv\d+) = seq\.compreg[^\n]*: (i\d+)", m)
    }
    svw = sorted(widths.values())
    # t rides [0, 100] (i8), idx [0, 49] (i7); the data-stepped s keeps i32.
    assert svw == ["i32", "i7", "i8"], widths

    rng = np.random.default_rng(3)
    A = rng.integers(-40, 40, size=N).astype(np.int32)
    out = np.zeros(3, np.int32)
    rtl.cosim(A, out)
    taken = A > 0
    assert out[0] == 2 * int(taken.sum())
    assert out[1] == int(np.max(np.where(taken)[0]))
    assert out[2] == int(A.sum())


def test_int_product_reduction_cosim():
    # An integer *multiply* reduction (distinct from the add reductions): the
    # multiply-latency recurrence pipelines and the frozen product returns as a
    # scalar. Small values keep the product within i32 (exact, no wrap ambiguity).
    @kernel
    def prod(x: i32[16]) -> i32:
        acc: i32 = 1
        for i in range(16, name="i"):
            acc *= x[i]
        return acc

    x = np.ones(16, dtype=np.int32)
    x[:6] = np.array([2, 3, 1, 2, 5, 1], np.int32)  # product 360, fits i32
    r = _to_rtl(prod).cosim(x)
    assert int(r.result) == int(np.prod(x.astype(np.int64)))


# --- the measured model ---------------------------------------------------------


def _fabrics():
    """Every fabric module paired with a part built on it."""
    from allo.backend.rtl.devices import kv260, pynqz2, series7, u55c, ultrascalex
    from allo.backend.rtl.devices import versal, vck190

    return (
        (ultrascalex, u55c),
        (ultrascalex, kv260),
        (versal, vck190),
        (series7, pynqz2),
    )


def _points(cost):
    """The ``[point, value]`` pairs of a `table` or an `interp`."""
    return dict(zip(cost.coeffs[0::2], cost.coeffs[1::2]))


def _rounded(value: float) -> int:
    """As the cost evaluator rounds a total: half away from zero."""
    return math.floor(value + 0.5)


def test_a_tiled_factor_may_sit_anywhere_in_a_term():
    dev = default_device.copy()
    ff = dev.resources["ff"]
    dev.set_chain_uses({ff: (Tiled(32), Linear(1.0))})
    assert dev.price(dev.chain_uses, (256, 8)) == {"ff": 8 * 8}
    dev.set_chain_uses({ff: (Linear(1.0), Tiled(32))})
    assert dev.price(dev.chain_uses, (8, 256)) == {"ff": 8 * 8}
    # Anything other than a full set of factors, or one `tiled` alone, is an
    # arity error.
    for wrong in ((Linear(1.0),), (Tiled(32), Linear(1.0), Const(1.0))):
        with pytest.raises(ValueError, match="2 factor"):
            dev.set_chain_uses({ff: wrong})
    dev.set_chain_uses({ff: Tiled(1024)})  # the whole-tuple spelling still holds
    assert dev.price(dev.chain_uses, (256, 8)) == {"ff": 2}


# Delay is continuous in the operand width, so every characterized row is read
# as a line through its measurements.
def test_every_comb_delay_row_interpolates_its_measurements():
    for mod, dev in _fabrics():
        grade = next(g for g in mod.TIMING if g.name == dev.grade)
        for kind, cost in mod.TIMING[grade].comb.items():
            assert cost.form == "interp", (mod.NAME, kind)
            for p, v in _points(cost).items():
                assert dev.comb_delay(kind, int(p)) == pytest.approx(v, abs=1e-9)
    # Between two measurements the curve is the line joining them, and 48 bits
    # is the midpoint of 32 and 64.
    dev = default_device
    for kind in ("add", "mul", "div"):
        lo, hi = dev.comb_delay(kind, 32), dev.comb_delay(kind, 64)
        assert dev.comb_delay(kind, 48) == pytest.approx((lo + hi) / 2, abs=1e-9)
        assert dev.comb_delay(kind, 48) > lo


# An area row is a staircase only where the quantity really steps: a DSP count
# is a whole number of slices, while the fabric logic around them grows with the
# width.
def test_comb_area_rows_reproduce_their_measurements():
    for _, dev in _fabrics():
        for kind in ("shl", "shr", "mul", "div", "rem"):
            for name, factors in dev.comb_uses[kind]:
                if factors[0].form not in ("table", "interp"):
                    continue
                for p, v in _points(factors[0]).items():
                    one_term = ((name, factors),)
                    assert dev.price(one_term, (int(p),)) == {name: _rounded(v)}
        mul = dict(dev.comb_uses["mul"])
        assert mul["dsp"][0].form == "table"
        assert mul["lut"][0].form == "interp"
        assert dict(dev.comb_uses["div"])["lut"][0].form == "interp"
        assert dict(dev.comb_uses["shl"])["lut"][0].form == "interp"


# An extracted chain keeps its first and last stage in flip-flops, so its shift
# registers hold `depth - 2` and occupy `ceil((depth-2)/32)` SLICEM sites a bit.
# Measured at the depths that tell that apart from `ceil(depth/32)` and
# `ceil((depth-1)/32)`: 32, 33 and 34 take one site, 64, 65 and 66 take two.
def test_the_srl_site_count_is_the_formula_its_table_sampled():
    dev = default_device
    sites = {1: 0, 4: 1} | {32 * i + 3: i + 1 for i in range(1, 17)}

    def sampled(depth):
        return sites[max(k for k in sites if k <= depth)]

    probes = sorted(set(sites) | {2, 3, 5, 32, 33, 34, 64, 65, 66, 96, 97, 512})
    for depth in probes:
        got = dev.price(dev.chain_uses, (depth, 8))["slicem_lut"]
        assert got == 8 * sampled(depth), (depth, got)
    assert dev.price(dev.chain_uses, (1024, 8))["slicem_lut"] == 8 * 32
    # A one-bit chain whose stages carry a reset is never extracted.
    for depth in (4, 33, 64, 128, 1024):
        assert dev.price(dev.chain_uses, (depth, 1))["slicem_lut"] == 0
        assert dev.price(dev.chain_uses_norst, (depth, 1))["slicem_lut"] > 0
    # A chain below the extraction cliff is `depth * width` registers, one above
    # it a head and tail per bit plus one per stage.
    for depth in (1, 2, 3, 4, 5, 64, 1024):
        for width in (1, 8, 32):
            want = depth * width if depth < 4 else 2 * width + depth - 1
            assert dev.price(dev.chain_uses, (depth, width))["ff"] == want


# A select's LUT count per bit is measured up to a fan-in of 64 and read as that
# staircase there; past it the least-squares line through the upper regime takes
# over, since a region can share an operator over more sources than the sweep.
def test_the_mux_curve_is_measured_over_its_sweep_and_fitted_past_it():
    from allo.backend.rtl.devices.spec import MUX_LUT_PER_BIT

    slope, base = 0.5388, 1.6478
    last = max(MUX_LUT_PER_BIT)
    for _, dev in _fabrics():
        for k, v in MUX_LUT_PER_BIT.items():
            assert dev.price(dev.mux_uses, (k, 1)) == {"lut": v}, (dev.name, k)
        for k in (last + 1, 96, 128):
            want = _rounded((base + slope * k) * 100)
            assert dev.price(dev.mux_uses, (k, 100)) == {"lut": want}
        assert dev.price(dev.mux_uses, (128, 1))["lut"] > MUX_LUT_PER_BIT[last]


# The curve is two regimes: a LUT6 absorbs three (data, select) pairs up to 24
# sources and the cost per source rises by a quarter above that, so a line
# fitted to the low end reads far short at the top of the sweep.
def test_the_mux_curve_steepens_past_two_dozen_sources():
    from allo.backend.rtl.devices.spec import MUX_LUT_PER_BIT

    def slope_between(lo, hi):
        return (MUX_LUT_PER_BIT[hi] - MUX_LUT_PER_BIT[lo]) / (hi - lo)

    assert slope_between(4, 24) < 0.45
    assert slope_between(26, 64) > 0.5
    # The measured staircase never steps down.
    values = [MUX_LUT_PER_BIT[k] for k in sorted(MUX_LUT_PER_BIT)]
    assert values == sorted(values)


# Several rows under one archetype are several cores, each declared under its
# own symbol: `mnemonic` tells apart two rows of one kind and signature, and the
# archetype's own row keeps the plain name.
def test_a_fabric_declares_every_row_of_an_archetype():
    from allo.backend.rtl.devices import ultrascalex
    from allo.backend.rtl.devices import ip as catalog
    from allo.backend.rtl.devices.spec import IPRow, Part

    fast = IPRow(
        FADD - 1,
        {"lut": 400, "ff": 300, "dsp": 3},
        in_delay_ns=0.5,
        min_period_ns=0.0,
        out_delay_ns=0.5,
        mnemonic="add_dsp",
    )
    part = Part(
        name="twoadds",
        part="xcu55c-fsvh2892-2L-e",
        grade=ultrascalex.GRADE_2L,
        capacity={
            "lut": 1_303_680,
            "ff": 2_607_360,
            "dsp": 9_024,
            "bram36": 2_016,
            "uram288": 960,
        },
    )
    saved = ultrascalex.IP[catalog.fadd]
    # The fabric's entry is already a candidate set where it measured more than
    # one core, so the fabricated row joins it rather than nesting inside.
    ultrascalex.IP[catalog.fadd] = (
        *(saved if isinstance(saved, tuple) else (saved,)),
        fast,
    )
    try:
        dev = ultrascalex.build(part)
    finally:
        ultrascalex.IP[catalog.fadd] = saved

    symbols = {o.symbol for o in dev.operators}
    assert f"add_f32_f32_f32_l{FADD}" in symbols  # the archetype's own name
    assert f"add_dsp_f32_f32_f32_l{FADD - 1}" in symbols
    assert dev.operator_uses[f"add_dsp_f32_f32_f32_l{FADD - 1}"]

    @kernel
    def addk(A: f32[8], B: f32[8], C: f32[8]):
        for i in range(8):
            C[i] = A[i] + B[i]

    rtl = _to_rtl(addk, device=dev)
    injected = Dcp(rtl).attrs("allo.dcp.operator", "sym_name")
    assert {f"add_f32_f32_f32_l{FADD}", f"add_dsp_f32_f32_f32_l{FADD - 1}"} <= set(
        injected
    )
    impls = _impls(rtl.schedule())
    assert f"add_dsp_f32_f32_f32_l{FADD - 1}" in impls
    assert f"add_f32_f32_f32_l{FADD}" not in impls


def test_remove_operator_withdraws_a_candidate_and_its_cost():
    @operator_ip(
        optype=OperatorType.ADD,
        mnemonic="add_fast",
        latency=FADD - 1,
        in_delay_ns=0.5,
        pipelined=True,
        style="ce",
    )
    def add_fast(a: f32, b: f32) -> f32: ...

    @kernel
    def addk(A: f32[8], B: f32[8], C: f32[8]):
        for i in range(8):
            C[i] = A[i] + B[i]

    dev = default_device.copy()
    dev.add_operator(add_fast)
    dev.set_operator_uses(add_fast, {dev.resources["lut"]: Const(300.0)})
    assert add_fast.symbol in _impls(_to_rtl(addk, device=dev).schedule())

    dev.remove_operator(add_fast)
    assert all(o.symbol != add_fast.symbol for o in dev.operators)
    assert add_fast.symbol not in dev.operator_uses
    rtl = _to_rtl(addk, device=dev)
    assert add_fast.symbol not in Dcp(rtl).attrs("allo.dcp.operator", "sym_name")
    assert f"add_f32_f32_f32_l{FADD}" in _impls(rtl.schedule())
    # A core the device does not declare cannot be withdrawn.
    with pytest.raises(ValueError):
        dev.remove_operator(add_fast.symbol)


# --- index casts and standalone affine expressions ---------------------------


# The frontend only builds the signed `index_cast`, so the unsigned variant is
# injected by rewriting the cast in place. The emitter must zero-extend it: the
# i8 bits 0xC8 subscript A[200], where sign extension into the 9-bit address
# reads A[456]. A 512-deep array so the extension bits survive the address
# truncation; at 256 the two lowerings agree.
def test_index_castui_zero_extends():
    from allo._mlir import ir
    from allo._mlir.dialects import arith as arith_d

    @kernel
    def castui(idx: i8[1], A: i32[512], out: i32[1]):
        out[0] = A[idx[0]]

    rtl = _to_rtl(castui)
    module = rtl.module
    with module.context, ir.Location.unknown():
        casts = [
            op
            for op in _walk(module.operation)
            if op.name == "arith.index_cast" and str(op.operands[0].type) == "i8"
        ]
        assert casts, "the i8 subscript lowers through one index_cast"
        for op in casts:
            new = arith_d.IndexCastUIOp(
                op.results[0].type, op.operands[0], ip=ir.InsertionPoint(op)
            )
            op.results[0].replace_all_uses_with(new.operation.results[0])
            op.erase()

    idx = np.array([-56], np.int8)  # bit pattern 0xC8, 200 read unsigned
    A = np.arange(512, dtype=np.int32)
    out = np.zeros(1, np.int32)
    rtl.cosim(idx, A, out)
    assert out[0] == 200


# `evalAffine` lowers a standalone apply's floordiv unsigned, so an argument
# that can go negative is refused before scheduling. The frontend never builds
# one; the apply is injected over the loop counter.
def test_signed_affine_division_is_refused():
    from allo._mlir import ir

    @kernel
    def sdiv(out: i32[16]):
        for i in range(16):
            out[i] = 7

    rtl = _to_rtl(sdiv)
    _inject_apply(
        rtl,
        lambda d0: ir.AffineExpr.get_floor_div(d0 - 100, ir.AffineConstantExpr.get(3)),
    )
    with pytest.raises(RuntimeError, match="ALLO-N0017"):
        rtl.schedule()


def _inject_apply(rtl, build_expr, dims=1):
    """Insert `apply(build_expr(d0..))` over ``dims`` copies of the loop
    counter, cast to i32 as the stored value of the kernel's one store. The
    frontend leaves no standalone apply behind, so the pricing path is reached
    by injection."""
    from allo._mlir import ir
    from allo._mlir.dialects import affine as affine_d
    from allo._mlir.dialects import arith as arith_d

    module = rtl.module
    with module.context, ir.Location.unknown():
        stores = [op for op in _walk(module.operation) if op.name == "affine.store"]
        loops = [op for op in _walk(module.operation) if op.name == "affine.for"]
        assert len(stores) == 1 and len(loops) == 1
        iv = loops[0].regions[0].blocks[0].arguments[0]
        exprs = [ir.AffineDimExpr.get(i) for i in range(dims)]
        amap = ir.AffineMap.get(dims, 0, [build_expr(*exprs)])
        ip = ir.InsertionPoint(stores[0])
        apply_op = affine_d.AffineApplyOp(
            amap, [iv] * dims, results=[ir.IndexType.get()], ip=ip
        )
        cast = arith_d.IndexCastOp(
            ir.IntegerType.get_signless(32), apply_op.operation.results[0], ip=ip
        )
        stores[0].operands[0] = cast.operation.results[0]


# A standalone apply is priced as its map's cone rather than as the free
# default row: `d0 * 3` is one shift-add, so the schedule carries one add
# step, and the emitted cone computes the map.
def test_a_standalone_apply_is_priced_as_its_cone():
    @kernel
    def scaled(out: i32[16]):
        for i in range(16):
            out[i] = 7

    rtl = _to_rtl(scaled)
    _inject_apply(rtl, lambda d0: d0 * 3)
    rtl.schedule()
    applies = [
        op
        for op in _walk(rtl.dcp_module.operation, "allo.dcp.compute")
        if "map" in op.attributes
    ]
    assert len(applies) == 1
    stamped = applies[0].attributes["in_delay"].value
    assert stamped == pytest.approx(round(comb_step_ns("add"), 2), abs=1e-3)
    out = np.zeros(16, np.int32)
    rtl.cosim(out)
    assert np.array_equal(out, np.arange(16, dtype=np.int32) * 3)
    # The report row exports the cone's operator counts, and the estimator
    # prices them rather than dropping the unit as unmodelled.
    units = [
        u
        for f in rtl.microarch.funcs
        for r in f.regions
        for u in r.units
        if u.identity.startswith("apply")
    ]
    assert len(units) == 1
    assert (units[0].adders, units[0].multipliers, units[0].dividers) == (1, 0, 0)
    assert "apply" not in rtl.estimation.unmodelled


# Every layer reads `applyExprOf`, the map's simplified form: term collection
# folds this cone to `d1 * 3`, one shift-add, priced, reported, and built as
# such rather than as the three-adder raw form.
def test_an_apply_is_priced_on_its_simplified_form():
    @kernel
    def folded(out: i32[16]):
        for i in range(16):
            out[i] = 7

    rtl = _to_rtl(folded)
    _inject_apply(rtl, lambda d0, d1: (d0 + d1) * 3 - d0 * 3, dims=2)
    rtl.schedule()
    applies = [
        op
        for op in _walk(rtl.dcp_module.operation, "allo.dcp.compute")
        if "map" in op.attributes
    ]
    assert len(applies) == 1
    assert applies[0].attributes["in_delay"].value == pytest.approx(
        round(comb_step_ns("add"), 2), abs=1e-3
    )
    out = np.zeros(16, np.int32)
    rtl.cosim(out)
    assert np.array_equal(out, np.arange(16, dtype=np.int32) * 3)
    units = [
        u
        for f in rtl.microarch.funcs
        for r in f.regions
        for u in r.units
        if u.identity.startswith("apply")
    ]
    assert [(u.adders, u.multipliers, u.dividers) for u in units] == [(1, 0, 0)]


# An index division by a non-power-of-two constant is a reciprocal multiply
# sized by the dividend's proven range, so it schedules at the requested
# period instead of derating the module for a combinational divider. The
# typed path is untouched and still binds the device's divider IP.
def test_a_constant_index_division_is_a_reciprocal_multiply():
    @kernel
    def label(out: i32[180], rem: i32[180]):
        for n in range(180):
            out[n] = n // 18
            rem[n] = n % 18

    rtl = _to_rtl(label)
    sched = rtl.schedule()
    assert (sched.cycle_ns or PERIOD_NS) == pytest.approx(PERIOD_NS)
    rtl.compile()
    assert "comb.divu" not in rtl.mlir and "comb.modu" not in rtl.mlir
    out = np.zeros(180, np.int32)
    rem = np.zeros(180, np.int32)
    rtl.cosim(out, rem)
    assert np.array_equal(out, np.arange(180) // 18)
    assert np.array_equal(rem, np.arange(180) % 18)


# A typed constant division always expands to the reciprocal: a narrow
# dividend's product multiply fits the clock combinationally, a full-width
# one rides the pipelined multiplier core, and only a dividend too wide for
# the widest such core keeps the divider IP. The signed forms preserve `//`'s
# truncation toward zero, INT_MIN included.
def test_a_typed_constant_division_expands_where_the_multiply_fits():
    @kernel
    def narrow(A: i8[256], q: i8[256], r: i8[256]):
        for i in range(256):
            q[i] = A[i] // 3
            r[i] = A[i] % 5

    rtl = _to_rtl(narrow)
    rtl.schedule()
    rtl.compile()
    assert not [
        op.impl
        for iface in rtl.interfaces.values()
        for op in iface.operators
        if "div" in op.impl or "rem" in op.impl
    ]
    A = np.arange(-128, 128, dtype=np.int8)
    q = np.zeros(256, np.int8)
    r = np.zeros(256, np.int8)
    rtl.cosim(A, q, r)
    assert np.array_equal(q, np.trunc(A / 3).astype(np.int8))
    assert np.array_equal(r, (A - np.trunc(A / 5) * 5).astype(np.int8))

    # `// 18` strips its even factor before the reciprocal; `// 7` unsigned is
    # the overflowed-magic form with the add fixup; `// 7` signed rounds toward
    # zero off the quotient's own sign.
    @kernel
    def wide(A: u32[8], q: u32[8], u: u32[8], B: i32[8], s: i32[8], r: i32[8]):
        for i in range(8):
            q[i] = A[i] // 18
            u[i] = A[i] // 7
            s[i] = B[i] // 7
            r[i] = B[i] % 7

    rtl = _to_rtl(wide)
    rtl.schedule()
    rtl.compile()
    impls = [op.impl for iface in rtl.interfaces.values() for op in iface.operators]
    assert impls and all(im.startswith("mulw_i64") for im in impls)
    A = np.array([0, 1, 17, 18, 4294967295, 4294967294, 2147483647, 12345], np.uint32)
    B = np.array([-2147483648, 2147483647, -7, 7, -1, 1, 0, -123456789], np.int32)
    q = np.zeros(8, np.uint32)
    u = np.zeros(8, np.uint32)
    s = np.zeros(8, np.int32)
    r = np.zeros(8, np.int32)
    rtl.cosim(A.copy(), q, u, B.copy(), s, r)
    assert np.array_equal(q, A // 18)
    assert np.array_equal(u, A // 7)
    truncated = np.trunc(B / 7).astype(np.int32)
    assert np.array_equal(s, truncated)
    assert np.array_equal(r, B - truncated * 7)

    @kernel
    def past_the_row(A: i64[8], out: i64[8]):
        for i in range(8):
            out[i] = A[i] // 3

    rtl = _to_rtl(past_the_row)
    rtl.schedule()
    rtl.compile()
    assert [op.impl for iface in rtl.interfaces.values() for op in iface.operators] == [
        "divsi_i64_i64_i64_l68"
    ]


# A genuinely wide product of narrow operands binds the narrow widening core:
# a 33x33 multiplier is real hardware, where a delivered 64x64 core netlist
# prunes nothing however its inputs are extended. A product of full 64-bit
# values must keep the full core, since the narrow one computes another number.
def test_a_widening_product_binds_the_narrow_multiplier():
    @kernel
    def widen(A: i32[16], B: i32[16], out: i64[16]):
        for i in range(16):
            out[i] = A[i] * B[i]

    rtl = _to_rtl(widen)
    rtl.schedule()
    rtl.compile()
    assert [op.impl for iface in rtl.interfaces.values() for op in iface.operators] == [
        "mulw_i64_i64_i64_l3"
    ]
    rng = np.random.default_rng(11)
    A = rng.integers(-(2**31), 2**31, 16).astype(np.int32)
    B = rng.integers(-(2**31), 2**31, 16).astype(np.int32)
    A[0], B[0] = -2147483648, -2147483648
    A[1], B[1] = 2147483647, -2147483648
    out = np.zeros(16, np.int64)
    rtl.cosim(A.copy(), B.copy(), out)
    assert np.array_equal(out, A.astype(np.int64) * B.astype(np.int64))

    @kernel
    def full(A: i64[8], B: i64[8], out: i64[8]):
        for i in range(8):
            out[i] = A[i] * B[i]

    rtl = _to_rtl(full)
    rtl.schedule()
    rtl.compile()
    assert [op.impl for iface in rtl.interfaces.values() for op in iface.operators] == [
        "mul_i64_i64_i64_l6"
    ]


# A constant table read at literal indices folds to the element it names, and
# a multiply by a small literal recodes to that constant's shift-add network:
# a 3-tap filter builds no multiplier, no DSP and no table.
def test_a_constant_table_coefficient_multiply_is_shift_adds():
    taps = np.array([3, 10, 3], np.int32)

    @kernel
    def fir(A: i32[64], out: i32[64]):
        w: i32[3] = taps
        for i in range(62):
            out[i] = A[i] * w[0] + A[i + 1] * w[1] + A[i + 2] * w[2]

    rtl = _to_rtl(fir)
    rtl.schedule()
    rtl.compile()
    assert not [op.impl for iface in rtl.interfaces.values() for op in iface.operators]
    assert rtl.estimation.area.dsp == 0
    assert not [
        m for f in rtl.microarch.funcs for m in f.mems if m.realization == "rom"
    ]
    A = (np.arange(64, dtype=np.int32) - 32) * 7
    out = np.zeros(64, np.int32)
    rtl.cosim(A, out)
    golden = np.zeros(64, np.int32)
    golden[:62] = A[:62] * 3 + A[1:63] * 10 + A[2:64] * 3
    assert np.array_equal(out, golden)


# The recoding is exact for either sign of the factor; a factor whose
# non-adjacent form needs more than three add/subs keeps the multiplier IP,
# which a DSP slice serves better than a deep adder tree.
def test_a_small_constant_multiply_recodes_and_a_wide_one_keeps_the_ip():
    @kernel
    def scale(A: i32[32], b: i32[32], c: i32[32], d: i32[32]):
        for i in range(32):
            b[i] = A[i] * 6
            c[i] = A[i] * -5
            d[i] = A[i] * 341

    rtl = _to_rtl(scale)
    rtl.schedule()
    rtl.compile()
    impls = [op.impl for iface in rtl.interfaces.values() for op in iface.operators]
    assert len(impls) == 1 and impls[0].startswith("mul_i32")
    A = (np.arange(32, dtype=np.int32) - 16) * 1000
    b = np.zeros(32, np.int32)
    c = np.zeros(32, np.int32)
    d = np.zeros(32, np.int32)
    rtl.cosim(A, b, c, d)
    assert np.array_equal(b, A * 6)
    assert np.array_equal(c, A * -5)
    assert np.array_equal(d, A * 341)


# A product of two loop counters spans the counters' ranges, not the `index`
# carrier it is written in: under the `% 16` mask the multiply narrows to four
# bits and no multiplier IP or DSP survives.
def test_a_masked_counter_product_narrows_past_the_index_width():
    taps = np.arange(16, dtype=np.int32) * 5 - 40

    @kernel
    def twiddle(out: i32[16, 16]):
        w: i32[16] = taps
        for i in range(16):
            for j in range(16):
                out[i, j] = w[(i * j) % 16]

    rtl = _to_rtl(twiddle)
    rtl.schedule()
    rtl.compile()
    assert not [op.impl for iface in rtl.interfaces.values() for op in iface.operators]
    assert rtl.estimation.area.dsp == 0
    out = np.zeros((16, 16), np.int32)
    rtl.cosim(out)
    i, j = np.meshgrid(np.arange(16), np.arange(16), indexing="ij")
    assert np.array_equal(out, taps[(i * j) % 16])


# A division whose divisor is data reaches the backend at `index` width, where
# no divider core is declared. Moved to the typed width the datapath builds
# `index` at, it binds a pipelined divider and the module keeps its clock
# instead of derating around a combinational one.
def test_a_variable_index_division_binds_the_typed_divider():
    @kernel
    def linearize(n: i32, out: i32[64]):
        for k in range(64):
            out[k] = k % n + k // n

    rtl = _to_rtl(linearize)
    sched = rtl.schedule()
    assert sched.cycle_ns == pytest.approx(PERIOD_NS)
    rtl.compile()
    kinds = {
        op.impl.split("_", 1)[0]
        for iface in rtl.interfaces.values()
        for op in iface.operators
    }
    assert {"divui", "remui"} <= kinds or {"divsi", "remsi"} <= kinds
    out = np.zeros(64, np.int32)
    rtl.cosim(np.int32(7), out)
    k = np.arange(64)
    assert np.array_equal(out, k % 7 + k // 7)


# A mixed-signedness pair promotes to a signed type holding both ranges
# (i16 with u16 gives i17), so the comparison and the division follow the
# operand values rather than C's unsigned-domain reinterpretation. The i17
# division that promotion mints has no divider row; it widens to the i32 row
# instead of derating the clock as a combinational divider.
def test_a_mixed_sign_pair_computes_values_not_bit_patterns():
    @kernel
    def mixed(a: i16[16], b: u16[16], lt: i32[16], q: i32[16], r: i32[16]):
        for i in range(16):
            lt[i] = a[i] < b[i]
            q[i] = a[i] // b[i]
            r[i] = a[i] % b[i]

    rtl = _to_rtl(mixed)
    sched = rtl.schedule()
    assert sched.cycle_ns == pytest.approx(PERIOD_NS)
    impls = _impls(sched)
    assert any(m.startswith("divsi_i32") for m in impls)
    assert any(m.startswith("remsi_i32") for m in impls)
    rtl.compile()
    av = np.arange(-8, 8, dtype=np.int64)
    bv = np.array([3, 40000, 7, 65535, 5, 50000, 3, 40000] * 2, dtype=np.int64)
    lt_out = np.zeros(16, np.int32)
    q_out = np.zeros(16, np.int32)
    r_out = np.zeros(16, np.int32)
    rtl.cosim(av.astype(np.int16), bv.astype(np.uint16), lt_out, q_out, r_out)
    # Integer `//` and `%` truncate toward zero, computed on the values.
    q_g = np.where(av < 0, -(-av // bv), av // bv)
    assert np.array_equal(lt_out, (av < bv).astype(np.int32))
    assert np.array_equal(q_out, q_g)
    assert np.array_equal(r_out, av - q_g * bv)


# The map is ONE operator: a division cone past the clock period has no seam a
# register could land on, so the scheduler lowers the clock to fit it whole
# and reports the achieved period.
def test_an_apply_division_over_the_period_derates_the_clock():
    from allo._mlir import ir

    @kernel
    def divs(out: i32[16]):
        for i in range(16):
            out[i] = 7

    rtl = _to_rtl(divs)
    _inject_apply(
        rtl,
        lambda d0: ir.AffineExpr.get_floor_div(d0 + 5, ir.AffineConstantExpr.get(3)),
    )
    sched = rtl.schedule()
    assert sched.cycle_ns > PERIOD_NS
    assert sched.compiler.options.cycle_ns == pytest.approx(PERIOD_NS)


# The cone's division is built and priced as the reciprocal multiply: no
# divider unit, the multiplier's shift-adds counted as adders, and the
# hardware still computes the map.
def test_an_apply_division_builds_the_reciprocal():
    from allo._mlir import ir

    @kernel
    def rdiv(out: i32[16]):
        for i in range(16):
            out[i] = 7

    rtl = _to_rtl(rdiv)
    _inject_apply(
        rtl,
        lambda d0: ir.AffineExpr.get_floor_div(d0 + 5, ir.AffineConstantExpr.get(3)),
    )
    rtl.schedule()
    units = [
        u
        for f in rtl.microarch.funcs
        for r in f.regions
        for u in r.units
        if u.identity.startswith("apply")
    ]
    assert len(units) == 1
    assert units[0].dividers == 0 and units[0].adders > 1
    out = np.zeros(16, np.int32)
    rtl.cosim(out)
    assert np.array_equal(out, (np.arange(16) + 5) // 3)


def test_an_apply_residue_builds_the_reciprocal():
    from allo._mlir import ir

    @kernel
    def rmod(out: i32[16]):
        for i in range(16):
            out[i] = 7

    rtl = _to_rtl(rmod)
    _inject_apply(
        rtl,
        lambda d0: ir.AffineExpr.get_mod(d0 * 3 + 2, ir.AffineConstantExpr.get(5)),
    )
    rtl.schedule()
    units = [
        u
        for f in rtl.microarch.funcs
        for r in f.regions
        for u in r.units
        if u.identity.startswith("apply")
    ]
    assert len(units) == 1
    assert units[0].dividers == 0
    out = np.zeros(16, np.int32)
    rtl.cosim(out)
    assert np.array_equal(out, (np.arange(16) * 3 + 2) % 5)


# A multiply whose only consumer is one add becomes `allo.muladd` and binds the
# device's fused row where one exists, so the multiply-to-add hop never crosses
# the fabric. A width no row declares stays unfused, as does a multiply with a
# second consumer or a constant factor, which recodes to shift-adds instead.
def test_a_multiply_feeding_one_add_fuses_onto_the_device_row():
    @kernel
    def mac(A: i32[16], B: i32[16], C: i32[16], out: i32[16]):
        for i in range(16):
            out[i] = A[i] * B[i] + C[i]

    rtl = mac.schedule().export("rtl")
    assert "muladd_i32_i32_i32_i32_l3" in rtl.mlir
    # The pair fused whole: no standalone multiplier core remains.
    assert "mul_i32_i32_i32" not in rtl.mlir

    rng = np.random.default_rng(7)
    A = rng.integers(-(2**31), 2**31, 16, dtype=np.int32)
    B = rng.integers(-(2**31), 2**31, 16, dtype=np.int32)
    C = rng.integers(-(2**31), 2**31, 16, dtype=np.int32)
    out = np.zeros(16, np.int32)
    rtl.cosim(A, B, C, out)
    ref = (A.astype(np.int64) * B.astype(np.int64) + C.astype(np.int64)).astype(
        np.int32
    )
    assert np.array_equal(out, ref)

    @kernel
    def mac16(A: i16[16], B: i16[16], C: i16[16], out: i16[16]):
        for i in range(16):
            out[i] = A[i] * B[i] + C[i]

    @kernel
    def shared(A: i32[16], B: i32[16], o1: i32[16], o2: i32[16]):
        for i in range(16):
            t: i32 = A[i] * B[i]
            o1[i] = t + A[i]
            o2[i] = t + B[i]

    @kernel
    def constf(A: i32[16], out: i32[16]):
        for i in range(16):
            out[i] = A[i] * 100 + A[i]

    # A reduction tree's leaf: the addend is the sibling product, which the
    # fused core would wait a whole multiplier for.
    @kernel
    def tree(A: i32[16], B: i32[16], out: i32[16]):
        for i in range(16):
            out[i] = A[i] * A[i] + B[i] * B[i]

    for kern in (mac16, shared, constf, tree):
        assert "muladd" not in kern.schedule().export("rtl").mlir


# A resource weight scales the scarcity price, so it decides which fabric
# realizes an operation when two cores share timing and differ only in spend.
def test_resource_weights_steer_a_realization_between_fabrics():

    @operator_ip(
        optype=OperatorType.MUL,
        mnemonic="mul_on_lut",
        latency=1,
        in_delay_ns=0.5,
        pipelined=True,
        style="ce",
    )
    def mul_on_lut(a: i16, b: i16) -> i16: ...

    @kernel
    def mulk(x: i16[8], y: i16[8], out: i16[8]):
        for i in range(8):
            out[i] = x[i] * y[i]

    dev = default_device.copy()
    dev.add_operator(mul_on_lut)
    dev.set_operator_uses(mul_on_lut, {dev.resources["lut"]: Const(200.0)})

    def chosen(**weights):
        rtl = _to_rtl(mulk, device=dev)
        rtl.set_scheduler_opt(scheduler="exact", O="area", resource_weights=weights)
        return _impls(rtl.schedule())

    # One DSP (2311) undercuts 200 LUTs (3200); at eight times the price the
    # LUT core wins instead.
    assert mul_on_lut.symbol not in chosen()
    assert mul_on_lut.symbol in chosen(dsp=8.0)


# Under the area objective the combinational row joins the candidates, so a
# weight can move a multiply off its IP onto the fabric.
def test_a_weight_moves_a_multiply_onto_the_comb_row():
    @kernel
    def mulk(x: i16[8], y: i16[8], out: i16[8]):
        for i in range(8):
            out[i] = x[i] * y[i]

    # 300 LUTs (4800) for the comb multiply, which an eightfold DSP price ranks
    # past the builtin DSP cores.
    dev = default_device.copy()
    dev.set_comb_delay(CombKind.MUL, 2.0, uses={dev.resources["lut"]: Const(300.0)})

    rtl = _to_rtl(mulk, device=dev)
    rtl.set_scheduler_opt(scheduler="exact", O="area")
    assert any("mul" in s for s in _impls(rtl.schedule()))

    rtl = _to_rtl(mulk, device=dev)
    rtl.set_scheduler_opt(scheduler="exact", O="area", resource_weights={"dsp": 8.0})
    assert "weight = 8" in rtl.dcp
    assert not _impls(rtl.schedule())
    rng = np.random.default_rng(3)
    x = rng.integers(-(2**15), 2**15, 8, dtype=np.int16)
    y = rng.integers(-(2**15), 2**15, 8, dtype=np.int16)
    out = np.zeros(8, np.int16)
    rtl.cosim(x, y, out)
    assert np.array_equal(out, (x.astype(np.int32) * y).astype(np.int16))


# The area objective takes the muladd fusion only when it prices below the
# multiply plus the add it replaces; on this device it does not, so the pair
# fuses under cycles and stays apart under area.
def test_the_area_objective_gates_the_muladd_fusion_by_price():
    @kernel
    def mac(A: i32[16], B: i32[16], C: i32[16], out: i32[16]):
        for i in range(16):
            out[i] = A[i] * B[i] + C[i]

    # The device row's declaration is always in the module; a second mention
    # is a compute op bound to it.
    row = "@muladd_i32_i32_i32_i32_l3"
    rtl = _to_rtl(mac)
    rtl.set_scheduler_opt(scheduler="exact", O="cycles")
    assert rtl.dcp.count(row) > 1

    rtl = _to_rtl(mac)
    rtl.set_scheduler_opt(scheduler="exact", O="area")
    assert rtl.dcp.count(row) == 1


# Each multiply is declared twice at one depth, as DSP and as fabric; the rows
# differ only in spend, so a DSP weight moves the design onto LUTs without
# changing its schedule.
# i64, not i32: rank is latency before price, so a fabric row only competes
# with the DSP row declared at its own depth. At 32 bits the DSP row that holds
# this clock is a cycle shallower than the fabric one, and no weight can move a
# multiply onto a longer row.
def test_a_dsp_weight_moves_a_multiply_onto_the_fabric_row():
    @kernel
    def mulk(x: i64[8], y: i64[8], out: i64[8]):
        for i in range(8):
            out[i] = x[i] * y[i]

    def built(scheduler, **weights):
        rtl = _to_rtl(mulk)
        rtl.set_scheduler_opt(scheduler=scheduler, resource_weights=weights)
        res = rtl.schedule()
        return rtl, _impls(res), [r.latency for r in res.regions()]

    # Both scheduler paths read the same prices: the heuristic ranks the rows,
    # the exact solve drops the dearer of a same-timing pair before selection.
    for scheduler in ("heuristic", "exact"):
        _, dsp, dsp_lat = built(scheduler)
        rtl, lut, lut_lat = built(scheduler, dsp=8.0)
        assert dsp == {"mul_i64_i64_i64_l6"}
        assert lut == {"mullut_i64_i64_i64_l6"}
        assert dsp_lat == lut_lat

    rng = np.random.default_rng(5)
    x = rng.integers(-(2**31), 2**31, 8, dtype=np.int64)
    y = rng.integers(-(2**31), 2**31, 8, dtype=np.int64)
    out = np.zeros(8, np.int64)
    rtl.cosim(x, y, out)
    assert np.array_equal(out, x * y)


# The i16 fabric product is combinational up to its consumer's register, so
# taking it over the DSP row costs the chain one cycle; the i32 pair costs none.
def test_the_fabric_i16_row_pays_a_cycle_for_the_dsp_it_saves():
    @kernel
    def mulk(x: i16[8], y: i16[8], out: i16[8]):
        for i in range(8):
            out[i] = x[i] * y[i]

    def built(**weights):
        rtl = _to_rtl(mulk)
        rtl.set_scheduler_opt(resource_weights=weights)
        res = rtl.schedule()
        return _impls(res), [r.latency for r in res.regions()]

    dsp, dsp_lat = built()
    lut, lut_lat = built(dsp=8.0)
    assert dsp == {"mul_i16_i16_i16_l1"}
    assert lut == {"mullut_i16_i16_i16_l1"}
    assert [a + 1 for a in dsp_lat] == lut_lat
