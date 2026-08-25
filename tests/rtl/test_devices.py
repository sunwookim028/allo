# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vivado recipes against the operator tables, and the generator over them."""

import json
import re

import pytest

from allo import kernel
from allo.backend.rtl.core import RealizationWarning
from allo.lang import f32, i32, i64
from allo.lang.ip import operator_ip
from allo.operators import arith as allo_arith
from allo.operators import math as amath
from allo.backend.rtl.devices import (
    default_device,
    ip,
    series7,
    ultrascalex,
    versal,
    vivado,
)
from allo.backend.rtl.devices.spec import IPRow

from _common import FMUL


def _tables():
    return [
        ultrascalex.IP,
        series7.IP,
        versal.IP,
        *ultrascalex.IP_BY_GRADE.values(),
    ]


def _rows(table):
    for arche, rows in table.items():
        yield arche, (rows,) if isinstance(rows, IPRow) else rows


def test_recipes_cover_the_catalog():
    # Both directions: an archetype without a recipe cannot be synthesized, and
    # a recipe naming no archetype is dead data.
    assert {a.func_name for a in vivado.RECIPES} == {a.func_name for a in ip.CATALOG}
    assert set(vivado.RECIPES) == set(ip.CATALOG)


def test_every_declared_row_resolves():
    # Every fabric row must reach a recipe whose core family has a base and a
    # latency knob, so a re-characterization cannot outrun the realizer. An
    # `rtl` recipe generates no core; its shape is the staged expression.
    for table in _tables():
        for arche, _ in _rows(table):
            recipe = vivado.RECIPES[arche]
            if recipe.core == "rtl":
                assert re.fullmatch(r"\w+ \* \w+ \+ \w+", recipe.shape)
                continue
            assert recipe.core in vivado.CE_BASE
            assert "{lat}" in vivado.LATENCY[recipe.core]


def test_config_fragments_are_key_value_pairs():
    for recipe in vivado.RECIPES.values():
        if recipe.core == "rtl":
            continue  # no create_ip: the shape is an expression, not CONFIG
        fragments = (
            vivado.CE_BASE[recipe.core],
            recipe.shape,
            recipe.no_dsp,
            vivado.LATENCY[recipe.core].format(lat=1),
        )
        for fragment in fragments:
            for kv in filter(None, fragment.split(",")):
                assert kv.count("=") == 1, kv


def test_dsp_free_candidates_have_the_fragment():
    # An archetype with one row spending DSPs and another spending none needs
    # the DSP-free fragment, or both rows would build the same core and the
    # table would price one piece of hardware under two names.
    for table in _tables():
        for arche, rows in _rows(table):
            if len({"dsp" in r.area for r in rows}) == 2:
                assert vivado.RECIPES[arche].no_dsp, arche.func_name


def test_divider_sign_follows_mnemonic():
    assert "operand_sign=Signed" in vivado.RECIPES[ip.idiv32].shape
    assert "operand_sign=Signed" in vivado.RECIPES[ip.irem8].shape
    assert "operand_sign=Unsigned" in vivado.RECIPES[ip.udiv64].shape
    assert "operand_sign=Unsigned" in vivado.RECIPES[ip.urem16].shape


# --- the generator, driven off a compiled kernel's manifest ------------------


def _generate(k, device=default_device, **kw):
    rtl = k.schedule().export("rtl", **kw)
    rtl.compile()
    return rtl, vivado.generate(rtl.interfaces, device)


def _operators(rtl):
    return [op for iface in rtl.interfaces.values() for op in iface.operators]


def _shim_of(generated, module):
    body = generated.shims.split(f"module {module}(\n", 1)[1]
    return body.split("endmodule", 1)[0]


def _tcl_of(generated, impl):
    block = generated.ip_tcl.split(f"-module_name {impl}_core", 1)[1]
    return block.split("}", 1)[0]


def test_generate_wraps_float_cores_off_the_manifest():
    @kernel
    def fk(A: f32[8], B: f32[8], out: f32[8]):
        for i in range(8):
            out[i] = allo_arith.max(A[i] * B[i], A[i] + B[i])

    rtl, g = _generate(fk)
    assert g.missing == ()
    ops = _operators(rtl)
    for op in ops:
        shim = _shim_of(g, op.module)
        assert f"{op.impl}_core u (" in shim
        assert ".aclken(ce)" in shim and ".s_axis_a_tvalid(1'b1)" in shim
        # The core's own depth is pinned, never left at its silent maximum.
        lat = int(re.search(r"_l(\d+)$", op.impl).group(1))
        tcl = _tcl_of(g, op.impl)
        assert f"CONFIG.Maximum_Latency false CONFIG.C_Latency {lat}" in tcl
    # One create_ip per distinct core: the two compare predicates share one.
    assert g.ip_tcl.count("create_ip") == len(g.cores) < len(ops)


def test_generate_drives_the_predicate_on_the_operation_channel():
    @kernel
    def fk(A: f32[8], B: f32[8], out: f32[8]):
        for i in range(8):
            out[i] = allo_arith.max(A[i], B[i])

    rtl, g = _generate(fk)
    assert g.missing == ()
    # The NaN-propagating max expands into a ugt and a uno compare. Each
    # wrapper drives its own opcode constant into the one Programmable core,
    # and takes the single result bit off the byte-padded channel.
    by_pred = {op.predicate: op for op in _operators(rtl) if op.predicate}
    ugt = _shim_of(g, by_pred["ugt"].module)
    uno = _shim_of(g, by_pred["uno"].module)
    for shim in (ugt, uno):
        assert ".s_axis_operation_tvalid(1'b1)" in shim
        assert "assign y = result[0:0];" in shim
    assert ".s_axis_operation_tdata(8'b00100100)" in ugt
    assert ".s_axis_operation_tdata(8'b00000100)" in uno


def test_generate_builds_the_row_the_area_measured():
    # A row whose area spends no DSPs is rebuilt with the DSP-free fragment,
    # and one whose area spends them is not.
    @kernel
    def fk(a: f32, b: f32) -> f32:
        return a * b + a

    rtl, g = _generate(fk)
    assert g.missing == ()
    for op in _operators(rtl):
        spends_dsp = "dsp" in dict(default_device.operator_uses[op.impl])
        block = _tcl_of(g, op.impl)
        assert ("CONFIG.C_Mult_Usage No_Usage" in block) != spends_dsp


def test_generate_builds_the_fabric_multiply_out_of_luts():
    # The multiply's fabric row spends no DSPs, so it rebuilds with the recipe's
    # DSP-free fragment, which repeats a key the shape sets and must land after
    # it; a `set_property -dict` list resolves the order.
    #
    # i64, because rank is latency before price: a fabric row only competes with
    # the DSP row declared at its own depth, and at 32 bits the DSP row that
    # holds this clock is a cycle shallower than the fabric one.
    @kernel
    def mk(x: i64[8], y: i64[8], out: i64[8]):
        for i in range(8):
            out[i] = x[i] * y[i]

    rtl = mk.schedule().export("rtl")
    rtl.set_scheduler_opt(resource_weights={"dsp": 8.0})
    rtl.compile()
    g = vivado.generate(rtl.interfaces, default_device)
    (mul,) = [op for op in _operators(rtl) if op.impl.startswith("mullut")]
    block = _tcl_of(g, mul.impl)
    assert block.index("Use_LUTs") > block.index("Use_Mults")


def test_generate_wraps_integer_cores():
    # `t` has two consumers, so it keeps the plain multiplier core; the
    # single-use `x * z` over a ready addend fuses and builds the `rtl` shim.
    @kernel
    def ik(x: i32, z: i32) -> i32:
        t: i32 = x * x
        u: i32 = x * z + z
        return (t + x // z) + (u + x % z) + t

    rtl, g = _generate(ik)
    assert g.missing == ()
    by_stem = {op.impl.split("_", 1)[0]: op for op in _operators(rtl)}
    div = _shim_of(g, by_stem["divsi"].module)
    rem = _shim_of(g, by_stem["remsi"].module)
    mul = _shim_of(g, by_stem["mul"].module)
    mad = _shim_of(g, by_stem["muladd"].module)
    # The divider core packs quotient over remainder; each mnemonic slices its
    # own half of the one dout channel.
    for shim in (div, rem):
        assert ".s_axis_dividend_tdata(a)" in shim
        assert ".s_axis_divisor_tdata(b)" in shim
    assert "assign y = dout[63:32];" in div
    assert "assign y = dout[31:0];" in rem
    assert ".CLK(clk), .CE(ce), .A(a), .B(b), .P(y)" in mul
    # The fused core is the shim itself: a staged product, the addend delayed
    # beside it, the add in the last stage, and no generated IP.
    assert "m0 <= a_q * b_q;" in mad and "r <= m0 + c_d1;" in mad
    assert "muladd" not in g.ip_tcl
    assert "CONFIG.latency_configuration Manual" in _tcl_of(g, by_stem["divsi"].impl)
    assert "CONFIG.PipeStages" in _tcl_of(g, by_stem["mul"].impl)


def test_generate_selects_add_against_sub_on_the_shared_core():
    # The Add_Subtract shape is the "Both" core: add and sub are one measured
    # piece of hardware told apart by the operation-channel constant. Leaving
    # the channel undriven would silently compute an add for both.
    @kernel
    def fk(A: f32[8], B: f32[8], o1: f32[8], o2: f32[8]):
        for i in range(8):
            o1[i] = A[i] + B[i]
            o2[i] = A[i] - B[i]

    rtl, g = _generate(fk)
    assert g.missing == ()
    by_stem = {op.impl.split("_", 1)[0]: op for op in _operators(rtl)}
    add = _shim_of(g, by_stem["add"].module)
    sub = _shim_of(g, by_stem["sub"].module)
    assert ".s_axis_operation_tdata(8'b00000000)" in add
    assert ".s_axis_operation_tdata(8'b00000001)" in sub


# --- project scaffolding -----------------------------------------------------


def test_scaffold_writes_split_rtl_and_realization(tmp_path):
    @kernel
    def sf1(A: f32[8], B: f32[8]):
        for i in range(8):
            B[i] = A[i] * A[i]

    @kernel
    def sf_top(A: f32[8], B: f32[8], out: f32[8]):
        sf1(A, B)
        for i in range(8):
            out[i] = B[i] + A[i]

    rtl = sf_top.schedule().export("rtl")
    root = rtl.scaffold_project(str(tmp_path / "prj"))
    # One RTL file per module, listed in the filelist.
    listed = (root / "filelist.f").read_text().split()
    assert "sf_top.sv" in listed and len(listed) == 2
    for name in listed:
        assert (root / name).exists()
    # The manifest keys the boundary by module name.
    manifest = json.loads((root / "manifest.json").read_text())
    assert "sf_top" in manifest
    # The library binds the shortest multiply that fits the clock, so the depth
    # comes off the table rather than being restated here.
    assert f"module mul_f32_f32_f32_l{FMUL}(" in (root / "shims.v").read_text()
    assert "create_ip" in (root / "gen_ip.tcl").read_text()
    # Split emission ran on a copy: the compiled module still exports whole.
    assert rtl.verilog


def test_scaffold_without_a_realizer_degrades_to_rtl_only(tmp_path):
    dev = default_device.copy()
    assert dev.realizer is not None  # a copy keeps the fabric's realizer
    dev.realizer = None

    @kernel
    def fk(a: f32, b: f32) -> f32:
        return a * b

    rtl = fk.schedule().export("rtl", device=dev)
    root = rtl.scaffold_project(str(tmp_path / "prj"))
    assert (root / "fk.sv").exists()
    assert not (root / "shims.v").exists()
