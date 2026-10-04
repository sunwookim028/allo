# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""README D-15: an engine is declared, and a swap that changes the accumulate
order is a different function.

``compose.Engine`` is the record (types and widths, ``mul``/``add``/``pack``,
their D-10 latencies, ``order``, numpy references, directives); a ``Unit``
binds its names through ``engines=`` slots; ``Architecture(engines=,
order=, accepts=)`` binds a record to each slot and holds it to the bodies,
the channels and the composite's order. The engines and the PE come from the
U3 prototype (``examples/minitpu/template/``).
"""

from __future__ import annotations

import dataclasses
import os
import tempfile

import numpy as np
import pytest

import allo.dataflow as df
from allo.compose import Architecture, Channel, Memory, engine_slot, unit
from allo.ir.types import UInt, int8, int32  # noqa: F401  pylint: disable=unused-import
from examples.minitpu.template import mac_pe
from examples.minitpu.template import matrix_engine as me
from examples.minitpu.template.engines import BF16_ACC24, INT8_INT32

needs_csim = pytest.mark.skipif(
    not (os.environ.get("MGC_HOME") and os.environ.get("SYSTEMC_HOME")),
    reason="SystemC csim needs Catapult (MGC_HOME) and SYSTEMC_HOME",
)

PE_MEMS = (Memory("A", "UInt(32)[N_WORK]"), Memory("W", "UInt(32)[N_WORK]"),
           Memory("P", "UInt(32)[N_WORK]"), Memory("OUT", "UInt(32)[N_WORK]"))


def _pe(engine, channels=None, **kw):
    return Architecture(name=kw.pop("name", "pe_t"), parameters={"N_WORK": 8, "QD": 4},
                        engines={"MAC": engine}, memories=PE_MEMS,
                        channels=channels or mac_pe._channels(),
                        units=(mac_pe.pe_feed, mac_pe.mac_pe, mac_pe.pe_sink), **kw)


# --- the record -------------------------------------------------------------


def test_engine_record():
    assert engine_slot("MAC_IN_BITS") == ("MAC", "IN_BITS")
    assert engine_slot("MAC__a_ADD") == ("MAC__a", "ADD")
    assert engine_slot("N_WORK") is None
    ns = BF16_ACC24.namespace("MAC")
    assert ns["MAC_ADD"] is BF16_ACC24.add and ns["MAC_IN_BITS"] == 16
    assert BF16_ACC24.add_latency == 3 and INT8_INT32.pack_latency is None
    with pytest.raises(AssertionError, match="IN_BITS=16 but IN"):
        dataclasses.replace(INT8_INT32, IN_BITS=16)
    with pytest.raises(AssertionError, match="order 'diagonal'"):
        dataclasses.replace(INT8_INT32, order="diagonal")
    with pytest.raises(AssertionError, match="latency names 'div'"):
        dataclasses.replace(INT8_INT32, latency={"div": 2})


def test_reference_takes_the_order():
    """The contract reference with the order as an argument: at bf16 the two
    orders are different functions on random data, one function on exact
    sums; at int8 (no rounding) always one."""
    A, W = me.stimulus(BF16_ACC24, 8, 64, seed=7)
    seq, tree = BF16_ACC24.dot(A, W, "sequential"), BF16_ACC24.dot(A, W, "tree")
    assert np.array_equal(seq, me._matrix_rows_prototype(A, W, BF16_ACC24, "sequential"))
    assert np.array_equal(tree, me._matrix_rows_prototype(A, W, BF16_ACC24, "tree"))
    assert np.sum(seq != tree) > 0
    A, W = me.stimulus(BF16_ACC24, 8, 64, seed=7, exact=True)
    assert np.array_equal(BF16_ACC24.dot(A, W, "sequential"), BF16_ACC24.dot(A, W, "tree"))


# --- one PE source, two engines, one region ----------------------------------


def _two_engine_check(mod, n):
    aa, wa, pa = mac_pe.stimulus(BF16_ACC24, n, 0)
    ab, wb, pb = mac_pe.stimulus(INT8_INT32, n, 1)
    oa, ob = np.zeros(n, np.uint32), np.zeros(n, np.uint32)
    mod(aa, wa, pa, oa, ab, wb, pb, ob)
    assert np.array_equal(oa, mac_pe.reference(BF16_ACC24, aa, wa, pa))
    assert np.array_equal(ob, mac_pe.reference(INT8_INT32, ab, wb, pb))


def test_two_engines_one_region_simulator():
    arch = mac_pe.pe_rig_two(BF16_ACC24, INT8_INT32, 128)
    assert {s: e.name for s, e in arch.engines.items()} == {
        "MAC__a": "bf16_acc24", "MAC__b": "int8_int32"}
    _two_engine_check(df.build(arch.region(), target="simulator"), 128)


@needs_csim
def test_two_engines_one_region_csim():
    arch = mac_pe.pe_rig_two(BF16_ACC24, INT8_INT32, 64, name="pe_two_sc")
    with tempfile.TemporaryDirectory() as tmp:
        mod = df.build(arch.region(), target="systemc", mode="csim",
                       project=os.path.join(tmp, "pe_two"))
        _two_engine_check(mod, 64)


# --- refusals ----------------------------------------------------------------


def test_type_mismatch_refused():
    """The PE moves ``MAC_IN`` (int8 at this engine) on a channel declared
    ``UInt(16)``: the bf16 engine's width, not this one's."""
    chans = (Channel("lhs", "UInt(16)", "4"),) + mac_pe._channels()[1:]
    with pytest.raises(AssertionError, match=r"moves `MAC_IN`.*int8_int32.*channel 'lhs'.*UInt\(16\)"):
        _pe(INT8_INT32, chans)
    _pe(BF16_ACC24, chans)  # the same channel IS the bf16 engine's type


def test_lane_bits_mismatch_refused():
    """A packed channel's ``lane_bits`` is the engine width the body slices
    it by: ``lane_bits == IN_BITS``."""
    good = me.mxu_rig(me.SYSTOLIC, INT8_INT32, 4, 4)
    chans = (Channel("me_lhs", depth="QD", lanes="DIM", lane_bits="16"),
             good.channels[1])
    with pytest.raises(AssertionError, match=r"lane_bits='16' \(16\).*MAC_IN_BITS = 8"):
        Architecture(name="mxu_bad", parameters=good.parameters, engines=good.engines,
                     memories=good.memories, channels=chans, units=good.units)


def test_order_mismatch_refused():
    """A composite whose reference is ``sequential`` refuses a ``tree`` part
    -- a matrix engine unit or an engine record -- naming it and the orders,
    unless it accepts ``tree``; then the verdict uses the tree reference."""
    with pytest.raises(AssertionError, match=r"unit matrix_engine accumulates in order 'tree'.*takes 'sequential'"):
        me.mxu_rig(me.TREE, BF16_ACC24, 4, 4, accepts=())
    assert me.mxu_rig(me.TREE, BF16_ACC24, 4, 4).reference_order == "tree"
    assert me.mxu_rig(me.SYSTOLIC, BF16_ACC24, 4, 4).reference_order == "sequential"
    tree_mac = dataclasses.replace(INT8_INT32, name="int8_tree", order="tree")
    with pytest.raises(AssertionError, match=r"engine slot 'MAC' \(int8_tree\) accumulates in order 'tree'.*takes 'sequential' and accepts no other"):
        _pe(tree_mac, order="sequential")
    assert _pe(tree_mac, order="sequential", accepts=("tree",)).reference_order == "tree"
    assert _pe(INT8_INT32, order="sequential", accepts=("tree",)).reference_order == "sequential"


def test_tree_matrix_engine_verified_with_its_order():
    got, want = me.run_rig(me.TREE, BF16_ACC24, 4, 16, seed=5)
    assert np.array_equal(got, want)


@unit(reads=("lhs",), writes=("psum_out",), parameters=("N_WORK", "MAC_MUL"),
      engines=("MAC_IN", "MAC_ACC"))
def pe_param_mul():
    for _ in range(N_WORK):
        a: MAC_IN = lhs.get()
        p: MAC_ACC = MAC_MUL(a, a)
        psum_out.put(p)


@unit(reads=("lhs",), writes=("psum_out",), parameters=("N_WORK", "MUL"),
      engines=("MAC_IN", "MAC_ACC"))
def pe_bare_mul():
    for _ in range(N_WORK):
        a: MAC_IN = lhs.get()
        p: MAC_ACC = MUL(a, a)
        psum_out.put(p)


@unit(reads=("lhs",), writes=("psum_out",), parameters=("N_WORK",),
      engines=("MAC_IN", "MAC_ACC", "MAC_MUL"))
def pe_mul_only():
    for _ in range(N_WORK):
        a: MAC_IN = lhs.get()
        p: MAC_ACC = MAC_MUL(a, a)
        psum_out.put(p)


@unit(memories=("X",), writes=("lhs",), reads=("psum_out",), parameters=("N_WORK",),
      engines=("MAC_IN", "MAC_ACC"))
def sq_io(x: int32[N_WORK]):
    for i in range(N_WORK):
        a: MAC_IN = 1
        lhs.put(a)
        p: MAC_ACC = psum_out.get()
        x[i] = p


def _square(eng, u, params=None):
    return Architecture(
        name="sq", parameters=params or {"N_WORK": 4, "QD": 4}, engines=eng,
        memories=(Memory("X", "int32[N_WORK]"),),
        channels=(Channel("lhs", "MAC_IN"), Channel("psum_out", "MAC_ACC")),
        units=(sq_io, u))


def test_slot_rules():
    # an engine name declared a parameter, not a slot
    with pytest.raises(AssertionError, match="'MAC_MUL' names engine slot 'MAC' but is declared a parameter"):
        pe_param_mul.check()
    # a bare function bound as a parameter
    with pytest.raises(AssertionError, match="binds function 'mul_int8' as bare parameter 'MUL'"):
        _square({"MAC": INT8_INT32}, pe_bare_mul,
                {"N_WORK": 4, "QD": 4, "MUL": INT8_INT32.mul})
    # a slot with no engine, and an engine on no slot
    with pytest.raises(AssertionError, match="binds engine slot 'MAC'.*binds no engine"):
        _square({"MXU": INT8_INT32}, pe_mul_only)
    with pytest.raises(AssertionError, match="bound to slot 'MXU', which no unit binds"):
        _square({"MAC": INT8_INT32, "MXU": INT8_INT32}, pe_mul_only)
    _square({"MAC": INT8_INT32}, pe_mul_only)
    # a slot that is not <slot>_<field>
    with pytest.raises(AssertionError, match="engine slot 'MACIN' is not"):
        dataclasses.replace(pe_mul_only, engines=("MACIN",)).check()


# --- directives travel with the engine ----------------------------------------


def test_engine_directive_applied():
    """``BF16_ACC24.directives`` unrolls the adder's leading-zero count (C10);
    the architecture applies it for the unit binding the engine, and the
    emitted HLS carries the pragma in ``leading_zeros19``."""
    arch = mac_pe.pe_rig(BF16_ACC24, 16, name="pe_dir")
    plain = str(df.customize(arch.region()).build(target="vhls"))
    s = df.customize(arch.region())
    arch.directives(s)
    code = str(s.build(target="vhls"))

    def lz(text):
        i = text.index("leading_zeros19(")
        return text[i:text.index("\n}", i)]

    assert "#pragma HLS unroll" in lz(code)
    assert "#pragma HLS unroll" not in lz(plain)
