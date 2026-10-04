# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""README D-18: a schedule belongs to the function that needs it and travels
with it.

``Architecture.directives`` runs, after each unit's own directives, the
directives of every engine the unit binds -- once per distinct engine, since
the loop they name sits in one shared ``func.func``. A directive naming a
function the module does not hold as its own ``func.func`` is refused,
naming the function, never dropped.
"""

from __future__ import annotations

import dataclasses

import pytest

import allo.dataflow as df
from examples.minitpu.template import mac_pe
from examples.minitpu.template.engines import BF16_ACC24, INT8_INT32


def _offset_loops(module):
    return [l for l in str(module).splitlines() if 'loop_name = "offset"' in l]


def test_engine_directive_travels_with_the_engine():
    """``pe_rig`` at bf16: the region's schedule names nothing, and the MLIR
    holds the one ``offset`` loop of ``leading_zeros19``, unrolled."""
    arch = mac_pe.pe_rig(BF16_ACC24, 8, name="pe_d18")
    s = df.customize(arch.region())
    assert [("unroll" in l) for l in _offset_loops(s.module)] == [False]
    arch.directives(s)
    assert [("unroll" in l) for l in _offset_loops(s.module)] == [True]


def test_once_per_distinct_engine():
    calls = []

    def count(s, ctx):
        calls.append((ctx.unit, ctx.engine))
        s.unroll("leading_zeros19:offset")

    eng = dataclasses.replace(BF16_ACC24, directives=count)
    arch = mac_pe.pe_rig_two(eng, eng, 8, name="pe_two_bf16")
    s = df.customize(arch.region())
    arch.directives(s)
    # pe_feed_a binds the engine first (types only); one call covers the six units
    assert calls == [("pe_feed_a", "MAC__a")]
    assert [("unroll" in l) for l in _offset_loops(s.module)] == [True]
    calls.clear()
    arch = mac_pe.pe_rig_two(eng, INT8_INT32, 8, name="pe_two_mixed")
    arch.directives(df.customize(arch.region()))
    assert calls == [("pe_feed_a", "MAC__a")]


def test_directive_on_an_inlined_function_refused():
    """``leading_zeros19`` marked to be inlined: its loop would vanish into
    every caller, so the engine's unroll is refused naming it."""
    arch = mac_pe.pe_rig(BF16_ACC24, 8, name="pe_inl")
    s = df.customize(arch.region())
    s.inline("leading_zeros19")
    with pytest.raises(AssertionError, match="function 'leading_zeros19' is marked to be inlined"):
        arch.directives(s)


def test_directive_on_a_function_with_no_func_refused():
    def helper(s, ctx):
        del ctx
        s.unroll("normalize_helper:k")

    eng = dataclasses.replace(BF16_ACC24, directives=helper)
    arch = mac_pe.pe_rig(eng, 8, name="pe_nofunc")
    with pytest.raises(AssertionError, match="function 'normalize_helper' is not a func.func"):
        arch.directives(df.customize(arch.region()))

    def top_loop(s, ctx):
        del ctx
        s.unroll("offset")

    eng = dataclasses.replace(BF16_ACC24, directives=top_loop)
    arch = mac_pe.pe_rig(eng, 8, name="pe_noname")
    with pytest.raises(AssertionError, match="'offset' names no function"):
        arch.directives(df.customize(arch.region()))
