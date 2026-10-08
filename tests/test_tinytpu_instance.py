# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""TinyTPU-isa as an instance of the template (README D-4, D-15, D-17,
D-19, D-20; ``dev/records/minitpu/tinytpu_instance_2026-10-08.rst``).

The composition is held to the frozen ``ip/tinytpu.py`` wiring (same units
in the same order, same channels, memories and parameter values), its ISA
slots to ``isa_spec.json``, the base to D-19's refusal, the geometry to
D-20's relations, and the ``Architecture(draft=True)`` mechanism the
instance needed to its two obligations: a draft is checked as the composed
result and never emitted alone.
"""

from __future__ import annotations

import numpy as np
import pytest

import allo.dataflow as df
from allo.compose import Architecture, Channel, Memory, isa_slots, unit
from examples.minitpu.template.instances.tinytpu import accumulator as ACC
from examples.minitpu.template.instances.tinytpu import instance as I
from examples.minitpu.template.instances.tinytpu.geometry import TinyTpuGeometry
from examples.tinytpu.ip.tinytpu import architecture as frozen_architecture


def test_composition_equals_the_frozen_wiring():
    g = TinyTpuGeometry()
    a, f = I.architecture(g), frozen_architecture()
    assert [u.name for u in a.units] == [u.name for u in f.units]
    assert sorted(c.name for c in a.channels) == sorted(c.name for c in f.channels)
    assert [m.name for m in a.memories] == [m.name for m in f.memories]
    assert a.parameters == f.parameters
    assert a.geometry is g and not a.draft
    assert a.reference_order == "sequential"
    assert a.engines["MAC"].name == "int8_int32"


def test_isa_slots_equal_the_spec():
    a = I.architecture()
    assert isa_slots(a) == {s: "base" for s in I.BASE_SLOTS} | {
        s: "accumulator" for s in ACC.SLOTS}
    r = I.compare_isa(a)
    assert r["missing_in_instance"] == r["extra_in_instance"] == r["module_differs"] == []
    assert set(r["spec"]) == set(r["instance"])


def test_base_alone_is_refused_naming_the_accumulator_queue():
    b = I.base()
    assert b.draft
    with pytest.raises(AssertionError, match="unit sequencer writes undeclared channel 'c_acc'"):
        Architecture(name="base_checked", parameters=b.parameters, memories=b.memories,
                     channels=b.channels, units=b.units, engines=b.engines,
                     order=b.order, slots=b.slots)
    for what in ("source", "region", "machine"):
        with pytest.raises(AssertionError, match="of a draft"):
            getattr(b, what)()
    with pytest.raises(AssertionError, match="of a draft"):
        b.directives(None)


def test_geometry_relations_are_legality():
    g = TinyTpuGeometry()
    assert (g.VW, g.AW, g.WPR, g.OPERAND_ROWS) == (32, 128, 16, 1024)
    assert g.params.VW == g.VW and g.params.AW == g.AW
    with pytest.raises(AssertionError, match="AR_RAW_DIST=5 > T=4"):
        TinyTpuGeometry(AR_RAW_DIST=5)
    with pytest.raises(AssertionError, match="AR_RAW_DIST=4 > T=2"):
        TinyTpuGeometry(T=2)
    with pytest.raises(AssertionError, match="11-bit address"):
        TinyTpuGeometry(MAXDIM=96)
    # A geometry at a different T moves every derived width through the
    # engine's relation, and the option's contract parameter rides along.
    g8 = TinyTpuGeometry(T=8, MAXDIM=32)
    a8 = I.architecture(g8)
    assert a8.parameters["VW"] == 64 and a8.parameters["AW"] == 256
    assert a8.parameters["AR_RAW_DIST"] == 4


def test_option_contract_parameter_must_agree():
    g = TinyTpuGeometry(AR_RAW_DIST=3)
    with pytest.raises(AssertionError, match="option accumulator sets AR_RAW_DIST=4"):
        Architecture.with_options(I.base(g), ACC.option(4))


# --- Architecture(draft=True) on its own ------------------------------------

@unit(memories=("X",), writes=("a", "b"), parameters=("N",))
def src(x: int32[N]):
    for i in range(N):
        a.put(x[i])
        b.put(x[i] + 1)


@unit(reads=("a",), memories=("Y",), parameters=("N",))
def snk_a(y: int32[N]):
    for i in range(N):
        y[i] = a.get()


@unit(reads=("b",), memories=("Z",), parameters=("N",))
def snk_b(z: int32[N]):
    for i in range(N):
        z[i] = b.get()


def _draft(n):
    return Architecture(name="half", parameters={"N": n, "QD": 4},
                        memories=(Memory("X", "int32[N]"), Memory("Y", "int32[N]")),
                        channels=(Channel("a", "int32", "QD"),),
                        units=(src, snk_a), slots=("one",), draft=True)


def test_draft_is_completed_by_an_option_and_checked_as_the_result():
    from allo.compose import Option
    with pytest.raises(AssertionError, match="unit src writes undeclared channel 'b'"):
        Architecture(name="half_checked", parameters={"N": 4, "QD": 4},
                     memories=_draft(4).memories, channels=_draft(4).channels,
                     units=_draft(4).units)
    half = _draft(8)
    with pytest.raises(AssertionError, match="source of a draft"):
        half.source()
    whole = Architecture.with_options(
        half, Option(name="second", units=(snk_b,), memories=(Memory("Z", "int32[N]"),),
                     channels=(Channel("b", "int32", "QD"),), isa=("two",)))
    assert not whole.draft and isa_slots(whole) == {"one": "base", "two": "second"}
    with pytest.raises(AssertionError, match="nothing reads 'b'"):
        Architecture.with_options(
            half, Option(name="chan_only", channels=(Channel("b", "int32", "QD"),)))
    x = np.arange(8, dtype=np.int32)
    y, z = np.zeros(8, np.int32), np.zeros(8, np.int32)
    df.build(whole.region(), target="simulator")(x, y, z)
    assert np.array_equal(y, x) and np.array_equal(z, x + 1)
