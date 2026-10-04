# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""README D-17: a unit is instantiated, and the instantiation binds its
parameters, channels and engines.

``compose.Instance(unit, name, bind)`` composes one ``Unit`` any number of
times in one region; ``bind`` renames the body's free names (and the
memories it binds) for that instance, and every check ``compose`` makes
runs per instance. The two-engine PE in one region (simulator and csim) is
``tests/test_compose_engines.py``; it is built from these instances.
"""

from __future__ import annotations

import numpy as np
import pytest

import allo.dataflow as df
from allo.compose import Architecture, Channel, Instance, Memory, unit
from allo.ir.types import UInt, int32  # noqa: F401  pylint: disable=unused-import
from examples.minitpu.template import mac_pe
from examples.minitpu.template import matrix_engine as me
from examples.minitpu.template.engines import BF16_ACC24, INT8_INT32


def _pos(p):
    assert p["N"] > 0, f"N={p['N']} must be positive"


@unit(memories=("X",), writes=("q",), parameters=("N", "K"), legality=_pos)
def produce(x: int32[N]):
    for i in range(N):
        q.put(x[i] * K)


@unit(memories=("Y",), reads=("q",), parameters=("N",))
def consume(y: int32[N]):
    for i in range(N):
        v: int32 = q.get()
        y[i] = v


def _pair(sfx, n_name, k_name):
    b = {"N": n_name, "q": f"q_{sfx}"}
    return (Instance(produce, f"produce_{sfx}", b | {"K": k_name, "X": f"X_{sfx}"}),
            Instance(consume, f"consume_{sfx}", b | {"Y": f"Y_{sfx}"}))


def test_one_unit_twice_two_bindings():
    """One ``produce`` and one ``consume``, each instantiated twice in one
    region at two sizes and two scales."""
    arch = Architecture(
        name="twice", parameters={"N_a": 4, "N_b": 6, "K_a": 2, "K_b": 3, "QD": 2},
        memories=tuple(Memory(m, f"int32[N_{s}]") for s in "ab" for m in (f"X_{s}", f"Y_{s}")),
        channels=(Channel("q_a", "int32"), Channel("q_b", "int32")),
        units=_pair("a", "N_a", "K_a") + _pair("b", "N_b", "K_b"))
    assert [u.unit for u in arch.units] == [produce, consume, produce, consume]
    assert "def produce_b(x: int32[N_b]):" in arch.units[2].source()
    mod = df.build(arch.region(), target="simulator")
    xa, xb = np.arange(4, dtype=np.int32), np.arange(6, dtype=np.int32)
    ya, yb = np.zeros(4, np.int32), np.zeros(6, np.int32)
    mod(xa, ya, xb, yb)
    assert list(ya) == list(2 * xa) and list(yb) == list(3 * xb)


def test_two_engine_pe_is_instances_of_one_source():
    arch = mac_pe.pe_rig_two(BF16_ACC24, INT8_INT32, 8)
    pes = [u for u in arch.units if isinstance(u, Instance) and u.unit is mac_pe.mac_pe]
    assert [u.name for u in pes] == ["mac_pe_a", "mac_pe_b"]
    assert pes[0].engines == ("MAC__a_IN", "MAC__a_ACC", "MAC__a_MUL", "MAC__a_ADD")


def test_bind_of_a_non_free_name_refused():
    with pytest.raises(AssertionError, match=r"bind names \['i'\], which is not free in the body"):
        Instance(produce, "p", {"i": "j"})
    with pytest.raises(AssertionError, match=r"bind names \['M'\]"):
        Instance(produce, "p", {"M": "M_a"})


def test_legality_runs_per_instance():
    """``legality`` sees the parameter set through the instance's binding."""
    with pytest.raises(AssertionError, match="N=0 must be positive"):
        Architecture(
            name="bad", parameters={"N_a": 0, "K_a": 1, "QD": 2},
            memories=(Memory("X_a", "int32[N_a]"), Memory("Y_a", "int32[N_a]")),
            channels=(Channel("q_a", "int32"),), units=_pair("a", "N_a", "K_a"))
    with pytest.raises(AssertionError, match="DIM=6: the tree engine is a BALANCED"):
        me.mxu_rig(me.TREE, BF16_ACC24, 6, 4)


def test_rebinding_an_instance_composes_the_bindings():
    i1 = Instance(consume, "c", {"q": "q_a", "N": "N_a"})
    i2 = Instance(i1, "c", {"q_a": "q_b"})
    assert i2.unit is consume and i2.bind == {"q": "q_b", "N": "N_a"}


@unit(memories=("X",), writes=("q",), parameters=("N", "W_BITS"))
def slices_by_parameter(x: UInt(32)[N]):
    for i in range(N):
        w: UInt(32) = x[i]
        q.put(w[0:W_BITS])


def test_parameter_in_slice_bound_refused():
    with pytest.raises(AssertionError, match=r"slice `w\[0:W_BITS\]` has bounds that are only "
                       r"parameters \(W_BITS\); Allo cannot infer the bitwidth of the slice"):
        slices_by_parameter.check()
    # inside meta_for the bound names a bound index and folds: accepted
    me.systolic_engine.check()
