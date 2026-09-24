# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""One test per unfilled row of ``docs/source/designs/ip_gaps.rst``.

Each one either **fails** (the capability is missing and the test says so
loudly, via ``xfail(strict=True)``, so filling the gap also fails the test and
forces the row to be closed) or **asserts the hole is still there** (for a
"cannot refuse" row, where the evidence IS that something composes cleanly
that should not).

A gap with no test is a gap nobody can see. Running this file is how you find
out what the library still cannot do, without reading the page.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import allo.dataflow as df
from examples.tinytpu.ip import placeholders
from allo.compose import (
    Architecture, Channel, Memory, Unit, unit)


# ---------------------------------------------------------------- fixtures
@unit(reads=("a",), writes=("b",), parameters=())
def copy_unit():
    x: int32 = a.get()          # noqa
    b.put(x)                    # noqa


@unit(instances=("T",), writes=("narrow",), parameters=("T",))
def chain_src():
    i = df.get_pid()                          # noqa
    with allo.meta_if(i < T):                 # noqa
        narrow[i].put(1)                      # noqa


@unit(instances=("T",), reads=("narrow",), memories=("out",),
      parameters=("T",))
def chain_over(sink: int32[4]):
    i = df.get_pid()                          # noqa
    with allo.meta_if(i < T):                 # noqa
        sink[i] = narrow[i + 1000].get()      # noqa


@unit(instances=("T",), reads=("chain",), writes=("chain",), parameters=("T",))
def stage_a():
    i = df.get_pid()                          # noqa
    with allo.meta_if(i != T - 1):            # noqa
        chain[i + 1].put(chain[i].get())      # noqa


@unit(instances=("T",), reads=("chain",), writes=("chain",), parameters=("T",))
def stage_b():
    i = df.get_pid()                          # noqa
    with allo.meta_if(i != T - 1):            # noqa
        chain[i + 1].put(chain[i].get())      # noqa


@unit(writes=("in0", "in1"), parameters=("N",))
def two_site_src():
    for i in range(N):          # noqa
        in0.put(1)              # noqa
    for j in range(N):          # noqa
        in1.put(2)              # noqa


@unit(reads=("in0", "in1"), memories=("out5",), parameters=("N",))
def two_site_mem(sink: int32[8]):
    buf: int32[N]               # noqa
    for i in range(N):          # noqa
        buf[i] = in0.get()      # noqa
    for j in range(N):          # noqa
        buf[j] = in1.get()      # noqa
    for k in range(N):          # noqa
        sink[k] = buf[k]


# ---------------------------------------------------------- cannot express
@pytest.mark.xfail(strict=True, reason=placeholders.SECOND_INSTANCE.needs)
def test_second_instance_of_a_unit():
    """One Unit object, twice in one region, against different channels.

    The FRONT END can do this now -- `@df.unit` binds ports positionally and
    tests/dataflow/test_stream_ports.py instantiates one unit twice. This
    library cannot, because compose.py nests kernel source and a channel is
    reached by lexical name. The row is adoption, not a missing mechanism."""
    Architecture(name="two_copies", parameters={}, memories=(),
                 channels=(Channel("a", "int32", "4"),
                           Channel("b", "int32", "4"),
                           Channel("c", "int32", "4")),
                 units=(copy_unit, copy_unit))


def test_no_placement_concept():
    """compose.py has no way to say where an instance lives, so the
    Jalapeno requirement -- explicit compile-time placement -- has nowhere to
    be written down."""
    fields = {f.name for f in dataclasses.fields(Unit)}
    assert fields == {"body", "instances", "memories", "reads", "writes",
                      "parameters", "isa", "directives", "legality"}, (
        "Unit grew a field -- if it is a placement, close the PLACEMENT row")
    with pytest.raises(placeholders.NotBuilt):
        placeholders.PLACEMENT()


def test_no_collective_topology():
    """A Channel is a point-to-point stream or an array of them. There is no
    kind, so a collective is whatever chain the unit bodies happen to index."""
    fields = {f.name for f in dataclasses.fields(Channel)}
    assert fields == {"name", "dtype", "depth", "shape", "carries"}, (
        "Channel grew a field -- if it is a topology, close the COLLECTIVE row")
    with pytest.raises(placeholders.NotBuilt):
        placeholders.COLLECTIVE()


def test_unit_cannot_declare_latency():
    """Flagged to `unit-actions`: a latency belongs to an action, not to a
    unit, and reduce_tree's two outputs are the case that needs it."""
    from examples.tinytpu.ip.reduce import ReduceParams
    p = ReduceParams()
    assert p.RED_DEPTH - p.RED_TAP_LEVEL == 1, (
        "the tap and the root are one adder level apart at the default "
        "parameter set, and nothing says what that is in cycles")
    assert "latency" not in {f.name for f in dataclasses.fields(Unit)}
    with pytest.raises(placeholders.NotBuilt):
        placeholders.UNIT_LATENCY()


def test_datatype_parametric_slices_widen_to_32_bits():
    """A unit CAN be datatype-parametric -- reduce_tree is -- but every bit
    slice whose bounds are expressions over a named width loses its inferred
    width. The results are right and the circuit is wider than declared."""
    import warnings
    from examples.tinytpu.ip.reduce import DotTree
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        df.build(DotTree(name="dot_tree_widths").region, target="simulator")
    widened = [w for w in caught
               if "bitwidth of the slice" in str(w.message)]
    assert widened, (
        "no slice widened -- if the front end learned to fold a constant into "
        "a slice bound, close this row and re-measure the tree's area")


def test_full_width_bit_slice_does_not_build():
    """`w[0:32]` on a `UInt(32)` lowers to an `arith.trunci` from i32 to i32.
    It is why reduce_tree refuses RED_GROUPS=1 rather than emitting a tap that
    happens to be the root."""
    import allo                                       # noqa: F401
    from allo.ir.types import Stream, UInt, int32      # noqa: F401

    @df.region()
    def full_width_slice(out: int32[1]):
        s: Stream[UInt(32), 4]

        @df.kernel(mapping=[1])
        def prod():
            w: UInt(32) = 5
            s.put(w)

        @df.kernel(mapping=[1], args=[out])
        def cons(o: int32[1]):
            w: UInt(32) = s.get()
            v: int32 = w[0:32]
            o[0] = v

    with pytest.raises(Exception):
        df.build(full_width_slice, target="simulator")


# ----------------------------------------------------------- cannot refuse
def test_two_writers_on_a_chain_not_refused():
    """The evidence IS that this composes. A scalar channel with two writers
    is refused; a stream ARRAY with two writers is not, because the owner is
    per element and the element is a runtime index."""
    Architecture(name="two_writers_chain", parameters={"T": 4}, memories=(),
                 channels=(Channel("chain", "int32", "4", ("T",)),),
                 units=(stage_a, stage_b))
    with pytest.raises(AssertionError, match="writes in both"):
        Architecture(name="two_writers_scalar", parameters={}, memories=(),
                     channels=(Channel("a", "int32", "4"),
                               Channel("b", "int32", "4")),
                     units=(copy_unit,
                            Unit(body=copy_unit.body, reads=("a",),
                                 writes=("b",))))
    with pytest.raises(placeholders.NotBuilt):
        placeholders.CHAIN_OWNERSHIP()


def test_chain_index_outside_shape_not_refused():
    """An index 1000 past a stream array's declared shape composes; the front
    end then reports it as an undefined variable."""
    arch = Architecture(
        name="overrun", parameters={"T": 4},
        memories=(Memory("out", "int32[4]"),),
        channels=(Channel("narrow", "int32", "4", ("T",)),),
        units=(chain_src, chain_over))
    assert "narrow[i + 1000]" in arch.source()


def test_two_write_sites_not_refused():
    """The ELAB-366 shape: a local array written in two places. It composes,
    it builds, and nothing remarks on it -- which is how a design reached
    Design Compiler with a true dual-write-port RAM named _1R1W."""
    arch = Architecture(
        name="two_write_sites", parameters={"N": 8},
        memories=(Memory("out5", "int32[8]"),),
        channels=(Channel("in0", "int32", "4"), Channel("in1", "int32", "4")),
        units=(two_site_src, two_site_mem))
    mod = df.build(arch.region(), target="simulator")
    out = np.zeros(8, np.int32)
    mod(out)
    assert (out == 2).all()
    with pytest.raises(placeholders.NotBuilt):
        placeholders.ASIC_LEGAL_MEMORY()


def test_units_do_not_declare_arithmetic():
    """pe computes an int8 x int8 product into an int32 and its interface says
    only T, VW, AW. reduce_tree declares RED_IN and RED_ACC and is the shape
    the other eight would take."""
    from examples.tinytpu.ip.units.pe import pe
    from examples.tinytpu.ip.units.reduction_tree import (
        reduce_tree)
    assert "int8" in pe.source() and "int8" not in pe.declared_names
    assert {"RED_IN", "RED_ACC"} <= set(reduce_tree.parameters)
    with pytest.raises(placeholders.NotBuilt):
        placeholders.DECLARED_ARITHMETIC()


# ------------------------------------------------------------- the register
def test_every_gap_has_a_test_and_states_what_would_close_it():
    source = open(__file__, encoding="utf-8").read()
    for gap in placeholders.GAPS:
        assert gap.real_when and gap.evidence and gap.needs, gap.name
        assert gap.kind in ("cannot express", "cannot refuse"), gap.name
        assert gap.name.upper() in source or gap.name in source, (
            f"{gap.name} is declared in placeholders.py and no test here "
            f"exercises it")
