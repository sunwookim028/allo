# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A ``@df.kernel`` that declares its stream ports, and the rules that wiring
one has to satisfy. See ``docs/source/developer/stream_ports.rst``."""

import warnings

import numpy as np
import pytest

import allo.dataflow as df
from allo.ir.types import int32, Stream
from allo.netlist import NetlistError, UndeclaredPremise

N = 8


@df.unit()
def produce(dst: Stream[int32, 4], mem: int32[N]):
    for i in range(N):
        dst.put(mem[i])


@df.unit()
def increment(src: Stream[int32, 4], dst: Stream[int32, 4]):
    for i in range(N):
        dst.put(src.get() + 1)


@df.unit()
def consume(src: Stream[int32, 4], mem: int32[N]):
    for i in range(N):
        mem[i] = src.get()


def refused(rule, build):
    with pytest.raises(NetlistError) as refusal:
        build()
    rules = [violation.rule for violation in refusal.value.violations]
    assert rules == [rule], rules
    for violation in refusal.value.violations:
        assert violation.where and violation.found and violation.repair
    return refusal.value


def test_a_unit_is_instantiated_twice_against_different_streams():
    @df.region()
    def two(A: int32[N], B: int32[N]):
        a: Stream[int32, 4]
        b: Stream[int32, 4]
        c: Stream[int32, 4]
        produce(dst=a, mem=A)
        first = increment(src=a, dst=b)
        second = increment(src=b, dst=c)
        consume(src=c, mem=B)

    module = df.build(two, target="simulator")
    source = np.arange(N, dtype=np.int32)
    result = np.zeros(N, dtype=np.int32)
    module(source, result)
    np.testing.assert_array_equal(result, source + 2)


def test_the_same_units_compose_into_a_second_topology_unedited():
    @df.region()
    def three(A: int32[N], B: int32[N]):
        w: Stream[int32, 4]
        x: Stream[int32, 4]
        y: Stream[int32, 4]
        z: Stream[int32, 4]
        produce(dst=w, mem=A)
        increment(src=w, dst=x)
        increment(src=x, dst=y)
        increment(src=y, dst=z)
        consume(src=z, mem=B)

    module = df.build(three, target="simulator")
    source = np.arange(N, dtype=np.int32)
    result = np.zeros(N, dtype=np.int32)
    module(source, result)
    np.testing.assert_array_equal(result, source + 3)


def test_a_port_is_named_by_the_unit_not_by_the_region():
    @df.region()
    def renamed(A: int32[N], B: int32[N]):
        nothing_like_src: Stream[int32, 4]
        produce(dst=nothing_like_src, mem=A)
        consume(src=nothing_like_src, mem=B)

    module = df.build(renamed, target="simulator")
    source = np.arange(N, dtype=np.int32)
    result = np.zeros(N, dtype=np.int32)
    module(source, result)
    np.testing.assert_array_equal(result, source)


def test_an_array_port_carries_a_chain():
    T = 4

    @df.unit(mapping=[T])
    def stage(chain: Stream[int32, 4][T + 1]):
        (i,) = df.get_pid()
        for k in range(N):
            chain[i + 1].put(chain[i].get() + 1)

    @df.unit()
    def drive(chain: Stream[int32, 4][T + 1], A: int32[N], B: int32[N]):
        for k in range(N):
            chain[0].put(A[k])
        for k in range(N):
            B[k] = chain[T].get()

    @df.region(deadlock_free_because="a chain is not a loop; it drains forward")
    def pipeline(A: int32[N], B: int32[N]):
        link: Stream[int32, 4][T + 1]
        stage(chain=link)
        drive(chain=link, A=A, B=B)

    module = df.build(pipeline, target="simulator")
    source = np.arange(N, dtype=np.int32)
    result = np.zeros(N, dtype=np.int32)
    module(source, result)
    np.testing.assert_array_equal(result, source + T)


def test_a_unit_whose_port_has_two_directions_is_refused():
    def define():
        @df.unit()
        def both_ways(p: Stream[int32, 4]):
            for i in range(N):
                p.put(p.get())

    refused("direction-single-valued", define)


def test_a_port_the_body_never_uses_is_refused():
    def define():
        @df.unit()
        def spare_port(p: Stream[int32, 4], mem: int32[N]):
            for i in range(N):
                mem[i] = 0

    refused("dangling-port", define)


def test_a_wiring_that_does_not_type_check_is_refused():
    @df.unit()
    def deeper(src: Stream[int32, 8], mem: int32[N]):
        for i in range(N):
            mem[i] = src.get()

    def compose():
        @df.region()
        def mismatched(A: int32[N], B: int32[N]):
            a: Stream[int32, 4]
            produce(dst=a, mem=A)
            deeper(src=a, mem=B)

    refusal = refused("wiring-type", compose)
    assert "Stream[i32, 8]" in refusal.violations[0].found


def test_a_parameter_that_is_not_wired_is_refused():
    def compose():
        @df.region()
        def short(A: int32[N], B: int32[N]):
            a: Stream[int32, 4]
            produce(dst=a, mem=A)
            consume(src=a)

    refused("wiring-arity", compose)


def test_a_name_the_unit_does_not_have_is_refused():
    def compose():
        @df.region()
        def stray(A: int32[N], B: int32[N]):
            a: Stream[int32, 4]
            produce(dst=a, mem=A)
            consume(src=a, mem=B, elsewhere=a)

    refused("wiring-arity", compose)


def test_two_readers_on_one_stream_are_refused():
    def compose():
        @df.region()
        def forked(A: int32[N], B: int32[N], C: int32[N]):
            a: Stream[int32, 4]
            produce(dst=a, mem=A)
            consume(src=a, mem=B)
            consume(src=a, mem=C)

    refused("single-producer-single-consumer", compose)


def test_a_stream_nothing_writes_is_refused():
    def compose():
        @df.region()
        def spare(A: int32[N], B: int32[N]):
            a: Stream[int32, 4]
            unused: Stream[int32, 4]
            produce(dst=a, mem=A)
            consume(src=a, mem=B)

    refused("unconnected-stream", compose)


@df.unit()
def ask(fwd: Stream[int32, 0], back: Stream[int32, 0], mem: int32[N]):
    for i in range(N):
        fwd.put(mem[i])
        back.get()


@df.unit()
def answer(fwd: Stream[int32, 0], back: Stream[int32, 0], mem: int32[N]):
    for i in range(N):
        mem[i] = fwd.get()
        back.put(0)


def test_a_cycle_with_no_capacity_anywhere_is_refused():
    def compose():
        @df.region()
        def tight(A: int32[N], B: int32[N]):
            f: Stream[int32, 0]
            b: Stream[int32, 0]
            ask(fwd=f, back=b, mem=A)
            answer(fwd=f, back=b, mem=B)

    refused("zero-capacity-cycle", compose)


@df.unit()
def ask_buffered(fwd: Stream[int32, 2], back: Stream[int32, 2], mem: int32[N]):
    for i in range(N):
        fwd.put(mem[i])
        back.get()


@df.unit()
def answer_buffered(fwd: Stream[int32, 2], back: Stream[int32, 2], mem: int32[N]):
    for i in range(N):
        mem[i] = fwd.get()
        back.put(0)


def test_a_cycle_with_capacity_is_accepted_and_leaves_an_obligation():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        @df.region()
        def loop(A: int32[N], B: int32[N]):
            f: Stream[int32, 2]
            b: Stream[int32, 2]
            ask_buffered(fwd=f, back=b, mem=A)
            answer_buffered(fwd=f, back=b, mem=B)

    premises = [w for w in caught if issubclass(w.category, UndeclaredPremise)]
    assert premises, [str(w.message) for w in caught]
    obligation = df.netlist_of("loop").obligation
    assert obligation is not None and obligation.premise is None
    assert obligation.cycles


def test_a_declared_premise_silences_the_warning_and_is_recorded():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        @df.region(deadlock_free_because="one token in flight, one slot each way")
        def declared(A: int32[N], B: int32[N]):
            f: Stream[int32, 2]
            b: Stream[int32, 2]
            ask_buffered(fwd=f, back=b, mem=A)
            answer_buffered(fwd=f, back=b, mem=B)

    assert not [w for w in caught if issubclass(w.category, UndeclaredPremise)]
    obligation = df.netlist_of("declared").obligation
    assert obligation.premise == "one token in flight, one slot each way"


def test_an_acyclic_netlist_leaves_no_obligation_at_all():
    @df.region()
    def straight(A: int32[N], B: int32[N]):
        a: Stream[int32, 4]
        produce(dst=a, mem=A)
        consume(src=a, mem=B)

    assert df.netlist_of("straight").obligation is None


def test_a_region_with_no_units_is_not_touched():
    @df.region()
    def lexical(A: int32[N], B: int32[N]):
        pipe: Stream[int32, 4]

        @df.kernel(mapping=[1], args=[A])
        def writer(mem: int32[N]):
            for i in range(N):
                pipe.put(mem[i])

        @df.kernel(mapping=[1], args=[B])
        def reader(mem: int32[N]):
            for i in range(N):
                mem[i] = pipe.get()

    assert df.netlist_of("lexical") is None
    module = df.build(lexical, target="simulator")
    source = np.arange(N, dtype=np.int32)
    result = np.zeros(N, dtype=np.int32)
    module(source, result)
    np.testing.assert_array_equal(result, source)


def test_a_unit_and_a_nested_kernel_share_one_region():
    @df.region()
    def mixed(A: int32[N], B: int32[N]):
        a: Stream[int32, 4]
        produce(dst=a, mem=A)

        @df.kernel(mapping=[1], args=[B])
        def reader(mem: int32[N]):
            for i in range(N):
                mem[i] = a.get()

    module = df.build(mixed, target="simulator")
    source = np.arange(N, dtype=np.int32)
    result = np.zeros(N, dtype=np.int32)
    module(source, result)
    np.testing.assert_array_equal(result, source)


def test_a_nested_kernel_that_also_reads_the_stream_is_refused():
    def compose():
        @df.region()
        def contended(A: int32[N], B: int32[N], C: int32[N]):
            a: Stream[int32, 4]
            produce(dst=a, mem=A)
            consume(src=a, mem=B)

            @df.kernel(mapping=[1], args=[C])
            def also(mem: int32[N]):
                for i in range(N):
                    mem[i] = a.get()

    refused("single-producer-single-consumer", compose)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
