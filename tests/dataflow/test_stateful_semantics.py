# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``@ Stateful`` semantics (README D-11).

1. A ``Stateful`` keeps its value from one call of the built module to the
   next, on the simulator and in SystemC csim (one process per call: the
   testbench saves the state at exit and resumes it on the next call), and
   ``mod.reset()`` returns it to the declared initial value.
2. One ``Stateful`` has one kernel: two kernels touching the same one are
   refused at build, before any backend, with a message that names the
   Stateful and the kernels. A mapped kernel's PEs count as kernels.
3. What must still build: a single kernel owning a region-scope Stateful
   beside ``@df.unit`` instances that own Statefuls of their own (one global
   per instance), and a sharing region under an explicit ``shared_stateful``
   premise (the simulator honours it; HLS backends still refuse).

Record: ``dev/records/limitations/stateful_semantics_2026-10-02.rst``.
"""

import os

import numpy as np
import pytest

import allo.dataflow as df
from allo.ir.types import float32, int32, Stateful, Stream

N = 4
A_IN = np.array([1, 2, 3, 4], dtype=np.int32)


def _counter_region():
    """One kernel: a region-scope array, a kernel-local scalar, a float array."""

    @df.region()
    def top(A: int32[N], B: int32[N], C: float32[N]):
        total: int32[N] @ Stateful = 0  # region-scope, owned by one kernel
        ftotal: float32[N] @ Stateful = 0.0  # the float path (raw-bits file format)

        @df.kernel(mapping=[1], args=[A, B, C])
        def acc(a: int32[N], b: int32[N], c: float32[N]):
            calls: int32 @ Stateful = 0  # kernel-local scalar
            calls = calls + 1
            for i in range(N):
                total[i] = total[i] + a[i]
                ftotal[i] = ftotal[i] + 0.5
                b[i] = total[i] * 10 + calls
                c[i] = ftotal[i]

    return top


def _expect(k):
    """Outputs after the k-th call since the last reset, with A_IN every time."""
    b = (A_IN * k * 10 + k).astype(np.int32)
    c = np.full(N, 0.5 * k, dtype=np.float32)
    return b, c


def _run_persistence(mod):
    B = np.zeros(N, dtype=np.int32)
    C = np.zeros(N, dtype=np.float32)
    for k in (1, 2, 3):
        mod(A_IN, B, C)
        b, c = _expect(k)
        np.testing.assert_array_equal(B, b, err_msg=f"call {k}")
        np.testing.assert_array_equal(C, c, err_msg=f"call {k} (float)")
    mod.reset()
    mod(A_IN, B, C)
    b, c = _expect(1)
    np.testing.assert_array_equal(B, b, err_msg="first call after reset")
    np.testing.assert_array_equal(C, c, err_msg="first call after reset (float)")
    mod(A_IN, B, C)
    np.testing.assert_array_equal(B, _expect(2)[0], err_msg="resumes after reset")


def test_stateful_persists_across_calls_simulator():
    mod = df.build(_counter_region(), target="simulator")
    _run_persistence(mod)


@pytest.mark.skipif(
    not (os.environ.get("MGC_HOME") and os.environ.get("SYSTEMC_HOME")),
    reason="SystemC csim needs Catapult's g++ (MGC_HOME) and SYSTEMC_HOME",
)
def test_stateful_persists_across_calls_systemc_csim(tmp_path):
    prj = str(tmp_path / "stateful_csim.prj")
    mod = df.build(_counter_region(), target="systemc", mode="csim", project=prj)
    B = np.zeros(N, dtype=np.int32)
    C = np.zeros(N, dtype=np.float32)
    mod(A_IN, B, C)
    # The testbench saved the one stateful instance's members at exit ...
    state = os.path.join(prj, "allo_state_u0.data")
    assert os.path.exists(state), os.listdir(prj)
    # ... as one value per element: total[4], ftotal[4] (raw bits), calls.
    assert len(open(state, encoding="utf-8").read().split()) == 2 * N + 1
    mod.reset()
    assert not os.path.exists(state)
    _run_persistence(mod)


# ---------------------------------------------------------------------------
# One Stateful has one kernel.
# ---------------------------------------------------------------------------


def _shared_region(**region_kwargs):
    @df.region(**region_kwargs)
    def top(out_a: int32[N], out_b: int32[N]):
        acc: int32[N] @ Stateful = 0

        @df.kernel(mapping=[1], args=[out_a])
        def producer(po: int32[N]):
            for i in range(N):
                acc[i] += 1
                po[i] = acc[i]

        @df.kernel(mapping=[1], args=[out_b])
        def reader(rb: int32[N]):
            for i in range(N):
                rb[i] = acc[i]

    return top


def test_two_kernels_sharing_a_stateful_are_refused():
    with pytest.raises(RuntimeError) as e:
        df.build(_shared_region(), target="simulator")
    msg = str(e.value)
    assert "Stateful `acc`" in msg and "__stateful_top_acc" in msg, msg
    assert "2 kernels" in msg and "producer_0" in msg and "reader_0" in msg, msg
    assert "shared_stateful" in msg, msg
    # Refused before any backend: the same answer from df.customize.
    with pytest.raises(RuntimeError, match="Stateful `acc`"):
        df.customize(_shared_region())


def test_mapped_kernel_pes_count_as_kernels():
    @df.region()
    def top(out: int32[2, N]):
        rs: int32[N] @ Stateful = 0

        @df.kernel(mapping=[2], args=[out])
        def pe(o: int32[2, N]):
            p = df.get_pid()
            for i in range(N):
                rs[i] += 1
                o[p, i] = rs[i]

    with pytest.raises(RuntimeError) as e:
        df.customize(top)
    assert "pe_0" in str(e.value) and "pe_1" in str(e.value), str(e.value)


def test_shared_stateful_premise_is_honoured_by_the_simulator_only():
    top = _shared_region(
        shared_stateful={"acc": "test: the reader tolerates any interleaving"}
    )
    mod = df.build(top, target="simulator")
    a = np.zeros(N, dtype=np.int32)
    b = np.zeros(N, dtype=np.int32)
    for _ in range(3):
        mod(a, b)
    np.testing.assert_array_equal(a, np.full(N, 3, dtype=np.int32))
    for v in b:
        assert v in (2, 3), b
    # The HLS emitters refuse the sharing regardless of the premise.
    with pytest.raises(RuntimeError):
        df.customize(top).build(target="vhls")


def test_shared_stateful_premise_must_name_a_region_stateful():
    with pytest.raises(RuntimeError, match="not region-scope"):
        df.customize(_shared_region(shared_stateful={"typo": "nothing"}))


# ---------------------------------------------------------------------------
# What must still build.
# ---------------------------------------------------------------------------


@df.unit()
def running_sum(src: Stream[int32, 4], dst: Stream[int32, 4]):
    acc: int32 @ Stateful = 0  # one global per instance
    for i in range(N):
        acc = acc + src.get()
        dst.put(acc)


def test_single_kernel_and_unit_instances_build_and_persist():
    @df.region()
    def top(A: int32[N], B: int32[N]):
        a: Stream[int32, 4]
        b: Stream[int32, 4]
        c: Stream[int32, 4]
        first = running_sum(src=a, dst=b)
        second = running_sum(src=b, dst=c)
        hist: int32[N] @ Stateful = 0  # region-scope, one kernel

        @df.kernel(mapping=[1], args=[A])
        def feed(x: int32[N]):
            for i in range(N):
                hist[i] = hist[i] + x[i]
                a.put(hist[i])

        @df.kernel(mapping=[1], args=[B])
        def drain(y: int32[N]):
            for i in range(N):
                y[i] = c.get()

    s = df.customize(top)
    ir = str(s.module)
    for g in ("first_0_acc", "second_0_acc", "top_hist"):
        assert f"__stateful_{g}" in ir, ir
    mod = df.build(top, target="simulator")
    B = np.zeros(N, dtype=np.int32)
    hist = np.zeros(N, dtype=np.int64)
    acc1 = acc2 = 0
    for _ in range(2):
        mod(A_IN, B)
        hist += A_IN
        exp = []
        for v in hist:
            acc1 += int(v)
            acc2 += acc1
            exp.append(acc2)
        np.testing.assert_array_equal(B, np.array(exp, dtype=np.int32))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
