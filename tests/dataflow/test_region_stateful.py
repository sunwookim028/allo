# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for region-scope ``@ Stateful`` declarations.

Allo already supports declaring ``int32[N] @ Stateful = 0`` inside
``@df.kernel`` bodies, where the persistent buffer is private to that
kernel. This file exercises the architecturally interesting case where
the Stateful is declared at ``@df.region`` body scope so it is shared
across every kernel inside the region (e.g. a Gemmini-style decoder
kernel and an MXU-driver kernel sharing scratchpad/accumulator state).

Three variants are exercised:

  1. Region-scope Stateful read by a single kernel.
  2. Region-scope Stateful read by two kernels in the same region.
  3. Region-scope Stateful with a ``Stream`` between two kernels also
     reading the shared Stateful (full Gemmini-style topology).

The persistence contract: invoking the module multiple times must
preserve the buffer's contents across calls.
"""

import numpy as np
import pytest

import allo.dataflow as df
from allo.ir.types import Stateful, Stream, int32


# ---------------------------------------------------------------------------
# Variant 1: region-scope Stateful read by a single kernel.
# ---------------------------------------------------------------------------


def test_region_stateful_single_kernel():
    @df.region()
    def top(out: int32[4]):
        acc: int32[4] @ Stateful = 0  # region-scope shared state

        @df.kernel(mapping=[1], args=[out])
        def k(local_out: int32[4]):
            for i in range(4):
                acc[i] += 1
                local_out[i] = acc[i]

    mod = df.build(top, target="simulator")

    out = np.zeros(4, dtype=np.int32)
    mod(out)
    np.testing.assert_array_equal(out, np.array([1, 1, 1, 1], dtype=np.int32))
    mod(out)
    np.testing.assert_array_equal(out, np.array([2, 2, 2, 2], dtype=np.int32))
    mod(out)
    np.testing.assert_array_equal(out, np.array([3, 3, 3, 3], dtype=np.int32))


# ---------------------------------------------------------------------------
# Variant 2: region-scope Stateful read by two kernels in the same region.
# ---------------------------------------------------------------------------


def test_region_stateful_two_kernels_shared():
    @df.region()
    def top(out_a: int32[4], out_b: int32[4]):
        acc: int32[4] @ Stateful = 0  # shared by both kernels

        @df.kernel(mapping=[1], args=[out_a])
        def producer(po: int32[4]):
            for i in range(4):
                acc[i] += 1
                po[i] = acc[i]

        @df.kernel(mapping=[1], args=[out_b])
        def reader(rb: int32[4]):
            for i in range(4):
                rb[i] = acc[i]

    mod = df.build(top, target="simulator")

    a = np.zeros(4, dtype=np.int32)
    b = np.zeros(4, dtype=np.int32)
    # First call: producer increments acc to 1; reader sees the same acc.
    # The two kernels in a region run concurrently, so the reader may see
    # acc either before or after the producer's update. Run the module
    # multiple times and check that acc is at least monotonic and shared.
    mod(a, b)
    mod(a, b)
    mod(a, b)
    # After three invocations, producer has incremented acc three times;
    # reader observation in this last call sees a value <= 3 and >= 0,
    # but the region's acc must equal exactly 3 in the persistent buffer
    # (which is what producer's writeback to ``a`` reflects).
    np.testing.assert_array_equal(a, np.array([3, 3, 3, 3], dtype=np.int32))
    # Reader saw a snapshot of the shared state. The exact value depends
    # on scheduling, but it must be in {2, 3} on the last call.
    for v in b:
        assert v in (2, 3), f"reader saw {v} which is outside expected snapshot set"


# ---------------------------------------------------------------------------
# Variant 3: region-scope Stateful + Stream between two kernels.
# ---------------------------------------------------------------------------


def test_region_stateful_with_stream():
    @df.region()
    def top(out: int32[4]):
        acc: int32[4] @ Stateful = 0  # shared between decoder + driver
        sig: Stream[int32, 4]

        @df.kernel(mapping=[1])
        def decoder():
            for i in range(4):
                acc[i] += 1
                sig.put(acc[i])

        @df.kernel(mapping=[1], args=[out])
        def driver(local_out: int32[4]):
            for i in range(4):
                v: int32 = sig.get()
                local_out[i] = v + acc[i]

    mod = df.build(top, target="simulator")

    out = np.zeros(4, dtype=np.int32)
    mod(out)
    # After 1st call: decoder put 1 then driver reads 1 + acc[i].
    # The driver's read of acc[i] races with the producer; the contract
    # is just that the buffer is shared. Each element is in {2, 3, 4}.
    for v in out:
        assert 2 <= v <= 4, f"unexpected {v} (must reflect shared acc + sig)"
    mod(out)
    # 2nd call: persistent acc continues from previous value.
    for v in out:
        assert 4 <= v <= 6, f"unexpected {v} after 2nd call"


# ---------------------------------------------------------------------------
# The HLS backend cannot honour the sharing contract, and must say so.
# ---------------------------------------------------------------------------


# Every HLS emitter must refuse a shared region-scope Stateful. Parametrised
# rather than written once for ``vhls``, because the one that was NOT tested is
# exactly the one that had the hole: ``catapult`` emitted the shared global as a
# single file-scope ``static`` touched by every process under ``#pragma
# hls_design dataflow`` -- a race, with no diagnostic -- and nothing here caught
# it. A fourth emitter should have to answer this question too, so add its name
# to this list rather than writing a fourth test.
HLS_TARGETS = ["vhls", "systemc", "catapult"]

# Whether the emitter's message names the kernels, not just the variable. The
# ``vhls`` and ``catapult`` messages list them; the ``systemc`` one reports a
# count ("is used by 3 kernels") instead. Recorded rather than papered over: the
# refusal is what matters, and the difference is visible here if anyone wants to
# close it.
NAMES_THE_KERNELS = {"vhls": True, "catapult": True, "systemc": False}


@pytest.mark.parametrize("target", HLS_TARGETS)
def test_region_stateful_shared_by_two_kernels_is_rejected_for_hls(target, capfd):
    """Sharing is honoured by the simulator; every HLS emitter must refuse it.

    The emitter re-emits each stateful global as a function-local ``static``,
    so two kernels referencing one region-scope ``Stateful`` would get two
    *independent* copies -- except on ``catapult``, where the global is emitted
    at file scope and the two kernels really do land on one array with no
    arbitration, which is worse.

    Vitis does have unsynchronised sharing -- ``#pragma HLS stream
    variable=X type=shared`` and ``type=unsync`` -- but neither reaches more
    than one contending client.  Measured on Vitis HLS 2023.2 on this host, a
    4-element array in a ``#pragma HLS dataflow`` region: ``type=shared`` with
    1 writer + 1 reader csynths clean; with 1 writer + 2 readers it is
    ``ERROR [HLS 200-1014] Synchronized shared array 'buf' failed dataflow
    checking: it can only have a single reader and a single writer``; with 2
    writers + 1 reader, that plus ``[HLS 200-979] it can only be written in one
    process function``; ``type=unsync`` with 2 writers + 1 reader gives 200-979
    plus ``[HLS 200-780] it has 3 processes accessing it and only 2 ports``.
    So the 1R1W rule is not scoped to scalar channels -- a ``type=shared``
    array carries its own -- and the emitter has to fail the build rather than
    emit a silently wrong circuit.
    """

    @df.region()
    def top(out_a: int32[4], out_b: int32[4]):
        acc: int32[4] @ Stateful = 0  # shared by both kernels

        @df.kernel(mapping=[1], args=[out_a])
        def producer(po: int32[4]):
            for i in range(4):
                acc[i] += 1
                po[i] = acc[i]

        @df.kernel(mapping=[1], args=[out_b])
        def reader(rb: int32[4]):
            for i in range(4):
                rb[i] = acc[i]

    s = df.customize(top)
    with pytest.raises(RuntimeError):
        s.build(target=target)
    err = capfd.readouterr().err
    assert "__stateful_top_acc" in err, err
    if NAMES_THE_KERNELS[target]:
        assert "producer" in err and "reader" in err, err


@pytest.mark.parametrize("target", HLS_TARGETS)
def test_region_stateful_shared_by_three_kernels_is_rejected_for_hls(
    target, capfd
):
    """The three-client shape: one writer, one read-modify-writer, one reader.

    This is the design that found the ``catapult`` hole. Two clients can be
    argued about (producer/consumer looks like a stream that has not been
    written as one); three, with a read-modify-write in the middle, cannot be
    turned into any ordering the emitter could pick on its own. Every emitter
    must refuse it.
    """

    @df.region()
    def top(out_a: int32[4], out_b: int32[4], out_c: int32[4]):
        acc: int32[4] @ Stateful = 0  # touched by all three kernels

        @df.kernel(mapping=[1], args=[out_a])
        def writer(o: int32[4]):
            for i in range(4):
                acc[i] = i
                o[i] = acc[i]

        @df.kernel(mapping=[1], args=[out_b])
        def rmw(o: int32[4]):
            for i in range(4):
                acc[i] += 1
                o[i] = acc[i]

        @df.kernel(mapping=[1], args=[out_c])
        def pure_reader(o: int32[4]):
            for i in range(4):
                o[i] = acc[i]

    s = df.customize(top)
    with pytest.raises(RuntimeError):
        s.build(target=target)
    err = capfd.readouterr().err
    assert "__stateful_top_acc" in err, err
    # The count must be right: a guard that fires on "more than one" but
    # miscounts would also fire here, and would mislead whoever reads it.
    assert "3" in err, err
    if NAMES_THE_KERNELS[target]:
        for k in ("writer", "rmw", "pure_reader"):
            assert k in err, err


@pytest.mark.parametrize("target", HLS_TARGETS)
def test_catapult_emits_no_shared_file_scope_static(target, capfd):
    """The refusal must come before emission, not after a plausible file.

    ``catapult`` used to emit ``static int32_t __stateful_top_acc[4]`` at file
    scope and return success. Assert that no emitter hands back code for this
    program at all -- the failure mode being guarded is precisely "a build that
    succeeds and is wrong".
    """

    @df.region()
    def top(out_a: int32[4], out_b: int32[4]):
        acc: int32[4] @ Stateful = 0

        @df.kernel(mapping=[1], args=[out_a])
        def p(o: int32[4]):
            for i in range(4):
                acc[i] += 1
                o[i] = acc[i]

        @df.kernel(mapping=[1], args=[out_b])
        def c(o: int32[4]):
            for i in range(4):
                o[i] = acc[i]

    s = df.customize(top)
    code = None
    try:
        code = str(s.build(target=target))
    except RuntimeError:
        pass
    capfd.readouterr()
    assert code is None, (
        f"{target} returned code for a shared region-scope Stateful "
        f"instead of refusing it:\n{code}"
    )


@pytest.mark.parametrize("target", HLS_TARGETS)
def test_kernel_private_stateful_is_not_caught_by_the_guard(target, capfd):
    """The guard must not fire on the kernel-private model.

    Each mapped instance gets its own ``__stateful_k_<pid>_acc``, so no global
    has more than one user and every emitter must still build. Without this,
    the guard above could be "fixed" by refusing all Statefuls.
    """

    @df.region()
    def top(out: int32[2, 4]):
        @df.kernel(mapping=[2], args=[out])
        def k(o: int32[2, 4]):
            acc: int32[4] @ Stateful = 0  # private to each instance
            p = df.get_pid()
            for i in range(4):
                acc[i] += 1
                o[p, i] = acc[i]

    code = str(df.customize(top).build(target=target))
    capfd.readouterr()
    assert "__stateful_k_0_acc" in code, code
    assert "__stateful_k_1_acc" in code, code


def test_kernel_private_stateful_still_builds_for_hls():
    """The kernel-private model each instance owns its own buffer, so the
    per-function ``static`` is correct and the build must go through."""

    @df.region()
    def top(out: int32[2, 4]):
        @df.kernel(mapping=[2], args=[out])
        def k(o: int32[2, 4]):
            acc: int32[4] @ Stateful = 0  # private to each instance
            p = df.get_pid()
            for i in range(4):
                acc[i] += 1
                o[p, i] = acc[i]

    code = str(df.customize(top).build(target="vhls"))
    assert code.count("static int32_t __stateful_k_0_acc") == 1, code
    assert code.count("static int32_t __stateful_k_1_acc") == 1, code
