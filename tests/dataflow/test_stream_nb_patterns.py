# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Non-Blocking Stream Pattern Tests
==================================
Focused tests for patterns that are *only possible* with non-blocking streams.
These go beyond simple try_put/try_get smoke tests to demonstrate control-flow
patterns that blocking FIFOs fundamentally cannot express.

Pattern catalogue
-----------------
1. Priority interrupt — a long-running data kernel checks a control stream
   each iteration; an interrupt signal preempts normal processing.
2. Multi-stream select — one consumer polls empty() on N streams, services
   whichever is ready first (round-robin arbiter).
3. Graceful drain — consumer uses empty() to drain a variable-length stream
   without knowing the count up front; a sentinel marks end-of-stream.
4. Timeout / bounded retry — try_get with a finite retry budget; if the
   producer is slow, the consumer gives up and takes a fallback path.

Why blocking streams cannot do this
------------------------------------
Blocking .get() suspends the caller until data arrives. This means:
  - You cannot poll a side-channel (interrupt) while waiting on data.
  - You cannot check multiple streams and pick the ready one.
  - You cannot bound the wait time or fall back on timeout.
  - You cannot probe whether data remains without committing to a read.

Non-blocking primitives (try_get, try_put, empty, full) decouple the
*query* from the *commit*, enabling all of the above.
"""
from __future__ import annotations
import pytest
import allo
from allo.ir.types import int32, int1, Stream
import allo.dataflow as df
import numpy as np


# ---------------------------------------------------------------------------
# Pattern 1: Priority Interrupt
# ---------------------------------------------------------------------------
# A data producer sends values 0..N-1. An interrupt producer sends a kill
# signal after K values. The consumer processes data until the interrupt
# arrives, then stops early and writes -1 as a sentinel.
#
# With blocking streams this is impossible: if the consumer blocks on
# data_stream.get(), it can never check the interrupt stream. With NB
# streams the consumer polls both each iteration.
# ---------------------------------------------------------------------------

def test_priority_interrupt_sim():
    """Consumer processes data until an interrupt arrives on a side-channel.

    Design: The data producer sends items one at a time, and after each put
    it signals an "ack" stream. The interrupt producer waits for K acks
    (meaning K data items have been sent), then fires the interrupt.
    This makes the interrupt timing deterministic regardless of simulator
    thread scheduling.

    The consumer polls both data and interrupt streams each iteration.
    With blocking streams this would be impossible — blocking on data.get()
    would prevent checking the interrupt channel.
    """
    N = 8   # total data items producer will send
    K = 4   # interrupt fires after K items

    @df.region()
    def top_interrupt(out: int32[N]):
        data_s: Stream[int32, 4][1]
        ack_s: Stream[int32, 4][1]    # data_producer -> interrupt_producer
        intr_s: Stream[int32, 1][1]   # interrupt: any value = "stop"

        @df.kernel(mapping=[1])
        def data_producer():
            for i in range(N):
                while not data_s[0].try_put(i + 1):
                    pass
                # Signal that item i was sent
                while not ack_s[0].try_put(1):
                    pass

        @df.kernel(mapping=[1])
        def interrupt_producer():
            # Wait for K acks (K data items sent), then fire interrupt
            ack_count: int32 = 0
            for _cycle in range(N * 4):
                if ack_count < K:
                    a_val: int32 = 0
                    a_ok: int1 = 0
                    a_val, a_ok = ack_s[0].try_get()
                    if a_ok:
                        ack_count += 1
            while not intr_s[0].try_put(1):
                pass

        @df.kernel(mapping=[1], args=[out])
        def consumer(out_buf: int32[N]):
            for i in range(N):
                out_buf[i] = 0
            idx: int32 = 0
            done: int1 = 0
            for _cycle in range(N * 4):
                if done == 0:
                    # Poll interrupt channel first (priority)
                    intr_val: int32 = 0
                    intr_ok: int1 = 0
                    intr_val, intr_ok = intr_s[0].try_get()
                    if intr_ok:
                        done = 1
                    # Poll data channel only if not interrupted
                    if done == 0:
                        d_val: int32 = 0
                        d_ok: int1 = 0
                        d_val, d_ok = data_s[0].try_get()
                        if d_ok:
                            out_buf[idx] = d_val
                            idx += 1

    sim = df.build(top_interrupt, target="simulator")
    np_out = np.zeros(N, dtype=np.int32)
    sim(np_out)

    # Consumer should have received data items, then stopped after interrupt.
    # With ack-based synchronization, the interrupt fires after K items are
    # sent. The consumer may have received up to all K items plus any that
    # were already in the FIFO when the interrupt arrived.
    received = int(np.count_nonzero(np_out))
    assert 1 <= received < N, (
        f"Expected fewer than {N} items (interrupt should preempt), got {received}: {np_out}"
    )
    # Verify received items are the correct prefix
    for i in range(received):
        assert np_out[i] == i + 1, f"out[{i}] = {np_out[i]}, expected {i + 1}"
    print(f"test_priority_interrupt_sim PASSED — received {received}/{N} before interrupt")


# ---------------------------------------------------------------------------
# Pattern 2: Multi-Stream Select (Round-Robin Arbiter)
# ---------------------------------------------------------------------------
# Two producers feed separate streams at different rates. One consumer
# polls empty() on both and services whichever is ready, round-robin.
#
# This is a fundamental NoC/interconnect pattern: an arbiter must not
# block on one port while another has data waiting.
# ---------------------------------------------------------------------------

def test_multi_stream_select_sim():
    """Round-robin arbiter drains two streams into one output.

    A single producer fills both streams, then the arbiter drains them.
    A "ready" stream synchronizes: producer signals when both streams are
    loaded, arbiter waits for the signal before draining. This ensures
    the polling pattern exercises both streams reliably.
    """
    N = 4   # items per stream
    TOTAL = 2 * N

    @df.region()
    def top_select(out: int32[TOTAL]):
        s0: Stream[int32, 8][1]
        s1: Stream[int32, 8][1]
        ready: Stream[int32, 1][1]

        @df.kernel(mapping=[1])
        def producer_both():
            # Fill both streams
            for i in range(N):
                while not s0[0].try_put(100 + i):
                    pass
            for i in range(N):
                while not s1[0].try_put(200 + i):
                    pass
            # Signal arbiter that data is ready
            while not ready[0].try_put(1):
                pass

        @df.kernel(mapping=[1], args=[out])
        def arbiter(out_buf: int32[TOTAL]):
            for i in range(TOTAL):
                out_buf[i] = 0
            # Wait for producer to finish loading both streams
            rdy_val: int32 = 0
            rdy_ok: int1 = 0
            while rdy_ok == 0:
                rdy_val, rdy_ok = ready[0].try_get()
            # Now drain both streams by polling
            count: int32 = 0
            for _cycle in range(TOTAL * 4):
                if count < TOTAL:
                    v0: int32 = 0
                    ok0: int1 = 0
                    v0, ok0 = s0[0].try_get()
                    if ok0:
                        out_buf[count] = v0
                        count += 1
                if count < TOTAL:
                    v1: int32 = 0
                    ok1: int1 = 0
                    v1, ok1 = s1[0].try_get()
                    if ok1:
                        out_buf[count] = v1
                        count += 1

    sim = df.build(top_select, target="simulator")
    np_out = np.zeros(TOTAL, dtype=np.int32)
    sim(np_out)

    received = set(np_out.tolist())
    expected_s0 = {100 + i for i in range(N)}
    expected_s1 = {200 + i for i in range(N)}
    assert expected_s0.issubset(received), f"Missing s0 values: {expected_s0 - received}"
    assert expected_s1.issubset(received), f"Missing s1 values: {expected_s1 - received}"
    print(f"test_multi_stream_select_sim PASSED — received {sorted(np_out.tolist())}")


# ---------------------------------------------------------------------------
# Pattern 3: Graceful Drain with End-of-Stream Sentinel
# ---------------------------------------------------------------------------
# Producer sends a variable number of items followed by a sentinel (0).
# Consumer uses try_get in a loop, accumulating values until it sees the
# sentinel. The consumer does not know the count in advance.
#
# With blocking get(), the consumer would need to know the exact count
# or risk deadlocking on an extra get() after the last real item.
# ---------------------------------------------------------------------------

def test_graceful_drain_sim():
    """Consumer drains a stream of unknown length using try_get + sentinel."""
    DATA = [10, 20, 30, 40, 50]
    N = len(DATA)
    SENTINEL = 0
    MAX_OUT = 8  # output buffer size (larger than data)

    @df.region()
    def top_drain(out: int32[MAX_OUT]):
        s: Stream[int32, 8][1]

        @df.kernel(mapping=[1])
        def producer():
            for i in range(N):
                # Send data values (all non-zero)
                val: int32 = (i + 1) * 10
                while not s[0].try_put(val):
                    pass
            # Send sentinel
            while not s[0].try_put(SENTINEL):
                pass

        @df.kernel(mapping=[1], args=[out])
        def consumer(out_buf: int32[MAX_OUT]):
            for i in range(MAX_OUT):
                out_buf[i] = -1  # mark unused slots
            idx: int32 = 0
            done: int1 = 0
            for _cycle in range(MAX_OUT * 4):
                if done == 0:
                    v: int32 = 0
                    ok: int1 = 0
                    v, ok = s[0].try_get()
                    if ok:
                        if v == SENTINEL:
                            done = 1
                        else:
                            out_buf[idx] = v
                            idx += 1

    sim = df.build(top_drain, target="simulator")
    np_out = np.full(MAX_OUT, -1, dtype=np.int32)
    sim(np_out)

    # First N slots should contain DATA, rest should be -1 (unused)
    for i in range(N):
        assert np_out[i] == DATA[i], f"out[{i}] = {np_out[i]}, expected {DATA[i]}"
    for i in range(N, MAX_OUT):
        assert np_out[i] == -1, f"out[{i}] = {np_out[i]}, expected -1 (unused)"
    print(f"test_graceful_drain_sim PASSED — drained {N} items + sentinel")


# ---------------------------------------------------------------------------
# Pattern 4: Bounded Retry / Timeout
# ---------------------------------------------------------------------------
# Consumer tries to receive from a stream with a finite retry budget.
# If data does not arrive within the budget, it writes a fallback value.
#
# This models hardware timeout logic: a PE waits a bounded number of
# cycles for a response, then takes an alternative action (e.g., use a
# cached value, signal an error, skip the tile).
# ---------------------------------------------------------------------------

def test_bounded_retry_sim():
    """Consumer gives up after N retries and uses a fallback value."""
    TIMEOUT = 4
    FALLBACK = -999

    @df.region()
    def top_timeout(out: int32[2]):
        fast_s: Stream[int32, 2][1]   # data arrives immediately
        slow_s: Stream[int32, 2][1]   # no data ever sent (simulates timeout)

        @df.kernel(mapping=[1])
        def fast_producer():
            while not fast_s[0].try_put(42):
                pass

        # No slow_producer — slow_s stays empty to trigger timeout

        @df.kernel(mapping=[1])
        def slow_placeholder():
            # Kernel exists so slow_s is properly constructed in the region,
            # but never puts anything.
            pass

        @df.kernel(mapping=[1], args=[out])
        def consumer(out_buf: int32[2]):
            # Try fast stream — should succeed quickly
            val0: int32 = FALLBACK
            for _retry in range(TIMEOUT):
                v: int32 = 0
                ok: int1 = 0
                v, ok = fast_s[0].try_get()
                if ok:
                    val0 = v
            out_buf[0] = val0

            # Try slow stream — should timeout
            val1: int32 = FALLBACK
            for _retry2 in range(TIMEOUT):
                v2: int32 = 0
                ok2: int1 = 0
                v2, ok2 = slow_s[0].try_get()
                if ok2:
                    val1 = v2
            out_buf[1] = val1

    sim = df.build(top_timeout, target="simulator")
    np_out = np.zeros(2, dtype=np.int32)
    sim(np_out)

    assert np_out[0] == 42, f"Fast stream should succeed, got {np_out[0]}"
    assert np_out[1] == FALLBACK, f"Slow stream should timeout to fallback, got {np_out[1]}"
    print(f"test_bounded_retry_sim PASSED — fast={np_out[0]}, slow(timeout)={np_out[1]}")


# ---------------------------------------------------------------------------
# Pattern 5: HLS Codegen — verify all patterns lower correctly
# ---------------------------------------------------------------------------

def test_nb_patterns_hls_codegen():
    """Verify that NB patterns produce correct Vitis HLS API calls."""
    @df.region()
    def top_patterns_hls():
        data_s: Stream[int32, 4][1]
        ctrl_s: Stream[int32, 1][1]

        @df.kernel(mapping=[1])
        def sender():
            while not data_s[0].try_put(1):
                pass
            while not ctrl_s[0].try_put(1):
                pass

        @df.kernel(mapping=[1])
        def receiver():
            # Multi-stream polling pattern
            done: int1 = 0
            for _c in range(16):
                if done == 0:
                    # Check interrupt first
                    ci: int32 = 0
                    ci_ok: int1 = 0
                    ci, ci_ok = ctrl_s[0].try_get()
                    if ci_ok:
                        done = 1
                    # Check data
                    if done == 0:
                        e: int1 = data_s[0].empty()
                        if e == 0:
                            dv: int32 = 0
                            dv_ok: int1 = 0
                            dv, dv_ok = data_s[0].try_get()

    mod = allo.customize(top_patterns_hls)
    hls_mod = mod.build(target="vhls")
    code = hls_mod.hls_code

    assert ".read_nb(" in code, "Expected .read_nb() in HLS output"
    assert ".write_nb(" in code, "Expected .write_nb() in HLS output"
    assert ".empty()" in code, "Expected .empty() in HLS output"
    print("test_nb_patterns_hls_codegen PASSED")


if __name__ == "__main__":
    test_priority_interrupt_sim()
    test_multi_stream_select_sim()
    test_graceful_drain_sim()
    test_bounded_retry_sim()
    test_nb_patterns_hls_codegen()
    print("\nAll NB pattern tests PASSED!")
