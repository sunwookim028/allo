# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""SystemC / Catapult (MatchLib Connections) backend — target="systemc".

Emits a dataflow @df.region as SystemC: each @df.kernel becomes an SC_MODULE
with Connections::In/Out ports + a clocked free-running thread, streams become
Connections::Combinational channels, and region boundary arrays are hoisted to
top-level array members (memory-port style, direction from arg_dirs).

test_systemc_emit runs anywhere (pure codegen). test_systemc_csim compiles +
simulates the emitted SystemC and is SKIPPED unless Catapult is available
(MGC_HOME / zhang-21 only).
"""

import os
import tempfile

import numpy as np
import pytest

import allo
from allo.ir.types import int8, int16, int32, int1, UInt, Stream
import allo.dataflow as df


def _producer_consumer():
    N = 8

    @df.region()
    def top(A: int32[N], B: int32[N]):
        fifo: Stream[int32, 4][1]

        @df.kernel(mapping=[1], args=[A])
        def producer(a: int32[N]):
            for i in range(N):
                fifo[0].put(a[i])

        @df.kernel(mapping=[1], args=[B])
        def consumer(b: int32[N]):
            for i in range(N):
                b[i] = fifo[0].get() + 1

    return top


def _systolic_chain(P=4, N=8):
    """array->stream feed + mapping=[P] PE chain (each +1) + stream->array drain."""

    @df.region()
    def top(A: int32[N], B: int32[N]):
        link: Stream[int32, 4][P + 1]

        @df.kernel(mapping=[1], args=[A])
        def feed(a: int32[N]):
            for k in range(N):
                link[0].put(a[k])

        @df.kernel(mapping=[P])
        def pe():
            i = df.get_pid()
            for k in range(N):
                v: int32 = link[i].get()
                w: int32 = v + 1
                link[i + 1].put(w)

        @df.kernel(mapping=[1], args=[B])
        def drain(b: int32[N]):
            for k in range(N):
                b[k] = link[P].get()

    return top, P, N


def test_systemc_golden():
    """The designs the SystemC backend emits are functionally correct, checked
    against the backend-agnostic Allo simulator. This makes the reference
    trustworthy for the csim / RTL-cosim checks (numpy -> simulator -> csim -> RTL)."""
    # producer/consumer: B = A + 1
    top = _producer_consumer()
    A = np.arange(8, dtype=np.int32)
    B = np.zeros(8, dtype=np.int32)
    df.build(top, target="simulator")(A, B)
    np.testing.assert_array_equal(B, A + 1)
    # systolic chain: B = A + P
    chain, P, N = _systolic_chain()
    A2 = np.arange(N, dtype=np.int32)
    B2 = np.zeros(N, dtype=np.int32)
    df.build(chain, target="simulator")(A2, B2)
    np.testing.assert_array_equal(B2, A2 + P)
    print("simulator golden OK: producer/consumer B=A+1, chain B=A+P")


def test_systemc_mem_port_reread():
    """A re-read (identity a[k] under an OUTER loop, each element touched twice)
    is not a single-pass scan, so the INPUT array a routes to a random-access
    memory port — a read may fire many times, which is correct."""

    N = 8

    @df.region()
    def top(A: int32[N], B: int32[N]):
        fifo: Stream[int32, 4][1]

        @df.kernel(mapping=[1], args=[A])
        def producer(a: int32[N]):
            for r in range(2):
                for k in range(N):
                    fifo[0].put(a[k])  # a[k] re-read across the r loop -> mem port

        @df.kernel(mapping=[1], args=[B])
        def consumer(b: int32[N]):
            for i in range(N):
                s: int32 = 0
                for r in range(2):
                    s = fifo[0].get()
                b[i] = s

    code = df.build(top, target="systemc").hls_code
    assert "AlloMemPins<" in code  # re-read input -> memory behind RAM pins
    assert "_re.write(true)" in code and "_q.read()" in code  # the read pins
    print("re-read input -> memory port")


def test_systemc_nonblocking():
    """try_get / try_put emit as Connections .PopNB() / .PushNB() (fire-on-valid).
    NOTE: a *working* fire-on-valid design must do one try per cycle (not spin) —
    spin-until-success deadlocks a free-running SC_THREAD. This checks emission."""

    N = 8

    @df.region()
    def top(A: int32[N], B: int32[N]):
        S: Stream[int32, 4][1]

        @df.kernel(mapping=[1], args=[A])
        def producer(a: int32[N]):
            for i in range(N):
                ok: int1 = 0
                while ok == 0:
                    ok = S[0].try_put(a[i])

        @df.kernel(mapping=[1], args=[B])
        def consumer(b: int32[N]):
            for i in range(N):
                v: int32 = 0
                ok: int1 = 0
                while ok == 0:
                    v, ok = S[0].try_get()
                b[i] = v

    code = df.build(top, target="systemc").hls_code
    assert ".PopNB(" in code and ".PushNB(" in code
    print("try_get/try_put -> PopNB/PushNB")


def _empty_full():
    """Producer checks full() (its Out port), consumer checks empty() (its In
    port). MatchLib ports DO provide these: Out<T>.Full() / In<T>.Empty(). The
    checks are read-only here so the design still streams B == A."""
    N = 8

    @df.region()
    def top(A: int32[N], B: int32[N]):
        S: Stream[int32, 4][1]

        @df.kernel(mapping=[1], args=[A])
        def producer(a: int32[N]):
            for i in range(N):
                f: int1 = S[0].full()  # -> Out.Full()
                S[0].put(a[i])

        @df.kernel(mapping=[1], args=[B])
        def consumer(b: int32[N]):
            for i in range(N):
                e: int1 = S[0].empty()  # -> In.Empty()
                b[i] = S[0].get()

    return top, N


def test_systemc_empty_full_emit():
    """empty()/full() are served by AlloFifoC's empty_o/full_o SIDEBAND PINS, which
    the kernels read as plain sc_in<bool>. NOT Connections In.Empty()/Out.Full():
    a regular In/Out cannot report Empty/Full in HLS (that needs InBuffered/
    OutBuffered, absent in Catapult 2024.2), so the FIFO exports them instead."""
    top, _ = _empty_full()
    code = df.build(top, target="systemc").hls_code
    assert "full_o" in code and "empty_o" in code  # AlloFifoC exports the sidebands
    assert "_full.read();" in code  # producer reads the full pin
    assert "_empty.read();" in code  # consumer reads the empty pin
    print("empty()/full() -> AlloFifoC empty_o/full_o sideband pins")


@pytest.mark.skipif(
    not os.environ.get("MGC_HOME"),
    reason="Catapult (MGC_HOME) not available — csim needs zhang-21",
)
def test_systemc_empty_full_csim():
    """A design using empty()/full() compiles + streams correctly: B == A."""
    top, N = _empty_full()
    with tempfile.TemporaryDirectory() as tmp:
        mod = df.build(top, target="systemc", mode="csim", project=tmp)
        A = np.arange(N, dtype=np.int32)
        B = np.zeros(N, dtype=np.int32)
        mod(A, B)
        np.testing.assert_array_equal(B, A)
    print("SystemC empty()/full() csim B == A")
    print("empty() correctly rejected")


def test_systemc_emit():
    """Pure codegen — no toolchain needed. Checks the Connections stream shape."""
    top = _producer_consumer()
    code = df.build(top, target="systemc").hls_code

    # Connections shell (synthesizable), not the old sc_fifo model
    assert "sc_fifo" not in code
    assert "Connections::In<" in code
    assert "Connections::Out<" in code
    assert "Connections::Combinational<" in code
    assert ".Pop()" in code and ".Push(" in code
    assert "while (1)" in code
    assert "SC_HAS_PROCESS" in code
    # sequential-stream: boundary arrays become Connections stream PORTS
    # (no array members, no pointers), and a[i]/b[i]=v become Pop()/Push()
    assert "[8]" not in code  # no int32_t vN[8]; array member survives
    assert "nullptr" not in code  # no kernel array pointer
    print("SystemC emit shape OK")


def _mem_port_reverse():
    """B[i] = A[N-1-i] + 1 : A read reversed -> random-access INPUT memory port;
    B is a normal sequential output stream."""
    N = 8

    @df.region()
    def top(A: int32[N], B: int32[N]):
        @df.kernel(mapping=[1], args=[A, B])
        def rev(a: int32[N], b: int32[N]):
            for i in range(N):
                b[i] = a[N - 1 - i] + 1

    return top, N


def test_systemc_mem_port_emit():
    """A non-sequential INPUT array is routed to a memory behind RAM PINS
    (radr/re/q, driven by a `modulario` _rd accessor), NOT silently mis-streamed.
    The flat address is the reversed index."""
    top, _ = _mem_port_reverse()
    code = df.build(top, target="systemc").hls_code
    assert "AlloMemPins<" in code  # random-access memory instantiated
    assert "_re.write(true)" in code and "_q.read()" in code  # the read pins
    assert "_radr" in code  # address pin wired at top
    assert ".mem[f] =" in code  # tb preloads the memory from input file
    print("random INPUT access -> memory port emitted")


def _mem_port_scatter():
    """B[N-1-i] = A[i] + 1 : B WRITTEN reversed -> random-access OUTPUT (write-
    only) memory port; A is a sequential input stream. No stream output,
    so completion is time-based.  B == (A+1)[::-1]."""
    N = 8

    @df.region()
    def top(A: int32[N], B: int32[N]):
        @df.kernel(mapping=[1], args=[A, B])
        def scat(a: int32[N], b: int32[N]):
            for i in range(N):
                b[N - 1 - i] = a[i] + 1

    return top, N


def test_systemc_mem_port_store_emit():
    """A non-sequential OUTPUT array routes to a write-only memory driven by the
    WRITE pins (wadr/d/we) via a `modulario` _wr accessor; the tb reads mem[] out
    after a time-based run (no stream output)."""
    top, _ = _mem_port_scatter()
    code = df.build(top, target="systemc").hls_code
    assert "AlloMemPins<" in code  # write-only memory
    # write pins driven, read pins never used for a store-only array
    assert "_we.write(true)" in code and "_re.write(true)" not in code
    assert "sc_start(" in code and "SC_NS);" in code  # time-based completion
    assert "_mem.mem[f];" in code  # tb reads the memory out (sum-merge) to file
    print("random OUTPUT access -> write memory port emitted")


def _both_mem_port():
    """A read+write random-access array ('both') -> one memory, both pin bundles,
    preloaded AND read back (in-place). Reads a[4..7], writes a[0..3] (disjoint,
    so re-execution under the free-running kernel is idempotent). Final:
    A[0:4] = A[4:8] + 1, A[4:8] unchanged."""
    N = 8

    @df.region()
    def top(A: int32[N]):
        @df.kernel(mapping=[1], args=[A])
        def rmw(a: int32[N]):
            for i in range(4):
                a[i] = a[i + 4] + 1

    return top, N


def test_systemc_both_mem_port_emit():
    """A read+write random-access array is ONE memory carrying BOTH pin bundles:
    the read pins (radr/re/q) and the write pins (wadr/d/we)."""
    top, _ = _both_mem_port()
    code = df.build(top, target="systemc").hls_code
    assert code.count("AlloMemPins<") == 1  # ONE memory, not one per direction
    assert "_re.write(true)" in code and "_we.write(true)" in code  # both bundles
    assert "_q.read()" in code  # the read data pin
    print("both (read+write) memory port -> one memory, read + write pins")


@pytest.mark.skipif(
    not os.environ.get("MGC_HOME"),
    reason="Catapult (MGC_HOME) not available — csim needs zhang-21",
)
def test_systemc_both_mem_port_csim():
    """In-place read-modify-write: A preloaded, modified, read back.
    A[0:4] == A[4:8]+1, A[4:8] unchanged."""
    top, N = _both_mem_port()
    with tempfile.TemporaryDirectory() as tmp:
        mod = df.build(top, target="systemc", mode="csim", project=tmp)
        A = np.arange(N, dtype=np.int32)
        exp = A.copy()
        exp[0:4] = A[4:8] + 1
        mod(A)
        np.testing.assert_array_equal(A, exp)
    print("SystemC 'both' mem-port csim: in-place read-modify-write")


@pytest.mark.skipif(
    not os.environ.get("MGC_HOME"),
    reason="Catapult (MGC_HOME) not available — csim needs zhang-21",
)
def test_systemc_mem_port_csim():
    """Compile + simulate the memory-port design: A preloaded into the memory, read
    reversed over the RAM pins, +1, streamed to B. Assert B == A[::-1]+1."""
    top, N = _mem_port_reverse()
    with tempfile.TemporaryDirectory() as tmp:
        mod = df.build(top, target="systemc", mode="csim", project=tmp)
        A = np.arange(N, dtype=np.int32)
        B = np.zeros(N, dtype=np.int32)
        mod(A, B)
        np.testing.assert_array_equal(B, A[::-1] + 1)
    print("SystemC memory-port csim B == A[::-1] + 1")


@pytest.mark.skipif(
    not os.environ.get("MGC_HOME"),
    reason="Catapult (MGC_HOME) not available — csim needs zhang-21",
)
def test_systemc_mem_port_store_csim():
    """Compile + simulate the store-side design: A streamed in, +1, scattered to B
    reversed via the write pins into the memory, read out after a time-based run. Assert
    B == (A+1)[::-1]."""
    top, N = _mem_port_scatter()
    with tempfile.TemporaryDirectory() as tmp:
        mod = df.build(top, target="systemc", mode="csim", project=tmp)
        A = np.arange(N, dtype=np.int32)
        B = np.zeros(N, dtype=np.int32)
        mod(A, B)
        np.testing.assert_array_equal(B, (A + 1)[::-1])
    print("SystemC store-side memory-port csim B == (A+1)[::-1]")


def test_systemc_grid():
    """mapping=[P] is unrolled into P kernel SC_MODULEs, wired as a neighbor chain."""

    P, N = 4, 8

    @df.region()
    def top(A: int32[N], B: int32[N]):
        link: Stream[int32, 4][P + 1]

        @df.kernel(mapping=[1], args=[A])
        def feed(a: int32[N]):
            for k in range(N):
                link[0].put(a[k])

        @df.kernel(mapping=[P])
        def pe():
            i = df.get_pid()
            for k in range(N):
                v: int32 = link[i].get()
                w: int32 = v + 1
                link[i + 1].put(w)

        @df.kernel(mapping=[1], args=[B])
        def drain(b: int32[N]):
            for k in range(N):
                b[k] = link[P].get()

    code = df.build(top, target="systemc").hls_code
    for p in range(P):  # each grid instance is its own module
        assert f"SC_MODULE(pe_{p})" in code
    # P+1 link channels in the top (+ tb channels), so at least P+1
    assert code.count("Connections::Combinational<") >= P + 1
    print(f"grid of {P} PEs emitted + wired")


@pytest.mark.skipif(
    not os.environ.get("MGC_HOME"),
    reason="Catapult (MGC_HOME) not available — csim needs zhang-21",
)
def test_systemc_csim():
    """Compile + simulate the emitted SystemC; results flow back into B (Option B:
    A -> input0.data -> design -> output0.data -> B), so we assert B == A + 1."""
    top = _producer_consumer()
    with tempfile.TemporaryDirectory() as tmp:
        mod = df.build(top, target="systemc", mode="csim", project=tmp)
        A = np.arange(8, dtype=np.int32)
        B = np.zeros(8, dtype=np.int32)
        mod(A, B)
        np.testing.assert_array_equal(B, A + 1)
    print("SystemC csim B == A + 1")


@pytest.mark.skipif(
    not os.environ.get("MGC_HOME"),
    reason="Catapult (MGC_HOME) not available — csim needs zhang-21",
)
def test_systemc_csim_grid():
    """End-to-end csim of a mapping=[P] grid with data-through-args: B == A + P."""
    chain, P, N = _systolic_chain()
    with tempfile.TemporaryDirectory() as tmp:
        mod = df.build(chain, target="systemc", mode="csim", project=tmp)
        A = np.arange(N, dtype=np.int32)
        B = np.zeros(N, dtype=np.int32)
        mod(A, B)
        np.testing.assert_array_equal(B, A + P)
    print(f"SystemC csim grid B == A + {P}")


def _depth_pc(depth):
    """producer -> [Stream depth] -> +1 -> consumer, with the given stream depth."""
    N = 8

    @df.region()
    def top(A: int32[N], B: int32[N]):
        s_in: Stream[int32, depth][1]
        s_out: Stream[int32, depth][1]

        @df.kernel(mapping=[1], args=[A])
        def source(a: int32[N]):
            for i in range(N):
                s_in[0].put(a[i])

        @df.kernel(mapping=[1])
        def compute():
            for i in range(N):
                s_out[0].put(s_in[0].get() + 1)

        @df.kernel(mapping=[1], args=[B])
        def sink(b: int32[N]):
            for i in range(N):
                b[i] = s_out[0].get()

    return top, N


def test_systemc_stream_depth_flavor():
    """Stream depth is honored: depth 0 -> bare Connections::Combinational (wire),
    depth >= 1 -> a schedulable Connections::Fifo<T,depth> between _in/_out wires.

    The buffered flavor is the VENDOR FIFO, not our own: since 9b1eac7 the emitter
    instantiates Connections::Fifo directly (AlloFifoC subclasses it only where the
    empty/full sidebands are needed), so the ports are enq/deq, not in/out."""
    code0 = df.build(_depth_pc(0)[0], target="systemc").hls_code
    assert "_fifo;" not in code0  # depth 0 -> no FIFO instance, a plain wire
    # Integer stream payloads emit as ac_int<W> (not native int32_t) so Connections'
    # marshaller can serialize them -- native int8_t lacks a Wrapped<> specialization.
    assert "Connections::Combinational< ac_int<32, true> > v" in code0

    code4 = df.build(_depth_pc(4)[0], target="systemc").hls_code
    # both streams buffered
    assert code4.count("Connections::Fifo< ac_int<32, true>, 4 >") == 2
    assert "_fifo.enq(" in code4 and "_fifo.deq(" in code4  # wired through
    print("stream depth honored: 0 -> Combinational, >=1 -> Connections::Fifo")


@pytest.mark.skipif(
    not os.environ.get("MGC_HOME"),
    reason="Catapult (MGC_HOME) not available — csim needs zhang-21",
)
def test_systemc_stream_depth0_csim():
    """A depth-0 (combinational-wire) stream still simulates correctly: B == A+1."""
    top, N = _depth_pc(0)
    with tempfile.TemporaryDirectory() as tmp:
        mod = df.build(top, target="systemc", mode="csim", project=tmp)
        A = np.arange(N, dtype=np.int32)
        B = np.zeros(N, dtype=np.int32)
        mod(A, B)
        np.testing.assert_array_equal(B, A + 1)
    print("SystemC depth-0 wire csim B == A + 1")


def _multi_client_grid():
    """Grid of N independent PEs, each reading shared A (reversed) and writing a
    DISTINCT element of shared C. A -> N replicated read memories (multi-client
    read); C -> N replicated write memories + element-wise sum-merge at readout
    (multi-client write, disjoint elements). No streams/feedback. C = A[::-1]+1."""
    N = 4

    @df.region()
    def top(A: int32[N], C: int32[N]):
        @df.kernel(mapping=[N], args=[A, C])
        def pe(a: int32[N], c: int32[N]):
            p = df.get_pid()
            c[p] = a[N - 1 - p] + 1

    return top, N


def test_systemc_multi_client_emit():
    """A boundary array shared by several grid replicas is REPLICATED per client
    (each gets its own memory + channels), so there is no multi-driver on a single
    channel. N read replicas for A, N write replicas for C."""
    top, N = _multi_client_grid()
    code = df.build(top, target="systemc").hls_code
    # one memory per (kernel, arg): N readers of A + N writers of C
    assert code.count("AlloMemPins<") == 2 * N
    assert code.count("_re.write(true)") == N  # A read replicas
    assert code.count("_we.write(true)") == N  # C write replicas
    print(f"shared arrays replicated into {N} read + {N} write memories")


@pytest.mark.skipif(
    not os.environ.get("MGC_HOME"),
    reason="Catapult (MGC_HOME) not available — csim needs zhang-21",
)
def test_systemc_multi_client_csim():
    """End-to-end csim of the multi-client grid: replicated read memories all
    preloaded from A, disjoint writes summed back into C. C == A[::-1] + 1."""
    top, N = _multi_client_grid()
    with tempfile.TemporaryDirectory() as tmp:
        mod = df.build(top, target="systemc", mode="csim", project=tmp)
        A = np.arange(N, dtype=np.int32)
        C = np.zeros(N, dtype=np.int32)
        mod(A, C)
        np.testing.assert_array_equal(C, A[::-1] + 1)
    print("SystemC multi-client csim C == A[::-1] + 1")


def _tiled_systolic_gemm():
    """Standard Allo tiled systolic GEMM (C = A @ B) at Mt=Nt=1: a mapping=[P0,P1]
    grid of feeder/body/drain PEs; 2-D boundaries A,B (reads) + C (write) each
    touched by exactly one PE -> single-client memory ports; int accumulation
    widens past 64 bits (ap_int<65>)."""
    M = N = K = 4
    Mt = Nt = 1
    P0, P1 = Mt + 2, Nt + 2

    @df.region()
    def top(A: int32[M, K], B: int32[K, N], C: int32[M, N]):
        fifo_A: Stream[int32, 4][P0, P1]
        fifo_B: Stream[int32, 4][P0, P1]

        @df.kernel(mapping=[P0, P1], args=[A, B, C])
        def gemm(local_A: int32[M, K], local_B: int32[K, N], local_C: int32[M, N]):
            i, j = df.get_pid()
            for m in range(M // Mt):
                for n in range(N // Nt):
                    with allo.meta_if(i in {0, Mt + 1} and j in {0, Nt + 1}):
                        pass
                    with allo.meta_elif(j == 0):
                        for k in range(K):
                            fifo_A[i, j + 1].put(local_A[m * Mt + i - 1, k])
                    with allo.meta_elif(i == 0):
                        for k in range(K):
                            fifo_B[i + 1, j].put(local_B[k, n * Nt + j - 1])
                    with allo.meta_elif(i == Mt + 1):
                        for k in range(K):
                            b: int32 = fifo_B[i, j].get()
                    with allo.meta_elif(j == Nt + 1):
                        for k in range(K):
                            a: int32 = fifo_A[i, j].get()
                    with allo.meta_else():
                        c: int32 = 0
                        for k in range(K):
                            a: int32 = fifo_A[i, j].get()
                            b: int32 = fifo_B[i, j].get()
                            c += a * b
                            fifo_A[i, j + 1].put(a)
                            fifo_B[i + 1, j].put(b)
                        local_C[m * Mt + i - 1, n * Nt + j - 1] = c

    return top, M, N, K


def test_systemc_tiled_systolic_emit():
    """The standard Allo tiled-systolic GEMM maps to a grid of PEs + memory ports:
    A,B -> read pins, C -> write pins. (Non-corner PEs only.)"""
    top, *_ = _tiled_systolic_gemm()
    code = df.build(top, target="systemc").hls_code
    assert code.count("AlloMemPins<") == 3  # A, B, C
    assert code.count("_re.write(true)") == 2  # A, B read ports
    assert code.count("_we.write(true)") == 1  # C write port
    assert "SC_MODULE(gemm_1_1)" in code  # the main-body PE survives DCE
    print("tiled systolic GEMM -> grid + 2 read / 1 write memory ports")


@pytest.mark.skipif(
    not os.environ.get("MGC_HOME"),
    reason="Catapult (MGC_HOME) not available — csim needs zhang-21",
)
def test_systemc_tiled_systolic_csim():
    """End-to-end csim of the tiled systolic GEMM: C == A @ B."""
    top, M, N, K = _tiled_systolic_gemm()
    with tempfile.TemporaryDirectory() as tmp:
        mod = df.build(top, target="systemc", mode="csim", project=tmp)
        A = np.random.randint(0, 10, (M, K)).astype(np.int32)
        B = np.random.randint(0, 10, (K, N)).astype(np.int32)
        C = np.zeros((M, N), dtype=np.int32)
        mod(A, B, C)
        np.testing.assert_array_equal(C, A @ B)
    print("SystemC tiled-systolic csim C == A @ B")


def _saxpy_helper(x: int32, y: int32) -> int32:
    """A pure compute helper (no streams/kernels) called from a kernel body."""
    return x * 2 + y


def _kernel_with_helper():
    """A @df.kernel whose body calls a typed compute helper -> the helper must be
    emitted as a plain C++ free function (return-by-pointer) before the modules."""
    N = 8

    @df.region()
    def top(A: int32[N], B: int32[N]):
        @df.kernel(mapping=[1], args=[A, B])
        def k(a: int32[N], b: int32[N]):
            for i in range(N):
                b[i] = _saxpy_helper(a[i], 1)

    return top, N


def test_systemc_helper_function_emit():
    """A kernel calling a compute helper emits the helper as a C++ free function
    (definition precedes the SC_MODULEs that call it)."""
    top, _ = _kernel_with_helper()
    code = df.build(top, target="systemc").hls_code
    assert "void _saxpy_helper(" in code  # helper defined
    assert code.index("void _saxpy_helper(") < code.index("SC_MODULE(k_0)")  # before use
    print("compute helper emitted as a C++ free function")


@pytest.mark.skipif(
    not os.environ.get("MGC_HOME"),
    reason="Catapult (MGC_HOME) not available — csim needs zhang-21",
)
def test_systemc_helper_function_csim():
    """The kernel+helper compiles and runs: B == A*2 + 1."""
    top, N = _kernel_with_helper()
    with tempfile.TemporaryDirectory() as tmp:
        mod = df.build(top, target="systemc", mode="csim", project=tmp)
        A = np.arange(N, dtype=np.int32)
        B = np.zeros(N, dtype=np.int32)
        mod(A, B)
        np.testing.assert_array_equal(B, A * 2 + 1)
    print("SystemC kernel+helper csim B == A*2 + 1")


def _smith_waterman():
    """Smith-Waterman local-alignment scoring on a P0xP1 systolic grid. Unlike a
    feed-forward GEMM, it has a DIAGONAL FEEDBACK stream (fifo_C: PE(i,j) ->
    PE(i+1,j+1)) plus horizontal/vertical streams, and 1-D inputs A,B read at
    pid-indexed positions (-> replicated read memory ports) and a 2-D output S
    written at pid positions (-> replicated write memory ports). Exercises the
    whole stack together: feedback systolic + memory ports + multi-client."""
    Wp, Sim, Mis = 2, 3, -3
    M = N = 4
    P0, P1 = M + 2, N + 2

    @df.region()
    def top(A: int8[M], B: int8[N], S: int32[P0 - 1, P1 - 1]):
        fifo_A: Stream[int32, 4][P0, P1]
        fifo_B: Stream[int32, 4][P0, P1]
        fifo_C: Stream[int32, 4][P0, P1]

        @df.kernel(mapping=[P0, P1], args=[A, B, S])
        def sw(local_A: int8[M], local_B: int8[N], local_S: int32[P0 - 1, P1 - 1]):
            i, j = df.get_pid()
            with allo.meta_if((i == 0 and j == P1 - 1) or (i == P0 - 1 and j == 0)):
                pass
            with allo.meta_elif(i == 0 and j == 0):
                fifo_C[i + 1, j + 1].put(0)
            with allo.meta_elif(i == 0):
                fifo_B[i + 1, j].put(0); fifo_C[i + 1, j + 1].put(0)
            with allo.meta_elif(j == 0):
                fifo_A[i, j + 1].put(0); fifo_C[i + 1, j + 1].put(0)
            with allo.meta_elif(i == P0 - 1 and j == P1 - 1):
                fifo_C[i, j].get()
            with allo.meta_elif(i == P0 - 1):
                fifo_B[i, j].get(); fifo_C[i, j].get()
            with allo.meta_elif(j == P1 - 1):
                fifo_A[i, j].get(); fifo_C[i, j].get()
            with allo.meta_else():
                a = fifo_A[i, j].get(); b = fifo_B[i, j].get(); c = fifo_C[i, j].get()
                aligning: int32 = c + (Sim if local_A[i - 1] == local_B[j - 1] else Mis)
                gap_A: int32 = a - Wp
                gap_B: int32 = b - Wp
                score: int32 = max(max(0, aligning), max(gap_A, gap_B))
                local_S[i, j] = score
                fifo_A[i, j + 1].put(max(gap_A, score))
                fifo_B[i + 1, j].put(max(gap_B, score))
                fifo_C[i + 1, j + 1].put(score)

    def golden(seqA, seqB):
        sm = np.zeros((len(seqA) + 1, len(seqB) + 1), dtype=int)
        for i in range(sm.shape[0]):
            for j in range(sm.shape[1]):
                if i == 0 or j == 0:
                    continue
                sim = Sim if seqA[i - 1] == seqB[j - 1] else Mis
                sm[i][j] = max(0, sm[i - 1][j - 1] + sim,
                               max([sm[a, j] - 2 * (i - a) for a in range(i)]),
                               max([sm[i, b] - 2 * (j - b) for b in range(j)]))
        return sm

    return top, M, N, P0, P1, golden


@pytest.mark.skipif(
    not os.environ.get("MGC_HOME"),
    reason="Catapult (MGC_HOME) not available — csim needs zhang-21",
)
def test_systemc_smith_waterman_csim():
    """Feedback systolic array (diagonal fifo_C) + memory ports end-to-end: the
    emitted SystemC score matrix matches the Smith-Waterman golden. Guards
    against regressing feedback-systolic correctness (a prior 'wrong' reading was
    a stale-build test artifact; the design is correct)."""
    top, M, N, P0, P1, golden = _smith_waterman()
    with tempfile.TemporaryDirectory() as tmp:
        mod = df.build(top, target="systemc", mode="csim", project=tmp)
        for seed in range(3):
            np.random.seed(seed)
            A = np.random.randint(0, 4, M).astype(np.int8)
            B = np.random.randint(0, 4, N).astype(np.int8)
            S = np.zeros((P0 - 1, P1 - 1), dtype=np.int32)
            mod(A.copy(), B.copy(), S)
            np.testing.assert_array_equal(S[1:, 1:], golden(A, B)[1:, 1:])
    print("SystemC smith-waterman csim == golden (feedback systolic)")


# --- @df.region HIERARCHY (regions must be defined at module level: Allo
# compiles their source via inspect, which fails on indented nested defs) ---
_HN, _HCAP = 4, 2


@df.region()
def _hier_inner(result: int32[1]):
    s: Stream[int32, _HCAP]

    @df.kernel(mapping=[1])
    def producer():
        for i in range(_HN):
            v: int32 = i
            s.put(v)

    @df.kernel(mapping=[1], args=[result])
    def consumer(out: int32[1]):
        acc: int32 = 0
        for i in range(_HN):
            acc += s.get()
        out[0] = acc


@df.region()
def _hier_top(result: int32[1]):
    @df.kernel(mapping=[1], args=[result])
    def driver(out: int32[1]):
        _hier_inner(out)


def test_systemc_hierarchy_flatten_emit():
    """A sub-region invoked from a kernel is FLATTENED: the leaf kernels become
    top-level modules; the delegating kernel + sub-region are inlined away.
    (SystemC can't nest a region inside an SC_THREAD; C++/HLS keeps the nesting.)"""
    code = df.build(_hier_top, target="systemc").hls_code
    assert "SC_MODULE(producer" in code and "SC_MODULE(consumer" in code
    assert "SC_MODULE(driver" not in code  # delegating kernel inlined away
    assert "SC_MODULE(_hier_inner" not in code  # sub-region inlined away
    print("sub-region flattened: leaf kernels hoisted, delegators removed")


@pytest.mark.skipif(
    not os.environ.get("MGC_HOME"),
    reason="Catapult (MGC_HOME) not available — csim needs zhang-21",
)
def test_systemc_hierarchy_flatten_csim():
    """The flattened hierarchy compiles + runs: result == sum(0.._HN-1)."""
    with tempfile.TemporaryDirectory() as tmp:
        mod = df.build(_hier_top, target="systemc", mode="csim", project=tmp)
        R = np.zeros(1, dtype=np.int32)
        mod(R)
        assert R[0] == sum(range(_HN)), f"got {R[0]}, expected {sum(range(_HN))}"
    print("SystemC flattened-hierarchy csim result == sum(0..N-1)")


@df.region()
def _pack_top(A: int16[2], B: int16[2]):
    s: Stream[UInt(32), 2][1]

    @df.kernel(mapping=[1], args=[A])
    def packer(a: int16[2]):
        p: UInt(32) = 0
        for m in range(2):
            p[m * 16 : (m + 1) * 16] = a[m]  # bit-slice WRITE (pack)
        s[0].put(p)

    @df.kernel(mapping=[1], args=[B])
    def unpacker(b: int16[2]):
        p: UInt(32) = s[0].get()
        for m in range(2):
            b[m] = p[m * 16 : (m + 1) * 16]  # bit-slice READ (unpack)


def test_systemc_bitslice_emit():
    """Packed streams pack/unpack with Vitis bit-slice syntax x(hi,lo). The ac_int
    shim lacks it, so ap_(u)int gets an operator()(hi,lo) bit-range proxy."""
    code = df.build(_pack_top, target="systemc").hls_code
    assert "struct ap_rng" in code  # the bit-range proxy is emitted
    assert "operator()(int hi, int lo)" in code
    print("bit-slice x(hi,lo) supported via ap_rng proxy")


@pytest.mark.skipif(
    not os.environ.get("MGC_HOME"),
    reason="Catapult (MGC_HOME) not available — csim needs zhang-21",
)
def test_systemc_bitslice_csim():
    """Pack 2x int16 into a UInt(32) stream and unpack -> B == A (exercises the
    bit-slice write AND read on a packed stream, like daisy_chain_gemm)."""
    with tempfile.TemporaryDirectory() as tmp:
        mod = df.build(_pack_top, target="systemc", mode="csim", project=tmp)
        A = np.array([7, 9], dtype=np.int16)
        B = np.zeros(2, dtype=np.int16)
        mod(A, B)
        np.testing.assert_array_equal(B, A)
    print("SystemC packed-stream bit-slice csim B == A")


if __name__ == "__main__":
    test_systemc_emit()
    if os.environ.get("MGC_HOME"):
        test_systemc_csim()
    else:
        print("skipped csim (set MGC_HOME on zhang-21 to run it)")
