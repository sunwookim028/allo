# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from allo.lang.core import APInt, i32, range as arange, Stream
from allo.lang.kernel import kernel


def test_simulator_apint_buffers():
    i5 = APInt(5, signed=True)
    u5 = APInt(5, signed=False)

    @kernel
    def addsub(A: i5[8], B: u5[8], C: i5[8]):
        for i in arange(8, name="i"):
            C[i] = A[i] + B[i]

    A = np.array([-4, -3, -2, -1, 0, 1, 2, 3], dtype=np.int8)
    B = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint8)
    C = np.zeros(8, dtype=np.int8)
    addsub(A, B, C)
    expected = ((A.astype(np.int16) + B + 16) % 32 - 16).astype(np.int8)
    np.testing.assert_array_equal(C, expected)


def test_simulator_apint_scalar_return():
    i13 = APInt(13, signed=True)

    @kernel
    def acc(A: i13[6]) -> i13:
        s: i13 = 0
        for i in arange(6, name="i"):
            s = s + A[i]
        return s

    A = np.array([-4000, 4000, -100, 100, -1, 1], dtype=np.int16)
    expected = int((int(A.sum()) + 4096) % 8192 - 4096)
    assert int(acc(A)) == expected


def test_simulator_scalar_stream():
    @kernel
    def top(x: i32[8], out: i32[8]):
        fifo: Stream[i32]

        @kernel
        def producer(src: i32[8], stream: Stream[i32]):
            for i in range(8):
                stream.put(src[i] + 1)

        @kernel
        def consumer(stream: Stream[i32], dst: i32[8]):
            for i in range(8):
                dst[i] = stream.get() * 2

        producer(x, fifo)
        consumer(fifo, out)

    x = np.arange(8, dtype=np.int32)
    out = np.zeros((8,), dtype=np.int32)

    top(x, out)

    np.testing.assert_array_equal(out, (x + 1) * 2)


def test_simulator_block_stream():
    @kernel
    def top(out: i32[2, 2, 2]):
        fifo: Stream[i32[2, 2]]

        @kernel
        def producer(stream: Stream[i32[2, 2]]):
            buf: i32[2, 2]
            buf[0, 0] = 1
            buf[0, 1] = 2
            buf[1, 0] = 3
            buf[1, 1] = 4
            stream.put(buf)
            buf[0, 0] = 10
            buf[0, 1] = 20
            buf[1, 0] = 30
            buf[1, 1] = 40
            stream.put(buf)

        @kernel
        def consumer(stream: Stream[i32[2, 2]], dst: i32[2, 2, 2]):
            first = stream.get()
            second = stream.get()
            dst[0, 0, 0] = first[0, 0]
            dst[0, 0, 1] = first[0, 1]
            dst[0, 1, 0] = first[1, 0]
            dst[0, 1, 1] = first[1, 1]
            dst[1, 0, 0] = second[0, 0]
            dst[1, 0, 1] = second[0, 1]
            dst[1, 1, 0] = second[1, 0]
            dst[1, 1, 1] = second[1, 1]

        producer(fifo)
        consumer(fifo, out)

    out = np.zeros((2, 2, 2), dtype=np.int32)

    top(out)

    expected = np.array(
        [[[1, 2], [3, 4]], [[10, 20], [30, 40]]],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(out, expected)


def test_simulator_bit_get_slice():
    """get_slice: unpack four bytes from each packed 32-bit word at a dynamic offset."""
    u32 = APInt(32, signed=False)

    @kernel
    def unpack(packed: u32[6], out: i32[6, 4]):
        for i in arange(6, name="i"):
            for p in arange(4, name="p"):
                out[i, p] = packed[i][p * 8 : p * 8 + 8]

    lanes = np.random.randint(0, 256, (6, 4)).astype(np.uint32)
    packed = np.zeros(6, dtype=np.uint32)
    for p in range(4):
        packed |= lanes[:, p] << (8 * p)

    out = np.zeros((6, 4), dtype=np.int32)
    unpack(packed, out)

    np.testing.assert_array_equal(out, lanes.astype(np.int32))


def test_simulator_bit_set_slice():
    """set_slice: pack four bytes into a 32-bit word at a dynamic offset."""
    u32 = APInt(32, signed=False)

    @kernel
    def pack(lanes: i32[6, 4], out: u32[6]):
        for i in arange(6, name="i"):
            word: u32 = 0
            for p in arange(4, name="p"):
                word[p * 8 : p * 8 + 8] = lanes[i, p]
            out[i] = word

    lanes = np.random.randint(0, 256, (6, 4)).astype(np.int32)
    out = np.zeros(6, dtype=np.uint32)
    pack(lanes, out)

    expected = np.zeros(6, dtype=np.uint32)
    for p in range(4):
        expected |= lanes[:, p].astype(np.uint32) << (8 * p)

    np.testing.assert_array_equal(out, expected)


def test_simulator_bit_extract_insert():
    """get_bit / set_bit: reverse the eight bits of each byte (width-1 slices)."""
    u8 = APInt(8, signed=False)

    @kernel
    def rev(src: u8[8], out: u8[8]):
        for i in arange(8, name="i"):
            r: u8 = 0
            for b in arange(8, name="b"):
                r[7 - b] = src[i][b]
            out[i] = r

    src = np.random.randint(0, 256, 8).astype(np.uint8)
    out = np.zeros(8, dtype=np.uint8)
    rev(src, out)

    expected = np.array([int(f"{int(v):08b}"[::-1], 2) for v in src], dtype=np.uint8)
    np.testing.assert_array_equal(out, expected)


def test_simulator_feedback_init_tokens():
    # A feedback CYCLE seeded with an initial token.
    N = 8

    @kernel
    async def emit(t: Stream[i32], s: Stream[i32], out: i32[N]):
        for i in range(N):
            x = t.get()
            out[i] = x
            s.put(x + 1)

    @kernel
    async def fwd(s: Stream[i32], t: Stream[i32]):
        for i in range(N):
            t.put(s.get())

    @kernel
    async def top(out: i32[N]):
        s: Stream[i32]
        t: Stream[i32] = [0]  # feedback channel, one initial token
        await emit(t, s, out)
        await fwd(s, t)

    out = np.zeros(N, np.int32)
    top(out)
    assert np.array_equal(out, np.arange(N, dtype=np.int32)), list(out)


def test_dataflow_nested_container_golden():
    # P-hier Slice 0: the CPU golden must run NESTED dataflow containers -- a
    # process that is itself a container. `mid` awaits inner_a/inner_b and is
    # awaited by `top`, carrying two boundary streams (s in, t out). The runtime
    # flattens the nest onto one marl scheduler (a nested `allo_df_open` reuses
    # the enclosing scheduler instead of binding a second one to the fiber's
    # thread, which aborts); each level keeps its own WaitGroup so joins are
    # scoped. csim ONLY -- RTL emit of a container-as-callee is Slice 1/2 and
    # still asserts; this pins the golden half.
    N = 16
    a = (np.arange(N, dtype=np.int32) * 7 + 13) & 0xFF

    @kernel
    async def produce(a: i32[N], s: Stream[i32]):
        for i in range(N):
            s.put(a[i])

    @kernel
    async def inner_a(x: Stream[i32], y: Stream[i32]):
        for i in range(N):
            y.put(x.get() + 1)

    @kernel
    async def inner_b(y: Stream[i32], z: Stream[i32]):
        for i in range(N):
            z.put(y.get() * 2)

    @kernel
    async def mid(x: Stream[i32], z: Stream[i32]):
        y: Stream[i32]
        await inner_a(x, y)
        await inner_b(y, z)

    @kernel
    async def consume(t: Stream[i32], out: i32[N]):
        for i in range(N):
            out[i] = t.get()

    @kernel
    async def top(a: i32[N], out: i32[N]):
        s: Stream[i32]
        t: Stream[i32]
        await produce(a, s)
        await mid(s, t)
        await consume(t, out)

    exp = (a + 1) * 2
    # csim is deterministic (KPN); repeat to surface any scheduler/WaitGroup race.
    for _ in range(8):
        out = np.zeros(N, np.int32)
        top(a, out)
        assert np.array_equal(out, exp), list(out)
