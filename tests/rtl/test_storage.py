# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""On-chip storage realization: banking, ROM-vs-RAM classification, multi-cycle access latency, and cross-region/container buffer identity."""

import os
import re
import shutil
import sys

import numpy as np
import pytest

from allo import kernel
from allo.lang import f32, i32, u8, index, Stateful, Stream
from allo.schedule import Schedule
from allo.schedule.errors import InvalidScheduleArgumentError
from allo.backend.base import run_pipeline
from allo.backend.rtl import Memory, RegisterFile, qor
from allo.backend.rtl.device import Resource, Tiled
from allo.backend.rtl.devices import default_device
from allo.backend.rtl.schedule import RTL_PREPARE_PIPELINE

sys.path.insert(0, os.path.dirname(__file__))
from _common import (  # noqa: E402
    Dcp,
    _walk,
    _sched,
    _to_rtl,
    _iis,
    MEM,
    MEM_URAM,
)

pytestmark = pytest.mark.skipif(
    shutil.which("verilator") is None, reason="verilator not available"
)

N = 8
A8 = np.arange(1, 9, dtype=np.int32)
A16 = (np.arange(16, dtype=np.int32) * 7 + 13) & 0xFF


# --- banking ------------------------------------------------------------


def test_banked_internal_buffer():
    # A partitioned internal buffer splits into per-bank on-chip memories: a
    # statically-resolvable index routes to its bank directly, a runtime-varying
    # one gets a crossbar (read every bank + mux, write-enable demux).
    @kernel
    def ibuf(A: i32[8], out: i32[8]):
        buf: i32[8] = 0
        for i in range(8):
            buf[i] = A[i] & 5
        for i in range(8):
            out[i] = buf[i] & A[i]

    out = np.zeros(8, np.int32)
    _to_rtl(ibuf).cosim(A8, out)
    assert np.array_equal(out, (A8 & 5) & A8)

    # Cyclic-2 accessed at even/odd indices -> two statically-banked halves. Each
    # bank runs a distinct op (+1 vs +100), so a swapped route or a fall-back to
    # one memory corrupts the golden.
    @kernel
    def bank(A: i32[16], out: i32[16]):
        buf: i32[16]
        for i in range(8):
            buf[2 * i] = A[2 * i] + 1
            buf[2 * i + 1] = A[2 * i + 1] + 100
        for i in range(8):
            out[2 * i] = buf[2 * i] & 255
            out[2 * i + 1] = buf[2 * i + 1] & 255

    s = bank.schedule()
    s.partition("buf", dim=1, kind=s.Cyclic, factor=2)
    mod = s.export("rtl")
    assert mod.mlir.count("= seq.hlmem") == 2  # two per-bank memories, not one

    out = np.zeros(16, np.int32)
    mod.cosim(A16, out)
    ref = A16.copy()
    ref[0::2] = (A16[0::2] + 1) & 255
    ref[1::2] = (A16[1::2] + 100) & 255
    assert np.array_equal(out, ref)

    # buf[i] under cyclic-2 is NOT statically banked (the bank alternates with the
    # loop counter), so the emitter builds the crossbar: bank (i & 1) / offset
    # (i >> 1), aligned with the 1-cycle read latency.
    @kernel
    def dbank(A: i32[16], out: i32[16]):
        buf: i32[16]
        for i in range(16):
            buf[i] = A[i] + 1
        for i in range(16):
            out[i] = buf[i] & 255

    s = dbank.schedule()
    s.partition("buf", dim=1, kind=s.Cyclic, factor=2)
    mod = s.export("rtl")
    assert mod.mlir.count("= seq.hlmem") == 2  # two banks, crossbar-addressed

    out = np.zeros(16, np.int32)
    mod.cosim(A16, out)
    assert np.array_equal(out, (A16 + 1) & 255)


def test_banked_boundary_argument():
    # A partitioned argument array becomes one boundary interface per bank; the
    # cosim harness splits the numpy argument into cyclic bank slices, joining on
    # writeback. A runtime-varying bank crossbars over those interfaces.
    @kernel
    def ext(A: i32[16], out: i32[16]):
        for i in range(8):
            out[2 * i] = A[2 * i] + 1
            out[2 * i + 1] = A[2 * i + 1] + 100

    s = ext.schedule()
    s.partition("A", dim=1, kind=s.Cyclic, factor=2)
    s.partition("out", dim=1, kind=s.Cyclic, factor=2)
    mod = s.export("rtl")
    # The boundary carries per-port bank info (both banks reached).
    iface = mod.interfaces[mod.top]
    assert {r.bank for acc in iface.reads for r in acc} == {0, 1}
    # Both halves of the split are derived on the affine expression, so both
    # fold: bank `(2i) mod 2` is a constant and offset `(2i) floordiv 2` is the
    # counter itself. A statically banked access therefore emits NO address
    # arithmetic at all. Deriving it on emitted values instead costs a multiply
    # and a shift per port, which nothing downstream can fold away.
    assert "comb.mul" not in mod.mlir and "comb.shru" not in mod.mlir

    out = np.zeros(16, np.int32)
    mod.cosim(A16, out)
    ref = A16.copy()
    ref[0::2] = A16[0::2] + 1
    ref[1::2] = A16[1::2] + 100
    assert np.array_equal(out, ref)

    @kernel
    def dext(A: i32[16], out: i32[16]):
        for i in range(16):
            out[i] = A[i] + 1

    s = dext.schedule()
    s.partition("A", dim=1, kind=s.Cyclic, factor=2)
    s.partition("out", dim=1, kind=s.Cyclic, factor=2)
    mod = s.export("rtl")
    # Each argument presents two bank interfaces (_b0/_b1), not one flat port.
    iface = mod.interfaces[mod.top]
    rbases = {r.base for acc in iface.reads for r in acc}
    wbases = {w.base for acc in iface.writes for w in acc}
    assert {"A_rd0_b0", "A_rd0_b1"} <= rbases
    assert {"out_wr0_b0", "out_wr0_b1"} <= wbases

    out = np.zeros(16, np.int32)
    mod.cosim(A16, out)
    assert np.array_equal(out, A16 + 1)


def test_banking_beyond_1d_pow2_cyclic():
    # Banking is decomposed in element space, not restricted to the 1-D
    # power-of-two cyclic case flat address arithmetic can express: covers a
    # BLOCK partition kind, a non-power-of-two cyclic FACTOR, and multi-dim
    # RANK, on both internal buffers and boundary arguments.

    # Kind: a BLOCK partition, internal. Reading at `15 - i` breaks index
    # symmetry, so a self-consistent but wrong bank select still scrambles.
    @kernel
    def blk(A: i32[16], out: i32[16]):
        buf: i32[16]
        for i in range(16):
            buf[i] = A[i] + 1
        for i in range(16):
            out[i] = buf[15 - i] & 255

    s = blk.schedule()
    s.partition("buf", dim=1, kind=s.Block, factor=2)
    mod = s.export("rtl")
    assert mod.mlir.count("= seq.hlmem") == 2
    out = np.zeros(16, np.int32)
    mod.cosim(A16, out)
    assert np.array_equal(out, (A16[::-1] + 1) & 255)

    # Kind, boundary side: the manifest publishes the block decomposition and
    # the host shards the argument by it.
    @kernel
    def eblk(A: i32[16], out: i32[16]):
        for i in range(16):
            out[i] = A[i] + 1

    s = eblk.schedule()
    s.partition("A", dim=1, kind=s.Block, factor=2)
    mod = s.export("rtl")
    rd = [r for acc in mod.interfaces[mod.top].reads for r in acc]
    assert rd[0].axes == (Memory.Axis(0, 2, "block"),)
    assert rd[0].shape == (16,)
    out = np.zeros(16, np.int32)
    mod.cosim(A16, out)
    assert np.array_equal(out, A16 + 1)

    # Factor: cyclic by 3 (a divider, not a shift) over a length the factor does
    # not divide, so banks 1..2 each carry a padding slot.
    A10 = (np.arange(10, dtype=np.int32) * 3 + 1) & 0xFF

    @kernel
    def np2(A: i32[10], out: i32[10]):
        for i in range(10):
            out[i] = A[i] + 1

    s = np2.schedule()
    s.partition("A", dim=1, kind=s.Cyclic, factor=3)
    mod = s.export("rtl")
    out = np.zeros(10, np.int32)
    mod.cosim(A10, out)
    assert np.array_equal(out, A10 + 1)

    # Rank: a 2-D ARGUMENT, cyclic on the last dim, with an ODD row length, so
    # the element-space bank (`j % 2`) and a flat one (`(i*5 + j) % 2`) differ on
    # every odd row.
    A45 = ((np.arange(20, dtype=np.int32) * 7 + 13) & 0xFF).reshape(4, 5)

    @kernel
    def ext2d(A: i32[4, 5], out: i32[4, 5]):
        for i in range(4):
            for j in range(5):
                out[i, j] = A[i, j] + 1

    s = ext2d.schedule()
    s.partition("A", dim=2, kind=s.Cyclic, factor=2)
    mod = s.export("rtl")
    out = np.zeros((4, 5), np.int32)
    mod.cosim(A45, out)
    assert np.array_equal(out, A45 + 1)

    # Rank, internal + data-dependent: cyclic on dim 1 (ROWS) of a 2-D buffer.
    # The bank is the row parity `i % 2`, which a flat address cannot express.
    A48 = ((np.arange(32, dtype=np.int32) * 7 + 13) & 0xFF).reshape(4, 8)

    @kernel
    def int2d(A: i32[4, 8], out: i32[4, 8]):
        buf: i32[4, 8]
        for i in range(4):
            for j in range(8):
                buf[i, j] = A[i, j] + 1
        for i in range(4):
            for j in range(8):
                out[i, j] = buf[3 - i, j] & 255

    s = int2d.schedule()
    s.partition("buf", dim=1, kind=s.Cyclic, factor=2)
    mod = s.export("rtl")
    assert mod.mlir.count("= seq.hlmem") == 2
    out = np.zeros((4, 8), np.int32)
    mod.cosim(A48, out)
    assert np.array_equal(out, (A48[::-1, :] + 1) & 255)


def test_nested_banked_static_split():
    # A 2D nest accessing a cyclic-partitioned buffer on its inner (partitioned)
    # dim. `loop-canonicalization` must not coalesce the inner loop -- coalescing
    # would delinearize j and defeat static bank resolution, falling back to a
    # runtime crossbar. With the skip, buf banks statically (two per-bank
    # memories, no `_b<k>` crossbar).
    @kernel
    def nb(A: i32[4, 8], out: i32[4, 8]):
        buf: i32[4, 8]
        for i in range(4):
            for j in range(4):
                buf[i, 2 * j] = A[i, 2 * j] + 1
                buf[i, 2 * j + 1] = A[i, 2 * j + 1] + 100
        for i in range(4):
            for j in range(4):
                out[i, 2 * j] = buf[i, 2 * j] & 255
                out[i, 2 * j + 1] = buf[i, 2 * j + 1] & 255

    s = nb.schedule()
    s.partition("buf", dim=2, kind=s.Cyclic, factor=2)
    mod = s.export("rtl")
    assert mod.mlir.count("= seq.hlmem") == 2
    assert "@buf_b" not in mod.mlir

    A = ((np.arange(32, dtype=np.int32) * 7 + 13) & 0xFF).reshape(4, 8)
    out = np.zeros((4, 8), np.int32)
    mod.cosim(A, out)
    ref = A.copy()
    ref[:, 0::2] = (A[:, 0::2] + 1) & 255
    ref[:, 1::2] = (A[:, 1::2] + 100) & 255
    assert np.array_equal(out, ref)


def test_a_dynamic_bank_costs_a_port_on_every_bank():
    # The crossbar a data-dependent bank falls back to READS EVERY BANK, so one
    # logical access holds one port on each: a partitioned array's concurrent
    # capacity is `portsPerBank`, not `portsPerBank * factor`. Three reads per
    # iteration therefore cannot all issue in one cycle on a 2-port RAM, and the
    # port model has to say so rather than bill them against a lumped pool.
    @kernel
    def stencil(A: i32[18], out: i32[16]):
        buf: i32[18]
        for i in range(18):
            buf[i] = A[i] & 255
        for i in range(16):
            out[i] = (buf[i] + buf[i + 1] + buf[i + 2]) & 255

    s = stencil.schedule()
    s.partition("buf", dim=1, kind=s.Cyclic, factor=2)
    mod = s.export("rtl")
    # Two banks, three crossbarred reads: six read ports, three on each bank.
    assert len(re.findall(r"\bseq\.read\b", mod.mlir)) == 6
    # A distributed RAM instance serves one addressed read, so a bank taking
    # three of them is held in three copies: the reads split over them and every
    # write reaches all.
    assert mod.mlir.count("= seq.hlmem") == 6
    # ... which is one more than the two banks hold, so the read loop runs at
    # II=2 while the copy loop stays fully pipelined.
    assert _iis(mod.schedule().cyclic()) == [1, 2]

    A18 = (np.arange(18, dtype=np.int32) * 7 + 13) & 0xFF
    out = np.zeros(16, np.int32)
    mod.cosim(A18, out)
    ref = A18 & 255
    assert np.array_equal(out, (ref[:16] + ref[1:17] + ref[2:18]) & 255)


def test_a_static_bank_skips_the_crossbar():
    # `resolve-banking` bails on the whole alloc because ONE access has a
    # data-dependent bank, so the buffer reaches emit still partitioned. The
    # accesses whose bank IS a compile-time constant must still route straight
    # to their own memory, the way the boundary path already does, rather than
    # be dragged into the crossbar by their sibling.
    @kernel
    def mixed(A: i32[16], out: i32[16]):
        buf: i32[16]
        for i in range(16):
            buf[i] = A[i] + 1  # data-dependent bank: no static split
        for i in range(8):
            out[2 * i] = buf[2 * i] & 255  # bank 0
            out[2 * i + 1] = buf[2 * i + 1] & 255  # bank 1

    s = mixed.schedule()
    s.partition("buf", dim=1, kind=s.Cyclic, factor=2)
    mod = s.export("rtl")
    assert mod.mlir.count("= seq.hlmem") == 2
    # The dynamic write drives both banks (a write-enable demux); each static
    # read touches only its own.
    assert len(re.findall(r"\bseq\.write\b", mod.mlir)) == 2
    assert len(re.findall(r"\bseq\.read\b", mod.mlir)) == 2

    out = np.zeros(16, np.int32)
    mod.cosim(A16, out)
    assert np.array_equal(out, (A16 + 1) & 255)

    # The same on a BLOCK axis, which is statically banked only at a constant
    # subscript and so never reaches a split. `loop-canonicalization` coalesces
    # both nests here, so the read's map is `(iv) -> (iv floordiv 6, 3)`: the
    # constant has to survive the row-major round trip for the bank to resolve.
    @kernel
    def blk2(A: i32[4, 6], out: i32[4, 6]):
        buf: i32[4, 6]
        for i in range(4):
            for j in range(6):
                buf[i, j] = A[i, j] + 1
        for i in range(4):
            for j in range(6):
                out[i, j] = buf[i, 3] & 255

    s = blk2.schedule()
    s.partition("buf", dim=2, kind=s.Block, factor=2)
    mod = s.export("rtl")
    assert mod.mlir.count("= seq.hlmem") == 2
    assert len(re.findall(r"\bseq\.write\b", mod.mlir)) == 2  # roaming write
    assert len(re.findall(r"\bseq\.read\b", mod.mlir)) == 1  # bank 1 only

    A46 = ((np.arange(24, dtype=np.int32) * 7 + 13) & 0xFF).reshape(4, 6)
    out = np.zeros((4, 6), np.int32)
    mod.cosim(A46, out)
    col = ((A46[:, 3] + 1) & 255)[:, None]
    assert np.array_equal(out, np.broadcast_to(col, (4, 6)))


def test_a_block_partition_resolves_a_loop_per_chunk():
    # The standard block idiom: partition an array into chunks and give each a
    # loop of its own. Every access provably lies in ONE block, but the digit
    # `i floordiv 8` folds for no `i`, so the bank resolves only if the fold is
    # given the loop's TRIP COUNT and not just its lower bound and step.
    # Without that, all four accesses crossbar over both banks.
    @kernel
    def perchunk(A: i32[16], out: i32[16]):
        buf: i32[16]
        for i in range(8):
            buf[i] = A[i] + 1  # block 0, whatever i is
        for i in range(8, 16):
            buf[i] = A[i] + 2  # block 1, whatever i is
        for i in range(8):
            out[i] = buf[i] & 255
            out[i + 8] = buf[i + 8] & 255

    s = perchunk.schedule()
    s.partition("buf", dim=1, kind=s.Block, factor=2)
    mod = s.export("rtl")
    assert mod.mlir.count("= seq.hlmem") == 2
    # One port per access on its own bank, not one on every bank: two writes and
    # two reads total, where a crossbar would take four of each.
    assert len(re.findall(r"\bseq\.write\b", mod.mlir)) == 2
    assert len(re.findall(r"\bseq\.read\b", mod.mlir)) == 2

    out = np.zeros(16, np.int32)
    mod.cosim(A16, out)
    exp = np.concatenate([A16[:8] + 1, A16[8:] + 2]) & 255
    assert np.array_equal(out, exp)


def test_a_reversed_index_carries_its_digits_in_down_counters():
    # `buf[N-1-k*i]` runs every digit of the address BACKWARDS. A digit is still
    # a register, counting down and wrapping by ADDING the modulus on borrow
    # rather than subtracting it on overflow, so the residue and the quotient
    # cost the same as they do forwards. Non-power-of-two factors, so the wrap
    # is real arithmetic and not a mask, and a stride so the decrement is > 1.
    @kernel
    def revstep(A: i32[24], out: i32[12]):
        buf: i32[24]
        for i in range(24):
            buf[i] = A[i] + 1
        for i in range(12):
            out[i] = buf[22 - 2 * i] & 255

    A24 = (np.arange(24, dtype=np.int32) * 11 + 5) & 0xFF
    for kind in (Schedule.Cyclic, Schedule.Block):
        s = revstep.schedule()
        s.partition("buf", dim=1, kind=kind, factor=3)
        mod = s.export("rtl")
        assert mod.mlir.count("= seq.hlmem") == 3
        # Nothing divides: every digit of a decreasing address is a register.
        assert "comb.divu" not in mod.mlir and "comb.modu" not in mod.mlir
        out = np.zeros(12, np.int32)
        mod.cosim(A24, out)
        assert np.array_equal(out, (A24[22::-2] + 1) & 255)


def test_the_bank_an_access_reaches_is_decided_once():
    # The bank is DECIDED before the scheduler (`assign-banks`) and recorded on
    # the access, so the port model, the static split and the emitter's routing
    # read one fact instead of each re-deriving it on the map form its own layer
    # sees. It is visible in the reified IR: an assigned bank on the two static
    # reads, and none on the write, which reaches every bank.
    @kernel
    def mixed(A: i32[16], out: i32[16]):
        buf: i32[16]
        for i in range(16):
            buf[i] = A[i] + 1  # roaming: no bank
        for i in range(8):
            out[2 * i] = buf[2 * i] & 255  # bank 0
            out[2 * i + 1] = buf[2 * i + 1] & 255  # bank 1

    s = mixed.schedule()
    s.partition("buf", dim=1, kind=s.Cyclic, factor=2)
    mod = s.export("rtl")
    fn = Dcp(mod).func("mixed")
    (alloc,) = fn.ops("memref.alloc")
    access = fn.accesses(alloc.results[0])
    assert [a.attributes["bank"].value for a in access if "bank" in a.attributes] == [
        0,
        1,
    ]
    assert sum("bank" not in a.attributes for a in access) == 1  # the roaming write

    # A BLOCK partition now splits, which it never could while the split ran its
    # own cyclic-only predicate: a block axis is statically banked at a constant
    # subscript, so an array every access of which was assigned a bank is
    # materializable whatever the axis kind. `@buf_b<k>` is the crossbar naming
    # a still-partitioned memref keeps.
    @kernel
    def blk(A: i32[4, 6], out: i32[4, 6]):
        buf: i32[4, 6]
        for i in range(4):
            for j in range(6):
                buf[i, 1] = A[i, j] + 1  # bank 0 (columns 0..2)
                buf[i, 4] = A[i, j] + 100  # bank 1 (columns 3..5)
        for i in range(4):
            for j in range(6):
                out[i, j] = (buf[i, 1] + buf[i, 4]) & 255

    s = blk.schedule()
    s.partition("buf", dim=2, kind=s.Block, factor=2)
    mod = s.export("rtl")
    assert mod.mlir.count("= seq.hlmem") == 2
    assert "@buf_b" not in mod.mlir  # split into two memrefs, not crossbarred

    A46 = ((np.arange(24, dtype=np.int32) * 7 + 13) & 0xFF).reshape(4, 6)
    out = np.zeros((4, 6), np.int32)
    mod.cosim(A46, out)
    row = ((2 * A46[:, 5] + 101) & 255)[:, None]  # the last `j` wins both slots
    assert np.array_equal(out, np.broadcast_to(row, (4, 6)))


def test_composed_banking():
    # Banking a COMPOSED array: a partition stated once where the array lives
    # (`reconcile-array-directives`) reaches every callee parameter, so each child
    # emits a port group per bank and the container materializes one memory per
    # bank with no crossbar of its own. Covers a container-local buffer and a
    # container argument.
    @kernel
    def cbi_prod(A: i32[16], tmp: i32[16]):
        for i in range(8):
            tmp[2 * i] = A[2 * i] + 1
            tmp[2 * i + 1] = A[2 * i + 1] + 100

    @kernel
    def cbi_cons(tmp: i32[16], out: i32[16]):
        for i in range(8):
            out[2 * i] = tmp[2 * i] & 255
            out[2 * i + 1] = tmp[2 * i + 1] & 255

    @kernel
    def cbi_top(A: i32[16], out: i32[16]):
        tmp: i32[16]  # container-local, partitioned -> two on-chip banks
        cbi_prod(A, tmp)
        cbi_cons(tmp, out)

    s = cbi_top.schedule()
    s.partition("tmp", dim=1, kind=s.Cyclic, factor=2)
    mod = s.export("rtl")
    # the leaf CallUnit path
    assert "cbi_top.cbi_prod" in Dcp(mod).func("cbi_top").callees()
    assert re.findall(r"seq\.hlmem @(\w+) [^:]*: <(\d+)x", mod.mlir) == [
        ("tmp_b0", "8"),
        ("tmp_b1", "8"),
    ]
    A = (np.arange(16, dtype=np.int32) * 5 + 2) & 0x3F
    out = np.zeros(16, np.int32)
    mod.cosim(A, out)
    # even lanes +1, odd lanes +100 -- a swapped bank route corrupts the golden
    assert np.array_equal(out, np.where(np.arange(16) % 2 == 0, A + 1, A + 100) & 255)

    # The boundary dual: a partitioned container ARGUMENT. The child exposes one
    # port group per bank and the container mirrors them onto the top, each
    # carrying its own `bank`/`factor` for the cosim harness to shard by.
    @kernel
    def cbb(A: i32[16], o: i32[16]):
        for i in range(8):
            o[2 * i] = A[2 * i] + 1
            o[2 * i + 1] = A[2 * i + 1] + 100

    @kernel
    def cbb_top(A: i32[16], o: i32[16]):
        cbb(A, o)

    s = cbb_top.schedule()
    s.partition("A", dim=1, kind=s.Cyclic, factor=2)
    mod = s.export("rtl")
    assert "cbb_top.cbb" in Dcp(mod).func("cbb_top").callees()
    rd = [g[0] for g in mod.interfaces["cbb_top"].reads]
    assert {(r.arg, r.bank, r.factor) for r in rd} == {(0, 0, 2), (0, 1, 2)}
    o = np.zeros(16, np.int32)
    mod.cosim(A16, o)
    assert np.array_equal(o, np.where(np.arange(16) % 2 == 0, A16 + 1, A16 + 100))


def test_a_partition_stated_on_a_leaf_reaches_its_container():
    # A sub-kernel MASTERS PORTS on its caller's array rather than receiving a
    # copy, so which end the directive is written on is not a property of the
    # array. Stated on the leaf's own parameter it has to reach the container
    # that allocates the buffer, and through it the leaf's sibling, or the
    # container would materialize one memory for banks its children address.
    @kernel
    def upw_prod(A: i32[16], tmp: i32[16]):
        for i in range(8):
            tmp[2 * i] = A[2 * i] + 1
            tmp[2 * i + 1] = A[2 * i + 1] + 100

    @kernel
    def upw_cons(tmp: i32[16], out: i32[16]):
        for i in range(8):
            out[2 * i] = tmp[2 * i] & 255
            out[2 * i + 1] = tmp[2 * i + 1] & 255

    @kernel
    def upw_top(A: i32[16], out: i32[16]):
        tmp: i32[16]
        upw_prod(A, tmp)
        upw_cons(tmp, out)

    ps = upw_prod.schedule()
    ps.partition("tmp", dim=1, kind=ps.Cyclic, factor=2)
    s = upw_top.schedule()
    s.compose(ps)
    mod = s.export("rtl")
    assert re.findall(r"seq\.hlmem @(\w+) [^:]*: <(\d+)x", mod.mlir) == [
        ("tmp_b0", "8"),
        ("tmp_b1", "8"),
    ]
    A = (np.arange(16, dtype=np.int32) * 5 + 2) & 0x3F
    out = np.zeros(16, np.int32)
    mod.cosim(A, out)
    assert np.array_equal(out, np.where(np.arange(16) % 2 == 0, A + 1, A + 100) & 255)


def test_partitions_of_two_kernels_compose_across_dimensions():
    # Two kernels asking for different axes of one array is not a conflict: a
    # directive is a LOWER BOUND on the banks its kernel needs, and splitting
    # both ways refines both, so each still gets the conflict-free groups it
    # asked for. Four banks of a 2x2 tile, and both children address them.
    @kernel
    def xd_prod(A: i32[4, 4], buf: i32[4, 4]):
        for i in range(2):
            for j in range(2):
                buf[2 * i, 2 * j] = A[2 * i, 2 * j] + 1
                buf[2 * i, 2 * j + 1] = A[2 * i, 2 * j + 1] + 2
                buf[2 * i + 1, 2 * j] = A[2 * i + 1, 2 * j] + 3
                buf[2 * i + 1, 2 * j + 1] = A[2 * i + 1, 2 * j + 1] + 4

    @kernel
    def xd_cons(buf: i32[4, 4], out: i32[4, 4]):
        for i in range(2):
            for j in range(2):
                out[2 * i, 2 * j] = buf[2 * i, 2 * j] & 255
                out[2 * i, 2 * j + 1] = buf[2 * i, 2 * j + 1] & 255
                out[2 * i + 1, 2 * j] = buf[2 * i + 1, 2 * j] & 255
                out[2 * i + 1, 2 * j + 1] = buf[2 * i + 1, 2 * j + 1] & 255

    @kernel
    def xd_top(A: i32[4, 4], out: i32[4, 4]):
        buf: i32[4, 4]
        xd_prod(A, buf)
        xd_cons(buf, out)

    ps = xd_prod.schedule()
    ps.partition("buf", dim=1, kind=ps.Cyclic, factor=2)  # rows
    cs = xd_cons.schedule()
    cs.partition("buf", dim=2, kind=cs.Cyclic, factor=2)  # columns
    s = xd_top.schedule()
    s.compose(ps, cs)
    mod = s.export("rtl")
    # Both axes, one attribute, on the buffer and on both children alike.
    d = Dcp(mod)
    (alloc,) = d.func("xd_top").ops("memref.alloc")
    part = alloc.attributes["allo.part"]
    assert d.func("xd_top.xd_prod").arg_attrs("allo.part") == [part]
    assert d.func("xd_top.xd_cons").arg_attrs("allo.part") == [part]
    assert re.findall(r"seq\.hlmem @(\w+) [^:]*: <(\d+)x", mod.mlir) == [
        ("buf_b0", "4"),
        ("buf_b1", "4"),
        ("buf_b2", "4"),
        ("buf_b3", "4"),
    ]
    A44 = ((np.arange(16, dtype=np.int32) * 7 + 3) & 0x3F).reshape(4, 4)
    out = np.zeros((4, 4), np.int32)
    mod.cosim(A44, out)
    # A swapped bank route on either side lands the wrong element in the tile.
    r, c = np.indices((4, 4))
    assert np.array_equal(out, (A44 + 1 + (r % 2) * 2 + (c % 2)) & 255)


def test_partition_factors_join_to_the_finer_one():
    # On ONE dimension the join has to stay a SINGLE axis, so it exists exactly
    # when one factor divides the other: cyclic-4 keeps apart every pair
    # cyclic-2 did (residues mod 2 are unions of residues mod 4), so the
    # consumer that asked for two banks is served by four.
    @kernel
    def fj_prod(A: i32[16], tmp: i32[16]):
        for i in range(4):
            tmp[4 * i] = A[4 * i] + 1
            tmp[4 * i + 1] = A[4 * i + 1] + 2
            tmp[4 * i + 2] = A[4 * i + 2] + 3
            tmp[4 * i + 3] = A[4 * i + 3] + 4

    @kernel
    def fj_cons(tmp: i32[16], out: i32[16]):
        for i in range(8):
            out[2 * i] = tmp[2 * i] & 255
            out[2 * i + 1] = tmp[2 * i + 1] & 255

    @kernel
    def fj_top(A: i32[16], out: i32[16]):
        tmp: i32[16]
        fj_prod(A, tmp)
        fj_cons(tmp, out)

    ps = fj_prod.schedule()
    ps.partition("tmp", dim=1, kind=ps.Cyclic, factor=4)
    cs = fj_cons.schedule()
    cs.partition("tmp", dim=1, kind=cs.Cyclic, factor=2)
    s = fj_top.schedule()
    s.compose(ps, cs)
    mod = s.export("rtl")
    # By bank, with the instance suffix stripped: the number of copies a bank is
    # held in follows from its read ports and is not what this checks.
    cells = {
        (re.sub(r"_c\d+$", "", name), depth)
        for name, depth in re.findall(r"seq\.hlmem @(\w+) [^:]*: <(\d+)x", mod.mlir)
    }
    assert sorted(cells) == [
        ("tmp_b0", "4"),
        ("tmp_b1", "4"),
        ("tmp_b2", "4"),
        ("tmp_b3", "4"),
    ]
    A = (np.arange(16, dtype=np.int32) * 5 + 2) & 0x3F
    out = np.zeros(16, np.int32)
    mod.cosim(A, out)
    assert np.array_equal(out, (A + 1 + np.arange(16) % 4) & 255)


def test_a_block_and_a_cyclic_axis_on_one_dimension_are_reported():
    # The one pair with no join: block chunks a dimension and cyclic
    # interleaves it, so the two send the same element to different banks and no
    # single banking serves both. Reported rather than silently picking a side,
    # since the loser would address the wrong elements. Asserted on the refusal
    # CODE, the only stable token a diagnostic carries, so the check neither
    # reads the log nor depends on the wording.
    @kernel
    def bc_prod(A: i32[16], tmp: i32[16]):
        for i in range(16):
            tmp[i] = A[i] + 1

    @kernel
    def bc_cons(tmp: i32[16], out: i32[16]):
        for i in range(16):
            out[i] = tmp[i] & 255

    @kernel
    def bc_top(A: i32[16], out: i32[16]):
        tmp: i32[16]
        bc_prod(A, tmp)
        bc_cons(tmp, out)

    ps = bc_prod.schedule()
    ps.partition("tmp", dim=1, kind=ps.Block, factor=2)
    cs = bc_cons.schedule()
    cs.partition("tmp", dim=1, kind=cs.Cyclic, factor=2)
    s = bc_top.schedule()
    s.compose(ps, cs)
    with pytest.raises(RuntimeError):
        s.export("rtl").schedule()


def test_a_partition_stated_on_a_process_reaches_its_container():
    # The dual of `test_a_partitioned_container_local_buffer`: an async spawn is
    # a call like any other, so a directive on the PROCESS's own parameter joins
    # the same storage class as the container's buffer, and the sibling process
    # reading it learns the banking through that class.
    @kernel
    async def pp_src(s: Stream[i32]):
        s.put(42)

    @kernel
    async def pp_side(s: Stream[i32], o0: i32[1]):
        o0[0] = s.get()

    @kernel
    async def pp_prod(x: i32[16], tmp: i32[16]):
        for i in range(8):
            tmp[2 * i] = x[2 * i] + 1
            tmp[2 * i + 1] = x[2 * i + 1] + 100

    @kernel
    def pp_cons(tmp: i32[16], out: i32[16]):
        for i in range(16):
            out[i] = tmp[i] & 255

    @kernel
    async def pp_top(x: i32[16], out: i32[16], o0: i32[1]):
        f: Stream[i32]
        tmp: i32[16]
        await pp_src(f)
        await pp_side(f, o0)
        await pp_prod(x, tmp)
        pp_cons(tmp, out)

    ps = pp_prod.schedule()
    ps.partition("tmp", dim=1, kind=ps.Cyclic, factor=2)
    s = pp_top.schedule()
    s.compose(ps)
    mod = s.export("rtl")
    m = mod.mlir
    assert "seq.hlmem @tmp_b0" in m and "seq.hlmem @tmp_b1" in m
    x = (np.arange(16, dtype=np.int32) * 5 + 2) & 0x3F
    out = np.zeros(16, np.int32)
    o0 = np.zeros(1, np.int32)
    mod.cosim(x, out, o0)
    exp = np.where(np.arange(16) % 2 == 0, x + 1, x + 100) & 255
    assert np.array_equal(out, exp), list(out)


def _transpose_pair(part, factor=4):
    """`out[i,j] = buf[i,j] + buf[j,i]` over an internal 8x8 tile, unrolled by
    4 on `j`, with `out`/`Ain` cyclic so `buf` is the only port bottleneck."""

    @kernel
    def sym(Ain: i32[8, 8], out: i32[8, 8]):
        buf: i32[8, 8]
        for i in range(8, name="ci"):
            for j in range(8, name="cj"):
                buf[i, j] = Ain[i, j]
        for i in range(8, name="i"):
            for j in range(8, name="j"):
                out[i, j] = buf[i, j] + buf[j, i]

    s = sym.schedule()
    s.unroll("j", factor=4)
    s.partition("out", dim=2, kind=s.Cyclic, factor=4)
    s.partition("Ain", dim=2, kind=s.Cyclic, factor=4)
    if part:
        s.partition("buf", dim=part[0], kind=part[1], factor=factor)
    return s.export("rtl")


A88 = ((np.arange(64, dtype=np.int32) * 7 + 3) & 255).reshape(8, 8)


def test_a_skewed_partition_serves_a_row_and_a_column():
    # Block and cyclic are each a function of ONE subscript, so no choice of
    # axis serves an array read as both `buf[i,j]` and `buf[j,i]`: whichever
    # axis is partitioned, the other pattern walks a single bank. A SKEW banks
    # on the sum of the subscripts, and the four unrolled copies of each access
    # then reach four distinct banks. Not a compile-time bank each, but a
    # distinct one, which is what a port is billed against.
    cyc = _transpose_pair((2, Schedule.Cyclic))
    skew = _transpose_pair((2, Schedule.Skew))

    # Four banks either way; what differs is the ports into them, and so how
    # many copies of the row a bank needs, a distributed RAM instance serving
    # one addressed read. Under cyclic the four row reads bank statically (one
    # port each) and the four column reads crossbar (a port on every bank):
    # 4 + 4*4 = 20 reads, five to a bank, five copies of each. Under the skew
    # the eight reads fall into two lanes of four distinct slots and a lane
    # shares one port per bank: 2 * 4 = 8 reads, two to a bank, two copies.
    assert skew.mlir.count("= seq.hlmem") == 8
    assert cyc.mlir.count("= seq.hlmem") == 20
    assert len(re.findall(r"\bseq\.read\b", cyc.mlir)) == 20
    assert len(re.findall(r"\bseq\.read\b", skew.mlir)) == 8

    # ... so the read loop closes at II=1 where cyclic bottoms out at 3.
    assert _iis(cyc.schedule().cyclic()) == [1, 3]
    assert _iis(skew.schedule().cyclic()) == [1, 1]

    # The rotation, the lane muxes and the host's copy of the layout all have to
    # agree or the sum is of the wrong two elements.
    for mod in (cyc, skew):
        out = np.zeros((8, 8), np.int32)
        mod.cosim(A88, out)
        assert np.array_equal(out, A88 + A88.T)


def test_a_skewed_argument_keeps_the_conservative_billing():
    # An argument's banks are boundary interfaces the manifest published, one
    # set per access, so there is no shared port for a lane to economize and no
    # slot to bill. The LAYOUT still applies end to end: the host reproduces the
    # skew to shard the array, so a disagreement between it and the emitted
    # address arithmetic shows up as scrambled data.
    @kernel
    def ext(Ain: i32[8, 8], out: i32[8, 8]):
        for i in range(8, name="i"):
            for j in range(8, name="j"):
                out[i, j] = Ain[i, j] + 1

    s = ext.schedule()
    s.partition("Ain", dim=2, kind=s.Skew, factor=4)
    mod = s.export("rtl")
    rd = [r for acc in mod.interfaces[mod.top].reads for r in acc]
    assert rd[0].axes == (Memory.Axis(1, 4, "skew"),)
    out = np.zeros((8, 8), np.int32)
    mod.cosim(A88, out)
    assert np.array_equal(out, A88 + 1)


def test_a_skew_whose_accesses_disagree_resolves_nothing():
    # A slot is billable only because the array's contending accesses share one
    # bank expression up to a constant: then a distinct slot IS a distinct bank
    # at every rotation. `buf[i,j]` and `buf[i,2*j]` do not, so they can collide
    # and the array falls back to the crossbar, which it must REPORT, since a
    # partition that resolves nothing is pure area.
    @kernel
    def bad(Ain: i32[8, 8], out: i32[8, 8]):
        buf: i32[8, 8]
        for i in range(8, name="ci"):
            for j in range(8, name="cj"):
                buf[i, j] = Ain[i, j]
        for i in range(8, name="i"):
            for j in range(4, name="j"):
                out[i, j] = buf[i, j] + buf[i, 2 * j]

    s = bad.schedule()
    s.partition("buf", dim=2, kind=s.Skew, factor=4)
    mod = s.export("rtl")
    banked = mod.microarch.mem("buf_b0")
    # The partition is BUILT, four banks of it, and buys nothing: no access is
    # fixed to a bank, so each takes a port on every one of them.
    assert banked.layout == "skew" and banked.banks == 4
    assert not banked.partition_resolved
    out = np.zeros((8, 8), np.int32)
    mod.cosim(A88, out)
    ref = np.zeros((8, 8), np.int32)
    ref[:, :4] = A88[:, :4] + A88[:, 0:8:2]
    assert np.array_equal(out, ref)


def test_a_skew_must_name_its_distribution_dimension():
    # `dim=0` means "every dimension" for block and cyclic. A skew's bank
    # already reads every subscript, so the flag has no meaning for it: `dim`
    # names the one dimension divided down to make room for the banks.
    @kernel
    def k(A: i32[8, 8], out: i32[8, 8]):
        for i in range(8):
            for j in range(8):
                out[i, j] = A[i, j]

    s = k.schedule()
    with pytest.raises(InvalidScheduleArgumentError):
        s.partition("A", dim=0, kind=s.Skew, factor=4)


def test_a_partitioned_container_local_buffer():
    # `reconcile-array-directives` gives every child the same `allo.part`, so the
    # container just materializes the banks they already agree on: `bk_prod`
    # writes STATIC banks (one single-bank port group each), while `bk_cons`
    # reads a DATA-DEPENDENT one (crossbarred inside the child). Same shape the
    # leaf path takes, on the structural top.
    @kernel
    async def bk_src(s: Stream[i32]):
        s.put(42)

    @kernel
    async def bk_side(s: Stream[i32], o0: i32[1]):
        o0[0] = s.get()

    @kernel
    async def bk_prod(x: i32[16], tmp: i32[16]):
        for i in range(8):
            tmp[2 * i] = x[2 * i] + 1
            tmp[2 * i + 1] = x[2 * i + 1] + 100

    @kernel
    def bk_cons(tmp: i32[16], out: i32[16]):
        for i in range(16):
            out[i] = tmp[i] & 255

    @kernel
    async def bk_top(x: i32[16], out: i32[16], o0: i32[1]):
        f: Stream[i32]
        tmp: i32[16]
        await bk_src(f)
        await bk_side(f, o0)
        await bk_prod(x, tmp)
        bk_cons(tmp, out)

    s = bk_top.schedule()
    s.partition("tmp", dim=1, kind=s.Cyclic, factor=2)
    mod = s.export("rtl")
    m = mod.mlir
    assert "seq.hlmem @tmp_b0" in m and "seq.hlmem @tmp_b1" in m
    x = (np.arange(16, dtype=np.int32) * 5 + 2) & 0x3F
    out = np.zeros(16, np.int32)
    o0 = np.zeros(1, np.int32)
    mod.cosim(x, out, o0)
    # even lanes +1, odd lanes +100; a swapped bank route corrupts the golden
    exp = np.where(np.arange(16) % 2 == 0, x + 1, x + 100) & 255
    assert np.array_equal(out, exp), list(out)


# --- address linearization -------------------------------------------------


@pytest.mark.parametrize("cols", [24, 16])
def test_a_coalesced_nest_addresses_with_the_bare_counter(cols):
    # `loop-canonicalization` coalesces the nest and delinearizes the subscripts
    # against the single counter (`A[iv floordiv N, iv mod N]`); the memref's
    # row-major linearization composes straight back to `iv`. That cancellation
    # must happen on the affine EXPRESSION: rebuilding it out of comb ops costs
    # a divider, a modulo and a multiplier per port (a shift pair when N is a
    # power of two) to recompute an index the counter already holds, and nothing
    # downstream can fold them away.
    @kernel
    def flat(A: i32[6, cols], out: i32[6, cols]):
        for i in range(6):
            for j in range(cols):
                out[i, j] = A[i, j] + 1

    mod = _to_rtl(flat)
    for op in ("comb.divu", "comb.modu", "comb.mul", "comb.shru"):
        assert op not in mod.mlir, f"{op} in the address path of a flat nest"

    A = (np.arange(6 * cols, dtype=np.int32) % 251).reshape(6, cols)
    out = np.zeros((6, cols), np.int32)
    mod.cosim(A, out)
    assert np.array_equal(out, A + 1)


def test_a_nest_whose_address_would_need_a_divider_is_not_coalesced():
    # The cancellation above holds only for an access that walks the nest's own
    # iteration space in order. Offset one subscript and the row-major fold no
    # longer composes back to the counter, so coalescing would leave a real
    # `floordiv 9` on the address of every port, at the full datapath width and
    # priced by nothing. `loop-canonicalization` gives the band back a level
    # instead: two counters, and an address that is affine in both.
    @kernel
    def offset(A: i32[6, 10], out: i32[6, 10]):
        for i in range(5):
            for j in range(9):
                out[i + 1, j + 1] = A[i, j]

    mod = _to_rtl(offset)
    for op in ("comb.divu", "comb.modu"):
        assert op not in mod.mlir, f"{op} in the address path of an offset nest"

    A = (np.arange(60, dtype=np.int32) % 251).reshape(6, 10)
    out = np.zeros((6, 10), np.int32)
    mod.cosim(A, out)
    exp = np.zeros((6, 10), np.int32)
    exp[1:6, 1:10] = A[0:5, 0:9]
    assert np.array_equal(out, exp)


def test_a_normalized_subscript_is_read_before_the_band_is_measured():
    # The same refusal, reached through a normalized loop. `s.unroll` leaves a
    # strided loop, and normalizing it moves the stride out of the bound and
    # into an `affine.apply` the subscript then names INSTEAD of the counter.
    # Until that is composed back into the map, the band is measured on an
    # expression its own induction variable is missing from: the recovered
    # `mod 5` never appears, the divider check clears the nest, and coalescing
    # puts a runtime `mod 5` on every port. Nine trips, so the divisor is not a
    # power of two and the mask that would hide it does not exist.
    @kernel
    def strided_gap(A: i32[18], out: i32[18]):
        for i in range(4):
            for j in range(0, 18, 2):
                out[j] = A[j] + i

    mod = _to_rtl(strided_gap)
    for op in ("comb.divu", "comb.modu"):
        assert op not in mod.mlir, f"{op} in the address path of a strided nest"

    A = (np.arange(18, dtype=np.int32) * 5 + 3) & 0xFF
    out = np.zeros(18, np.int32)
    mod.cosim(A, out)
    exp = np.zeros(18, np.int32)
    exp[0::2] = A[0::2] + 3
    assert np.array_equal(out, exp)


def test_a_partial_coalesced_subscript_keeps_its_address_arithmetic():
    # The counterpart: an inner loop the coalescing does not absorb leaves the
    # map with two live dims, so the row-major fold is a real shift/add rather
    # than the identity. It must still be emitted, and correctly.
    @kernel
    def part(A: i32[4, 8], out: i32[4]):
        for i in range(4):
            acc: i32 = 0
            for k in range(8):
                acc += A[i, k]
            out[i] = acc

    mod = _to_rtl(part)
    A = (np.arange(32, dtype=np.int32) % 13).reshape(4, 8)
    out = np.zeros(4, np.int32)
    mod.cosim(A, out)
    assert np.array_equal(out, A.sum(axis=1))


# --- ROM vs RAM classification -------------------------------------------


def test_constant_rom_cosim():
    # A constant-initialized local array lowers to a read-only ROM (an indexed
    # constant table) rather than a writable on-chip buffer: a byte table read
    # under a data-dependent index, and a wider (i32) table of non-power-of-two
    # length read by a scalar index.
    TBL = [10, 20, 30, 40, 50, 60, 70, 80]

    @kernel
    def table_lookup(A: u8[16], out: u8[16]):
        tbl: u8[8] = [10, 20, 30, 40, 50, 60, 70, 80]
        for i in range(16):
            idx: index = A[i] % 8
            out[i] = tbl[idx]

    m = _to_rtl(table_lookup)
    assert "hw.aggregate_constant" in m.mlir  # a ROM, not a writable hlmem
    A = np.arange(16, dtype=np.uint8) * 5 + 1
    out = np.zeros(16, np.uint8)
    m.cosim(A, out)
    assert np.array_equal(out, np.array(TBL, np.uint8)[A % 8])

    SQ = [i * i for i in range(12)]

    @kernel
    def square_table(x: i32, out: i32[1]):
        sq: i32[12] = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121]
        idx: index = x
        out[0] = sq[idx]

    m = _to_rtl(square_table)
    out = np.zeros(1, np.int32)
    for x in (0, 3, 7, 11):
        m.cosim(np.int32(x), out)
        assert out[0] == SQ[x]


def test_a_padded_constant_table():
    # The non-power-of-two ROM edge case: the `hw.aggregate_constant` needs
    # spare fields, and the padding must land past the real elements (a
    # hw.array indexes element 0 as its last field, so the initializer is
    # reversed). The indices stay variable on purpose: a literal index folds
    # to the element it names and drops the table outright.
    @kernel
    def padrom(A: i32[8], B: i32[8]):
        tbl: i32[3] = [77, 88, 99]
        for i in range(8):
            B[i] = tbl[A[i] % 3] + A[i]

    mod = _to_rtl(padrom)
    assert "aggregate_constant" in mod.mlir  # a table, not an hlmem
    B = np.zeros(8, np.int32)
    mod.cosim(A8, B)
    tbl = np.array([77, 88, 99], np.int32)
    assert np.array_equal(B, tbl[A8 % 3] + A8)


def test_constant_table_reads_are_unlimited_port():
    # Three table reads per iteration pipeline at II=1. A 2-port budget would
    # force II=2 (ceil(3/2)) for hardware that has no ports at all.
    @kernel
    def rom3(A: i32[8], B: i32[8]):
        tbl: i32[8] = [10, 20, 30, 40, 50, 60, 70, 80]
        for i in range(8):
            B[i] = tbl[A[i] % 8] + tbl[(A[i] + 1) % 8] + tbl[(A[i] + 2) % 8]

    mod = _to_rtl(rom3)
    # One combinational constant array, one array_get per access, no hlmem.
    assert mod.mlir.count("aggregate_constant") == 1
    assert len(re.findall(r"hw\.array_get", mod.mlir)) == 3
    assert "seq.hlmem" not in mod.mlir
    iis = _iis(mod.schedule().func("rom3").regions)
    assert iis == [1], "a constant table must not limit reads"

    A = np.array([0, 3, 5, 7, 1, 2, 6, 4], dtype=np.int32)
    B = np.zeros(8, np.int32)
    mod.cosim(A, B)
    t = np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=np.int32)
    assert np.array_equal(B, t[A % 8] + t[(A + 1) % 8] + t[(A + 2) % 8])


def test_a_constant_table_is_priced_as_the_logic_it_is_built_from():
    # A ROM holds no memory bits and occupies no storage row: it is emitted as a
    # constant array read by an index, so the part builds it out of LUTs. One
    # LUT6 is a 64-entry one-bit lookup, so a 64-deep table costs one per bit.
    table = (np.arange(64, dtype=np.int32) * 7) & 0xFF

    @kernel
    def rom(A: i32[8], B: i32[8]):
        tbl: i32[64] = table
        for i in range(8):
            B[i] = tbl[A[i] % 64]

    rtl = _to_rtl(rom)
    rtl.compile()
    mem = next(
        m
        for f in rtl.report.microarch.funcs
        for m in f.mems
        if m.owner.startswith("tbl")
    )
    assert mem.realization == "rom"
    est = qor.estimate(rtl.report)
    assert est.mem_bits == 0, "a constant table is logic, not memory"
    # One LUT6 per bit of the 64-deep table, quoted after LUT packing.
    assert est.by_kind["memories"].lut == round(32 * rtl.device.lut_packing)

    # Binding the array to a block RAM overrides the table: it becomes a real
    # memory that powers on with the same contents.
    s = rom.schedule()
    s.bind_storage("tbl", impl=Schedule.BRAM, mem_type=s.RAM_T2P)
    rtl = s.export("rtl")
    rtl.compile()
    assert qor.estimate(rtl.report).area.bram36 == 1
    B = np.zeros(8, np.int32)
    rtl.cosim(A8, B)
    assert np.array_equal(B, table[A8 % 64])


def test_a_table_too_deep_to_read_quickly_becomes_a_memory():
    # A table's read is a LUT cone that grows with the depth, while an
    # addressed memory's read delay is flat. Past the depth at which the
    # memory reads faster, the array is realized as that memory, powering on
    # with the same contents.
    def table(depth):
        vals = ((np.arange(depth, dtype=np.int32) * 2654435) >> 3) & 0xFFFF

        @kernel
        def look(A: i32[8], B: i32[8]):
            tbl: i32[depth] = vals
            for i in range(8):
                idx: index = A[i] % depth
                B[i] = tbl[idx]

        m = _to_rtl(look)
        mem = next(
            mm
            for f in m.report.microarch.funcs
            for mm in f.mems
            if mm.depth_words == depth
        )
        mlir = m.mlir
        B = np.zeros(8, np.int32)
        m.cosim(A8, B)
        assert np.array_equal(B, vals[A8 % depth]), (depth, list(B))
        return mem.realization, mem.storage, mlir

    shallow, shallow_row, shallow_mlir = table(256)
    assert (shallow, shallow_row) == ("rom", "rom")
    assert "hw.aggregate_constant" in shallow_mlir

    deep, deep_row, deep_mlir = table(1024)
    assert (deep, deep_row) == ("ram", "bram")
    # The memory comes up holding the table: the contents reach the backing
    # register as an `initial` block.
    assert "hw.aggregate_constant" not in deep_mlir
    assert "seq.hlmem @tbl" in deep_mlir


def test_binding_a_written_array_to_the_table_row_is_refused():
    # The table row has compile-time contents and no write port, so an array
    # that is stored to cannot be bound there.
    @kernel
    def rmw(A: i32[8], B: i32[8]):
        tbl: i32[8] = [1, 2, 3, 4, 5, 6, 7, 8]
        for i in range(8):
            tbl[i] = tbl[i] + A[i]
        for i in range(8):
            B[i] = tbl[i]

    s = rmw.schedule()
    s.bind_storage("tbl", impl=Schedule.ROM, mem_type=s.RAM_1P)
    with pytest.raises(RuntimeError):
        s.export("rtl").schedule()


def test_a_table_a_sub_kernel_only_reads_is_still_a_table():
    # Read-only is a property of the use, and a sub-kernel is one of the users:
    # only a sweep holding the callee's port directions can tell that a child
    # merely reads an array and leave it a combinational table.
    tbl_vals = (np.arange(8, dtype=np.int32) * 9) & 0xFF

    @kernel
    def rom_reader(tbl: i32[8], A: i32[8], B: i32[8]):
        for i in range(8):
            B[i] = tbl[A[i] % 8] + A[i]

    @kernel
    def rom_caller(A: i32[8], B: i32[8]):
        tbl: i32[8] = tbl_vals
        rom_reader(tbl, A, B)

    rtl = _to_rtl(rom_caller)
    rtl.compile()
    # The caller's copy: to the callee the array is a boundary argument, whose
    # cells belong to whoever passed it.
    caller = next(f for f in rtl.report.microarch.funcs if f.func == "rom_caller")
    mem = next(m for m in caller.mems if m.owner.startswith("tbl"))
    assert mem.realization == "rom", "a child that only reads leaves it a table"
    assert mem.cost.call_reads == 1 and mem.cost.call_writes == 0
    assert "seq.hlmem" not in rtl.mlir, "a table needs no writable storage"

    B = np.zeros(8, np.int32)
    rtl.cosim(A8, B)
    assert np.array_equal(B, tbl_vals[A8 % 8] + A8)


def test_a_design_asking_for_more_of_a_resource_than_the_part_has_is_named():
    @kernel
    def k(A: i32[64], out: i32[64]):
        buf: i32[64]
        for i in range(64):
            buf[i] = A[i] + 1
        for i in range(64):
            out[i] = buf[i] & 255

    rtl = _to_rtl(k)
    rtl.compile()
    est = qor.estimate(rtl.report)
    assert (
        est.utilization["lut"]
        == est.area.lut / default_device.resources["lut"].capacity
    )
    assert not est.over_capacity, "this fits a u55c many times over"

    # The same design against a part with a handful of LUTs: the LUT rows go
    # over capacity and nothing else does.
    small = default_device.copy()
    for name in ("lut", "slicem_lut"):
        small.resources[name] = Resource(name, 8)
    over = qor.estimate(rtl.report, small).over_capacity
    assert set(over) == {"lut", "slicem_lut"}
    assert over["lut"] == est.area.lut / 8


def test_a_banked_array_cannot_be_declared_with_contents():
    # Declared contents are realized as one bank, so a banking partition on such
    # an array is refused. A complete partition is not banking and stays legal.
    table = (np.arange(16, dtype=np.int32) * 7) & 0xFF

    @kernel
    def tbl16(A: i32[16], B: i32[16]):
        tbl: i32[16] = table
        for i in range(16):
            B[i] = tbl[i] + A[i]

    s = tbl16.schedule()
    s.partition("tbl", kind=Schedule.Cyclic, dim=1, factor=2)
    with pytest.raises(RuntimeError):
        s.export("rtl").schedule()

    s = tbl16.schedule()
    s.partition("tbl")
    s.export("rtl").schedule()


def test_written_array_keeps_its_port_limit():
    # The contrast that keeps the ROM grant narrow: the SAME three reads off an
    # array the kernel writes are still bound by its two ports (II=2). Read-only
    # is a property of the use, so writing it once makes it a real memory.
    @kernel
    def ram3(A: i32[8], B: i32[8]):
        t: i32[8]
        for i in range(8):
            t[i] = A[i]
        for i in range(8):
            B[i] = t[A[i] % 8] + t[(A[i] + 1) % 8] + t[(A[i] + 2) % 8]

    # Ports are only a limit while the array is a memory, so the automatic
    # partition that would scatter this one into registers is off.
    res = _to_rtl(ram3).set_scheduler_opt(scalarize_threshold=0).schedule()
    iis = _iis(res.func("ram3").regions)
    assert iis == [1, 2], f"a written array must keep its port limit, got {iis}"


def test_read_only_initialized_array_is_a_rom():
    # The classification is on the USE, so a never-written initialized array
    # keeps its combinational constant-table realization.
    @kernel
    def lookup(A: i32[8], B: i32[8]):
        tbl: i32[8] = [10, 20, 30, 40, 50, 60, 70, 80]
        for i in range(8):
            B[i] = tbl[i] + A[i]

    m = _to_rtl(lookup)
    assert "hw.aggregate_constant" in m.mlir
    assert "seq.hlmem" not in m.mlir
    B = np.zeros(8, dtype=np.int32)
    m.cosim(A8, B)
    assert np.array_equal(B, np.arange(1, 9, dtype=np.int32) * 10 + A8)


@pytest.mark.parametrize("decl", ["const", "stateful"])
def test_initialized_and_written_array(decl):
    # The same array written even once is not a constant table: it needs a real
    # write port AND the declared contents as power-on state (an `initial` block
    # over the backing storage). Both declaration forms that carry a
    # compile-time initializer -- a list-initialized local and `Stateful` --
    # realize identically.
    if decl == "const":

        @kernel
        def rmw(A: i32[8], B: i32[8]):
            tbl: i32[8] = [1, 2, 3, 4, 5, 6, 7, 8]
            for i in range(8):
                tbl[i] = tbl[i] + A[i]
            for i in range(8):
                B[i] = tbl[i]

    else:

        @kernel
        def rmw(A: i32[8], B: i32[8]):
            tbl: Stateful[i32[8]] = [1, 2, 3, 4, 5, 6, 7, 8]
            for i in range(8):
                tbl[i] = tbl[i] + A[i]
            for i in range(8):
                B[i] = tbl[i]

    m = _to_rtl(rmw)
    # A writable memory, not the ROM a read-only table would give.
    assert "seq.hlmem @tbl" in m.mlir and "hw.aggregate_constant" not in m.mlir
    B = np.zeros(8, np.int32)
    m.cosim(A8, B)
    assert np.array_equal(B, np.arange(1, 9, dtype=np.int32) + A8), list(B)


def test_initialized_and_written_scalar():
    # A `Stateful` scalar is the same case at depth 1: a rank-0 memref, whose
    # single element addresses at 0 with no subscript at all. It must not slip
    # through as a ROM (which would drop every store) or as an uninitialized
    # register (which would start at X).
    @kernel
    def counter(A: i32[8], B: i32[8]):
        acc: Stateful[i32] = 100
        for i in range(8):
            acc = acc + A[i]
        for i in range(8):
            B[i] = acc

    m = _to_rtl(counter)
    B = np.zeros(8, np.int32)
    m.cosim(A8, B)
    assert np.all(B == 100 + A8.sum()), list(B)


@pytest.mark.parametrize("written", [False, True])
def test_initialized_float_array(written):
    # A float table's contents are its element bit patterns, the same
    # convention the datapath carries every float by. The constant-table and
    # written-memory forms share one conversion, so they cannot disagree on
    # what the declared values are.
    if written:

        @kernel
        def scale(A: f32[8], B: f32[8]):
            tbl: f32[8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
            for i in range(8):
                tbl[i] = tbl[i] * A[i]
            for i in range(8):
                B[i] = tbl[i]

    else:

        @kernel
        def scale(A: f32[8], B: f32[8]):
            tbl: f32[8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
            for i in range(8):
                B[i] = tbl[i] * A[i]

    m = _to_rtl(scale)
    assert ("seq.hlmem @tbl" in m.mlir) == written
    Af = np.arange(1, 9, dtype=np.float32)
    B = np.zeros(8, dtype=np.float32)
    m.cosim(Af, B)
    assert np.allclose(B, np.arange(1, 9, dtype=np.float32) * Af), list(B)


@pytest.mark.parametrize("child", ["reads", "writes"])
def test_initialized_array_handed_to_a_sub_kernel(child):
    # Read-only is a property of the USE, calls included. A sub-kernel that
    # WRITES the table needs a real memory that merely starts with its
    # contents; a table every child only READS stays a constant array, the
    # parent serving the child's address off the aggregate.
    if child == "reads":

        @kernel
        def use(t: i32[8], A: i32[8], B: i32[8]):
            for i in range(8):
                B[i] = t[i] + A[i]

        @kernel
        def top(A: i32[8], B: i32[8]):
            tbl: i32[8] = [1, 2, 3, 4, 5, 6, 7, 8]
            use(tbl, A, B)

    else:

        @kernel
        def bump(t: i32[8], A: i32[8]):
            for i in range(8):
                t[i] = t[i] + A[i]

        @kernel
        def top(A: i32[8], B: i32[8]):
            tbl: i32[8] = [1, 2, 3, 4, 5, 6, 7, 8]
            bump(tbl, A)
            for i in range(8):
                B[i] = tbl[i]

    m = _to_rtl(top)
    written = child == "writes"
    assert ("seq.hlmem @tbl" in m.mlir) == written, m.mlir
    assert ("hw.aggregate_constant" in m.mlir) != written, m.mlir
    B = np.zeros(8, np.int32)
    m.cosim(A8, B)
    assert np.array_equal(B, np.arange(1, 9, dtype=np.int32) + A8), list(B)


# --- storage realizations the device declares -------------------------------


# A storage realization is not a resource: a resource is a counter the compiler
# adds up, and a realization is something the device BUILDS out of counters,
# with timing of its own. That split is why the vocabulary is open, and an
# `impl=` resolves against it BY NAME.
def test_a_device_can_declare_a_storage_of_its_own():
    # A name the compiler has never heard of resolves all the same: `bind_storage`
    # maps `y` onto the device's own `mram` row, which every kernel then reads at
    # the latency the device declared for it.
    @kernel
    def mv(A: f32[8, 8], x: f32[8], out: f32[8]):
        y: f32[8] = 0
        for i in range(8):
            for k in range(8):
                y[i] += A[i, k] * x[k]
        for i in range(8):
            out[i] = y[i]

    dev = default_device.copy()
    mram = dev.add_storage(
        "mram",
        read_latency=2,
        write_latency=1,
        read_delay_ns=0.5,
        write_delay_ns=0.5,
    )
    s = mv.schedule()
    s.bind_storage("y", impl=mram, mem_type=s.RAM_T2P)
    rtl = s.export("rtl", device=dev).set_scheduler_opt(scalarize_threshold=0)
    rtl.compile()
    mems = _shared_mems(rtl, "y")
    assert mems and all(m.storage == "mram" and m.read_latency == 2 for m in mems)


def test_binding_storage_to_a_resource_is_a_type_error():
    # `@lut` is a counter, not a place an array can live. Resources and storage
    # realizations being different types is what makes that a type error at the
    # call rather than a name that fails to resolve at export.
    @kernel
    def k(A: i32[8], out: i32[8]):
        for i in range(8):
            out[i] = A[i] + 1

    lut = default_device.resources["lut"]
    s = k.schedule()
    with pytest.raises(InvalidScheduleArgumentError):
        s.bind_storage("A", impl=lut, mem_type=s.RAM_T2P)
    with pytest.raises(TypeError):
        default_device.copy().set_default_storage(lut)


def test_an_undeclared_storage_is_reported():
    # An `impl=` the device declares no row for would fall to zero timing and
    # schedule combinationally, reading before the data is valid.
    @kernel
    def k(A: i32[8], out: i32[8]):
        for i in range(8):
            out[i] = A[i] + 1

    dev = default_device.copy()
    del dev.storage["uram"]
    s = k.schedule()
    s.bind_storage("A", impl=Schedule.URAM, mem_type=s.RAM_T2P)
    with pytest.raises(RuntimeError):
        s.export("rtl", device=dev).schedule()


def test_a_storage_that_powers_up_undefined_cannot_hold_declared_contents():
    # An UltraRAM powers up with no contents, so an array that must start
    # holding them cannot be one.
    @kernel
    def rmw(A: i32[8], B: i32[8]):
        tbl: i32[8] = [1, 2, 3, 4, 5, 6, 7, 8]
        for i in range(8):
            tbl[i] = tbl[i] + A[i]
        for i in range(8):
            B[i] = tbl[i]

    s = rmw.schedule()
    s.bind_storage("tbl", impl=Schedule.URAM, mem_type=s.RAM_T2P)
    with pytest.raises(RuntimeError):
        s.export("rtl").schedule()

    # A read-only table is refused too: an explicit `impl=` names the structure
    # to build, and this one cannot hold declared contents. Unbound, the same
    # table is realized as logic.
    @kernel
    def ro(A: i32[8], B: i32[8]):
        tbl: i32[8] = [1, 2, 3, 4, 5, 6, 7, 8]
        for i in range(8):
            B[i] = tbl[i] + A[i]

    s = ro.schedule()
    s.bind_storage("tbl", impl=Schedule.URAM, mem_type=s.RAM_T2P)
    with pytest.raises(RuntimeError):
        s.export("rtl").schedule()
    assert "hw.aggregate_constant" in _to_rtl(ro).mlir


@kernel
def sh_prod(A: i32[64], buf: i32[64]):
    for i in range(64):
        buf[i] = A[i] + 1


@kernel
def sh_cons(buf: i32[64], out: i32[64]):
    for i in range(64):
        out[i] = buf[i] & 255


@kernel
def sh_top(A: i32[64], out: i32[64]):
    buf: i32[64]
    sh_prod(A, buf)
    sh_cons(buf, out)


def _shared_mems(rtl, owner):
    """Every kernel's view of the one array named `owner`."""
    return [
        m
        for f in rtl.report.microarch.funcs
        for m in f.mems
        if m.owner.rstrip("_") == owner
    ]


def test_a_storage_binding_reaches_every_kernel_the_array_is_visible_in():
    # `bind_storage` is stated once where the array lives, and a callee sees
    # only its own parameter, so a child left on the default row would read a
    # 2-cycle UltraRAM at 1-cycle timing and every element would come back
    # shifted.
    s = sh_top.schedule()
    s.bind_storage("buf", impl=Schedule.URAM, mem_type=s.RAM_T2P)
    rtl = s.export("rtl")
    rtl.compile()
    mems = _shared_mems(rtl, "buf")
    assert len(mems) == 3, "the owner and both children name it"
    assert all(m.storage == "uram" and m.read_latency == 2 for m in mems)
    A = np.arange(64, dtype=np.int32)
    out = np.zeros(64, np.int32)
    rtl.cosim(A, out)
    assert np.array_equal(out, (A + 1) & 255)

    # A boundary array is the same: its cells are the caller's, but the latency
    # the two sides read them at is one number.
    @kernel
    def sh_arg_top(A: i32[64], out: i32[64]):
        sh_cons(A, out)

    s = sh_arg_top.schedule()
    s.bind_storage("A", impl=Schedule.URAM, mem_type=s.RAM_T2P)
    rtl = s.export("rtl")
    rtl.compile()
    shared = _shared_mems(rtl, "A") + _shared_mems(rtl, "buf")
    assert len(shared) == 2 and all(m.read_latency == 2 for m in shared)
    out = np.zeros(64, np.int32)
    rtl.cosim(A, out)
    assert np.array_equal(out, A & 255)


def test_two_kernels_binding_one_array_to_different_structures_is_refused():
    # Refused on the `impl` axis whatever the timing says: `bram` and `lutram`
    # are both 1-cycle here and still different hardware.
    s = sh_top.schedule()
    cs = sh_cons.schedule()
    cs.bind_storage("buf", impl=Schedule.BRAM, mem_type=cs.RAM_T2P)
    s.compose(cs)
    s.bind_storage("buf", impl=Schedule.LUTRAM, mem_type=s.RAM_T2P)
    with pytest.raises(RuntimeError, match="ALLO-E0018"):
        s.export("rtl").schedule()


def test_the_port_topology_two_kernels_ask_for_is_covered_not_matched():
    # The three topologies form a chain: t2p serves everything s2p does, and s2p
    # everything 1p does, so the array takes the one that covers both kernels.
    def buf_ports(build):
        s = sh_top.schedule()
        build(s)
        rtl = s.export("rtl")
        rtl.compile()
        mems = _shared_mems(rtl, "buf")
        assert len(mems) == 3 and all(m.storage == "bram" for m in mems)
        return {(m.cost.row_reads, m.cost.row_writes) for m in mems}, rtl

    def mixed(s):
        cs = sh_cons.schedule()
        cs.bind_storage("buf", impl=Schedule.BRAM, mem_type=cs.RAM_1P)
        s.compose(cs)
        s.bind_storage("buf", impl=Schedule.BRAM, mem_type=s.RAM_T2P)

    # The block RAM's 2 read / 2 write reaches every kernel: t2p covers the
    # child's 1p, so the 1p narrows nothing.
    ports, rtl = buf_ports(mixed)
    assert ports == {(2, 2)}
    A = np.arange(64, dtype=np.int32)
    out = np.zeros(64, np.int32)
    rtl.cosim(A, out)
    assert np.array_equal(out, (A + 1) & 255)

    # 1p stated on the owner does narrow it.
    only_1p, _ = buf_ports(
        lambda s: s.bind_storage("buf", impl=Schedule.BRAM, mem_type=s.RAM_1P)
    )
    assert only_1p == {(1, 1)}


def test_a_multi_cycle_read_registers_the_datum_not_the_address():
    # A 2-cycle read is a port that reads in one cycle and holds the datum in
    # the output register a block RAM and an UltraRAM have. Registering the
    # address instead lands the same datum on the same cycle but throws that
    # register away.
    @kernel
    def k(A: i32[32], out: i32[32]):
        buf: i32[32] = 0
        for i in range(32):
            buf[i] = A[i] + 1
        for i in range(32):
            out[i] = buf[i]

    s = k.schedule()
    s.bind_storage("buf", impl=Schedule.URAM, mem_type=s.RAM_T2P)
    rtl = s.export("rtl")
    rtl.compile()
    v = rtl.verilog
    assert re.search(r"buf__rd0_reg <= buf_\[\w+\];", v), "the port reads in one"
    assert re.search(r"buf__rd0_dly1 <= buf__rd0_reg;", v), "the datum is held"
    assert "_rdaddr" not in v, "no address delay stage"
    assert '(* ram_style = "ultra" *)' in v

    out = np.zeros(32, np.int32)
    A = np.arange(32, dtype=np.int32)
    rtl.cosim(A, out)
    assert np.array_equal(out, A + 1)


def test_a_complete_partition_conflicting_with_a_bind_is_reported():
    # Layout and realization are different axes, but a complete partition
    # scatters the array into flip-flops whatever `impl=` asked for, so the two
    # directives have one silent winner unless it is reported. Agreeing is not
    # a conflict: `impl=` naming the register row states what the partition
    # already implies.
    @kernel
    def k(A: i32[8], out: i32[8]):
        for i in range(8):
            out[i] = A[i] + 1

    s = k.schedule()
    s.partition("A", kind=s.Complete)
    s.bind_storage("A", impl=Schedule.BRAM, mem_type=s.RAM_T2P)
    with pytest.raises(RuntimeError):
        s.export("rtl").schedule()

    agree = k.schedule()
    agree.partition("A", kind=agree.Complete)
    agree.bind_storage(
        "A", impl=default_device.storage["register"], mem_type=agree.RAM_T2P
    )
    agree.export("rtl").schedule()


def test_the_device_names_the_storage_a_scatter_goes_into():
    # The compiler spells no storage name of its own. A complete partition
    # resolves to whichever row the DEVICE marked `is_scatter`, so a part whose
    # flip-flops go by another name marks that one and nothing in the tree
    # switches on the list; a part that has none cannot hold a scatter at all.
    @kernel
    def k(A: i32[8], out: i32[8]):
        for i in range(8):
            out[i] = A[i] + 1

    renamed = default_device.copy()
    del renamed.storage["register"]
    renamed.add_storage(
        "ff_cell",
        read_latency=0,
        write_latency=1,
        read_delay_ns=0.1,
        write_delay_ns=0.1,
        is_scatter=True,
    )
    s = k.schedule()
    s.partition("A", kind=s.Complete)
    s.export("rtl", device=renamed).schedule()

    bare = default_device.copy()
    del bare.storage["register"]
    s = k.schedule()
    s.partition("A", kind=s.Complete)
    with pytest.raises(RuntimeError):
        s.export("rtl", device=bare).schedule()

    with pytest.raises(ValueError):
        default_device.copy().add_storage(
            "another",
            read_latency=0,
            write_latency=1,
            is_scatter=True,
        )


def test_a_scatter_row_declares_no_port_limit():
    # One cell per element is not addressed, so a `scatter` row has no port to
    # limit.
    for limit in ({"inst_reads": 1}, {"inst_writes": 1}):
        dev = default_device.copy()
        del dev.storage["register"]
        with pytest.raises(ValueError):
            dev.add_storage(
                "ff_cell",
                read_latency=0,
                write_latency=1,
                is_scatter=True,
                **limit,
            )


def test_reads_share_the_ports_the_row_has_rather_than_taking_one_each():
    # Six reads of one array in one region: the scheduler bills the row's two
    # read ports and staggers the reads across cycles, and the binding puts them
    # on those two ports rather than one each.
    @kernel
    def k(A: i32[32], out: i32[8]):
        buf: i32[32] = 0
        for i in range(32):
            buf[i] = A[i]
        for j in range(8):
            out[j] = buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5]

    def buf_of(rtl):
        rtl.compile()
        mem = next(
            m
            for f in rtl.report.microarch.funcs
            for m in f.mems
            if m.owner.startswith("buf")
        )
        return mem, rtl

    mem, _ = buf_of(_to_rtl(k))
    assert mem.storage == "lutram"
    assert mem.reads == 6 and mem.cost.read_ports == 2
    # A LUT RAM's write port and its replicated reads are separate structures,
    # so its three ports really are three and it has all three.
    assert mem.cost.ports == 3 and mem.realization == "ram"

    # A block RAM's two ports are one pool, each serving a read or a write in a
    # cycle. The fill loop and the drain loop never run together, so the write
    # rides a read's port: two ports over three accesses, and one block RAM.
    s = k.schedule()
    s.bind_storage("buf", impl=Schedule.BRAM, mem_type=s.RAM_T2P)
    mem, rtl = buf_of(s.export("rtl"))
    assert mem.storage == "bram"
    assert mem.cost.read_ports == 2 and mem.cost.write_ports == 1
    assert mem.cost.ports == 2 and mem.realization == "ram"
    assert '(* ram_style = "block" *)' in rtl.verilog
    assert qor.estimate(rtl.report).area.bram36 == 1


def test_a_row_with_one_write_port_is_scheduled_to_one_rather_than_overrun():
    # Two stores per iteration against a row with a single write port: the
    # scheduler bills that port, so the stores land on different cycles (II 2)
    # and the binding puts them both on it. Deeper than the auto-partition
    # threshold, since a completely partitioned array is realized as registers
    # and never reaches a port at all.
    @kernel
    def k(A: i32[32], out: i32[32]):
        buf: i32[32] = 0
        for i in range(16):
            buf[2 * i] = A[2 * i] + 1
            buf[2 * i + 1] = A[2 * i + 1] + 2
        for i in range(32):
            out[i] = buf[i]

    def buf_of(rtl):
        rtl.compile()
        m = next(
            m
            for f in rtl.report.microarch.funcs
            for m in f.mems
            if m.owner.startswith("buf")
        )
        return m, qor.estimate(rtl.report), rtl

    mem, est, rtl = buf_of(_to_rtl(k))
    assert mem.storage == "lutram"
    assert mem.writes == 2 and mem.cost.write_ports == 1
    assert mem.realization == "ram" and est.mem_bits == 32 * 32
    # One port carries both stores, so nothing claims they are independent.
    assert not mem.writes_independent and "independent" not in rtl.mlir

    # Two write ports where the row has two: each gets its own `always` block,
    # which is what infers a true dual port. The drain loop's read rides one of
    # them, so the three accesses still fit the two ports the row has.
    s = k.schedule()
    s.bind_storage("buf", impl=Schedule.BRAM, mem_type=s.RAM_T2P)
    mem, est, rtl = buf_of(s.export("rtl"))
    assert mem.storage == "bram"
    assert mem.cost.write_ports == 2 and mem.writes_independent
    assert mem.cost.ports == 2 and mem.realization == "ram"
    assert "independent" in rtl.mlir and est.area.bram36 == 1
    # The port a read and a write share addresses the array once, in one
    # `always` block: a dual-port RAM has to see two address buses over the
    # three accesses, and three would infer nothing.
    assert re.search(
        r"buf_\[(\w+)\] <= \w+;.*\n.*buf__rd0_reg <= buf_\[\1\]", rtl.verilog
    ), "the shared port's write and read take one address"


def test_a_tiled_cost_prices_the_whole_shape():
    # `tiled` is the one cost form reading the WHOLE parameter tuple: a block
    # RAM tile holds 36864 bits however a depth-by-width array is cut, so the
    # product sits inside the ceiling and does not separate into one factor per
    # parameter the way every other form does.
    @kernel
    def k(A: i32[8], out: i32[8]):
        for i in range(8):
            out[i] = A[i] + 1

    dev = default_device.copy()
    dev.add_storage(
        "bram",
        read_latency=1,
        write_latency=1,
        read_delay_ns=0.7,
        write_delay_ns=0.7,
        uses={dev.resources["bram36"]: Tiled(36864)},
    )
    text = _to_rtl(k, device=dev).dcp
    assert "allo.dcp.storage @bram" in text
    assert "@bram36" in text and "tiled" in text


# --- multi-cycle access latency -------------------------------------------


def _uram_buffer_rtl(impl):
    """A producer/consumer pair through an internal buffer, optionally bound to
    a storage impl. The consumer reads `buf` inside an II=1 pipeline, so a read
    port built at the wrong latency shifts the result by one iteration."""

    @kernel
    def urambuf(A: i32[16], out: i32[16]):
        buf: i32[16] = 0
        for i in range(16):
            buf[i] = A[i] * 3
        for i in range(16):
            out[i] = buf[i] + 1

    s = urambuf.schedule()
    if impl is not None:
        s.bind_storage("buf", impl=impl, mem_type=s.RAM_T2P)
    # `buf` is small enough to be complete-partitioned by default, which would
    # replace the read port whose latency is the subject here.
    return s.export("rtl").set_scheduler_opt(scalarize_threshold=0)


def test_multicycle_storage_read_cosim():
    # The emitted read port must be built at the memory's DEVICE read latency,
    # not a hardcoded 1: URAM reads in 2 cycles, and the scheduler places the
    # consumer accordingly. The extra read cycle shows up in the whole-kernel
    # latency.
    def reader_depth(result):
        # Region 1 is the consumer loop, the one whose load carries the port.
        region = next(x for x in result.func("urambuf").regions if x.order == 1)
        return region.iteration_latency

    exp = A16 * 3 + 1
    out_default = np.zeros(16, np.int32)
    r = _uram_buffer_rtl(None)
    depth_default = reader_depth(r.schedule())
    r.cosim(A16, out_default)
    np.testing.assert_array_equal(out_default, exp)

    out_uram = np.zeros(16, np.int32)
    r = _uram_buffer_rtl(Schedule.URAM)
    depth_uram = reader_depth(r.schedule())
    r.cosim(A16, out_uram)
    np.testing.assert_array_equal(out_uram, exp)
    assert depth_uram - depth_default == MEM_URAM - MEM


def test_multicycle_storage_on_argument_cosim():
    # A boundary array's port latency is a contract with the driver: the
    # emitted RTL expects the read datum `latency` cycles after the address,
    # with no delay elements of its own. That number rides the interface
    # manifest and the cosim harness honors it, so a multi-cycle ARGUMENT is
    # emittable and the extra cycle shows up as whole-kernel latency.
    def argmem_rtl(impl):
        @kernel
        def argmem(A: i32[16], out: i32[16]):
            for i in range(16):
                out[i] = A[i] + 1

        s = argmem.schedule()
        if impl is not None:
            s.bind_storage("A", impl=impl, mem_type=s.RAM_T2P)
        return s.export("rtl")

    exp = A16 + 1
    out_default = np.zeros(16, np.int32)
    argmem_rtl(None).cosim(A16, out_default)
    np.testing.assert_array_equal(out_default, exp)

    out_uram = np.zeros(16, np.int32)
    argmem_rtl(Schedule.URAM).cosim(A16, out_uram)
    np.testing.assert_array_equal(out_uram, exp)

    # The contract must be stated in the manifest, not just honored by luck:
    # the URAM argument's read ports declare 2 cycles, the 1-cycle default 1.
    def read_latencies(rtl):
        iface = rtl.interfaces["argmem"]
        return {p.latency for acc in iface.reads for p in acc}

    assert read_latencies(argmem_rtl(Schedule.URAM)) == {2}
    assert read_latencies(argmem_rtl(None)) == {1}


def _dev(write_latency: int):
    """The default device with the default on-chip storage rebound to a
    ``write_latency``-cycle write."""
    d = default_device.copy()
    d.set_default_storage(
        d.add_storage(
            "lutram",
            read_latency=1,
            write_latency=write_latency,
            read_delay_ns=0.5,
            write_delay_ns=0.5,
        )
    )
    return d


def test_internal_array_multi_cycle_write():
    # An on-chip buffer bound to a 2- and 3-cycle write. Both a plain
    # producer/consumer hand-off (the write must land before the next region
    # reads it) and a same-address accumulate (the recurrence's II is read +
    # add + write, so a mistimed write reads back a stale partial) are covered.
    @kernel
    def xfer(A: i32[8], B: i32[8]):
        buf: i32[8] = 0
        for i in range(8):
            buf[i] = A[i] * 2
        for i in range(8):
            B[i] = buf[i] + 1

    @kernel
    def accumulate(A: i32[8], B: i32[8]):
        s: i32[8] = 0
        for i in range(8):
            s[0] = s[0] + A[i]
        for i in range(8):
            B[i] = s[0]

    for wr in (1, 2, 3):
        B = np.zeros(8, dtype=np.int32)
        _to_rtl(xfer, device=_dev(wr)).cosim(A8, B)
        assert np.array_equal(B, A8 * 2 + 1), f"wr_lat={wr}: {list(B)}"

        C = np.zeros(8, dtype=np.int32)
        _to_rtl(accumulate, device=_dev(wr)).cosim(A8, C)
        assert np.all(C == A8.sum()), f"wr_lat={wr}: {list(C)}"


def test_multi_cycle_write_through_sub_kernel_call():
    # A buffer whose write port is mastered by a child kernel takes the same
    # pipelining: the parent drives the port, so it owes the child's write the
    # same delay as one of its own.
    @kernel
    def fill(b: i32[8], A: i32[8]):
        for i in range(8):
            b[i] = A[i] * 5

    @kernel
    def top(A: i32[8], B: i32[8]):
        buf: i32[8] = 0
        fill(buf, A)
        for i in range(8):
            B[i] = buf[i] + 2

    for wr in (1, 2, 3):
        B = np.zeros(8, dtype=np.int32)
        _to_rtl(top, device=_dev(wr)).cosim(A8, B)
        assert np.array_equal(B, A8 * 5 + 2), f"wr_lat={wr}: {list(B)}"


# --- container-local storage ----------------------------------------------


def test_a_container_local_buffer_is_on_chip_storage():
    # A buffer declared in a dataflow container is storage the top OWNS, not a
    # port it forwards: one `seq.hlmem` and one port per accessing process,
    # driven straight from that child's addr/data/we, invisible at the
    # boundary interface.
    @kernel
    async def ia_prod(s: Stream[i32]):
        for i in range(N):
            s.put(i * 2)

    @kernel
    async def ia_cons(s: Stream[i32], tmp: i32[N]):
        for i in range(N):
            tmp[i] = s.get() + 1

    @kernel
    def ia_post(tmp: i32[N], out: i32[N]):
        for i in range(N):
            out[i] = tmp[i] * 3

    @kernel
    async def ia_top(out: i32[N]):
        f: Stream[i32]
        tmp: i32[N]  # declared HERE, not an argument
        await ia_prod(f)
        await ia_cons(f, tmp)
        ia_post(tmp, out)

    mod = _to_rtl(ia_top)
    assert "seq.hlmem @tmp" in mod.mlir
    # The buffer is internal: it must not show up as a boundary interface.
    top = mod.interfaces[mod.top]
    assert not any(
        m.base.startswith("tmp") for acc in top.reads + top.writes for m in acc
    ), top
    out = np.zeros(N, np.int32)
    mod.cosim(out)
    assert np.array_equal(out, (np.arange(N) * 2 + 1) * 3), list(out)


def test_two_sync_processes_share_a_container_local_buffer():
    # Both accessors are determinate, so neither takes the `done` handshake:
    # the reader fires at the static offset the scheduler placed it at, past
    # the writer's latency. That ordering is the schedule's to make, so the
    # emitter's whole-array gate is inert here.
    @kernel
    async def q_prod(s: Stream[i32]):
        for i in range(N):
            s.put(i * 2)

    @kernel
    async def q_cons(s: Stream[i32], o: i32[N]):
        for i in range(N):
            o[i] = s.get()

    @kernel
    def q_w(b: i32[N]):
        for i in range(N):
            b[i] = i * 7

    @kernel
    def q_r(b: i32[N], o2: i32[N]):
        for i in range(N):
            o2[i] = b[i] + 1

    @kernel
    async def q_top(o: i32[N], o2: i32[N]):
        f: Stream[i32]
        tmp: i32[N]
        await q_prod(f)
        await q_cons(f, o)
        q_w(tmp)
        q_r(tmp, o2)

    mod = _to_rtl(q_top)
    o = np.zeros(N, np.int32)
    o2 = np.zeros(N, np.int32)
    mod.cosim(o, o2)
    assert np.array_equal(o, np.arange(N) * 2), list(o)
    assert np.array_equal(o2, np.arange(N) * 7 + 1), list(o2)


def test_two_processes_read_one_container_local_buffer_concurrently():
    # Two readers do not hazard, so nothing orders them and they run together
    # on ports of their own: each accessor gets its own port instead of sharing
    # an arbitrated one, since a mux would time-share exactly the pair that is
    # safe to run in parallel.
    @kernel
    async def tr_fill(s: Stream[i32], tmp: i32[N]):
        for i in range(N):
            tmp[i] = i * 2
        s.put(1)

    @kernel
    async def tr_sink(s: Stream[i32], o0: i32[1]):
        o0[0] = s.get()

    @kernel
    def tr_a(tmp: i32[N], o1: i32[N]):
        for i in range(N):
            o1[i] = tmp[i] + 1

    @kernel
    def tr_b(tmp: i32[N], o2: i32[N]):
        for i in range(N):
            o2[i] = tmp[i] + 100

    @kernel
    async def tr_top(o0: i32[1], o1: i32[N], o2: i32[N]):
        f: Stream[i32]
        tmp: i32[N]
        await tr_fill(f, tmp)
        await tr_sink(f, o0)
        tr_a(tmp, o1)
        tr_b(tmp, o2)

    mod = _to_rtl(tr_top)
    o0 = np.zeros(1, np.int32)
    o1 = np.zeros(N, np.int32)
    o2 = np.zeros(N, np.int32)
    mod.cosim(o0, o1, o2)
    assert np.array_equal(o1, np.arange(N) * 2 + 1), list(o1)
    assert np.array_equal(o2, np.arange(N) * 2 + 100), list(o2)


def test_a_multidimensional_container_local_buffer():
    # Shape is the child's business: it flattens its own addressing and drives
    # a linear address, so the container declares one cell of `prod(shape)`
    # words whatever the rank.
    @kernel
    async def s5_prod(s: Stream[i32], tmp: i32[4, 4]):
        for i in range(4):
            for j in range(4):
                tmp[i, j] = i * 4 + j
        s.put(1)

    @kernel
    async def s5_cons(s: Stream[i32], o1: i32[1]):
        o1[0] = s.get()

    @kernel
    def s5_post(tmp: i32[4, 4], out: i32[16]):
        for i in range(4):
            for j in range(4):
                out[i * 4 + j] = tmp[i, j] * 3

    @kernel
    async def s5_top(out: i32[16], o1: i32[1]):
        f: Stream[i32]
        tmp: i32[4, 4]
        await s5_prod(f, tmp)
        await s5_cons(f, o1)
        s5_post(tmp, out)

    mod = _to_rtl(s5_top)
    assert re.search(r"seq\.hlmem @tmp %0, %rst[^\n]* : <16xi32>", mod.mlir)
    out = np.zeros(16, np.int32)
    o1 = np.zeros(1, np.int32)
    mod.cosim(out, o1)
    assert np.array_equal(out, np.arange(16) * 3), list(out)


def test_a_container_local_constant_table():
    # A table nothing writes is a ROM even when it is container-local: one
    # `hw.aggregate_constant` read combinationally and registered to the
    # latency the children were timed against. Classification comes from the
    # accessors, not the declaration.
    @kernel
    async def ct_src(s: Stream[i32]):
        for i in range(N):
            s.put(i)

    @kernel
    async def ct_use(tbl: i32[N], s: Stream[i32], out: i32[N]):
        for i in range(N):
            out[i] = tbl[i] + s.get()

    @kernel
    async def ct_top(out: i32[N]):
        f: Stream[i32]
        tbl: i32[N] = [10, 20, 30, 40, 50, 60, 70, 80]
        await ct_src(f)
        await ct_use(tbl, f, out)

    mod = _to_rtl(ct_top)
    m = mod.mlir
    assert "hw.aggregate_constant" in m and "seq.hlmem" not in m
    out = np.zeros(N, np.int32)
    mod.cosim(out)
    assert np.array_equal(out, np.arange(1, N + 1) * 10 + np.arange(N)), list(out)


def test_a_written_container_table_keeps_its_contents():
    # The same container-local declaration, once a process WRITES it, is not a
    # ROM but a real memory that starts with the declared contents: the
    # classification comes from the accessors, and the container owns the
    # initialized storage either way.
    @kernel
    async def wt_src(s: Stream[i32]):
        for i in range(N):
            s.put(i)

    @kernel
    async def wt_use(tbl: i32[N], s: Stream[i32], out: i32[N]):
        for i in range(N):
            tbl[i] = tbl[i] + s.get()
            out[i] = tbl[i]

    @kernel
    async def wt_top(out: i32[N]):
        f: Stream[i32]
        tbl: i32[N] = [10, 20, 30, 40, 50, 60, 70, 80]
        await wt_src(f)
        await wt_use(tbl, f, out)

    mod = _to_rtl(wt_top)
    m = mod.mlir
    assert "seq.hlmem @tbl" in m and "hw.aggregate_constant" not in m
    out = np.zeros(N, np.int32)
    mod.cosim(out)
    assert np.array_equal(out, np.arange(1, N + 1) * 10 + np.arange(N)), list(out)


# --- cross-region buffer identity ------------------------------------------


def test_single_element_internal_buffer():
    # A depth-1 internal buffer written and read every iteration; rank does not
    # matter (`i32[1, 1]` behaves the same), only the element COUNT does.
    @kernel
    def one(A: i32[8], B: i32[8]):
        t: i32[1]
        for i in range(8):
            t[0] = A[i] * 2
            B[i] = t[0] + 1

    B = np.zeros(8, np.int32)
    _to_rtl(one).cosim(A8, B)
    assert np.array_equal(B, A8 * 2 + 1)

    @kernel
    def one2d(A: i32[8], B: i32[8]):
        t: i32[1, 1]
        for i in range(8):
            t[0, 0] = A[i] * 3
            B[i] = t[0, 0] - 1

    B2 = np.zeros(8, np.int32)
    _to_rtl(one2d).cosim(A8, B2)
    assert np.array_equal(B2, A8 * 3 - 1)


@pytest.mark.parametrize("depth", [1, 2, 4])
def test_buffer_threaded_across_regions_is_one_memory(depth):
    # A straight-line store, then a loop reading it: the store is its own
    # (acyclic) region, so the buffer crosses a region boundary as a region
    # result. Every depth is covered because the split has nothing to do with
    # depth: it must remain ONE memory, not one per accessing region.
    #
    # The loop accumulates INTO the element it reads, which is what keeps the
    # buffer storage at all: `scalarize-memory` forwards a store to a later load
    # only when nothing writes in between, and the loop's own store does. With a
    # plain read the value would travel as dataflow and there would be no memory
    # to be identical about.
    if depth == 1:

        @kernel
        def cross(A: i32[8], B: i32[8]):
            t: i32[1]
            t[0] = A[0]
            for i in range(8):
                t[0] = t[0] + A[i]
                B[i] = t[0]

    elif depth == 2:

        @kernel
        def cross(A: i32[8], B: i32[8]):
            t: i32[2]
            t[0] = A[0]
            for i in range(8):
                t[0] = t[0] + A[i]
                B[i] = t[0]

    else:

        @kernel
        def cross(A: i32[8], B: i32[8]):
            t: i32[4]
            t[0] = A[0]
            for i in range(8):
                t[0] = t[0] + A[i]
                B[i] = t[0]

    mod = _to_rtl(cross)
    # Counted off the memories rather than the emitted cells: `t` is small
    # enough to be auto-partitioned, so it is one register per element and
    # holds no `seq.hlmem` to count.
    assert len([m for f in mod.microarch.funcs for m in f.mems if not m.external]) == 1
    B = np.zeros(8, np.int32)
    mod.cosim(A8, B)
    assert np.array_equal(B, A8[0] + np.cumsum(A8))


def test_two_stores_threaded_across_regions():
    # Two straight-line stores feeding one downstream loop: both must reach the
    # reader's memory (a split loses both, so the reads come back zero). Both are
    # accumulated into in the loop, so neither seed is forwarded away and both
    # genuinely travel through the memory.
    @kernel
    def cross2(A: i32[8], B: i32[8]):
        t: i32[4]
        t[0] = A[0]
        t[1] = A[1]
        for i in range(8):
            t[0] = t[0] + A[i]
            t[1] = t[1] + 2 * A[i]
            B[i] = t[0] + t[1]

    mod = _to_rtl(cross2)
    assert len([m for f in mod.microarch.funcs for m in f.mems if not m.external]) == 1
    B = np.zeros(8, np.int32)
    mod.cosim(A8, B)
    c = np.cumsum(A8)
    assert np.array_equal(B, (A8[0] + c) + (A8[1] + 2 * c))


def test_a_forwarded_buffer_costs_no_storage():
    # The complement of the two tests above, and the point of `scalarize-memory`:
    # when every read of a local buffer has a unique reaching store, the value is
    # dataflow and the buffer must not be built at all. This is the shape the
    # region-crossing tests used to have.
    @kernel
    def fwd(A: i32[8], B: i32[8]):
        t: i32[4]
        t[0] = A[0]
        t[1] = A[1]
        for i in range(8):
            B[i] = t[0] + t[1] + A[i]

    mod = _to_rtl(fwd)
    assert "seq.hlmem" not in mod.mlir, mod.mlir
    B = np.zeros(8, np.int32)
    mod.cosim(A8, B)
    assert np.array_equal(B, A8[0] + A8[1] + A8)


def test_a_callee_write_shares_the_port_of_ordered_regions():
    # Three writers of one buffer, a declaration init, an init loop, and a
    # child's write port, run in program order under one serial loop, so all
    # three share one write port. The golden reads one element from the child
    # and one from the loop, so a write lost to the shared port breaks it.
    @kernel
    def bump(a: i32[4], v: i32):
        a[v] = v + 40

    @kernel
    def seqw(A: i32[8], out: i32[8]):
        buf: i32[4] = 0
        for t in range(8):
            for i in range(4):
                buf[i] = A[t] + i
            bump(buf, 2)
            out[t] = buf[2] + buf[3]

    mod = _to_rtl(seqw)
    buf = next(m for f in mod.microarch.funcs for m in f.mems if m.realization == "ram")
    assert buf.cost.write_ports == 1
    out = np.zeros(8, np.int32)
    mod.cosim(A8, out)
    assert np.array_equal(out, 42 + A8 + 3)


# --- dead-initializer elision -----------------------------------------------

_NO_ELIDE = RTL_PREPARE_PIPELINE.replace("elide-dead-init,\n", "")

MIXED = np.array([3, -1, 7, -4, 0, 5, -9, 2], np.int32)


def _stores(kernel_fn):
    """Store sites the prepare pipeline leaves with ``elide-dead-init`` disabled
    and enabled."""
    counts = []
    for pipeline in (_NO_ELIDE, RTL_PREPARE_PIPELINE):
        module = kernel_fn.schedule().module
        run_pipeline(module, pipeline)
        counts.append(
            len(_walk(module, "affine.store")) + len(_walk(module, "memref.store"))
        )
    return tuple(counts)


def test_dead_init_is_elided():
    # The shape the pass exists for. Every element is overwritten before anything
    # reads one, so the fill is dead and the cosim pins that dropping it leaves no
    # element reading as uninitialized.
    @kernel
    def dead_init(A: i32[8], out: i32[8]):
        buf: i32[8] = 0
        for i in range(8):
            buf[i] = A[i] + 1
        for i in range(8):
            out[i] = buf[i]

    assert _stores(dead_init) == (3, 2)

    out = np.zeros(8, np.int32)
    _to_rtl(dead_init).cosim(A8, out)
    assert np.array_equal(out, A8 + 1)


def test_dead_init_is_elided_inside_an_enclosing_loop():
    # The array is declared in a loop body, so the fill costs a full pass over it
    # on every trip. Its regions stay symbolic in the trip counter rather than
    # being folded into a hull.
    @kernel
    def dead_init_inner(A: i32[16], out: i32[16]):
        for t in range(2):
            buf: i32[8] = 0
            for i in range(8):
                buf[i] = A[t * 8 + i] + 1
            for i in range(8):
                out[t * 8 + i] = buf[i]

    assert _stores(dead_init_inner) == (3, 2)


def test_dead_init_is_elided_when_the_overwrite_is_reordered_or_tiled():
    # Coverage is a property of the written SET, so neither the loop order nor a
    # strip-mined subscript changes it. Both defeat any match on nest shape.
    @kernel
    def permuted_init(A: i32[4, 4], out: i32[4, 4]):
        buf: i32[4, 4] = 0
        for j in range(4):
            for i in range(4):
                buf[i, j] = A[i, j] + 1
        for i in range(4):
            for j in range(4):
                out[i, j] = buf[i, j]

    @kernel
    def tiled_init(A: i32[16], out: i32[16]):
        buf: i32[16] = 0
        for o in range(4):
            for t in range(4):
                buf[o * 4 + t] = A[o * 4 + t] + 1
        for i in range(16):
            out[i] = buf[i]

    assert _stores(permuted_init) == (3, 2)
    assert _stores(tiled_init) == (3, 2)


def test_dead_init_is_elided_across_a_call():
    # The sub-kernel writes every element and reads none, which its per-parameter
    # summary carries back to the caller.
    @kernel
    def overwrite_all(dst: i32[8], A: i32[8]):
        for i in range(8):
            dst[i] = A[i] + 1

    @kernel
    def call_init(A: i32[8], out: i32[8]):
        buf: i32[8] = 0
        overwrite_all(buf, A)
        for i in range(8):
            out[i] = buf[i]

    assert _stores(call_init) == (3, 2)

    out = np.zeros(8, np.int32)
    _to_rtl(call_init).cosim(A8, out)
    assert np.array_equal(out, A8 + 1)


def test_dead_init_is_elided_when_every_branch_writes_it():
    # One of the two blocks runs whatever the condition does, so an access both
    # make is unconditional. Regions carry no guard, so this has to be settled on
    # the accesses rather than by intersecting their footprints.
    @kernel
    def branch_init(A: i32[8], out: i32[8]):
        buf: i32[8] = 0
        for i in range(8):
            if A[i] > 0:
                buf[i] = A[i] + 1
            else:
                buf[i] = 0
        for i in range(8):
            out[i] = buf[i]

    assert _stores(branch_init) == (4, 3)

    out = np.zeros(8, np.int32)
    _to_rtl(branch_init).cosim(MIXED, out)
    assert np.array_equal(out, np.where(MIXED > 0, MIXED + 1, 0))


def test_dead_init_is_elided_when_only_a_prefix_is_touched():
    # Written and read regions are both `0..k`, so no element is ever read at its
    # initial value even though the declared shape is never covered. Both regions
    # keep `k` as a symbol, which is what makes them comparable.
    @kernel
    def prefix_init(A: i32[8], out: i32[8]):
        for k in range(8):
            buf: i32[8] = 0
            for i in range(k):
                buf[i] = A[i] + 1
            for i in range(k):
                out[i] = buf[i]

    assert _stores(prefix_init) == (3, 2)

    out = np.zeros(8, np.int32)
    _to_rtl(prefix_init).cosim(A8, out)
    ref = np.zeros(8, np.int32)
    for k in range(8):
        for i in range(k):
            ref[i] = A8[i] + 1
    assert np.array_equal(out, ref)


def test_init_is_live_when_the_overwriting_nest_reads():
    # The covering nest reads the element it is about to write, so the initial
    # value reaches that read: eliding here would change the result, not just the
    # schedule. This is what keeps a `+=` accumulator correct.
    @kernel
    def acc_init(A: i32[8], out: i32[8]):
        buf: i32[8] = 0
        for i in range(8):
            buf[i] = buf[i] + A[i]
        for i in range(8):
            out[i] = buf[i]

    assert _stores(acc_init) == (3, 3)

    out = np.zeros(8, np.int32)
    _to_rtl(acc_init).cosim(A8, out)
    assert np.array_equal(out, A8)


def test_init_is_live_when_the_callee_reads_first():
    # Same call shape as the elided case, opposite answer: the summary says the
    # parameter is read before it is written, so the fill feeds the accumulation.
    @kernel
    def accumulate_into(dst: i32[8], A: i32[8]):
        for i in range(8):
            dst[i] = dst[i] + A[i]

    @kernel
    def call_acc_init(A: i32[8], out: i32[8]):
        buf: i32[8] = 0
        accumulate_into(buf, A)
        for i in range(8):
            out[i] = buf[i]

    assert _stores(call_acc_init) == (3, 3)

    out = np.zeros(8, np.int32)
    _to_rtl(call_acc_init).cosim(A8, out)
    assert np.array_equal(out, A8)


def test_init_is_live_when_only_one_branch_writes_it():
    # With no else block the store is genuinely conditional, so the elements the
    # condition skips are read at their initial value.
    @kernel
    def half_branch_init(A: i32[8], out: i32[8]):
        buf: i32[8] = 0
        for i in range(8):
            if A[i] > 0:
                buf[i] = A[i] + 1
        for i in range(8):
            out[i] = buf[i]

    assert _stores(half_branch_init) == (3, 3)


def test_init_is_live_when_a_read_sits_between_the_writes():
    # One element is read before the overwrite, so the fill has to stay.
    @kernel
    def probe_init(A: i32[8], out: i32[8], probe: i32[1]):
        buf: i32[8] = 0
        probe[0] = buf[3]
        for i in range(8):
            buf[i] = A[i] + 1
        for i in range(8):
            out[i] = buf[i]

    assert _stores(probe_init) == (4, 4)


def test_init_is_live_when_a_read_reaches_past_the_written_set():
    # The prefix shape with the read widened to the whole array: `0..k` written,
    # `0..8` read, so the tail is read at its initial value.
    @kernel
    def prefix_overread(A: i32[8], out: i32[8]):
        for k in range(8):
            buf: i32[8] = 0
            for i in range(k):
                buf[i] = A[i] + 1
            for i in range(8):
                out[i] = buf[i]

    assert _stores(prefix_overread) == (3, 3)


def test_init_is_live_when_a_write_sits_under_a_loop_that_may_not_run():
    # The subscript ignores the inner induction variable, so a region carries
    # nothing about the trip count: at `k == 0` the store never runs and `buf[0]`
    # is still read at its initial value. The fill is non-zero because an
    # uninitialized read gives 0 here and would hide a wrong answer.
    @kernel
    def invariant_store(A: i32[8], out: i32[8]):
        for k in range(8):
            buf: i32[8] = 99
            for i in range(k):
                buf[0] = A[i]
            out[k] = buf[0]

    assert _stores(invariant_store) == (3, 3)

    out = np.zeros(8, np.int32)
    _to_rtl(invariant_store).cosim(A8, out)
    ref = np.zeros(8, np.int32)
    for k in range(8):
        value = 99
        for i in range(k):
            value = A8[i]
        ref[k] = value
    assert np.array_equal(out, ref)


def test_init_is_live_when_the_overwrite_is_partial():
    # The second nest covers half the array, so the other half is still read at
    # its initial value.
    @kernel
    def partial_init(A: i32[8], out: i32[8]):
        buf: i32[8] = 0
        for i in range(4):
            buf[i] = A[i] + 1
        for i in range(8):
            out[i] = buf[i]

    assert _stores(partial_init) == (3, 3)


# --- coverage by an earlier write of the same access -------------------------


def test_dead_init_is_elided_when_the_writer_reads_back_what_it_wrote():
    # The callee writes `dst[i]` and reads it back in the same iteration, so the
    # read is loop-independent on that write and never reaches the fill. Nothing
    # orders them at the call site: the whole callee is one step there.
    @kernel
    def clamp_all(dst: i32[8], A: i32[8]):
        for i in range(8):
            dst[i] = A[i] + 1
            if dst[i] > 4:
                dst[i] = 4

    @kernel
    def call_clamp_init(A: i32[8], out: i32[8]):
        buf: i32[8] = 77
        clamp_all(buf, A)
        for i in range(8):
            out[i] = buf[i]

    assert _stores(call_clamp_init) == (4, 3)

    out = np.zeros(8, np.int32)
    _to_rtl(call_clamp_init).cosim(A8, out)
    assert np.array_equal(out, np.minimum(A8 + 1, 4))


def test_dead_init_is_elided_for_a_scan():
    # Iteration i reads what i-1 wrote and element 0 comes from a sibling store.
    # No step covers the array, so this needs the carried dependence.
    @kernel
    def scan_init(A: i32[8], out: i32[8]):
        total: i32[8] = 77
        total[0] = A[0]
        for i in range(1, 8):
            total[i] = total[i - 1] + A[i]
        for i in range(8):
            out[i] = total[i]

    assert _stores(scan_init) == (4, 3)

    out = np.zeros(8, np.int32)
    _to_rtl(scan_init).cosim(A8, out)
    assert np.array_equal(out, np.cumsum(A8))


def test_dead_init_is_elided_for_a_wavefront():
    # The interior reads three neighbours the earlier iterations produced, and the
    # first row and column come from sibling loops, so coverage is split across
    # both sources.
    @kernel
    def wavefront_init(A: i32[4, 4], out: i32[4, 4]):
        M: i32[4, 4] = 77
        for i in range(4):
            M[0, i] = A[0, i]
        for j in range(4):
            M[j, 0] = A[j, 0]
        for bi in range(1, 4):
            for ai in range(1, 4):
                M[bi, ai] = M[bi - 1, ai - 1] + M[bi - 1, ai] + M[bi, ai - 1]
        for i in range(4):
            for j in range(4):
                out[i, j] = M[i, j]

    assert _stores(wavefront_init) == (5, 4)

    A = np.arange(1, 17, dtype=np.int32).reshape(4, 4)
    ref = np.zeros((4, 4), np.int32)
    ref[0, :] = A[0, :]
    ref[:, 0] = A[:, 0]
    for bi in range(1, 4):
        for ai in range(1, 4):
            ref[bi, ai] = ref[bi - 1, ai - 1] + ref[bi - 1, ai] + ref[bi, ai - 1]

    out = np.zeros((4, 4), np.int32)
    _to_rtl(wavefront_init).cosim(A, out)
    assert np.array_equal(out, ref)


def test_init_is_live_when_the_read_sits_in_an_else_region():
    # `getEnclosingAffineOps` collects the guard whichever region an access is
    # in, and the index set then gets the THEN condition, so this read's
    # iteration domain comes back as `j < 4` while it really runs on `j >= 4`.
    # Coverage argued over that domain would say the `0..3` writes reach it.
    @kernel
    def else_read(A: i32[8], out: i32[8]):
        buf: i32[8] = 77
        for i in range(4):
            buf[i] = A[i]
        for j in range(8):
            if j < 4:
                out[j] = 0
            else:
                out[j] = buf[j]

    assert _stores(else_read) == (4, 4)

    out = np.zeros(8, np.int32)
    _to_rtl(else_read).cosim(A8, out)
    assert np.array_equal(out, np.where(np.arange(8) < 4, 0, 77))


def test_init_is_live_for_a_sliding_window():
    # A shift register reads the taps the fill left, so the fill IS the filter's
    # startup state. The window shape does not make an initializer dead, and the
    # golden here depends on the fill value.
    @kernel
    def shift_init(A: i32[8], out: i32[8]):
        delay: i32[4] = 77
        for j in range(8):
            for i in range(3):
                delay[3 - i] = delay[2 - i]
            delay[0] = A[j]
            out[j] = delay[0] + delay[3]

    assert _stores(shift_init) == (4, 4)

    ref = np.zeros(8, np.int32)
    delay = [77, 77, 77, 77]
    for j in range(8):
        for i in range(3):
            delay[3 - i] = delay[2 - i]
        delay[0] = int(A8[j])
        ref[j] = delay[0] + delay[3]

    out = np.zeros(8, np.int32)
    _to_rtl(shift_init).cosim(A8, out)
    assert np.array_equal(out, ref)


# --- composing effects inside a nested region --------------------------------
#
# The walk carries what has definitely been written into a loop body and a
# branch, so the ops inside one are ordered against each other rather than
# lumped into a single step. A loop body starts from what was written BEFORE the
# loop and nothing else: an element the body writes at one trip is not there yet
# at the first one.


def test_init_is_dead_when_two_callees_hand_over_inside_a_loop():
    # Every trip of `t` writes the whole array before reading it. Dependence
    # analysis cannot pair a call with anything, so this rests entirely on the
    # body's own ordering.
    @kernel
    def fill_stage(dst: i32[8], A: i32[8], t: i32):
        for i in range(8):
            dst[i] = A[i] + t

    @kernel
    def read_stage(src: i32[8], out: i32[8], t: i32):
        for i in range(8):
            out[i] = src[i] + t

    @kernel
    def staged_init(A: i32[8], out: i32[8]):
        buf: i32[8] = 77
        for t in range(2):
            fill_stage(buf, A, t)
            read_stage(buf, out, t)

    assert _stores(staged_init) == (3, 2)

    out = np.zeros(8, np.int32)
    _to_rtl(staged_init).cosim(A8, out)
    assert np.array_equal(out, A8 + 2)


def test_init_is_live_when_a_loop_body_reads_what_an_earlier_trip_wrote():
    # `buf` is written at trip `t` and read at trip `t + 1`, so the first trip
    # reads the fill. Seeding the body with what was written before the loop is
    # what keeps this one live.
    @kernel
    def carry_init(A: i32[8], out: i32[8]):
        buf: i32[8] = 77
        for t in range(4):
            for i in range(8):
                out[i] = buf[i]
            for i in range(8):
                buf[i] = A[i] + t

    assert _stores(carry_init) == (3, 3)

    out = np.zeros(8, np.int32)
    _to_rtl(carry_init).cosim(A8, out)
    assert np.array_equal(out, A8 + 2)


def test_init_is_dead_when_the_branches_of_an_affine_guard_add_up():
    # Neither arm writes the whole array, and neither is the same access as the
    # other, so nothing survives an intersection. Their guards partition the
    # iteration space, which is what makes the union total.
    @kernel
    def split_init(A: i32[8], out: i32[8]):
        buf: i32[8] = 77
        for i in range(8):
            if i < 3:
                buf[i] = A[i]
            else:
                buf[10 - i] = A[i] * 2
        for i in range(8):
            out[i] = buf[i]

    assert _stores(split_init) == (4, 3)

    ref = np.array([A8[j] if j < 3 else A8[10 - j] * 2 for j in range(8)], np.int32)
    out = np.zeros(8, np.int32)
    _to_rtl(split_init).cosim(A8, out)
    assert np.array_equal(out, ref)


def test_init_is_dead_when_a_guarded_write_covers_the_array_alone():
    # One store below a guard with no else at all. Its footprint over the
    # iterations the guard admits is the whole array.
    @kernel
    def guarded_init(A: i32[8], out: i32[8]):
        buf: i32[8] = 77
        for i in range(10):
            if i >= 2:
                buf[i - 2] = A[i - 2]
        for i in range(8):
            out[i] = buf[i]

    assert _stores(guarded_init) == (3, 2)

    out = np.zeros(8, np.int32)
    _to_rtl(guarded_init).cosim(A8, out)
    assert np.array_equal(out, A8)


def test_init_is_live_when_a_data_dependent_branch_writes_it():
    # An `scf.if` on a runtime value leaves no condition to fold, so only what
    # both arms write is certain, and here one arm writes nothing.
    @kernel
    def maybe_init(A: i32[8], out: i32[8]):
        buf: i32[8] = 77
        for i in range(8):
            if A[i] > 0:
                buf[i] = A[i]
        for i in range(8):
            out[i] = buf[i]

    assert _stores(maybe_init) == (3, 3)

    ref = np.where(MIXED > 0, MIXED, 77)
    out = np.zeros(8, np.int32)
    _to_rtl(maybe_init).cosim(MIXED, out)
    assert np.array_equal(out, ref)


# --- scalarization of forwardable arrays ------------------------------------


def test_a_small_local_buffer_under_a_pipeline_costs_no_storage():
    # The shape `scalarize-memory` exists for. Unrolling under the pipeline makes
    # every subscript of `buf` a constant, so each read has a unique reaching
    # store and the buffer is pure dataflow. It used to be a 1-port LUTRAM whose
    # ports set the resource-min II: II 4 and latency 69 against II 2 here.
    @kernel
    def small(A: i32[16, 4], out: i32[16]):
        for i in range(16):
            buf: i32[4]
            for j in range(4):
                buf[j] = A[i, j] * 2
            acc: i32 = 0
            for j in range(4):
                acc += buf[j]
            out[i] = acc

    s = small.schedule()
    s.pipeline("i")
    mod = s.export("rtl")
    res = mod.schedule()
    # The residual II is `A` at 4 reads over its 2 ports, not `buf`.
    assert _iis(res.cyclic()) == [2]
    # And the buffer is DELETED, not merely turned into registers, which is what
    # "costs no storage" means: the design holds no such array at all, in any
    # realization.
    assert not [m for m in mod.microarch.top.mems if m.owner.startswith("buf")]

    A = (np.arange(64, dtype=np.int32) % 11).reshape(16, 4)
    out = np.zeros(16, np.int32)
    mod.cosim(A, out)
    assert np.array_equal(out, (A * 2).sum(1))


def test_a_data_dependent_subscript_keeps_its_storage():
    # The complement, and the boundary `scalarize-memory` must not cross: a
    # subscript that is not a constant has no unique reaching store, so the
    # buffer stays storage. What `auto-complete-partition` then decides is the
    # KIND of storage: small enough, so registers rather than a ported memory.
    @kernel
    def roam(A: i32[16, 4], out: i32[16]):
        for i in range(16):
            buf: i32[4]
            for j in range(4):
                buf[j] = A[i, j] * 2
            out[i] = buf[A[i, 0] & 3]

    s = roam.schedule()
    s.pipeline("i")
    mod = s.export("rtl")
    # Still one storage, and `register` is realized as it is priced: a cell per
    # element, so no addressed memory is built for it at all.
    assert len([m for f in mod.microarch.funcs for m in f.mems if not m.external]) == 1
    assert mod.microarch.mem("buf_").storage == "register"
    assert "seq.hlmem" not in mod.mlir, mod.mlir

    # Threshold 0 disqualifies every array, so the same buffer stays on ports.
    s2 = roam.schedule()
    s2.pipeline("i")
    on_ports = s2.export("rtl").set_scheduler_opt(scalarize_threshold=0)
    assert on_ports.microarch.mem("buf_").storage != "register"
    assert len(re.findall(r"= seq\.hlmem", on_ports.mlir)) == 1, on_ports.mlir

    A = (np.arange(64, dtype=np.int32) % 11).reshape(16, 4)
    out = np.zeros(16, np.int32)
    mod.cosim(A, out)
    assert np.array_equal(out, (A * 2)[np.arange(16), A[:, 0] & 3])


@pytest.mark.parametrize("depth", [16, 32])
def test_the_auto_partition_threshold_is_a_boundary(depth):
    # A register file with a runtime subscript costs a mux per read and a demux
    # per write, so the element-count threshold is what bounds the area the
    # automatic partition can spend. It has to be a real boundary. Three reads
    # in one body is what qualifies the array: past any ported row's bandwidth,
    # only the register file serves the block.
    if depth == 16:

        @kernel
        def sized(A: i32[16], out: i32[16]):
            buf: i32[16]
            for i in range(16):
                buf[i] = A[i] * 2
            for i in range(16):
                out[i] = buf[A[i] & 15] + buf[(A[i] + 1) & 15] + buf[(A[i] + 2) & 15]

    else:

        @kernel
        def sized(A: i32[16], out: i32[16]):
            buf: i32[32]
            for i in range(16):
                buf[i] = A[i] * 2
            for i in range(16):
                out[i] = buf[A[i] & 15] + buf[(A[i] + 1) & 15] + buf[(A[i] + 2) & 15]

    mod = _to_rtl(sized)
    registered = mod.microarch.mem("buf_").storage == "register"
    assert registered == (depth == 16)

    out = np.zeros(16, np.int32)
    mod.cosim(A16, out)
    doubled = A16 * 2
    expect = doubled[A16 & 15] + doubled[(A16 + 1) & 15] + doubled[(A16 + 2) & 15]
    assert np.array_equal(out, expect)


def test_a_rolling_small_array_keeps_ported_storage():
    # A dual-ported row serves one or two touches per iteration, so a register
    # file would spend its read mux and write decode on nothing: the array
    # stays on the priced storage tables.
    @kernel
    def roll(A: i32[16], out: i32[16]):
        buf: i32[16]
        for i in range(16):
            buf[i] = A[i] * 2
        for i in range(16):
            out[i] = buf[A[i] & 15]

    mod = _to_rtl(roll)
    assert mod.microarch.mem("buf_").storage != "register"

    out = np.zeros(16, np.int32)
    mod.cosim(A16, out)
    assert np.array_equal(out, (A16 * 2)[A16 & 15])


def test_a_large_all_constant_array_dissolves_to_registers():
    # The element-count cap bounds only a variable subscript, which pays a read
    # mux and a write demux. Every subscript here is the constant 0, so the array
    # dissolves into wired registers with no mux and is scalarized at any size:
    # 24 elements is past the 16-element cap yet still becomes a register file,
    # its untouched cells folding away.
    def build():
        @kernel
        def cross(A: i32[8], B: i32[8]):
            t: i32[24]
            t[0] = A[0]
            for i in range(8):
                t[0] = t[0] + A[i]
                B[i] = t[0]

        return cross

    mod = _to_rtl(build())
    assert mod.microarch.mem("t").storage == "register"
    assert "seq.hlmem" not in mod.mlir, mod.mlir

    # A threshold well below the array's size still scalarizes it, since the cap
    # does not gate a constant-subscript array; only turning auto-partition off
    # (threshold 0) keeps it a ported memory.
    below = build().schedule().export("rtl").set_scheduler_opt(scalarize_threshold=4)
    assert below.microarch.mem("t").storage == "register"
    off = build().schedule().export("rtl").set_scheduler_opt(scalarize_threshold=0)
    assert off.microarch.mem("t").storage != "register"

    B = np.zeros(8, np.int32)
    mod.cosim(A8, B)
    assert np.array_equal(B, A8[0] + np.cumsum(A8))


def test_invariant_reads_preload_ahead_of_a_pipelined_loop():
    # An unrolled body re-reading the same words of a ported array every
    # iteration would pay its port count in II. Reads whose address does not
    # change across iterations move in front of the loop instead, so the body
    # meets held registers and the array's ports serve only the preload.
    N, TAPS = 32, 8

    @kernel
    def fir(x: i32[N], taps: i32[TAPS], out: i32[N]):
        for j in range(N):
            xv: i32 = x[j]
            acc: i32 = 0
            for k in range(TAPS):
                acc += xv * taps[k]
            out[j] = acc

    s = fir.schedule()
    s.pipeline(s.loop("j"), ii=1)
    mod = s.export("rtl")
    assert _iis(mod.schedule().func("fir").regions) == [1]

    rng = np.random.default_rng(0)
    x = rng.integers(-16, 16, N, dtype=np.int32)
    taps = rng.integers(-8, 8, TAPS, dtype=np.int32)
    out = np.zeros(N, np.int32)
    mod.cosim(x, taps, out)
    assert np.array_equal(out, x * taps.sum())


def test_an_array_handed_to_a_sub_kernel_is_not_partitioned():
    # A child masters ports on storage the parent owns, and a Complete partition
    # across that boundary has never been tried, so any use that is not a direct
    # access disqualifies the array.
    @kernel
    def bump(b: i32[8]):
        for j in range(8):
            b[j] = b[j] + 100

    @kernel
    def outer(A: i32[8], out: i32[8]):
        buf: i32[8]
        for i in range(8):
            buf[i] = A[i] * 2
        bump(buf)
        for i in range(8):
            out[i] = buf[i]

    mod = _to_rtl(outer)
    assert mod.microarch.mem("buf_").storage != "register"

    out = np.zeros(8, np.int32)
    mod.cosim(A8, out)
    assert np.array_equal(out, A8 * 2 + 100)


def test_a_sub_kernel_masters_a_complete_partitioned_buffer():
    # Asking for the partition explicitly is a different thing from the
    # automatic one declined above: it makes the parent's buffer a 0-cycle
    # (combinational) read, which the child then masters ports on. The child's
    # parameter is a BlockArgument but NOT a boundary port -- the storage is the
    # parent's `seq.hlmem`, not a driver's memory on the far side of the top
    # module -- so the >= 1 cycle boundary contract does not apply to it.
    @kernel
    def bump(b: i32[8]):
        for j in range(8):
            b[j] = b[j] + 100

    @kernel
    def outer(A: i32[8], out: i32[8]):
        buf: i32[8]
        for i in range(8):
            buf[i] = A[i] * 2
        bump(buf)
        for i in range(8):
            out[i] = buf[i]

    s = outer.schedule()
    s.partition("buf", kind=s.Complete)
    mod = s.export("rtl")

    out = np.zeros(8, np.int32)
    mod.cosim(A8, out)
    assert np.array_equal(out, A8 * 2 + 100)


# --- completely-partitioned arguments (one boundary port per element) --------


def _scattered():
    """A kernel reading a complete-partitioned argument at a runtime subscript."""

    @kernel
    def scatter(A: i32[8], out: i32[8]):
        for i in range(8):
            out[i] = A[i] * 2

    s = scatter.schedule()
    s.partition("A", kind=s.Complete)
    return s.export("rtl")


def test_a_complete_partitioned_argument_becomes_element_ports():
    # An argument's storage lives outside the module, so "scatter it into
    # registers" can only mean every element arrives at once: N input ports, no
    # address and no access latency. This is what Vitis emits for the same
    # directive (`a_0 .. a_7`), and it is the only shape that delivers the
    # unlimited combinational ports a Complete partition bills the scheduler for.
    mod = _scattered()
    iface = mod.interfaces["scatter"]
    (rf,) = iface.registers
    # Read-only, so the bare name and no output side at all.
    assert rf.elements == tuple(RegisterFile.Element(f"A_{k}") for k in range(8)), rf
    assert rf.width == 32 and rf.shape == (8,)
    # It is a register file, NOT an addressed port group: `A` appears in neither
    # read nor write interfaces, and takes no address port.
    assert all(p.arg != 0 for acc in iface.reads for p in acc), iface.reads
    assert "A_rd0_addr" not in mod.verilog

    out = np.zeros(8, np.int32)
    mod.cosim(A8, out)
    assert np.array_equal(out, A8 * 2)


def test_a_scattered_argument_read_folds_at_a_constant_index():
    # A constant subscript selects one input port, so the N:1 read mux folds away
    # entirely in CIRCT and costs no hardware. Nothing special-cases it -- the
    # crossbar is emitted and the folder collapses it.
    @kernel
    def konst(A: i32[8], out: i32[8]):
        for i in range(8):
            out[i] = A[3] * 2

    s = konst.schedule()
    s.partition("A", kind=s.Complete)
    mod = s.export("rtl")
    out = np.zeros(8, np.int32)
    mod.cosim(A8, out)
    assert np.array_equal(out, np.full(8, A8[3] * 2))


def _four_read_ii(complete):
    """II of a body reading one argument four times, with and without the
    Complete partition. The addresses roll with the iteration so no read is
    loop-invariant and none can preload ahead of the loop."""

    @kernel
    def dot4(A: i32[8], out: i32[8]):
        for i in range(8):
            out[i] = A[i % 4] + A[(i + 1) % 4] + A[(i + 2) % 4] + A[(i + 3) % 4]

    s = dot4.schedule()
    if complete:
        s.partition("A", kind=s.Complete)
    return _iis(s.export("rtl").schedule().func("dot4").regions)


def test_a_scattered_argument_has_no_port_pressure():
    # What the feature BUYS. Four reads of one argument in one iteration need
    # four ports, and an addressed argument has 2, so its II is 2. Scattered,
    # every element is already on the boundary, so the reads contend for no port
    # at all (`MemoryBankModel` bills none) and the loop pipelines at II=1.
    assert _four_read_ii(complete=False) == [2]
    assert _four_read_ii(complete=True) == [1]


def test_a_read_write_scattered_argument_splits_its_directions():
    # An argument used BOTH ways needs its two ports told apart, so the bare name
    # gives way to `_in` / `_out` (+ its write enable). This is the one case that
    # renames, and it is the rule Vitis follows (`a_0_i` / `a_0_o`).
    @kernel
    def rmw(A: i32[8]):
        for i in range(8):
            A[i] = A[i] * 2

    s = rmw.schedule()
    s.partition("A", kind=s.Complete)
    mod = s.export("rtl")
    (rf,) = mod.interfaces["rmw"].registers
    assert rf.elements[0] == RegisterFile.Element("A_0_in", "A_0_out", "A_0_out_we"), rf

    a = A8.copy()
    mod.cosim(a)
    assert np.array_equal(a, A8 * 2)


def test_a_write_only_scattered_argument_keeps_the_bare_name():
    # Only one direction is live, so no disambiguation: `out` takes the bare
    # name and there is no input port at all. Elements the kernel never stores
    # to are never enabled, so they pass the driver's value through.
    @kernel
    def wo(A: i32[8]):
        for i in range(4):
            A[i] = 7

    s = wo.schedule()
    s.partition("A", kind=s.Complete)
    mod = s.export("rtl")
    (rf,) = mod.interfaces["wo"].registers
    assert rf.elements[0] == RegisterFile.Element(None, "A_0", "A_0_we"), rf

    a = A8.copy()
    mod.cosim(a)
    assert np.array_equal(a, np.concatenate([np.full(4, 7), A8[4:]]))


def test_scattered_writes_share_their_element_ports():
    # The reason the port drivers are built after every region rather than at
    # each store: N element ports serve ALL of an argument's writes, where an
    # addressed argument gets a port group per access. Two stores in one region
    # and a third in another would each drive them.
    @kernel
    def multi(A: i32[8], b: i32[8]):
        for i in range(4):
            A[2 * i] = b[2 * i] + 1
            A[2 * i + 1] = b[2 * i + 1] + 2
        for i in range(8):
            A[i] = A[i] * 10

    s = multi.schedule()
    s.partition("A", kind=s.Complete)
    mod = s.export("rtl")

    a, b = np.zeros(8, np.int32), A8.copy()
    mod.cosim(a, b)
    expect = np.array([b[k] + (1 if k % 2 == 0 else 2) for k in range(8)]) * 10
    assert np.array_equal(a, expect)


def test_a_scattered_argument_carries_a_recurrence_through_the_boundary():
    # The registers sit OUTSIDE the module, so a RAW closes through the boundary:
    # a store presents at its cycle and the driver commits on that edge, and the
    # combinational read sees it the next cycle. Write 1 / read 0 is the same
    # model an internal complete-partitioned array runs, just relocated, so the
    # II is the recurrence and nothing else.
    @kernel
    def acc(A: i32[8]):
        for i in range(1, 8):
            A[i] = A[i - 1] + A[i]

    s = acc.schedule()
    s.partition("A", kind=s.Complete)
    mod = s.export("rtl")

    a = A8.copy()
    mod.cosim(a)
    assert np.array_equal(a, np.cumsum(A8))


# --- internal memory names --------------------------------------------------


def test_unrolled_copies_of_one_array_get_distinct_symbols():
    # Unrolling a body that declares an array gives every copy the same source
    # name, so the tie-break has to fire on an internal memory too. CIRCT would
    # rename the duplicate downstream, which is exactly the desync `Naming.h`
    # exists to prevent.
    @kernel
    def dup(A: i32[8], out: i32[8]):
        for i in range(4):
            buf: i32[32]
            for k in range(32):
                buf[k] = A[i] + k
            out[i] = buf[A[i] & 31]
            out[i + 4] = buf[(A[i] + 3) & 31]

    s = dup.schedule()
    s.unroll(s.loop("i"), factor=2)
    mod = s.export("rtl")
    syms = re.findall(r"seq\.hlmem @(\S+)", mod.mlir)
    assert len(syms) == 2 and len(set(syms)) == 2, syms

    A8x = np.arange(8, dtype=np.int32)
    out = np.zeros(8, np.int32)
    mod.cosim(A8x, out)
    exp = np.zeros(8, np.int32)
    for i in range(4):
        exp[i] = A8x[i] + (A8x[i] & 31)
        exp[i + 4] = A8x[i] + ((A8x[i] + 3) & 31)
    assert np.array_equal(out, exp)


# An array whose linear extent needs more address bits than the index carrier
# has would wrap its addresses, so the compile is refused instead.
def test_an_array_past_the_index_carrier_is_refused():
    @kernel
    def bigarr(A: i32[2**33], out: i32[1]):
        out[0] = A[0]

    with pytest.raises(RuntimeError, match="ALLO-N0018"):
        _sched(bigarr)
