# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The port-naming grammar and the boundary port manifest it produces."""

import os
import re
import shutil
import sys

import ml_dtypes
import numpy as np
import pytest

import allo
from allo import kernel
from allo.backend.rtl import Control, Operator
from allo.lang import Stream, bf16, f16, f32, i32
from allo.lang.core import APInt

sys.path.insert(0, os.path.dirname(__file__))
from _common import Dcp, _to_rtl  # noqa: E402

pytestmark = pytest.mark.skipif(
    shutil.which("verilator") is None, reason="verilator not available"
)


def _bases(rtl, module):
    """The port-group bases of one emitted module, by role."""
    m = rtl.interfaces[module]
    return {
        "reads": [[p.base for p in group] for group in m.reads],
        "writes": [[p.base for p in group] for group in m.writes],
        "scalars": [s.name for s in m.scalars],
        "results": [r.name for r in m.results],
        "streams": [s.base for s in m.streams],
    }


# ---------------------------------------------------------------------------
# Naming-grammar basics: `<owner>_<role><group>`, group counted per (argument,
# role), no index on a boundary whose count the source signature already fixes.
# ---------------------------------------------------------------------------


# Several accesses to one argument, and accesses spread over two: the group
# index counts per argument, so `B`'s first read is `B_rd0` and not `B_rd2`.
def test_group_index_counts_per_argument():
    @kernel
    def multi_access(A: i32[16], B: i32[16], C: i32[16]):
        for i in range(16):
            C[i] = A[i] + A[15 - i] + B[i]

    assert _bases(_to_rtl(multi_access), "multi_access") == {
        "reads": [["A_rd0"], ["A_rd1"], ["B_rd0"]],
        "writes": [["C_wr0"]],
        "scalars": [],
        "results": [],
        "streams": [],
    }


# An accumulator argument takes one group per ROLE, so the read and the write
# are both group 0: the counter is per (argument, role), not per argument.
def test_read_and_write_groups_number_independently():
    @kernel
    def accumulate_arg(A: i32[8, 8], B: i32[8]):
        for i in range(8):
            for j in range(8):
                B[i] += A[i, j]

    assert _bases(_to_rtl(accumulate_arg), "accumulate_arg") == {
        "reads": [["B_rd0"], ["A_rd0"]],
        "writes": [["B_wr0"]],
        "scalars": [],
        "results": [],
        "streams": [],
    }


# A partitioned argument presents one interface per bank. A statically routed
# access still spans them in the manifest (the host needs every bank's layout);
# a data-dependent one crossbars over all of them.
def test_banked_argument_names_every_bank():
    @kernel
    def banked(A: i32[16], idx: i32[8], B: i32[8]):
        for i in range(8):
            B[i] = A[idx[i]]

    s = banked.schedule()
    s.partition("A", dim=1, kind=s.Cyclic, factor=4)
    rtl = s.export("rtl")
    assert _bases(rtl, "banked") == {
        "reads": [["idx_rd0"], ["A_rd0_b0", "A_rd0_b1", "A_rd0_b2", "A_rd0_b3"]],
        "writes": [["B_wr0"]],
        "scalars": [],
        "results": [],
        "streams": [],
    }
    # `factor` travels with every interface of the group; `bank` identifies it.
    group = rtl.interfaces["banked"].reads[1]
    assert [p.bank for p in group] == [0, 1, 2, 3]
    assert {p.factor for p in group} == {4}

    A = np.arange(16, dtype=np.int32)
    idx = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=np.int32)
    B = np.zeros(8, dtype=np.int32)
    rtl.cosim(A, idx, B)
    assert np.array_equal(B, A[idx])


# One counter numbers an array's groups across parent and child. `D` is
# written by the parent and then handed to a child instance; the two writes
# are serial, so the binding colours them onto one port and the child joins
# the parent's `D_wr0` group. The child's read is numbered after the parent's
# (absent) reads, so it is `D_rd0`.
def test_call_write_joins_the_parents_boundary_group():
    @kernel
    def _bump(D: i32[8]):
        for i in range(8):
            D[i] = D[i] + 1

    @kernel
    def seed_then_bump(D: i32[8], S: i32[8]):
        for i in range(8):
            D[i] = S[i]
        _bump(D)

    rtl = _to_rtl(seed_then_bump)
    assert _bases(rtl, "seed_then_bump") == {
        "reads": [["S_rd0"], ["D_rd0"]],  # parent's S, then the child's D read
        "writes": [["D_wr0"]],  # parent and child share the serial write bus
        "scalars": [],
        "results": [],
        "streams": [],
    }
    # The child module numbers its own boundary from scratch.
    assert _bases(rtl, "seed_then_bump__bump")["writes"] == [["D_wr0"]]

    D = np.zeros(8, dtype=np.int32)
    S = np.arange(8, dtype=np.int32)
    rtl.cosim(D, S)
    assert np.array_equal(D, S + 1)


# The non-memory boundaries: a scalar argument, a scalar result, and a stream
# handshake. None carries a group index, since the source signature fixes the
# count, so there is nothing a scheduling decision could renumber.
def test_scalar_result_and_stream_names_carry_no_index():
    @kernel
    def scalar_io(A: i32[16], s: i32) -> i32:
        acc: i32 = s
        for i in range(16):
            acc += A[i]
        return acc

    assert _bases(_to_rtl(scalar_io), "scalar_io") == {
        "reads": [["A_rd0"]],
        "writes": [],
        "scalars": ["s_in"],
        "results": ["ret_out"],
        "streams": [],
    }

    @kernel
    def prod(srm: Stream[i32]):
        for i in range(10):
            srm.put(i)

    @kernel
    def cons(srm: Stream[i32], out: i32[1]):
        acc: i32 = 0
        for i in range(10):
            acc += srm.get()
        out[0] = acc

    @kernel
    def top(out: i32[1]):
        srm: Stream[i32]
        prod(srm)
        cons(srm, out)

    rtl = _to_rtl(top)
    assert _bases(rtl, "top_prod")["streams"] == ["srm_st"]
    assert _bases(rtl, "top_cons")["streams"] == ["srm_st"]
    # The container forwards only the boundary array; the channel is internal.
    assert _bases(rtl, "top") == {
        "reads": [],
        "writes": [["out_wr0"]],
        "scalars": [],
        "results": [],
        "streams": [],
    }


# The SV-legal port-name grammar: a keyword-named argument still composes to a
# legal identifier (`input_in`), two same-named stream args split by direction,
# and a port group's index is stable as more groups get added to an argument.
def test_port_name_grammar():
    @kernel
    def kw(input: i32, wire: i32[16], reg: i32[16]):
        for i in range(16):
            reg[i] = wire[i] + input

    mod = _to_rtl(kw)
    iface = mod.interfaces[mod.top]
    # Every port carries a role suffix, so a keyword owner is already legal
    # once composed: `input_in`, not an escaped `input__in`.
    assert [s.name for s in iface.scalars] == ["input_in"]
    assert [r.base for acc in iface.reads for r in acc] == ["wire_rd0"]
    assert [w.base for acc in iface.writes for w in acc] == ["reg_wr0"]
    # Every manifest name appears verbatim in the emitted module header (a
    # rewritten `input_0` would not match `\binput_in\b`).
    head = mod.verilog.split(");", 1)[0]
    for name in ["input_in", "wire_rd0_addr", "wire_rd0_data", "reg_wr0_we"]:
        assert re.search(rf"\b{name}\b", head), f"{name} absent from {head}"
    out = np.zeros(16, np.int32)
    A16 = (np.arange(16, dtype=np.int32) * 7 + 13) & 0xFF
    mod.cosim(3, A16, out)  # the harness binds `input_in` by manifest name
    assert np.array_equal(out, A16 + 3)

    # A bare owner that is a keyword still needs the escape: an on-chip buffer
    # named `buf` carries no role suffix and collides with the gate primitive.
    # Deeper than the auto-partition threshold: a completely partitioned array
    # is realized per element, and those cells carry an index that already
    # separates them from the keyword.
    @kernel
    def kwbuf(A: i32[32], out: i32[32]):
        buf: i32[32] = 0
        for i in range(32):
            buf[i] = A[i] * 2
        for j in range(32):
            out[j] = buf[j] + 1

    assert re.search(r"\bbuf_\b", _to_rtl(kwbuf).verilog)

    # Stability: `A` reaches one read group above (`wire_rd0`) and two here.
    # The first group keeps index 0 either way, so growing the port set never
    # renames a port that already existed. A result carries a role, no index.
    @kernel
    def two_reads(A: i32[16]) -> i32:
        acc: i32 = 0
        for i in range(15):
            acc += A[i] * A[i + 1]
        return acc

    iface = _to_rtl(two_reads).interfaces["two_reads"]
    assert [r.base for acc in iface.reads for r in acc] == ["A_rd0", "A_rd1"]
    assert [r.name for r in iface.results] == ["ret_out"]

    # A relay chain: the middle process gets `fifo[p]` and puts `fifo[p+1]`, so
    # both stream arguments carry the source name `fifo`. The colliding pair is
    # split by direction; a process with a single stream keeps the plain name.
    N, P = 16, 3

    @kernel
    def chain(A: i32[N], out: i32[N]):
        fifo: Stream[i32, 2][P]

        @kernel(mapping=[P])
        def pe(A: i32[N], out: i32[N], fifo: Stream[i32, 2][P]):
            p = allo.get_wid(0)
            if p == 0:
                for i in range(N):
                    fifo[p + 1].put(A[i])
            elif p == P - 1:
                for i in range(N):
                    out[i] = fifo[p].get() + 1
            else:
                for i in range(N):
                    fifo[p + 1].put(fifo[p].get() * 2)

        pe(A, out, fifo)

    mod = _to_rtl(chain)
    bases = {k: [s.base for s in v.streams] for k, v in mod.interfaces.items()}
    assert sorted(bases["chain_pe_1"]) == ["fifo_st_in", "fifo_st_out"], bases
    assert bases["chain_pe_0"] == ["fifo_st"], bases
    assert bases["chain_pe_2"] == ["fifo_st"], bases

    out = np.zeros(N, np.int32)
    mod.cosim(A16, out)
    assert np.array_equal(out, A16 * 2 + 1)


# The manifest alone is a complete binding contract: the RTL module name vs.
# the legalized MLIR symbol, the fixed control ABI, and each extern operator's
# port roles. No cosim needed; the manifest is the only thing inspected.
def test_manifest_is_the_whole_binding_contract():
    @kernel
    def mk(A: f32[8], out: f32[8]):
        for i in range(8):
            out[i] = A[i] * 2.0

    iface = _to_rtl(mk).interfaces["mk"]
    assert iface.module == "mk" and iface.symbol == "mk"
    assert iface.control == Control("clk", "rst", "start", "done")
    # The f32 multiply is an IP: the manifest names its module, the device
    # operator it joins to, and each port's role (so `clk`/`ce`/the result are
    # found structurally, not by matching their names back out).
    ops = iface.operators
    assert ops, "an f32 multiply must instantiate an extern operator"
    roles = [p.role for p in ops[0].ports]
    assert roles.count(Operator.Role.DATA) >= 1 and roles.count(Operator.Role.CLK) == 1
    assert roles[-1] is Operator.Role.OUT and ops[0].impl

    # A nested callee's symbol carries a dot; the module name is the legalized
    # form, and the manifest is keyed by (and reports) that.
    N = 16

    @kernel
    async def mp(a: i32[N], s: Stream[i32]):
        for i in range(N):
            s.put(a[i] + 1)

    @kernel
    async def mc(s: Stream[i32], out: i32[N]):
        for i in range(N):
            out[i] = s.get() * 2

    @kernel
    async def mtop(a: i32[N], out: i32[N]):
        fifo: Stream[i32]
        await mp(a, fifo)
        await mc(fifo, out)

    ifaces = _to_rtl(mtop).interfaces
    assert ifaces.of_symbol("mtop.mp").module == "mtop_mp"
    assert all(k == i.module for k, i in ifaces.items()), "keyed by module name"


# Internal (non-manifest) cells still carry readable names: region-scoped
# control signals, a container loop's own IV, a delay tap's suffix, and an IP
# instance's port folding its source variable in, all surviving CIRCT
# uniquification without becoming an anonymous `_GEN_37`.
def test_internal_signal_names():
    @kernel
    def sx(a: i32, X: i32[16], Y: i32[16], Z: i32[16]):
        for i in range(16):
            # `buf` feeds the first multiply and then the multiply of its
            # result, so it stays live across the first core's latency
            # wherever the schedule places it (a single late read would be
            # re-placed next to its producer, and a mul-add pair would fuse).
            # That wait is the delay tap named below.
            buf: i32 = X[i] + a
            Z[i] = (buf * Y[i]) * buf

    v = _to_rtl(sx).verilog
    assert re.search(r"\bbuf_d1\b", v), "a delay tap keeps its value + delay"
    assert re.search(r"\br\d+_run\b", v) and re.search(r"\br\d+_done\b", v)
    assert re.search(r"\br\d+_v\d+\b", v), "the valid chain is region-tagged"

    @kernel
    def gv(A: f32[8, 8], x: f32[8], out: f32[8]):
        for i in range(8):
            acc: f32 = 0.0
            for j in range(8):
                acc += A[i, j] * x[j]
            out[i] = acc

    v = _to_rtl(gv).verilog
    # Both loop counters, including the outer (container) one. The declared
    # width is whatever each loop's own range needs, so this asks for the NAME
    # and leaves the width to the test that measures it.
    assert re.search(r"\breg\s+(\[\d+:0\]\s+)?i;", v), "container counter lost its IV"
    assert re.search(r"\breg\s+(\[\d+:0\]\s+)?j;", v)
    assert re.search(r"\br\d+_sv\d+\b", v), "the survivor latch is named"
    assert re.search(r"_acc_u\d+_y", v), "an IP result reaches the wire as `acc`"


# ---------------------------------------------------------------------------
# Boundary port groups: shared and disjoint accessors of one array argument.
# ---------------------------------------------------------------------------


# A boundary array read by two serial children (the minimal atax shape). The
# binding colours provably-serial accessors onto one port, so both children
# share the one A read group, each driving its address only in its own run
# window. `t` is the internal buffer chaining them; the loose RMW seed makes
# the container mixed (leaf-routed) and keeps the init live.
def test_two_serial_children_reading_one_boundary_array_share_a_group():
    @kernel
    def acc_t(A: i32[8], t: i32[8]):
        for i in range(8):
            t[i] += A[i]  # RMW internal t (needs the seed), reads boundary A

    @kernel
    def scale_t(A: i32[8], t: i32[8], out: i32[8]):
        for i in range(8):
            out[i] = A[i] * 2 + t[i]  # reads boundary A (SHARED) + internal t

    @kernel
    def shared_read_top(A: i32[8], out: i32[8]):
        t: i32[8]
        for i in range(8):
            t[i] = 0  # loose init region -> mixed container; seeds the RMW
        acc_t(A, t)  # adjacent calls, one region; both master A's read port
        scale_t(A, t, out)

    rtl = _to_rtl(shared_read_top)
    rd = [p for acc in rtl.interfaces["shared_read_top"].reads for p in acc]
    assert [p.base for p in rd] == ["A_rd0"]  # one group per (bank, port) colour
    assert {p.arg for p in rd} == {0}  # off the one boundary argument

    A = np.arange(8, dtype=np.int32) + 1
    out = np.zeros(8, dtype=np.int32)
    rtl.cosim(A, out)
    assert np.array_equal(out, A * 3)  # A*2 + t, with t == A after the RMW


# The write-side mirror: two children writing disjoint halves of one boundary
# array. The disjoint slices leave the calls free to overlap, so the binding
# colours the writes apart and each keeps its own group; each child gates its
# own we outside its run, so the idle master never writes. `s` is a
# loose-written internal both children read (mixed container).
def test_two_children_writing_one_boundary_array_mux_addr_data_we():
    @kernel
    def write_lo(s: i32[8], out: i32[8]):
        for i in range(4):
            out[i] = s[i] + 1  # writes out[0:4]

    @kernel
    def write_hi(s: i32[8], out: i32[8]):
        for i in range(4):
            out[i + 4] = s[i + 4] * 2  # writes out[4:8] (shares out's wr port)

    @kernel
    def shared_write_top(A: i32[8], out: i32[8]):
        s: i32[8]
        for i in range(8):
            s[i] = A[i] + 5  # loose region -> mixed; s read by both children
        write_lo(s, out)
        write_hi(s, out)

    A = np.arange(8, dtype=np.int32) + 1
    out = np.zeros(8, dtype=np.int32)
    _to_rtl(shared_write_top).cosim(A, out)
    exp = np.empty(8, dtype=np.int32)
    exp[:4] = (A[:4] + 5) + 1
    exp[4:] = (A[4:] + 5) * 2
    assert np.array_equal(out, exp)


# Two children writing DISJOINT slices of one boundary array run concurrently
# on the leaf: each accessor gets its own boundary port group (`B_wr0`,
# `B_wr1`), no serial mux, and the cosim harness backs both against the one
# array. The `dcp.instance` check pins the leaf path (the port-group naming
# alone also arises on the structural top).
def test_disjoint_writers_get_separate_port_groups():
    @kernel
    def pgw1(A: i32[16], B: i32[16]):
        for i in range(8):
            B[i] = A[i] + 1

    @kernel
    def pgw2(A: i32[16], B: i32[16]):
        for i in range(8):
            B[i + 8] = A[i + 8] * 2

    @kernel
    def pg_top(A: i32[16], B: i32[16]):
        pgw1(A, B)
        pgw2(A, B)  # disjoint slice of B -> concurrent, its own port group

    rtl = _to_rtl(pg_top)
    assert Dcp(rtl).func(rtl.top).callees()  # leaf CallUnit path (structural lock)
    wr = [w[0] for w in rtl.interfaces["pg_top"].writes]
    assert [w.base for w in wr] == ["B_wr0", "B_wr1"]  # per accessor, not a mux
    assert {w.arg for w in wr} == {1}  # both groups master the same argument
    A = np.arange(16, dtype=np.int32) + 1
    B = np.zeros(16, np.int32)
    rtl.cosim(A, B)
    assert np.array_equal(B, np.concatenate([A[:8] + 1, A[8:] * 2]))


# A boundary array the parent writes and a child then read-modify-writes. The
# shared-memref sibling edge keeps them serial, so the two writes share one
# top port group, the child's arm selected on its run window.
def test_call_shares_boundary_arg_with_parent_write():
    @kernel
    def sb_child(B: i32[16]):
        for i in range(16):
            B[i] = B[i] + 1

    @kernel
    def sb_top(A: i32[16], B: i32[16]):
        for i in range(16):
            B[i] = A[i] * 2  # parent masters B's write port group 0
        sb_child(B)  # child masters its own read + write groups

    A = (np.arange(16, dtype=np.int32) * 3 + 1) & 0x3F
    B = np.zeros(16, dtype=np.int32)
    _to_rtl(sb_top).cosim(A, B)
    assert np.array_equal(B, A * 2 + 1)


# The read side: a child and the parent both read one boundary argument. The
# sibling chain keeps them serial, so both drive the one read group.
def test_call_shares_boundary_arg_with_parent_read():
    @kernel
    def sbr_child(A: i32[16], out: i32[16]):
        for i in range(16):
            out[i] = A[i] * 5

    @kernel
    def sbr_top(A: i32[16], out: i32[16], t: i32[16]):
        sbr_child(A, out)  # child reads A
        for i in range(16):
            t[i] = A[i] + 100  # the parent reads A too

    A = (np.arange(16, dtype=np.int32) * 3 + 1) & 0x3F
    out = np.zeros(16, dtype=np.int32)
    t = np.zeros(16, dtype=np.int32)
    _to_rtl(sbr_top).cosim(A, out, t)
    assert np.array_equal(out, A * 5)
    assert np.array_equal(t, A + 100)


# --- the host boundary --------------------------------------------------------
# A width the host cannot name (anything but 8/16/32/64) crosses in a wider numpy
# container while the design's ports stay as wide as the type, so the host
# truncates on the way in and sign-extends on the way out. The values below set
# the top bit of the 48-bit type, where a missing extension shows.
def test_nonstandard_width_crosses_the_host_boundary():
    i48 = APInt(48, signed=True)

    @kernel
    def copy48(a: i48[4], out: i48[4]):
        for k in range(4, name="k"):
            out[k] = a[k]

    @kernel
    def scale48(a: i48[4]) -> i48:
        acc: i48 = 0
        for k in range(4, name="k"):
            acc = acc + a[k]
        return acc

    vals = np.array([-1, -(2**47), 5, 2**47 - 1], np.int64)
    out = np.zeros(4, np.int64)
    _to_rtl(copy48).cosim(vals, out)
    assert np.array_equal(out, vals), f"{out} != {vals}"

    total = ((int(vals.sum()) + 2**47) % 2**48) - 2**47
    assert int(_to_rtl(scale48).cosim(vals).result) == total


# The same kernel on the CPU (which widens the boundary in the IR) and in cosim
# (which keeps the exact width and closes the gap on the host) must agree.
def test_nonstandard_width_agrees_with_cpu():
    i48 = APInt(48, signed=True)

    @kernel
    def dot48(x: i32[8], y: i32[8]) -> i48:
        acc: i48 = 0
        for k in range(8, name="k"):
            acc = acc + x[k] * y[k]
        return acc

    rng = np.random.default_rng(2)
    x = rng.integers(-(2**31), 2**31, size=8, dtype=np.int64).astype(np.int32)
    y = rng.integers(-(2**31), 2**31, size=8, dtype=np.int64).astype(np.int32)
    exact = sum(int(a) * int(b) for a, b in zip(x, y))
    assert ((exact + 2**47) % 2**48) - 2**47 < 0, "the wrapped value must be negative"
    # The product narrows to 48 bits, and a 48-bit combinational multiply
    # measures past the default clock's period on this part.
    assert int(_to_rtl(dot48, freq_mhz=200).cosim(x, y).result) == int(dot48(x, y))


# A boundary made of ports rather than C types carries any bit layout, which is
# what `RTL_ABI` declares: binary16 and bfloat16 cross as their own 16 bits.
def test_narrow_floats_cross_the_host_boundary():
    @kernel
    def hcopy(a: f16[8], out: f16[8]):
        for k in range(8, name="k"):
            out[k] = a[k]

    @kernel
    def bfsum(a: bf16[8], b: bf16[8], out: bf16[8]):
        for k in range(8, name="k"):
            out[k] = a[k] + b[k]

    # 65504 is the largest binary16, 6e-8 a subnormal: both survive only if the
    # boundary reads the format's own layout rather than the nearest-looking one.
    a = np.array([1.5, -2.25, 65504.0, 6e-8, 0.0, -0.5, 3.14159, -1e-5], np.float16)
    out = np.zeros(8, np.float16)
    _to_rtl(hcopy).cosim(a, out)
    assert np.array_equal(out, a), f"{out} != {a}"

    x = np.array([1.5, -2.25, 3.5, 7.0, -8.0, 0.5, 100.0, -0.25], ml_dtypes.bfloat16)
    y = np.array([0.5, 1.25, -3.5, 2.0, 8.0, 0.25, 28.0, 0.75], ml_dtypes.bfloat16)
    z = np.zeros(8, ml_dtypes.bfloat16)
    _to_rtl(bfsum).cosim(x, y, z)
    assert np.array_equal(z, x + y), f"{z} != {x + y}"


# An integer past every numpy container has no array form, but a scalar one
# crosses as a Python int, which has no width to run out of.
def test_wide_scalar_crosses_as_a_python_int():
    i128 = APInt(128, signed=True)

    @kernel
    def wide(x: i32[8], y: i32[8]) -> i128:
        acc: i128 = 0
        for k in range(8, name="k"):
            acc = acc + x[k] * y[k]
        return acc

    big = np.full(8, 2**31 - 1, np.int32)
    exact = sum(int(p) * int(q) for p, q in zip(big, big))
    assert exact.bit_length() > 64, "the sum must not fit a 64-bit container"
    assert _to_rtl(wide).cosim(big, big).result == exact

    rng = np.random.default_rng(2)
    x = rng.integers(-(2**31), 2**31, size=8, dtype=np.int64).astype(np.int32)
    y = rng.integers(-(2**31), 2**31, size=8, dtype=np.int64).astype(np.int32)
    negative = sum(int(p) * int(q) for p, q in zip(x, y))
    assert negative < 0, "a negative case pins the sign extension"
    assert _to_rtl(wide).cosim(x, y).result == negative
