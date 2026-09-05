# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pipeline elasticity (the region-wide stall shell) and clock-frequency-aware chaining/timing."""

import os
import re
import shutil
import sys

import numpy as np
import pytest

import allo
from allo import kernel
from allo.lang import i32, f32, index, Stream
from allo.backend.rtl.devices import default_device

sys.path.insert(0, os.path.dirname(__file__))
from _common import (  # noqa: E402
    Mod,
    _impls,
    _sched,
    _to_rtl,
    _iis,
    comb_ns,
    comb_step_ns,
    IMUL64,
    REG_NS,
    PERIOD_NS,
)

pytestmark = pytest.mark.skipif(
    shutil.which("verilator") is None, reason="verilator not available"
)

_STALLS = [0.0, 0.5, 0.8]

# A cell carrying the region's TIME BASE: a valid-chain stage (`r1_v3`) or a
# register tap (`acc_d2`). Survivors (`r1_sv0`) are excluded, since a survivor
# is enabled by its own capture pulse, not by the shell.
_TIME_BASE = re.compile(r"^(r\d+_v\d+|.+_d\d+)$")


class _Mod(Mod):
    # Mod plus the time-base classification this file's locks read.

    def time_base(self):
        # (label, register, input) of every time-base cell.
        return [(lb, r, i) for lb, r, i in self.regs if _TIME_BASE.match(lb)]


# --- elasticity: one shell per region ----------------------------------------


# An elastic region's chain stages all ride ONE `chainEnable`. Each of these
# cells is built by a different helper (register chain, valid-delay, put/get
# pulses), and they agree only because each names the same region's shell.
def test_one_shell_enables_every_time_base_cell():
    @kernel
    def stage(x_in: Stream[i32], y_out: Stream[i32]):
        for i in range(16):
            y_out.put(x_in.get() + 7)

    rtl = _to_rtl(stage)
    m = _Mod(rtl.mlir, "stage")

    ce = m.hinted("r0_ce")
    assert m.hints_like(r"_ce$") == ["r0_ce"], "one region, one shell"

    cells = m.time_base()
    assert cells, "an elastic region must have time-base cells to freeze"
    enables = {m.enable_of(reg, inp) for _, reg, inp in cells}
    assert enables == {ce}, f"time-base cells not on one shell: {enables}"

    # G's half: issue is the run flag gated by the shell.
    issue = m.hinted("r0_issue")
    assert ce in m.cone(issue)

    # The done drain is held through back-pressure by the same signal, so the
    # region cannot report completion on a token that was never accepted.
    done_reg, done_in = m.reg_named("r0_done")
    assert done_reg  # the latch itself
    assert ce in m.cone(done_in)

    x = np.arange(16, dtype=np.int32) * 5 - 3
    for gap in _STALLS:
        y = np.zeros(16, dtype=np.int32)
        rtl.cosim(x, y, stall_prob=gap)
        assert np.array_equal(y, x + 7), f"gap={gap}: {list(y)}"


# The two halves part on what a blocked handshake blocks. An empty input holds
# back only the pass that would read it, so the datapath keeps advancing and
# the tokens already in flight still leave: `_valid` reaches `issueEnable` and
# not `chainEnable`. Draining under starvation is what lets a feedback cycle
# through two processes turn on fewer tokens than the pipeline is deep.
def test_a_starved_input_defers_the_pass_without_freezing_the_chain():
    @kernel
    def defer(x_in: Stream[i32], y_out: Stream[i32]):
        for i in range(16):
            y_out.put(x_in.get() * 3 + 7)

    rtl = _to_rtl(defer)
    m = _Mod(rtl.mlir, "defer")
    assert m.hints_like(r"_ien$") == ["r0_ien"], "the shell did not split"
    ce, ien = m.hinted("r0_ce"), m.hinted("r0_ien")

    assert "x_in_st_valid" in m.cone(ien), "issue does not wait for its input"
    assert "x_in_st_valid" not in m.cone(ce), "a starved input froze the chain"
    # An in-flight token that cannot leave still freezes it: back-pressure on
    # the output reaches both halves.
    assert "y_out_st_ready" in m.cone(ce)
    assert ien in m.cone(m.signal("r0_issue"))
    # And the chain cells all stay on F's half.
    assert {m.enable_of(reg, inp) for _, reg, inp in m.time_base()} == {ce}

    x = np.arange(16, dtype=np.int32) * 5 - 3
    for gap in _STALLS:
        y = np.zeros(16, dtype=np.int32)
        rtl.cosim(x, y, stall_prob=gap)
        assert np.array_equal(y, x * 3 + 7), f"gap={gap}: {list(y)}"


# The one region that cannot take the bubble a deferred pass leaves. A value
# read back by a later iteration off a register tap is indexed in cycles, so a
# skipped issue would fold the slot beside it in. Such a region freezes on a
# starved input instead, which is the two halves collapsing back into one.
def test_a_register_recurrence_freezes_rather_than_deferring():
    @kernel
    def fold(x_in: Stream[i32], out: i32[1]):
        a: i32 = 0
        for i in range(16):
            a += x_in.get()
        out[0] = a

    rtl = _to_rtl(fold)
    m = _Mod(rtl.mlir, "fold")
    assert m.hints_like(r"_ien$") == [], "an accumulator region may not bubble"
    assert "x_in_st_valid" in m.cone(m.hinted("r0_ce"))

    x = np.arange(16, dtype=np.int32) * 5 - 3
    for gap in _STALLS:
        out = np.zeros(1, dtype=np.int32)
        rtl.cosim(x, out, stall_prob=gap)
        assert out[0] == x.sum(), f"gap={gap}: {out[0]} != {x.sum()}"


# A clock-enabled IP's `ce` port IS the region's `chainEnable`. The shell is
# consumed at the IP boundary too: a free-running IP would keep clocking
# while the shift chains are frozen and fold a stale result.
def test_ce_ip_rides_the_region_shell():
    @kernel
    def fstage(x_in: Stream[f32], y_out: Stream[f32]):
        for i in range(16):
            y_out.put(x_in.get() * 2.0 + 1.0)

    rtl = _to_rtl(fstage)
    m = _Mod(rtl.mlir, "fstage")
    ce = m.hinted("r0_ce")

    ports = re.findall(r"hw\.instance \"(\w+)\" @\w+\((.*?)\) ->", rtl.mlir)
    assert len(ports) >= 2, f"expected the fmul -> fadd chain, got {ports}"
    for name, args in ports:
        got = re.search(r"ce: %([\w.$-]+):", args)
        assert got, f"instance {name} has no ce port: {args}"
        assert got.group(1) == ce, f"instance {name} rides {got.group(1)}, not {ce}"

    fx = (np.arange(16, dtype=np.float32) * 0.5 - 3.0).astype(np.float32)
    for gap in _STALLS:
        fy = np.zeros(16, dtype=np.float32)
        rtl.cosim(fx, fy, stall_prob=gap)
        assert np.allclose(fy, fx * 2.0 + 1.0), f"gap={gap}: {list(fy)}"


# A banked memory read inside a stream region freezes with the chain. Both
# halves of the split (bank and offset) are held by the same enable: a
# disagreement about when to freeze would read the wrong element.
def test_held_read_address_rides_the_region_shell():
    @kernel
    def banked(A: i32[32], y_out: Stream[i32]):
        for i in range(32):
            y_out.put(A[i] * 3)

    s = banked.schedule()
    s.partition("A", dim=1, kind=s.Cyclic, factor=4)
    rtl = s.export("rtl")
    m = _Mod(rtl.mlir, "banked")
    ce = m.hinted("r0_ce")

    # Every self-holding cell in the region, chain stages and held address
    # halves alike, is enabled by the one shell.
    held = {m.enable_of(reg, inp) for _, reg, inp in m.regs}
    assert held - {None} == {ce}, f"not one shell: {held}"

    A = np.arange(32, dtype=np.int32) * 7 - 11
    for gap in _STALLS:
        y = np.zeros(32, dtype=np.int32)
        rtl.cosim(A, y, stall_prob=gap)
        assert np.array_equal(y, A * 3), f"gap={gap}: {list(y)}"


# No stream accesses => no shell, and no trace of one in the RTL. A rigid
# shell is the IDENTITY: every timing primitive reduces to its unconditional
# form, not a constant-true-enabled special case.
def test_rigid_region_emits_no_shell():
    @kernel
    def gemm(A: f32[8, 8], B: f32[8, 8], C: f32[8, 8]):
        for i, j in allo.grid(8, 8):
            acc: f32 = 0.0
            for k in range(8):
                acc += A[i, k] * B[k, j]
            C[i, j] = acc

    rtl = _to_rtl(gemm)
    m = _Mod(rtl.mlir, "gemm")

    assert m.hints_like(r"_ce$") == [], "a rigid region derives no shell"
    cells = m.time_base()
    assert cells, "the deep f32 datapath must emit valid-chain stages"
    for label, reg, inp in cells:
        assert m.enable_of(reg, inp) is None, f"{label} is enabled under a rigid shell"

    A = np.random.rand(8, 8).astype(np.float32)
    B = np.random.rand(8, 8).astype(np.float32)
    C = np.zeros((8, 8), dtype=np.float32)
    rtl.cosim(A, B, C)
    assert np.allclose(C, A @ B, atol=1e-4), C


# --- multi-cycle write timing ------------------------------------------------


def _dev(write_latency: int):
    # The default device with the default on-chip storage rebound to a
    # write_latency-cycle write.
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


# The registers that carry a multi-cycle write ride the region's clock
# enable, so a stream region's back-pressure freezes the in-flight write with
# the rest of the datapath instead of committing it a cycle early.
def test_multi_cycle_write_freezes_under_back_pressure():
    @kernel
    def strbuf(out: i32[8]):
        fifo: Stream[i32]

        @kernel(mapping=[2])
        def pe(out: i32[8], fifo: Stream[i32]):
            p = allo.get_wid(0)
            if p == 0:
                for i in range(8):
                    fifo.put(i * 3)
            else:
                buf: i32[8] = 0
                for i in range(8):
                    buf[i] = fifo.get() + 1
                for i in range(8):
                    out[i] = buf[i]

        pe(out, fifo)

    expect = np.arange(8, dtype=np.int32) * 3 + 1
    for wr in (1, 2, 3):
        for gap in (0.0, 0.6):
            out = np.zeros(8, dtype=np.int32)
            _to_rtl(strbuf, device=_dev(wr)).cosim(out, stall_prob=gap)
            assert np.array_equal(out, expect), f"wr_lat={wr} gap={gap}: {list(out)}"


# --- clock-frequency-aware chaining -------------------------------------------


# The timing/chaining model is clock-frequency sensitive: a combinational
# int-add chain too deep for one cycle splits across more cycles under a tight
# clock than under a loose one.
_ADD_CHAIN = 8


def test_chaining_inserts_register():
    def chain():
        @kernel
        def c(A: i32[8], out: i32[8]):
            for i in range(8):
                t1: i32 = A[i] + A[i]
                t2: i32 = t1 + A[i]
                t3: i32 = t2 + A[i]
                t4: i32 = t3 + A[i]
                t5: i32 = t4 + A[i]
                t6: i32 = t5 + A[i]
                t7: i32 = t6 + A[i]
                out[i] = t7 + A[i]

        return c

    # The premise, stated against the device rather than assumed: this many
    # chained int adds do not fit one default cycle. The register floor is paid
    # once per cycle and each add contributes its own step on top, which is the
    # sum the chaining solve cuts against.
    assert REG_NS + _ADD_CHAIN * comb_step_ns("add") > PERIOD_NS
    # So the chaining scheduler splits the chain across cycles, leaving more
    # register stages than a huge cycle time, where the whole chain settles in one.
    tight = _sched(chain()).cyclic()[0]
    loose = _sched(chain(), freq_mhz=1.0).cyclic()[0]  # a 1000ns cycle
    assert tight.last_t() > loose.last_t()


def test_a_symbolic_bound_binds_like_any_other_arithmetic():
    # A symbolic loop bound is an affine MAP, and the constraint system has a
    # vertex only for an operation, so it used to be expanded after the solve
    # and reach the datapath as one combinational cone nothing could break.
    # `expand-region-bounds` reifies it before the solve, at the datapath's
    # index width: `index` carries no width for an operator row to be priced at
    # or for an IP signature to match, so the same divide that is unbuildable
    # combinationally binds the device's divider core once it is typed.
    def band():
        @kernel
        def k(A: i32[64], out: i32[8]):
            for i in range(8):
                s: i32 = 0
                for j in range(i // 3 * 2 + 1):  # a floordiv bound map
                    s = s + A[j]
                out[i] = s

        return k

    # The premise, against the device: a signed floordiv expands to a divider
    # plus its sign correction (cmp, sub, select on each side), and the divider
    # alone overruns the default period, so a combinational one is not
    # buildable at all.
    assert (
        REG_NS + comb_step_ns("div") + comb_step_ns("sub") + comb_step_ns("select")
        > PERIOD_NS
    )

    # A cell the emitter reaches with more delay on its inputs than the schedule
    # left it is a REFUSAL (`checkCombPathsMeetPeriod`), so compiling at all is
    # what says the bound was scheduled against the same clock as everything
    # else.
    _to_rtl(band()).compile()

    # A device core carries it rather than an unrealizable index cone: the
    # constant divide expands to a reciprocal multiply on the pipelined
    # multiplier, so the region holding it is at least as deep as that core.
    res = _sched(band())
    assert any(im.startswith("mulw_i64") for im in _impls(res))
    spans = [r for r in res.regions() if r.kind == "acyclic" and r.ops]
    assert max(r.last_t() for r in spans) >= IMUL64


def test_a_sequential_whiles_condition_is_cut_like_any_other_chain():
    # A sequential (CHECK/RUN) while's continue-condition is the last expression
    # the reifier used to synthesize after the solve. The scheduler solves it as
    # its own straight-line span now, so `t_cond`, the cycles the controller
    # waits before deciding, is a CUT depth rather than an ASAP walk over
    # unpriced cells.
    def loop():
        @kernel
        def k(A: i32[64], out: i32[1]):
            i: i32 = 0
            s: i32 = 0
            # The load is what forces the CHECK/RUN controller; the arith behind
            # it is what has to be cut. Adds and not a multiply: an integer
            # multiply binds to a DSP core, whose latency the solve cannot cut.
            while A[i] + A[i] + A[i] + A[i] + A[i] + A[i] + A[i] + A[i] < 100:
                s += A[i]
                i += 1
            out[0] = s

        return k

    # The premise, against the device: the condition's own chain does not fit
    # one default cycle, so it is only buildable if something breaks it. Seven
    # adds chain the eight terms, and the register floor is paid once.
    assert REG_NS + 7 * comb_step_ns("add") + comb_step_ns("cmp") > PERIOD_NS

    def cond_depth(res):
        # The conditional container's own ops ARE the condition cone: its body
        # is a nested region.
        cone = [r for r in res.func("k").regions if r.conditional and r.ops]
        assert len(cone) == 1, cone
        return cone[0].last_t()

    assert cond_depth(_sched(loop())) > cond_depth(
        _sched(loop(), freq_mhz=1.0)  # a 1000ns cycle
    )

    # As above: a path the schedule did not leave room for is a refusal, so the
    # compile completing is the check.
    rtl = _to_rtl(loop())
    rtl.compile()

    # And the multi-cycle wait is real hardware: the controller must hold RUN
    # off until the condition settles, or it runs an iteration too many.
    A = (np.arange(64, dtype=np.int32) * 7) % 90
    out = np.zeros(1, np.int32)
    rtl.cosim(A, out)
    i = s = 0
    while A[i] * 3 + A[i] * 5 + 11 < 100:
        s += int(A[i])
        i += 1
    assert out[0] == s


def test_an_address_cone_is_charged_to_the_port_it_feeds():
    # An address never becomes an operation: it is folded into the access's
    # affine map, so no dependence carries its delay and only the access's own
    # operator type can account for it. These two kernels run the same four adds
    # over the same trip count and differ only in what it costs to reach the
    # element -- `flat` addresses with the bare counter, `cone` sums three
    # shifted terms.
    @kernel
    def flat(A: i32[512], B: i32[512], out: i32[512]):
        for i in range(64):
            out[i] = A[i] + B[i] + A[i] + B[i]

    @kernel
    def cone(A: i32[8, 8, 8], B: i32[8, 8, 8], out: i32[8, 8, 8]):
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    out[i + 1, j + 1, k + 1] = (
                        A[i, j, k] + B[i, j, k] + A[i, j, k] + B[i, j, k]
                    )

    def fits(k, mhz):
        # The schedule keeps the asked-for period only while every cone fits
        # it; one past the period lowers the clock instead of failing.
        return _sched(k, freq_mhz=mhz).cycle_ns < 1000.0 / mhz + 1e-6

    base = default_device.default_freq_mhz
    assert fits(flat, base) and fits(cone, base)

    parted = [
        mhz
        for mhz in range(int(base) + 10, 501, 10)
        if fits(flat, mhz) and not fits(cone, mhz)
    ]
    assert parted, "no clock charges the address cone more than the bare counter"


def test_an_address_that_follows_the_counters_is_carried_in_a_register():
    # Address strength reduction. Every term of `i*400 + j*20 + k + c` is a
    # constant multiple of an enclosing counter, so consecutive iterations
    # differ by a constant: each term becomes a register the controller
    # advances beside the counter it follows, and the address is their sum. The
    # constant multiplies -- the arithmetic that dominates an address, and the
    # reason it was the widest cone in the datapath -- are gone entirely.
    @kernel
    def stencil(A: i32[20, 20, 20], out: i32[20, 20, 20]):
        for i in range(18):
            for j in range(18):
                for k in range(18):
                    out[i + 1, j + 1, k + 1] = A[i, j, k] + 1

    mod = _to_rtl(stencil)
    m = mod.mlir
    assert "comb.mul" not in m, "a constant stride survived on the address path"
    # One scaled counter per non-unit level: the outer two multiply their
    # counter (400, 20) and ride their own registers, at the outermost also a
    # second one whose reset value carries `out`'s constant 421 off the memory
    # port's setup. The innermost term is the bare counter `k` for both accesses
    # (`out`'s `k+1` puts the 1 in the base, not the stride), so it reads the `k`
    # register directly rather than duplicating it into an `r2_addr0`.
    assert sorted(set(re.findall(r"r\d+_addr\d+", m))) == [
        "r0_addr0",
        "r0_addr1",
        "r1_addr0",
    ]
    inits = dict(
        re.findall(r"%(r\d+_addr\d+) = seq\.compreg [^\n]*reset %rst, %(\w+)", m)
    )
    consts = dict(re.findall(r"%(\w+) = hw\.constant (-?\d+)", m))
    assert sorted(consts[inits[r]] for r in ("r0_addr0", "r0_addr1")) == ["0", "421"]

    A = (np.arange(8000, dtype=np.int32) % 251).reshape(20, 20, 20)
    out = np.zeros((20, 20, 20), np.int32)
    mod.cosim(A, out)
    exp = np.zeros((20, 20, 20), np.int32)
    exp[1:19, 1:19, 1:19] = A[0:18, 0:18, 0:18] + 1
    assert np.array_equal(out, exp)


def test_a_unit_stride_address_reads_the_counter_and_builds_no_register():
    # A term that is `1 * counter + 0` is the counter, so the reduction builds
    # no register for it: the access reads the counter directly and no `r0_addr`
    # is built. `B[i]`'s write and `A[i]`'s read share the one counter.
    @kernel
    def copy(A: i32[64], B: i32[64]):
        for i in range(64):
            B[i] = A[i] + 1

    mod = _to_rtl(copy)
    m = mod.mlir
    assert not re.findall(r"r\d+_addr\d+", m), "a unit-stride address kept a register"
    # The report agrees: a counter-aliased stride counts as no register and is
    # flagged is_counter.
    region = mod.microarch.funcs[0].regions[0]
    assert region.cost.addr_strides == 0, region.cost.addr_strides
    assert any(s.is_counter for s in region.cost.strides), region.cost.strides

    A = (np.arange(64, dtype=np.int32) * 5 + 1) & 0xFF
    B = np.zeros(64, np.int32)
    mod.cosim(A, B)
    assert np.array_equal(B, A + 1)


def test_a_subscript_that_cannot_be_carried_keeps_the_row_its_register():
    # PARTIAL strength reduction. An address is not one decision. `A[i, c]` has a
    # row that follows a counter and a column that never can: `c` is a boundary
    # scalar, so no register advances with it. Taking the address as one decision
    # would cost the row its register as well, rebuilding `i*20` every cycle to
    # add `c` to it. The row reduces on its own.
    @kernel
    def colsum(A: i32[12, 20], c: index, out: i32[12]):
        for i in range(12):
            out[i] = A[i, c]

    mod = _to_rtl(colsum)
    m = mod.mlir
    # The row stride is a register the controller advances by 20, not a multiply
    # on the address path: 20 is no power of two, so one left there would be a
    # visible `comb.mul` by it (`mulConst` leaves the recoding to synthesis).
    # Asked of that constant rather than of `comb.mul` at large, since a runtime
    # loop bound negates with one too and that is control, not address.
    # Any width: the stride register is built at the range it walks, not at the
    # counter's width, so `20` is a constant of that register's own type.
    twenty = set(re.findall(r"(%c20_i\d+\w*) = hw\.constant 20", m))
    assert twenty, "no stride of 20 anywhere: the test measures nothing"
    # The advance-by-20 reads the stride register, either directly or through
    # its start-cycle bypass mux.
    carriers = {f"%r0_addr{k}" for k in re.findall(r"%r0_addr(\d+)\b", m)}
    carriers |= {
        f"%{name}"
        for name in re.findall(r"%([\w.]+) = comb\.mux [^\n]*%r0_addr\d+\b", m)
    }
    assert any(
        re.search(rf"comb\.add {re.escape(o)}, {c}\b", m)
        for o in carriers
        for c in twenty
    ), "the row stride is not carried in a register"
    assert not any(
        re.search(rf"comb\.mul [^\n]*{c}\b", m) for c in twenty
    ), "a row stride survived beside a column that did"
    A = (np.arange(240, dtype=np.int32) % 251).reshape(12, 20)
    out = np.zeros(12, np.int32)
    mod.cosim(A, 7, out)
    assert np.array_equal(out, A[:, 7])


def test_normalizing_a_strided_loop_lets_its_nest_coalesce():
    # Loop normalization, and what it is FOR. Coalescing states a precondition
    # nothing else establishes (lower bound 0, step 1), so a nest `s.unroll`
    # left stepping by 2 would be refused for a property nothing fixes.
    # Normalized, the step moves into the subscript and the band coalesces into
    # one region running at II=1.
    #
    # The stride is on the INNER loop in the first kernel and on the OUTER loop
    # in the second, and the two are not the same case. Normalizing leaves the
    # original induction variable behind as an `affine.apply`; on an outer level
    # that op stands between the two loops, so the nest stops being perfect and
    # the normalization meant to open the band is what closes it. It only
    # coalesces because the leftover is sunk into the innermost body.
    @kernel
    def inner_stride(A: i32[8, 8], out: i32[8, 8]):
        for i in range(8):
            for j in range(0, 8, 2):
                out[i, j] = A[i, j] + 1
                out[i, j + 1] = A[i, j + 1] + 1

    @kernel
    def outer_stride(A: i32[8, 8], out: i32[8, 8]):
        for i in range(0, 8, 2):
            for j in range(8):
                out[i, j] = A[i, j] + 1
                out[i + 1, j] = A[i + 1, j] + 1

    A = (np.arange(64, dtype=np.int32) % 251).reshape(8, 8)
    for k in (inner_stride, outer_stride):
        mod = _to_rtl(k)
        assert (
            len(mod.schedule().func(k.__name__).cyclic(wrappers=True)) == 1
        ), f"{k.__name__}: the strided band did not coalesce into one region"
        # Both forms store two elements of `out` and distributed RAM has one
        # write port, so the coalesced band issues one iteration every 2 cycles.
        assert _iis(mod.schedule().func(k.__name__).regions) == [2]
        out = np.zeros((8, 8), np.int32)
        mod.cosim(A, out)
        assert np.array_equal(out, A + 1)


# ---------------------------------------------------------------------------
# The estimated clock
# ---------------------------------------------------------------------------


# The QoR reports fmax from the longest modelled path. This kernel shares no
# port and no unit, so nothing is built after the schedule's cut and the
# estimate stays inside the target period.
def test_the_qor_publishes_the_clock_the_model_holds():
    @kernel
    def vsum(A: i32[64], B: i32[64], C: i32[64]):
        for i in range(64):
            C[i] = A[i] + B[i]

    rtl = _to_rtl(vsum)
    est = rtl.estimation
    period = 1000.0 / est.fmax_target
    worst = max(f.critical_ns for f in rtl.microarch.funcs)
    assert worst > 0 and est.fmax == pytest.approx(1000.0 / worst)
    assert worst <= period + 1e-9
    assert est.fmax >= est.fmax_target

    # A path starts at a register or port launch and its steps sum to its
    # total; several paths are reported per compile.
    assert len(est.critical_paths) > 1
    for p in est.critical_paths:
        assert p.steps and p.steps[0].what.startswith("launch:")
        assert sum(s.delay for s in p.steps) == pytest.approx(p.total)
        assert p.slack == pytest.approx(period - p.total)


# ---------------------------------------------------------------------------
# The value-chain report
# ---------------------------------------------------------------------------


# The microarch report holds one row per value delay chain of the model, with
# the value range the interval walk proves through its driving cells. Reading
# the IV as data forces a chain to align it with the memory read, and the walk
# follows it back to the counter's constant bounds.
def test_a_chain_report_carries_the_proven_range():
    @kernel
    def ranged(a: i32[16], out: i32[16]):
        for i in range(16):
            out[i] = a[i] * 3 + i

    rtl = _to_rtl(ranged)
    f = rtl.microarch.func("ranged")
    assert f.chains, "the IV must be delayed to meet the read's landing"
    from_counter = [c for c in f.chains if c.source == "counter"]
    assert from_counter, f"no counter-fed chain in {f.chains}"
    for c in from_counter:
        # i runs [0, 16); the one-past value 16 needs 5 unsigned bits, and the
        # non-negative counter is built unsigned, so range never exceeds width.
        assert c.range_bits == 5 and c.width >= c.range_bits
    # Chains are a subset of the value-role ledger: read-data alignment delays
    # are charged there but built outside the model.
    from allo.backend.rtl.reports.microarch import RegRole

    plain = sum(c.width * c.depth for c in f.chains)
    assert 0 < plain <= f.reg_bits_by_role()[RegRole.VALUE]

    x = np.arange(16, dtype=np.int32)
    out = np.zeros(16, np.int32)
    rtl.cosim(x, out)
    assert np.array_equal(out, x * 3 + np.arange(16))


# An address cone delayed to a deep access stage holds one datum per in-flight
# iteration, so at II > 1 it folds onto the region's phase: ceil(stage / II)
# enabled registers rather than one per cycle of the stage.
def test_a_deep_access_address_folds_onto_the_phase():
    @kernel
    def deepstore(A: f32[16], B: f32[16]):
        acc: f32 = 1.0
        for i in range(16):
            acc = acc * A[i]  # recurrence: II = the multiplier's latency
            t: f32 = acc + acc
            t = t * t
            t = t * t
            B[i] = t  # lands stages past the II; its address rides a chain

    s = deepstore.schedule()
    s.pipeline(ii=1).apply()
    rtl = s.export("rtl")
    from allo.backend.rtl.reports.microarch import RegRole

    f = rtl.microarch.func("deepstore")
    deep = [c for c in f.regs if c.role == RegRole.VALUE and c.depth > 1]
    assert deep, "the deep store's address must ride a multi-stage chain"
    assert all(c.enable for c in deep), "an unfolded per-cycle address chain"

    x = np.linspace(1.0, 1.3, 16, dtype=np.float32)
    out = np.zeros(16, np.float32)
    rtl.cosim(x, out)
    acc, ref = np.float32(1.0), np.zeros(16, np.float32)
    for i in range(16):
        acc = acc * x[i]
        t = acc + acc
        t = t * t
        t = t * t
        ref[i] = t
    assert np.array_equal(out, ref)
