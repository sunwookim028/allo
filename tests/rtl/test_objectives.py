# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scheduler knobs: the ``O`` direction, the clock margin and the search loop."""

import os
import shutil
import sys

import numpy as np
import pytest

from allo import kernel
from allo.lang import f32, i32

sys.path.insert(0, os.path.dirname(__file__))
from _common import _impls, _to_rtl  # noqa: E402
from allo.backend.rtl.devices import default_device  # noqa: E402

pytestmark = pytest.mark.skipif(
    shutil.which("verilator") is None, reason="verilator not available"
)


def _mixed_kernel():
    @kernel
    def mx(A: i32[32], B: i32[32], out: i32[1]):
        s: i32 = 0
        for i in range(32):
            s = s + A[i] * B[i]
        t: i32 = s * 3
        u: i32 = t * 5
        out[0] = t + u

    return mx


def _run(rtl):
    A = np.arange(32, dtype=np.int32)
    B = np.arange(32, dtype=np.int32) + 2
    out = np.zeros(1, dtype=np.int32)
    rtl.cosim(A, B, out)
    s = int((A.astype(np.int64) * B).sum())
    assert out[0] == np.int32(s * 18)


def test_exact_cycles_ships_no_slower_than_the_heuristic():
    # The exact scheduler minimizes span, then area under it; at the default
    # area_slack of zero it ships no slower than the heuristic.
    heuristic = _to_rtl(_mixed_kernel()).schedule().func("mx").latency
    rtl = _to_rtl(_mixed_kernel()).set_scheduler_opt(scheduler="exact")
    latency = rtl.schedule().func("mx").latency
    assert latency is not None and latency <= heuristic
    _run(rtl)


def test_area_slack_respects_an_explicit_pipeline_ii_ceiling():
    # A positive area_slack opens wider intervals for the area fold, but an
    # explicit pipeline(ii=n) caps that envelope at n, so the solved interval
    # never exceeds it.
    s = _mixed_kernel().schedule()
    s.pipeline("i", ii=2)
    rtl = s.export("rtl").set_scheduler_opt(scheduler="exact", area_slack=1.0)
    assert rtl.schedule().cyclic()[0].interval == 2
    _run(rtl)


def test_area_slack_needs_the_exact_scheduler():
    # area_slack trades span for area in the exact area solve; the heuristic
    # minimizes span only, so it ignores the knob and ships the same schedule.
    base = _to_rtl(_mixed_kernel()).schedule().func("mx").latency
    rtl = _to_rtl(_mixed_kernel()).set_scheduler_opt(area_slack=1.0)
    assert rtl.schedule().func("mx").latency == base
    _run(rtl)


def test_rejects_the_removed_area_objective():
    # O="area" is refused, not silently mapped; area is the cycles solve second
    # pass, steered by area_slack.
    with pytest.raises(ValueError, match="unknown objective"):
        _to_rtl(_mixed_kernel()).set_scheduler_opt(O="area")


def test_racing_workers_are_reported_as_not_reproducible():
    # deterministic=False lets the workers race: every cpsat solve, proven or
    # not, reports its schedule may not reproduce; interleaved within budget
    # they report it does.
    steady = _to_rtl(_mixed_kernel()).set_scheduler_opt(scheduler="exact")
    assert steady.schedule().compiler.deterministic
    rtl = _to_rtl(_mixed_kernel()).set_scheduler_opt(
        scheduler="exact", deterministic=False
    )
    report = rtl.schedule().compiler
    cpsat = [s for s in report.solves if s.solver == "cpsat"]
    assert cpsat and not any(s.deterministic for s in cpsat)
    assert not report.deterministic
    _run(rtl)


def test_freq_objective_sweeps_the_period_and_writes_the_clock_back():
    # O="freq" probes periods below the requested clock, holds the span within
    # span_tolerance, and the handle's clock follows the winner; compile then
    # tightens it once more to the realized critical path, held under every
    # bound row's warranted period.
    rtl = _to_rtl(_mixed_kernel(), freq_mhz=50.0).set_scheduler_opt(O="freq")
    result = rtl.schedule()
    assert result.sweep and result.sweep[0].cycle_ns == pytest.approx(20.0)
    n0 = result.sweep[0].latency
    assert rtl.freq_mhz > 50.0
    fn = result.func("mx")
    assert fn.latency <= n0 * 1.1
    est = rtl.estimation  # compiles, which tightens the clock to fmax
    floors = {o.symbol: o.timing.min_period_ns for o in default_device.operators}
    cap = max(floors[i] for i in _impls(result))
    assert rtl.freq_mhz == pytest.approx(min(est.fmax, 1000.0 / cap))
    assert est.clock_mhz == pytest.approx(rtl.freq_mhz)
    _run(rtl)


def test_freq_objective_sweeps_a_kernel_with_no_composed_span():
    # A data-dependent trip publishes no span, so the sweep leashes the
    # per-region quantities a span composes from instead of refusing: the
    # clock still becomes an output.
    @kernel
    def acc(A: i32[64], out: i32[1]):
        i: i32 = 0
        s: i32 = 0
        while A[i] < 100:
            s += A[i]
            i += 1
        out[0] = s

    rtl = _to_rtl(acc).set_scheduler_opt(O="freq")
    result = rtl.schedule()
    assert result.func("acc").latency is None
    assert result.sweep and result.sweep[0].latency is None
    assert rtl.freq_mhz > 1000.0 / result.sweep[0].achieved_ns - 0.5
    A = np.zeros(64, dtype=np.int32)
    A[:5] = 1
    A[5] = 100
    out = np.zeros(1, dtype=np.int32)
    rtl.cosim(A, out)
    assert out[0] == 5


def test_area_slack_pays_span_for_unit_folds():
    # Concurrent float adds need their own units at the natural II, and the
    # tight span the cycles solve minimizes to refuses the wider II that folds
    # them; a paid area_slack opens it, trading span for the fewer units.
    @kernel
    def pairsum(A: f32[8], B: f32[8], C: f32[8], D: f32[8], out: f32[8]):
        for i in range(8):
            out[i] = (A[i] + B[i]) + (C[i] + D[i])

    def units(rtl):
        rtl.compile()
        return sum(
            1
            for f in rtl.microarch.funcs
            for r in f.regions
            for u in r.units
            if (u.impl or "").startswith("add_f32")
        )

    strict = _to_rtl(pairsum).set_scheduler_opt(scheduler="exact", budget=2.0)
    n_strict = strict.schedule().func("pairsum").latency
    slack = _to_rtl(pairsum).set_scheduler_opt(
        scheduler="exact", area_slack=1.0, budget=2.0
    )
    n_slack = slack.schedule().func("pairsum").latency
    assert units(slack) < units(strict)
    assert n_strict <= n_slack <= n_strict * 2.0
    A = np.arange(8, dtype=np.float32)
    out = np.zeros(8, dtype=np.float32)
    slack.cosim(A, A, A, A, out)
    assert np.array_equal(out, 4.0 * A)


def test_wall_objective_trades_the_clock_for_iterations():
    # O="wall" minimizes span times period. A float accumulation is II-bound
    # by the adder's depth at the default clock; the latency-1 row at a slower
    # clock costs less wall time per iteration, so the sweep slows the clock
    # down and the shallow row wins.
    @kernel
    def acc(A: f32[64], out: f32[1]):
        s: f32 = 0.0
        for i in range(64):
            s = s + A[i]
        out[0] = s

    base = _to_rtl(acc).set_scheduler_opt(accumulators=0)
    wall0 = base.schedule().func("acc").latency * (1000.0 / base.freq_mhz)

    rtl = _to_rtl(acc).set_scheduler_opt(O="wall", accumulators=0)
    result = rtl.schedule()
    assert result.sweep and len(result.sweep) > 2
    assert rtl.freq_mhz < base.freq_mhz
    assert "add_f32_f32_f32_l1" in _impls(result)
    assert result.func("acc").latency * (1000.0 / rtl.freq_mhz) < wall0
    A = np.ones(64, dtype=np.float32)
    out = np.zeros(1, dtype=np.float32)
    rtl.cosim(A, out)
    assert out[0] == 64.0


def test_wall_objective_refuses_a_kernel_with_no_composed_span():
    # Wall time is span times period; with no span there is nothing to
    # minimize, unlike O="freq", which leashes the per-region vector instead.
    @kernel
    def find(A: i32[64], out: i32[1]):
        i: i32 = 0
        while A[i] < 100:
            i += 1
        out[0] = i

    rtl = _to_rtl(find).set_scheduler_opt(O="wall")
    with pytest.raises(RuntimeError, match="publishes no span"):
        rtl.schedule()


def test_tighten_clock_moves_the_operating_clock_to_the_realized_path():
    # Any compiled design may be reclocked at its realized critical path
    # without recompiling; the report's clock follows. A bound row's warranted
    # period caps the move, since its internal stages are not paths the
    # estimator sees, so the target is the slower of the two.
    rtl = _to_rtl(_mixed_kernel(), freq_mhz=200.0)
    bound = {op.impl for m in rtl.interfaces.values() for op in m.operators}
    cap = max(
        (o.timing.min_period_ns for o in default_device.operators if o.symbol in bound),
        default=0.0,
    )
    want = 1000.0 / max(1000.0 / rtl.estimation.fmax, cap)
    mhz = rtl.tighten_clock()
    assert mhz == pytest.approx(want) and rtl.freq_mhz == pytest.approx(want)
    assert mhz > 200.0  # the clock did move
    assert rtl.estimation.clock_mhz == pytest.approx(mhz)
    _run(rtl)


def test_clock_margin_splits_model_from_operating_period():
    # A margin cuts every chain to (1 - u) * cycle_ns while the design stays
    # clocked at cycle_ns; the QoR reports both periods.
    rtl = _to_rtl(_mixed_kernel(), freq_mhz=200.0).set_scheduler_opt(clock_margin=0.25)
    assert rtl.schedule().cycle_ns == pytest.approx(3.75)
    est = rtl.estimation
    assert est.fmax_target == pytest.approx(1000.0 / 3.75)
    assert est.clock_mhz == pytest.approx(200.0)
    assert "clocked at 200.0 MHz" in est.timing_report()
    _run(rtl)
