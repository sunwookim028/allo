# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The RTL backend"""

from __future__ import annotations

import json
import warnings

from dataclasses import fields, replace
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

from ..base import Backend, run_pipeline
from ..cpu import CPU
from ..._mlir.ir import Module
from ..._mlir._mlir_libs._allo import ir_ext
from ..._mlir.dialects.allo import (
    emit_verilog,
    emit_split_verilog,
    emit_datapath_to_hw,
)
from .device import (
    Device,
    inject_operators,
    inject_device,
    operator_descs,
)
from .devices import default_device
from .interface import Interfaces
from .options import PERIOD_POLICIES, PrepassOptions, SchedulerOptions
from .qor import QoR, estimate
from .reports import CompileReport, MicroarchReport, ScheduleResult
from .schedule import run_schedule, sweep_freq, sweep_wall
from .sim import shell
from ...lang.core import ShapedType
from ...lang.kernel import Kernel

P = ParamSpec("P")
R = TypeVar("R")

# The one DCP normalization before emit: it materializes the per-bank memrefs of
# a partitioned array. Addresses stay in element space until the emitter.
_NORMALIZE_PIPELINE = "builtin.module(dcp-resolve-banking)"

# The one option the handle derives rather than takes: the scheduler's view of
# `freq_mhz`, which the emitter and the cosim clock read too.
_DERIVED_OPTIONS = {"cycle_ns"}

# The values the ``O`` knob accepts: minimize span (area traded within
# `area_slack`), or one of the two period policies that make the clock an output.
_OBJECTIVES = {"cycles"} | PERIOD_POLICIES


def _operator_period_cap(device: Device) -> float:
    """The slowest period any operator row is built for, which caps the wall
    ladder. Zero where the device declares no operator."""
    return max(
        (
            max(
                device.reg_delay_ns + o.timing.in_delay_ns,
                o.timing.out_delay_ns,
                o.timing.min_period_ns,
            )
            for o in device.operators
        ),
        default=0.0,
    )


class LatencyModelWarning(UserWarning):
    """A cosim ran for fewer cycles than the exact contract the kernel
    publishes."""


class RealizationWarning(UserWarning):
    """A scaffolded project instantiates an extern module its device cannot
    build, so it synthesizes as a black box until an implementation is
    supplied."""


# pylint: disable-next=too-many-instance-attributes
class RTL(Backend[P, R]):
    name = "rtl"

    def __init__(
        self,
        kernel: Kernel[P, R],
        *,
        device: Device | None = None,
        freq_mhz: float | None = None,
        simulator: str = "verilator",
    ):
        """Build an RTL handle for one hardware configuration.

        These three fix the target and how hardware is built for it. The
        compiler's own knobs are set on the handle with
        :meth:`set_scheduler_opt`, by the field names of
        :class:`SchedulerOptions` and :class:`PrepassOptions`.

        Args:
            device: the hardware platform: storage primitives, native chaining
                delays, operator IPs and a default clock.
            freq_mhz: target frequency, overriding the device default. Drives
                both the SDC cycle time and the cosim clock.
            simulator: the engine cocotb drives for ``cosim``.
        """
        super().__init__(kernel)
        self._device = device if device is not None else default_device
        self.freq_mhz = (
            freq_mhz if freq_mhz is not None else self._device.default_freq_mhz
        )
        self._cycle_time = 1000.0 / self.freq_mhz
        self.simulator = simulator
        # Operator-sharing binding, resolved from the scheduler when the schedule
        # is built; `use_trivial_binding` overrides it.
        self.binding: str | None = None
        self._sched_opts = SchedulerOptions(cycle_ns=self._cycle_time)
        self._prepass_opts = PrepassOptions()
        self.arg_types = kernel.parse_argument_annotations()
        self.res_types = kernel.parse_return_annotation()
        # The stage artifacts, each built once on first use. `self.module` stays
        # the pristine snapshot.
        self._dcp_ir: Module | None = None
        self._schedule_result: ScheduleResult | None = None
        self._hw_ir: Module | None = None
        self._verilog: str | None = None
        self._cpu: CPU[P, R] | None = None
        self._interfaces: Interfaces | None = None
        self._microarch: MicroarchReport | None = None
        self._manifest: dict | None = None

    @property
    def top(self) -> str:
        """The DUT module name"""
        return self.kernel.func_name

    @property
    def device(self) -> Device:
        """The device this handle compiles for"""
        return self._device

    # -- scheduling -------------------------------------------------------

    def set_scheduler_opt(self, **opts: Any) -> RTL:
        """Turn one or more of the compiler's knobs and return the handle, so
        calls chain. The names are the fields of :class:`SchedulerOptions`,
        which ``report.compiler.options`` publishes back, and of
        :class:`PrepassOptions`, which shapes the IR the scheduler is handed.

        ``cycle_ns`` is not among them: it follows ``freq_mhz``, and setting it
        here would leave the schedule describing a different design than the one
        cosim drives.
        """
        assert self._schedule_result is None, (
            "the schedule is already built, and everything downstream describes "
            "it, so a knob turned now would not reach the design"
        )
        sched = {f.name for f in fields(SchedulerOptions)} - _DERIVED_OPTIONS
        prepass = {f.name for f in fields(PrepassOptions)}
        unknown = sorted(set(opts) - sched - prepass)
        if unknown:
            raise ValueError(
                f"unknown scheduler option(s) {unknown}; "
                f"expected any of {sorted(sched | prepass)}"
            )
        if (o := opts.get("O")) is not None and o not in _OBJECTIVES:
            raise ValueError(
                f"unknown objective O={o!r}; expected any of {sorted(_OBJECTIVES)}"
            )
        self._sched_opts = replace(
            self._sched_opts, **{k: v for k, v in opts.items() if k in sched}
        )
        self._prepass_opts = replace(
            self._prepass_opts, **{k: v for k, v in opts.items() if k in prepass}
        )
        return self

    def use_trivial_binding(self) -> RTL:
        """Give every operation its own unit, folding nothing.

        A development diagnostic that strips binding-time sharing out of the
        datapath. The default binding follows the scheduler: ``exact-share``
        under the heuristic, ``planned`` under an exact scheduler.
        """
        assert self._schedule_result is None, (
            "the schedule is already built, and everything downstream describes "
            "it, so a binding forced now would not reach the design"
        )
        self.binding = "trivial"
        return self

    def schedule(self) -> ScheduleResult:
        """Schedule the kernel and return the result: per-func regions with their
        II, latency and per-op start times. Computed once and reused by
        ``compile()``, so it always describes the RTL that ``cosim`` runs."""
        if self._schedule_result is None:
            # The default binding follows the scheduler: an exact solve carries
            # its own allocation, so `planned` realizes it; the heuristic
            # decides none, so `exact-share` solves the fold instead.
            if self.binding is None:
                self.binding = (
                    "exact-share"
                    if self._sched_opts.scheduler == "heuristic"
                    else "planned"
                )

            # The schedule is reified in place, so it runs on a copy. Operator
            # and device timing is injected into that copy only, keeping the CPU
            # functional path clear of it.
            def make_module() -> Module:
                m = ir_ext.clone_module(self.module)
                inject_operators(m, self._device)
                inject_device(
                    m, self._device, weights=self._sched_opts.resource_weights
                )
                return m

            # An allocation is only worth deciding where the emitter builds it:
            # the trivial binding keeps one unit per operation.
            allocate = self.binding != "trivial"
            if self._sched_opts.O in PERIOD_POLICIES:
                # The clock is an output: the sweep probes candidates on
                # fresh copies and the handle follows the winner.
                if self._sched_opts.O == "freq":
                    self._dcp_ir, self._schedule_result = sweep_freq(
                        self.top,
                        make_module,
                        self._sched_opts,
                        self._prepass_opts,
                        allocate,
                        self._device.reg_delay_ns,
                    )
                else:
                    self._dcp_ir, self._schedule_result = sweep_wall(
                        self.top,
                        make_module,
                        self._sched_opts,
                        self._prepass_opts,
                        allocate,
                        self._device.reg_delay_ns,
                        _operator_period_cap(self._device),
                    )
                self._set_clock(
                    self._schedule_result.cycle_ns
                    / (1.0 - self._sched_opts.clock_margin)
                )
            else:
                self._dcp_ir = make_module()
                self._schedule_result = run_schedule(
                    self.top,
                    self._dcp_ir,
                    self._sched_opts,
                    self._prepass_opts,
                    allocate,
                )
        return self._schedule_result

    def _set_clock(self, period_ns: float) -> None:
        """Move the operating clock the design ships at. ``freq_mhz``, the
        published options and the QoR's ``clock_mhz`` follow, and cosim drives
        the new clock. Nothing is recompiled: the schedule's model period,
        which the chains were cut to, stays what it was."""
        self.freq_mhz = 1000.0 / period_ns
        self._cycle_time = period_ns
        result = self._schedule_result
        self._sched_opts = replace(result.compiler.options, cycle_ns=period_ns)
        self._schedule_result = replace(
            result, compiler=replace(result.compiler, options=self._sched_opts)
        )

    @property
    def dcp_module(self) -> Module:
        """The scheduled DCP module object."""
        self.schedule()
        assert self._dcp_ir is not None  # set by schedule()
        return self._dcp_ir

    @property
    def dcp(self) -> str:
        """The textual scheduled DCP MLIR module.
        NOTE: the textual form is not stable"""
        return str(self.dcp_module)

    # -- emission ---------------------------------------------------------

    def compile(self) -> Module:
        """Compile the kernel to hw/comb/seq MLIR"""
        if self._hw_ir is None:
            # An array return has no meaning at a hardware port. Emission only:
            # such a kernel still schedules.
            if any(isinstance(t, ShapedType) for t in self.res_types):
                raise TypeError(
                    "RTL does not support returning arrays; use an out-parameter "
                    "instead"
                )
            schedule = self.schedule()
            # The period the schedule holds: the target, or the one the
            # scheduler lowered the clock to when the target was unreachable.
            # Emission checks and reports against it.
            cycle_ns = schedule.cycle_ns or self._cycle_time
            # Emit on a copy, so `dcp` keeps reading the scheduled module.
            work = ir_ext.clone_module(self._dcp_ir)
            run_pipeline(work, _NORMALIZE_PIPELINE)
            # The emitter is a direct call rather than a pass, so its diagnostics
            # do not reach the PassManager -> MLIRError path. Capture them here.
            diagnostics: list[str] = []
            handler = work.context.attach_diagnostic_handler(
                lambda d: bool(diagnostics.append(d.message)) or True
            )
            try:
                manifests = emit_datapath_to_hw(work, self.binding, self.top, cycle_ns)
            finally:
                handler.detach()
            if manifests is None:
                raise RuntimeError(
                    "An error occurred during code generation process:\n"
                    + "\n".join(diagnostics)
                )
            # One envelope, two documents: the boundary the cosim harness drives
            # and the allocation the emitter decided.
            envelope = json.loads(manifests)
            self._interfaces = Interfaces.from_json(envelope["interfaces"])
            self._microarch = MicroarchReport.from_json(envelope["microarch"])
            # The boundary document verbatim, for `scaffold_project` to write.
            self._manifest = envelope["interfaces"]
            self._hw_ir = work
            # Last step of the period policies: clock at the realized
            # critical path.
            if self._sched_opts.O in PERIOD_POLICIES:
                self.tighten_clock()
        return self._hw_ir

    @property
    def mlir(self) -> str:
        """The emitted hw/comb/seq MLIR module"""
        return str(self.compile())

    @property
    def verilog(self) -> str:
        """The emitted (System)Verilog via CIRCT"""
        if self._verilog is None:
            verilog = emit_verilog(self.compile())
            assert verilog is not None, "RTL Verilog emission failed"
            self._verilog = verilog
        return self._verilog

    @property
    def interfaces(self) -> Interfaces:
        """The emitted modules' port interfaces, keyed by RTL module name"""
        self.compile()
        return self._interfaces

    @property
    def microarch(self) -> MicroarchReport:
        """What the emitter BUILT: per region its units and muxes, per array its
        storage and ports, and the design's register ledger. The allocation half
        of the compile, joined to ``schedule()`` on (func, region order)."""
        self.compile()
        assert self._microarch is not None  # set by compile()
        return self._microarch

    @property
    def report(self) -> CompileReport:
        """The whole compile: schedule, allocation and boundary in one object.
        Emission has run by the time this returns, so every member is present;
        for the schedule alone, call ``schedule()``."""
        self.compile()  # before schedule() is read: compiling can move the clock
        return CompileReport(self.schedule(), self.microarch, self.interfaces)

    @property
    def estimation(self) -> QoR:
        """What this compile costs: its span, and the area its structures price
        at against the device it was compiled for. A model, and never a
        substitute for synthesis; see :mod:`allo.backend.rtl.qor`."""
        return estimate(self.report, self._device)

    def tighten_clock(self) -> float:
        """Clock the compiled design at its realized critical path (the QoR's
        ``fmax``), recompiling nothing, with ``clock_margin`` withheld on top.
        The clock moves in both directions: a design whose paths came in under
        the target speeds up, one that missed it slows down. A bound operator
        row's warranted period caps the move, since its internal pipeline
        stages are not paths the estimator sees. Returns the new ``freq_mhz``,
        which ``cosim`` then drives. Runs by itself at compile under every
        period policy."""
        period = 1000.0 / self.estimation.fmax
        floors = {o.symbol: o.timing.min_period_ns for o in self._device.operators}
        bound = {op.impl for m in self.interfaces.values() for op in m.operators}
        period = max(period, *(floors[i] for i in bound)) if bound else period
        self._set_clock(period / (1.0 - self._sched_opts.clock_margin))
        return self.freq_mhz

    # -- verbs ------------------------------------------------------------

    def csim(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Functional golden: run the kernel on the CPU/LLVM-JIT path (in place)."""
        if self._cpu is None:
            self._cpu = CPU(self.kernel)
        return self._cpu(*args, **kwargs)

    def cosim(
        self,
        *args: Any,
        simulator: str | None = None,
        timeout: int = 40000,
        waves: bool = False,
        stall_prob: float = 0.0,
    ) -> shell.CosimResult:
        """Drive the emitted RTL under cocotb; write outputs back in place and
        return the cycle count. Does not compare: keep a ``csim`` golden.

        An array output crosses as an out-parameter, so pass a pre-allocated
        buffer for each; a scalar result stays an output port sampled at
        ``done``. A ``Stream[...]`` argument is driven token-by-token over its
        FIFO handshake: a 1-D array of tokens for an input, a pre-allocated
        buffer for an output. ``stall_prob`` (0..1) randomly starves inputs and
        back-pressures outputs; the result must be unchanged.
        """
        self.compile()  # fills self._interfaces
        result = shell.cosim(
            self.verilog,
            self.interfaces,
            self.top,
            self.arg_types,
            list(args),
            result_types=self.res_types,
            operators=operator_descs(self._device.operators),
            simulator=simulator or self.simulator,
            freq_mhz=self.freq_mhz,
            timeout=timeout,
            waves=waves,
            stall_prob=stall_prob,
        )
        if stall_prob == 0:
            self._check_latency(result.cycles)
        return result

    def _check_latency(self, cycles: int) -> None:
        """Hold the published latency contract to the measured cycle count: a
        run that outlasts it fails, a run that finishes early warns. The only
        check in the compiler that compares a model against a measurement
        rather than against another model.
        """
        iface = self.interfaces.of_symbol(self.top)
        fn = self.schedule().func(self.top)
        assert (iface.latency, iface.latency_is_bound, iface.determinacy) == (
            fn.latency,
            fn.latency_is_bound,
            fn.determinacy,
        ), "the manifest and the schedule report disagree about the kernel's span"
        # A data-dependent span publishes no figure, and a concurrent kernel's
        # is a completion floor, so neither is held to a count.
        if iface.latency is None or iface.determinacy == "concurrent":
            return
        if cycles > iface.latency:
            kind = "bounds" if iface.latency_is_bound else "is"
            raise RuntimeError(
                f"the latency contract is UNSOUND for '{self.top}': it publishes "
                f"{iface.latency} cycles, which {kind} the whole start->done "
                f"span, and the hardware ran {cycles} "
                f"({cycles - iface.latency:+d}); a caller time-triggered against "
                "the published figure samples before this kernel writes"
            )
        # Slack under a bound is expected; only an exact contract is held to
        # the cycle.
        if cycles < iface.latency and not iface.latency_is_bound:
            warnings.warn(
                f"DEV-ONLY: the latency model is pessimistic for '{self.top}': "
                f"declared latency = {iface.latency}, measured {cycles} cycles "
                f"(delta {cycles - iface.latency:+d}), which may indicate a bug "
                "in the compiler or the RTL.",
                LatencyModelWarning,
                stacklevel=2,
            )

    # pylint: disable-next=arguments-differ
    def run(self, mode: str, *args: Any, **kwargs: Any) -> Any:
        if mode == "csim":
            return self.csim(*args, **kwargs)
        if mode == "cosim":
            return self.cosim(*args, **kwargs)
        raise NotImplementedError(f"RTL mode '{mode}' is not implemented")

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Run the kernel with CPU functional simulation"""
        return self.csim(*args, **kwargs)

    def scaffold_project(
        self, project: str | None = None, *, exist_ok: bool = True
    ) -> Path:
        """Write the compiled design as a project directory: the emitted RTL
        one file per module, the port manifest, and whatever the device's
        realizer contributes (operator-core wrappers and build scripts).
        Extern modules the realizer cannot build raise a
        :class:`RealizationWarning` and stay black boxes."""
        module = self.compile()
        root = Path(project or f"{self.top}.prj")
        root.mkdir(parents=True, exist_ok=exist_ok)
        # Verilog export lowers the module in place, so split-emit a copy and
        # keep the compiled module pristine for `verilog` and `cosim`.
        work = ir_ext.clone_module(module)
        ok = emit_split_verilog(work, str(root))
        assert ok, "RTL Verilog emission failed"
        (root / "manifest.json").write_text(json.dumps(self._manifest, indent=2))
        if self._device.realizer is not None:
            realized = self._device.realizer(self.interfaces, self._device)
            for name, text in realized.files.items():
                (root / name).write_text(text)
            if realized.missing:
                warnings.warn(
                    "extern operator modules with no realization: "
                    + "; ".join(realized.missing),
                    RealizationWarning,
                )
        return root
