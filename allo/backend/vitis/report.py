# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

_RESOURCE_KEYS = (
    ("LUT", "lut"),
    ("FF", "ff"),
    ("DSP", "dsp"),
    ("BRAM_18K", "bram"),
    ("URAM", "uram"),
)


def _text(el: ET.Element | None, tag: str) -> str:
    if el is None:
        return ""
    return (el.findtext(tag) or "").strip()


def _int(el: ET.Element | None, tag: str, default: int = 0) -> int:
    text = _text(el, tag)
    return int(text) if text else default


def _opt_int(el: ET.Element | None, tag: str) -> int | None:
    # Latency/interval fields are "undef" for data-dependent (unbounded) designs.
    text = _text(el, tag)
    return int(text) if text.lstrip("-").isdigit() else None


def _float(el: ET.Element | None, tag: str, default: float = 0.0) -> float:
    text = _text(el, tag)
    return float(text) if text else default


@dataclass(frozen=True)
class ResourceUsage:
    """Resource counts (BRAM is in 18K-block units, matching the report)."""

    lut: int
    ff: int
    dsp: int
    bram: int
    uram: int


@dataclass(frozen=True)
class TimingReport:
    target_clock_ns: float
    estimated_clock_ns: float
    clock_uncertainty_ns: float

    @property
    def fmax_mhz(self) -> float:
        """Achievable clock frequency from the estimated critical path."""
        return 1000.0 / self.estimated_clock_ns if self.estimated_clock_ns else 0.0


@dataclass(frozen=True)
class LatencyReport:
    best_cycles: int | None  # None when latency is data-dependent ("undef")
    avg_cycles: int | None
    worst_cycles: int | None
    worst_time: str  # equivalent wall-clock latency, e.g. "25.976 us"
    interval_min: int | None
    interval_max: int | None
    pipeline_type: str | None
    pipeline_ii: int | None
    pipeline_depth: int | None


@dataclass(frozen=True)
class Interface:
    """A grouped hardware interface (one bundle / object of the RTL design)."""

    name: str  # bundle / object, e.g. "gmem0", "control"
    protocol: str  # m_axi, s_axi, axis, ap_ctrl_chain, ...
    data_bits: int | None  # data-bus width when applicable
    num_ports: int  # number of underlying RTL signals


@dataclass(frozen=True)
class LoopReport:
    """One loop nest inside a module, as Vitis reports it.

    A module's own ``latency`` covers a whole call; this covers the loop that
    call spends its time in. ``trip_count`` and ``latency_cycles`` are ``None``
    for a data-dependent bound, where the loop's ``ii``/``depth`` are still
    reported -- that pair is what survives to describe an unbounded mover."""

    name: str
    trip_count: int | None  # None when the bound is data-dependent ("undef")
    latency_cycles: int | None
    ii: int | None  # initiation interval, when the loop is pipelined
    depth: int | None  # iteration latency of one pipelined iteration
    pipelined: bool


@dataclass(frozen=True)
class ModuleReport:
    name: str
    timing: TimingReport
    latency: LatencyReport
    resources: ResourceUsage
    loops: list[LoopReport] = field(default_factory=list)


@dataclass(repr=False)
class SynthReport:
    version: str  # Vitis tool version, e.g. "2023.2"
    part: str  # target FPGA part number
    product_family: str
    top: str  # top-level module name
    flow_target: str  # "vitis" or "vivado"
    timing: TimingReport
    latency: LatencyReport
    resources: ResourceUsage  # top-level (whole-design) usage
    available: ResourceUsage  # device capacity
    interfaces: list[Interface] = field(default_factory=list)
    modules: dict[str, ModuleReport] = field(default_factory=dict)

    @property
    def fmax(self) -> float:
        """Achievable clock frequency (MHz)."""
        return self.timing.fmax_mhz

    @property
    def utilization(self) -> dict[str, float]:
        """Per-resource utilization as a percentage of device capacity."""
        util = {}
        for _, attr in _RESOURCE_KEYS:
            used, avail = getattr(self.resources, attr), getattr(self.available, attr)
            util[attr] = 100.0 * used / avail if avail else 0.0
        return util

    def __repr__(self) -> str:
        return (
            f"SynthReport(top={self.top!r}, part={self.part!r}, "
            f"fmax={self.fmax:.1f}MHz, lut={self.utilization['lut']:.1f}%, "
            f"modules={len(self.modules)})"
        )

    def __str__(self) -> str:
        t, lat = self.timing, self.latency
        ii = lat.interval_max if lat.interval_max is not None else lat.pipeline_ii
        lines = [
            "Vitis HLS Synthesis Report",
            f"  Tool version : {self.version}",
            f"  Part         : {self.part}  ({self.product_family})",
            f"  Top module   : {self.top}  [flow={self.flow_target}]",
            f"  Clock        : target {t.target_clock_ns:.3f} ns | "
            f"estimated {t.estimated_clock_ns:.3f} ns | Fmax {self.fmax:.1f} MHz",
            f"  Latency      : {lat.worst_cycles if lat.worst_cycles is not None else 'undef'}"
            f" cycles ({lat.worst_time or 'undef'})"
            + (f" | II {ii}" if ii is not None else ""),
            "",
            f"  {'Resource':<9}{'Used':>10}{'Avail':>10}{'Util%':>8}",
        ]
        util = self.utilization
        for label, attr in _RESOURCE_KEYS:
            lines.append(
                f"  {label:<9}{getattr(self.resources, attr):>10}"
                f"{getattr(self.available, attr):>10}{util[attr]:>8.1f}"
            )
        if self.interfaces:
            lines += ["", "  HW Interfaces"]
            for itf in self.interfaces:
                bits = f"{itf.data_bits}-bit" if itf.data_bits else "-"
                lines.append(f"    {itf.name:<12}{itf.protocol:<16}{bits}")
        subs = [m for m in self.modules.values() if m.name != self.top]
        lines += ["", f"  Sub-modules  : {len(subs)} (access via .modules[name])"]
        if subs:
            lines.append(
                f"    {'Module':<36}{'LUT':>8}{'FF':>8}{'DSP':>6}{'BRAM':>6}{'II':>6}"
            )
            for m in sorted(subs, key=lambda m: m.resources.lut, reverse=True)[:8]:
                name = m.name if len(m.name) <= 35 else m.name[:32] + "..."
                mii = m.latency.pipeline_ii
                lines.append(
                    f"    {name:<36}{m.resources.lut:>8}{m.resources.ff:>8}"
                    f"{m.resources.dsp:>6}{m.resources.bram:>6}"
                    f"{('-' if mii is None else mii)!s:>6}"
                )
        return "\n".join(lines)


def _parse_resources(res: ET.Element | None) -> ResourceUsage:
    return ResourceUsage(
        lut=_int(res, "LUT"),
        ff=_int(res, "FF"),
        dsp=_int(res, "DSP"),
        bram=_int(res, "BRAM_18K"),
        uram=_int(res, "URAM"),
    )


def _parse_timing(
    perf: ET.Element | None,
    target_fallback: float = 0.0,
    uncertainty_fallback: float = 0.0,
) -> TimingReport:
    # Top-level keeps target clock / uncertainty under <UserAssignments>; each
    # module repeats them inside its own <SummaryOfTimingAnalysis>.
    sta = perf.find("SummaryOfTimingAnalysis") if perf is not None else None
    return TimingReport(
        target_clock_ns=_float(sta, "TargetClockPeriod", target_fallback),
        estimated_clock_ns=_float(sta, "EstimatedClockPeriod"),
        clock_uncertainty_ns=_float(sta, "ClockUncertainty", uncertainty_fallback),
    )


def _parse_latency(perf: ET.Element | None) -> LatencyReport:
    lat = perf.find("SummaryOfOverallLatency") if perf is not None else None
    # Top-level carries <PipelineType> on <PerformanceEstimates>; modules carry
    # it inside <SummaryOfOverallLatency>.
    pipeline_type = _text(lat, "PipelineType") or _text(perf, "PipelineType")
    return LatencyReport(
        best_cycles=_opt_int(lat, "Best-caseLatency"),
        avg_cycles=_opt_int(lat, "Average-caseLatency"),
        worst_cycles=_opt_int(lat, "Worst-caseLatency"),
        worst_time=_text(lat, "Worst-caseRealTimeLatency"),
        interval_min=_opt_int(lat, "Interval-min"),
        interval_max=_opt_int(lat, "Interval-max"),
        pipeline_type=pipeline_type or None,
        pipeline_ii=_opt_int(lat, "PipelineInitiationInterval"),
        pipeline_depth=_opt_int(lat, "PipelineDepth"),
    )


def _parse_interfaces(root: ET.Element) -> list[Interface]:
    summary = root.find("InterfaceSummary")
    if summary is None:
        return []
    # Group the per-signal RtlPorts into one entry per (bundle, protocol).
    groups: dict[tuple[str, str], dict] = {}
    for port in summary.findall("RtlPorts"):
        key = (_text(port, "Object"), _text(port, "IOProtocol"))
        group = groups.setdefault(key, {"data_bits": None, "count": 0})
        group["count"] += 1
        bits = _opt_int(port, "Bits")
        if bits is not None and "DATA" in _text(port, "name").upper():
            group["data_bits"] = max(group["data_bits"] or 0, bits)
    return [
        Interface(
            name=name, protocol=proto, data_bits=g["data_bits"], num_ports=g["count"]
        )
        for (name, proto), g in groups.items()
    ]


def _parse_loops(perf: ET.Element | None) -> list[LoopReport]:
    """Every loop under a module's ``SummaryOfLoopLatency``, outermost first."""
    summary = perf.find("SummaryOfLoopLatency") if perf is not None else None
    if summary is None:
        return []
    return [
        LoopReport(
            name=_text(loop, "Name") or loop.tag,
            trip_count=_opt_int(loop, "TripCount"),
            latency_cycles=_opt_int(loop, "Latency"),
            ii=_opt_int(loop, "PipelineII"),
            depth=_opt_int(loop, "PipelineDepth"),
            pipelined=_text(loop, "PipelineType").lower() == "yes",
        )
        for loop in summary
    ]


def _parse_modules(root: ET.Element) -> dict[str, ModuleReport]:
    info = root.find("ModuleInformation")
    if info is None:
        return {}
    modules: dict[str, ModuleReport] = {}
    for module in info.findall("Module"):
        name = _text(module, "Name")
        perf = module.find("PerformanceEstimates")
        modules[name] = ModuleReport(
            name=name,
            timing=_parse_timing(perf),
            latency=_parse_latency(perf),
            resources=_parse_resources(module.find("AreaEstimates/Resources")),
            loops=_parse_loops(perf),
        )
    return modules


def parse_report(path: Path | str) -> SynthReport:
    """Parse a Vitis HLS ``csynth.xml`` report into a :class:`SynthReport`.

    ``path`` may be the ``csynth.xml`` file (as returned by ``mod.synth_report``)
    or the directory containing it.
    """
    path = Path(path)
    if path.is_dir():
        path = path / "csynth.xml"
    if not path.exists():
        raise FileNotFoundError(f"Synthesis report not found at: {path}")

    root = ET.parse(path).getroot()
    assignments = root.find("UserAssignments")
    perf = root.find("PerformanceEstimates")
    area = root.find("AreaEstimates")
    return SynthReport(
        version=_text(root.find("ReportVersion"), "Version"),
        part=_text(assignments, "Part"),
        product_family=_text(assignments, "ProductFamily"),
        top=_text(assignments, "TopModelName"),
        flow_target=_text(assignments, "FlowTarget"),
        timing=_parse_timing(
            perf,
            _float(assignments, "TargetClockPeriod"),
            _float(assignments, "ClockUncertainty"),
        ),
        latency=_parse_latency(perf),
        resources=_parse_resources(
            area.find("Resources") if area is not None else None
        ),
        available=_parse_resources(
            area.find("AvailableResources") if area is not None else None
        ),
        interfaces=_parse_interfaces(root),
        modules=_parse_modules(root),
    )
