# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typing stubs for the Vitis HLS backend."""

from pathlib import Path
from typing import Any, Generic, ParamSpec, TypeVar, Literal

P = ParamSpec("P")
R = TypeVar("R")

AxiOffset = Literal["off", "direct", "slave"]
AxisRegisterMode = Literal["forward", "reverse", "both", "off"]
AxiliteStorageImpl = Literal["auto", "bram", "uram"]
VitisMode = Literal["csim", "csyn", "sw_emu", "hw_emu", "hw"]

HLS_PREPARE_PIPELINE: str = ...
DEFAULT_DEVICE: str = ...
DEFAULT_FREQ_MHZ: str = ...
DEFAULT_PART: str = ...

class Vitis(Generic[P, R]):
    part: str
    device: str
    freq_mhz: float
    flow: Literal["vitis", "vivado"]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Wrap a compiled kernel for the Vitis HLS backend.

        Normally constructed via ``Schedule.export("vitis", ...)``. Keyword
        options: ``vitis_home`` (Vitis install dir; falls back to
        ``$XILINX_HLS``/``$XILINX_VITIS``), ``project_path`` (where projects are
        scaffolded), ``device`` (board name, e.g. ``"u280"``) or ``part`` (FPGA
        part number; mutually exclusive with ``device``), ``freq_mhz`` (target
        clock, default 300), and ``flow`` (``"vitis"`` or ``"vivado"``).
        """

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Run C simulation with the given runtime arguments and return its
        result (shorthand for ``run("csim", ...)``)."""

    @property
    def hls_code(self) -> str:
        """The emitted Vitis HLS C++ source for the top kernel."""

    @property
    def synth_report(self) -> Path:
        """Path to the Vitis HLS C-synthesis report (``csynth.xml``) for the
        configured project. Pass it to ``allo.backend.vitis.parse_report`` to
        inspect synthesis results without re-synthesizing. Requires a project
        path (set via ``project_path=`` or a prior ``scaffold_project``)."""

    def run(self, mode: VitisMode, *args: Any, exist_ok: bool = ...) -> Any:
        """Build and/or run the kernel in the requested ``mode``:

        - ``"csim"``: compile and run C simulation with ``*args`` (returns its result).
        - ``"sw_emu"``: deprecated alias for ``"csim"``.
        - ``"csyn"``: run Vitis HLS synthesis (no runtime args); returns the report path.
        - ``"hw_emu"``: build and run hardware emulation with ``*args``.
        - ``"hw"``: build the hardware bitstream (no host execution).

        ``exist_ok=False`` rebuilds even when cached artifacts exist.
        """

    def synth(self, *, exist_ok: bool = ...):
        """Scaffold an HLS project, invoke Vitis HLS C synthesis. Requires a part number."""

    def precheck(
        self, mode: VitisMode, project: str | None = ..., *, exist_ok: bool = ...
    ) -> Path:
        """Scaffold and run the fast ``hw_emu``/``hw`` pre-check (kernel ``.xo`` +
        XRT host, plus emconfig for emulation) without the multi-hour, platform-
        locked link step, validating that the project is buildable. Returns the
        project directory."""

    def scaffold_project(
        self, project: str | None = ..., *args, exist_ok: bool = ...
    ) -> Path:
        """Write the HLS project files (kernel sources, Makefile, config) to
        ``project`` (or the configured path) without invoking Vitis HLS, and
        return the project directory. ``args`` are the runtime arguments used
        to generate input examples used in hw_emu."""

    def set_axi(
        self,
        index: int,
        *,
        bundle: str | None = None,
        depth: int | None = None,
        offset: AxiOffset | None = None,
        channel: str | None = None,
        latency: int | None = None,
        num_read_outstanding: int | None = None,
        num_write_outstanding: int | None = None,
        max_read_burst_length: int | None = None,
        max_write_burst_length: int | None = None,
        max_widen_bitwidth: int | None = None,
        alignment_byte_size: int | None = None,
        name: str | None = None,
        **kwargs: str | int | bool | None,
    ) -> None:
        """Bind argument ``index`` to an AXI master (``m_axi``) interface with the
        given options (``bundle``, ``depth``, ``offset``, burst lengths, ...; see
        the Vitis HLS interface pragma). Only valid on buffer (pointer)
        arguments."""

    def set_axis(
        self,
        index: int,
        *,
        register: bool | None = None,
        register_mode: AxisRegisterMode | None = None,
        depth: int | None = None,
        name: str | None = None,
        bundle: str | None = None,
        **kwargs: str | int | bool | None,
    ) -> None:
        """Bind stream argument ``index`` to an AXI4-Stream (``axis``) interface
        with the given options (``register``, ``register_mode``, ``depth``, ...;
        see the Vitis HLS interface pragma). Only valid on ``Stream``
        arguments."""

    def set_axilite(
        self,
        index: int,
        *,
        bundle: str | None = None,
        register: bool | None = None,
        clock: str | None = None,
        name: str | None = None,
        offset: str | None = None,
        storage_impl: AxiliteStorageImpl | None = None,
        **kwargs: str | int | bool | None,
    ) -> None:
        """Bind argument ``index`` (or the return value with ``index=-1``) to an
        AXI4-Lite (``s_axilite``) slave interface, typically for control/status,
        with the given options (``bundle``, ``register``, ``offset``,
        ``storage_impl``, ...; see the Vitis HLS interface pragma)."""
