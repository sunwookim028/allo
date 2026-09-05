# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the DUT and run it through ``cocotb_tools.runner``."""

from __future__ import annotations

import json
import os
import shutil
import tempfile

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import ip_models
from . import ports as _ports
from ... import marshal
from ..interface import Interfaces, ModuleInterface

_TB_MODULE = "allo.backend.rtl.sim.cocotb_tb"


def available(simulator: str = "verilator") -> bool:
    # Verilog is emitted in-process, so only the simulator is external.
    return shutil.which(simulator) is not None


@dataclass
class CosimResult:
    cycles: int
    latency_ns: float
    waveform: Path | None = None
    # The scalar return value sampled at `done`: bare for one result, a tuple
    # for several, None for none. An array result is written back in place.
    result: object = None


def _write_sources(
    verilog: str, module: str, workdir: Path, operators, interfaces: Interfaces
) -> tuple[list[Path], list[str]]:
    """Write the DUT Verilog (plus extern-IP behavioral models) and DPI C, and
    return ``(verilog_sources, build_args)`` for the runner. The models come from
    the device ``operators`` joined to the extern instances the manifest
    declares."""
    dut = workdir / f"{module}.sv"
    dut.write_text(verilog + "\n" + ip_models.sv_models(interfaces, operators))
    build_args: list[str] = []
    dpi = ip_models.dpi_c(interfaces, operators)
    if dpi:
        cpp = workdir / "dpi.cpp"
        cpp.write_text(dpi)
        build_args.append(str(cpp))
    return [dut], build_args


def _build_config(
    interface: ModuleInterface,
    mems,
    regfiles,
    streams,
    arg_types,
    args,
    *,
    clock_ps,
    timeout,
    workdir,
    stall_prob=0.0,
) -> dict:
    """Serialize each backing array to ``.npy`` and build the testbench config.

    The testbench reads this as plain JSON from the simulator's embedded Python,
    so every port object is projected back into a dict here; the model stops at
    this boundary."""
    mem_cfgs = []
    for m in mems:
        tag = f"{m.arg}_b{m.bank}"  # one backing array per (argument, bank)
        file_in = None
        if m.readers or not m.writeback:  # read/RMW args are preloaded from the arg
            bits = m.slice_in(args[m.arg])
            file_in = workdir / f"in_arg{tag}.npy"
            np.save(file_in, bits.astype(np.uint64))
        file_out = str(workdir / f"out_arg{tag}.npy") if m.writeback else None
        mem_cfgs.append(
            {
                "file_in": str(file_in) if file_in else None,
                "file_out": file_out,
                "size": m.size,
                "readers": [
                    {"addr": r.addr, "data": r.data, "latency": r.latency}
                    for r in m.readers
                ],
                "writers": [
                    {"addr": w.addr, "data": w.data, "we": w.we, "latency": w.latency}
                    for w in m.writers
                ],
            }
        )
    # A completely-partitioned argument: its whole bit pattern in flat element
    # order, one value per port, always preloaded so an element the kernel never
    # stores to passes through.
    reg_cfgs = []
    for rf in regfiles:
        arg = rf.port.arg
        bits = marshal.to_bits(args[arg], rf.host)
        file_in = workdir / f"in_reg{arg}.npy"
        np.save(file_in, bits.astype(np.uint64))
        # An unused direction stays absent rather than null.
        elements = []
        for e in rf.port.elements:
            cfg = {}
            if e.in_ is not None:
                cfg["in"] = e.in_
            if e.out is not None:
                cfg["out"], cfg["we"] = e.out, e.we
            elements.append(cfg)
        reg_cfgs.append(
            {
                "file_in": str(file_in),
                "file_out": (
                    str(workdir / f"out_reg{arg}.npy") if rf.port.writeback else None
                ),
                "elements": elements,
            }
        )
    scalars = [
        {
            "name": sc.name,
            "value": _ports.scalar_bits(args[sc.arg], arg_types[sc.arg]),
        }
        for sc in interface.scalars
    ]
    # An input stream's tokens are serialized to `.npy` for the feeder; an
    # output records where to write the drained ones and how many to expect.
    stream_cfgs = []
    for s in streams:
        p = s.port
        cfg = {
            "base": p.base,
            "data": p.data,
            "valid": p.valid,
            "ready": p.ready,
            "input": p.is_input,
        }
        if p.is_input:
            bits = marshal.to_bits(np.asarray(args[p.arg]), s.host)
            file_in = workdir / f"in_stream{p.arg}.npy"
            np.save(file_in, bits.astype(np.uint64))
            cfg["file_in"] = str(file_in)
        else:
            cfg["count"] = int(np.asarray(args[p.arg]).reshape(-1).shape[0])
            cfg["file_out"] = str(workdir / f"out_stream{p.arg}.npy")
        stream_cfgs.append(cfg)
    ctl = interface.control
    return {
        "top": interface.module,
        "control": {
            "clk": ctl.clk,
            "rst": ctl.rst,
            "start": ctl.start,
            "done": ctl.done,
        },
        "clock_ps": clock_ps,
        "timeout": timeout,
        "reset_cycles": 3,
        "settle_cycles": 2,
        "mems": mem_cfgs,
        "regfiles": reg_cfgs,
        "scalars": scalars,
        "streams": stream_cfgs,
        "stream_gap": stall_prob,
        "result_ports": [r.name for r in interface.results],
        "results_out": str(workdir / "results.json"),
        "cycles_out": str(workdir / "cycles.txt"),
    }


# pylint: disable-next=too-many-arguments
def cosim(
    verilog: str,
    interfaces: Interfaces,
    top: str,
    arg_types,
    args,
    *,
    result_types=(),
    operators=(),
    simulator: str = "verilator",
    freq_mhz: float = 300.0,
    timeout: int = 40000,
    workdir: str | os.PathLike | None = None,
    waves: bool = False,
    stall_prob: float = 0.0,
) -> CosimResult:
    """Drive the emitted RTL under cocotb + ``simulator`` with the numpy ``args``,
    bound to ports by the manifest of the module ``top`` (an MLIR symbol).
    ``interfaces`` is the whole map, since the extern-IP models cover every
    emitted module. Writes each output argument back in place and returns the
    cycle count."""
    from cocotb_tools.runner import get_runner

    assert len(args) == len(
        arg_types
    ), f"cosim expected {len(arg_types)} kernel arguments, got {len(args)}"
    interface = interfaces.of_symbol(top)
    module = interface.module
    mems = _ports.plan_mems(interface, arg_types)
    regfiles = _ports.plan_regfiles(interface, arg_types)
    streams = _ports.plan_streams(interface, arg_types)

    tmp = workdir is None
    wd = Path(tempfile.mkdtemp(prefix="allo_cosim_")) if tmp else Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    # the whole simulator build/run is one unit; `finally` owns the cleanup
    # pylint: disable-next=too-many-try-statements
    try:
        verilog_sources, build_args = _write_sources(
            verilog, module, wd, operators, interfaces
        )
        if simulator == "verilator":
            # Each DUT builds in a fresh directory and recompiles the same
            # verilator runtime units, which OBJCACHE turns into cache hits. An
            # explicit setting wins, including an empty one that opts out.
            if "OBJCACHE" not in os.environ:
                cache = next(
                    (c for c in ("ccache", "sccache") if shutil.which(c)), None
                )
                if cache:
                    os.environ["OBJCACHE"] = cache
            # A `.gch` records the path it was built from, so a cached one
            # names a deleted temp dir and the compile fails. Drop `pch_defines`
            # to decline it and keep `time_macros`, which caches the rest.
            os.environ.setdefault("CCACHE_SLOPPINESS", "time_macros")
        # An even integer ps, since cocotb splits it into two half periods. It
        # only affects sim time, not the reported cycle count.
        clock_ps = round(1.0e6 / freq_mhz)
        clock_ps += clock_ps & 1
        cfg = _build_config(
            interface,
            mems,
            regfiles,
            streams,
            arg_types,
            args,
            clock_ps=clock_ps,
            timeout=timeout,
            workdir=wd,
            stall_prob=stall_prob,
        )
        cfg_path = wd / "cosim.json"
        cfg_path.write_text(json.dumps(cfg))

        runner = get_runner(simulator)
        runner.build(
            sources=verilog_sources,
            build_args=build_args,
            hdl_toplevel=module,
            build_dir=str(wd / "sim_build"),
            always=True,
            waves=waves,
        )
        from cocotb_tools.runner import get_results

        xml = runner.test(
            hdl_toplevel=module,
            test_module=_TB_MODULE,
            test_dir=str(wd),
            extra_env={"ALLO_COSIM_CFG": str(cfg_path)},
            waves=waves,
        )
        _, failed = get_results(xml)
        assert failed == 0, f"cosim testbench failed (see {wd}/sim.log)"

        cycles = int((wd / "cycles.txt").read_text().strip())
        for m in mems:
            if m.writeback:
                bits = np.load(wd / f"out_arg{m.arg}_b{m.bank}.npy")
                vals = marshal.from_bits(bits, m.host, (m.size,))
                m.scatter_out(args[m.arg], vals)
        # A written scattered argument: its registers are flat and unbanked, so
        # the drained values land straight back in the caller's array.
        for rf in regfiles:
            if rf.port.writeback:
                buf = args[rf.port.arg]
                bits = np.load(wd / f"out_reg{rf.port.arg}.npy")
                buf[...] = marshal.from_bits(bits, rf.host, buf.shape)
        # Drained output-stream tokens, written into the caller's buffer in place.
        for s in streams:
            if not s.port.is_input:
                buf = np.asarray(args[s.port.arg])
                bits = np.load(wd / f"out_stream{s.port.arg}.npy")
                buf[...] = marshal.from_bits(bits, s.host, buf.shape)
        # Decode each sampled result port by its return type; the manifest order
        # matches `result_types`.
        raw = json.loads((wd / "results.json").read_text())
        assert len(raw) == len(
            result_types
        ), f"cosim sampled {len(raw)} result ports, expected {len(result_types)}"
        vals = [_ports.from_scalar_bits(b, t) for b, t in zip(raw, result_types)]
        result = vals[0] if len(vals) == 1 else (tuple(vals) if vals else None)
        wave = next(iter(wd.glob("sim_build/*.fst")), None) if waves else None
        return CosimResult(cycles, cycles * 1000.0 / freq_mhz, wave, result)
    finally:
        if tmp:
            shutil.rmtree(wd, ignore_errors=True)
