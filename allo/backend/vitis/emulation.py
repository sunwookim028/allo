# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vitis emulation / hardware (sw_emu, hw_emu, hw) support for the new backend.

The kernel is built into an ``.xclbin`` with ``v++`` and driven by an XRT-native
host (``xrt::device``/``xrt::kernel``/``xrt::bo``). Build orchestration is a
self-contained Makefile (``impl.mk``) whose ``PLATFORM`` is read from the
environment, so no ``.xpfm`` is baked into the project. Both the host and the
Makefile are emitted into the synth project directory at scaffold time, so a
single directory serves csyn, hw_emu, and hw."""

from __future__ import annotations

import os

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from ..marshal import HLS_HW_ABI, as_array, host_type
from .utils import _render_template
from ...lang.core import BufferType, DType, TypeBase
from ...logging import run_command, stage

IMPL_MAKEFILE = "Makefile"
HOST_CPP = "host.cpp"


def _validate_impl_dtype(dtype: TypeBase, index: int) -> None:
    """Reject element types :data:`HLS_HW_ABI` cannot carry at the m_axi /
    s_axilite boundary. Emulation and hardware use the real kernel types rather
    than the widened ``generate-apint-wrapper`` ABI C simulation uses."""
    where = "return value" if index == -1 else f"argument {index}"
    if not isinstance(dtype, DType):
        raise TypeError(
            f"Vitis emulation/hardware {where}: unsupported type {dtype!r}."
        )
    try:
        host_type(dtype, HLS_HW_ABI)
    except TypeError as exc:
        raise TypeError(f"{exc} ({where})") from exc


def _element_bytes(dtype: DType) -> int:
    """Host byte size of one element of ``dtype`` (its standard-width container)."""
    return host_type(dtype, HLS_HW_ABI).container_bits // 8


def _buffer_bytes(buffer_type: BufferType) -> int:
    """Total byte size of a buffer's flattened contents at the host boundary."""
    return _element_bytes(buffer_type.dtype) * int(np.prod(buffer_type.shape))


def validate_impl_abi(arg_types: list[TypeBase], res_types: list[TypeBase]) -> None:
    """Raise a clear error if any argument or return type cannot cross the
    XRT host boundary (see ``_validate_impl_dtype``)."""
    for index, arg_type in enumerate(arg_types):
        dtype = arg_type.dtype if isinstance(arg_type, BufferType) else arg_type
        _validate_impl_dtype(dtype, index)
    for res_type in res_types:
        _validate_impl_dtype(res_type, -1)


def generate_impl_host(top: str, arg_types: list[TypeBase]) -> str:
    """Render the XRT-native host. Buffer arguments are staged through
    ``input<i>.data`` BOs and synced back to ``output<i>.data`` after the run;
    scalars are read by value. Argument index ``i`` maps to ``group_id(i)``."""
    body: list[str] = []
    run_args: list[str] = []
    buffer_indices: list[int] = []
    for i, arg_type in enumerate(arg_types):
        if isinstance(arg_type, BufferType):
            nbytes = _buffer_bytes(arg_type)
            body += [
                f'  auto in{i} = read_data("input{i}.data", {nbytes});',
                f"  auto bo{i} = xrt::bo(device, {nbytes}, kernel.group_id({i}));",
                f"  bo{i}.write(in{i}.data());",
                f"  bo{i}.sync(XCL_BO_SYNC_BO_TO_DEVICE);",
            ]
            run_args.append(f"bo{i}")
            buffer_indices.append(i)
        elif isinstance(arg_type, DType):
            ctype = host_type(arg_type, HLS_HW_ABI).c_scalar
            nbytes = _element_bytes(arg_type)
            body += [
                f'  auto raw{i} = read_data("input{i}.data", {nbytes});',
                f"  {ctype} arg{i} = *reinterpret_cast<{ctype} *>(raw{i}.data());",
            ]
            run_args.append(f"arg{i}")
        else:
            raise TypeError(
                f"Vitis emulation/hardware argument {i}: unsupported type "
                f"{arg_type!r} (only buffers and scalars are supported)."
            )

    body.append("")
    body.append(f'  auto run = kernel({", ".join(run_args)});')
    body.append("  run.wait();")
    body.append("")
    for i in buffer_indices:
        body += [
            f"  bo{i}.sync(XCL_BO_SYNC_BO_FROM_DEVICE);",
            f"  bo{i}.read(in{i}.data());",
            f'  write_data("output{i}.data", in{i}.data(), in{i}.size());',
        ]
    return _render_template("host.cpp", top=top, body="\n".join(body))


def generate_impl_makefile(top: str, freq_mhz: float, vitis_root: Path) -> str:
    """Render ``impl.mk`` (the emulation/hardware Makefile) for ``top``."""
    # v++ --kernel_frequency expects an integer MHz.
    freq = int(freq_mhz) if float(freq_mhz).is_integer() else freq_mhz
    n_jobs = int(os.getenv("VIVADO_IMPL_JOBS", "4"))
    return _render_template(
        "impl.mk",
        top=top,
        freq_mhz=freq,
        vitis_root=os.fspath(vitis_root),
        vivado_impl_jobs=n_jobs,
    )


def write_impl_inputs(
    project_path: Path, arg_types: list[TypeBase], args: tuple
) -> None:
    """Serialize each kernel argument to ``input<i>.data`` for the host to read."""
    for i, (arg_type, arg) in enumerate(zip(arg_types, args)):
        if isinstance(arg_type, BufferType):
            data = as_array(arg, arg_type, HLS_HW_ABI).tobytes()
        else:
            assert isinstance(arg_type, DType)
            np_dtype = host_type(arg_type, HLS_HW_ABI).np_dtype
            data = np.asarray(arg, dtype=np_dtype).tobytes()
        (project_path / f"input{i}.data").write_bytes(data)


def read_impl_outputs(
    project_path: Path, arg_types: list[TypeBase], args: tuple
) -> None:
    """Read ``output<i>.data`` back into each buffer argument (in place)."""
    for i, (arg_type, arg) in enumerate(zip(arg_types, args)):
        if not isinstance(arg_type, BufferType):
            continue
        np_dtype = host_type(arg_type.dtype, HLS_HW_ABI).np_dtype
        data = np.fromfile(project_path / f"output{i}.data", dtype=np_dtype)
        arg[...] = data.reshape(arg_type.shape).astype(arg.dtype, copy=False)


class VitisEmulator:
    """Drives the generated ``impl.mk`` for sw_emu/hw_emu/hw and marshals numpy
    buffers to/from the XRT host's ``input<i>.data``/``output<i>.data`` files."""

    def __init__(
        self,
        *,
        top: str,
        project_path: Path,
        env: Mapping[str, str],
        arg_types: list[TypeBase],
        res_types: list[TypeBase],
    ):
        self.top = top
        self.project_path = Path(project_path)
        self.env = dict(env)
        self.arg_types = list(arg_types)
        self.res_types = list(res_types)

    @property
    def xclbin_path(self) -> Path:
        return self.project_path / f"{self.top}.xclbin"

    def precheck(self, mode: str) -> None:
        """Build only the fast, frontend-validating targets (.xo + host, plus
        emconfig for emulation), skipping the multi-hour / platform-locked link."""
        with stage(f"Vitis {mode} pre-check (kernel .xo + XRT host)"):
            self._make(["precheck"], mode)

    def build(self, mode: str) -> Path:
        """Full build to an ``.xclbin`` (emulation link or hw synth+impl)."""
        self._run(["all"], mode)
        return self.xclbin_path

    def _run(self, targets: list[str], mode: str) -> None:
        with stage(f"Vitis {mode} build/run"):
            self._make(targets, mode)

    def run(self, mode: str, *args) -> None:
        if len(args) != len(self.arg_types):
            raise ValueError(
                f"Expected {len(self.arg_types)} arguments, got {len(args)}"
            )
        write_impl_inputs(self.project_path, self.arg_types, args)
        self._run(["run"], mode)
        read_impl_outputs(self.project_path, self.arg_types, args)

    def _make(self, targets: list[str], mode: str):
        cmd = ["make", "-f", IMPL_MAKEFILE, f"TARGET={mode}", *targets]
        return run_command(cmd, cwd=self.project_path, env=self.env)
