# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import functools

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping
from typing import Any

from ...logging import (
    CommandError,
    completed_output,
    log_info,
    log_tail,
    read_text_tail,
    stage,
)

SYNTH_LOG = Path("logs") / "hls_run_tcl.log"
LOG_FAILURE_TAIL_LINES = 100
TEMPLATE_DIR = Path(__file__).with_name("templates")
DEFAULT_VITIS_HOME = Path("/opt/xilinx/2025.2/Vitis")

INTERFACE_MODES = ("m_axi", "axis", "s_axilite")
_AXI_OFFSET_VALUES = {"off", "direct", "slave"}
_AXIS_REGISTER_MODE_VALUES = {"forward", "reverse", "both", "off"}
_AXILITE_STORAGE_IMPL_VALUES = {"auto", "bram", "uram"}
_INTERFACE_OPTION_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_AXI_OPTION_ORDER = (
    "offset",
    "bundle",
    "channel",
    "depth",
    "latency",
    "num_read_outstanding",
    "num_write_outstanding",
    "max_read_burst_length",
    "max_write_burst_length",
    "max_widen_bitwidth",
    "alignment_byte_size",
    "name",
)
_AXIS_OPTION_ORDER = ("register", "register_mode", "depth", "name", "bundle")
_AXILITE_OPTION_ORDER = (
    "bundle",
    "register",
    "clock",
    "name",
    "offset",
    "storage_impl",
)
_VITIS_VERSION_RE = re.compile(r"\bv?(\d{4}\.\d+(?:\.\d+)?)\b")


@dataclass(frozen=True)
class VitisTool:
    name: str
    executable: Path
    env: dict[str, str]
    version: str = "unknown"


def _render_template(name: str, **kwargs) -> str:
    return (TEMPLATE_DIR / name).read_text(encoding="utf-8").format(**kwargs)


def generate_hls_cfg(top: str, part: str, freq_mhz: float, flow_target: str) -> str:
    """Render the ``hls.cfg`` config consumed by ``v++ -c --mode hls`` for the
    standalone C-synthesis (csynth) flow."""
    clock_period = 1000.0 / freq_mhz
    return _render_template(
        "hls.cfg",
        top=top,
        part=part,
        flow_target=flow_target,
        clock_period=clock_period,
    )


def generate_run_tcl(top: str, part: str, freq_mhz: float, flow_target: str) -> str:
    """Render the ``run.tcl`` script consumed by ``vitis-run --mode hls`` for
    the emulation/hardware flow."""
    period = 1000.0 / freq_mhz
    return _render_template(
        "run.tcl",
        top=top,
        part=part,
        period=period,
        flow_target=flow_target,
    )


def _top_signature_marker(top: str) -> str:
    return f" {top}("


def _strip_line_comment(text: str) -> str:
    """Drop a trailing ``// ...`` comment (e.g. a ``with_location`` annotation)
    so line-structure checks see the real code ending (``;`` / ``{``). Signature
    lines contain no other ``//``, so the first occurrence starts the comment."""
    idx = text.find("//")
    return (text[:idx] if idx != -1 else text).rstrip()


def _extract_top_declaration(hls_code: str, top: str) -> str:
    marker = _top_signature_marker(top)
    for line in hls_code.splitlines():
        stripped = _strip_line_comment(line.strip())
        if marker in stripped and stripped.endswith(";"):
            return stripped.removeprefix('extern "C" ').strip()
    raise RuntimeError(f"Failed to find emitted declaration for top function {top}")


def generate_kernel_header(hls_code: str, top: str) -> str:
    declaration = _extract_top_declaration(hls_code, top)
    return _render_template("kernel.h", declaration=declaration)


def _split_cpp_arguments(arguments: str) -> list[str]:
    parts = []
    start = 0
    angle_depth = 0
    bracket_depth = 0
    for i, char in enumerate(arguments):
        if char == "<":
            angle_depth += 1
        elif char == ">" and angle_depth:
            angle_depth -= 1
        elif char == "[":
            bracket_depth += 1
        elif char == "]" and bracket_depth:
            bracket_depth -= 1
        elif char == "," and angle_depth == 0 and bracket_depth == 0:
            parts.append(arguments[start:i].strip())
            start = i + 1
    tail = arguments[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _extract_cpp_arg_name(argument: str) -> str:
    arg = argument.strip()
    while arg.endswith("]"):
        arg = re.sub(r"\s*\[[^\]]*\]\s*$", "", arg).rstrip()
    match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*$", arg)
    if match is None:
        raise RuntimeError(f"Failed to parse C++ argument name from '{argument}'")
    return match.group(1)


def _extract_top_port_names(hls_code: str, top: str) -> list[str]:
    marker = _top_signature_marker(top)
    for line in hls_code.splitlines():
        stripped = _strip_line_comment(line.strip())
        if marker not in stripped or not stripped.endswith("{"):
            continue
        args_begin = stripped.find(marker) + len(marker)
        args_end = stripped.rfind(")")
        if args_end < args_begin:
            raise RuntimeError(f"Failed to parse emitted definition for {top}")
        arguments = stripped[args_begin:args_end].strip()
        if not arguments:
            return []
        return [_extract_cpp_arg_name(arg) for arg in _split_cpp_arguments(arguments)]
    raise RuntimeError(f"Failed to find emitted definition for top function {top}")


def _validate_interface_option_name(name: str) -> None:
    if not _INTERFACE_OPTION_RE.match(name):
        raise ValueError(f"Invalid Vitis HLS interface option name '{name}'")


def _validate_non_empty_string(name: str, value: object) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Vitis HLS interface option '{name}' must be a string")


def _validate_positive_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"Vitis HLS interface option '{name}' must be a positive integer"
        )


def _validate_optional_bool(name: str, value: object) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"Vitis HLS interface option '{name}' must be a boolean")


def normalize_interface_options(
    mode: str,
    options: Mapping[str, Any],
) -> dict[str, str | int | bool | None]:
    normalized: dict[str, str | int | bool | None] = {}
    for name, value in options.items():
        _validate_interface_option_name(name)
        if value is None:
            continue
        if name == "register":
            _validate_optional_bool(name, value)
            normalized[name] = value
            continue
        if name in {
            "depth",
            "latency",
            "num_read_outstanding",
            "num_write_outstanding",
            "max_read_burst_length",
            "max_write_burst_length",
            "max_widen_bitwidth",
            "alignment_byte_size",
        }:
            _validate_positive_int(name, value)
            normalized[name] = value
            continue
        if name == "offset" and mode == "m_axi":
            _validate_non_empty_string(name, value)
            if value not in _AXI_OFFSET_VALUES:
                raise ValueError(
                    "Vitis HLS m_axi offset must be one of: off, direct, slave"
                )
            normalized[name] = value
            continue
        if name in {"bundle", "channel", "name", "clock", "offset"}:
            _validate_non_empty_string(name, value)
            normalized[name] = value
            continue
        if name == "register_mode":
            _validate_non_empty_string(name, value)
            if value not in _AXIS_REGISTER_MODE_VALUES:
                raise ValueError(
                    "Vitis HLS axis register_mode must be one of: "
                    "forward, reverse, both, off"
                )
            normalized[name] = value
            continue
        if name == "storage_impl":
            _validate_non_empty_string(name, value)
            if value not in _AXILITE_STORAGE_IMPL_VALUES:
                raise ValueError(
                    "Vitis HLS s_axilite storage_impl must be one of: "
                    "auto, bram, uram"
                )
            normalized[name] = value
            continue
        if isinstance(value, os.PathLike):
            value = os.fspath(value)
        if isinstance(value, bool):
            normalized[name] = value
        elif isinstance(value, int):
            normalized[name] = value
        elif isinstance(value, str):
            if not value:
                raise ValueError(
                    f"Vitis HLS interface option '{name}' must not be empty"
                )
            normalized[name] = value
        else:
            raise TypeError(
                f"Unsupported Vitis HLS interface option value for '{name}': "
                f"{type(value).__name__}"
            )
    return normalized


def _interface_option_order(mode: str) -> tuple[str, ...]:
    if mode == "m_axi":
        return _AXI_OPTION_ORDER
    if mode == "axis":
        return _AXIS_OPTION_ORDER
    if mode == "s_axilite":
        return _AXILITE_OPTION_ORDER
    raise ValueError(f"Unsupported Vitis HLS interface mode '{mode}'")


def _render_interface_options(
    options: Mapping[str, str | int | bool | None],
    order: tuple[str, ...],
) -> list[str]:
    rendered = []
    remaining = dict(options)
    for name in order:
        if name in remaining:
            value = remaining.pop(name)
            if value is True:
                rendered.append(name)
            elif value not in (False, None):
                rendered.append(f"{name}={value}")
    for name in sorted(remaining):
        value = remaining[name]
        if value is True:
            rendered.append(name)
        elif value not in (False, None):
            rendered.append(f"{name}={value}")
    return rendered


def _render_interface_pragma(pragma: Any, port: str) -> str:
    options = _render_interface_options(
        pragma.options, _interface_option_order(pragma.mode)
    )
    suffix = " " + " ".join(options) if options else ""
    return f"#pragma HLS interface mode={pragma.mode} port={port}{suffix}"


def apply_interface_pragmas(
    hls_code: str,
    top: str,
    pragmas: Mapping[int, Mapping[str, Any]],
) -> str:
    if not pragmas:
        return hls_code

    ports = _extract_top_port_names(hls_code, top)
    lines = []
    inserted = False
    marker = _top_signature_marker(top)
    mode_order = {mode: i for i, mode in enumerate(INTERFACE_MODES)}
    index_order = sorted(pragmas, key=lambda index: (index == -1, index))
    for line in hls_code.splitlines():
        lines.append(line)
        stripped = _strip_line_comment(line.strip())
        if inserted or marker not in stripped or not stripped.endswith("{"):
            continue
        indent = line[: len(line) - len(line.lstrip())] + "  "
        for index in index_order:
            port = "return" if index == -1 else ports[index]
            for mode, pragma in sorted(
                pragmas[index].items(), key=lambda item: mode_order[item[0]]
            ):
                lines.append(indent + _render_interface_pragma(pragma, port))
        inserted = True
    if not inserted:
        raise RuntimeError(f"Failed to insert interface pragmas for {top}")
    return "\n".join(lines) + ("\n" if hls_code.endswith("\n") else "")


def log_failure_tail(cmd_name: str, log_path: Path, error: Exception) -> None:
    tail = read_text_tail(log_path, max_lines=LOG_FAILURE_TAIL_LINES)
    if not tail and isinstance(error, CommandError):
        tail = error.output_tail(LOG_FAILURE_TAIL_LINES)
    log_tail(f"{cmd_name} log tail", tail, max_lines=LOG_FAILURE_TAIL_LINES)


def source_settings_env(settings64: Path) -> dict[str, str] | None:
    if not settings64.exists():
        return None
    command = f"source {shlex.quote(str(settings64))} >/dev/null 2>&1 && env"
    result = subprocess.run(
        ["bash", "-lc", command],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    env = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            env[key] = value
    return env


def _version_commands(tool_name: str) -> tuple[str, ...]:
    if tool_name == "vitis-run":
        return ("--version",)
    return ("-version", "--version")


def _probe_vitis_version(
    executable: Path, tool_name: str, env: Mapping[str, str]
) -> str:
    for version_arg in _version_commands(tool_name):
        result = subprocess.run(
            [os.fspath(executable), version_arg],
            env=dict(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        output = completed_output(result)
        if output:
            match = _VITIS_VERSION_RE.search(output)
            # return match.group(1) if match is not None else "unknown"
            assert match is not None, f"invalid Vitis version output: {output}"
            version = match.group(1)
            return version
    assert False, f"Failed to detect Vitis version from output: {output}"


def _find_tool_in_env(env: Mapping[str, str]) -> VitisTool | None:
    path = env.get("PATH", "")
    for tool_name in ("vitis-run", "vitis_hls"):
        executable = shutil.which(tool_name, path=path)
        if executable:
            tool_path = Path(executable)
            return VitisTool(
                tool_name,
                tool_path,
                dict(env),
                _probe_vitis_version(tool_path, tool_name, env),
            )
    return None


@functools.cache
def probe_vitis_tool(settings64: Path) -> VitisTool:
    sourced_env = source_settings_env(settings64)
    if sourced_env is not None:
        tool = _find_tool_in_env(sourced_env)
        if tool is not None:
            return tool

    tool = _find_tool_in_env(os.environ)
    if tool is None:
        raise RuntimeError(f"Failed to detect Vitis HLS toolchain with {settings64}")
    return tool


def detect_vitis_tool(settings64: Path) -> VitisTool:
    with stage("Detecting Vitis HLS Toolchain"):
        tool = probe_vitis_tool(settings64)
    log_info(f"Using Vitis {tool.executable}, Version: {tool.version}")
    return tool


def detect_vitis_home(vitis_home: str | None) -> Path:
    """Best guess of the Vitis install root: explicit argument, then
    ``$XILINX_HLS``/``$XILINX_VITIS``, then the packaged default."""
    if vitis_home:
        return Path(vitis_home)
    vitis_env = os.environ.get("XILINX_VITIS") or os.environ.get("XILINX_HLS")
    if vitis_env:
        return Path(vitis_env)
    return DEFAULT_VITIS_HOME


@functools.cache
def is_vitis_available(vitis_home: str | None = None) -> bool:
    """Whether a Vitis HLS toolchain can be detected, as a plain cached bool.

    Unlike ``detect_vitis_tool`` this never raises and emits no logs, so it is
    safe to use directly in ``pytest.mark.skipif`` predicates."""
    settings64 = detect_vitis_home(vitis_home) / "settings64.sh"
    try:
        probe_vitis_tool(settings64)
        return True
    except Exception:
        return False


def vitis_supports_apfloat(tool: VitisTool) -> bool:
    """Whether the detected tool can emit ap_float (bf16/tf32) types. Supported
    by the 2023.1+ ``vitis-run`` launcher; the legacy ``vitis_hls`` is not."""
    return tool.name == "vitis-run"
