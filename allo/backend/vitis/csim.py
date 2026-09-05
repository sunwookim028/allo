# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ctypes
import functools
import os
import re
import shutil
import subprocess

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from collections.abc import Mapping

import numpy as np

from .utils import _render_template
from ..marshal import HLS_CSIM_ABI, as_array, host_type, writeback
from ..base import write_text_if_changed
from ...lang.core import BufferType, DType, TypeBase
from ...logging import completed_output, log_debug, log_detail, run_command, stage

CSIM_MAKEFILE = "csim.mk"
CSIM_SHARED_LIBRARY = "libkernel.so"
# Source templates rendered into the project's CSIM_MAKEFILE. Vitis HLS compiled
# C simulation with gcc through 2024.2 (the legacy flow); Vitis 2025.2 switched
# the csim compiler to an AMD clang fork (driven by `-fhls-csim`, the native
# flow). The two differ only in toolchain/flags -- same emitted C++, same ABI.
CSIM_NATIVE_TEMPLATE = "csim.mk"
CSIM_LEGACY_TEMPLATE = "csim_legacy.mk"


def _version_key(path: Path) -> tuple[int, ...]:
    """Sort key from the trailing version digits of a dir name (e.g. clang-16,
    gcc-8.3.0), so globbed toolchains can be ordered newest-first."""
    nums = re.findall(r"\d+", path.name)
    return tuple(int(n) for n in nums) if nums else (0,)


@functools.cache
def _clang_supports_hls_csim(clang: str, host_lib: str) -> bool:
    """Whether ``clang`` accepts ``-fhls-csim`` (the AMD clang shipped since Vitis
    2025.2). Probed by compiling a trivial unit; cached per (clang, host_lib)."""
    env = dict(os.environ)
    if host_lib:
        env["LD_LIBRARY_PATH"] = host_lib + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    try:
        proc = subprocess.run(
            [
                clang,
                "-x",
                "c++",
                "-std=gnu++17",
                "-c",
                "-fhls-csim",
                "-fhlstoplevel=__allo_probe",
                "-",
                "-o",
                os.devnull,
            ],
            input='extern "C" void __allo_probe() {}\n',
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        return proc.returncode == 0
    except OSError:
        return False


def discover_csim_make_vars(vitis_root: Path) -> dict[str, str]:
    """Probe a Vitis install and return make vars for the *native* csim flow,
    which uses the AMD clang fork (the csim compiler since Vitis 2025.2). The
    directory *structure* is stable across versions while the version numbers in
    names drift, so paths are globbed rather than hardcoded. Raises if no clang
    accepting ``-fhls-csim`` is found, i.e. on pre-2025.2 (gcc-era) installs;
    ``discover_csim`` catches that and falls back to the legacy gcc flow."""
    vitis_root = Path(vitis_root)
    host_lib = vitis_root / "lib" / "lnx64.o"
    clangs = sorted(
        vitis_root.glob("lnx64/tools/clang-*/bin/clang++"),
        key=_version_key,
        reverse=True,
    )
    clang = next(
        (
            c
            for c in clangs
            if _clang_supports_hls_csim(os.fspath(c), os.fspath(host_lib))
        ),
        None,
    )
    if clang is None:
        raise RuntimeError(
            f"No Vitis clang supporting '-fhls-csim' found under {vitis_root}. "
            "Python-native C simulation requires the AMD clang shipped with "
            "Vitis 2025.2 or newer."
        )
    make_vars = {
        "CXX": os.fspath(clang),
        "VITIS_HOST_LIB": os.fspath(host_lib),
        "MATHHLS_LIB": os.fspath(vitis_root / "lnx64" / "lib" / "csim"),
    }
    gccs = sorted(vitis_root.glob("tps/lnx64/gcc-*"), key=_version_key, reverse=True)
    if gccs:
        make_vars["GCC_TOOLCHAIN"] = os.fspath(gccs[0])
    fpos = sorted(vitis_root.glob("lnx64/tools/fpo_*"))
    if fpos:
        make_vars["FPO_LIB"] = os.fspath(fpos[0])
    return make_vars


def _resolve_hls_root(vitis_root: Path) -> Path:
    """Return the dir whose ``include/`` holds ``ap_int.h``. Vitis 2024.2+ merged
    HLS into the Vitis install; 2023.2 keeps it in a sibling ``Vitis_HLS/<ver>``."""
    if (vitis_root / "include" / "ap_int.h").exists():
        return vitis_root
    sibling = vitis_root.parent.parent / "Vitis_HLS" / vitis_root.name
    if (sibling / "include" / "ap_int.h").exists():
        return sibling
    raise RuntimeError(
        f"Could not locate HLS headers (ap_int.h) under {vitis_root} or a sibling "
        "Vitis_HLS install. Legacy C simulation needs the Vitis HLS include dir."
    )


@functools.cache
def _detect_crt_dir() -> str:
    """Directory holding the system C-runtime startup objects (crti.o), queried
    from the system compiler. The bundled Vitis gcc needs ``-B`` to find them."""
    for cc in ("cc", "gcc", "g++"):
        exe = shutil.which(cc)
        if not exe:
            continue
        try:
            out = subprocess.run(
                [exe, "-print-file-name=crti.o"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
        except OSError:
            continue
        if out and out != "crti.o" and os.path.exists(out):
            return os.fspath(Path(out).parent)
    return ""


def discover_legacy_csim_make_vars(vitis_root: Path) -> dict[str, str]:
    """Probe a pre-2025.2 (gcc-era) Vitis install and return make vars for the
    legacy gcc C simulation: the version-matched bundled gcc, the HLS include
    dir, the hls_math/FP model lib dirs, and the system crt directory the bundled
    gcc needs via ``-B``."""
    vitis_root = Path(vitis_root)
    hls_root = _resolve_hls_root(vitis_root)
    gxxs = sorted(
        hls_root.glob("tps/lnx64/gcc-*/bin/g++"),
        key=lambda p: _version_key(p.parent.parent),
        reverse=True,
    )
    if not gxxs:
        raise RuntimeError(
            f"No bundled gcc found under {hls_root}/tps/lnx64. Legacy C simulation "
            "needs the gcc shipped with Vitis HLS."
        )
    make_vars = {
        "CXX": os.fspath(gxxs[0]),
        # HLS root: the template derives the official include set from it.
        "HLS_ROOT": os.fspath(hls_root),
        # hls_math / hls::half bit-accurate model libs (resolved at .so dlopen).
        "MATHHLS_LIB": os.fspath(hls_root / "lnx64" / "lib" / "csim"),
    }
    fpos = sorted((hls_root / "lnx64" / "tools").glob("fpo_*"))
    if fpos:
        make_vars["FPO_LIB"] = os.fspath(fpos[0])
    crt_dir = _detect_crt_dir()
    if crt_dir:
        make_vars["CRT_DIR"] = crt_dir
    return make_vars


@dataclass(frozen=True)
class CsimToolchain:
    """The chosen C-simulation flavor: which makefile template to render and the
    concrete, version-discovered make vars to drive it."""

    flavor: str  # "native" (2025.2+ -fhls-csim) | "legacy" (plain g++)
    template: str
    make_vars: dict[str, str] = field(default_factory=dict)


@functools.cache
def discover_csim(vitis_root: Path) -> CsimToolchain:
    """Pick the C-simulation flavor for a Vitis install. Vitis HLS used gcc as the
    csim compiler through 2024.2; 2025.2 switched to an AMD clang fork (driven by
    ``-fhls-csim``). We probe for that clang and use the native flow if present,
    else fall back to the legacy gcc flow. Both yield a ctypes-callable .so with
    the same ABI -- only the toolchain and link libraries differ."""
    vitis_root = Path(vitis_root)
    try:
        return CsimToolchain(
            "native", CSIM_NATIVE_TEMPLATE, discover_csim_make_vars(vitis_root)
        )
    except RuntimeError:
        return CsimToolchain(
            "legacy", CSIM_LEGACY_TEMPLATE, discover_legacy_csim_make_vars(vitis_root)
        )


def _generate_csim_makefile(
    vitis_root: Path, template: str = CSIM_NATIVE_TEMPLATE
) -> str:
    return _render_template(
        template,
        csim_shared_library=CSIM_SHARED_LIBRARY,
        vitis_root=os.fspath(vitis_root),
    )


def _csim_argtype(arg_type: TypeBase):
    if isinstance(arg_type, BufferType):
        return np.ctypeslib.ndpointer(
            dtype=host_type(arg_type.dtype, HLS_CSIM_ABI).np_dtype,
            ndim=len(arg_type.shape),
            flags="C_CONTIGUOUS",
        )
    if isinstance(arg_type, DType):
        return host_type(arg_type, HLS_CSIM_ABI).ctype
    raise TypeError(f"Unsupported Vitis Python-native csim argument type: {arg_type}")


def _pack_csim_arg(arg, arg_type: TypeBase):
    if isinstance(arg_type, BufferType):
        array = as_array(arg, arg_type, HLS_CSIM_ABI)
        return array, array
    if isinstance(arg_type, DType):
        return host_type(arg_type, HLS_CSIM_ABI).ctype(arg), None
    raise TypeError(f"Unsupported Vitis Python-native csim argument type: {arg_type}")


def _csim_return_type(res_types: list[TypeBase]):
    if not res_types:
        return None
    if len(res_types) == 1 and isinstance(res_types[0], DType):
        return host_type(res_types[0], HLS_CSIM_ABI).ctype
    raise TypeError("Vitis Python-native csim only supports void or scalar return")


class PythonNativeCSimulator:
    def __init__(
        self,
        *,
        top: str,
        project_path: Path,
        vitis_root: Path,
        env: Mapping[str, str],
        arg_types: list[TypeBase],
        res_types: list[TypeBase],
        make_vars: Mapping[str, str] | None = None,
        makefile_template: str = CSIM_NATIVE_TEMPLATE,
    ):
        self.top = top
        self.project_path = project_path
        self.vitis_root = vitis_root
        self.env = dict(env)
        self.arg_types = list(arg_types)
        self.res_types = list(res_types)
        self.make_vars = dict(make_vars or {})
        self.makefile_template = makefile_template
        self.library_path = self._resolve_project_path(
            self.make_vars.get("OUT", CSIM_SHARED_LIBRARY)
        )
        self._library: ctypes.CDLL | None = None
        self._function = None

    def run(self, *args, exist_ok: bool = True) -> Any:
        if len(args) != len(self.arg_types):
            raise ValueError(
                f"Expected {len(self.arg_types)} arguments, got {len(args)}"
            )
        self.build(exist_ok=exist_ok)
        func = self._get_function()
        packed_args = []
        arg_arrays = []
        for arg, arg_type in zip(args, self.arg_types):
            packed, array = _pack_csim_arg(arg, arg_type)
            packed_args.append(packed)
            if array is not None:
                arg_arrays.append((arg, array))

        with stage("Running Vitis C Simulation"):
            result = func(*packed_args)
            writeback(arg_arrays)
            return result

    def build(self, *, exist_ok: bool = True) -> Path:
        write_text_if_changed(
            self.project_path / CSIM_MAKEFILE,
            _generate_csim_makefile(self.vitis_root, self.makefile_template),
        )
        if self.library_path.exists() and exist_ok:
            log_debug(
                f"Building Vitis C Simulation Shared Library: {self.library_path} (cache hit)"
            )
            return self.library_path

        self._library = None
        self._function = None
        # The makefile recipe self-prepends VITIS_HOST_LIB to LD_LIBRARY_PATH for
        # the native clang; the legacy gcc needs nothing (the .so is rpath-baked).
        with stage("Building Vitis C Simulation Shared Library"):
            dry_run = run_command(
                self._make_command(dry_run=True),
                cwd=self.project_path,
                env=self.env,
            )
            self._log_make_commands(dry_run)
            run_command(self._make_command(), cwd=self.project_path, env=self.env)
        return self.library_path

    def _make_command(self, *, dry_run: bool = False) -> list[str]:
        cmd = ["make"]
        if dry_run:
            cmd.append("-n")
        cmd.extend(
            [
                "-f",
                CSIM_MAKEFILE,
                f"TOP={self.top}",
                *[f"{key}={value}" for key, value in self.make_vars.items()],
            ]
        )
        return cmd

    def _log_make_commands(self, result: subprocess.CompletedProcess[str]) -> None:
        output = completed_output(result)
        if output:
            log_detail(f"Make command:\n{output}")

    def _resolve_project_path(self, path: object) -> Path:
        resolved = Path(os.fspath(path) if isinstance(path, os.PathLike) else str(path))
        if resolved.is_absolute():
            return resolved
        return self.project_path / resolved

    def _get_function(self):
        if self._function is not None:
            return self._function
        if self._library is None:
            self._library = ctypes.CDLL(os.fspath(self.library_path))
        func = getattr(self._library, self.top)
        func.argtypes = [_csim_argtype(arg_type) for arg_type in self.arg_types]
        func.restype = _csim_return_type(self.res_types)
        self._function = func
        return func
