# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ctypes
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import ParamSpec, TypeVar

import numpy as np

from .marshal import (
    LLVM_ABI,
    as_array,
    from_ctype_scalar,
    host_type,
    to_ctype_scalar,
    writeback,
)
from .utils import make_project_path

from ..lang.core import (
    BufferType,
    DType,
    StreamType,
    TypeBase,
)
from ..logging import stage, terminate_on_error
from .base import Backend, run_pipeline, set_top_llvm_c_wrapper
from ..lang.kernel import Kernel
from .._mlir import ir
from .._mlir.execution_engine import ExecutionEngine
from .._mlir.runtime import (
    as_ctype,
    get_ranked_memref_descriptor,
    make_nd_memref_descriptor,
    ranked_memref_to_numpy,
)


@dataclass
class _CPUCompileCacheEntry:
    module: ir.Module
    engine: ExecutionEngine
    arg_types: list[TypeBase]
    res_types: list[TypeBase]


def _dataflow_runtime_lib() -> str:
    lib_dir = Path(__file__).resolve().parent.parent / "_mlir" / "_mlir_libs"
    lib_name = "libAlloDataflowRuntime"
    if platform.system() == "Darwin":
        lib_name += ".dylib"
    elif platform.system() == "Linux":
        lib_name += ".so"
    else:
        raise RuntimeError(f"Unsupported platform: {platform.system()}")
    lib_path = lib_dir / lib_name
    if not lib_path.exists():
        raise RuntimeError(
            f"Cannot find Allo dataflow runtime library at {lib_path}. Installation may be broken."
        )
    return str(lib_path)


def _make_output_struct(memref_descriptors):
    fields = [
        (f"memref{i}", memref.__class__) for i, memref in enumerate(memref_descriptors)
    ]
    output_struct = type("OutputStruct", (ctypes.Structure,), {"_fields_": fields})()
    for i, memref in enumerate(memref_descriptors):
        setattr(output_struct, f"memref{i}", memref)
    return output_struct


def _pack_kernel_args(args, arg_types: list[TypeBase], res_types: list[TypeBase]):
    if len(args) != len(arg_types):
        raise ValueError(f"Expected {len(arg_types)} arguments, got {len(args)}")

    keepalive = []
    packed_args = []
    arg_arrays = []
    for arg, arg_type in zip(args, arg_types):
        ptr, obj, array = _pack_arg(arg, arg_type)
        packed_args.append(ptr)
        keepalive.append(obj)
        if array is not None:
            arg_arrays.append((arg, array))

    result_state = _pack_results(res_types)
    if result_state is None:
        return packed_args, keepalive, arg_arrays, None

    result_ptr, result_keepalive, result_decode = result_state
    keepalive.extend(result_keepalive)
    if len(res_types) == 1 and isinstance(res_types[0], DType):
        packed_args.append(result_ptr)
    else:
        packed_args.insert(0, result_ptr)
    return packed_args, keepalive, arg_arrays, result_decode


def _pack_arg(arg, arg_type: TypeBase):
    if isinstance(arg_type, BufferType):
        array = as_array(arg, arg_type, LLVM_ABI)
        desc = get_ranked_memref_descriptor(array)
        ptr = ctypes.pointer(ctypes.pointer(desc))
        return ptr, (array, desc, ptr), array

    if isinstance(arg_type, DType):
        value = to_ctype_scalar(arg, host_type(arg_type, LLVM_ABI))
        return value, value, None

    raise TypeError(f"Unsupported CPU argument type: {arg_type}")


def _pack_results(res_types: list[TypeBase]):
    if not res_types:
        return None

    if len(res_types) == 1 and isinstance(res_types[0], DType):
        host = host_type(res_types[0], LLVM_ABI)
        scalar = to_ctype_scalar(-1, host)
        return scalar, [scalar], lambda: from_ctype_scalar(scalar[0], host)

    descriptors = []
    keepalive = []
    for res_type in res_types:
        if not isinstance(res_type, BufferType):
            raise TypeError("Multiple CPU return values must be buffers")
        ctp = as_ctype(np.dtype(host_type(res_type.dtype, LLVM_ABI).np_dtype))
        desc = make_nd_memref_descriptor(len(res_type.shape), ctp)()
        descriptors.append(desc)
        keepalive.append(desc)

    if len(descriptors) == 1:
        ptr = ctypes.pointer(ctypes.pointer(descriptors[0]))
        keepalive.append(ptr)
        return ptr, keepalive, lambda: ranked_memref_to_numpy(ptr[0])

    output = _make_output_struct(descriptors)
    ptr = ctypes.pointer(ctypes.pointer(output))
    keepalive.extend([output, ptr])
    return (
        ptr,
        keepalive,
        lambda: [
            ranked_memref_to_numpy(ctypes.pointer(getattr(ptr[0][0], f"memref{i}")))
            for i in range(len(descriptors))
        ],
    )


P = ParamSpec("P")
R = TypeVar("R")


class CPU(Backend[P, R]):
    """
    Backend for executing kernels on the CPU using LLVM's JIT compilation.

    This backend lowers the kernel to LLVMIR Dialect, compiles it using MLIR's ExecutionEngine (LLVM JIT),
    and executes it directly on the CPU. It supports buffer arguments as numpy arrays and scalar arguments
    as Python scalars.

    Currently the CPU backend does not support the tensor ABI, or arbitrary APInt/APFloat types
    """

    name = "cpu"

    def __init__(
        self,
        kernel: Kernel[P, R],
        *,
        opt_level: int = 2,
        shared_libs: list[str] | None = None,
    ):
        super().__init__(kernel)
        self.opt_level = opt_level
        self.shared_libs = [_dataflow_runtime_lib()]
        self.shared_libs.extend(shared_libs or [])
        self.engine: ExecutionEngine | None = None
        self.arg_types: list[TypeBase] = []
        self.res_types: list[TypeBase] = []

    @terminate_on_error
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self.run(*args, **kwargs)

    @terminate_on_error
    def compile(self):
        if self.kernel.options.enable_tensor:
            raise NotImplementedError("CPU backend does not support tensor ABI yet")
        cache_key = self._cache_key(
            {
                "backend": self.name,
                "opt_level": self.opt_level,
                "shared_libs": self.shared_libs,
            }
        )
        cache = self._pcache_get("cpu.compile", cache_key)
        if cache is not None:
            self.module = cache.module
            self.engine = cache.engine
            self.arg_types = cache.arg_types
            self.res_types = cache.res_types
            return self.module
        cache = self._build_pcache(self.shared_libs)
        self._pcache_set("cpu.compile", cache_key, cache)
        self.module = cache.module
        self.engine = cache.engine
        self.arg_types = cache.arg_types
        self.res_types = cache.res_types
        return self.module

    def _build_pcache(self, shared_libs: list[str]) -> _CPUCompileCacheEntry:
        with stage("Compiling CPU Kernels"):
            arg_types = self.kernel.parse_argument_annotations()
            res_types = self.kernel.parse_return_annotation()
            if any(isinstance(ty, StreamType) for ty in arg_types):
                raise NotImplementedError(
                    "CPU backend does not support stream top-level arguments"
                )

            # Wrap a non-standard-width APInt boundary with a std-width interface
            # so the LLVM memref ABI is numpy-representable. No-op otherwise. Runs
            # before set_top_llvm_c_wrapper so the wrapper takes the public name.
            run_pipeline(
                self.module,
                "builtin.module(generate-apint-wrapper{"
                f"top={self.kernel.func_name}}})",
            )
            if not set_top_llvm_c_wrapper(self.module, self.kernel.func_name):
                raise RuntimeError(
                    f"Cannot find top function '{self.kernel.func_name}'"
                )
            run_pipeline(self.module, "builtin.module(lower-to-llvm)")
            engine = ExecutionEngine(
                self.module,
                opt_level=self.opt_level,
                shared_libs=shared_libs,
            )
            return _CPUCompileCacheEntry(
                module=self.module,
                engine=engine,
                arg_types=arg_types,
                res_types=res_types,
            )

    @terminate_on_error
    def run(self, *args: P.args, **kwargs: P.kwargs) -> R:
        self._ensure_compiled()
        packed_args, _, arg_arrays, result_decode = _pack_kernel_args(
            args, self.arg_types, self.res_types
        )
        with stage("Running CPU Kernels (JIT)"):
            assert self.engine is not None
            self.engine.invoke(self.kernel.func_name, *packed_args)
            writeback(arg_arrays)
            if result_decode is None:
                return None  # type: ignore
            return result_decode()  # type: ignore

    @terminate_on_error
    def scaffold_project(
        self,
        project: str | None = None,
        *,
        exist_ok: bool = True,
    ) -> Path:
        project_path = make_project_path(project, self.kernel.func_name, exist_ok)
        self._ensure_compiled()
        assert self.module is not None
        (project_path / "lowered.mlir").write_text(str(self.module), encoding="utf-8")
        return project_path

    def _ensure_compiled(self):
        if self.engine is None:
            self.compile()
