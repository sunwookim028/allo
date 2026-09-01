# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Functional simulator for hand-written assembly (``@oracle``).

An ``@oracle`` body is *traced* (run once); each instruction call appends an
``EmitRecord`` and each ``inspect`` appends an ``InspectRecord`` to the
active ``OracleProgram``. ``simulate`` then turns this stream into a module of
``allo.buffer`` + ``allo.define`` (catalog) + ``func @main`` (the emits, with a
``call @__inspect_k`` anchor per inspect) -> ``lower-instructions`` -> patch
buffer-global initializers (inputs) and replace each anchor with a
``get_global`` + ``copy`` into ``@main``'s output args (snapshots) ->
``allo-lower-to-llvm`` -> JIT, then read the snapshot arrays back.
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path

import ml_dtypes
import numpy as np

from ..._mlir import ir
from ..._mlir.ir import InsertionPoint, Location, Module
from ..._mlir.dialects import allo as allo_d, memref
from ..._mlir.execution_engine import ExecutionEngine
from ..._mlir.runtime import get_ranked_memref_descriptor

from ..._mlir.passmanager import PassManager
from ..._mlir.dialects.allo import register_passes as _register_allo_passes
from .codegen import INSPECT_PREFIX, build_main, emit_catalog
from .errors import AcceleratorDescriptionError, AssemblyError

_register_allo_passes()


def run_pipeline(module: ir.Module, pipeline: str) -> None:
    """Run a textual pass pipeline on ``module`` in its own context."""
    PassManager.parse(pipeline, module.context).run(module.operation)


_RUNNER_STEMS = ("libmlir_runner_utils", "libmlir_c_runner_utils")


@lru_cache(maxsize=1)
def _runner_utils() -> list[str]:
    """The MLIR runner-utils shared libraries, for the JIT symbols a lowered program
    can call into — ``_memrefCopy``, which a copy between differently-laid-out memrefs
    lowers to. Searched under ``$LLVM_BASE_DIR/lib`` then the in-tree LLVM build.

    Deliberately not ``backend.cpu._default_shared_libs``: that one also demands the
    dataflow runtime (raising if absent), which the functional simulator never calls.
    Returning ``[]`` when nothing is found leaves programs that need no runtime symbol
    working, and the ones that do fail with the JIT's own ``Symbols not found``."""
    base = os.environ.get("LLVM_BASE_DIR")
    roots = [Path(base) / "lib"] if base else []
    roots += [
        p / "externals" / "llvm-project" / "build" / "lib"
        for p in Path(__file__).resolve().parents
    ]
    for lib_dir in roots:
        found = [
            str(lib_dir / f"{stem}{ext}")
            for stem in _RUNNER_STEMS
            for ext in (".dylib", ".so")
            if (lib_dir / f"{stem}{ext}").exists()
        ]
        if len(found) == len(_RUNNER_STEMS):
            return found
    return []


# ==========================================================================#
# Recorded program (the traced emit/inspect stream) + simulator config
# ==========================================================================#


@dataclass
class EmitRecord:
    """One serialized instruction. In a *compiled* stream this is one epoch's
    total configuration on the wire — ``epoch.Epoch`` is its denotational
    reading; an ``@oracle`` stream is hand-written and makes no such promise
    (it may omit schedule fields, which the simulator ignores anyway)."""

    name: str  # instruction mnemonic
    addr: list  # address-param values (static ints)
    compute: list  # computational attributes (α): the instruction's immediates
    # Schedule params: instruction-word fields the compiler *chose*. They change the
    # configuration, never the value, so the simulator ignores them.
    schedule: list = field(default_factory=list)


@dataclass
class InspectRecord:
    buffer: object  # BufferSpec
    sl: object  # slice | int | None  (None = whole buffer)
    label: str | None


class OracleProgram:
    """The ordered emit/inspect stream collected while tracing an oracle body."""

    def __init__(self):
        self.steps: list[tuple[str, object]] = []

    def record_emit(self, name, addr, compute):
        self.steps.append(("emit", EmitRecord(name, list(addr), list(compute))))

    def record_inspect(self, buffer, sl, label):
        self.steps.append(("inspect", InspectRecord(buffer, sl, label)))

    @property
    def inspects(self) -> list[InspectRecord]:
        return [rec for kind, rec in self.steps if kind == "inspect"]


@dataclass
class OracleConfig:
    """Functional-simulator configuration for ``@oracle``."""

    # I/O & initial memory
    init: dict = field(
        default_factory=dict
    )  # {BufferSpec: ndarray}, written at offset 0
    mem_init: str = "zero"  # "zero" (HW reset) | "random" | "uninit"
    seed: int | None = None  # determinism for mem_init="random"
    # execution / backend
    opt_level: int = 2  # LLVM JIT optimization level
    # observability
    verbose: bool = False  # print each instruction + inspected values
    print_ir: bool = False  # dump built / lowered IR
    # validation
    verify: bool = True  # run the MLIR verifier on the built module
    # differential ("oracle") check
    reference: object = None  # callable() -> {label: ndarray} to diff against
    rtol: float = 1e-5
    atol: float = 1e-6


_NUMPY_DTYPE = {
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "uint64": np.uint64,
    "uint1": np.bool_,
    "bfloat16": ml_dtypes.bfloat16,
}


def _np_dtype(dtype) -> np.dtype:
    if dtype.name not in _NUMPY_DTYPE:
        raise AcceleratorDescriptionError(f"no host numpy dtype for {dtype.name}")
    return np.dtype(_NUMPY_DTYPE[dtype.name])


class Oracle:
    """A traced + simulatable assembly function produced by ``@oracle``."""

    def __init__(self, isa, fn, config: OracleConfig):
        self.isa = isa
        self.fn = fn
        self.config = config
        self.__name__ = getattr(fn, "__name__", "oracle")

    def __call__(self, **overrides):
        config = replace(self.config, **overrides) if overrides else self.config
        program = OracleProgram()
        prev = self.isa._active_oracle
        self.isa._active_oracle = program
        try:
            self.fn()
        finally:
            self.isa._active_oracle = prev
        return simulate(self.isa, program, config)


def simulate(isa, program: OracleProgram, config: OracleConfig) -> dict:
    context = ir.Context()
    allo_d.register_dialect(context)
    with context, Location.unknown(context):
        module = Module.create()
        with InsertionPoint(module.body):
            emit_catalog(context, isa, program)
            build_main(context, isa, program)
        if config.verify:
            if not module.operation.verify():
                raise AcceleratorDescriptionError("oracle module failed verification")
        if config.print_ir:
            print(module)

        run_pipeline(module, "builtin.module(lower-instructions)")
        # The inlined semantics are value-semantics TOSA; legalize to linalg/arith
        # (the resulting tensor.empty inits live only in the flat func, never in a
        # define region — so the SemanticsBuilder never has to clone them).
        run_pipeline(
            module,
            "builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,"
            "tosa-to-arith,tosa-to-tensor),canonicalize)",
        )
        _apply_initializers(module, isa, config)
        _wire_inspect_captures(context, module, isa, program)
        if config.print_ir:
            print(module)

        run_pipeline(module, "builtin.module(lower-to-llvm)")
        out_arrays = _jit_run(module, isa, program, config)

    results = _collect(program, out_arrays)
    if config.verbose:
        _report(program, results)
    if config.reference is not None:
        _diff(results, config)
    return results


# ==========================================================================#
# Post-lowering IR surgery
# ==========================================================================#


def _find_global(module: Module, name: str):
    for op in module.body.operations:
        if op.operation.name == "memref.global" and op.sym_name.value == name:
            return op
    return None


def _dense_initial_value(data, buf, context) -> ir.DenseElementsAttr:
    """A global's ``initial_value`` attribute from ``data``. bf16 has no numpy
    buffer-protocol exposure, so route its raw 2-byte values through a uint16 view
    plus an explicit bf16 tensor type (the form numpy + MLIR can both ingest)."""
    data = np.ascontiguousarray(data)
    if buf.kind.dtype.is_bf16():
        tt = ir.RankedTensorType.get(
            buf.memref_shape, buf.kind.dtype.materialize(context)
        )
        return ir.DenseElementsAttr.get(data.view(np.uint16), type=tt)
    return ir.DenseElementsAttr.get(data)


def _apply_initializers(module: Module, isa, config: OracleConfig):
    """Patch buffer-global initial values for inputs / random memory init."""
    rng = np.random.default_rng(config.seed)
    for buf in isa.buffers.values():
        glob = _find_global(module, buf.name)
        if glob is None:
            continue
        shape = buf.memref_shape
        np_dt = _np_dtype(buf.kind.dtype)
        if buf in config.init:
            data = np.zeros(shape, np_dt)
            flat = np.asarray(config.init[buf], np_dt).reshape(-1)
            if flat.size > data.size:
                raise AssemblyError(
                    f"init for '{buf.name}' has {flat.size} elems > "
                    f"capacity {data.size}"
                )
            data.reshape(-1)[: flat.size] = flat
        elif config.mem_init == "random":
            data = rng.standard_normal(shape).astype(np_dt)
        else:
            data = np.zeros(shape, np_dt)
        # Every global needs a defined initializer; an uninitialized memref.global
        # lowers to an external symbol the JIT cannot resolve.
        glob.attributes["initial_value"] = _dense_initial_value(
            data, buf, module.context
        )


def _wire_inspect_captures(context, module: Module, isa, program: OracleProgram):
    """Replace each ``call @__inspect_k`` with a snapshot of the inspected buffer
    into ``@main``'s k-th output arg, then drop the placeholder declarations."""
    main = ir.SymbolTable(module.operation)["main"]
    out_args = list(main.regions[0].blocks[0].arguments)
    inspects = program.inspects

    calls = [
        op
        for op in main.regions[0].blocks[0].operations
        if op.operation.name == "func.call"
        and op.attributes["callee"].value.startswith(INSPECT_PREFIX)
    ]
    for call in calls:
        k = int(call.attributes["callee"].value[len(INSPECT_PREFIX) :])
        buf = inspects[k].buffer
        mtype = ir.MemRefType.get(buf.memref_shape, buf.kind.dtype.materialize(context))
        with InsertionPoint(call):
            g = memref.get_global(mtype, buf.name)
            memref.copy(g, out_args[k])
        call.operation.erase()

    for op in list(module.body.operations):
        if op.operation.name == "func.func" and op.sym_name.value.startswith(
            INSPECT_PREFIX
        ):
            op.operation.erase()


# ==========================================================================#
# JIT execution + readback
# ==========================================================================#


def _jit_run(module: Module, isa, program: OracleProgram, config: OracleConfig):
    inspects = program.inspects
    packed, keepalive, out_arrays = [], [], []
    for ins in inspects:
        arr = np.zeros(ins.buffer.memref_shape, _np_dtype(ins.buffer.kind.dtype))
        desc = get_ranked_memref_descriptor(arr)
        ptr = ctypes.pointer(ctypes.pointer(desc))
        packed.append(ptr)
        keepalive += [arr, desc, ptr]
        out_arrays.append(arr)
    # A copy between differently-laid-out memrefs (any relayout: a strided gather, a
    # multi-dimensional block) lowers to a `memref.copy` call into `_memrefCopy`, so
    # the MLIR runner utils have to be loaded. Contiguous copies inline and do not.
    engine = ExecutionEngine(
        module, opt_level=config.opt_level, shared_libs=_runner_utils()
    )
    engine.invoke("main", *packed)
    return out_arrays


def _collect(program: OracleProgram, out_arrays) -> dict:
    results: dict = {}
    for ins, arr in zip(program.inspects, out_arrays):
        value = arr if ins.sl is None else arr[ins.sl]
        key = ins.label or ins.buffer.name
        if key in results:
            i = 2
            while f"{key}#{i}" in results:
                i += 1
            key = f"{key}#{i}"
        results[key] = value
    return results


def _report(program: OracleProgram, results: dict):
    for kind, rec in program.steps:
        if kind == "emit":
            args = ", ".join(map(str, rec.addr + rec.compute))
            print(f"  {rec.name}({args})")
    for key, value in results.items():
        print(f"  inspect {key} = {np.asarray(value).tolist()}")


def _diff(results: dict, config: OracleConfig):
    reference = config.reference()
    for key, expected in reference.items():
        if key not in results:
            raise AssemblyError(f"reference key '{key}' was not inspected")
        np.testing.assert_allclose(
            results[key], expected, rtol=config.rtol, atol=config.atol
        )
