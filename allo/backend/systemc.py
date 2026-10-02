# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-name-in-module
"""SystemC backend (``target="systemc"``).

The emitter is ``mlir/lib/Translation/EmitSystemC.cpp`` (``emit_systemc``). It
shares the Catapult flow: :class:`~allo.backend.hls.HLSModule` drives it, the
project uses ``allo/harness/catapult``, and ``run.tcl`` comes from
:func:`~allo.backend.catapult.codegen_tcl_catapult`. This module holds the
SystemC-only steps of that flow. See ``docs/source/backends/systemc.rst``.
"""

import os
import re

import numpy as np

from .._mlir.ir import StringAttr
from ..ir.transform import find_func_in_module
from ..passes import analyze_arg_load_store
from .vitis import read_tensor_from_file, write_tensor_to_file

_DIR_CHAR = {"in": "i", "out": "o", "both": "b", "scalar": "_"}

# Catapult 2024.2 dropped `solution app` csim, so the project gets a standalone
# runner for the self-contained kernel.cpp (it carries its own sc_main).
CSIM_SCRIPT = (
    "#!/bin/bash\n"
    "# Standalone SystemC behavioral csim for the self-contained\n"
    "# kernel.cpp (compile+run its sc_main tb with Catapult's g++).\n"
    "set -e\n"
    ': "${MGC_HOME:?set MGC_HOME to your Catapult Mgc_home}"\n'
    'GXX="$MGC_HOME/bin/g++"\n'
    'INC="$MGC_HOME/shared/include"\n'
    'LIB=$(ls -d "$MGC_HOME"/shared/lib/Linux/gcc-*-64 2>/dev/null | head -1)\n'
    '"$GXX" -std=c++11 -DSC_INCLUDE_DYNAMIC_PROCESSES -I"$INC" \\\n'
    '  kernel.cpp -o csim_sim -L"$LIB" -Wl,-rpath,"$LIB" -lsystemc\n'
    'LD_LIBRARY_PATH="$MGC_HOME/lib:$LIB" ./csim_sim\n'
)


def stamp_arg_dirs(module):
    """Stamp every function's per-argument direction as ``arg_dirs``.

    The emitter picks port directions (``Connections::In`` vs ``Out``) from it.
    ``analyze_arg_load_store`` derives directions from the actual loads and
    stores, propagated through calls, which ``itypes`` does not carry.
    """
    for fname, dirs in analyze_arg_load_store(module).items():
        func = find_func_in_module(module, fname)
        if func is not None:
            func.attributes["arg_dirs"] = StringAttr.get(
                "".join(_DIR_CHAR.get(d, "_") for d in dirs)
            )


# Floats cross the testbench's data files as their raw IEEE bits, written as an
# unsigned integer per line: the emitted tb rebuilds the value with set_data()
# (``_ffrombits``) and writes it back with ``_fbits``. Decimal text was not
# exact: libstdc++'s ``>> float`` fails on "nan"/"inf" (and every read after
# it), a NaN's sign and payload are lost, and ac::bfloat16(float) truncates, so
# the shortest decimal of a bf16 value read back as the bf16 below it.
_FLOAT_BITS = {"f16": 16, "bf16": 16, "f32": 32, "f64": 64}
_UINT = {16: np.uint16, 32: np.uint32, 64: np.uint64}
_FLOAT_NP = {"f16": np.float16, "f32": np.float32, "f64": np.float64}


def _bf16_np():
    try:
        import ml_dtypes  # pylint: disable=import-outside-toplevel

        return ml_dtypes.bfloat16
    except ImportError:
        return None


def float_to_bits(dtype, arr):
    """``arr`` as the unsigned bit patterns of float type ``dtype``."""
    dtype = str(dtype)
    arr = np.asarray(arr)
    if dtype == "bf16":
        bf16 = _bf16_np()
        if bf16 is not None and arr.dtype == bf16:
            return arr.view(np.uint16)
        if bf16 is not None:
            # ml_dtypes rounds to nearest even, like the simulator's bf16
            return np.ascontiguousarray(arr.astype(bf16)).view(np.uint16)
        f = np.ascontiguousarray(arr.astype(np.float32)).view(np.uint32)
        return (f >> 16).astype(np.uint16)  # exact for bf16-valued float32
    t = _FLOAT_NP[dtype]
    return np.ascontiguousarray(arr.astype(t)).view(_UINT[_FLOAT_BITS[dtype]])


def bits_to_float(dtype, bits):
    """The inverse of :func:`float_to_bits`; bf16 comes back as ml_dtypes'
    bfloat16 when available, else as the (exactly equal) float32."""
    dtype = str(dtype)
    bits = np.asarray(bits).astype(_UINT[_FLOAT_BITS[dtype]])
    if dtype == "bf16":
        bf16 = _bf16_np()
        if bf16 is not None:
            return bits.view(bf16)
        return (bits.astype(np.uint32) << 16).view(np.float32)
    return bits.view(_FLOAT_NP[dtype])


def write_data(dtype, arr, shape, path):
    """Write one ``input<k>.data`` file: floats as bits, integers as text."""
    if str(dtype) not in _FLOAT_BITS:
        write_tensor_to_file(arr, shape, path)
        return
    bits = float_to_bits(dtype, arr).reshape(-1)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(str(int(b)) for b in bits.tolist()))
        f.write("\n")


def _int_container(dtype):
    """The numpy container of an ``i<N>``/``ui<N>`` of any width up to 64
    (``ui24`` -> uint32), as the simulator's argument path accepts it; None
    for anything else."""
    m = re.fullmatch(r"(u?)i(\d+)", str(dtype))
    if not m or int(m.group(2)) > 64:
        return None
    width = next(w for w in (8, 16, 32, 64) if int(m.group(2)) <= w)
    return np.dtype(f"{'u' if m.group(1) else ''}int{width}")


def _wide_int_width(dtype):
    """The width of an ``i<N>``/``ui<N>`` wider than 64 bits, else 0."""
    m = re.fullmatch(r"(u?)i(\d+)", str(dtype))
    return int(m.group(2)) if m and int(m.group(2)) > 64 else 0


def read_data(dtype, shape, path):
    """Read one ``output<k>.data`` file written by the emitted testbench."""
    if str(dtype) not in _FLOAT_BITS:
        container = _int_container(dtype)
        if container is not None:
            # the tb prints integers in decimal; any width, not only 8/16/32/64
            vals = np.loadtxt(path, dtype=np.int64 if container.kind == "i" else np.uint64, ndmin=1)
            return vals.astype(container).reshape(shape)
        if _wide_int_width(dtype):
            # > 64 bits: the tb prints decimal text of any length (_wrwide);
            # numpy has no such integer, so hand back Python ints. Storing them
            # into a narrower array refuses (OverflowError) a value it cannot hold.
            with open(path, encoding="utf-8") as f:
                vals = [int(t) for t in f.read().split()]
            out = np.empty(len(vals), dtype=object)
            out[:] = vals
            return out.reshape(shape)
        return read_tensor_from_file(dtype, shape, path)
    bits = np.loadtxt(path, dtype=np.uint64, ndmin=1)
    return bits_to_float(dtype, bits).reshape(shape)


def bits_equal(dtype, a, b):
    """Exact equality; floats by bit pattern (NaN == NaN, -0 != +0)."""
    if str(dtype) in _FLOAT_BITS:
        return np.array_equal(float_to_bits(dtype, a), float_to_bits(dtype, b))
    return np.array_equal(a, b)


def write_inputs(module, top, inputs, args, project):
    """Write the arrays the testbench reads as ``input<k>.data``.

    A SystemC region is a void function, so every argument is an input by
    signature. Only ``in`` and ``both`` arguments are preloaded; ``out``
    arguments start from the testbench's own zero-initialised memory.
    Returns the directions, for :func:`read_outputs`.
    """
    dirs = analyze_arg_load_store(module)[top]
    k = 0
    for (dtype, shape), arg, d in zip(inputs, args, dirs):
        if d in {"in", "both"}:
            write_data(dtype, arg, shape, f"{project}/input{k}.data")
            k += 1
    return dirs


def read_outputs(inputs, args, dirs, project, store_output):
    """Read ``output<k>.data`` back into the ``out`` and ``both`` arguments."""
    k = 0
    for (dtype, shape), arg, d in zip(inputs, args, dirs):
        if d in {"out", "both"}:
            fpath = f"{project}/output{k}.data"
            if not os.path.exists(fpath):
                raise RuntimeError(
                    f"Output file {fpath} not found. Simulation might have failed."
                )
            store_output(arg, read_data(dtype, shape, fpath))
            k += 1


def compile_command(project, ac_include, what):
    """The g++ command that builds the emitted testbench into ``sim``.

    ``SYSTEMC_HOME`` names the SystemC install; ``ALLO_CXX_EXTRA`` carries
    host-specific flags (e.g. a newer libstdc++ for libsystemc's GLIBCXX).
    """
    systemc_home = os.environ.get("SYSTEMC_HOME", "")
    if not systemc_home:
        raise RuntimeError(f"Set SYSTEMC_HOME for systemc {what}.")
    lib_dir = os.path.join(systemc_home, "lib-linux64")
    if not os.path.isdir(lib_dir):
        lib_dir = os.path.join(systemc_home, "lib")
    cxx_extra = os.environ.get("ALLO_CXX_EXTRA", "")
    return (
        f"cd {project}; g++ -std=c++17 "
        f"-I{ac_include} -I{systemc_home}/include "
        f"{cxx_extra} kernel.cpp "
        f"-L{lib_dir} -Wl,-rpath,{lib_dir} -lsystemc -o sim"
    )
