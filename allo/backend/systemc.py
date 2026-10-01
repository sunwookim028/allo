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


def write_inputs(module, top, inputs, args, project):
    """Write the arrays the testbench reads as ``input<k>.data``.

    A SystemC region is a void function, so every argument is an input by
    signature. Only ``in`` and ``both`` arguments are preloaded; ``out``
    arguments start from the testbench's own zero-initialised memory.
    Returns the directions, for :func:`read_outputs`.
    """
    dirs = analyze_arg_load_store(module)[top]
    k = 0
    for (_, shape), arg, d in zip(inputs, args, dirs):
        if d in {"in", "both"}:
            write_tensor_to_file(arg, shape, f"{project}/input{k}.data")
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
            store_output(arg, read_tensor_from_file(dtype, shape, fpath))
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
