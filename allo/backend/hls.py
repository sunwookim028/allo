# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=consider-using-with, no-name-in-module, too-many-branches

import os
import re
import io
import signal
import subprocess
import time
import numpy as np
from .._mlir.dialects import allo as allo_d
from .._mlir.ir import (
    Context,
    Location,
    Module,
    StringAttr,
    UnitAttr,
)
from .._mlir.passmanager import PassManager

from .config import DEFAULT_CONFIG, PART_NUMBER
from .vitis import (
    codegen_host,
    postprocess_hls_code,
    generate_description_file,
    write_tensor_to_file,
    read_tensor_from_file,
    generate_hbm_config,
    extract_hls_arg_names,
)
from .pynq import (
    postprocess_hls_code_pynq,
    codegen_pynq_host,
)
from .tapa import (
    codegen_tapa_host,
)
from .catapult import (
    codegen_tcl as codegen_tcl_catapult,
    codegen_host as codegen_host_catapult,
    parse_catapult_report,
    parse_catapult_hierarchical_report,
)
from .ip import IPModule
from .report import parse_xml
from ..passes import (
    _mlir_lower_pipeline,
    decompose_library_function,
    generate_input_output_buffers,
    analyze_arg_load_store,
)
from ..harness.makefile_gen.makegen import generate_makefile
from ..ir.transform import find_func_in_module
from ..utils import (
    get_func_inputs_outputs,
    c2allo_type,
    get_bitwidth_from_type,
    np_supported_types,
)


PIPELINE_STYLE_PLATFORMS = {"vivado_hls", "vitis_hls", "pynq"}


def styled_pipelines(op):
    """Every loop carrying `s.pipeline(style=)`, as (axis, style)."""
    for region in op.regions:
        for block in region.blocks:
            for child in block.operations:
                inner = child.operation
                if "pipeline_style" in inner.attributes:
                    axis = (
                        StringAttr(inner.attributes["loop_name"]).value
                        if "loop_name" in inner.attributes
                        else inner.name
                    )
                    yield axis, StringAttr(inner.attributes["pipeline_style"]).value
                yield from styled_pipelines(inner)


def check_pipeline_style_reaches_emitter(mod, platform):
    """Only the Vivado/Vitis emitter writes `style=`. A style is chosen to stop
    an RTL deadlock, so dropping it silently would restore the deadlock the
    author was fixing; the emitters that ignore it refuse instead."""
    if platform in PIPELINE_STYLE_PLATFORMS:
        return
    styled = list(styled_pipelines(mod.operation))
    if not styled:
        return
    named = ", ".join(f"{axis} (style={style})" for axis, style in styled)
    raise RuntimeError(
        f"pipeline: the {platform} emitter does not write `style=`, so the "
        f"pipeline control style on {named} would be dropped and the RTL would "
        f"use that tool's default. This is refused rather than dropped because "
        f"a style is chosen to stop an RTL deadlock that no simulation shows. "
        f"Build for vitis_hls/vivado_hls/pynq, or drop `style=` if the style is "
        f"not needed on {platform}."
    )


def _find_catapult_binary():
    """Return the path to the catapult binary, searching common install locations."""
    import shutil
    import glob as _glob

    # 1. Explicit MGC_HOME env var (highest priority)
    mgc_home = os.environ.get("MGC_HOME", "")
    if mgc_home:
        candidate = os.path.join(mgc_home, "bin", "catapult")
        if os.path.isfile(candidate):
            return candidate

    # 2. Siemens standard install path: /opt/siemens/catapult/<version>/
    siemens_root = "/opt/siemens/catapult"
    if os.path.isdir(siemens_root):
        # Pick the most recent version (sort descending)
        versions = sorted(
            (d for d in os.listdir(siemens_root)
             if os.path.isdir(os.path.join(siemens_root, d))),
            reverse=True,
        )
        for ver in versions:
            candidate = os.path.join(siemens_root, ver, "bin", "catapult")
            if os.path.isfile(candidate):
                os.environ.setdefault("MGC_HOME", os.path.join(siemens_root, ver))
                return candidate

    # 3. System PATH fallback
    found = shutil.which("catapult")
    if found:
        return found

    raise RuntimeError(
        "Catapult binary not found. Set MGC_HOME to the Catapult installation "
        "directory (e.g. MGC_HOME=/opt/siemens/catapult/2024.2) or add it to PATH."
    )


def is_available(backend="vivado_hls"):
    if backend == "vivado_hls":
        return os.system("which vivado_hls >> /dev/null") == 0
    if backend == "tapa":
        return os.system("which tapa >> /dev/null") == 0
    return os.system("which vitis_hls >> /dev/null") == 0


def run_process(cmd, pattern=None):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    out, err = p.communicate()
    if err:
        raise RuntimeError("Error raised: ", err.decode())
    if pattern:
        return re.findall(pattern, out.decode("utf-8"))
    return out.decode("utf-8")


def codegen_tcl(top, configs):
    out_str = """# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#=============================================================================
# run.tcl 
#=============================================================================
# Project name
set hls_prj out.prj

# Open/reset the project
open_project ${hls_prj} -reset

open_solution -reset solution1 -flow_target vivado

"""
    out_str += f'# Top function of the design is "{top}"\n'
    out_str += f"set_top {top}\n"
    out_str += """
# Add design and testbench files
add_files kernel.cpp
add_files -tb host.cpp -cflags "-std=gnu++0x"
open_solution "solution1"
"""
    device = configs["device"]
    frequency = configs["frequency"]
    mode = configs["mode"]
    if device not in PART_NUMBER:
        raise RuntimeError(
            f"Device {device} not supported. Available devices: {list(PART_NUMBER.keys())}"
        )
    out_str += f"\n# Target device is {device}\n"
    out_str += f"set_part {{{PART_NUMBER[device]}}}\n\n"
    out_str += "# Target frequency\n"
    out_str += f"create_clock -period {1000 / frequency:.2f}\n\n"
    out_str += "# Run HLS\n"
    if "csim" in mode or "sw_emu" in mode:
        out_str += "csim_design -O\n"
    if "csyn" in mode or "debug" in mode:
        out_str += "csynth_design\n"
    if "cosim" in mode or "hw_emu" in mode:
        out_str += "cosim_design\n"
    if "impl" in mode or "hw" in mode:
        if device in {"ultra96v2", "pynqz2", "zedboard"}:
            # Embedded boards: export IP only, bitstream happens in Python/Vivado later
            out_str += "export_design -rtl verilog -format ip_catalog\n"
        else:
            # Other platforms: run full impl in HLS
            out_str += "export_design -flow impl\n"
    out_str += "\nexit\n"
    return out_str


def copy_ext_libs(ext_libs, project):
    for ext_lib in ext_libs:
        impl_path = ext_lib.impl
        cpp_file = impl_path.split("/")[-1]
        assert cpp_file != "kernel.cpp", "kernel.cpp is reserved for the top function"
        os.system(f"cp {impl_path} {project}/{cpp_file}")


def store_output(out_arg, value):
    """Write a result back into a caller-supplied output argument, in place.

    Returns True if the value was stored, False if `out_arg` is immutable.

    Why this exists: the read-back sites used to do `out_arg[:] = value`
    unconditionally. That is correct for an ndarray but raises
    `TypeError: 'numpy.int32' object does not support item assignment` when the
    design's output is a SCALAR -- numpy scalars are immutable, so there is no way
    to propagate a value back through one. That turned every scalar-output design
    (empty_full, try_put_try_get, scalar_*) into a harness error even though the
    design itself synthesized and ran fine.

    A 0-d ndarray IS mutable, but only via `arr[...]`, not `arr[:]` -- hence the
    ndim check rather than a bare slice assignment.
    """
    if np.isscalar(out_arg) or isinstance(out_arg, np.generic):
        # Immutable: nothing to write into. Callers that only need to COMPARE
        # (e.g. cosim RTL-vs-golden) are unaffected; callers that need the value
        # should have been passed a 0-d array.
        return False
    if getattr(out_arg, "ndim", None) == 0:
        out_arg[...] = value
        return True
    out_arg[:] = value
    return True


def separate_header(hls_code, top=None, extern_c=True):
    func_decl = False
    sig_str = "#ifndef KERNEL_H\n"
    sig_str += "#define KERNEL_H\n\n"
    args = []
    if extern_c:
        sig_str += 'extern "C" {\n'
    for line in hls_code.split("\n"):
        if line.startswith(f"void {top}"):
            func_decl = True
            sig_str += line + "\n"
        elif func_decl and line.startswith(") {"):
            func_decl = False
            sig_str += ");\n"
            break
        elif func_decl:
            arg_type = line.strip()
            _, var = arg_type.rsplit(" ", 1)
            comma = "," if var[-1] == "," else ""
            ele_type = arg_type.split("[")[0].split(" ")[0].strip()
            allo_type = None
            if ele_type in c2allo_type:
                allo_type = c2allo_type[ele_type]
            else:
                pattern = r"^ap_(u?)int<(\d+)>$"
                match = re.match(pattern, ele_type)
                if not match:
                    raise ValueError(f"Fail to resolve ctype {ele_type}")
                unsigned_flag, width = match.groups()
                allo_type = f"{'u' if unsigned_flag else ''}int{int(width)}"
            shape = tuple(s.split("]")[0] for s in arg_type.split("[")[1:])
            args.append((allo_type, shape))
            if "[" in var:  # array
                var = var.split("[")[0]
                sig_str += "  " + ele_type + " *" + var + f"{comma}\n"
            else:  # scalar
                var = var.split(",")[0]
                sig_str += "  " + ele_type + " " + var + f"{comma}\n"
    if extern_c:
        sig_str += '} // extern "C"\n'
    sig_str += "\n#endif // KERNEL_H\n"
    return sig_str, args



def _run_group_timeout(cmd, timeout, what, **kwargs):
    """subprocess.run(timeout=...) whose timeout actually kills the process TREE.

    subprocess.run kills only the process it spawned. With shell=True, or with a
    tool that forks helpers -- Catapult spawns catapult-pm, CLIP and
    salt_mgls_asy -- the children are reparented to init and keep running
    FOREVER. Measured on this repo: two orphaned Catapult synthesis runs left by
    timed-out cosims, alive 25h and 12h at ~86% CPU each, still holding
    tests/dataflow/test_mlp/cosb. They also skew every later measurement, since a
    synthesis that merely got starved then looks like one that needs a longer
    budget.

    start_new_session puts the child in its own process group, so on timeout the
    whole group can be signalled at once.
    """
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        **kwargs,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        proc.communicate()
        raise RuntimeError(
            f"cosim: {what} timed out after {timeout}s "
            f"(raise ALLO_COSIM_{'SYNTH' if 'synthesis' in what else 'SIM'}_TIMEOUT "
            f"to allow longer)."
        ) from None
    return subprocess.CompletedProcess(cmd, proc.returncode, out, err)


class HLSModule:
    def __init__(
        self,
        mod,
        top_func_name,
        platform="vivado_hls",
        mode=None,
        project=None,
        ext_libs=None,
        configs=None,
        func_args=None,
        wrap_io=True,
    ):
        self.top_func_name = top_func_name
        self.mode = mode
        self.project = project
        self.platform = platform
        self.ext_libs = [] if ext_libs is None else ext_libs
        self.num_output_args = None  # Will be set from configs if provided
        user_configs = configs if configs is not None else {}
        # For Catapult (ASIC), start with ASIC-appropriate defaults instead of FPGA defaults.
        if platform in {"catapult", "systemc"}:
            base_configs = {"device": "nangate-45nm_beh", "frequency": 500}
        else:
            base_configs = DEFAULT_CONFIG.copy()
        base_configs.update(user_configs)
        configs = base_configs
        self.num_output_args = configs.get("num_output_args", None)
        if self.mode is not None:
            configs["mode"] = self.mode
        with Context() as ctx, Location.unknown():
            allo_d.register_dialect(ctx)
            self.module = Module.parse(str(mod), ctx)
            func = find_func_in_module(self.module, top_func_name)
            func.attributes["top"] = UnitAttr.get()
            # Wire/Channel links are SystemC-flow only. Other HLS backends have no
            # emission for them (the base emitter no-ops), which would silently
            # produce wrong output -- so fail loud here instead.
            if platform != "systemc" and (
                "!allo.wire" in str(self.module)
                or "!allo.channel" in str(self.module)
            ):
                raise NotImplementedError(
                    "Wire and Channel links are only supported by the SystemC "
                    f'backend (target="systemc"), not "{platform}". Use '
                    'target="systemc" for wire/channel, or Stream for other '
                    "HLS backends."
                )
            # Stamp per-arg I/O direction (in/out/both/scalar) so the SystemC
            # backend can emit the right port direction — analyze_arg_load_store
            # derives it from actual loads/stores (propagated through calls),
            # unlike `itypes` which only carries the datatype/signedness.
            if platform == "systemc":
                _dir_char = {"in": "i", "out": "o", "both": "b", "scalar": "_"}
                # stamp EVERY func (top + kernels): the emitter needs each memref
                # arg's direction — kernel memref args too, to pick Connections::In
                # vs Out when stream-ifying a boundary array.
                for _fname, _dirs in analyze_arg_load_store(self.module).items():
                    _f = find_func_in_module(self.module, _fname)
                    if _f is not None:
                        _f.attributes["arg_dirs"] = StringAttr.get(
                            "".join(_dir_char.get(d, "_") for d in _dirs)
                        )
            # fix: num_output_args
            if self.num_output_args is None and len(func.type.results) == 0:
                load_store_mapping = analyze_arg_load_store(self.module)
                cnt = 0
                for io_type in load_store_mapping[top_func_name]:
                    if io_type in {"both", "out"}:
                        cnt += 1
                    elif io_type == "in" and cnt > 0 and platform in {"vitis_hls"}:
                        raise RuntimeError("Output arguments must appear at the end.")
                self.num_output_args = cnt

            if platform in {"vitis_hls", "pynq"}:
                assert func_args is not None, "Need to specify func_args"
                if wrap_io:
                    generate_input_output_buffers(
                        self.module,
                        top_func_name,
                        flatten=True,
                        mappings=configs.get("mappings", None),
                    )

            self.module = decompose_library_function(self.module)
            _mlir_lower_pipeline(self.module, lower_linalg=True)
            # Run through lowering passes
            pm = PassManager.parse(
                "builtin.module("
                # used for lowering tensor.empty
                "empty-tensor-to-alloc-tensor,"
                # translate tensor dialect (virtual) to memref dialect (physical)
                # "one-shot-bufferize{bufferize-function-boundaries},"
                # common lowering passes
                "func.func(convert-linalg-to-affine-loops)"
                # DO NOT LOWER AFFINE DIALECT
                ")"
            )
            pm.run(self.module.operation)
        check_pipeline_style_reaches_emitter(self.module, platform)
        buf = io.StringIO()
        success = True
        match platform:
            # [NOTE]: If the HLS backend reports "<func_name> was not declared in this scope", it is likely caused by a forward reference.
            #           MLIR allows calling functions before their definition, but C++ HLS kernels require a prior declaration.
            case "tapa":
                success = allo_d.emit_thls(self.module, buf)
            case "intel_hls":
                success = allo_d.emit_ihls(self.module, buf)
            case "catapult":
                success = allo_d.emit_catapult(self.module, buf)
            case "systemc":
                success = allo_d.emit_systemc(self.module, buf)
            case _:
                # wrap_io=True has already linearized array indexing in
                # generate_input_output_buffers, so we don't need to do it again
                flatten = False if platform == "vivado_hls" else (not wrap_io)
                success = allo_d.emit_vhls(self.module, buf, flatten=flatten)

        if not success:
            raise RuntimeError(
                "Failed to emit HLS code. Check error messages above for details. "
                "Common issues: nested functions with multi-dimensional arrays when wrap_io=False."
            )

        buf.seek(0)
        self.hls_code = buf.read()
        if project is not None:
            assert mode is not None, "mode must be specified when project is specified"
            os.makedirs(project, exist_ok=True)
            path = os.path.dirname(__file__)
            path = os.path.join(path, "../harness/")
            if platform in {"vivado_hls", "vitis_hls", "tapa", "pynq", "catapult", "systemc"}:
                harness_dir = "catapult" if platform == "systemc" else platform.split("_")[0]
                os.system("cp " + path + f"{harness_dir}/* " + project)
                configs["platform"] = platform  # tcl codegen distinguishes systemc
                with open(f"{project}/run.tcl", "w", encoding="utf-8") as outfile:
                    if platform in {"catapult", "systemc"}:
                        outfile.write(codegen_tcl_catapult(top_func_name, configs))
                    else:
                        outfile.write(codegen_tcl(top_func_name, configs))
                # The systemc flow's kernel.cpp is a self-contained sc_main tb.
                # Catapult 2024.2 dropped `solution app` csim, so provide a
                # standalone runner: compile+run with the bundled g++/libsystemc.
                if platform == "systemc" and mode == "csim":
                    csim_sh = (
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
                    with open(f"{project}/csim.sh", "w", encoding="utf-8") as f:
                        f.write(csim_sh)
                    os.chmod(f"{project}/csim.sh", 0o755)
            copy_ext_libs(ext_libs, project)
            if self.platform == "vitis_hls":
                assert self.mode in {
                    "csim",
                    "csyn",
                    "sw_emu",
                    "hw_emu",
                    "hw",
                }, "Invalid mode"
                assert (
                    self.top_func_name != "kernel"
                ), "kernel is a reserved keyword for vitis_hls"
                path = os.path.dirname(__file__)
                path = os.path.join(path, "../harness/")
                dst_path = os.path.join(project, "description.json")
                generate_description_file(
                    self.top_func_name,
                    path + "makefile_gen/description.json",
                    dst_path,
                    frequency=configs["frequency"],
                )
                hbm_mapping = configs.get("hbm_mapping", None)
                generate_makefile(dst_path, project, self.platform, hbm_mapping)
                header, self.args = separate_header(self.hls_code, self.top_func_name)
                with open(f"{project}/kernel.h", "w", encoding="utf-8") as outfile:
                    outfile.write(header)
                # `align_value` promises Vitis the pointer is N-byte aligned,
                # which is what lets `-m_axi_max_widen_bitwidth` actually widen
                # the port. Opt-in: it is a promise the HOST must keep.
                self.hls_code = postprocess_hls_code(
                    self.hls_code,
                    self.top_func_name,
                    align_value=(configs or {}).get("align_value", None),
                )

                # Generate HBM/DDR configuration file if hbm_mapping is provided
                # This must be done AFTER postprocess_hls_code to get correct arg names
                if hbm_mapping is not None:
                    # Extract HLS argument names from the postprocessed code
                    hls_arg_names = extract_hls_arg_names(
                        self.hls_code, self.top_func_name
                    )
                    # Build mapping from user arg names to HLS arg names
                    user_arg_names = []
                    if func_args is not None and self.top_func_name in func_args:
                        for arg in func_args[self.top_func_name]:
                            if hasattr(arg, "name"):
                                user_arg_names.append(arg.name)
                            else:
                                user_arg_names.append(str(arg))
                    # Add return value name - it becomes the last argument
                    # Use the last HLS arg name count to determine if there's a return
                    if len(hls_arg_names) > len(user_arg_names):
                        # There's a return value, add placeholder names
                        for i in range(len(hls_arg_names) - len(user_arg_names)):
                            user_arg_names.append(f"output_{i}")

                    arg_name_mapping = None
                    if len(user_arg_names) == len(hls_arg_names):
                        arg_name_mapping = dict(zip(user_arg_names, hls_arg_names))

                    cfg_content = generate_hbm_config(
                        self.top_func_name, hbm_mapping, arg_name_mapping
                    )
                    cfg_path = os.path.join(project, f"{self.top_func_name}.cfg")
                    with open(cfg_path, "w", encoding="utf-8") as cfg_file:
                        cfg_file.write(cfg_content)
                for lib in self.ext_libs:
                    cpp_file = lib.impl.split("/")[-1]
                    with open(f"{project}/{cpp_file}", "r", encoding="utf-8") as infile:
                        new_code = postprocess_hls_code(
                            infile.read(), lib.top, pragma=False
                        )
                    with open(
                        f"{project}/{cpp_file}", "w", encoding="utf-8"
                    ) as outfile:
                        outfile.write(new_code)
                self.host_code = codegen_host(
                    self.top_func_name,
                    self.module,
                    num_output_args=self.num_output_args,
                )
            elif self.platform in {"catapult", "systemc"}:
                assert self.mode in {
                    "csim",
                    "csyn",
                    "ppa",
                    "cosim",
                }, "Invalid mode for catapult"

                if self.mode == "csim":
                    if self.platform == "systemc":
                        # Option A: the SystemC emitter produces a self-contained
                        # kernel.cpp with its own sc_main testbench, so there is no
                        # separate host.cpp. (Option B will split the testbench out
                        # into a data-file-driven host.cpp via codegen_systemc_host.)
                        self.host_code = ""
                    else:
                        self.host_code = codegen_host_catapult(
                            self.top_func_name,
                            self.module,
                        )
                else:
                    self.host_code = ""

                # For Catapult, we don't have separate kernel.h generation logic yet
                # similar to separate_header. The kernel.cpp contains everything needed
                # or headers are handled differently.
                # If we want to support csim, kernel.cpp usually needs a header
                # referenced by host.cpp.
                # allo/backend/catapult.py's codegen_host includes "kernel.h".
                # So we SHOULD generate kernel.h.
                # Re-using separate_header which is generic enough for C-style headers.
                #
                # However, separate_header currently only understands builtin and
                # ap_(u)int<...> types. When Catapult emits ac_int<...> (e.g., for
                # non-standard integer widths), separate_header can raise ValueError.
                # Fall back to including kernel.cpp directly if that happens.
                try:
                    header, self.args = separate_header(
                        self.hls_code, self.top_func_name
                    )
                except ValueError:
                    header = '#pragma once\n#include "kernel.cpp"\n'
                    self.args = []
                with open(f"{project}/kernel.h", "w", encoding="utf-8") as outfile:
                    outfile.write(header)
            elif self.platform == "tapa":
                assert self.mode in {
                    "csim",
                    "fast_hw_emu",
                    "hw_emu",
                    "hw",
                }, "Invalid mode"
                assert (
                    self.top_func_name != "kernel"
                ), "kernel is a reserved keyword for tapa"
                path = os.path.dirname(__file__)
                path = os.path.join(path, "../harness/")
                dst_path = os.path.join(project, "description.json")
                generate_description_file(
                    self.top_func_name,
                    path + "makefile_gen/description.json",
                    dst_path,
                    frequency=configs["frequency"],
                )
                self.args = []
                hbm_mapping = configs.get("hbm_mapping", None)
                generate_makefile(dst_path, project, self.platform, hbm_mapping)
                # Generate HBM/DDR configuration file if hbm_mapping is provided
                if hbm_mapping is not None:
                    # Extract HLS argument names from the code
                    hls_arg_names = extract_hls_arg_names(
                        self.hls_code, self.top_func_name
                    )
                    # Build mapping from user arg names to HLS arg names
                    user_arg_names = []
                    if func_args is not None and self.top_func_name in func_args:
                        for arg in func_args[self.top_func_name]:
                            if hasattr(arg, "name"):
                                user_arg_names.append(arg.name)
                            else:
                                user_arg_names.append(str(arg))
                    # Add placeholder for return values if needed
                    if len(hls_arg_names) > len(user_arg_names):
                        for i in range(len(hls_arg_names) - len(user_arg_names)):
                            user_arg_names.append(f"output_{i}")

                    arg_name_mapping = None
                    if len(user_arg_names) == len(hls_arg_names):
                        arg_name_mapping = dict(zip(user_arg_names, hls_arg_names))

                    cfg_content = generate_hbm_config(
                        self.top_func_name, hbm_mapping, arg_name_mapping
                    )
                    cfg_path = os.path.join(project, f"{self.top_func_name}.cfg")
                    with open(cfg_path, "w", encoding="utf-8") as cfg_file:
                        cfg_file.write(cfg_content)
                # [NOTE] (Shihan): I guess tapa backend do not use this one. I modified codegen_host for vitis, similar logic should be updated for tapa if self.host_code is useful here
                self.host_code = codegen_host(
                    self.top_func_name,
                    self.module,
                )
                self.tapa_host = codegen_tapa_host(
                    self.top_func_name,
                    self.module,
                    self.hls_code,
                )
                with open(f"{project}/tapa_host.cpp", "w", encoding="utf-8") as outfile:
                    outfile.write(self.tapa_host)
            elif self.platform == "pynq":
                assert self.mode in {"csim", "csyn", "impl"}, "Invalid mode for pynq"
                kernel_h = os.path.join(project, "kernel.h")

                # Generate kernel.h
                header, self.args = separate_header(self.hls_code, self.top_func_name)
                with open(kernel_h, "w", encoding="utf-8") as outfile:
                    outfile.write(header)

                # Apply PYNQ-specific HLS code tweaks and write kernel.cpp
                self.hls_code = postprocess_hls_code_pynq(
                    self.hls_code, self.top_func_name
                )
            else:
                self.host_code = ""
            with open(f"{project}/kernel.cpp", "w", encoding="utf-8") as outfile:
                outfile.write(self.hls_code)
            if hasattr(self, "host_code") and self.host_code:
                with open(f"{project}/host.cpp", "w", encoding="utf-8") as outfile:
                    outfile.write(self.host_code)
            if len(ext_libs) > 0:
                for lib in ext_libs:
                    # Update kernel.cpp
                    new_kernel = ""
                    with open(
                        os.path.join(project, "kernel.cpp"), "r", encoding="utf-8"
                    ) as kernel:
                        for line in kernel:
                            new_kernel += line
                            if "#include <stdint.h>" in line:
                                new_kernel += f'#include "{lib.impl.split("/")[-1]}"\n'
                    with open(
                        os.path.join(project, "kernel.cpp"), "w", encoding="utf-8"
                    ) as kernel:
                        kernel.write(new_kernel)
                    # Update tcl file
                    new_tcl = ""
                    with open(
                        os.path.join(project, "run.tcl"), "r", encoding="utf-8"
                    ) as tcl_file:
                        for line in tcl_file:
                            new_tcl += line
                            if "# Add design and testbench files" in line:
                                cpp_file = lib.impl.split("/")[-1]
                                new_tcl += f"add_files {cpp_file}\n"
                    with open(
                        os.path.join(project, "run.tcl"), "w", encoding="utf-8"
                    ) as tcl_file:
                        tcl_file.write(new_tcl)

    def __repr__(self):
        if self.mode is None:
            return self.hls_code
        return f"HLSModule({self.top_func_name}, {self.mode}, {self.project})"

    def __call__(self, *args, shell=True):
        if self.platform == "vivado_hls":
            assert is_available("vivado_hls"), "vivado_hls is not available"
            ver = run_process("g++ --version", r"\d+\.\d+\.\d+")[0].split(".")
            assert (
                int(ver[0]) * 10 + int(ver[1]) >= 48
            ), f"g++ version too old {ver[0]}.{ver[1]}.{ver[2]}"

            cmd = f"cd {self.project}; make "
            if self.mode == "csim":
                cmd += "csim"
                out = run_process(cmd + " 2>&1")
                runtime = [k for k in out.split("\n") if "seconds" in k][0]
                print(
                    f"[{time.strftime('%H:%M:%S', time.gmtime())}] Simulation runtime {runtime}"
                )

            elif "csyn" in self.mode or self.mode == "custom" or self.mode == "debug":
                cmd += self.platform
                print(
                    f"[{time.strftime('%H:%M:%S', time.gmtime())}] Begin synthesizing project ..."
                )
                if shell:
                    process = subprocess.Popen(cmd, shell=True)
                else:
                    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
                process.wait()
                if process.returncode != 0:
                    raise RuntimeError("Failed to synthesize the design")
                if self.mode != "custom":
                    out = parse_xml(
                        self.project,
                        "Vivado HLS",
                        top=self.top_func_name,
                        print_flag=True,
                    )

            else:
                raise RuntimeError(f"{self.platform} does not support {self.mode} mode")
        elif self.platform == "vitis_hls":
            assert is_available("vitis_hls"), "vitis_hls is not available"
            if self.mode == "csim":
                mod = IPModule(
                    top=self.top_func_name,
                    impl=f"{self.project}/kernel.cpp",
                    include_paths=[self.project],
                    link_hls=True,
                )
                mod(*args)
                return
            if self.mode == "csyn":
                cmd = f"cd {self.project}; vitis_hls -f run.tcl"
                assert len(args) == 0, "csyn mode does not need to pass in arguments"
                print(
                    f"[{time.strftime('%H:%M:%S', time.gmtime())}] Begin synthesizing project ..."
                )
                if shell:
                    process = subprocess.Popen(cmd, shell=True)
                else:
                    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
                process.wait()
                if process.returncode != 0:
                    raise RuntimeError("Failed to synthesize the design")
                return
            # Use Makefile (sw_emu, hw_emu, hw)
            assert "XDEVICE" in os.environ, "Please set XDEVICE in your environment"
            # prepare data
            func = find_func_in_module(self.module, self.top_func_name)
            inputs, outputs = get_func_inputs_outputs(func)
            assert len(args) == len(inputs) + len(
                outputs
            ), f"Number of arguments mismatch, got {len(args)}, expected {len(inputs) + len(outputs)}"
            for i, ((in_dtype, in_shape), arg) in enumerate(zip(inputs, args)):
                assert (len(in_shape) == 0 and np.isscalar(arg)) or np.prod(
                    arg.shape
                ) == np.prod(
                    in_shape
                ), f"invalid arguemnt {i}, {np.asarray(arg).shape}-{in_shape}"
                ele_bitwidth = get_bitwidth_from_type(in_dtype)
                assert (
                    ele_bitwidth == 1 or ele_bitwidth % 8 == 0
                ), "can only handle bytes"
                # store as byte stream
                with open(f"{self.project}/input{i}.data", "wb") as f:
                    if np.isscalar(arg):
                        arg = np.array(arg, dtype=np_supported_types[in_dtype])
                    f.write(arg.tobytes())
            # check if the build folder exists
            bitstream_folder = f"{self.project}/build_dir.{self.mode}.{os.environ['XDEVICE'].rsplit('/')[-1].split('.')[0]}"
            if not os.path.exists(
                os.path.join(bitstream_folder, f"{self.top_func_name}.xclbin")
            ):
                cmd = (
                    f"cd {self.project}; make run TARGET={self.mode} PLATFORM=$XDEVICE"
                )
                print(cmd)
                if shell:
                    process = subprocess.Popen(cmd, shell=True)
                else:
                    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
                process.wait()
                if process.returncode != 0:
                    raise RuntimeError("Failed to build the project")
            else:
                print("Build folder exists, skip building")
                # run the executable
                prefix = f"cd {self.project};"
                if not os.path.exists(f"{self.project}/{self.top_func_name}"):
                    prefix += " make host PLATFORM=$XDEVICE;"
                prefix += (
                    f" XCL_EMULATION_MODE={self.mode}" if self.mode != "hw" else ""
                )
                cmd = f"{prefix} ./{self.top_func_name} ../{bitstream_folder}/{self.top_func_name}.xclbin"
                print(cmd)
                process = subprocess.Popen(cmd, shell=True)
                process.wait()
                if process.returncode != 0:
                    raise RuntimeError("Failed to run the executable")
            # Read output tensors from files
            # Determine how many output files to read
            func = find_func_in_module(self.module, self.top_func_name)
            _, outputs = get_func_inputs_outputs(func)
            if len(outputs) > 0:
                # Original behavior: single output.data file
                if np.isscalar(args[-1]):
                    raise RuntimeError("The output must be a tensor")
                arr = np.fromfile(f"{self.project}/output.data", dtype=args[-1].dtype)
                args[-1][:] = arr.reshape(args[-1].shape)
            else:
                # Multiple output files: output0.data, output1.data, etc.
                num_out = self.num_output_args if self.num_output_args > 0 else 1
                for idx in range(num_out):
                    out_arg_idx = len(inputs) - num_out + idx
                    if out_arg_idx < 0 or out_arg_idx >= len(args):
                        continue
                    out_arg = args[out_arg_idx]
                    if np.isscalar(out_arg):
                        continue
                    arr = np.fromfile(
                        f"{self.project}/output{idx}.data", dtype=out_arg.dtype
                    )
                    out_arg[:] = arr.reshape(out_arg.shape)
            return
        elif self.platform == "pynq":
            # Do not assert PYNQ availability here; the presence of a physical
            # PYNQ device should be checked by callers that need it.
            if self.mode == "csim":
                cwd = os.getcwd()
                mod = IPModule(
                    top=self.top_func_name,
                    impl=f"{cwd}/{self.project}/kernel.cpp",
                    link_hls=True,
                )
                mod(*args)
                return
            if self.mode in {"csyn", "impl"}:
                # HLS synthesis
                cmd = f"cd {self.project}; vitis_hls -f run.tcl"
                print(
                    f"[{time.strftime('%H:%M:%S', time.gmtime())}] Begin synthesizing project ..."
                )
                process = subprocess.Popen(cmd, shell=True)
                process.wait()
                if process.returncode != 0:
                    raise RuntimeError("Failed to synthesize the design")

                if self.mode == "impl":
                    # Produce host (deploy.py)
                    host_code = codegen_pynq_host(
                        self.top_func_name,
                        self.module,
                        self.project,
                    )
                    with open(
                        f"{self.project}/deploy.py", "w", encoding="utf-8"
                    ) as outfile:
                        outfile.write(host_code)

                    # Vivado block design
                    bd_script = "block_design.tcl"
                    bd_script = os.path.basename(bd_script)
                    cmd = f"cd {self.project}; vivado -mode batch -source {bd_script}"
                    print(
                        f"[{time.strftime('%H:%M:%S', time.gmtime())}] Running Vivado Block Design ..."
                    )
                    process = subprocess.Popen(cmd, shell=True)
                    process.wait()
                    if process.returncode != 0:
                        raise RuntimeError(
                            "Failed to create block design / generate bitstream"
                        )

                    # Package .bit / .hwh / deploy.py into deploy/ folder
                    deploy_dir = os.path.join(self.project, "deploy")
                    cmd = (
                        f"mkdir -p {deploy_dir}; "
                        f"cp {self.project}/build_vivado/project_1.runs/impl_1/project_1_bd_wrapper.bit {deploy_dir}/{self.top_func_name}.bit; "
                        f"cp {self.project}/build_vivado/project_1.gen/sources_1/bd/project_1_bd/hw_handoff/project_1_bd.hwh {deploy_dir}/{self.top_func_name}.hwh; "
                        f"cp {self.project}/deploy.py {deploy_dir}/deploy.py"
                    )
                    print(
                        f"[{time.strftime('%H:%M:%S', time.gmtime())}] Collecting files for deployment ..."
                    )
                    print(f"Files for deployment located in {deploy_dir}")
                    process = subprocess.Popen(cmd, shell=True)
                    process.wait()
                    if process.returncode != 0:
                        raise RuntimeError("Failed to collect files")
                return
        elif self.platform == "tapa":
            assert is_available("tapa"), "tapa is not available"
            # Use Makefile (sw_emu, hw_emu, hw)
            assert "XDEVICE" in os.environ, "Please set XDEVICE in your environment"
            # prepare data
            func = find_func_in_module(self.module, self.top_func_name)
            inputs, _ = get_func_inputs_outputs(func)
            for i, ((_, in_shape), arg) in enumerate(zip(inputs, args)):
                write_tensor_to_file(
                    arg,
                    in_shape,
                    f"{self.project}/input{i}.data",
                )
            # check if the build folder exists
            if self.mode in {"csim", "fast_hw_emu"}:
                cmd = f"cd {self.project}; make {self.mode}"
                print(cmd)
                process = subprocess.Popen(cmd, shell=True)
                process.wait()
                if process.returncode != 0:
                    raise RuntimeError("Failed to run tapa executable")
                return
            bitstream_folder = f"{self.project}/build_dir.{self.mode}.{os.environ['XDEVICE'].rsplit('/')[-1].split('.')[0]}"
            if not os.path.exists(
                os.path.join(bitstream_folder, f"{self.top_func_name}.xclbin")
            ):
                cmd = (
                    f"cd {self.project}; make run TARGET={self.mode} PLATFORM=$XDEVICE"
                )
                print(cmd)
                if shell:
                    process = subprocess.Popen(cmd, shell=True)
                else:
                    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
                process.wait()
                if process.returncode != 0:
                    raise RuntimeError("Failed to build the project")
            else:
                print("Build folder exists, skip building")
                # run the executable
                prefix = f"cd {self.project};"
                if not os.path.exists(f"{self.project}/{self.top_func_name}"):
                    prefix += " make host PLATFORM=$XDEVICE;"
                prefix += (
                    f" XCL_EMULATION_MODE={self.mode}" if self.mode != "hw" else ""
                )
                cmd = f"{prefix} ./{self.top_func_name} ../{bitstream_folder}/{self.top_func_name}.xclbin"
                print(cmd)
                process = subprocess.Popen(cmd, shell=True)
                process.wait()
                if process.returncode != 0:
                    raise RuntimeError("Failed to run the executable")
            # suppose the last argument is the output tensor
            result = read_tensor_from_file(
                inputs[-1][0], args[-1].shape, f"{self.project}/output.data"
            )
            args[-1][:] = result
            return
        if self.platform in {"catapult", "systemc"}:
            if self.mode == "csim":
                # Check for input arguments
                func = find_func_in_module(self.module, self.top_func_name)
                inputs, outputs = get_func_inputs_outputs(func)
                assert len(args) == len(inputs) + len(
                    outputs
                ), f"Number of arguments mismatch, got {len(args)}, expected {len(inputs) + len(outputs)}"

                # Generate kernel.h
                # self.args might be updated by separate_header if needed, but for csim we use passed args
                header, _ = separate_header(
                    self.hls_code, self.top_func_name, extern_c=False
                )
                with open(
                    os.path.join(self.project, "kernel.h"), "w", encoding="utf-8"
                ) as outfile:
                    outfile.write(header)

                # Write input data. A systemc region is a void function whose args
                # are all "inputs" by signature, so split by actual direction
                # (arg_dirs): 'in' -> input<k>.data (read by the emitted testbench),
                # 'out' -> filled from output<k>.data after the run.
                if self.platform == "systemc":
                    dirs = analyze_arg_load_store(self.module)[self.top_func_name]
                    _ii = 0
                    for (in_dtype, in_shape), arg, d in zip(inputs, args, dirs):
                        if d in ("in", "both"):  # 'both' arrays are preloaded too
                            write_tensor_to_file(
                                arg, in_shape, f"{self.project}/input{_ii}.data"
                            )
                            _ii += 1
                else:
                    for i, ((in_dtype, in_shape), arg) in enumerate(
                        zip(inputs, args[: len(inputs)])
                    ):
                        write_tensor_to_file(arg, in_shape, f"{self.project}/input{i}.data")

                # Compilation with g++
                # Assuming 'g++' is in PATH.
                # Include path for ac_types
                # Resolve MGC_HOME via the same logic as synthesis mode
                _find_catapult_binary()  # sets MGC_HOME as side-effect if found
                mgc_home = os.environ.get("MGC_HOME", "")
                if not mgc_home:
                    raise RuntimeError(
                        "Catapult not found. Set MGC_HOME or install to /opt/siemens/catapult/."
                    )

                ac_include = os.path.join(mgc_home, "shared/include")
                if not os.path.isdir(ac_include):
                    raise RuntimeError(
                        f"Catapult headers not found at {ac_include}. Check MGC_HOME."
                    )

                if self.platform == "systemc":
                    systemc_home = os.environ.get("SYSTEMC_HOME", "")
                    if not systemc_home:
                        raise RuntimeError("Set SYSTEMC_HOME for systemc csim.")
                    lib_dir = os.path.join(systemc_home, "lib-linux64")
                    if not os.path.isdir(lib_dir):
                        lib_dir = os.path.join(systemc_home, "lib")
                    # Connections/matchlib ship under $MGC_HOME/shared/include alongside ac_types
                    # Option A: kernel.cpp is self-contained (its own sc_main); no host.cpp.
                    # ALLO_CXX_EXTRA: host-specific extra flags (e.g. a newer libstdc++
                    # dir: libsystemc needs GLIBCXX_3.4.26, absent from zhang-21's system
                    # libstdc++ -> `-L<conda>/lib -Wl,-rpath,<conda>/lib`).
                    cxx_extra = os.environ.get("ALLO_CXX_EXTRA", "")
                    cmd = (
                        f"cd {self.project}; g++ -std=c++17 "
                        f"-I{ac_include} -I{systemc_home}/include "
                        f"{cxx_extra} kernel.cpp "
                        f"-L{lib_dir} -Wl,-rpath,{lib_dir} -lsystemc "
                        f"-o sim"
                    )
                else:
                    cmd = f"cd {self.project}; g++ -std=c++11 -I{ac_include} kernel.cpp host.cpp -o sim"
                print(
                    f"[{time.strftime('%H:%M:%S', time.gmtime())}] Compiling with g++ ..."
                )
                if shell:
                    process = subprocess.Popen(cmd, shell=True)
                else:
                    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
                process.wait()
                if process.returncode != 0:
                    raise RuntimeError(
                        "Failed to compile with g++. Check if g++ is installed and ac_types headers are correct."
                    )

                # Execution
                cmd = f"cd {self.project}; ./sim"
                print(
                    f"[{time.strftime('%H:%M:%S', time.gmtime())}] Running simulation ..."
                )
                if shell:
                    process = subprocess.Popen(cmd, shell=True)
                else:
                    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
                process.wait()
                if process.returncode != 0:
                    raise RuntimeError("Simulation failed.")

                # Read outputs
                if self.platform == "systemc":
                    # Option B: the emitted testbench wrote each 'out' arg to
                    # output<k>.data; read them back into the numpy args (same
                    # arg_dirs split as the inputs above; `dirs` is in scope).
                    _oo = 0
                    for (out_dtype, out_shape), out_arg, d in zip(inputs, args, dirs):
                        if d in ("out", "both"):  # 'both' arrays are read back too
                            fpath = f"{self.project}/output{_oo}.data"
                            if not os.path.exists(fpath):
                                raise RuntimeError(
                                    f"Output file {fpath} not found. Simulation might have failed."
                                )
                            store_output(
                                out_arg,
                                read_tensor_from_file(out_dtype, out_shape, fpath),
                            )
                            _oo += 1
                    return
                for i, ((out_dtype, out_shape), out_arg) in enumerate(
                    zip(outputs, args[len(inputs) :])
                ):
                    if not os.path.exists(f"{self.project}/output{i}.data"):
                        raise RuntimeError(
                            f"Output file output{i}.data not found. Simulation might have failed."
                        )
                    result = read_tensor_from_file(
                        out_dtype, out_shape, f"{self.project}/output{i}.data"
                    )
                    store_output(out_arg, result)
                return

            if self.mode == "cosim":
                # Bit-exact RTL cosim: run the emitted SystemC testbench once in
                # software (the golden, in $project) and once against Catapult-
                # synthesized RTL via SCVerify/Xcelium (in the build subdir cosb), then
                # diff the two. Only the systemc flow emits a self-contained sc_main
                # testbench + SCVerify-wrappable DUT, so cosim is systemc-only. The
                # golden's output<k>.data (in $project) is renamed golden_output<k>.data
                # so it does not collide with the RTL run's output<k>.data (in cosb).
                # Catapult MUST synthesize from a subdir, not $project -- see step 2.
                # See notes: catapult-connections-cwd-quirk, systemc-nonblocking-no-wait.
                import glob as _glob
                import shutil as _shutil

                if self.platform != "systemc":
                    raise NotImplementedError(
                        "cosim mode is only supported by the SystemC backend "
                        '(target="systemc").'
                    )

                func = find_func_in_module(self.module, self.top_func_name)
                inputs, _outputs = get_func_inputs_outputs(func)
                assert len(args) == len(inputs) + len(_outputs), (
                    f"Number of arguments mismatch, got {len(args)}, "
                    f"expected {len(inputs) + len(_outputs)}"
                )

                # kernel.h next to the self-contained kernel.cpp (mirror csim).
                header, _ = separate_header(
                    self.hls_code, self.top_func_name, extern_c=False
                )
                with open(
                    os.path.join(self.project, "kernel.h"), "w", encoding="utf-8"
                ) as outfile:
                    outfile.write(header)

                # A systemc region is a void function whose args are all "inputs" by
                # signature; split by actual direction (arg_dirs) as csim does.
                dirs = analyze_arg_load_store(self.module)[self.top_func_name]
                _ii = 0
                for (in_dtype, in_shape), arg, d in zip(inputs, args, dirs):
                    if d in ("in", "both"):
                        write_tensor_to_file(
                            arg, in_shape, f"{self.project}/input{_ii}.data"
                        )
                        _ii += 1

                # ---- 1. golden: compile+run the SystemC testbench in software ----
                _find_catapult_binary()  # resolves MGC_HOME as a side effect
                mgc_home = os.environ.get("MGC_HOME", "")
                if not mgc_home:
                    raise RuntimeError(
                        "Catapult not found. Set MGC_HOME or install to "
                        "/opt/siemens/catapult/."
                    )
                ac_include = os.path.join(mgc_home, "shared/include")
                systemc_home = os.environ.get("SYSTEMC_HOME", "")
                if not systemc_home:
                    raise RuntimeError("Set SYSTEMC_HOME for systemc cosim.")
                lib_dir = os.path.join(systemc_home, "lib-linux64")
                if not os.path.isdir(lib_dir):
                    lib_dir = os.path.join(systemc_home, "lib")
                cxx_extra = os.environ.get("ALLO_CXX_EXTRA", "")
                gcmd = (
                    f"cd {self.project}; g++ -std=c++17 "
                    f"-I{ac_include} -I{systemc_home}/include "
                    f"{cxx_extra} kernel.cpp "
                    f"-L{lib_dir} -Wl,-rpath,{lib_dir} -lsystemc -o sim"
                )
                print(
                    f"[{time.strftime('%H:%M:%S', time.gmtime())}] "
                    "cosim: building software golden ..."
                )
                if subprocess.Popen(gcmd, shell=True).wait() != 0:
                    raise RuntimeError("cosim: golden g++ compile failed.")
                if subprocess.Popen(f"cd {self.project}; ./sim", shell=True).wait() != 0:
                    raise RuntimeError("cosim: golden simulation failed.")
                # Set the golden aside; the RTL run overwrites output<k>.data.
                for f in _glob.glob(f"{self.project}/output*.data"):
                    _shutil.move(
                        f, os.path.join(self.project, "golden_" + os.path.basename(f))
                    )

                # ---- 2. synthesize with SCVerify (run.tcl already requires it) ----
                # IMPORTANT: run Catapult in a BUILD SUBDIR, not in self.project where
                # kernel.cpp lives. Empirically, when Catapult's cwd is the source dir
                # the Connections In/Out ports degrade to raw sc_signals (CIN-124 on
                # in.rdy) -> iomode=fixed -> SCHD-30 in the AlloFifo partition; running
                # from a separate build dir (run.tcl adds "$sfd/kernel.cpp", so sources
                # stay in the parent) keeps the handshake intact and it schedules.
                syn = os.path.join(self.project, "cosb")
                os.makedirs(syn, exist_ok=True)
                catapult_cmd = _find_catapult_binary()
                print(
                    f"[{time.strftime('%H:%M:%S', time.gmtime())}] "
                    "cosim: synthesizing RTL (Catapult + SCVerify) ..."
                )
                # 900s was too small for real designs: test_multi_cache_gemm times
                # out at 900s on an OTHERWISE IDLE machine (914s, and 914s again under
                # load -- contention was not the cause) yet PASSES in 1825s end-to-end.
                # A legitimate design failing is worse than a stuck one taking longer to
                # report, and since the timeout now kills the whole process group a long
                # budget no longer leaves anything running behind it.
                synth_to = int(os.environ.get("ALLO_COSIM_SYNTH_TIMEOUT", "2400"))
                r = _run_group_timeout(
                    f"cd {syn}; {catapult_cmd} -shell -f {self.project}/run.tcl",
                    synth_to,
                    "Catapult synthesis",
                    shell=True,
                    text=True,
                )
                with open(f"{self.project}/synth.log", "w", encoding="utf-8") as lf:
                    lf.write((r.stdout or "") + (r.stderr or ""))
                if r.returncode != 0:
                    raise RuntimeError(
                        "cosim: Catapult synthesis failed (see "
                        f"{self.project}/synth.log)."
                    )

                # ---- 3. patch the SCVerify SC shim ----
                # sysc_sim.cpp is a SEPARATE translation unit from kernel.cpp: it sees
                # only what sysc_sim.h pulls in, and it is included BEFORE Connections'
                # marshaller.h. Both AC headers have to be visible by then:
                #   * ac_int.h        -- ac_int-typed ports
                #   * ac_std_float.h  -- float ports
                #
                # It is the ORDER that matters, not mere presence. Catapult already
                # emits `#include <ac_std_float.h>` into this header -- but AFTER
                # mc_connections.h, which is too late for the marshaller to pick up a
                # specialisation for the type. The RTL-cosim COMPILE then dies with
                # `class "ac_ieee_float<binary32>" has no member "Marshall"`
                # (marshaller.h:208). Injecting the same header BEFORE mc_connections.h
                # fixes it. Invisible to csim, which never builds this TU, so an int32
                # memory design cosims fine while a float one does not.
                #
                # MEASURED three ways: ac_int.h-only reproduces the Marshall error;
                # adding ac_std_float.h early fixes it; and a guard keyed on the string
                # "ac_std_float.h" silently matches Catapult's own late include and
                # skips the patch -- hence the distinct sentinel below.
                _SC_SIM_INCLUDES = (
                    "\n// ALLO_SC_SIM_INCLUDES: must precede mc_connections.h"
                    "\n#include <ac_int.h>\n#include <ac_std_float.h>"
                )
                for sh in _glob.glob(f"{syn}/**/sysc_sim.h", recursive=True):
                    s = open(sh, encoding="utf-8").read()
                    if "ALLO_SC_SIM_INCLUDES" not in s:
                        with open(sh, "w", encoding="utf-8") as f:
                            f.write(
                                s.replace(
                                    "#include <systemc.h>",
                                    "#include <systemc.h>" + _SC_SIM_INCLUDES,
                                )
                            )

                # ---- 4. run the RTL cosim (Xcelium/ncsim) ----
                # SCVerify runs the sim with cwd = the build dir (syn), so stage the
                # inputs there and read the RTL outputs back from there.
                for f in _glob.glob(f"{self.project}/input*.data"):
                    _shutil.copy(f, os.path.join(syn, os.path.basename(f)))
                mk = _glob.glob(
                    f"{syn}/**/Verify_concat_sim_rtl_v_ncsim.mk",
                    recursive=True,
                )
                if not mk:
                    raise RuntimeError(
                        "cosim: SCVerify did not emit the ncsim makefile "
                        "(synthesis may have stopped before `go extract`; see "
                        f"{self.project}/synth.log)."
                    )
                v1 = os.path.dirname(os.path.dirname(mk[0]))
                nc_root = os.environ.get("NC_ROOT", "/opt/cadence/XCELIUM2403")
                if not os.path.isdir(nc_root):
                    raise RuntimeError(
                        f"cosim: Xcelium not found at {nc_root}. Set NC_ROOT."
                    )
                env = dict(os.environ)
                env["NC_ROOT"] = nc_root
                env["NCSim_NC_ROOT"] = nc_root
                print(
                    f"[{time.strftime('%H:%M:%S', time.gmtime())}] "
                    "cosim: running RTL simulation (Xcelium) ..."
                )
                cosim_to = int(os.environ.get("ALLO_COSIM_SIM_TIMEOUT", "900"))
                # make -> ncsim forks a tree of its own; same group-kill treatment.
                r2 = _run_group_timeout(
                    [
                        f"{mgc_home}/bin/make",
                        "-f",
                        "./scverify/Verify_concat_sim_rtl_v_ncsim.mk",
                        f"NC_ROOT={nc_root}",
                        f"NCSim_NC_ROOT={nc_root}",
                        "SIMTOOL=ncsim",
                        "sim",
                    ],
                    cosim_to,
                    "RTL simulation",
                    cwd=v1,
                    text=True,
                    env=env,
                )
                with open(f"{self.project}/cosim.log", "w", encoding="utf-8") as lf:
                    lf.write((r2.stdout or "") + (r2.stderr or ""))

                # ---- 5. compare RTL output vs golden, populate output args ----
                _oo = 0
                mismatches = []
                for (out_dtype, out_shape), out_arg, d in zip(inputs, args, dirs):
                    if d not in ("out", "both"):
                        continue
                    rtl_f = f"{syn}/output{_oo}.data"
                    gold_f = f"{self.project}/golden_output{_oo}.data"
                    if not os.path.exists(rtl_f):
                        raise RuntimeError(
                            f"cosim: RTL produced no output{_oo}.data (see "
                            f"{self.project}/cosim.log)."
                        )
                    rtl = read_tensor_from_file(out_dtype, out_shape, rtl_f)
                    store_output(out_arg, rtl)
                    if os.path.exists(gold_f):
                        gold = read_tensor_from_file(out_dtype, out_shape, gold_f)
                        if not np.array_equal(rtl, gold):
                            mismatches.append(_oo)
                    _oo += 1

                stamp = time.strftime("%H:%M:%S", time.gmtime())
                if mismatches:
                    raise RuntimeError(
                        f"[{stamp}] cosim MISMATCH: RTL != software golden on output "
                        f"index(es) {mismatches} (project {self.project})."
                    )
                print(
                    f"[{stamp}] cosim MATCH: RTL is bit-exact with the software "
                    f"golden ({_oo} output array(s))."
                )
                return

            if self.mode in {"csyn", "ppa"}:
                catapult_cmd = _find_catapult_binary()

                # For systemc, synthesize from a BUILD SUBDIR, not self.project where
                # kernel.cpp lives: when Catapult's cwd contains the source, matchlib
                # Connections In/Out ports degrade to raw sc_signals (CIN-124 on in.rdy)
                # -> iomode=fixed -> SCHD-30. Sources stay in the parent via
                # run.tcl's "$sfd/kernel.cpp". Reports then live under the subdir.
                if self.platform == "systemc":
                    rpt_dir = os.path.join(self.project, "build")
                    os.makedirs(rpt_dir, exist_ok=True)
                    cmd = f"cd {rpt_dir}; {catapult_cmd} -shell -f {self.project}/run.tcl"
                else:
                    rpt_dir = self.project
                    cmd = f"cd {self.project}; {catapult_cmd} -shell -f run.tcl"
                # Synthesis EXECUTES NOTHING, so output arrays are never written. Passing
                # them is therefore always a caller mistake, and it must fail HERE: this
                # was briefly downgraded to a warning so one call site could serve every
                # mode, and the result was that a test ran on to compare its untouched
                # (all-zero) output against a golden and reported a 100% mismatch -- a
                # confusing failure 20 lines from the real cause. Fail fast instead.
                assert len(args) == 0, (
                    f"{self.mode} mode synthesizes only and runs nothing, so it takes no "
                    f"arguments (got {len(args)}). Output arrays would stay unwritten. "
                    f'Call mod() with no arguments, and use mode="csim" or "cosim" if you '
                    f"want results back."
                )
                print(
                    f"[{time.strftime('%H:%M:%S', time.gmtime())}] Begin synthesizing project with Catapult HLS ({self.mode} mode)..."
                )
                if shell:
                    process = subprocess.Popen(cmd, shell=True)
                else:
                    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
                process.wait()
                if process.returncode != 0:
                    raise RuntimeError(
                        f"Failed to synthesize the design with Catapult HLS in {self.mode} mode"
                    )
                print(
                    f"[{time.strftime('%H:%M:%S', time.gmtime())}] Catapult HLS synthesis completed successfully"
                )

                if self.mode == "ppa":
                    print(
                        f"[{time.strftime('%H:%M:%S', time.gmtime())}] Extracting PPA metrics..."
                    )
                    stats = parse_catapult_report(rpt_dir, self.top_func_name)
                    print("| Metric              | Value                |")
                    print("|---------------------|----------------------|")
                    for k, v in stats.items():
                        print(f"| {k:<20} | {v:<20} |")

                    # Hierarchical breakdown: per-PE and interconnect
                    hier = parse_catapult_hierarchical_report(
                        rpt_dir, self.top_func_name
                    )
                    print()
                    print(hier["summary"])
                    stats["hierarchical"] = hier
                    return stats
                return
            raise RuntimeError(
                "Catapult backend currently only supports 'csim', 'csyn', 'cosim', "
                f"and 'ppa' mode, got '{self.mode}'"
            )
        raise RuntimeError("Not implemented")
