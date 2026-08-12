#!/usr/bin/env python3
"""Generate backend-specific, workload-driven simulation collateral."""

import ast
import json
import os
import re
import shutil
from pathlib import Path


INPUTS = Path("inputs")
OUTPUTS = Path("outputs")
TEMPLATES = Path("templates")


def _replace_template(path, replacements):
    text = path.read_text()
    for marker, value in replacements.items():
        text = text.replace(f"@{marker}@", value)
    leftovers = sorted(set(re.findall(r"@[A-Z0-9_]+@", text)))
    if leftovers:
        raise RuntimeError(f"unexpanded template markers in {path}: {leftovers}")
    return text


def _parse_cpp_arguments(kernel_cpp, top):
    match = re.search(rf"\bvoid\s+{re.escape(top)}\s*\((.*?)\)\s*\{{", kernel_cpp, re.S)
    if not match:
        raise RuntimeError(f"cannot find generated C++ top function {top}")
    arguments = []
    for declaration in match.group(1).split(","):
        declaration = declaration.strip()
        if not declaration:
            continue
        names = re.findall(r"[A-Za-z_]\w*", declaration)
        if not names:
            raise RuntimeError(f"cannot parse C++ argument declaration: {declaration}")
        arguments.append(names[-1])
    return arguments


def _parse_m_axi_pragmas(kernel_cpp):
    result = {}
    pattern = re.compile(
        r"#pragma\s+HLS\s+interface\s+m_axi\s+port=(\w+)"
        r"[^\n]*\bbundle=(\w+)"
    )
    for argument, bundle in pattern.findall(kernel_cpp):
        if argument in result:
            raise RuntimeError(f"duplicate m_axi pragma for {argument}")
        result[argument] = bundle
    return result


def _constant_expression(expression, parameters):
    """Evaluate the limited integer arithmetic used in Vitis width parameters."""
    tree = ast.parse(expression, mode="eval")

    def evaluate(node):
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Num) and isinstance(node.n, int):
            return node.n
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.Name) and node.id in parameters:
            return parameters[node.id]
        if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Div)
        ):
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if right == 0 or left % right:
                raise ValueError("non-integral parameter division")
            return left // right
        raise ValueError(f"unsupported constant expression: {expression}")

    return int(evaluate(tree))


def _parse_parameters(verilog):
    params = {}
    declarations = re.findall(r"parameter\s+(\w+)\s*=\s*([^;]+);", verilog)
    unresolved = list(declarations)
    while unresolved:
        next_unresolved = []
        progress = False
        for name, expression in unresolved:
            try:
                params[name] = _constant_expression(expression.strip(), params)
                progress = True
            except (SyntaxError, ValueError):
                next_unresolved.append((name, expression))
        if not progress:
            break
        unresolved = next_unresolved
    return params


def _range_width(bit_range, parameters):
    if bit_range is None:
        return 1
    expression = bit_range.strip()
    numeric = re.fullmatch(r"(\d+)\s*:\s*(\d+)", expression)
    if numeric:
        return abs(int(numeric.group(1)) - int(numeric.group(2))) + 1
    descending = re.fullmatch(r"(.+?)\s*:\s*(.+)", expression)
    if descending:
        try:
            high = _constant_expression(descending.group(1), parameters)
            low = _constant_expression(descending.group(2), parameters)
            return abs(high - low) + 1
        except (SyntaxError, ValueError):
            pass
    raise RuntimeError(f"unsupported Verilog port range [{expression}]")


def _parse_ports(verilog):
    parameters = _parse_parameters(verilog)
    ports = {}
    pattern = re.compile(
        r"^\s*(input|output)\s+(?:(?:wire|reg)\s+)?"
        r"(?:\[([^\]]+)\]\s+)?(\w+)\s*;",
        re.M,
    )
    for direction, bit_range, name in pattern.findall(verilog):
        ports[name] = {
            "direction": direction,
            "width": _range_width(bit_range or None, parameters),
        }
    if not ports:
        raise RuntimeError("no top-level RTL ports found")
    return ports, parameters


def _parse_pointer_registers(control_verilog, hls_arguments):
    addresses = {}
    for argument in hls_arguments:
        upper = argument.upper()
        halves = {}
        for half in (0, 1):
            pattern = rf"ADDR_{re.escape(upper)}_DATA_{half}\s*=\s*\d+'h([0-9a-fA-F]+)"
            match = re.search(pattern, control_verilog)
            if not match:
                raise RuntimeError(f"missing AXI-Lite pointer register for {argument} half {half}")
            halves[f"data_{half}"] = int(match.group(1), 16)
        addresses[argument] = halves
    return addresses


def _logic_declaration(name, width):
    return f"  logic {name};" if width == 1 else f"  logic [{width - 1}:0] {name};"


def _packing_ratio(element_width, data_width):
    """Return the number of semantic elements carried by one AXI beat."""
    if element_width <= 0 or data_width <= 0:
        raise RuntimeError(
            f"AXI and workload element widths must be positive: "
            f"element={element_width}, AXI={data_width}"
        )
    if element_width > data_width or data_width % element_width:
        raise RuntimeError(
            f"Vitis generator requires an integral AXI packing ratio: "
            f"element={element_width} bits, AXI={data_width} bits"
        )
    return data_width // element_width


def _simulation_path(relative_path):
    """Return the path used after collateral enters a simulation node."""
    return (Path("inputs") / relative_path).as_posix()


AXI_SUFFIXES = [
    "AWVALID", "AWREADY", "AWADDR", "AWID", "AWLEN", "AWSIZE", "AWBURST",
    "AWLOCK", "AWCACHE", "AWPROT", "AWQOS", "AWREGION", "AWUSER", "WVALID",
    "WREADY", "WDATA", "WSTRB", "WLAST", "WID", "WUSER", "ARVALID", "ARREADY",
    "ARADDR", "ARID", "ARLEN", "ARSIZE", "ARBURST", "ARLOCK", "ARCACHE",
    "ARPROT", "ARQOS", "ARREGION", "ARUSER", "RVALID", "RREADY", "RDATA",
    "RLAST", "RID", "RUSER", "RRESP", "BVALID", "BREADY", "BRESP", "BID",
    "BUSER",
]


AXIL_SUFFIXES = [
    "AWVALID", "AWREADY", "AWADDR", "WVALID", "WREADY", "WDATA", "WSTRB",
    "ARVALID", "ARREADY", "ARADDR", "RVALID", "RREADY", "RDATA", "RRESP",
    "BVALID", "BREADY", "BRESP",
]


def _bundle_ports(ports, prefix, suffixes):
    names = []
    for suffix in suffixes:
        name = f"{prefix}_{suffix}"
        if name in ports:
            names.append((suffix, name, ports[name]))
    return names


def _generate_vitis(workload, metadata):
    top = workload["top_function"]
    kernel_path = INPUTS / "allo-build" / "kernel.cpp"
    if not kernel_path.is_file():
        raise RuntimeError(f"missing Vitis kernel source: {kernel_path}")
    kernel_cpp = kernel_path.read_text()
    hls_arguments = _parse_cpp_arguments(kernel_cpp, top)
    semantic_arguments = workload["call_signature"]
    if len(hls_arguments) != len(semantic_arguments):
        raise RuntimeError(
            f"workload has {len(semantic_arguments)} arguments but generated C++ has "
            f"{len(hls_arguments)}"
        )
    pragmas = _parse_m_axi_pragmas(kernel_cpp)
    missing = [name for name in hls_arguments if name not in pragmas]
    if missing:
        raise RuntimeError(f"Vitis arguments lack expected Allo m_axi pragmas: {missing}")

    rtl_dir = INPUTS / "backend-rtl"
    rtl_path = rtl_dir / f"{top}.v"
    control_path = rtl_dir / f"{top}_control_s_axi.v"
    if not rtl_path.is_file() or not control_path.is_file():
        raise RuntimeError(f"missing Vitis top/control RTL for {top}")
    rtl_text = rtl_path.read_text()
    ports, parameters = _parse_ports(rtl_text)
    pointer_registers = _parse_pointer_registers(control_path.read_text(), hls_arguments)

    required_control = ["ap_clk", "ap_rst_n", "ap_start", "ap_done"]
    missing_control = [name for name in required_control if name not in ports]
    if missing_control:
        raise RuntimeError(f"missing Vitis control ports: {missing_control}")

    mappings = []
    for semantic, hls_name in zip(semantic_arguments, hls_arguments):
        bundle = pragmas[hls_name]
        prefix = f"m_axi_{bundle}"
        bundle_ports = _bundle_ports(ports, prefix, AXI_SUFFIXES)
        if not bundle_ports:
            raise RuntimeError(f"no RTL ports found for Vitis bundle {bundle}")
        first_call = workload["calls"][0]["arguments"][semantic]
        data_width = ports[f"{prefix}_WDATA"]["width"]
        element_width = first_call["element_bits"]
        packing_ratio = _packing_ratio(element_width, data_width)
        mappings.append(
            {
                "semantic_name": semantic,
                "hls_argument": hls_name,
                "bundle": bundle,
                "rtl_prefix": prefix,
                "data_width": data_width,
                "element_width": element_width,
                "packing_ratio": packing_ratio,
                "address_width": ports[f"{prefix}_AWADDR"]["width"],
                "id_width": ports[f"{prefix}_AWID"]["width"],
                "registers": pointer_registers[hls_name],
            }
        )

    declarations = "\n".join(
        _logic_declaration(name, info["width"]) for name, info in ports.items()
    )
    connections = ",\n".join(f"    .{name}({name})" for name in ports)

    bfm_instances = []
    for mapping in mappings:
        prefix = mapping["rtl_prefix"]
        port_map = _bundle_ports(ports, prefix, AXI_SUFFIXES)
        connections_bfm = ",\n".join(
            f"    .{suffix.lower()}({name})" for suffix, name, _ in port_map
        )
        bfm_instances.append(
            f"  vitis_axi_memory_bfm #(\n"
            f"    .ADDR_WIDTH({mapping['address_width']}),\n"
            f"    .DATA_WIDTH({mapping['data_width']}),\n"
            f"    .ELEMENT_WIDTH({mapping['element_width']}),\n"
            f"    .ID_WIDTH({mapping['id_width']}),\n"
            f"    .BASE_ADDR(64'h0000_0000_0000_0000)\n"
            f"  ) {mapping['bundle']}_bfm (\n"
            f"    .clk(ap_clk),\n"
            f"    .reset_n(ap_rst_n),\n{connections_bfm}\n  );"
        )

    axil_ports = _bundle_ports(ports, "s_axi_control", AXIL_SUFFIXES)
    if not axil_ports:
        raise RuntimeError("Vitis top has no s_axi_control ports")
    axil_connections = ",\n".join(
        f"    .{suffix.lower()}({name})" for suffix, name, _ in axil_ports
    )
    bfm_instances.append(
        "  vitis_axilite_master_bfm #(\n"
        f"    .ADDR_WIDTH({ports['s_axi_control_AWADDR']['width']}),\n"
        f"    .DATA_WIDTH({ports['s_axi_control_WDATA']['width']})\n"
        "  ) control_bfm (\n"
        "    .clk(ap_clk),\n"
        "    .reset_n(ap_rst_n),\n"
        f"{axil_connections}\n  );"
    )

    sequence = []
    for call_index, call in enumerate(workload["calls"]):
        sequence.append(f'    $display("Starting workload call {call_index}: {call["name"]}");')
        if call["reset_before"]:
            sequence.extend([
                "    @(negedge ap_clk);",
                "    #(allo_bagl_input_delay_ns);",
                "    ap_rst_n = 1'b0;",
                "    repeat (allo_bagl_num_reset_cycles) @(negedge ap_clk);",
                "    #(allo_bagl_input_delay_ns);",
                "    ap_rst_n = 1'b1;",
                "    repeat (2) @(posedge ap_clk);",
            ])
        for mapping in mappings:
            semantic = mapping["semantic_name"]
            argument = call["arguments"][semantic]
            vector = _simulation_path(argument["file"])
            sequence.append(
                f'    {mapping["bundle"]}_bfm.load_hex('
                f'"{vector}", {argument["element_count"]});'
            )
            registers = mapping["registers"]
            sequence.append(
                f"    control_bfm.write(32'h{registers['data_0']:08x}, 32'h00000000);"
            )
            sequence.append(
                f"    control_bfm.write(32'h{registers['data_1']:08x}, 32'h00000000);"
            )
        sequence.extend([
            "    @(negedge ap_clk);",
            "    #(allo_bagl_input_delay_ns);",
            "    ap_start = 1'b1;",
            "    @(negedge ap_clk);",
            "    #(allo_bagl_input_delay_ns);",
            "    ap_start = 1'b0;",
            f"    wait_for_done({workload['default_timeout_cycles']});",
        ])
        for semantic, expected in call["expected"].items():
            mapping = next(item for item in mappings if item["semantic_name"] == semantic)
            sequence.append(
                f'    {mapping["bundle"]}_bfm.check_hex('
                f'"{_simulation_path(expected["file"])}", '
                f"{expected['element_count']});"
            )

    clock_half = float(metadata["clock_period_ns"]) / 2.0
    testbench = _replace_template(
        TEMPLATES / "vitis" / "testbench.sv.tpl",
        {
            "CLOCK_HALF_PERIOD": f"{clock_half:g}",
            "SIGNAL_DECLARATIONS": declarations,
            "TOP_MODULE": top,
            "DUT_CONNECTIONS": connections,
            "BFM_INSTANTIATIONS": "\n\n".join(bfm_instances),
            "WORKLOAD_SEQUENCE": "\n".join(sequence),
        },
    )
    (OUTPUTS / "testbench.sv").write_text(testbench)
    contract = {
        "schema_version": 1,
        "backend": "vitis",
        "top_module": top,
        "clock_period_ns": float(metadata["clock_period_ns"]),
        "arguments": mappings,
        "calls": workload["calls"],
    }

    # Backend-owned collateral stays inside the backend generator. A future
    # backend can select entirely different templates and simulator inputs
    # without adding backend conditionals to main().
    vitis_templates = TEMPLATES / "vitis"
    shutil.copy(vitis_templates / "vitis-axi-memory-bfm.sv", OUTPUTS)
    shutil.copy(vitis_templates / "vitis-axilite-master-bfm.sv", OUTPUTS)
    rtl_files = sorted((INPUTS / "backend-rtl").glob("*.v"))
    file_list = [
        "outputs/vitis-axi-memory-bfm.sv",
        "outputs/vitis-axilite-master-bfm.sv",
        "outputs/testbench.sv",
        *[str(path.resolve()) for path in rtl_files],
    ]
    (OUTPUTS / "vcs-rtl.f").write_text("\n".join(file_list) + "\n")
    return contract


def main():
    OUTPUTS.mkdir(exist_ok=True)
    workload = json.loads((INPUTS / "workload-manifest.json").read_text())
    metadata = json.loads((INPUTS / "build-metadata.json").read_text())
    if not workload.get("enabled"):
        raise RuntimeError("testbench generation requested with disabled workload manifest")

    backend = os.environ.get("backend", metadata.get("backend"))
    generators = {"vitis": _generate_vitis}
    generator = generators.get(backend)
    if generator is None:
        raise RuntimeError(
            f"unsupported testbench backend {backend!r}; supported backends: "
            f"{', '.join(sorted(generators))}"
        )
    contract = generator(workload, metadata)

    vector_source = INPUTS / "workload-vectors"
    vector_output = OUTPUTS / "workload-vectors"
    if vector_output.exists():
        shutil.rmtree(vector_output)
    shutil.copytree(vector_source, vector_output)
    (OUTPUTS / "testbench-contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    report = {
        "schema_version": 1,
        "node": "allo-testbench-generation",
        "status": "passed",
        "backend": backend,
        "top_module": contract["top_module"],
        "argument_count": len(contract["arguments"]),
        "call_count": len(contract["calls"]),
    }
    (OUTPUTS / "testbench-generation-report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
